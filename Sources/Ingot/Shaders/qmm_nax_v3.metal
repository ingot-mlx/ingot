// Dense prefill GEMM, v3: Y[M, N] (+)= X[M, K] · Wᵀ with 4-bit weights, rows of X and columns of W through
// threadgroup memory and a grid of WM × WN simdgroups of 32 × 32 (32·WM·WN threads). Each simdgroup carries at most
// 2 pairs of live accumulators, which is what fits in registers. It started as the MoE v3 down (large tiles,
// retired: with real routing it did not win); the body stayed for the dense GEMMs, where it does win with large K.
// Concatenated after qmm_nax.metal (nax_*, NX_*, QmmParams).

template <int bits, int G, int rows, int BK, int NT>
struct NaxWStageN {
    static constant constexpr uint per = 32 / bits, upr = BK / per, NI = (rows * upr + NT - 1) / NT;
    uint v[NI];
    bfloat sc[NI], bi[NI];

    METAL_FUNC void load(const device uint* w, const device bfloat* s, const device bfloat* b,
                         uint K, uint n0, uint k0, uint tid) {
        NAX_UNROLL for (uint i = 0; i < NI; i++) {
            const uint idx = tid + i * NT;
            if (NI * NT != rows * upr && idx >= rows * upr) break;
            const uint r = idx / upr, u = idx % upr;
            const uint k = k0 + u * per;
            const uint g = (n0 + r) * (K / G) + k / G;
            v[i] = w[size_t(n0 + r) * (K / per) + k / per];
            sc[i] = s[g];
            if constexpr (bits != 2) bi[i] = b[g];
        }
    }

    METAL_FUNC void store(threadgroup half* ws, uint tid) const {
        constexpr int LD = BK + 8;
        static_assert(bits == 4 || bits == 8 || (bits == 2 && G == 128),
                      "4 or 8 bits, or 2 ternary bits with groups of 128");
        if constexpr (bits == 8) {
            // Like fn_load_w8 (HC of Flash-Next): fma in half4 with the scale and bias in half; same values.
            NAX_UNROLL for (uint i = 0; i < NI; i++) {
                const uint idx = tid + i * NT;
                if (NI * NT != rows * upr && idx >= rows * upr) break;
                const uint r = idx / upr, u = idx % upr;
                const half sc1 = half(float(sc[i])), bi1 = half(float(bi[i]));
                *(threadgroup half4*)(ws + r * LD + u * per) = fma(half4(as_type<uchar4>(v[i])), half4(sc1), half4(bi1));
            }
            return;
        }
        if constexpr (bits == 2) {
            // Bonsai ternary (xc_tern.metal): w = (q − 1)·s, s fp16 stored in the bfloat slot. 16 values per uint;
            // ((x >> 2c) & 0x00030003) | 0x6400 gives the values c and c + 8.
            NAX_UNROLL for (uint i = 0; i < NI; i++) {
                const uint idx = tid + i * NT;
                if (NI * NT != rows * upr && idx >= rows * upr) break;
                const uint r = idx / upr, u = idx % upr;
                threadgroup half* o = ws + r * LD + u * per;
                const half2 s2 = half2(as_type<half>(sc[i]));
                const uint x = v[i];
                half lo[8], hi[8];
                NAX_UNROLL for (int c = 0; c < 8; c++) {
                    const half2 q = (as_type<half2>(((x >> (2 * c)) & 0x00030003u) | 0x64006400u) - half2(1025.0h)) * s2;
                    lo[c] = q.x; hi[c] = q.y;
                }
                *(threadgroup half4*)o = half4(lo[0], lo[1], lo[2], lo[3]);
                *(threadgroup half4*)(o + 4) = half4(lo[4], lo[5], lo[6], lo[7]);
                *(threadgroup half4*)(o + 8) = half4(hi[0], hi[1], hi[2], hi[3]);
                *(threadgroup half4*)(o + 12) = half4(hi[4], hi[5], hi[6], hi[7]);
            }
            return;
        }
        NAX_UNROLL for (uint i = 0; i < NI; i++) {
            const uint idx = tid + i * NT;
            if (NI * NT != rows * upr && idx >= rows * upr) break;
            const uint r = idx / upr, u = idx % upr;
            threadgroup half* o = ws + r * LD + u * per;
            const half2 s2 = half2(half(float(sc[i]))), b2 = half2(half(float(bi[i])));
            const uint x = v[i];
            const half2 q0 = fma(as_type<half2>((x & 0x000F000Fu) | 0x64006400u) - half2(1024.0h), s2, b2);
            const half2 q1 = fma(as_type<half2>(((x >> 4) & 0x000F000Fu) | 0x64006400u) - half2(1024.0h), s2, b2);
            const half2 q2 = fma(as_type<half2>(((x >> 8) & 0x000F000Fu) | 0x64006400u) - half2(1024.0h), s2, b2);
            const half2 q3 = fma(as_type<half2>(((x >> 12) & 0x000F000Fu) | 0x64006400u) - half2(1024.0h), s2, b2);
            *(threadgroup half4*)o = half4(q0.x, q1.x, q2.x, q3.x);
            *(threadgroup half4*)(o + 4) = half4(q0.y, q1.y, q2.y, q3.y);
        }
    }
};

template <int BM, int BK, int NT>
struct NaxAStageN {
    static constant constexpr int CPR = BK / 8, NI = (BM * CPR + NT - 1) / NT;
    half4 v[NI][2];
    METAL_FUNC void load(const thread array<const device half*, NI>& rp, uint k0, uint tid) {
        NAX_UNROLL for (int i = 0; i < NI; i++) {
            const uint ch = tid + i * NT;
            if (ch < uint(BM * CPR) && rp[i]) {
                const device half* src = rp[i] + k0 + (ch % CPR) * 8;
                v[i][0] = *(const device half4*)src;
                v[i][1] = *(const device half4*)(src + 4);
            }
        }
    }
    METAL_FUNC void store(threadgroup half* xs, uint tid) const {
        NAX_UNROLL for (int i = 0; i < NI; i++) {
            const uint ch = tid + i * NT;
            if (ch < uint(BM * CPR)) {
                threadgroup half* o = xs + (ch / CPR) * (BK + 8) + (ch % CPR) * 8;
                *(threadgroup half4*)o = v[i][0];
                *(threadgroup half4*)(o + 4) = v[i][1];
            }
        }
    }
};

template <int bits, int G, int WM, int WN, int PF, bool FULL, int BK = NX_BK, typename OT = half, bool ACC = false>
METAL_FUNC void down3_body(
    const device uint* w, const device bfloat* s, const device bfloat* b, device OT* y,
    const thread array<const device half*, NaxAStageN<32 * WM, BK, 32 * WM * WN>::NI>& rp,
    threadgroup half* ws, threadgroup half* xs,
    uint K, uint N, uint n0, uint r0, uint rows, int tm, int tn, short2 c, uint tid)
{
    constexpr int NT = 32 * WM * WN, BM = 32 * WM, BN = 32 * WN, KF = BK / 16, LD = BK + 8;
    using AS = NaxAStageN<BM, BK, NT>;
    using WS = NaxWStageN<bits, G, BN, BK, NT>;
    const int live = FULL ? 2 : clamp((int(rows) - tm + 15) / 16, 0, 2);
    nax_ffrag a00 = 0, a01 = 0, a10 = 0, a11 = 0;
    WS stw[PF];
    AS sta[PF];
    NAX_UNROLL for (int q = 0; q < PF; q++) if (uint(q * BK) < K) {
        stw[q].load(w, s, b, K, n0, q * BK, tid);
        sta[q].load(rp, q * BK, tid);
    }
    for (uint k00 = 0; k00 < K; k00 += PF * BK) NAX_UNROLL for (int q = 0; q < PF; q++) {
        const uint k0 = k00 + q * BK;
        if (k0 >= K) break;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        stw[q].store(ws, tid);
        sta[q].store(xs, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (k0 + PF * BK < K) {
            stw[q].load(w, s, b, K, n0, k0 + PF * BK, tid);
            sta[q].load(rp, k0 + PF * BK, tid);
        }
        if (!FULL && live == 0) continue;
        nax_hfrag b0[KF], b1[KF];
        NAX_UNROLL for (int kk = 0; kk < KF; kk++) {
            nax_load(b0[kk], ws + tn * LD + kk * 16, LD, c);
            nax_load(b1[kk], ws + (tn + 16) * LD + kk * 16, LD, c);
        }
        if (FULL || live == 2) {
            NAX_UNROLL for (int kk = 0; kk < KF; kk++) {
                nax_hfrag a, a1;
                nax_load(a, xs + tm * LD + kk * 16, LD, c);
                nax_load(a1, xs + (tm + 16) * LD + kk * 16, LD, c);
                nax_mma2(a00, a01, a, b0[kk], b1[kk]);
                nax_mma2(a10, a11, a1, b0[kk], b1[kk]);
            }
        } else {
            NAX_UNROLL for (int kk = 0; kk < KF; kk++) {
                nax_hfrag a;
                nax_load(a, xs + tm * LD + kk * 16, LD, c);
                nax_mma2(a00, a01, a, b0[kk], b1[kk]);
            }
        }
    }
    NAX_UNROLL for (int i = 0; i < 2; i++) {
        if (i >= live) break;
        NAX_UNROLL for (int j = 0; j < 2; j++) {
            const nax_ffrag A8 = i == 0 ? (j == 0 ? a00 : a01) : (j == 0 ? a10 : a11);
            NAX_UNROLL for (int q = 0; q < 2; q++) {
                const uint r = tm + i * 16 + c.x + q * 8;
                if (!FULL && r >= rows) continue;
                device vec<OT, 4>* yp = (device vec<OT, 4>*)(y + size_t(r0 + r) * N + n0 + tn + j * 16 + c.y);
                const vec<OT, 4> v4 = vec<OT, 4>(A8[q * 4], A8[q * 4 + 1], A8[q * 4 + 2], A8[q * 4 + 3]);
                *yp = ACC ? *yp + v4 : v4;
            }
        }
    }
}


// Dense GEMM Y[M, N] (+)= X[M, K] · Wᵀ with the body of the v3 down (rows of X through threadgroup memory, grid WM × WN
// of 32 × 32 simdgroups): for prefill (M a multiple of 32·WM). The usual qmm_t_nax reads X from memory into
// registers at every step and with K = 4096 drops to 14 TFLOPS (with K ≤ 2048 it gives 18.5).
template <int WM, int WN, int BK, typename OT, bool ACC, int BITS = 4, int GS = 64>
[[kernel, max_total_threads_per_threadgroup(32 * WM * WN)]] void qmm_t_nax3(
    const device half*   x   [[buffer(0)]],
    const device uint*   w   [[buffer(1)]],
    const device bfloat* s   [[buffer(2)]],
    const device bfloat* b   [[buffer(3)]],
    device OT*           y   [[buffer(4)]],
    constant QmmParams&  p   [[buffer(5)]],
    uint3  tgid [[threadgroup_position_in_grid]],
    uint   tid  [[thread_index_in_threadgroup]],
    ushort sgid [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]])
{
    constexpr int NT = 32 * WM * WN, BM = 32 * WM, BN = 32 * WN, LD = BK + 8;
    using AS = NaxAStageN<BM, BK, NT>;
    threadgroup half ws[BN * LD];
    threadgroup half xs[BM * LD];
    const uint r0 = tgid.y * BM, n0 = tgid.x * BN;
    array<const device half*, AS::NI> rp;
    NAX_UNROLL for (int i = 0; i < AS::NI; i++) rp[i] = x + size_t(r0 + (tid + i * NT) / AS::CPR) * p.K;
    down3_body<BITS, GS, WM, WN, 1, true, BK, OT, ACC>(w, s, b, y, rp, ws, xs, p.K, p.N, n0, r0, BM,
                                                   (sgid / WN) * 32, (sgid % WN) * 32, nax_coord(lane), tid);
}

#define INST_QMM3(WM, WN, BK)                                                                                     \
  template [[host_name("qmm_t_nax3_w" #WM "x" #WN "_bk" #BK "_f32acc")]] [[kernel]] void qmm_t_nax3<WM, WN, BK, float, true>( \
      const device half*, const device uint*, const device bfloat*, const device bfloat*, device float*,         \
      constant QmmParams&, uint3, uint, ushort, ushort);                                                        \
  template [[host_name("qmm_t_nax3_w" #WM "x" #WN "_bk" #BK "_f32")]] [[kernel]] void qmm_t_nax3<WM, WN, BK, float, false>( \
      const device half*, const device uint*, const device bfloat*, const device bfloat*, device float*,         \
      constant QmmParams&, uint3, uint, ushort, ushort);                                                        \
  template [[host_name("qmm_t_nax3_w" #WM "x" #WN "_bk" #BK "_h")]] [[kernel]] void qmm_t_nax3<WM, WN, BK, half, false>( \
      const device half*, const device uint*, const device bfloat*, const device bfloat*, device half*,          \
      constant QmmParams&, uint3, uint, ushort, ushort);
INST_QMM3(2, 2, 32) INST_QMM3(2, 4, 64)

// Groups of 32: AffineSrc / NaxWStageN with G = 32, only if the model has them (LAB_G32).
#if LAB_G32
#define INST_QMM3G(WM, WN, BK)                                                                                    \
  template [[host_name("qmm_t_nax3_g32_w" #WM "x" #WN "_bk" #BK "_f32acc")]] [[kernel]] void qmm_t_nax3<WM, WN, BK, float, true, 4, 32>( \
      const device half*, const device uint*, const device bfloat*, const device bfloat*, device float*,         \
      constant QmmParams&, uint3, uint, ushort, ushort);                                                        \
  template [[host_name("qmm_t_nax3_g32_w" #WM "x" #WN "_bk" #BK "_f32")]] [[kernel]] void qmm_t_nax3<WM, WN, BK, float, false, 4, 32>( \
      const device half*, const device uint*, const device bfloat*, const device bfloat*, device float*,         \
      constant QmmParams&, uint3, uint, ushort, ushort);                                                        \
  template [[host_name("qmm_t_nax3_g32_w" #WM "x" #WN "_bk" #BK "_h")]] [[kernel]] void qmm_t_nax3<WM, WN, BK, half, false, 4, 32>( \
      const device half*, const device uint*, const device bfloat*, const device bfloat*, device half*,          \
      constant QmmParams&, uint3, uint, ushort, ushort);
INST_QMM3G(2, 2, 32) INST_QMM3G(2, 4, 64)
#endif

// Bonsai ternary (2 bits, groups of 128, fp16 scale in buffer s; b is not read).
#define INST_QMM3T(WM, WN, BK)                                                                                    \
  template [[host_name("qmm_t_nax3t_w" #WM "x" #WN "_bk" #BK "_f32acc")]] [[kernel]] void qmm_t_nax3<WM, WN, BK, float, true, 2, 128>( \
      const device half*, const device uint*, const device bfloat*, const device bfloat*, device float*,         \
      constant QmmParams&, uint3, uint, ushort, ushort);                                                        \
  template [[host_name("qmm_t_nax3t_w" #WM "x" #WN "_bk" #BK "_f32")]] [[kernel]] void qmm_t_nax3<WM, WN, BK, float, false, 2, 128>( \
      const device half*, const device uint*, const device bfloat*, const device bfloat*, device float*,         \
      constant QmmParams&, uint3, uint, ushort, ushort);                                                        \
  template [[host_name("qmm_t_nax3t_w" #WM "x" #WN "_bk" #BK "_h")]] [[kernel]] void qmm_t_nax3<WM, WN, BK, half, false, 2, 128>( \
      const device half*, const device uint*, const device bfloat*, const device bfloat*, device half*,          \
      constant QmmParams&, uint3, uint, ushort, ushort);
INST_QMM3T(2, 2, 32) INST_QMM3T(2, 4, 64)

// 8-bit g64 with fp32 output: the prefill HC down of Flash-Next (K = 10240). Same order of sums per
// element as fn_qmm8_f32_nax (steps of 16 in K, in order), so the result is bit-exact.
#define INST_QMM3_8(WM, WN, BK)                                                                                   \
  template [[host_name("qmm_t_nax3_8_w" #WM "x" #WN "_bk" #BK "_f32")]] [[kernel]] void qmm_t_nax3<WM, WN, BK, float, false, 8, 64>( \
      const device half*, const device uint*, const device bfloat*, const device bfloat*, device float*,         \
      constant QmmParams&, uint3, uint, ushort, ushort);
// Half output and accumulated fp32: the 8-bit dense FFN of prefill, with the splits of encodeQmmF32.
#define INST_QMM3_8HA(WM, WN, BK)                                                                                 \
  template [[host_name("qmm_t_nax3_8_w" #WM "x" #WN "_bk" #BK "_f32acc")]] [[kernel]] void qmm_t_nax3<WM, WN, BK, float, true, 8, 64>( \
      const device half*, const device uint*, const device bfloat*, const device bfloat*, device float*,         \
      constant QmmParams&, uint3, uint, ushort, ushort);                                                        \
  template [[host_name("qmm_t_nax3_8_w" #WM "x" #WN "_bk" #BK "_h")]] [[kernel]] void qmm_t_nax3<WM, WN, BK, half, false, 8, 64>( \
      const device half*, const device uint*, const device bfloat*, const device bfloat*, device half*,          \
      constant QmmParams&, uint3, uint, ushort, ushort);
INST_QMM3_8HA(2, 2, 32) INST_QMM3_8HA(2, 4, 64)
INST_QMM3_8(2, 2, 32) INST_QMM3_8(2, 4, 64) INST_QMM3_8(2, 2, 64)
