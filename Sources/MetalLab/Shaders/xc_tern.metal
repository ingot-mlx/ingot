// Ternary format of Bonsai 2 (prism_hadamard_qwen35, Prism ML, Inc.; weights Apache-2.0): MLX affine 2-bit with groups of 128,
// fp16 scale and bias = −scale, i.e. w = (q − 1)·s with q ∈ {0, 1, 2}. Only the scale is stored (the bias is
// checked at load). The weights are in a rotated basis: before each matrix the activation goes through
// x ← H₁₀₂₄(x ⊙ signs) / 32 in blocks of 1024 (normalized Sylvester Walsh-Hadamard); the embedding does the
// inverse (H₁₀₂₄(row) / 32 ⊙ signs).
//
// Concatenated after qmm_nax*.metal (nax_*) and verify*.metal (VParams, VUNROLL).

// FWHT of 1024 values spread over the 1024 threads of the threadgroup (one value per thread, thread t holds index t):
// steps 1..16 between lanes with simd_shuffle_xor and 32..512 through threadgroup memory. Not normalized.
inline float tern_fwht1024(float v, threadgroup float* sh, uint t, ushort lane) {
    VUNROLL for (ushort s = 1; s < 32; s <<= 1) {
        const float o = simd_shuffle_xor(v, s);
        v = (lane & s) ? o - v : v + o;
    }
    for (uint s = 32; s < 1024; s <<= 1) {
        sh[t] = v;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float o = sh[t ^ s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        v = (t & s) ? o - v : v + o;
    }
    return v;
}

struct HadParams { uint K, ldx, ldy, pad; };

// y[r, :] = H(x[r, :] ⊙ signs) / 32 in blocks of 1024 (half → half activations). Grid (K/1024, rows), 1024 threads.
[[kernel]] void had_rows_h(const device half*  x  [[buffer(0)]],
                           const device float* sg [[buffer(1)]],
                           device half*        y  [[buffer(2)]],
                           constant HadParams& p  [[buffer(3)]],
                           uint2 tg [[threadgroup_position_in_grid]],
                           uint t [[thread_index_in_threadgroup]],
                           ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float sh[1024];
    const uint k = tg.x * 1024 + t;
    const float v = tern_fwht1024(float(x[size_t(tg.y) * p.ldx + k]) * sg[k], sh, t, lane);
    y[size_t(tg.y) * p.ldy + k] = half(v * (1.0f / 32.0f));
}

// Ternary embedding: row tok[r] dequantized, inverse rotation, and output float (flags 1) and/or half (flags 2)
// with stride ldo. One threadgroup of 1024 threads per row; K a multiple of 1024.
[[kernel]] void embed_rows_t(const device uint*   tok [[buffer(0)]],
                             const device uint*   w   [[buffer(1)]],
                             const device half*   s   [[buffer(2)]],
                             const device float*  sg  [[buffer(3)]],
                             device float*        of  [[buffer(4)]],
                             device half*         oh  [[buffer(5)]],
                             constant uint4&      p   [[buffer(6)]],   // K, flags, ldo
                             uint row [[threadgroup_position_in_grid]],
                             uint t [[thread_index_in_threadgroup]],
                             ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float sh[1024];
    const uint K = p.x, tk = tok[row];
    const device uint* wr = w + size_t(tk) * (K / 16);
    const device half* sr = s + size_t(tk) * (K / 128);
    for (uint b0 = 0; b0 < K; b0 += 1024) {
        const uint k = b0 + t;
        const float q = float((wr[k / 16] >> (2 * (k % 16))) & 3u) - 1.0f;
        const float v = tern_fwht1024(q * float(sr[k / 128]), sh, t, lane) * (1.0f / 32.0f) * sg[k];
        if (p.y & 1) of[size_t(row) * p.z + k] = v;
        if (p.y & 2) oh[size_t(row) * p.z + k] = half(v);
    }
}

// Y[m, n] (+)= Σ_k x[m, k] · W[n, k] with unquantized fp32 W (in_proj_b | in_proj_a of Bonsai's GDN, 96 rows).
// Grid (⌈N/8⌉, M), 256 threads: one simdgroup per output. y in float or half (OT) with stride ldy.
struct RowDotParams { uint M, N, K, ldx, ldy, flags; };
template <typename OT>
[[kernel]] void rowdot_f32w(const device half*     x [[buffer(0)]],
                            const device float*    W [[buffer(1)]],
                            device OT*             y [[buffer(2)]],
                            constant RowDotParams& p [[buffer(3)]],
                            uint2 tg [[threadgroup_position_in_grid]],
                            ushort sg [[simdgroup_index_in_threadgroup]],
                            ushort lane [[thread_index_in_simdgroup]]) {
    const uint n = tg.x * 8 + sg, m = tg.y;
    if (n >= p.N) return;
    const device half4* xr = (const device half4*)(x + size_t(m) * p.ldx);
    const device float4* wr = (const device float4*)(W + size_t(n) * p.K);
    float acc = 0;
    for (uint i = lane; i < p.K / 4; i += 32) acc += dot(float4(xr[i]), wr[i]);
    acc = simd_sum(acc);
    if (lane == 0) {
        device OT* o = y + size_t(m) * p.ldy + n;
        *o = (p.flags & 1) ? OT(float(*o) + acc) : OT(acc);
    }
}
template [[host_name("rowdot_f32w_f")]] [[kernel]] void rowdot_f32w<float>(const device half*, const device float*, device float*,
                                                                           constant RowDotParams&, uint2, ushort, ushort);
template [[host_name("rowdot_f32w_h")]] [[kernel]] void rowdot_f32w<half>(const device half*, const device float*, device half*,
                                                                          constant RowDotParams&, uint2, ushort, ushort);

// ---- Ternary verification (B ≤ 16) on NAX: like vnax_qmv, but each K step is a group of 128 and each thread
// holds 32 contiguous values of its W row (a 2-bit uint2): [32·(fn/4), +32) of the group. In chunk c (0..7) it
// uses values {c, c+8, 16+c, 24+c}: ((w >> 2c) & 0x00030003) | 0x64006400 gives the half2 (1024 + q_c, 1024 + q_c+8).
// x uses the same permutation.

inline half2 vt_deq(uint w, int c, half2 s) {
    return (as_type<half2>(((w >> (2 * c)) & 0x00030003u) | 0x64006400u) - half2(1025.0h)) * s;
}

struct VNaxT {
    uint2 w0, w1;
    half2 s0, s1;
    inline nax_hfrag frag(int c) const {
        const half2 l0 = vt_deq(w0.x, c, s0), h0 = vt_deq(w0.y, c, s0);
        const half2 l1 = vt_deq(w1.x, c, s1), h1 = vt_deq(w1.y, c, s1);
        return nax_hfrag(l0.x, l0.y, h0.x, h0.y, l1.x, l1.y, h1.x, h1.y);
    }
};

inline VNaxT vt_loadw(const device uint* w, const device half* s, uint n, uint wpr, uint gpr, uint kk, uint kb) {
    VNaxT r;
    r.w0 = *(const device uint2*)(w + size_t(n) * wpr + kk / 16);
    r.w1 = *(const device uint2*)(w + size_t(n + 8) * wpr + kk / 16);
    r.s0 = half2(s[size_t(n) * gpr + kb]);
    r.s1 = half2(s[size_t(n + 8) * gpr + kb]);
    return r;
}

inline void vt_load32(thread half (&d)[32], const device half* p) {
    const device half4* q = (const device half4*)p;
    VUNROLL for (int i = 0; i < 8; i++) { half4 v = q[i]; d[4 * i] = v.x; d[4 * i + 1] = v.y; d[4 * i + 2] = v.z; d[4 * i + 3] = v.w; }
}

inline nax_hfrag vt_afrag(thread const half (&x0)[32], thread const half (&x1)[32], int c) {
    nax_hfrag a;
    a[0] = x0[c]; a[1] = x0[c + 8]; a[2] = x0[16 + c]; a[3] = x0[24 + c];
    a[4] = x1[c]; a[5] = x1[c + 8]; a[6] = x1[16 + c]; a[7] = x1[24 + c];
    return a;
}

template <int NF, int SGS, int U = 1>
[[kernel]] void vnax_qmvt(const device half*   x [[buffer(0)]],
                          const device uint*   w [[buffer(1)]],
                          const device half*   s [[buffer(2)]],
                          device float*        y [[buffer(4)]],
                          constant VParams&    p [[buffer(5)]],
                          uint tg [[threadgroup_position_in_grid]],
                          uint tid [[thread_index_in_threadgroup]],
                          ushort sg [[simdgroup_index_in_threadgroup]],
                          ushort lane [[thread_index_in_simdgroup]]) {
    constexpr int NC = 16 * NF;
    threadgroup float red[SGS * 16 * NC];
    const short2 co = nax_coord(lane);
    const uint fm = co.x, fn = co.y;
    const uint n0 = tg * NC, wpr = p.K / 16, gpr = p.K / 128;
    const uint r0 = min(fm, p.M - 1), r1 = min(fm + 8, p.M - 1);
    nax_ffrag acc[NF];
    VUNROLL for (int f = 0; f < NF; f++) acc[f] = 0;
    // U groups of K per step (the weight loads of the U groups go before the products).
    for (uint kb0 = sg; kb0 < gpr; kb0 += SGS * U) {
        VNaxT wt[U][NF];
        VUNROLL for (int u = 0; u < U; u++) {
            const uint kb = min(kb0 + u * SGS, gpr - 1), kk = kb * 128 + (fn / 4) * 32;
            VUNROLL for (int f = 0; f < NF; f++) wt[u][f] = vt_loadw(w, s, n0 + 16 * f + fm, wpr, gpr, kk, kb);
        }
        VUNROLL for (int u = 0; u < U; u++) {
            const uint kb = kb0 + u * SGS;
            if (U > 1 && kb >= gpr) break;
            const uint kk = kb * 128 + (fn / 4) * 32;
            half x0[32], x1[32];
            vt_load32(x0, x + size_t(r0) * p.ldx + kk);
            if (p.M > 8) vt_load32(x1, x + size_t(r1) * p.ldx + kk);
            else VUNROLL for (int i = 0; i < 32; i++) x1[i] = x0[i];
            VUNROLL for (int c = 0; c < 8; c++) {
                const nax_hfrag a = vt_afrag(x0, x1, c);
                if constexpr (NF == 1) {
                    nax_ffrag dummy = 0;
                    nax_mma2(acc[0], dummy, a, wt[u][0].frag(c), wt[u][0].frag(c));
                } else {
                    VUNROLL for (int f = 0; f < NF; f += 2) nax_mma2(acc[f], acc[f + 1], a, wt[u][f].frag(c), wt[u][f + 1].frag(c));
                }
            }
        }
    }
    vnax_reduce_store<NF, SGS>(acc, red, y, p, n0, tid, sg, fm, fn);
}

#define VNQT_INST(NF, SGS)                                                                                         \
  template [[host_name("vnax_qmvt_nf" #NF "_sg" #SGS)]] [[kernel]] void vnax_qmvt<NF, SGS, 1>(                     \
      const device half*, const device uint*, const device half*, device float*, constant VParams&, uint, uint,    \
      ushort, ushort);
VNQT_INST(1, 16) VNQT_INST(1, 8)

// ---- Ternary W8A8 prefill (option --int8 1 with Bonsai): the activations (already rotated by the format's 1024
// Hadamard) to int8 per row with no rotation of its own, and the weights to int8 with one scale per row (tern_i8row)
// for the normal qmm_i8. (The variant with exact weights and per-group scale, qmm_i8t, was removed.)

// Half rows [M, K] → int8 per row (s = max|x| / 127). One simdgroup per row, 4 per threadgroup.
[[kernel]] void quant_rows_i8(const device half* x  [[buffer(0)]],
                              device char*       xq [[buffer(1)]],
                              device float*      sx [[buffer(2)]],
                              constant uint2&    p  [[buffer(3)]],   // (rows, K)
                              uint   tg   [[threadgroup_position_in_grid]],
                              ushort sg   [[simdgroup_index_in_threadgroup]],
                              ushort lane [[thread_index_in_simdgroup]]) {
    const uint r = tg * 4 + sg;
    if (r >= p.x) return;
    const uint K = p.y;
    const device half4* xr = (const device half4*)(x + size_t(r) * K);
    float m = 0;
    for (uint i = lane; i < K / 4; i += 32) { const float4 a = abs(float4(xr[i])); m = max(m, max(max(a.x, a.y), max(a.z, a.w))); }
    m = simd_max(m);
    const float sc = m > 0 ? m / 127.0f : 1.0f, inv = 1.0f / sc;
    if (lane == 0) sx[r] = sc;
    device char4* o = (device char4*)(xq + size_t(r) * K);
    for (uint i = lane; i < K / 4; i += 32) o[i] = char4(clamp(rint(float4(xr[i]) * inv), -127.0f, 127.0f));
}

// "Per-row" weights (lossy in the weights): T[n, k]·s[n, g] → int8 with ONE scale per row,
// wq = t · round(s_g / s_row), s_row = max_g |s_g| / 127; then the normal qmm_i8 (no per-group scales).
// One simdgroup per row, 4 per threadgroup.
[[kernel]] void tern_i8row(const device uint* w  [[buffer(0)]],
                           const device half* s  [[buffer(1)]],
                           device char*       wq [[buffer(2)]],
                           device float*      sw [[buffer(3)]],
                           constant uint2&    p  [[buffer(4)]],   // (N, K)
                           uint   tg   [[threadgroup_position_in_grid]],
                           ushort sg   [[simdgroup_index_in_threadgroup]],
                           ushort lane [[thread_index_in_simdgroup]]) {
    const uint n = tg * 4 + sg;
    if (n >= p.x) return;
    const uint K = p.y, G = K / 128;
    const device half* sr = s + size_t(n) * G;
    float m = 0;
    for (uint g = lane; g < G; g += 32) m = max(m, abs(float(sr[g])));
    m = simd_max(m);
    const float srow = m > 0 ? m / 127.0f : 1.0f, inv = 1.0f / srow;
    if (lane == 0) sw[n] = srow;
    const device uint* wr = w + size_t(n) * (K / 16);
    device char4* o = (device char4*)(wq + size_t(n) * K);
    for (uint u = lane; u < K / 16; u += 32) {
        const int q = int(rint(float(sr[u / 8]) * inv));
        const uint x = wr[u];
        VUNROLL for (int c = 0; c < 4; c++)
            o[u * 4 + c] = char4((int4((x >> (8 * c)) & 3u, (x >> (8 * c + 2)) & 3u, (x >> (8 * c + 4)) & 3u, (x >> (8 * c + 6)) & 3u) - 1) * q);
    }
}
