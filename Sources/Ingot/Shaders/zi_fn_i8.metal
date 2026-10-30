// Q2 experts of Flash-Next prefill via the int8 route (matmul2d int8 × int8 → int32, ~2× the fp16 rate on
// the M6). Only under the INT8 prefill route (options.prefillInt8 and int8Experts); the fp16 route is unchanged.
//   - Weights: w = d·(q − 1) with q − 1 ∈ {−1, 0, 1, 2}, so (q − 1) in int8 is exact; the scale d of each group of 64
//     is applied to the group's int32 accumulator (one group = one K step of 64).
//   - Activations: int8 per group of 64 of each row (scale max|x|/127): the only loss. X with fn_q8g_perm; h comes out
//     already quantized from the gate|up epilogue (without going through half).
//   - Order within each block of 16 of K: word m holds elements m, m+4, m+8, m+12. This way the 2-bit codes are
//     unpacked with a shift, a mask and two operations per word ((q + 0x7F) ^ 0x80 = q − 1
//     per byte), and the activations are written already in that order (the dot product does not depend on the order).
// Same structure as fn_moe_gateup_nax / fn_moe_down_nax (zf_flashnext.metal): tiles of one expert × BM rows,
// 4 simdgroups, weights and rows to threadgroup with the next step's load in registers.
// Concatenated after zf_flashnext.metal (FnQ2Params, FN_*) and qmm_nax_i8.metal (nax_cfrag, nax_mma2_i8).

// NG groups of 64 of K per step (BK = 64·NG; only NG = 1 is instantiated: with 2 it was slower); threadgroup row of
// 64·NG + 16 bytes.
template <int NG> constexpr int fi8_ld() { return 64 * NG + 16; }

// x half [R, K] → xq int8 [R, K] in the permuted order and sx float [R, K/64]. One thread per block of 16 (n = R·K/16);
// the 4 threads of a group are consecutive lanes of the simdgroup.
[[kernel]] void fn_q8g_perm(const device half* x  [[buffer(0)]],
                            device uint4*      xq [[buffer(1)]],
                            device float*      sx [[buffer(2)]],
                            constant uint&     n  [[buffer(3)]],
                            uint gid [[thread_position_in_grid]]) {
    const bool ok = gid < n;
    float f[16];
    if (ok) {
        const device half4* p = (const device half4*)(x + size_t(gid) * 16);
        NAX_UNROLL for (int j = 0; j < 4; j++) {
            const half4 v = p[j];
            f[4 * j] = v.x; f[4 * j + 1] = v.y; f[4 * j + 2] = v.z; f[4 * j + 3] = v.w;
        }
    } else {
        NAX_UNROLL for (int j = 0; j < 16; j++) f[j] = 0;
    }
    float m = 0;
    NAX_UNROLL for (int j = 0; j < 16; j++) m = max(m, abs(f[j]));
    m = max(m, simd_shuffle_xor(m, 1));
    m = max(m, simd_shuffle_xor(m, 2));
    if (!ok) return;
    const float s = m > 0 ? m / 127.0f : 1.0f, inv = 1.0f / s;
    if ((gid & 3) == 0) sx[gid / 4] = s;
    uint4 o;
    NAX_UNROLL for (int w = 0; w < 4; w++) {
        const float4 v = float4(f[w], f[w + 4], f[w + 8], f[w + 12]) * inv;
        o[w] = as_type<uint>(char4(clamp(rint(v), -127.0f, 127.0f)));
    }
    xq[gid] = o;
}

// Q2 weights of `rows` rows × NG groups of 64 of K → (q − 1) int8 permuted in threadgroup, and the scales d → float
// (ds[g·rows + row]).
template <int rows, int NG>
struct FnQ2I8Stage {
    static constant constexpr uint CPR = 4 * NG, NI = rows * CPR / NX_THREADS, LD = fi8_ld<NG>();
    static_assert(rows * CPR % NX_THREADS == 0 && rows <= NX_THREADS, "FnQ2I8Stage: rows");
    uint v[NI];
    half sc[NG];
    METAL_FUNC void load(const device uint* codes, const device half* scales, uint K, uint n0, uint k0, uint tid) {
        NAX_UNROLL for (uint i = 0; i < NI; i++) {
            const uint idx = tid + i * NX_THREADS;
            v[i] = codes[size_t(n0 + idx / CPR) * (K / 16) + k0 / 16 + idx % CPR];
        }
        if (tid < uint(rows)) NAX_UNROLL for (int g = 0; g < NG; g++) sc[g] = scales[size_t(n0 + tid) * (K / 64) + k0 / 64 + g];
    }
    METAL_FUNC void store(threadgroup char* ws, threadgroup float* ds, uint tid) const {
        NAX_UNROLL for (uint i = 0; i < NI; i++) {
            const uint idx = tid + i * NX_THREADS;
            threadgroup uint* o = (threadgroup uint*)(ws + (idx / CPR) * LD + (idx % CPR) * 16);
            NAX_UNROLL for (int m = 0; m < 4; m++) o[m] = (((v[i] >> (2 * m)) & 0x03030303u) + 0x7F7F7F7Fu) ^ 0x80808080u;
        }
        if (tid < uint(rows)) NAX_UNROLL for (int g = 0; g < NG; g++) ds[g * rows + tid] = float(sc[g]);
    }
};

// BM rows × NG groups of 64 of K of xq (row i from rp[i]) to threadgroup, and their scales (xsc[g·BM + row]).
template <int BM, int NG>
struct FnI8AStage {
    static constant constexpr int CPR = 4 * NG, NI = (BM * CPR + NX_THREADS - 1) / NX_THREADS, LD = fi8_ld<NG>();
    uint4 v[NI];
    float s[NG];
    METAL_FUNC void load(const thread array<const device char*, NI>& rp, const device float* sp, uint k0, uint tid) {
        NAX_UNROLL for (int i = 0; i < NI; i++) {
            const uint ch = tid + i * NX_THREADS;
            if (ch < uint(BM * CPR) && rp[i]) v[i] = *(const device uint4*)(rp[i] + k0 + (ch % CPR) * 16);
        }
        NAX_UNROLL for (int g = 0; g < NG; g++) s[g] = sp ? sp[k0 / 64 + g] : 0.0f;
    }
    METAL_FUNC void store(threadgroup char* xs, threadgroup float* xsc, uint tid) const {
        NAX_UNROLL for (int i = 0; i < NI; i++) {
            const uint ch = tid + i * NX_THREADS;
            if (ch < uint(BM * CPR)) *(threadgroup uint4*)(xs + (ch / CPR) * LD + (ch % CPR) * 16) = v[i];
        }
        if (tid < uint(BM)) NAX_UNROLL for (int g = 0; g < NG; g++) xsc[g * BM + tid] = s[g];
    }
};

// Accumulates into f the rescaled int32 of a group: f[e] += a[e]·sx(row)·d(column), e = q·4 + j at (c.x + 8q, c.y + j).
METAL_FUNC void fi8_rescale(thread nax_ffrag& f, const thread nax_ifrag& a, float2 s, float4 d) {
    NAX_UNROLL for (int q = 0; q < 2; q++) NAX_UNROLL for (int j = 0; j < 4; j++)
        f[q * 4 + j] += float(a[q * 4 + j]) * (s[q] * d[j]);
}

template <int BM, int NG, int LIVE>
METAL_FUNC void fn_gateup_i8_body(
    const device uint* wg, const device half* sg, const device uint* wu, const device half* su, device char* hq,
    device float* hs, threadgroup float* rmax, ushort sgid, ushort lane,
    const thread array<const device char*, FnI8AStage<BM, NG>::NI>& rp, const device float* sp, threadgroup char* wgs,
    threadgroup char* wus, threadgroup float* dsg, threadgroup float* dsu, threadgroup char* xs, threadgroup float* xsc,
    uint n0, uint r0, uint rows, int tn, short2 c, uint tid)
{
    constexpr uint K = FN_D, N = FN_I;
    nax_ffrag fg[LIVE], fu[LIVE];
    NAX_UNROLL for (int i = 0; i < LIVE; i++) { fg[i] = 0; fu[i] = 0; }
    FnQ2I8Stage<64, NG> stg, stu;
    FnI8AStage<BM, NG> sta;
    constexpr int LD = fi8_ld<NG>();
    stg.load(wg, sg, K, n0, 0, tid);
    stu.load(wu, su, K, n0, 0, tid);
    sta.load(rp, sp, 0, tid);
    for (uint k0 = 0; k0 < K; k0 += 64 * NG) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        stg.store(wgs, dsg, tid);
        stu.store(wus, dsu, tid);
        sta.store(xs, xsc, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (k0 + 64 * NG < K) {
            stg.load(wg, sg, K, n0, k0 + 64 * NG, tid);
            stu.load(wu, su, K, n0, k0 + 64 * NG, tid);
            sta.load(rp, sp, k0 + 64 * NG, tid);
        }
        NAX_UNROLL for (int gq = 0; gq < NG; gq++) {
            nax_cfrag bg[4], bu[4];
            NAX_UNROLL for (int kk = 0; kk < 4; kk++) {
                nax_load_c(bg[kk], wgs + tn * LD + gq * 64 + kk * 16, LD, c);
                nax_load_c(bu[kk], wus + tn * LD + gq * 64 + kk * 16, LD, c);
            }
            const float4 dg = *(const threadgroup float4*)(dsg + gq * 64 + tn + c.y);
            const float4 du = *(const threadgroup float4*)(dsu + gq * 64 + tn + c.y);
            NAX_UNROLL for (int i = 0; i < LIVE; i++) {
                nax_ifrag ig = 0, iu = 0;
                NAX_UNROLL for (int kk = 0; kk < 4; kk++) {
                    nax_cfrag a;
                    nax_load_c(a, xs + (i * 16) * LD + gq * 64 + kk * 16, LD, c);
                    nax_mma2_i8(ig, iu, a, bg[kk], bu[kk]);
                }
                const float2 s = float2(xsc[gq * BM + i * 16 + c.x], xsc[gq * BM + i * 16 + c.x + 8]);
                fi8_rescale(fg[i], ig, s, dg);
                fi8_rescale(fu[i], iu, s, du);
            }
        }
    }
    // h = SiLU(g)·u quantized here to int8 (the 64 columns of the tile are a group of 64 of I): row max
    // across the 4 lanes sharing it (bits 0 and 3 of the lane) and across the 4 simdgroups (threadgroup), and each lane
    // writes word b0 of its simdgroup's block of 16 in the permuted order (byte b ← value b0 of lane b).
    float4 hv[LIVE][2];
    NAX_UNROLL for (int i = 0; i < LIVE; i++) NAX_UNROLL for (int q = 0; q < 2; q++) {
        const float4 g = float4(fg[i][q * 4], fg[i][q * 4 + 1], fg[i][q * 4 + 2], fg[i][q * 4 + 3]);
        const float4 u = float4(fu[i][q * 4], fu[i][q * 4 + 1], fu[i][q * 4 + 2], fu[i][q * 4 + 3]);
        hv[i][q] = g / (1.0f + exp(-g)) * u;
        const float4 a = abs(hv[i][q]);
        float m = max(max(a.x, a.y), max(a.z, a.w));
        m = max(m, simd_shuffle_xor(m, ushort(1)));
        m = max(m, simd_shuffle_xor(m, ushort(8)));
        if ((lane & 9) == 0) rmax[(i * 16 + c.x + q * 8) * 4 + sgid] = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint b0 = ((lane >> 3) & 1) * 2 + (lane & 1);
    NAX_UNROLL for (int i = 0; i < LIVE; i++) NAX_UNROLL for (int q = 0; q < 2; q++) {
        const uint r = i * 16 + c.x + q * 8;
        const float4 mm = *(const threadgroup float4*)(rmax + r * 4);
        const float m = max(max(mm.x, mm.y), max(mm.z, mm.w));
        const float sc = m > 0 ? m / 127.0f : 1.0f;
        const uint own = as_type<uint>(char4(clamp(rint(hv[i][q] / sc), -127.0f, 127.0f)));
        uint word = 0;
        NAX_UNROLL for (uint b = 0; b < 4; b++) {
            const uint src = simd_shuffle(own, ushort((lane & ~9u) | ((b >> 1) << 3) | (b & 1)));
            word |= ((src >> (8 * b0)) & 0xffu) << (8 * b);
        }
        if (r >= rows) continue;
        *(device uint*)(hq + size_t(r0 + r) * N + n0 + tn + 4 * b0) = word;
        if (sgid == 0 && b0 == 0) hs[size_t(r0 + r) * (N / 64) + n0 / 64] = sc;
    }
}

// Tiles BM × 64 columns of I (gate and up at once), 16 columns per simdgroup. Indirect grid (I / 64, tiles, 1).
template <int BM, int NG>
[[kernel, max_total_threads_per_threadgroup(NX_THREADS)]] void fn_moe_gateup_i8(
    const device char*   xq     [[buffer(0)]],   // [T, D] permuted
    const device float*  xsc    [[buffer(1)]],   // [T, D/64]
    const device uchar*  arena  [[buffer(2)]],
    const device uint*   smap   [[buffer(3)]],
    const device uint4*  tiles  [[buffer(4)]],
    const device uint*   sorted [[buffer(5)]],
    device char*         hq     [[buffer(6)]],   // [pairs, I] permuted (grouped order)
    device float*        hs     [[buffer(7)]],   // [pares, I/64]
    constant FnQ2Params& p      [[buffer(8)]],
    uint3  tgid [[threadgroup_position_in_grid]],
    uint   tid  [[thread_index_in_threadgroup]],
    ushort sgid [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]])
{
    constexpr int FM = BM / 16;
    using AS = FnI8AStage<BM, NG>;
    constexpr int LD = fi8_ld<NG>();
    threadgroup char wgs[64 * LD];
    threadgroup char wus[64 * LD];
    threadgroup char xs[BM * LD];
    threadgroup float dsg[64 * NG], dsu[64 * NG], xss[BM * NG], rmax[BM * 4];
    const uint4 td = tiles[tgid.y];
    const uint r0 = td.y, rows = td.z;
    const uint n0 = tgid.x * 64;
    const int tn = sgid * 16;
    const short2 c = nax_coord(lane);
    array<const device char*, AS::NI> rp;
    NAX_UNROLL for (int i = 0; i < AS::NI; i++) {
        const uint r = (tid + i * NX_THREADS) / AS::CPR;
        rp[i] = r < rows ? xq + size_t(sorted[r0 + r] / FN_K) * FN_D : nullptr;
    }
    const device float* sp = tid < rows && tid < uint(BM) ? xsc + size_t(sorted[r0 + tid] / FN_K) * (FN_D / 64) : nullptr;
    const device uchar* rec = arena + size_t(smap[td.x]) * p.rec;
    const device uint* wg = (const device uint*)(rec + p.gc);
    const device uint* wu = (const device uint*)(rec + p.uc);
    const device half* sg = (const device half*)(rec + p.gs);
    const device half* su = (const device half*)(rec + p.us);
    const int live = min(int(rows + 15) / 16, FM);
#define FGU8(L) fn_gateup_i8_body<BM, NG, L>(wg, sg, wu, su, hq, hs, rmax, sgid, lane, rp, sp, wgs, wus, dsg, dsu, xs, xss, n0, r0, rows, tn, c, tid)
    if (FM >= 4 && live == 4) { if constexpr (FM >= 4) FGU8((4 > FM ? FM : 4)); }
    else if (FM >= 3 && live == 3) { if constexpr (FM >= 3) FGU8((3 > FM ? FM : 3)); }
    else if (FM >= 2 && live == 2) { if constexpr (FM >= 2) FGU8((2 > FM ? FM : 2)); }
    else FGU8(1);
#undef FGU8
}

template <int BM, int NG, int LIVE>
METAL_FUNC void fn_down_i8_body(
    const device uint* w, const device half* s, device float* y,
    const thread array<const device char*, FnI8AStage<BM, NG>::NI>& rp, const device float* sp, threadgroup char* ws,
    threadgroup float* ds, threadgroup char* xs, threadgroup float* xsc, uint n0, uint r0, uint rows, int tn, short2 c,
    uint tid)
{
    constexpr uint K = FN_I, N = FN_D;
    nax_ffrag acc[LIVE][2];
    NAX_UNROLL for (int i = 0; i < LIVE; i++) { acc[i][0] = 0; acc[i][1] = 0; }
    FnQ2I8Stage<128, NG> stw;
    FnI8AStage<BM, NG> sta;
    constexpr int LD = fi8_ld<NG>();
    stw.load(w, s, K, n0, 0, tid);
    sta.load(rp, sp, 0, tid);
    for (uint k0 = 0; k0 < K; k0 += 64 * NG) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        stw.store(ws, ds, tid);
        sta.store(xs, xsc, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (k0 + 64 * NG < K) {
            stw.load(w, s, K, n0, k0 + 64 * NG, tid);
            sta.load(rp, sp, k0 + 64 * NG, tid);
        }
        NAX_UNROLL for (int gq = 0; gq < NG; gq++) {
            nax_cfrag bt[2][4];
            NAX_UNROLL for (int kk = 0; kk < 4; kk++) NAX_UNROLL for (int j = 0; j < 2; j++)
                nax_load_c(bt[j][kk], ws + (tn + j * 16) * LD + gq * 64 + kk * 16, LD, c);
            const float4 d0 = *(const threadgroup float4*)(ds + gq * 128 + tn + c.y);
            const float4 d1 = *(const threadgroup float4*)(ds + gq * 128 + tn + 16 + c.y);
            NAX_UNROLL for (int i = 0; i < LIVE; i++) {
                nax_ifrag a0 = 0, a1 = 0;
                NAX_UNROLL for (int kk = 0; kk < 4; kk++) {
                    nax_cfrag a;
                    nax_load_c(a, xs + (i * 16) * LD + gq * 64 + kk * 16, LD, c);
                    nax_mma2_i8(a0, a1, a, bt[0][kk], bt[1][kk]);
                }
                const float2 sx = float2(xsc[gq * BM + i * 16 + c.x], xsc[gq * BM + i * 16 + c.x + 8]);
                fi8_rescale(acc[i][0], a0, sx, d0);
                fi8_rescale(acc[i][1], a1, sx, d1);
            }
        }
    }
    NAX_UNROLL for (int i = 0; i < LIVE; i++) NAX_UNROLL for (int j = 0; j < 2; j++) NAX_UNROLL for (int q = 0; q < 2; q++) {
        const uint r = i * 16 + c.x + q * 8;
        if (r >= rows) continue;
        *(device float4*)(y + size_t(r0 + r) * N + n0 + tn + j * 16 + c.y) =
            float4(acc[i][j][q * 4], acc[i][j][q * 4 + 1], acc[i][j][q * 4 + 2], acc[i][j][q * 4 + 3]);
    }
}

// Tiles BM × 128 columns of D, 32 per simdgroup. Indirect grid (D / 128, tiles, 1).
template <int BM, int NG>
[[kernel, max_total_threads_per_threadgroup(NX_THREADS)]] void fn_moe_down_i8(
    const device char*   hq     [[buffer(0)]],   // [pairs, I] permuted (grouped order)
    const device float*  hsc    [[buffer(1)]],   // [pares, I/64]
    const device uchar*  arena  [[buffer(2)]],
    const device uint*   smap   [[buffer(3)]],
    const device uint4*  tiles  [[buffer(4)]],
    device float*        y      [[buffer(5)]],
    constant FnQ2Params& p      [[buffer(6)]],
    uint3  tgid [[threadgroup_position_in_grid]],
    uint   tid  [[thread_index_in_threadgroup]],
    ushort sgid [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]])
{
    constexpr int FM = BM / 16;
    using AS = FnI8AStage<BM, NG>;
    constexpr int LD = fi8_ld<NG>();
    threadgroup char ws[128 * LD];
    threadgroup char xs[BM * LD];
    threadgroup float ds[128 * NG], xss[BM * NG];
    const uint4 td = tiles[tgid.y];
    const uint r0 = td.y, rows = td.z;
    const uint n0 = tgid.x * 128;
    const int tn = sgid * 32;
    const short2 c = nax_coord(lane);
    array<const device char*, AS::NI> rp;
    NAX_UNROLL for (int i = 0; i < AS::NI; i++) {
        const uint r = (tid + i * NX_THREADS) / AS::CPR;
        rp[i] = r < rows ? hq + size_t(r0 + r) * FN_I : nullptr;
    }
    const device float* sp = tid < rows && tid < uint(BM) ? hsc + size_t(r0 + tid) * (FN_I / 64) : nullptr;
    const device uchar* rec = arena + size_t(smap[td.x]) * p.rec;
    const device uint* w = (const device uint*)(rec + p.dc);
    const device half* s = (const device half*)(rec + p.ds);
    const int live = min(int(rows + 15) / 16, FM);
#define FDN8(L) fn_down_i8_body<BM, NG, L>(w, s, y, rp, sp, ws, ds, xs, xss, n0, r0, rows, tn, c, tid)
    if (FM >= 4 && live == 4) { if constexpr (FM >= 4) FDN8((4 > FM ? FM : 4)); }
    else if (FM >= 3 && live == 3) { if constexpr (FM >= 3) FDN8((3 > FM ? FM : 3)); }
    else if (FM >= 2 && live == 2) { if constexpr (FM >= 2) FDN8((2 > FM ? FM : 2)); }
    else FDN8(1);
#undef FDN8
}

#define FN_I8_INST(BM, NG) \
  template [[host_name("fn_moe_gateup_i8_bm" #BM "_ng" #NG)]] [[kernel]] void fn_moe_gateup_i8<BM, NG>( \
      const device char*, const device float*, const device uchar*, const device uint*, const device uint4*, \
      const device uint*, device char*, device float*, constant FnQ2Params&, uint3, uint, ushort, ushort); \
  template [[host_name("fn_moe_down_i8_bm" #BM "_ng" #NG)]] [[kernel]] void fn_moe_down_i8<BM, NG>( \
      const device char*, const device float*, const device uchar*, const device uint*, const device uint4*, \
      device float*, constant FnQ2Params&, uint3, uint, ushort, ushort);
FN_I8_INST(16, 1)
FN_I8_INST(32, 1)
FN_I8_INST(48, 1)
FN_I8_INST(64, 1)
