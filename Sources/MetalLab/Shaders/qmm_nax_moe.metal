// MoE prefill (Qwen3.6-35B-A3B, qwen3_5_moe) on the matrix accelerators (matmul2d).
// Concatenated after qmm_nax.metal (uses nax_*, NX_* and QmmParams).
//
// Portions adapted from oMLX and splash (Apache-2.0; see THIRD_PARTY_NOTICES.md); changed: rewritten in Metal with
// our own tile, dispatch and epilogue code.
// Ideas taken (Apache-2.0, with attribution):
//   - oMLX (jundot/omlx, patches/m5_gather_qmm_nax.py, moe_routes.py, quantized_moe.h): single-expert tiles
//     with descriptors written on the GPU, gate+up with SwiGLU in the epilogue, rows read through `row_map` without
//     copying them, and unsort + weighted sum in fp32 in one kernel.
//   - splash (incoai/splash, kernels/shared/moe.metal, runtime/ops/MoE.cpp): grouping in a single threadgroup with
//     atomics and prefixes, tile descriptors and tile count as arguments of an indirect dispatch,
//     and skipping the 16-row fragments with no live rows inside the tile.
// Router semantics (mlx-lm qwen3_next.Qwen3NextSparseMoeBlock, MIT): softmax → top-k → renormalization of the
// k weights (norm_topk_prob), shared expert weighted by sigmoid(shared_expert_gate(x)).

// ---------------------------------------------------------------------------------------------
// 1) Router top-k + shared expert gate. One simdgroup per row, 4 rows per threadgroup.
// logits [T, MOE_LD] fp32 (E = 32*VPL, TOPK ≤ 32; row stride E rounded up to 64, MOE_LD in ab_moe_route.metal). weight_k = exp(l_k - l_max) / Σ_top exp(l_j - l_max)
// (the denominator of the full softmax cancels). Tie-break: the lower index.
// Two policies for Flash-Next (those of the retired fn_moe_route, with the same bits): SG_LOGIT, the shared gate is
// σ(l[E]) (column E of the logits; the logits row stride then goes in p.D) instead of the product with
// shared_expert_gate; SUM_SEQ, Σ_top is summed in k order (instead of the simd_sum tree).

struct RouteParams { uint T; uint E; uint D; uint topk; };

template <int VPL, int TOPK, bool SG_LOGIT = false, bool SUM_SEQ = false>
[[kernel]] void moe_route_topk(
    const device float*  logits [[buffer(0)]],
    const device half*   x      [[buffer(1)]],
    const device uint*   wsg    [[buffer(2)]],   // shared_expert_gate 8 bits [1, D]
    const device bfloat* ssg    [[buffer(3)]],
    const device bfloat* bsg    [[buffer(4)]],
    device uint*         inds   [[buffer(5)]],   // [T, TOPK]
    device float*        wts    [[buffer(6)]],   // [T, TOPK]
    device float*        sgate  [[buffer(7)]],   // [T]
    constant RouteParams& p     [[buffer(8)]],
    const device float*  bias   [[buffer(9)]],   // selection bias [E] (LAB_MOE_BIAS) or router bias (LAB_MOE_LBIAS)
    const device float*  pes    [[buffer(10)]],  // per-expert scale [E] (LAB_MOE_PES, Gemma 4)
    uint   tg   [[threadgroup_position_in_grid]],
    ushort sg   [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]])
{
    const uint t = tg * 4 + sg;
    if (t >= p.T) return;
    const device float* lr = logits + size_t(t) * (SG_LOGIT ? p.D : MOE_LD);
#if LAB_MOE_SCORE == 1
    // Sigmoid router: moe_route_sigmoid (ab_moe_route.metal); not in Flash-Next (SG_LOGIT).
    if constexpr (!SG_LOGIT) {
        float w; uint i;
        moe_route_sigmoid<VPL, TOPK>(lr, bias, lane, i, w);
        if (lane < TOPK) {
            inds[t * TOPK + lane] = i;
            wts[t * TOPK + lane] = w;
        }
    } else
#endif
    {
        float v[VPL];
#if LAB_MOE_LBIAS
        NAX_UNROLL for (int j = 0; j < VPL; j++) v[j] = lr[j * 32 + lane] + bias[j * 32 + lane];   // router bias
#else
        NAX_UNROLL for (int j = 0; j < VPL; j++) v[j] = lr[j * 32 + lane];
#endif
        float myv = -INFINITY;
        uint myi = 0;
        NAX_UNROLL for (int k = 0; k < TOPK; k++) {
            float bv = -INFINITY;
            uint bi = 0xffffffffu;
            NAX_UNROLL for (int j = 0; j < VPL; j++) if (v[j] > bv) { bv = v[j]; bi = j * 32 + lane; }
            const float gv = simd_max(bv);
            const uint gi = simd_min(bv == gv ? bi : 0xffffffffu);
            NAX_UNROLL for (int j = 0; j < VPL; j++) if (uint(j * 32 + lane) == gi) v[j] = -INFINITY;
            if (lane == k) { myv = gv; myi = gi; }
        }
        const float top1 = simd_broadcast(myv, 0);
        const float ex = lane < TOPK ? exp(myv - top1) : 0.0f;
        float sum;
        if constexpr (SUM_SEQ) {
            sum = 0;
            NAX_UNROLL for (int k = 0; k < TOPK; k++) sum += simd_broadcast(ex, k);
        } else {
            sum = simd_sum(ex);
        }
    #ifdef LAB_MOE_RSCALE
        if (!SG_LOGIT) sum /= LAB_MOE_RSCALE;   // routed_scaling_factor
    #endif
        if (lane < TOPK) {
            inds[t * TOPK + lane] = myi;
#if LAB_MOE_PES
            wts[t * TOPK + lane] = ex / sum * pes[myi];   // per_expert_scale after the renormalization
#else
            wts[t * TOPK + lane] = ex / sum;
#endif
        }
        if constexpr (SG_LOGIT) {
            if (lane == 0) sgate[t] = 1.0f / (1.0f + exp(-lr[p.E]));
            return;
        }
    }
#if LAB_MOE_SHARED != 1
    // Shared expert without gate (weight 1) or none (sgate is not read).
    if (lane == 0) sgate[t] = 1.0f;
    return;
#endif
    // Shared expert gate: dot product of x[t] with an 8-bit row (4 values per uint).
    float acc = 0;
    const device half* xr = x + size_t(t) * p.D;
    for (uint k = lane * 4; k < p.D; k += 128) {
        const uint w = wsg[k / 4];
        const float sc = float(ssg[k / 64]), bi = float(bsg[k / 64]);
        const half4 xv = *(const device half4*)(xr + k);
        NAX_UNROLL for (int j = 0; j < 4; j++) acc += float(xv[j]) * (sc * float((w >> (8 * j)) & 0xffu) + bi);
    }
    acc = simd_sum(acc);
    if (lane == 0) sgate[t] = 1.0f / (1.0f + exp(-acc));
}

// E and top-k from the config (LAB_MOE_E, LAB_MOE_K); _fn: with the Flash-Next policies.
template [[host_name("moe_route_topk_cfg")]] [[kernel]] void moe_route_topk<LAB_MOE_E / 32, LAB_MOE_K>(
    const device float*, const device half*, const device uint*, const device bfloat*, const device bfloat*,
    device uint*, device float*, device float*, constant RouteParams&, const device float*, const device float*, uint,
    ushort, ushort);
template [[host_name("moe_route_topk_fn")]] [[kernel]] void moe_route_topk<LAB_MOE_E / 32, LAB_MOE_K, true, true>(
    const device float*, const device half*, const device uint*, const device bfloat*, const device bfloat*,
    device uint*, device float*, device float*, constant RouteParams&, const device float*, const device float*, uint,
    ushort, ushort);

// ---------------------------------------------------------------------------------------------
// 2) Group the (row, k) pairs by expert, all in one threadgroup of 1024 threads:
//    count with atomics in threadgroup → prefixes (rows and tiles) → sorted position of each pair →
//    row_map (sorted position → pair) and pos (pair → sorted position) → tile descriptors
//    uint4(expert, start, rows, 0) for two tile sizes (gate+up and down) and the arguments of the
//    indirect dispatches (nx, number of tiles, 1). Within an expert the order depends on the atomics, but each
//    row is computed independently in the GEMMs, so the result does not change.

// T > 0: the shared expert enters as expert E with its T rows at positions P..P+T-1 (idea from splash,
// moe.metal: "shared expert as one more expert"); its tiles go first if flags bit 0. Flags bit 1 (the Flash-Next
// round, no shared expert): only the pairs of the first st[6] rows, with top-k = (flags >> 8) & 0xff.
struct GroupParams { uint P; uint E; uint bmA; uint bmB; uint nxA; uint nxB; uint T; uint flags; };

constant constexpr uint MOE_MAXE = 1024, MOE_GT = 1024;

METAL_FUNC uint tg_exclusive_scan(uint v, threadgroup uint* tmp, ushort sgid, ushort lane, thread uint& total) {
    const uint inc = simd_prefix_exclusive_sum(v);
    if (lane == 31) tmp[sgid] = inc + v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0) {
        const uint s = tmp[lane];
        tmp[lane] = simd_prefix_exclusive_sum(s);
        if (lane == 31) tmp[32] = tmp[lane] + s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint r = inc + tmp[sgid];
    total = tmp[32];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r;
}

[[kernel, max_total_threads_per_threadgroup(MOE_GT)]] void moe_group(
    const device uint* inds    [[buffer(0)]],   // [P] expert of each pair
    device uint*       offsets [[buffer(1)]],   // [E+1]
    device uint*       rowMap  [[buffer(2)]],   // [P] sorted position → pair
    device uint*       pos     [[buffer(3)]],   // [P] pair → sorted position
    device uint4*      tilesA  [[buffer(4)]],
    device uint4*      tilesB  [[buffer(5)]],
    device uint*       argsA   [[buffer(6)]],   // MTLDispatchThreadgroupsIndirectArguments
    device uint*       argsB   [[buffer(7)]],
    constant GroupParams& p    [[buffer(8)]],
    const device uint* st      [[buffer(9)]],   // round state (flags bit 1)
    uint   tid  [[thread_index_in_threadgroup]],
    ushort sgid [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]])
{
    threadgroup atomic_uint cnt[MOE_MAXE];
    threadgroup uint offs[MOE_MAXE];
    threadgroup uint tmp[33];
    const uint P = (p.flags & 2u) ? min(p.P, st[6] * ((p.flags >> 8) & 0xffu)) : p.P;
    for (uint e = tid; e < p.E; e += MOE_GT) atomic_store_explicit(&cnt[e], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < P; i += MOE_GT)
        pos[i] = atomic_fetch_add_explicit(&cnt[inds[i]], 1u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // E ≤ 1024: one thread per expert.
    const uint c = tid < p.E ? atomic_load_explicit(&cnt[tid], memory_order_relaxed) : 0u;
    const uint ta = (c + p.bmA - 1) / p.bmA, tb = (c + p.bmB - 1) / p.bmB;
    uint totC, totA, totB;
    const uint exC = tg_exclusive_scan(c, tmp, sgid, lane, totC);
    const uint exA = tg_exclusive_scan(ta, tmp, sgid, lane, totA);
    const uint exB = tg_exclusive_scan(tb, tmp, sgid, lane, totB);
    if (tid < p.E) { offs[tid] = exC; offsets[tid] = exC; }
    const uint nsA = (p.T + p.bmA - 1) / p.bmA, nsB = (p.T + p.bmB - 1) / p.bmB;
    const bool sharedFirst = p.flags & 1u;
    const uint shA = sharedFirst ? nsA : 0, shB = sharedFirst ? nsB : 0;
    if (tid == 0) {
        offsets[p.E] = totC;
        argsA[0] = p.nxA; argsA[1] = totA + nsA; argsA[2] = 1;
        argsB[0] = p.nxB; argsB[1] = totB + nsB; argsB[2] = 1;
    }
    if (p.T > 0) {
        const uint topk = p.P / p.T, a0 = sharedFirst ? 0 : totA, b0 = sharedFirst ? 0 : totB;
        for (uint t = tid; t < p.T; t += MOE_GT) rowMap[p.P + t] = t * topk;
        for (uint j = tid; j < nsA; j += MOE_GT) tilesA[a0 + j] = uint4(p.E, p.P + j * p.bmA, min(p.bmA, p.T - j * p.bmA), 0);
        for (uint j = tid; j < nsB; j += MOE_GT) tilesB[b0 + j] = uint4(p.E, p.P + j * p.bmB, min(p.bmB, p.T - j * p.bmB), 0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < P; i += MOE_GT) {
        const uint s = offs[inds[i]] + pos[i];
        pos[i] = s;
        rowMap[s] = i;
    }
    if (tid < p.E) {
        for (uint j = 0; j < ta; j++) tilesA[shA + exA + j] = uint4(tid, exC + j * p.bmA, min(p.bmA, c - j * p.bmA), 0);
        for (uint j = 0; j < tb; j++) tilesB[shB + exB + j] = uint4(tid, exC + j * p.bmB, min(p.bmB, c - j * p.bmB), 0);
    }
}

// ---------------------------------------------------------------------------------------------
// 3) Grouped expert GEMM. One tile = one expert × BM rows (the live ones may be fewer) × BN columns.
// The kernels (v2) are further down, after the combine; v1 (each simdgroup read its rows from memory) was retired.

// E: id of the shared expert (if a tile carries e == E it uses the shared expert's weights; see moe_group). sNK: the N
// (gate+up) or the K (down) of the shared expert, its intermediate size; P: row of h where its rows start (they go after
// those of the pairs with stride sNK; with sNK = that of the experts, the same positions as before).
struct MoeGemmParams { uint N; uint K; uint topk; uint E; uint sNK; uint P; };

// Two-phase weight loader (idea from splash, gguf_staged_tile.h: prefetch into registers + double buffer):
// `load` reads the next K block from memory into registers (packed uint32 + scale and bias) and `store`
// dequantizes it into threadgroup. This way the DRAM latency of step k+1 overlaps with the matmul2d of step k.
template <int bits, int G, int rows, int BK = NX_BK>
struct NaxWStage {
    static constant constexpr uint per = 32 / bits, upr = BK / per, NI = rows * upr / NX_THREADS;
    uint v[NI];
    bfloat sc[NI], bi[NI];

    METAL_FUNC void load(const device uint* w, const device bfloat* s, const device bfloat* b,
                         uint K, uint n0, uint k0, uint tid) {
        NAX_UNROLL for (uint i = 0; i < NI; i++) {
            const uint idx = tid + i * NX_THREADS;
            const uint r = idx / upr, u = idx % upr;
            const uint k = k0 + u * per;
            const uint g = (n0 + r) * (K / G) + k / G;
            v[i] = w[size_t(n0 + r) * (K / per) + k / per];
            sc[i] = s[g];
            bi[i] = b[g];
        }
    }

    // From a weight source (AffineSrc).
    template <class S>
    METAL_FUNC void load(const thread S& src, uint K, uint n0, uint k0, uint tid) {
        load(src.w, src.s, src.b, K, n0, k0, tid);
    }

    METAL_FUNC void store(threadgroup half* ws, uint tid) const {
        constexpr int LD = BK + 8;
        NAX_UNROLL for (uint i = 0; i < NI; i++) {
            const uint idx = tid + i * NX_THREADS;
            const uint r = idx / upr, u = idx % upr;
            threadgroup half* o = ws + r * LD + u * per;
            if constexpr (bits == 4) {
                const half2 s2 = half2(half(float(sc[i]))), b2 = half2(half(float(bi[i])));
                const uint x = v[i];
                const half2 q0 = fma(as_type<half2>((x & 0x000F000Fu) | 0x64006400u) - half2(1024.0h), s2, b2);
                const half2 q1 = fma(as_type<half2>(((x >> 4) & 0x000F000Fu) | 0x64006400u) - half2(1024.0h), s2, b2);
                const half2 q2 = fma(as_type<half2>(((x >> 8) & 0x000F000Fu) | 0x64006400u) - half2(1024.0h), s2, b2);
                const half2 q3 = fma(as_type<half2>(((x >> 12) & 0x000F000Fu) | 0x64006400u) - half2(1024.0h), s2, b2);
                *(threadgroup half4*)o = half4(q0.x, q1.x, q2.x, q3.x);
                *(threadgroup half4*)(o + 4) = half4(q0.y, q1.y, q2.y, q3.y);
            } else {
                const half s1 = half(float(sc[i])), b1 = half(float(bi[i]));
                for (uint j = 0; j < per; j++) o[j] = s1 * half((v[i] >> (bits * j)) & ((1u << bits) - 1)) + b1;
            }
        }
    }
};

// MXFP4 (mlx mode "mxfp4", LAB_MXFP4 instances): 4-bit e2m1 codes (sign in bit 3) packed like the affine 4 bits, one
// uint8 e8m0 scale 2^(e − 127) per group of 32, no bias. lab_fp4x2 decodes the low nibbles of both halves of v (bits
// 0–3 and 16–19, the order of the affine 0x6400 trick) with the trick of mlx's fp_quantized.h (MIT, Apple; copied, see THIRD_PARTY_NOTICES.md): (q & 7) << 9 as half bits is
// |fp4(q)| · 2⁻¹⁴ (exact; 0.5 is a half subnormal) and bit 3 goes to the sign bit. The 2¹⁴ and the scale (2⁻¹¹ … 2⁸ in
// gpt-oss) are one fp32 factor per group (lab_e8m0x): folded into a half they overflow (2⁸ · 2¹⁴), and as two half
// products fast math may reorder them into an underflow. Every weight of gpt-oss fits in half (at most 6 · 2⁸).
inline float lab_e8m0x(uchar e) { return as_type<float>(uint(e) << 23) * 16384.0f; }
inline half2 lab_fp4x2(uint v, float s) {
    return half2(float2(as_type<half2>(((v & 0x00070007u) << 9) | ((v & 0x00080008u) << 12))) * s);
}

// Two-phase loader of MXFP4 weights (NaxWStage's interface): the packed codes and the group's e8m0 scale in registers,
// dequantized into threadgroup by `store` in the order of the affine one (8 values per uint32, all in one group of 32).
template <int rows, int BK = NX_BK>
struct NaxF4Stage {
    static constant constexpr uint per = 8, upr = BK / per, NI = rows * upr / NX_THREADS;
    uint v[NI];
    uchar sc[NI];

    METAL_FUNC void load(const device uint* w, const device uchar* s, uint K, uint n0, uint k0, uint tid) {
        NAX_UNROLL for (uint i = 0; i < NI; i++) {
            const uint idx = tid + i * NX_THREADS;
            const uint r = idx / upr, u = idx % upr;
            const uint k = k0 + u * per;
            v[i] = w[size_t(n0 + r) * (K / per) + k / per];
            sc[i] = s[size_t(n0 + r) * (K / 32) + k / 32];
        }
    }

    template <class S>
    METAL_FUNC void load(const thread S& src, uint K, uint n0, uint k0, uint tid) {
        load(src.w, src.s, K, n0, k0, tid);
    }

    METAL_FUNC void store(threadgroup half* ws, uint tid) const {
        constexpr int LD = BK + 8;
        NAX_UNROLL for (uint i = 0; i < NI; i++) {
            const uint idx = tid + i * NX_THREADS;
            const uint r = idx / upr, u = idx % upr;
            threadgroup half* o = ws + r * LD + u * per;
            const float s = lab_e8m0x(sc[i]);
            const uint x = v[i];
            const half2 q0 = lab_fp4x2(x, s), q1 = lab_fp4x2(x >> 4, s), q2 = lab_fp4x2(x >> 8, s),
                        q3 = lab_fp4x2(x >> 12, s);
            *(threadgroup half4*)o = half4(q0.x, q1.x, q2.x, q3.x);
            *(threadgroup half4*)(o + 4) = half4(q0.y, q1.y, q2.y, q3.y);
        }
    }
};

// MXFP4 weight source for the shared NAX bodies (moe_gateup_body, moe_down_body).
struct Mxfp4Src {
    const device uint* w;
    const device uchar* s;
    template <int rows, int BK> using Stage = NaxF4Stage<rows, BK>;
    static constant constexpr int kbits = 4;
};

// MLX affine weight source (bits, groups of G, bf16 scale and bias) for the shared NAX bodies
// (moe_gateup_body, moe_down_body); the Flash-Next one (Q2 of our format) is FnQ2Src, in zf_flashnext.metal.
template <int bits, int G>
struct AffineSrc {
    const device uint* w;
    const device bfloat* s;
    const device bfloat* b;
    template <int rows, int BK> using Stage = NaxWStage<bits, G, rows, BK>;
    static constant constexpr int kbits = bits;
};

// Expert weight source of moe_gateup_swiglu2 and moe_down2: affine, or MXFP4 with FP4 (its uint8 scales bound where the
// bf16 scales go; `so` is the expert's offset in groups, bytes for MXFP4; the bias buffer is not read).
template <bool FP4, int bits, int G>
struct ExpertSrc {
    using T = AffineSrc<bits, G>;
    static METAL_FUNC T make(const device uint* w, const device bfloat* s, const device bfloat* b, size_t so) {
        return T{w, s + so, b + so};
    }
};
template <int bits, int G>
struct ExpertSrc<true, bits, G> {
    using T = Mxfp4Src;
    static METAL_FUNC T make(const device uint* w, const device bfloat* s, const device bfloat*, size_t so) {
        return T{w, (const device uchar*)s + so};
    }
};

// ---------------------------------------------------------------------------------------------
// 4) Unsort + weighted sum (fp32) + shared expert with its gate: one threadgroup of 256 threads per row.
// out[t] = Σ_k w[t,k]·Yd[pos[t,k]] + sgate[t]·Ys[t]

struct CombineParams { uint D; uint topk; };

[[kernel]] void moe_combine(
    const device half*  yd    [[buffer(0)]],
    const device uint*  pos   [[buffer(1)]],
    const device float* wts   [[buffer(2)]],
    const device half*  ys    [[buffer(3)]],
    const device float* sgate [[buffer(4)]],
    device half*        out   [[buffer(5)]],
    constant CombineParams& p [[buffer(6)]],
    uint t   [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint ntg [[threads_per_threadgroup]])
{
    const float g = sgate[t];
    for (uint c = tid * 8; c < p.D; c += ntg * 8) {
        const device half* sp = ys + size_t(t) * p.D + c;
        float4 a0 = g * float4(*(const device half4*)sp), a1 = g * float4(*(const device half4*)(sp + 4));
        for (uint k = 0; k < p.topk; k++) {
            const float w = wts[t * p.topk + k];
            const device half* yp = yd + size_t(pos[t * p.topk + k]) * p.D + c;
            a0 += w * float4(*(const device half4*)yp);
            a1 += w * float4(*(const device half4*)(yp + 4));
        }
        *(device half4*)(out + size_t(t) * p.D + c) = half4(a0);
        *(device half4*)(out + size_t(t) * p.D + c + 4) = half4(a1);
    }
}

// ---------------------------------------------------------------------------------------------
// v2: the tile rows (X through row_map, or contiguous H) are copied once per K step into threadgroup and are
// shared by the 4 simdgroups; each simdgroup does FM fragments of rows × FN pairs of column fragments.
// In v1 each simdgroup read its rows from memory for every 32 columns: the X traffic was ~4× that of the weights.
// Weights: prefetch of the next step into registers (NaxWStage) and a single threadgroup buffer.

constant constexpr int NX_LDA = NX_BK + 8;

// Load of BM rows × 64 columns (in registers: 16 B per chunk, BM*8/128 chunks per thread).
template <int BM, int BK = NX_BK>
struct NaxAStage {
    static constant constexpr int CPR = BK / 8, NI = (BM * CPR + NX_THREADS - 1) / NX_THREADS;
    half4 v[NI][2];
    METAL_FUNC void load(const thread array<const device half*, NI>& rp, uint k0, uint tid) {
        NAX_UNROLL for (int i = 0; i < NI; i++) {
            const uint ch = tid + i * NX_THREADS;
            if (ch < uint(BM * CPR) && rp[i]) {
                const device half* src = rp[i] + k0 + (ch % CPR) * 8;
                v[i][0] = *(const device half4*)src;
                v[i][1] = *(const device half4*)(src + 4);
            }
        }
    }
    METAL_FUNC void store(threadgroup half* xs, uint tid) const {
        NAX_UNROLL for (int i = 0; i < NI; i++) {
            const uint ch = tid + i * NX_THREADS;
            if (ch < uint(BM * CPR)) {
                threadgroup half* o = xs + (ch / CPR) * (BK + 8) + (ch % CPR) * 8;
                *(threadgroup half4*)o = v[i][0];
                *(threadgroup half4*)(o + 4) = v[i][1];
            }
        }
    }
};

// Bodies of the grouped GEMMs on NAX, shared by the Qwen3.6 MoE (AffineSrc, moe_gateup_swiglu2 and
// moe_down2) and the Flash-Next one (FnQ2Src, fn_moe_gateup_nax and fn_moe_down_nax). WSrc gives the two-phase
// weight loader (WSrc::Stage); KC and NC fix K and N at compile time (0: those of the arguments). LIVE live 16-row
// fragments as a compile-time constant: with a dynamic `if (i < live)` inside the unrolled loops, the fragment
// arrays go to private memory (from 2.3 to 5–20 ms). With SGM = 1 all simdgroups have the same rows, so
// the choice is uniform across the threadgroup.
// gate|up: GU_HALF rounds gate and up to half before SiLU (Qwen3.6); without it, in fp32 (Flash-Next).
// ACT: activation of the gated FFN (lab_glu). gb/ub (LAB_MOE_EBIAS): the expert's gate and up biases (fp32 [N], null
// for the shared expert), added before the activation.
template <class WSrc, bool GU_HALF, int BM, int FN, int BK, int PF, int LIVE, uint KC = 0, uint NC = 0, int ACT = 0>
METAL_FUNC void moe_gateup_body(
    const thread WSrc& wg, const thread WSrc& wu, device half* h,
    const thread array<const device half*, NaxAStage<BM, BK>::NI>& rp,
    threadgroup half* wgs, threadgroup half* wus, threadgroup half* xs,
    uint Kr, uint Nr, uint n0, uint r0, uint rows, int tn, short2 c, uint tid,
    const device float* gb = nullptr, const device float* ub = nullptr)
{
    constexpr int BN = 64 * FN;
    const uint K = KC ? KC : Kr, N = NC ? NC : Nr;
    nax_ffrag ag[LIVE][FN], au[LIVE][FN];
    NAX_UNROLL for (int i = 0; i < LIVE; i++) NAX_UNROLL for (int j = 0; j < FN; j++) { ag[i][j] = 0; au[i][j] = 0; }
    constexpr int LDW = BK + 8, LDA = BK + 8, KF = BK / 16;
    // PF K steps in flight in registers (PF = 2: steps k+1 and k+2 are loaded while k is computed).
    typename WSrc::template Stage<BN, BK> stg[PF], stu[PF];
    NaxAStage<BM, BK> sta[PF];
    NAX_UNROLL for (int q = 0; q < PF; q++) if (uint(q * BK) < K) {
        if constexpr (ACT != 3) stg[q].load(wg, K, n0, q * BK, tid);
        stu[q].load(wu, K, n0, q * BK, tid);
        sta[q].load(rp, q * BK, tid);
    }
    for (uint k00 = 0; k00 < K; k00 += PF * BK) NAX_UNROLL for (int q = 0; q < PF; q++) {
        const uint k0 = k00 + q * BK;
        if (k0 >= K) break;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if constexpr (ACT != 3) stg[q].store(wgs, tid);
        stu[q].store(wus, tid);
        sta[q].store(xs, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (k0 + PF * BK < K) {
            if constexpr (ACT != 3) stg[q].load(wg, K, n0, k0 + PF * BK, tid);
            stu[q].load(wu, K, n0, k0 + PF * BK, tid);
            sta[q].load(rp, k0 + PF * BK, tid);
        }
        nax_hfrag bgf[FN][KF], buf[FN][KF];
        NAX_UNROLL for (int kk = 0; kk < KF; kk++) NAX_UNROLL for (int j = 0; j < FN; j++) {
            if constexpr (ACT != 3) nax_load(bgf[j][kk], wgs + (tn + j * 16) * LDW + kk * 16, LDW, c);
            nax_load(buf[j][kk], wus + (tn + j * 16) * LDW + kk * 16, LDW, c);
        }
        // A is loaded just before each matmul2d (fewer live registers than loading it whole).
        NAX_UNROLL for (int i = 0; i < LIVE; i++) NAX_UNROLL for (int j = 0; j < FN; j++)
            NAX_UNROLL for (int kk = 0; kk < KF; kk++) {
                nax_hfrag a;
                nax_load(a, xs + (i * 16) * LDA + kk * 16, LDA, c);
                // relu² without gate (ACT 3): up only; the second half of the matmul goes to ag, which nobody reads.
                if constexpr (ACT == 3) nax_mma2(ag[i][j], au[i][j], a, buf[j][kk], buf[j][kk]);
                else nax_mma2(ag[i][j], au[i][j], a, bgf[j][kk], buf[j][kk]);
            }
    }
    NAX_UNROLL for (int i = 0; i < LIVE; i++) NAX_UNROLL for (int j = 0; j < FN; j++) NAX_UNROLL for (int q = 0; q < 2; q++) {
        const uint r = i * 16 + c.x + q * 8;
        if (r >= rows) continue;
        float4 g = float4(ag[i][j][q * 4], ag[i][j][q * 4 + 1], ag[i][j][q * 4 + 2], ag[i][j][q * 4 + 3]);
        float4 u = float4(au[i][j][q * 4], au[i][j][q * 4 + 1], au[i][j][q * 4 + 2], au[i][j][q * 4 + 3]);
#if LAB_MOE_EBIAS
        if (gb) {
            const uint n = n0 + tn + j * 16 + c.y;
            g += *(const device float4*)(gb + n);
            u += *(const device float4*)(ub + n);
        }
#endif
        if constexpr (GU_HALF) { g = float4(half4(g)); u = float4(half4(u)); }
        *(device half4*)(h + size_t(r0 + r) * N + n0 + tn + j * 16 + c.y) = half4(lab_glu<ACT>(g, u));
    }
}

// BM × (64·FN) tiles: the 4 simdgroups share the BM rows and split the columns (16·FN each).
// ACT: activation (lab_glu; `moe_gateup_swiglu2_gelu_*` instances with LAB_FFN_GELU, `_relu2_` with LAB_FFN_RELU2: no
// gate, its buffers bound but not read). FP4: MXFP4 experts (`_mxfp4`
// instances, LAB_MXFP4; G = 32), no shared expert.
template <int bits, int G, int BM, int FN, int BK, int PFX, int ACT = 0, bool FP4 = false>
[[kernel, max_total_threads_per_threadgroup(NX_THREADS)]] void moe_gateup_swiglu2(
    const device half*   x      [[buffer(0)]],
    const device uint*   wg     [[buffer(1)]],
    const device bfloat* sg     [[buffer(2)]],
    const device bfloat* bg     [[buffer(3)]],
    const device uint*   wu     [[buffer(4)]],
    const device bfloat* su     [[buffer(5)]],
    const device bfloat* bu     [[buffer(6)]],
    device half*         h      [[buffer(7)]],
    const device uint*   rowMap [[buffer(8)]],
    const device uint4*  tiles  [[buffer(9)]],
    constant MoeGemmParams& p   [[buffer(10)]],
    const device uint*   wgS    [[buffer(11)]],
    const device bfloat* sgS    [[buffer(12)]],
    const device bfloat* bgS    [[buffer(13)]],
    const device uint*   wuS    [[buffer(14)]],
    const device bfloat* suS    [[buffer(15)]],
    const device bfloat* buS    [[buffer(16)]],
    const device uint*   smap   [[buffer(17)]],   // expert → slot
    const device float*  ebg    [[buffer(18)]],   // gate and up biases [E, N] fp32 (LAB_MOE_EBIAS)
    const device float*  ebu    [[buffer(19)]],
    uint3  tgid [[threadgroup_position_in_grid]],
    uint   tid  [[thread_index_in_threadgroup]],
    ushort sgid [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]])
{
    // PFX = prefetch depth.
    constexpr int FM = BM / 16, BN = 64 * FN, PF = PFX;
    using AS = NaxAStage<BM, BK>;
    constexpr int LDW = BK + 8, LDA = BK + 8;
    threadgroup half wgs[BN * LDW];
    threadgroup half wus[BN * LDW];
    threadgroup half xs[BM * LDA];
    const uint4 td = tiles[tgid.y];
    const uint e = td.x, r0 = td.y, rows = td.z;
    const uint n0 = tgid.x * BN;
    const int tn = sgid * (16 * FN);
    const short2 c = nax_coord(lane);
    array<const device half*, AS::NI> rp;
    NAX_UNROLL for (int i = 0; i < AS::NI; i++) {
        const uint r = (tid + i * NX_THREADS) / AS::CPR;
        rp[i] = r < rows ? x + size_t(rowMap[r0 + r] / p.topk) * p.K : nullptr;
    }
    // The shared expert writes its h rows after those of the pairs (from P), with stride sNK (its intermediate size); the
    // grid covers the larger of the two widths and the extra tiles exit here.
    device half* hh = h;
    uint N = p.N, rh = r0;
    if (e == p.E) {
        N = p.sNK; hh = h + size_t(p.P) * p.N; rh = r0 - p.P;
    }
    if (n0 >= N) return;
    using XS = ExpertSrc<FP4, bits, G>;
    using Src = typename XS::T;
    Src g, u;
    if (e == p.E) {
        g = XS::make(wgS, sgS, bgS, 0); u = XS::make(wuS, suS, buS, 0);
    } else {
        const size_t se = smap[e];
        const size_t wOff = se * p.N * (p.K * bits / 32), sOff = se * p.N * (p.K / G);
        g = XS::make(wg + wOff, sg, bg, sOff); u = XS::make(wu + wOff, su, bu, sOff);
    }
    const int live = min(int(rows + 15) / 16, FM);
#if LAB_MOE_EBIAS
    const device float *gb = e < p.E ? ebg + size_t(e) * p.N : nullptr, *ub = e < p.E ? ebu + size_t(e) * p.N : nullptr;
#define GU2(L) moe_gateup_body<Src, true, BM, FN, BK, PF, L, 0, 0, ACT>(g, u, hh, rp, wgs, wus, xs, p.K, N, n0, rh, rows, tn, c, tid, gb, ub)
#else
#define GU2(L) moe_gateup_body<Src, true, BM, FN, BK, PF, L, 0, 0, ACT>(g, u, hh, rp, wgs, wus, xs, p.K, N, n0, rh, rows, tn, c, tid)
#endif
    if (FM >= 4 && live == 4) { if constexpr (FM >= 4) GU2((4 > FM ? FM : 4)); }
    else if (FM >= 3 && live == 3) { if constexpr (FM >= 3) GU2((3 > FM ? FM : 3)); }
    else if (FM >= 2 && live == 2) { if constexpr (FM >= 2) GU2((2 > FM ? FM : 2)); }
    else GU2(1);
#undef GU2
}

// down: output OutT (half in Qwen3.6, fp32 in Flash-Next). yb (LAB_MOE_EBIAS): the expert's down bias (fp32 [N], null
// for the shared expert), added before the output rounding. skip (LAB_MOE_H64): columns of the tile below n0 + skip are
// not written (the last tile, shifted back so it ends at N, recomputes columns of the previous one).
template <class WSrc, class OutT, int BM, int FN, int BK, int PF, int LIVE, uint KC = 0, uint NC = 0>
METAL_FUNC void moe_down_body(
    const thread WSrc& w, device OutT* y, const thread array<const device half*, NaxAStage<BM, BK>::NI>& rp,
    threadgroup half* ws, threadgroup half* xs, uint Kr, uint Nr, uint n0, uint r0, uint rows, int tn, short2 c,
    uint tid, const device float* yb = nullptr, uint skip = 0)
{
    constexpr int BN = 128 * FN;
    const uint K = KC ? KC : Kr, N = NC ? NC : Nr;
    nax_ffrag acc[LIVE][2 * FN];
    NAX_UNROLL for (int i = 0; i < LIVE; i++) NAX_UNROLL for (int j = 0; j < 2 * FN; j++) acc[i][j] = 0;
    constexpr int LDW = BK + 8, LDA = BK + 8, KF = BK / 16;
    typename WSrc::template Stage<BN, BK> stw[PF];
    NaxAStage<BM, BK> sta[PF];
    NAX_UNROLL for (int q = 0; q < PF; q++) if (uint(q * BK) < K) {
        stw[q].load(w, K, n0, q * BK, tid);
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
            stw[q].load(w, K, n0, k0 + PF * BK, tid);
            sta[q].load(rp, k0 + PF * BK, tid);
        }
        nax_hfrag bt[2 * FN][KF];
        NAX_UNROLL for (int kk = 0; kk < KF; kk++)
            NAX_UNROLL for (int j = 0; j < 2 * FN; j++) nax_load(bt[j][kk], ws + (tn + j * 16) * LDW + kk * 16, LDW, c);
        NAX_UNROLL for (int i = 0; i < LIVE; i++) NAX_UNROLL for (int j = 0; j < FN; j++)
            NAX_UNROLL for (int kk = 0; kk < KF; kk++) {
                nax_hfrag a;
                nax_load(a, xs + (i * 16) * LDA + kk * 16, LDA, c);
                nax_mma2(acc[i][2 * j], acc[i][2 * j + 1], a, bt[2 * j][kk], bt[2 * j + 1][kk]);
            }
    }
    using Out4 = vec<OutT, 4>;
    NAX_UNROLL for (int i = 0; i < LIVE; i++) NAX_UNROLL for (int j = 0; j < 2 * FN; j++) NAX_UNROLL for (int q = 0; q < 2; q++) {
        const uint r = i * 16 + c.x + q * 8;
        if (r >= rows) continue;
#if LAB_MOE_H64
        if (uint(tn + j * 16) < skip) continue;
#endif
#if LAB_MOE_EBIAS
        if (yb) {
            const float4 v = float4(acc[i][j][q * 4], acc[i][j][q * 4 + 1], acc[i][j][q * 4 + 2], acc[i][j][q * 4 + 3])
                + *(const device float4*)(yb + n0 + tn + j * 16 + c.y);
            *(device Out4*)(y + size_t(r0 + r) * N + n0 + tn + j * 16 + c.y) = Out4(v);
            continue;
        }
#endif
        *(device Out4*)(y + size_t(r0 + r) * N + n0 + tn + j * 16 + c.y) =
            Out4(acc[i][j][q * 4], acc[i][j][q * 4 + 1], acc[i][j][q * 4 + 2], acc[i][j][q * 4 + 3]);
    }
}

// BM × (128·FN) tiles: the 4 simdgroups share the BM rows and split the columns (32·FN each). FP4: MXFP4 experts
// (`_mxfp4` instances, LAB_MXFP4; G = 32), no shared expert, with an fp32 output (YT): gpt-oss's down weights reach 3072
// and its down biases ~700, so an expert's output can pass half's 65504 where mlx-lm's bf16 does not overflow.
template <int bits, int G, int BM, int FN, int BK, int PFX, bool FP4 = false, typename YT = half>
[[kernel, max_total_threads_per_threadgroup(NX_THREADS)]] void moe_down2(
    const device half*   hin    [[buffer(0)]],
    const device uint*   w      [[buffer(1)]],
    const device bfloat* s      [[buffer(2)]],
    const device bfloat* b      [[buffer(3)]],
    device YT*           y      [[buffer(4)]],
    const device uint4*  tiles  [[buffer(5)]],
    constant MoeGemmParams& p   [[buffer(6)]],
    const device uint*   wS     [[buffer(7)]],
    const device bfloat* sS     [[buffer(8)]],
    const device bfloat* bS     [[buffer(9)]],
    const device uint*   smap   [[buffer(10)]],   // expert → slot
    const device float*  ebd    [[buffer(11)]],   // down biases [E, N] fp32 (LAB_MOE_EBIAS)
    uint3  tgid [[threadgroup_position_in_grid]],
    uint   tid  [[thread_index_in_threadgroup]],
    ushort sgid [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]])
{
    constexpr int FM = BM / 16, BN = 128 * FN, PF = PFX;
    using AS = NaxAStage<BM, BK>;
    constexpr int LDW = BK + 8, LDA = BK + 8;
    threadgroup half ws[BN * LDW];
    threadgroup half xs[BM * LDA];
    const uint4 td = tiles[tgid.y];
    const uint e = td.x, r0 = td.y, rows = td.z;
#if LAB_MOE_H64
    // Hidden not a multiple of BN (N ≥ BN): the last tile ends at N and skips the columns the previous one wrote.
    uint n0 = tgid.x * BN, skip = 0;
    if (n0 + BN > p.N) { skip = n0 + BN - p.N; n0 = p.N - BN; }
#else
    const uint n0 = tgid.x * BN, skip = 0;
#endif
    const int tn = sgid * (32 * FN);
    const short2 c = nax_coord(lane);
    // The shared expert reads its h rows from P with stride sNK (its intermediate size, the K of its down).
    const uint K = e == p.E ? p.sNK : p.K;
    const device half* hb = e == p.E ? hin + size_t(p.P) * p.K : hin;
    const uint rh = e == p.E ? r0 - p.P : r0;
    array<const device half*, AS::NI> rp;
    NAX_UNROLL for (int i = 0; i < AS::NI; i++) {
        const uint r = (tid + i * NX_THREADS) / AS::CPR;
        rp[i] = r < rows ? hb + size_t(rh + r) * K : nullptr;
    }
    using XS = ExpertSrc<FP4, bits, G>;
    using Src = typename XS::T;
    Src wsrc;
    if (e == p.E) {
        wsrc = XS::make(wS, sS, bS, 0);
    } else {
        const size_t se = smap[e];
        wsrc = XS::make(w + se * p.N * (p.K * bits / 32), s, b, se * p.N * (p.K / G));
    }
    const int live = min(int(rows + 15) / 16, FM);
#if LAB_MOE_EBIAS
    const device float* yb = e < p.E ? ebd + size_t(e) * p.N : nullptr;
#else
    const device float* yb = nullptr;
#endif
#define DN2(L) moe_down_body<Src, YT, BM, FN, BK, PF, L>(wsrc, y, rp, ws, xs, K, p.N, n0, r0, rows, tn, c, tid, yb, skip)
    if (FM >= 4 && live == 4) { if constexpr (FM >= 4) DN2((4 > FM ? FM : 4)); }
    else if (FM >= 3 && live == 3) { if constexpr (FM >= 3) DN2((3 > FM ? FM : 3)); }
    else if (FM >= 2 && live == 2) { if constexpr (FM >= 2) DN2((2 > FM ? FM : 2)); }
    else DN2(1);
#undef DN2
}

#define INSTANTIATE_MOE2G(G, BM, FN, BK, PFX)                                                                   \
  template [[host_name("moe_gateup_swiglu2_b4_g" #G "_bm" #BM "_fn" #FN "_bk" #BK "_pf" #PFX)]] [[kernel]] void  \
  moe_gateup_swiglu2<4, G, BM, FN, BK, PFX>(                                                                \
      const device half*, const device uint*, const device bfloat*, const device bfloat*,                    \
      const device uint*, const device bfloat*, const device bfloat*, device half*, const device uint*,      \
      const device uint4*, constant MoeGemmParams&,                                                          \
      const device uint*, const device bfloat*, const device bfloat*,                                        \
      const device uint*, const device bfloat*, const device bfloat*, const device uint*, const device float*, \
      const device float*, uint3, uint, ushort, ushort);                                                     \
  template [[host_name("moe_down2_b4_g" #G "_bm" #BM "_fn" #FN "_bk" #BK "_pf" #PFX)]] [[kernel]] void           \
  moe_down2<4, G, BM, FN, BK, PFX>(                                                                         \
      const device half*, const device uint*, const device bfloat*, const device bfloat*, device half*,      \
      const device uint4*, constant MoeGemmParams&, const device uint*, const device bfloat*,                \
      const device bfloat*, const device uint*, const device float*, uint3, uint, ushort, ushort);
#if LAB_FFN_GELU
#define INSTANTIATE_MOE2G_GELU(G, BM, FN, BK, PFX)                                                              \
  template [[host_name("moe_gateup_swiglu2_gelu_b4_g" #G "_bm" #BM "_fn" #FN "_bk" #BK "_pf" #PFX)]] [[kernel]] \
  void moe_gateup_swiglu2<4, G, BM, FN, BK, PFX, 1>(                                                        \
      const device half*, const device uint*, const device bfloat*, const device bfloat*,                    \
      const device uint*, const device bfloat*, const device bfloat*, device half*, const device uint*,      \
      const device uint4*, constant MoeGemmParams&,                                                          \
      const device uint*, const device bfloat*, const device bfloat*,                                        \
      const device uint*, const device bfloat*, const device bfloat*, const device uint*, const device float*, \
      const device float*, uint3, uint, ushort, ushort);
#else
#define INSTANTIATE_MOE2G_GELU(G, BM, FN, BK, PFX)
#endif
#if LAB_FFN_OAI
#define INSTANTIATE_MOE2G_OAI(G, BM, FN, BK, PFX)                                                               \
  template [[host_name("moe_gateup_swiglu2_oai_b4_g" #G "_bm" #BM "_fn" #FN "_bk" #BK "_pf" #PFX)]] [[kernel]]  \
  void moe_gateup_swiglu2<4, G, BM, FN, BK, PFX, 2>(                                                        \
      const device half*, const device uint*, const device bfloat*, const device bfloat*,                    \
      const device uint*, const device bfloat*, const device bfloat*, device half*, const device uint*,      \
      const device uint4*, constant MoeGemmParams&,                                                          \
      const device uint*, const device bfloat*, const device bfloat*,                                        \
      const device uint*, const device bfloat*, const device bfloat*, const device uint*, const device float*, \
      const device float*, uint3, uint, ushort, ushort);
#else
#define INSTANTIATE_MOE2G_OAI(G, BM, FN, BK, PFX)
#endif
#if LAB_FFN_RELU2
#define INSTANTIATE_MOE2G_RELU2(G, BM, FN, BK, PFX)                                                             \
  template [[host_name("moe_gateup_swiglu2_relu2_b4_g" #G "_bm" #BM "_fn" #FN "_bk" #BK "_pf" #PFX)]] [[kernel]] \
  void moe_gateup_swiglu2<4, G, BM, FN, BK, PFX, 3>(                                                        \
      const device half*, const device uint*, const device bfloat*, const device bfloat*,                    \
      const device uint*, const device bfloat*, const device bfloat*, device half*, const device uint*,      \
      const device uint4*, constant MoeGemmParams&,                                                          \
      const device uint*, const device bfloat*, const device bfloat*,                                        \
      const device uint*, const device bfloat*, const device bfloat*, const device uint*, const device float*, \
      const device float*, uint3, uint, ushort, ushort);
#else
#define INSTANTIATE_MOE2G_RELU2(G, BM, FN, BK, PFX)
#endif
#define INSTANTIATE_MOE2(BM, FN, BK, PFX)                                                                      \
  INSTANTIATE_MOE2G(64, BM, FN, BK, PFX) INSTANTIATE_MOE2G_GELU(64, BM, FN, BK, PFX)                          \
  INSTANTIATE_MOE2G_OAI(64, BM, FN, BK, PFX) INSTANTIATE_MOE2G_RELU2(64, BM, FN, BK, PFX)

// The two tile shapes of ModelPrefill.configurePrefillMoE (PrefillMoE.Tiles).
INSTANTIATE_MOE2(32, 1, 64, 1)
INSTANTIATE_MOE2(48, 1, 32, 1)

// MXFP4 experts (LAB_MXFP4): the two tile shapes of ModelPrefill.configurePrefillMoE, SiLU and (with LAB_FFN_OAI)
// swiglu_oai.
#if LAB_MXFP4
#define INSTANTIATE_MOE2_FP4_GU(ACTN, ACT, BM, FN, BK, PFX)                                                     \
  template [[host_name("moe_gateup_swiglu2" ACTN "_mxfp4_bm" #BM "_fn" #FN "_bk" #BK "_pf" #PFX)]] [[kernel]]   \
  void moe_gateup_swiglu2<4, 32, BM, FN, BK, PFX, ACT, true>(                                               \
      const device half*, const device uint*, const device bfloat*, const device bfloat*,                    \
      const device uint*, const device bfloat*, const device bfloat*, device half*, const device uint*,      \
      const device uint4*, constant MoeGemmParams&,                                                          \
      const device uint*, const device bfloat*, const device bfloat*,                                        \
      const device uint*, const device bfloat*, const device bfloat*, const device uint*, const device float*, \
      const device float*, uint3, uint, ushort, ushort);
#if LAB_FFN_OAI
#define INSTANTIATE_MOE2_FP4_OAI(BM, FN, BK, PFX) INSTANTIATE_MOE2_FP4_GU("_oai", 2, BM, FN, BK, PFX)
#else
#define INSTANTIATE_MOE2_FP4_OAI(BM, FN, BK, PFX)
#endif
#define INSTANTIATE_MOE2_FP4(BM, FN, BK, PFX)                                                                  \
  INSTANTIATE_MOE2_FP4_GU("", 0, BM, FN, BK, PFX) INSTANTIATE_MOE2_FP4_OAI(BM, FN, BK, PFX)                    \
  template [[host_name("moe_down2_mxfp4_bm" #BM "_fn" #FN "_bk" #BK "_pf" #PFX)]] [[kernel]] void              \
  moe_down2<4, 32, BM, FN, BK, PFX, true, float>(                                                           \
      const device half*, const device uint*, const device bfloat*, const device bfloat*, device float*,     \
      const device uint4*, constant MoeGemmParams&, const device uint*, const device bfloat*,                \
      const device bfloat*, const device uint*, const device float*, uint3, uint, ushort, ushort);
INSTANTIATE_MOE2_FP4(32, 1, 64, 1)
INSTANTIATE_MOE2_FP4(48, 1, 32, 1)
#endif
