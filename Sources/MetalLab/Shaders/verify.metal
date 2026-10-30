// Kernels of the DFlash2 verification round on Qwen3.6-35B-A3B (B = 1..16 rows).
//
// Everything is memory-bound: the kernels read each weight byte once and apply it to all the rows that need it.
// Common `qmv` core: each simdgroup computes NC outputs for NR rows; each lane covers 16 values of K per iteration
// (32 lanes × 16 = 512), always inside a single group of 64 (one scale and one bias).
//
// Mask trick taken from MLX (MIT, Copyright © 2023-2024 Apple Inc., mlx/backend/metal/kernels/quantized.h,
// `load_vector` / `qdot`): the activation is scaled by 16^-i (4 bits) or 256^-i (8 bits) depending on the position of
// the value inside the word, so the weight is used with a mask without shifting it: x·q = (x/16^i)·(w & (15 << 4i)).
// Σ x (s q + b) = s Σ x q + b Σ x.
//
// Portions adapted from splash (Apache-2.0; see THIRD_PARTY_NOTICES.md); changed: the GDN tape and the shared-expert
// layout are rewritten for these kernels.
// Ideas from splash (Apache-2.0, incoai/splash 1.3.2) and oMLX (Apache-2.0, jundot/omlx 0.7.1rc1) used here:
//   - the shared expert as one more expert of the list (splash `moe.metal`);
//   - fp32 router (splash, comment of `moe_route_select_f32`);
//   - the GDN "tape": verification writes the state from before its block and the next one folds only the
//     accepted rows (splash `decode/gdn.metal`; oMLX `qwen35_gdn_verify_fused.py` does it in registers).

#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

#define VUNROLL _Pragma("clang loop unroll(full)")

struct VParams {
    uint M;      // rows of x
    uint N;      // outputs
    uint K;      // inputs
    uint flags;  // bit 0: y += result (residual); bit 1: x is float (otherwise half)
    uint ldx;    // row stride of x (elements)
    uint ldy;    // row stride of y
};

// Activation scale depending on the position inside the word.
template <int BITS> constexpr float vpos_scale(int j);
template <> constexpr float vpos_scale<4>(int j) {
    return j % 8 == 0 ? 1.0f : j % 8 == 1 ? 1.0f / 16 : j % 8 == 2 ? 1.0f / 256 : j % 8 == 3 ? 1.0f / 4096
         : j % 8 == 4 ? 1.0f / 65536 : j % 8 == 5 ? 1.0f / 1048576 : j % 8 == 6 ? 1.0f / 16777216 : 1.0f / 268435456;
}
template <> constexpr float vpos_scale<8>(int j) {
    return j % 4 == 0 ? 1.0f : j % 4 == 1 ? 1.0f / 256 : j % 4 == 2 ? 1.0f / 65536 : 1.0f / 16777216;
}

template <int BITS> using vword = vec<uint, BITS / 2>;  // 16 values: uint2 (4 bits) or uint4 (8 bits)

// 16 activations (half or float) -> scaled by position, and their sum.
template <int BITS, typename T>
inline void vload_x(thread float (&xs)[16], thread float& xsum, const device T* xp) {
    float v[16];
    if (is_same<T, half>::value) {
        const device half4* p = (const device half4*)xp;
        VUNROLL for (int i = 0; i < 4; i++) {
            half4 a = p[i];
            v[4 * i] = a.x; v[4 * i + 1] = a.y; v[4 * i + 2] = a.z; v[4 * i + 3] = a.w;
        }
    } else {
        const device float4* p = (const device float4*)xp;
        VUNROLL for (int i = 0; i < 4; i++) {
            float4 a = p[i];
            v[4 * i] = a.x; v[4 * i + 1] = a.y; v[4 * i + 2] = a.z; v[4 * i + 3] = a.w;
        }
    }
    float s = 0;
    VUNROLL for (int j = 0; j < 16; j++) { s += v[j]; xs[j] = v[j] * vpos_scale<BITS>(j); }
    xsum = s;
}

template <int BITS> inline float vqdot(thread const float (&xs)[16], vword<BITS> w);
template <> inline float vqdot<4>(thread const float (&xs)[16], uint2 w) {
    float a = 0, b = 0;
    VUNROLL for (int i = 0; i < 8; i++) {
        a = fma(xs[i], float(w.x & (0xfu << (4 * i))), a);
        b = fma(xs[8 + i], float(w.y & (0xfu << (4 * i))), b);
    }
    return a + b;
}
template <> inline float vqdot<8>(thread const float (&xs)[16], uint4 w) {
    float a = 0, b = 0;
    VUNROLL for (int i = 0; i < 4; i++) {
        a = fma(xs[i], float(w.x & (0xffu << (8 * i))), a);
        b = fma(xs[4 + i], float(w.y & (0xffu << (8 * i))), b);
        a = fma(xs[8 + i], float(w.z & (0xffu << (8 * i))), a);
        b = fma(xs[12 + i], float(w.w & (0xffu << (8 * i))), b);
    }
    return a + b;
}

// acc[r][c] += Σ_k x[rows[r]][k] · W[n0 + c][k] (one lane, its chunks of K).
template <int BITS, int NC, int NR, typename T, int G = 64>
inline void qmv_acc(const device T* x, uint ldx, thread const uint (&rows)[NR], const device uint* w,
                    const device bfloat* s, const device bfloat* b, uint K, uint n0,
                    thread float (&acc)[NR][NC], ushort lane) {
    const uint wpr = K * BITS / 32, gpr = K / G;   // groups of G (64, or 32): 16 values, a single group
    for (uint k0 = lane * 16; k0 < K; k0 += 512) {
        float xs[NR][16], xsum[NR];
        VUNROLL for (int r = 0; r < NR; r++) vload_x<BITS>(xs[r], xsum[r], x + size_t(rows[r]) * ldx + k0);
        VUNROLL for (int c = 0; c < NC; c++) {
            const size_t n = n0 + c;
            const vword<BITS> wv = *(const device vword<BITS>*)(w + n * wpr + k0 * BITS / 32);
            const float sc = s[n * gpr + k0 / G], bi = b[n * gpr + k0 / G];
            VUNROLL for (int r = 0; r < NR; r++) acc[r][c] += sc * vqdot<BITS>(xs[r], wv) + bi * xsum[r];
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Y[M, N] (+)= X[M, K] · Wᵀ with few rows. Grid: N / (SGS·NC) threadgroups of SGS simdgroups.
template <int BITS, int NC, int NR, int SGS, typename T, int G = 64>
[[kernel]] void vqmv(const device T*     x [[buffer(0)]],
                     const device uint*   w [[buffer(1)]],
                     const device bfloat* s [[buffer(2)]],
                     const device bfloat* b [[buffer(3)]],
                     device float*        y [[buffer(4)]],
                     constant VParams&    p [[buffer(5)]],
                     uint tg [[threadgroup_position_in_grid]],
                     ushort sg [[simdgroup_index_in_threadgroup]],
                     ushort lane [[thread_index_in_simdgroup]]) {
    const uint n0 = (tg * SGS + sg) * NC;
    if (n0 >= p.N) return;
    for (uint r0 = 0; r0 < p.M; r0 += NR) {
        uint rows[NR];
        VUNROLL for (int r = 0; r < NR; r++) rows[r] = min(r0 + r, p.M - 1);
        float acc[NR][NC];
        VUNROLL for (int r = 0; r < NR; r++) VUNROLL for (int c = 0; c < NC; c++) acc[r][c] = 0;
        qmv_acc<BITS, NC, NR, T, G>(x, p.ldx, rows, w, s, b, p.K, n0, acc, lane);
        VUNROLL for (int r = 0; r < NR; r++) VUNROLL for (int c = 0; c < NC; c++) {
            const float v = simd_sum(acc[r][c]);
            if (lane == 0 && r0 + r < p.M) {
                device float* o = y + size_t(r0 + r) * p.ldy + n0 + c;
                *o = (p.flags & 1) ? *o + v : v;
            }
        }
    }
}

#define VQMV_INST(BITS, NC, NR, SGS, T, TN)                                                                        \
  template [[host_name("vqmv_b" #BITS "_nc" #NC "_nr" #NR "_sg" #SGS "_" #TN)]] [[kernel]] void                   \
  vqmv<BITS, NC, NR, SGS, T>(const device T*, const device uint*, const device bfloat*, const device bfloat*,     \
                             device float*, constant VParams&, uint, ushort, ushort);
#define VQMV_NR(BITS, NC, SGS, T, TN) VQMV_INST(BITS, NC, 1, SGS, T, TN) VQMV_INST(BITS, NC, 2, SGS, T, TN)        \
  VQMV_INST(BITS, NC, 4, SGS, T, TN) VQMV_INST(BITS, NC, 8, SGS, T, TN)
VQMV_NR(4, 4, 4, half, h)
VQMV_NR(8, 4, 4, half, h)
VQMV_NR(4, 4, 4, float, f)
#if LAB_G32
#define VQMV_INST_G32(NC, NR, SGS, T, TN)                                                                          \
  template [[host_name("vqmv_b4_g32_nc" #NC "_nr" #NR "_sg" #SGS "_" #TN)]] [[kernel]] void                       \
  vqmv<4, NC, NR, SGS, T, 32>(const device T*, const device uint*, const device bfloat*, const device bfloat*,    \
                              device float*, constant VParams&, uint, ushort, ushort);
#define VQMV_NR_G32(NC, SGS, T, TN) VQMV_INST_G32(NC, 1, SGS, T, TN) VQMV_INST_G32(NC, 2, SGS, T, TN)              \
  VQMV_INST_G32(NC, 4, SGS, T, TN) VQMV_INST_G32(NC, 8, SGS, T, TN)
VQMV_NR_G32(4, 4, half, h)
VQMV_NR_G32(4, 4, float, f)
#endif

// ---------------------------------------------------------------------------------------------
// Verification MoE.
//
// vmoe_route: top-k over the fp32 router logits (ties broken by lower index), renormalized softmax weights (with
// LAB_MOE_SCORE = 1, the sigmoid router of moe_route_sigmoid),
// shared-expert gate (8 bits) and the expert union: list of distinct experts in increasing order, each with its
// (row, weight) pairs in row order, and the shared expert last (id E, all rows, weight sigmoid(gate)).
// One threadgroup of 512 threads (one simdgroup per row, B <= 16). E and top-k come from the config (LAB_MOE_E/K).
// Shared expert per LAB_MOE_SHARED: 1, with gate (the old behavior); 2, without gate (weight 1); 0, no shared expert
// (the union ends at the last routed expert: U entries instead of U + 1).
constant constexpr uint VMOE_E = LAB_MOE_E, VMOE_K = LAB_MOE_K, VMOE_VPL = VMOE_E / 32;
static_assert(VMOE_E % 32 == 0 && VMOE_E <= 512 && VMOE_K <= 16, "vmoe_route: E multiple of 32, <= 512; top-k <= 16");

struct VMoEParams {
    uint hidden;  // 2048
    uint inter;   // 512
    uint E;       // 256 (= LAB_MOE_E)
    uint B;       // rows
    uint tiles;   // column tiles of gate+up (inter / columns per threadgroup)
    uint flags;   // bit 0: out += (residual); bits 8-15: column blocks of down; 16-23: top-k (0 = 8);
                  // 24-31: combine slots (0 = 9: top-8 + shared)
    uint sinter;  // shared-expert intermediate (= inter in Qwen3.6 and Flash-Next)
};

[[kernel]] void vmoe_route(const device float*  logits  [[buffer(0)]],   // [B, MOE_LD]
                           const device half*   x       [[buffer(1)]],   // [B, hidden]
                           const device uint*   sgw     [[buffer(2)]],   // shared-expert gate 8 bits [1, hidden]
                           const device bfloat* sgs     [[buffer(3)]],
                           const device bfloat* sgb     [[buffer(4)]],
                           device uint4*        entries [[buffer(5)]],   // [U + 1] (expert, start, rows, 0)
                           device uint2*        pairs   [[buffer(6)]],   // [B·k + B] (row, fp32 weight)
                           device uint*         args    [[buffer(7)]],   // [tiles, U+1, 1, U+1]
                           device uint*         inds    [[buffer(8)]],   // [B, k] (for checking)
                           device float*        wts     [[buffer(9)]],   // [B, k]
                           constant VMoEParams& p       [[buffer(10)]],
                           const device uint*   rst     [[buffer(11)]],  // round state (flags bit 1: rows = st[6])
                           const device float*  bias    [[buffer(12)]],  // selection bias [E] (LAB_MOE_BIAS) or router bias (LAB_MOE_LBIAS)
                           const device float*  pes     [[buffer(13)]],  // per-expert scale [E] (LAB_MOE_PES, Gemma 4)
                           uint tid [[thread_index_in_threadgroup]],
                           ushort sg [[simdgroup_index_in_threadgroup]],
                           ushort lane [[thread_index_in_simdgroup]]) {
    constexpr int TOPK = VMOE_K, VPL = VMOE_VPL;
    constexpr uint NSG = VMOE_E / 32;   // simdgroups of the prefixes (32 experts each)
    threadgroup uint t_ind[16 * TOPK];
    threadgroup float t_w[16 * TOPK];
    threadgroup float t_sg[16];
    threadgroup atomic_uint t_cnt[VMOE_E];
    threadgroup uint t_off[VMOE_E], t_uid[VMOE_E], t_tot[2][NSG];
    // With bit 1 of flags only the first st[6] rows are routed (the rest do not enter the expert union
    // or the shared expert; their output is unused).
    const uint B = (p.flags & 2) ? min(p.B, rst[6]) : p.B, E = p.E;
    if (tid < VMOE_E) atomic_store_explicit(&t_cnt[tid], 0, memory_order_relaxed);
    if (sg < B) {
        const uint r = sg;
#if LAB_MOE_SCORE == 1
        {
            // Sigmoid router: moe_route_sigmoid (ab_moe_route.metal).
            float mine; uint mi;
            moe_route_sigmoid<VPL, TOPK>(logits + r * MOE_LD, bias, lane, mi, mine);
            if (lane < TOPK) {
                t_ind[r * TOPK + lane] = mi; t_w[r * TOPK + lane] = mine;
                inds[r * TOPK + lane] = mi; wts[r * TOPK + lane] = mine;
            }
        }
#else
        float v[VPL];
        // NaN -> -inf (otherwise top-k picks nothing and the indices go out of range).
        VUNROLL for (int i = 0; i < VPL; i++) {
#if LAB_MOE_LBIAS
            const float l = logits[r * MOE_LD + lane + 32 * i] + bias[lane + 32 * i];   // router bias (gpt-oss)
#else
            const float l = logits[r * MOE_LD + lane + 32 * i];
#endif
            v[i] = isnan(l) ? -INFINITY : l;
        }
        float top[TOPK];
        uint idx[TOPK];
        VUNROLL for (int k = 0; k < TOPK; k++) {
            float bv = -INFINITY; uint bi = 0xffffu;
            VUNROLL for (int i = 0; i < VPL; i++) if (v[i] > bv) { bv = v[i]; bi = lane + 32 * i; }
            const float m = simd_max(bv);
            uint sel = simd_min(bv == m ? bi : 0xffffu);
            if (sel >= E) sel = k;  // all -inf: any valid expert, different for each k
            VUNROLL for (int i = 0; i < VPL; i++) if (lane + 32 * i == sel) v[i] = -INFINITY;
            top[k] = m; idx[k] = sel;
        }
        float ex[TOPK], sum = 0;
        VUNROLL for (int k = 0; k < TOPK; k++) { ex[k] = exp(top[k] - top[0]); sum += ex[k]; }
#ifdef LAB_MOE_RSCALE
        sum /= LAB_MOE_RSCALE;   // routed_scaling_factor
#endif
        if (lane < TOPK) {
            float mine = 0; uint mi = 0;
            VUNROLL for (int k = 0; k < TOPK; k++) if (k == lane) { mine = ex[k]; mi = idx[k]; }
#if LAB_MOE_PES
            const float w = mine / sum * pes[mi];   // per_expert_scale after the renormalization
#else
            const float w = mine / sum;
#endif
            t_ind[r * TOPK + lane] = mi;
            t_w[r * TOPK + lane] = w;
            inds[r * TOPK + lane] = mi;
            wts[r * TOPK + lane] = w;
        }
#endif
#if LAB_MOE_SHARED == 1
        // Shared-expert gate (8 bits): each lane carries a group of 64 values per step of 2048 (hidden a multiple of
        // 64; with 2048, a single step).
        float acc = 0;
        for (uint base = 0; base < p.hidden; base += 2048) {
            const uint k0 = base + lane * 64;
            if (k0 >= p.hidden) break;
            for (uint kk = 0; kk < 64; kk += 16) {
                float xs[16], xsum;
                vload_x<8>(xs, xsum, x + size_t(r) * p.hidden + k0 + kk);
                const uint4 wv = *(const device uint4*)(sgw + (k0 + kk) / 4);
                acc += float(sgs[k0 / 64]) * vqdot<8>(xs, wv) + float(sgb[k0 / 64]) * xsum;
            }
        }
        acc = simd_sum(acc);
        if (lane == 0) t_sg[r] = 1.0f / (1.0f + exp(-acc));
#else
        if (lane == 0) t_sg[r] = 1.0f;
#endif
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint P = B * TOPK;
    if (tid < P) atomic_fetch_add_explicit(&t_cnt[t_ind[tid]], 1u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Prefixes of counts and of present experts (E / 32 simdgroups × 32 experts).
    uint c = 0, f = 0;
    if (tid < VMOE_E) {
        c = atomic_load_explicit(&t_cnt[tid], memory_order_relaxed);
        f = c > 0 ? 1u : 0u;
        const uint pc = simd_prefix_exclusive_sum(c), pf = simd_prefix_exclusive_sum(f);
        if (lane == 31) { t_tot[0][sg] = pc + c; t_tot[1][sg] = pf + f; }
        t_off[tid] = pc; t_uid[tid] = pf;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < VMOE_E) {
        uint bc = 0, bf = 0;
        for (uint s2 = 0; s2 < sg; s2++) { bc += t_tot[0][s2]; bf += t_tot[1][s2]; }
        t_off[tid] += bc; t_uid[tid] += bf;
        if (f) entries[t_uid[tid]] = uint4(tid, t_off[tid], c, 0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint U = 0;
    for (uint s2 = 0; s2 < NSG; s2++) U += t_tot[1][s2];
    if (tid < P) {
        const uint e = t_ind[tid], row = tid / TOPK;
        uint rank = 0;
        for (uint q = 0; q < P; q++) rank += (t_ind[q] == e && q / TOPK < row) ? 1u : 0u;
        pairs[t_off[e] + rank] = uint2(row, as_type<uint>(t_w[tid]));
    }
    constexpr uint NSH = LAB_MOE_SHARED ? 1u : 0u;   // shared-expert entries in the union
    if (NSH && tid < B) pairs[P + tid] = uint2(tid, as_type<uint>(t_sg[tid]));
    if (tid == 0) {
        if (NSH) entries[U] = uint4(E, P, B, 0);
        args[0] = p.tiles; args[1] = U + NSH; args[2] = 1; args[3] = U + NSH;
        // Down grid per expert (vnax_moe_down_split): (column blocks = flags >> 8, U + 1, 1).
        args[4] = (p.flags >> 8) & 0xffu; args[5] = U + NSH; args[6] = 1;
    }
}

// ---------------------------------------------------------------------------------------------
// Verification GatedDeltaNet (Qwen3.6: 16 k heads × 128, 32 v heads × 128, conv of 4).
// Row of concatenated projections (one qmv): qkv [0, 8192) | z [8192, 12288) | b [12288, 12320) | a [12320, 12352).
// Conv history: [3 window rows | B raw rows of the block]; the next round takes the window from rows
// [nacc, nacc + 3) of that history (nacc = rows accepted from the previous block).

struct GdnParams {
    uint B;      // rows of the current block
    uint nacc;   // rows of the previous block to fold (accepted)
    uint ldp;    // row stride of the projections (12352)
    float eps;   // rms_norm_eps
    uint fin;    // 1: also writes the state after the B rows (prefill: all count, no tape)
    uint dyn;    // bit 0: nacc is read from the GPU round state (st[1]) instead of p.nacc; bit 1: rows = st[6]
};

// q and k: 16 heads × 128; v: GDN_HV × 128 (32 in Qwen3.6, 48 in 27B); each k head serves GDN_R v heads.
// Projection per row: q | k | v (GDN_C) | z (GDN_V) | b (GDN_HV) | a (GDN_HV).
// Mamba2 (LAB_SSM, Nemotron-H): q = C and k = B with LAB_GDN_HK groups of the state (LAB_GDN_DK = 128), v = x with
// heads of GDN_DV; projection per row z (GDN_V) | x B C (GDN_C, conv channels in that order) | dt (GDN_HV).
constant constexpr uint GDN_HV = LAB_GDN_HV, GDN_R = GDN_HV / LAB_GDN_HK, GDN_DK = LAB_GDN_DK, GDN_DV = LAB_GDN_DV;
constant constexpr uint GDN_QK = LAB_GDN_HK * GDN_DK, GDN_V = GDN_HV * GDN_DV, GDN_C = 2 * GDN_QK + GDN_V;
static_assert(LAB_GDN_DK == 128 && (LAB_SSM || (LAB_GDN_HK == 16 && LAB_GDN_DV == 128)),
              "GDN: the kernels assume k heads of 128 (16 × 128 and v of 128 outside Mamba2)");
static_assert(GDN_DV % 16 == 0 && GDN_HV % LAB_GDN_HK == 0, "GDN: v heads a multiple of 16 wide, whole groups");
constant constexpr uint GDN_PB = GDN_C + GDN_V, GDN_PA = GDN_PB + GDN_HV;
// Columns of the projection: conv channels, gate z and the decay input (a; Mamba2's dt).
constant constexpr uint GDN_CONV0 = LAB_SSM ? GDN_V : 0, GDN_Z0 = LAB_SSM ? 0 : GDN_C;
constant constexpr uint GDN_DT0 = LAB_SSM ? GDN_V + GDN_C : GDN_PA;

// causal conv1d + SiLU, q and k norms (normalize_qk of mlx-lm), g and beta. Grid (64, B): 16 q heads, 16 k and
// 32 v heads (128 channels each) per row; 128 threads.
template <typename QT, typename PT>
[[kernel]] void gdn_prework(const device PT*     proj  [[buffer(0)]],
                            const device LAB_NWT* convw [[buffer(1)]],   // [8192, 4]
                            const device float*  hprev [[buffer(2)]],   // [3 + B_prev, 8192]
                            const device float*   alog  [[buffer(3)]],   // [32], always fp32
                            const device LAB_NWT* dtb   [[buffer(4)]],   // [32]
                            device float*        hcur  [[buffer(5)]],   // [3 + B, 8192]
                            device QT*           q     [[buffer(6)]],   // [B, 2048]
                            device QT*           k     [[buffer(7)]],   // [B, 2048]
                            device QT*           v     [[buffer(8)]],   // [B, 4096]
                            device float*        g     [[buffer(9)]],   // [B, 32]
                            device float*        beta  [[buffer(10)]],  // [B, 32]
                            constant GdnParams&  p     [[buffer(11)]],
                            const device uint*   rst   [[buffer(12)]],
#if LAB_SSM
                            const device LAB_NWT* convb [[buffer(13)]],  // Mamba2's conv bias [GDN_C]
#endif
                            uint2 tg [[threadgroup_position_in_grid]],
                            uint tid [[thread_index_in_threadgroup]],
                            ushort sg [[simdgroup_index_in_threadgroup]],
                            ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[4];
    const uint nacc = (p.dyn & 1) ? rst[1] : p.nacc;
    const uint t = tg.y, slot = tg.x, c = slot * 128 + tid;
    float acc = 0;
    VUNROLL for (uint j = 0; j < 4; j++) {
        const uint i = t + j;
        const float in = i < 3 ? hprev[(nacc + i) * GDN_C + c] : float(proj[(i - 3) * p.ldp + GDN_CONV0 + c]);
        acc = fma(float(convw[c * 4 + j]), in, acc);
    }
#if LAB_SSM
    acc += float(convb[c]);
#endif
    const float o = acc / (1.0f + exp(-acc));
    if (t == 0) VUNROLL for (uint i = 0; i < 3; i++) hcur[i * GDN_C + c] = hprev[(nacc + i) * GDN_C + c];
    // Prefill (fin): only the last 3 rows of the conv history are needed (copy_f32 copies them); writing all
    // B cost 67 MB per layer with chunks of 2048.
    // With captures inside the pass (dyn bits 8-19 and 20-31 = first and last captured row + 1), also
    // the 3 conv entries of each captured row (fnc: those of all the rows between the first and the last).
    const uint c0 = (p.dyn >> 8) & 0xFFFu, c1 = max(p.dyn >> 20, c0);
    const bool keep = c0 && t + 3 >= c0 && t < c1;
    if (!p.fin || 3 + t >= p.B || keep) hcur[(3 + t) * GDN_C + c] = float(proj[t * p.ldp + GDN_CONV0 + c]);
#if LAB_SSM
    // Mamba2: x → v, B → k, C → q, as they are (no norms); beta = dt = softplus(dt_raw + dt_bias) and the same g.
    if (c < GDN_V) v[t * GDN_V + c] = o;
    else if (c < GDN_V + GDN_QK) k[t * GDN_QK + c - GDN_V] = o;
    else q[t * GDN_QK + c - GDN_V - GDN_QK] = o;
    if (slot == 0) for (uint h = tid; h < GDN_HV; h += 128) {
        const float xa = float(proj[t * p.ldp + GDN_DT0 + h]) + float(dtb[h]);
        const float sp = xa > 20.0f ? xa : log(1.0f + exp(xa));
        beta[t * GDN_HV + h] = sp;
        g[t * GDN_HV + h] = exp(-exp(float(alog[h])) * sp);
    }
    (void)sg; (void)lane; (void)part;
#else
    if (slot < 32) {
        const float ss = simd_sum(o * o);
        if (lane == 0) part[sg] = ss;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float tot = part[0] + part[1] + part[2] + part[3];
        // normalize_qk: inv = 128^-1/2; rms_norm with eps·inv² (on the mean); q·inv², k·inv.
        constexpr float inv = 0.08838834764831845f;
        const float r = rsqrt(tot / 128.0f + 1e-6f * inv * inv);
        if (slot < 16) q[t * GDN_QK + c] = o * r * (inv * inv);
        else k[t * GDN_QK + c - GDN_QK] = o * r * inv;
    } else {
        v[t * GDN_V + c - 2 * GDN_QK] = o;
    }
    if (slot == 0 && tid < GDN_HV) {
        const float bb = float(proj[t * p.ldp + GDN_PB + tid]), aa = float(proj[t * p.ldp + GDN_PA + tid]);
        beta[t * GDN_HV + tid] = 1.0f / (1.0f + exp(-bb));
        const float xa = aa + float(dtb[tid]);
        const float sp = xa > 20.0f ? xa : log(1.0f + exp(xa));
        g[t * GDN_HV + tid] = exp(-exp(float(alog[tid])) * sp);
    }
#endif
}
#if LAB_SSM
#define GDN_PREWORK_CONVB , const device LAB_NWT*
#else
#define GDN_PREWORK_CONVB
#endif
#define GDN_PREWORK_INST(QT, PT, NAME)                                                                             \
  template [[host_name(NAME)]] [[kernel]] void gdn_prework<QT, PT>(                                                \
      const device PT*, const device LAB_NWT*, const device float*, const device float*, const device LAB_NWT*,   \
      device float*, device QT*, device QT*, device QT*, device float*, device float*, constant GdnParams&,         \
      const device uint* GDN_PREWORK_CONVB, uint2, uint, ushort, ushort);
GDN_PREWORK_INST(float, float, "gdn_prework")
// Also the qkv|z|b|a projection output in half (101 -> 51 MB per layer with chunks of 2048).
GDN_PREWORK_INST(half, half, "gdn_prework_hh")

// RMSNormGated: o = rms_norm(y)·w·silu(z) (σ(z) with LAB_GDN_SIG), per (row, v head). One simdgroup per head, 4/TG.
template <typename PT>
[[kernel]] void gdn_normgate(const device float*  y    [[buffer(0)]],   // [B, 4096]
                             const device PT*     proj [[buffer(1)]],   // z in [8192, 12288) of each row
                             const device LAB_NWT* nw   [[buffer(2)]],   // [128]
                             device half*         o    [[buffer(3)]],   // [B, 4096]
                             constant GdnParams&  p    [[buffer(4)]],
                             uint tg [[threadgroup_position_in_grid]],
                             ushort sg [[simdgroup_index_in_threadgroup]],
                             ushort lane [[thread_index_in_simdgroup]]) {
    const uint id = tg * 4 + sg, t = id / GDN_HV, hv = id % GDN_HV;
    if (t >= p.B) return;
    const float4 yy = ((const device float4*)(y + t * GDN_V + hv * 128))[lane];
    const float r = rsqrt(simd_sum(dot(yy, yy)) / 128.0f + p.eps);
    const float4 z = float4(((const device vec<PT, 4>*)(proj + t * p.ldp + GDN_C + hv * 128))[lane]);
    half4 out;
    VUNROLL for (int i = 0; i < 4; i++) {
        const float zz = z[i];
        out[i] = half(yy[i] * r * float(nw[lane * 4 + i]) * ((LAB_GDN_SIG ? 1.0f : zz) / (1.0f + exp(-zz))));
    }
    ((device half4*)(o + t * GDN_V + hv * 128))[lane] = out;
}
template [[host_name("gdn_normgate")]] [[kernel]] void gdn_normgate<float>(const device float*, const device float*,
    const device LAB_NWT*, device half*, constant GdnParams&, uint, ushort, ushort);
template [[host_name("gdn_normgate_h")]] [[kernel]] void gdn_normgate<half>(const device float*, const device half*,
    const device LAB_NWT*, device half*, constant GdnParams&, uint, ushort, ushort);

#if LAB_SSM
// Mamba2's MambaRMSNormGated (mlx-lm nemotron_h): u = y·silu(z), RMSNorm over groups of GDN_V / LAB_GDN_HK channels
// (no weight), then × w per channel, to half. One simdgroup per (row, group), 4 per threadgroup.
[[kernel]] void mamba_normgate(const device float*  y    [[buffer(0)]],   // [B, GDN_V]
                               const device float*  proj [[buffer(1)]],   // z in [0, GDN_V) of each row
                               const device LAB_NWT* nw   [[buffer(2)]],   // [GDN_V]
                               device half*         o    [[buffer(3)]],   // [B, GDN_V]
                               constant GdnParams&  p    [[buffer(4)]],
                               uint tg [[threadgroup_position_in_grid]],
                               ushort sg [[simdgroup_index_in_threadgroup]],
                               ushort lane [[thread_index_in_simdgroup]]) {
    constexpr uint GS = GDN_V / LAB_GDN_HK;
    const uint id = tg * 4 + sg, t = id / LAB_GDN_HK, c0 = (id % LAB_GDN_HK) * GS;
    if (t >= p.B) return;
    const device float* yr = y + t * GDN_V + c0;
    const device float* zr = proj + t * p.ldp + GDN_Z0 + c0;
    float ss = 0;
    for (uint i = lane; i < GS; i += 32) {
        const float z = zr[i], u = yr[i] * (z / (1.0f + exp(-z)));
        ss += u * u;
    }
    const float r = rsqrt(simd_sum(ss) / float(GS) + p.eps);
    for (uint i = lane; i < GS; i += 32) {
        const float z = zr[i], u = yr[i] * (z / (1.0f + exp(-z)));
        o[t * GDN_V + c0 + i] = half(u * r * float(nw[c0 + i]));
    }
}
#endif

// ---------------------------------------------------------------------------------------------
// Round utilities: RMSNorm of the residual stream (fp32) to half, and argmax per row.
struct NormParams { uint D; float eps; uint ldx; uint ldo; };

[[kernel]] void rmsnorm_rows(const device float*  x [[buffer(0)]],
                             const device LAB_NWT* w [[buffer(1)]],
                             device half*         o [[buffer(2)]],
                             constant NormParams& p [[buffer(3)]],
                             uint row [[threadgroup_position_in_grid]],
                             uint tid [[thread_index_in_threadgroup]],
                             ushort sg [[simdgroup_index_in_threadgroup]],
                             ushort lane [[thread_index_in_simdgroup]],
                             uint ntg [[threads_per_threadgroup]]) {
    threadgroup float part[32];
    const device float* xr = x + size_t(row) * p.ldx;
    float ss = 0;
    for (uint i = tid; i < p.D; i += ntg) ss += xr[i] * xr[i];
    ss = simd_sum(ss);
    if (lane == 0) part[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0;
    for (uint i = 0; i < ntg / 32; i++) tot += part[i];
    const float r = rsqrt(tot / float(p.D) + p.eps);
    for (uint i = tid; i < p.D; i += ntg) o[size_t(row) * p.ldo + i] = half(xr[i] * r * float(w[i]));
}

[[kernel]] void argmax_rows(const device float* x   [[buffer(0)]],
                            device uint*        out [[buffer(1)]],
                            constant uint&      N   [[buffer(2)]],
                            uint row [[threadgroup_position_in_grid]],
                            uint tid [[thread_index_in_threadgroup]],
                            ushort sg [[simdgroup_index_in_threadgroup]],
                            ushort lane [[thread_index_in_simdgroup]],
                            uint ntg [[threads_per_threadgroup]]) {
    threadgroup float bv[32];
    threadgroup uint bi[32];
    const device float* xr = x + size_t(row) * N;
    float m = -INFINITY; uint mi = 0;
    for (uint i = tid; i < N; i += ntg) if (xr[i] > m) { m = xr[i]; mi = i; }
    const float sm = simd_max(m);
    const uint si = simd_min(m == sm ? mi : 0xffffffffu);
    if (lane == 0) { bv[sg] = sm; bi[sg] = si; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float b = bv[0]; uint ix = bi[0];
        for (uint i = 1; i < ntg / 32; i++) if (bv[i] > b || (bv[i] == b && bi[i] < ix)) { b = bv[i]; ix = bi[i]; }
        out[row] = ix;
    }
}

[[kernel]] void copy_f32(const device float4* src [[buffer(0)]], device float4* dst [[buffer(1)]],
                         constant uint& n4 [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < n4) dst[i] = src[i];
}

// 16-byte copy per thread as integers (bit-exact with any type; fnt: captures of the prefill cuts).
[[kernel]] void copy_u32(const device uint4* src [[buffer(0)]], device uint4* dst [[buffer(1)]],
                         constant uint& n4 [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < n4) dst[i] = src[i];
}

// Delta rule recurrence. State [32, 128 (dv), 128 (dk)] fp32: each simdgroup carries DVS dv rows in registers
// (4 dk per lane). First it folds the nacc accepted rows of the previous block (the tape: k, v, g, beta saved
// by gdn_prework in the previous round), writes that state (the one from before the current block) and then walks the B
// rows of the block computing, without writing the state again.
// The data of G steps is loaded at once (the next group is prefetched while the current one computes): L2 latency
// is paid once every G steps instead of every step; only DVS = 4, G = 2 is instantiated. It is also the fp32
// sequential recurrence reference for prefill (options.bench.gdnChunk = false: `lab-dev kl --cfgs noise`).
template <int DVS, int G>
[[kernel]] void gdn_recur_g(const device float* sin   [[buffer(0)]],
                            device float*       sout  [[buffer(1)]],
                            const device float* kp    [[buffer(2)]],
                            const device float* vp    [[buffer(3)]],
                            const device float* gp    [[buffer(4)]],
                            const device float* bp    [[buffer(5)]],
                            const device float* q     [[buffer(6)]],
                            const device float* k     [[buffer(7)]],
                            const device float* v     [[buffer(8)]],
                            const device float* g     [[buffer(9)]],
                            const device float* beta  [[buffer(10)]],
                            device float*       y     [[buffer(11)]],
                            constant GdnParams& p     [[buffer(12)]],
                            const device uint*  rst   [[buffer(13)]],
#if LAB_SSM
                            const device float* Dskip [[buffer(14)]],  // Mamba2's D [GDN_HV] (y += D·x)
#endif
                            uint tg [[threadgroup_position_in_grid]],
                            ushort sg [[simdgroup_index_in_threadgroup]],
                            ushort lane [[thread_index_in_simdgroup]]) {
    constexpr uint per_head = GDN_DV / (4 * DVS);
    const uint hv = tg / per_head, dv0 = (tg % per_head) * 4 * DVS + sg * DVS, hk = hv / GDN_R;
    const uint nacc = (p.dyn & 1) ? rst[1] : p.nacc;
    // (dyn bit 1): only the first st[6] rows of the block (the rest are not verified).
    const uint T = nacc + ((p.dyn & 2) ? min(p.B, rst[6]) : p.B);
    struct Grp { float4 k[G], q[G]; float v[G][DVS], g[G], b[G]; };
    auto load = [&](uint t0, thread Grp& D) {
        VUNROLL for (int i = 0; i < G; i++) {
            const uint t = min(t0 + i, T - 1);
            const bool tape = t < nacc;
            const uint tc = tape ? t : t - nacc;
            D.k[i] = ((const device float4*)((tape ? kp : k) + tc * GDN_QK + hk * GDN_DK))[lane];
            D.q[i] = tape ? float4(0) : ((const device float4*)(q + tc * GDN_QK + hk * GDN_DK))[lane];
            VUNROLL for (int d = 0; d < DVS; d++) D.v[i][d] = (tape ? vp : v)[tc * GDN_V + hv * GDN_DV + dv0 + d];
            D.g[i] = (tape ? gp : g)[tc * GDN_HV + hv];
            D.b[i] = (tape ? bp : beta)[tc * GDN_HV + hv];
        }
    };
    const uint s0 = (hv * GDN_DV + dv0) * GDN_DK;
    float4 st[DVS];
    VUNROLL for (int d = 0; d < DVS; d++) st[d] = ((const device float4*)(sin + s0 + d * GDN_DK))[lane];
    if (nacc == 0) VUNROLL for (int d = 0; d < DVS; d++) ((device float4*)(sout + s0 + d * GDN_DK))[lane] = st[d];
    Grp cur, nxt;
    if (T > 0) load(0, cur);
    for (uint t0 = 0; t0 < T; t0 += G) {
        if (t0 + G < T) load(t0 + G, nxt);
        VUNROLL for (int i = 0; i < G; i++) {
            const uint t = t0 + i;
            if (t < T) {
                VUNROLL for (int d = 0; d < DVS; d++) {
#if LAB_SSM
                    // Mamba2: S = g·S + dt · x ⊗ B, y = S·C + D·x (no delta term). Explicit fma: with the
                    // contraction left to the compiler, the two unrolled steps of a group compiled differently and
                    // a row gave other bits depending on its place in the round (the Nano 4B's live state after a
                    // 9-row copy round differed from B = 1 steps).
                    st[d] = fma(cur.k[i], float4(cur.v[i][d] * cur.b[i]), st[d] * cur.g[i]);
#else
                    st[d] *= cur.g[i];
                    const float kv = simd_sum(dot(st[d], cur.k[i]));
                    st[d] += cur.k[i] * ((cur.v[i][d] - kv) * cur.b[i]);
#endif
                    if (t >= nacc) {
                        float yo = simd_sum(dot(st[d], cur.q[i]));
#if LAB_SSM
                        yo = fma(cur.v[i][d], Dskip[hv], yo);
#endif
                        if (lane == 0) y[(t - nacc) * GDN_V + hv * GDN_DV + dv0 + d] = yo;
                    }
                }
                if (t + 1 == nacc)
                    VUNROLL for (int d = 0; d < DVS; d++) ((device float4*)(sout + s0 + d * GDN_DK))[lane] = st[d];
            }
        }
        cur = nxt;
    }
    if (p.fin) VUNROLL for (int d = 0; d < DVS; d++) ((device float4*)(sout + s0 + d * GDN_DK))[lane] = st[d];
}
#if LAB_SSM
#define GDN_RECUR_DSKIP , const device float*
#else
#define GDN_RECUR_DSKIP
#endif
#define GDN_RECUR_G_INST(DVS, G)                                                                                   \
  template [[host_name("gdn_recur_dv" #DVS "_g" #G)]] [[kernel]] void gdn_recur_g<DVS, G>(                         \
      const device float*, device float*, const device float*, const device float*, const device float*,           \
      const device float*, const device float*, const device float*, const device float*, const device float*,     \
      const device float*, device float*, constant GdnParams&, const device uint* GDN_RECUR_DSKIP, uint, ushort,   \
      ushort);
GDN_RECUR_G_INST(4, 2)
