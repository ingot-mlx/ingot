// Full Qwen3.6-35B-A3B model and DFlash2 drafter (incoai/Qwen3.6-35B-A3B-DFlash2): the kernels that were missing for
// end-to-end decode. Concatenated after verify*.metal (VUNROLL, VParams, nax_*).
//
// DFlash2 drafter (6 dense qwen3 layers, hidden 2048, 32 × 128 of query, 8 KV, window 2048, non-causal within
// the block). Protocol and drafter shape per the mlx-serve implementation (`src/dflash.zig`, MIT) and splash
// (Apache-2.0): context = layers [1, 6, 11, 16, 22, 27, 32, 37] of the big model → fc → RMSNorm → K/V per layer;
// block = [anchor, mask × 7] with the big model's embedding; dynamic conv over groups of 16 channels before and
// after attention and the MLP; big model's lm_head over rows 1..7; selector with two rank-256 codebooks that
// chains the 16 candidates of each position greedily.

// ---------------------------------------------------------------------------------------------
// Quantized embedding (4 bits, groups of 64): row tok[r] → float (residual stream) and/or half.
struct EmbedParams { uint K; uint flags; uint ldo; uint pad; };   // flags: 1 = of, 2 = oh

// Groups of G: 64, or 32 (embed_rows_g32, only with LAB_G32).
template <int G>
[[kernel]] void embed_rows(const device uint*   tok [[buffer(0)]],
                           const device uint*   w   [[buffer(1)]],
                           const device bfloat* s   [[buffer(2)]],
                           const device bfloat* b   [[buffer(3)]],
                           device float*        of  [[buffer(4)]],
                           device half*         oh  [[buffer(5)]],
                           constant EmbedParams& p  [[buffer(6)]],
                           uint row [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]],
                           uint ntg [[threads_per_threadgroup]]) {
    const uint t = tok[row];
    const device uint* wr = w + size_t(t) * (p.K / 8);
    for (uint i = tid * 8; i < p.K; i += ntg * 8) {
        const uint word = wr[i / 8];
        const size_t g = size_t(t) * (p.K / G) + i / G;
        const float sc = float(s[g]), bi = float(b[g]);
        VUNROLL for (uint j = 0; j < 8; j++) {
            const float v = sc * float((word >> (4 * j)) & 15u) + bi;
            if (p.flags & 1) of[size_t(row) * p.ldo + i + j] = v;
            if (p.flags & 2) oh[size_t(row) * p.ldo + i + j] = half(v);
        }
    }
}

#define EMBED_ROWS_INST(NAME, G)                                                                                   \
  template [[host_name(NAME)]] [[kernel]] void embed_rows<G>(const device uint*, const device uint*,               \
      const device bfloat*, const device bfloat*, device float*, device half*, constant EmbedParams&, uint, uint,  \
      uint);
EMBED_ROWS_INST("embed_rows", 64)
#if LAB_G32
EMBED_ROWS_INST("embed_rows_g32", 32)
#endif

// Same with the 8-bit table (groups of 64, 4 values per uint; mixes from mlx-community/mlx-calib).
[[kernel]] void embed_rows_b8(const device uint*   tok [[buffer(0)]],
                              const device uint*   w   [[buffer(1)]],
                              const device bfloat* s   [[buffer(2)]],
                              const device bfloat* b   [[buffer(3)]],
                              device float*        of  [[buffer(4)]],
                              device half*         oh  [[buffer(5)]],
                              constant EmbedParams& p  [[buffer(6)]],
                              uint row [[threadgroup_position_in_grid]],
                              uint tid [[thread_index_in_threadgroup]],
                              uint ntg [[threads_per_threadgroup]]) {
    const uint t = tok[row];
    const device uint* wr = w + size_t(t) * (p.K / 4);
    for (uint i = tid * 4; i < p.K; i += ntg * 4) {
        const uint word = wr[i / 4];
        const size_t g = size_t(t) * (p.K / 64) + i / 64;
        const float sc = float(s[g]), bi = float(b[g]);
        VUNROLL for (uint j = 0; j < 4; j++) {
            const float v = sc * float((word >> (8 * j)) & 255u) + bi;
            if (p.flags & 1) of[size_t(row) * p.ldo + i + j] = v;
            if (p.flags & 2) oh[size_t(row) * p.ldo + i + j] = half(v);
        }
    }
}

// Same with the unquantized bf16 table (MLX-Serve pack of the 27B).
[[kernel]] void embed_rows_bf16(const device uint*   tok [[buffer(0)]],
                                const device bfloat* w   [[buffer(1)]],
                                const device bfloat* s   [[buffer(2)]],
                                const device bfloat* b   [[buffer(3)]],
                                device float*        of  [[buffer(4)]],
                                device half*         oh  [[buffer(5)]],
                                constant EmbedParams& p  [[buffer(6)]],
                                uint row [[threadgroup_position_in_grid]],
                                uint tid [[thread_index_in_threadgroup]],
                                uint ntg [[threads_per_threadgroup]]) {
    const device bfloat* wr = w + size_t(tok[row]) * p.K;
    for (uint i = tid; i < p.K; i += ntg) {
        const float v = float(wr[i]);
        if (p.flags & 1) of[size_t(row) * p.ldo + i] = v;
        if (p.flags & 2) oh[size_t(row) * p.ldo + i] = half(v);
    }
}

// Capture of the residual stream (fp32) into half with another row stride: grid (rows), 256 threads.
struct CopyParams { uint D; uint ldi; uint ldo; uint pad; };
[[kernel]] void capture_rows(const device float* x [[buffer(0)]],
                             device half*        o [[buffer(1)]],
                             constant CopyParams& p [[buffer(2)]],
                             uint row [[threadgroup_position_in_grid]],
                             uint tid [[thread_index_in_threadgroup]],
                             uint ntg [[threads_per_threadgroup]]) {
    for (uint i = tid; i < p.D; i += ntg) o[size_t(row) * p.ldo + i] = half(x[size_t(row) * p.ldi + i]);
}

// RMSNorm of half rows → half.
[[kernel]] void rmsnorm_rows_h(const device half*   x [[buffer(0)]],
                               const device LAB_NWT* w [[buffer(1)]],
                               device half*         o [[buffer(2)]],
                               constant NormParams& p [[buffer(3)]],
                               uint row [[threadgroup_position_in_grid]],
                               uint tid [[thread_index_in_threadgroup]],
                               ushort sg [[simdgroup_index_in_threadgroup]],
                               ushort lane [[thread_index_in_simdgroup]],
                               uint ntg [[threads_per_threadgroup]]) {
    threadgroup float part[32];
    const device half* xr = x + size_t(row) * p.ldx;
    float ss = 0;
    for (uint i = tid; i < p.D; i += ntg) { const float v = float(xr[i]); ss += v * v; }
    ss = simd_sum(ss);
    if (lane == 0) part[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0;
    for (uint i = 0; i < ntg / 32; i++) tot += part[i];
    const float r = rsqrt(tot / float(p.D) + p.eps);
    for (uint i = tid; i < p.D; i += ntg) o[size_t(row) * p.ldo + i] = half(float(xr[i]) * r * float(w[i]));
}

// Prefill MoE combine added to the fp32 residual stream (like moe_combine, which writes half).
[[kernel]] void moe_combine_acc(const device half*  yd    [[buffer(0)]],
                                const device uint*  pos   [[buffer(1)]],
                                const device float* wts   [[buffer(2)]],
                                const device half*  ys    [[buffer(3)]],
                                const device float* sgate [[buffer(4)]],
                                device float*       out   [[buffer(5)]],
                                constant uint2&     p     [[buffer(6)]],   // (D, topk)
                                uint t   [[threadgroup_position_in_grid]],
                                uint tid [[thread_index_in_threadgroup]],
                                uint ntg [[threads_per_threadgroup]]) {
    const float g = sgate[t];
    for (uint c = tid * 4; c < p.x; c += ntg * 4) {
        float4 a = g * float4(*(const device half4*)(ys + size_t(t) * p.x + c));
        for (uint k = 0; k < p.y; k++) {
            const float w = wts[t * p.y + k];
            a += w * float4(*(const device half4*)(yd + size_t(pos[t * p.y + k]) * p.x + c));
        }
        device float4* o = (device float4*)(out + size_t(t) * p.x + c);
        *o += a;
    }
}

// ---------------------------------------------------------------------------------------------
// Drafter DFlash2.

constant constexpr uint DF_H = LAB_HIDDEN, DF_GS = 16, DF_G = DF_H / DF_GS;   // hidden, channels per group, groups of the conv
constant constexpr uint DF_DYN = 4 * DF_G;                       // output of kernel_projection per row
// The context window and the ring slots are per layer (DAttnParams win/slots): 2048 and 2080 (window + up to 16 of
// block, to a multiple of 32) except the full-attention layer of a DFlash v1 drafter (8192 and 8224).
constant constexpr uint DF_HD = 128, DF_QH = 32, DF_KVH = 8, DF_QKV = 6144;

// Two-tap dynamic conv: out_t = Σ_tap (base[half][tap] + dyn_t[half][tap][c/16]) ⊙ x_{t−tap}, with x_{−1} = 0
// (the block starts from zero). base_kernel [2 (prepare|finish), 2 (tap), 2048]; dyn [L, 2, 2, 128] fp32 (output of
// kernel_projection over the sublayer's normalized input). Grid: L·2048/256 threadgroups of 256.
[[kernel]] void dconv_prep(const device half*   x    [[buffer(0)]],   // [L, 2048] (normalized input)
                           const device float*  dyn  [[buffer(1)]],   // [L, 512]
                           const device bfloat* base [[buffer(2)]],   // [2, 2, 2048]
                           device half*         o    [[buffer(3)]],   // [L, 2048]
                           uint i [[thread_position_in_grid]]) {
    const uint t = i / DF_H, c = i % DF_H, g = c / DF_GS;
    const float x0 = float(x[t * DF_H + c]), x1 = t > 0 ? float(x[(t - 1) * DF_H + c]) : 0.0f;
    const float k0 = float(base[c]) + dyn[t * DF_DYN + 0 * DF_G + g];
    const float k1 = float(base[DF_H + c]) + dyn[t * DF_DYN + 1 * DF_G + g];
    o[t * DF_H + c] = half(k0 * x0 + k1 * x1);
}

// finish: y (output of o_proj or down, fp32) convolved with base[1] and dyn[:, 1], added to the residual.
[[kernel]] void dconv_fin_add(const device float*  y    [[buffer(0)]],   // [L, 2048]
                              const device float*  dyn  [[buffer(1)]],
                              const device bfloat* base [[buffer(2)]],
                              device float*        x    [[buffer(3)]],   // residual [L, 2048]
                              uint i [[thread_position_in_grid]]) {
    const uint t = i / DF_H, c = i % DF_H, g = c / DF_GS;
    const float y0 = y[t * DF_H + c], y1 = t > 0 ? y[(t - 1) * DF_H + c] : 0.0f;
    const float k0 = float(base[2 * DF_H + c]) + dyn[t * DF_DYN + 2 * DF_G + g];
    const float k1 = float(base[3 * DF_H + c]) + dyn[t * DF_DYN + 3 * DF_G + g];
    x[t * DF_H + c] += k0 * y0 + k1 * y1;
}

// SwiGLU of [L, 2N] (gate | up, fp32) to half [L, N]. ACT: activation (lab_glu; `_gelu` instance with LAB_FFN_GELU;
// `_relu2` with LAB_FFN_RELU2, whose input is up only, [L, N]).
template <int ACT>
[[kernel]] void dswiglu(const device float* gu [[buffer(0)]], device half* h [[buffer(1)]],
                        constant uint& N [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if constexpr (ACT == 3) {
        h[i] = half(lab_glu<ACT>(0.0f, gu[i]));
    } else {
        const uint t = i / N, c = i % N;
        const float g = gu[size_t(t) * 2 * N + c], u = gu[size_t(t) * 2 * N + N + c];
        h[i] = half(lab_glu<ACT>(g, u));
    }
}

// Same with gate | up in half (dense prefill FFN): 4 columns per thread.
template <int ACT>
[[kernel]] void dswiglu_hh(const device half* gu [[buffer(0)]], device half* h [[buffer(1)]],
                           constant uint& N [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if constexpr (ACT == 3) {
        const float4 u = float4(*(const device half4*)(gu + size_t(4) * i));
        *(device half4*)(h + size_t(4) * i) = half4(lab_glu<ACT>(float4(0), u));
    } else {
        const uint t = (4 * i) / N, c = (4 * i) % N;
        const float4 g = float4(*(const device half4*)(gu + size_t(t) * 2 * N + c));
        const float4 u = float4(*(const device half4*)(gu + size_t(t) * 2 * N + N + c));
        *(device half4*)(h + size_t(4) * i) = half4(lab_glu<ACT>(g, u));
    }
}

#define DSWIGLU_INST(SUF, ACT)                                                                                     \
  template [[host_name("dswiglu" SUF)]] [[kernel]] void dswiglu<ACT>(const device float*, device half*,           \
      constant uint&, uint);                                                                                     \
  template [[host_name("dswiglu_hh" SUF)]] [[kernel]] void dswiglu_hh<ACT>(const device half*, device half*,      \
      constant uint&, uint);
DSWIGLU_INST("", 0)
#if LAB_FFN_GELU
DSWIGLU_INST("_gelu", 1)
#endif
#if LAB_FFN_RELU2
DSWIGLU_INST("_relu2", 3)
#endif

struct DAttnParams {
    uint P;       // absolute position of the anchor (= context tokens)
    uint nvalid;  // dctx_write: valid rows
    uint pos0;    // dctx_write: position of the first row
    uint C;       // keys per chunk
    uint nsplit;
    float scale, eps, theta;
    uint dyn;     // 1: P = st[0], nvalid = st[2], pos0 = st[3] (round state on the GPU)
    uint block;   // rows of the block: 8 or 16 (query rows per KV head: 4 · block)
    uint win;     // context window of the layer (ring of `win` positions, then the block's slots)
    uint slots;   // slots of the layer's ring: win + up to 16 of block, to a multiple of 32
};

// Per-head RMSNorm (128) with weight + standard RoPE (non-interleaved halves, 128 dimensions) in a threadgroup of
// 128 threads (one dimension each).
inline float dnorm_rope(float x, uint d, const device bfloat* nw, float pos, float eps, float theta,
                        threadgroup float* xs, threadgroup float* part, ushort sg, ushort lane) {
    const float ss = simd_sum(x * x);
    if (lane == 0) part[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float tot = part[0] + part[1] + part[2] + part[3];
    x = x * rsqrt(tot / float(DF_HD) + eps) * float(nw[d]);
    xs[d] = x;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint i = d % 64;
    const float ang = pos * pow(theta, -float(i) / 64.0f);
    const float c = cos(ang), s = sin(ang);
    return d < 64 ? x * c - xs[d + 64] * s : x * c + xs[d - 64] * s;
}

// Block: q (32 heads), k (8) and v (8) of the L = block rows. Grid (48, L), 128 threads.
[[kernel]] void dattn_prework(const device float*  qkv [[buffer(0)]],   // [L, 6144]
                              const device bfloat* qnw [[buffer(1)]],
                              const device bfloat* knw [[buffer(2)]],
                              device half*         q   [[buffer(3)]],   // [8, 4·block (m = row·4 + h), 128]
                              device half*         kc  [[buffer(4)]],   // [8, slots, 128]
                              device half*         vt  [[buffer(5)]],   // [8, 128, slots]
                              constant DAttnParams& p  [[buffer(6)]],
                              const device uint*   st  [[buffer(7)]],
                              uint2 tg [[threadgroup_position_in_grid]],
                              uint d [[thread_index_in_threadgroup]],
                              ushort sg [[simdgroup_index_in_threadgroup]],
                              ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float xs[DF_HD];
    threadgroup float part[4];
    const uint slot = tg.x, t = tg.y, P = p.dyn ? st[0] : p.P;
    const float x = qkv[t * DF_QKV + slot * DF_HD + d];
    if (slot >= 40) {
        vt[(size_t(slot - 40) * DF_HD + d) * p.slots + p.win + t] = half(x);
        return;
    }
    const float r = dnorm_rope(x, d, slot < 32 ? qnw : knw, float(P + t), p.eps, p.theta, xs, part, sg, lane);
    if (slot < 32) q[(size_t(slot / 4) * 4 * p.block + t * 4 + slot % 4) * DF_HD + d] = half(r);
    else kc[(size_t(slot - 32) * p.slots + p.win + t) * DF_HD + d] = half(r);
}

// Context: K (k_norm + RoPE) and V of the accepted rows into the window ring (slot = position % win).
// Grid (16, rows), 128 threads; only the first nvalid write.
[[kernel]] void dctx_write(const device float*  kv  [[buffer(0)]],   // [rows, 2048] (k 1024 | v 1024)
                           const device bfloat* knw [[buffer(1)]],
                           device half*         kc  [[buffer(2)]],
                           device half*         vt  [[buffer(3)]],
                           constant DAttnParams& p  [[buffer(4)]],
                           const device uint*   st  [[buffer(5)]],
                           uint2 tg [[threadgroup_position_in_grid]],
                           uint d [[thread_index_in_threadgroup]],
                           ushort sg [[simdgroup_index_in_threadgroup]],
                           ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float xs[DF_HD];
    threadgroup float part[4];
    const uint slot = tg.x, t = tg.y;
    if (t >= (p.dyn ? st[2] : p.nvalid)) return;
    const uint pos = (p.dyn ? st[3] : p.pos0) + t, ring = pos % p.win;
    const float x = kv[t * 2048 + slot * DF_HD + d];
    if (slot >= 8) {
        vt[(size_t(slot - 8) * DF_HD + d) * p.slots + ring] = half(x);
        return;
    }
    const float r = dnorm_rope(x, d, knw, float(pos), p.eps, p.theta, xs, part, sg, lane);
    kc[(size_t(slot) * p.slots + ring) * DF_HD + d] = half(r);
}

// Does row t of the block see slot n? Window: context position k visible if P + t − k < W; block: all B rows.
inline bool dvisible(uint n, uint t, uint P, uint B, uint W) {
    if (n >= W) return n < W + B;
    const int back = (int(P) - 1 - int(n)) % int(W);
    const int k = int(P) - 1 - (back < 0 ? back + int(W) : back);
    return k >= 0 && int(P + t) - k < int(W);
}

// Drafter attention on NAX (adapted from the fp16 verification kernel since removed): threadgroup = (chunk of C
// keys, 16 queries, KV head), with Q = 4 · block queries per KV head (grid (nsplit, block / 4, 8)); 4 simdgroups, s
// takes dimensions [32s, 32s + 32).
[[kernel]] void dattn_nax(const device half* q     [[buffer(0)]],
                          const device half* kc    [[buffer(1)]],
                          const device half* vt    [[buffer(2)]],
                          device float*      opart [[buffer(3)]],   // [8, nsplit, Q, 128]
                          device float2*     ml    [[buffer(4)]],   // [8, nsplit, Q]
                          constant DAttnParams& p  [[buffer(5)]],
                          const device uint*   st  [[buffer(6)]],
                          uint3 tg [[threadgroup_position_in_grid]],
                          ushort sg [[simdgroup_index_in_threadgroup]],
                          ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[2][4][16 * 32];
    const uint split = tg.x, qf = tg.y, kvh = tg.z, Q = 4 * p.block, P = p.dyn ? st[0] : p.P;
    const uint key0 = split * p.C, kend = min(key0 + p.C, p.slots), S = p.slots;
    const short2 co = nax_coord(lane);
    const uint fm = co.x, fn = co.y;
    const uint d0 = 32 * sg;
    const device half* qh = q + size_t(kvh) * Q * DF_HD;
    const device half* kh = kc + size_t(kvh) * S * DF_HD;
    const device half* vh = vt + size_t(kvh) * DF_HD * S;
    const uint qr0 = qf * 16 + fm, qr1 = qf * 16 + fm + 8;
    // A chunk of ring slots no position < P has reached yet (n ≥ P) sees nothing: the same (m, l) = (−∞, 0) the loop
    // would leave, which dattn_reduce skips, without reading its keys (the 8192 ring of a short context).
    if (key0 >= min(P, p.win) && key0 < p.win) {
        if (sg == 0 && (lane & 9) == 0) {
            device float2* mp = ml + (size_t(kvh) * p.nsplit + split) * Q;
            mp[qr0] = float2(-INFINITY, 0.0f);
            mp[qr1] = float2(-INFINITY, 0.0f);
        }
        return;
    }
    nax_hfrag qa[2];
    VUNROLL for (int c = 0; c < 2; c++) {
        const half4 v0 = *(const device half4*)(qh + size_t(qr0) * DF_HD + d0 + 16 * c + fn);
        const half4 v1 = *(const device half4*)(qh + size_t(qr1) * DF_HD + d0 + 16 * c + fn);
        qa[c] = nax_hfrag(v0, v1);
    }
    const uint t0 = qr0 / 4, t1 = qr1 / 4;
    nax_ffrag o[2];
    o[0] = 0; o[1] = 0;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0, l1 = 0;
    uint buf = 0;
    for (uint k0 = key0; k0 < kend; k0 += 32, buf ^= 1) {
        nax_ffrag s0 = 0, s1 = 0;
        VUNROLL for (int c = 0; c < 2; c++) {
            nax_hfrag kb0, kb1;
            nax_load(kb0, kh + size_t(k0) * DF_HD + d0 + 16 * c, DF_HD, co);
            nax_load(kb1, kh + size_t(k0 + 16) * DF_HD + d0 + 16 * c, DF_HD, co);
            nax_mma2(s0, s1, qa[c], kb0, kb1);
        }
        VUNROLL for (int e = 0; e < 8; e++) {
            const uint m = fm + 8 * (e / 4), n = fn + e % 4;
            red[buf][sg][m * 32 + n] = s0[e];
            red[buf][sg][m * 32 + 16 + n] = s1[e];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float sv[16];
        float mx0 = -INFINITY, mx1 = -INFINITY;
        VUNROLL for (int f = 0; f < 2; f++) VUNROLL for (int e = 0; e < 8; e++) {
            const uint m = fm + 8 * (e / 4), n = 16 * f + fn + e % 4;
            float v = red[buf][0][m * 32 + n] + red[buf][1][m * 32 + n] + red[buf][2][m * 32 + n] + red[buf][3][m * 32 + n];
            const uint key = k0 + n;
            const bool ok = key < kend && dvisible(key, e < 4 ? t0 : t1, P, p.block, p.win);
            v = ok ? v * p.scale : -INFINITY;
            sv[f * 8 + e] = v;
            if (e < 4) mx0 = max(mx0, v); else mx1 = max(mx1, v);
        }
        mx0 = max(mx0, simd_shuffle_xor(mx0, 1)); mx0 = max(mx0, simd_shuffle_xor(mx0, 8));
        mx1 = max(mx1, simd_shuffle_xor(mx1, 1)); mx1 = max(mx1, simd_shuffle_xor(mx1, 8));
        const float mn0 = max(m0, mx0), mn1 = max(m1, mx1);
        const float a0 = mn0 == -INFINITY ? 1.0f : exp(m0 - mn0), a1 = mn1 == -INFINITY ? 1.0f : exp(m1 - mn1);
        float rs0 = 0, rs1 = 0;
        nax_hfrag pa[2];
        VUNROLL for (int f = 0; f < 2; f++) VUNROLL for (int e = 0; e < 8; e++) {
            const float mn = e < 4 ? mn0 : mn1;
            const float pv = mn == -INFINITY ? 0.0f : exp(sv[f * 8 + e] - mn);
            if (e < 4) rs0 += pv; else rs1 += pv;
            pa[f][e] = half(pv);
        }
        rs0 += simd_shuffle_xor(rs0, 1); rs0 += simd_shuffle_xor(rs0, 8);
        rs1 += simd_shuffle_xor(rs1, 1); rs1 += simd_shuffle_xor(rs1, 8);
        l0 = l0 * a0 + rs0; l1 = l1 * a1 + rs1;
        m0 = mn0; m1 = mn1;
        VUNROLL for (int f = 0; f < 2; f++) VUNROLL for (int e = 0; e < 8; e++) o[f][e] *= (e < 4 ? a0 : a1);
        VUNROLL for (int kc2 = 0; kc2 < 2; kc2++) {
            nax_hfrag vb[2];
            VUNROLL for (int f = 0; f < 2; f++) nax_load(vb[f], vh + size_t(d0 + 16 * f) * S + k0 + 16 * kc2, S, co);
            nax_mma2(o[0], o[1], pa[kc2], vb[0], vb[1]);
        }
    }
    device float* op = opart + (size_t(kvh) * p.nsplit + split) * Q * DF_HD;
    VUNROLL for (int f = 0; f < 2; f++) VUNROLL for (int i = 0; i < 2; i++) {
        const uint m = qf * 16 + fm + 8 * i;
        *(device float4*)(op + size_t(m) * DF_HD + d0 + 16 * f + fn) =
            float4(o[f][4 * i], o[f][4 * i + 1], o[f][4 * i + 2], o[f][4 * i + 3]);
    }
    if (sg == 0 && (lane & 9) == 0) {
        device float2* mp = ml + (size_t(kvh) * p.nsplit + split) * Q;
        mp[qr0] = float2(m0, l0);
        mp[qr1] = float2(m1, l1);
    }
}

// Combine the chunks: out[t, (kvh·4 + h)·128 + d]. Grid (4 · block, 8), 128 threads.
[[kernel]] void dattn_reduce(const device float*  opart [[buffer(0)]],
                             const device float2* ml    [[buffer(1)]],
                             device half*         out   [[buffer(2)]],   // [block, 4096]
                             constant DAttnParams& p    [[buffer(3)]],
                             uint2 tg [[threadgroup_position_in_grid]],
                             uint d [[thread_index_in_threadgroup]]) {
    const uint m = tg.x, kvh = tg.y, Q = 4 * p.block, t = m / 4, h = kvh * 4 + m % 4;
    const device float2* mlh = ml + size_t(kvh) * p.nsplit * Q;
    float M = -INFINITY;
    for (uint s = 0; s < p.nsplit; s++) M = max(M, mlh[s * Q + m].x);
    float o = 0, L = 0;
    for (uint s = 0; s < p.nsplit; s++) {
        const float2 v = mlh[s * Q + m];
        if (v.x == -INFINITY) continue;
        const float w = exp(v.x - M);
        L += w * v.y;
        o += w * opart[((size_t(kvh) * p.nsplit + s) * Q + m) * DF_HD + d];
    }
    out[size_t(t) * 4096 + h * DF_HD + d] = half(o / L);
}

// Top-16 of each row of logits (fp32): each thread keeps its sorted list, each simdgroup merges those of its 32
// lanes and simdgroup 0 merges those of the 32 simdgroups. Ties go to the lower index. Grid (rows), 1024 threads.
constant constexpr int TK = 16;
inline bool tk_better(float v, uint i, float w, uint j) { return v > w || (v == w && i < j); }

[[kernel]] void topk16_rows(const device float* x   [[buffer(0)]],
                            device uint*        idx [[buffer(1)]],
                            device float*       val [[buffer(2)]],
                            constant uint&      V   [[buffer(3)]],
                            uint row [[threadgroup_position_in_grid]],
                            uint tid [[thread_index_in_threadgroup]],
                            ushort sg [[simdgroup_index_in_threadgroup]],
                            ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float sv[32 * TK];
    threadgroup uint si[32 * TK];
    const device float* xr = x + size_t(row) * V;
    float v[TK]; uint id[TK];
    VUNROLL for (int j = 0; j < TK; j++) { v[j] = -INFINITY; id[j] = 0xffffffffu; }
    for (uint i = tid; i < V; i += 1024) {
        float c = xr[i]; uint ci = i;
        if (!tk_better(c, ci, v[TK - 1], id[TK - 1])) continue;
        VUNROLL for (int j = 0; j < TK; j++) {
            const bool gt = tk_better(c, ci, v[j], id[j]);
            const float tv = v[j]; const uint ti = id[j];
            v[j] = gt ? c : tv; id[j] = gt ? ci : ti;
            c = gt ? tv : c; ci = gt ? ti : ci;
        }
    }
    // Merge in the simdgroup: TK rounds, each pulls the best list head.
    for (int r = 0; r < TK; r++) {
        const float best = simd_max(v[0]);
        const uint bid = simd_min(v[0] == best ? id[0] : 0xffffffffu);
        if (lane == 0) { sv[sg * TK + r] = best; si[sg * TK + r] = bid; }
        if (v[0] == best && id[0] == bid) {
            VUNROLL for (int j = 0; j + 1 < TK; j++) { v[j] = v[j + 1]; id[j] = id[j + 1]; }
            v[TK - 1] = -INFINITY; id[TK - 1] = 0xffffffffu;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg != 0) return;
    VUNROLL for (int j = 0; j < TK; j++) { v[j] = sv[lane * TK + j]; id[j] = si[lane * TK + j]; }
    for (int r = 0; r < TK; r++) {
        const float best = simd_max(v[0]);
        const uint bid = simd_min(v[0] == best ? id[0] : 0xffffffffu);
        if (lane == 0) { idx[row * TK + r] = bid; val[row * TK + r] = best; }
        if (v[0] == best && id[0] == bid) {
            VUNROLL for (int j = 0; j + 1 < TK; j++) { v[j] = v[j + 1]; id[j] = id[j + 1]; }
            v[TK - 1] = -INFINITY; id[TK - 1] = 0xffffffffu;
        }
    }
}

// DFlash v1: the draft of each position is its best candidate (the top-16 is sorted, so index 0). Its confidence is
// softmax(v/τ) of that candidate over the top-16 values, as in `dselect` (the verification rows of `spec_rows_choose`).
[[kernel]] void dpick_top1(const device uint*  cand [[buffer(0)]],   // [M, 16]
                           device uint*        tok  [[buffer(1)]],   // [1 + M]
                           constant uint&      M    [[buffer(2)]],
                           const device float* val  [[buffer(3)]],   // [M, 16], descending
                           device float*       conf [[buffer(4)]],   // [M]
                           constant float&     tau  [[buffer(5)]],
                           uint t [[thread_position_in_grid]]) {
    if (t >= M) return;
    tok[1 + t] = cand[t * TK];
    const float best = val[t * TK];
    float z = 0;
    for (int j = 0; j < TK; j++) z += exp((val[t * TK + j] - best) / tau);
    conf[t] = 1.0f / z;
}

// DFlash2 selector, greedy: at each position t, score of candidate j = logit_j + Σ_r pred[prev][r] ·
// hp[t][r] · succ[cand_j][r] (prev = anchor at t = 0, otherwise the one chosen at t − 1); the maximum is kept (the
// first if tied). Writes the drafts to tok[1..M]. One threadgroup of 256 threads (one per component r).
[[kernel]] void dselect(const device uint*   cand [[buffer(0)]],   // [M, 16]
                        const device float*  un   [[buffer(1)]],   // [M, 16]
                        const device float*  hp   [[buffer(2)]],   // [M, 256]
                        const device bfloat* pred [[buffer(3)]],   // [V, 256]
                        const device bfloat* succ [[buffer(4)]],   // [V, 256]
                        device uint*         tok  [[buffer(5)]],   // [1 + M]: tok[0] = ancla
                        constant uint&       M    [[buffer(6)]],
                        device float*        conf [[buffer(7)]],   // [M]: softmax(s/τ) of the chosen one over the top-16
                        constant float&      tau  [[buffer(8)]],
                        uint r [[thread_index_in_threadgroup]],
                        ushort sg [[simdgroup_index_in_threadgroup]],
                        ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[8][TK];
    threadgroup uint chosen;
    uint prev = tok[0];
    for (uint t = 0; t < M; t++) {
        const float a = float(pred[size_t(prev) * 256 + r]) * hp[t * 256 + r];
        VUNROLL for (int j = 0; j < TK; j++) {
            const float e = simd_sum(a * float(succ[size_t(cand[t * TK + j]) * 256 + r]));
            if (lane == 0) part[sg][j] = e;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (r == 0) {
            uint best = 0; float bs = -INFINITY;
            float scs[TK];
            for (int j = 0; j < TK; j++) {
                float e = 0;
                for (int q = 0; q < 8; q++) e += part[q][j];
                const float sc = un[t * TK + j] + e;
                scs[j] = sc;
                if (sc > bs) { bs = sc; best = j; }
            }
            chosen = cand[t * TK + best];
            tok[1 + t] = chosen;
            // Draft confidence: probability of the chosen one in softmax(s/τ) over the 16 candidates.
            float z = 0;
            for (int j = 0; j < TK; j++) z += exp((scs[j] - bs) / tau);
            conf[t] = 1.0f / z;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        prev = chosen;
    }
}

// Record of each round (rec[round % ring]): m, the m + 1 emitted tokens (≤ 17) and P before in REC_P.
constant constexpr uint REC_STRIDE = 24, REC_P = 20, REC_ROWS = 21, RND_PER_ROUND = 32;

// Greedy acceptance on the GPU at the end of each round (chained decode: the CPU encodes the next round w/o waiting).
// m = drafts that match the argmax of the previous row; stores in rec[round % RING] (m, emitted tokens,
// P before) and advances the state: st = [P, nacc, pending context rows, their position, anchor, round].
[[kernel]] void spec_accept(device uint*       st   [[buffer(0)]],
                            device uint*       vtok [[buffer(1)]],
                            const device uint* top1 [[buffer(2)]],
                            device uint*       dtok [[buffer(3)]],
                            device uint*       rec  [[buffer(4)]],
                            constant uint&     ring [[buffer(5)]],
                            constant uint&     rows [[buffer(6)]],   // verified rows (≤ 16; 0 = st[6])
                            uint tid [[thread_position_in_grid]]) {
    if (tid != 0) return;
    const uint nr = rows ? rows : st[6];   // 0: rows chosen by spec_rows_choose in this round
    uint m = 0;
    while (m + 1 < nr && vtok[m + 1] == top1[m]) m++;
    device uint* r = rec + (st[5] % ring) * REC_STRIDE;
    r[0] = m;
    r[REC_ROWS] = nr;
    for (uint i = 1; i <= m; i++) r[i] = vtok[i];
    r[m + 1] = top1[m];
    r[REC_P] = st[0];
    st[3] = st[0]; st[2] = m + 1; st[0] += m + 1; st[1] = m + 1; st[4] = top1[m]; st[5] += 1;
    vtok[0] = top1[m]; dtok[0] = top1[m];
}

// Verification rows of the round chosen on the GPU, after the drafter and before verification (the chained
// round does not let the CPU see the draft). With each draft's calibrated confidence (Platt per depth group
// 0, 1, 2, ≥ 3 over logit(conf)), expected tokens with k rows E(k) = 1 + Σ_{i<k−1} Π_{t≤i} q_t, and the k in
// [kmin, kmax] with the highest E(k)/cost(k) is kept (cost per rows measured, from the round's context table).
// Writes k to st[6], read by vmoe_route, vmoe_combine9, the GDN recurrence and spec_accept[_sample].
// Depends only on the drafter, never on the big model: the output is the same (greedy) or has the same distribution.
struct RowsParams {
    float a[4], b[4];    // Platt: q = sigmoid(a·logit(conf) + b)
    float cost[17];      // ms per round with k rows (index k)
    uint kmin, kmax;
};

[[kernel]] void spec_rows_choose(device uint*         st   [[buffer(0)]],
                                 const device float*  conf [[buffer(1)]],
                                 constant RowsParams& p    [[buffer(2)]],
                                 uint tid [[thread_position_in_grid]]) {
    if (tid != 0) return;
    float acc = 1, lp = 1, bv = -1;
    uint best = p.kmax;
    for (uint t = 0; t + 1 < p.kmax; t++) {
        const uint g = min(t, 3u);
        const float c = clamp(conf[t], 1e-6f, 1.0f - 1e-6f);
        const float q = 1.0f / (1.0f + exp(-(p.a[g] * log(c / (1.0f - c)) + p.b[g])));
        lp *= q; acc += lp;
        const uint k = t + 2;
        if (k >= p.kmin) {
            const float v = acc / p.cost[k];
            if (v > bv) { bv = v; best = k; }
        }
    }
    st[6] = best;
}
