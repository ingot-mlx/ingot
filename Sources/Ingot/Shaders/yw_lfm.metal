// LFM2 / LFM2.5 (lfm2_moe, e.g. LiquidAI LFM2.5-24B-A2B): the pieces it does not share with Qwen3.5/3.6.
// Semantics per mlx-lm `models/lfm2_moe.py` (MIT): short convolution blocks (ShortConv) and attention blocks (heads
// of 64 without gate, in verify_attn*.metal), MoE with sigmoid router (expert_bias only to choose the top-k, weights
// renormalized without bias), first two layers with dense FFN, tied 6-bit embedding that also serves as lm_head.
// Concatenated after verify*.metal (VUNROLL, VParams, VMoEParams; EmbedParams lives in yy_model.metal: copy here).

// ---------------------------------------------------------------------------------------------
// Short convolution: in_proj → (B | C | x) → Bx = B ⊙ x → per-channel causal conv with SC_L taps (with the SC_W
// previous Bx rows; LFM2: 3 taps, LFM3: 7) → y = C ⊙ conv → out_proj. History (like the GDN tape): [SC_W window rows |
// B Bx rows of the block]; the next round takes the window at rows [nacc, nacc + SC_W). The taps are summed from the
// oldest to the newest, `w0·a + w1·b + w2·bx` with 3 taps as before LFM3.

constant constexpr uint SC_D = LAB_HIDDEN;
constant constexpr uint SC_L = LAB_SC_L;
constant constexpr uint SC_W = SC_L - 1;

struct SConvParams {
    uint B;      // rows of the block (verification) or real rows of the chunk (prefill)
    uint nacc;   // rows of the previous block that are folded in
    uint M;      // prefill: rows with padding (multiple of 64)
    uint dyn;    // 1: nacc is read from the round state on the GPU (st[1])
};

#if LAB_SC_L == 3
// LFM2's 3 taps: the kernels as they were before LFM3, verbatim. The generic ones below with SC_L = 3 changed the bits
// of toy-lfm2's logits (same tokens): the compiler contracts the sum into fmas in another way (risk 1 of the
// plan).

// Verification (B ≤ 16 rows): one thread per channel walks the rows in order. Grid D / 128, 128 threads.
[[kernel]] void sconv_step(const device float*  proj [[buffer(0)]],   // [B, 3D] (B | C | x)
                           const device bfloat* w    [[buffer(1)]],   // [D, 3] (tap 0 = the oldest)
                           const device float*  hin  [[buffer(2)]],   // history of the previous round
                           device float*        hout [[buffer(3)]],   // history of this round
                           device half*         y    [[buffer(4)]],   // [B, D]
                           constant SConvParams& p   [[buffer(5)]],
                           const device uint*   st   [[buffer(6)]],
                           uint c [[thread_position_in_grid]]) {
    const uint nacc = p.dyn ? st[1] : p.nacc;
    const float w0 = float(w[c * 3]), w1 = float(w[c * 3 + 1]), w2 = float(w[c * 3 + 2]);
    float a = hin[nacc * SC_D + c], b = hin[(nacc + 1) * SC_D + c];
    hout[c] = a;
    hout[SC_D + c] = b;
    for (uint t = 0; t < p.B; t++) {
        const device float* r = proj + size_t(t) * 3 * SC_D;
        const float bx = r[c] * r[2 * SC_D + c];
        hout[(2 + t) * SC_D + c] = bx;
        const float cv = w0 * a + w1 * b + w2 * bx;
        y[t * SC_D + c] = half(r[SC_D + c] * cv);
        a = b;
        b = bx;
    }
}

METAL_FUNC float sconv_bx(const device float* proj, const device float* hin, int i, uint c) {
    if (i < 0) return hin[uint(2 + i) * SC_D + c];
    const device float* r = proj + size_t(i) * 3 * SC_D;
    return r[c] * r[2 * SC_D + c];
}

// Prefill: each thread carries one channel and 16 consecutive rows (the 2 previous ones are recomputed from proj or
// the window). The padding rows (B ≤ t < M) give 0. Grid (D / 128, M / 16), 128 threads.
[[kernel]] void sconv_prefill(const device float*  proj [[buffer(0)]],   // [M, 3D]
                              const device bfloat* w    [[buffer(1)]],
                              const device float*  hin  [[buffer(2)]],   // window in rows 0..1
                              device half*         y    [[buffer(3)]],   // [M, D]
                              constant SConvParams& p   [[buffer(4)]],
                              uint2 tg [[threadgroup_position_in_grid]],
                              uint tid [[thread_index_in_threadgroup]]) {
    const uint c = tg.x * 128 + tid, t0 = tg.y * 16;
    const float w0 = float(w[c * 3]), w1 = float(w[c * 3 + 1]), w2 = float(w[c * 3 + 2]);
    float a = t0 < p.B + 2 ? sconv_bx(proj, hin, int(t0) - 2, c) : 0.0f;
    float b = t0 < p.B + 1 ? sconv_bx(proj, hin, int(t0) - 1, c) : 0.0f;
    for (uint t = t0; t < min(t0 + 16, p.M); t++) {
        if (t >= p.B) { y[size_t(t) * SC_D + c] = 0.0h; continue; }
        const device float* r = proj + size_t(t) * 3 * SC_D;
        const float bx = r[c] * r[2 * SC_D + c];
        const float cv = w0 * a + w1 * b + w2 * bx;
        y[size_t(t) * SC_D + c] = half(r[SC_D + c] * cv);
        a = b;
        b = bx;
    }
}

// Window after the chunk: the last 2 of [previous window (2) | Bx of the B rows]. Grid D threads.
[[kernel]] void sconv_hist(const device float* proj [[buffer(0)]],
                           const device float* hin  [[buffer(1)]],
                           device float*       hout [[buffer(2)]],
                           constant SConvParams& p  [[buffer(3)]],
                           uint c [[thread_position_in_grid]]) {
    for (uint j = 0; j < 2; j++) hout[j * SC_D + c] = sconv_bx(proj, hin, int(p.B + j) - 2, c);
}

#else
// Any other width (LFM3: 7 taps).

// The conv of one row: the SC_W window values (oldest first) and the new Bx, oldest tap first; then the window slides.
METAL_FUNC float sconv_tap(thread const float (&w)[SC_L], thread float (&win)[SC_W], float bx) {
    float cv = w[0] * win[0];
    for (uint j = 1; j < SC_W; j++) cv += w[j] * win[j];
    cv += w[SC_W] * bx;
    for (uint j = 0; j + 1 < SC_W; j++) win[j] = win[j + 1];
    win[SC_W - 1] = bx;
    return cv;
}

// Verification (B ≤ 16 rows): one thread per channel walks the rows in order. Grid D / 128, 128 threads.
[[kernel]] void sconv_step(const device float*  proj [[buffer(0)]],   // [B, 3D] (B | C | x)
                           const device bfloat* wt   [[buffer(1)]],   // [D, SC_L] (tap 0 = the oldest)
                           const device float*  hin  [[buffer(2)]],   // history of the previous round
                           device float*        hout [[buffer(3)]],   // history of this round
                           device half*         y    [[buffer(4)]],   // [B, D]
                           constant SConvParams& p   [[buffer(5)]],
                           const device uint*   st   [[buffer(6)]],
                           uint c [[thread_position_in_grid]]) {
    const uint nacc = p.dyn ? st[1] : p.nacc;
    float w[SC_L], win[SC_W];
    for (uint j = 0; j < SC_L; j++) w[j] = float(wt[c * SC_L + j]);
    for (uint j = 0; j < SC_W; j++) {
        win[j] = hin[(nacc + j) * SC_D + c];
        hout[j * SC_D + c] = win[j];
    }
    for (uint t = 0; t < p.B; t++) {
        const device float* r = proj + size_t(t) * 3 * SC_D;
        const float bx = r[c] * r[2 * SC_D + c];
        hout[(SC_W + t) * SC_D + c] = bx;
        y[t * SC_D + c] = half(r[SC_D + c] * sconv_tap(w, win, bx));
    }
}

METAL_FUNC float sconv_bx(const device float* proj, const device float* hin, int i, uint c) {
    if (i < 0) return hin[uint(int(SC_W) + i) * SC_D + c];
    const device float* r = proj + size_t(i) * 3 * SC_D;
    return r[c] * r[2 * SC_D + c];
}

// Prefill: each thread carries one channel and 16 consecutive rows (the SC_W previous ones are recomputed from proj or
// the window). The padding rows (B ≤ t < M) give 0. Grid (D / 128, M / 16), 128 threads.
[[kernel]] void sconv_prefill(const device float*  proj [[buffer(0)]],   // [M, 3D]
                              const device bfloat* wt   [[buffer(1)]],
                              const device float*  hin  [[buffer(2)]],   // window in rows 0..SC_W-1
                              device half*         y    [[buffer(3)]],   // [M, D]
                              constant SConvParams& p   [[buffer(4)]],
                              uint2 tg [[threadgroup_position_in_grid]],
                              uint tid [[thread_index_in_threadgroup]]) {
    const uint c = tg.x * 128 + tid, t0 = tg.y * 16;
    float w[SC_L], win[SC_W];
    for (uint j = 0; j < SC_L; j++) w[j] = float(wt[c * SC_L + j]);
    for (uint j = 0; j < SC_W; j++) {
        const int i = int(t0) - int(SC_W) + int(j);
        win[j] = i < int(p.B) ? sconv_bx(proj, hin, i, c) : 0.0f;
    }
    for (uint t = t0; t < min(t0 + 16, p.M); t++) {
        if (t >= p.B) { y[size_t(t) * SC_D + c] = 0.0h; continue; }
        const device float* r = proj + size_t(t) * 3 * SC_D;
        const float bx = r[c] * r[2 * SC_D + c];
        y[size_t(t) * SC_D + c] = half(r[SC_D + c] * sconv_tap(w, win, bx));
    }
}

// Window after the chunk: the last SC_W of [previous window (SC_W) | Bx of the B rows]. Grid D threads.
[[kernel]] void sconv_hist(const device float* proj [[buffer(0)]],
                           const device float* hin  [[buffer(1)]],
                           device float*       hout [[buffer(2)]],
                           constant SConvParams& p  [[buffer(3)]],
                           uint c [[thread_position_in_grid]]) {
    for (uint j = 0; j < SC_W; j++) hout[j * SC_D + c] = sconv_bx(proj, hin, int(p.B + j) - int(SC_W), c);
}
#endif

// ---------------------------------------------------------------------------------------------
// 6 bits (MLX format: 4 values in 3 bytes, little endian; groups of 64 with bf16 scale and bias).

METAL_FUNC uint q6_at(thread const uint (&wv)[12], int j) {
    const int bit = 6 * j, word = bit / 32, off = bit % 32;
    uint v = wv[word] >> off;
    if (off > 26) v |= wv[word + 1] << (32 - off);
    return v & 63u;
}

struct EmbedParams6 { uint K; uint flags; uint ldo; uint pad; };   // flags: 1 = of, 2 = oh

[[kernel]] void embed_rows_b6(const device uint*   tok [[buffer(0)]],
                              const device uchar*  w   [[buffer(1)]],
                              const device bfloat* s   [[buffer(2)]],
                              const device bfloat* b   [[buffer(3)]],
                              device float*        of  [[buffer(4)]],
                              device half*         oh  [[buffer(5)]],
                              constant EmbedParams6& p [[buffer(6)]],
                              uint row [[threadgroup_position_in_grid]],
                              uint tid [[thread_index_in_threadgroup]],
                              uint ntg [[threads_per_threadgroup]]) {
    const uint t = tok[row];
    const device uchar* wr = w + size_t(t) * (p.K * 3 / 4);
    for (uint i = tid * 4; i < p.K; i += ntg * 4) {
        const device uchar* q = wr + i / 4 * 3;
        const uint b0 = q[0], b1 = q[1], b2 = q[2];
        const uint v[4] = {b0 & 63u, (b0 >> 6) | ((b1 & 15u) << 2), (b1 >> 4) | ((b2 & 3u) << 4), b2 >> 2};
        const size_t g = size_t(t) * (p.K / 64) + i / 64;
        const float sc = float(s[g]), bi = float(b[g]);
        VUNROLL for (uint j = 0; j < 4; j++) {
            const float val = sc * float(v[j]) + bi;
            if (p.flags & 1) of[size_t(row) * p.ldo + i + j] = val;
            if (p.flags & 2) oh[size_t(row) * p.ldo + i + j] = half(val);
        }
    }
}

// Y[M ≤ 16, N] (+)= X · Wᵀ with 6-bit W (tied lm_head of LFM2.5: 128000 × 2048). Each simdgroup takes NW rows
// of W; each lane one group of 64 of K per step (48 bytes = 3 uint4). Grid N / (8·NW), 256 threads.
template <int NW>
[[kernel]] void qmv6(const device half*   x [[buffer(0)]],
                     const device uchar*  w [[buffer(1)]],
                     const device bfloat* s [[buffer(2)]],
                     const device bfloat* b [[buffer(3)]],
                     device float*        y [[buffer(4)]],
                     constant VParams&    p [[buffer(5)]],
                     uint tg [[threadgroup_position_in_grid]],
                     ushort sg [[simdgroup_index_in_threadgroup]],
                     ushort lane [[thread_index_in_simdgroup]]) {
    const uint n0 = (tg * 8 + sg) * NW;
    if (n0 >= p.N) return;
    const uint gpr = p.K / 64, bpr = p.K * 3 / 4;
    for (uint m = 0; m < p.M; m++) {
        float acc[NW];
        VUNROLL for (int r = 0; r < NW; r++) acc[r] = 0;
        for (uint g = lane; g < gpr; g += 32) {
            const device half* xp = x + size_t(m) * p.ldx + g * 64;
            float xs[64], xsum = 0;
            VUNROLL for (int i = 0; i < 16; i++) {
                const float4 v = float4(*(const device half4*)(xp + 4 * i));
                xs[4 * i] = v.x; xs[4 * i + 1] = v.y; xs[4 * i + 2] = v.z; xs[4 * i + 3] = v.w;
                xsum += (v.x + v.y) + (v.z + v.w);
            }
            VUNROLL for (int r = 0; r < NW; r++) {
                const device uint4* wp = (const device uint4*)(w + size_t(n0 + r) * bpr + g * 48);
                const uint4 u0 = wp[0], u1 = wp[1], u2 = wp[2];
                const uint wv[12] = {u0.x, u0.y, u0.z, u0.w, u1.x, u1.y, u1.z, u1.w, u2.x, u2.y, u2.z, u2.w};
                float d = 0;
                VUNROLL for (int j = 0; j < 64; j++) d += xs[j] * float(q6_at(wv, j));
                const size_t gi = size_t(n0 + r) * gpr + g;
                acc[r] += float(s[gi]) * d + float(b[gi]) * xsum;
            }
        }
        VUNROLL for (int r = 0; r < NW; r++) {
            const float v = simd_sum(acc[r]);
            if (lane == 0) {
                device float* o = y + size_t(m) * p.ldy + n0 + r;
                *o = (p.flags & 1) ? *o + v : v;
            }
        }
    }
}
template [[host_name("qmv6_nw4")]] [[kernel]] void qmv6<4>(const device half*, const device uchar*, const device bfloat*,
    const device bfloat*, device float*, constant VParams&, uint, ushort, ushort);

// ---------------------------------------------------------------------------------------------
// LFM2's sigmoid router runs on the common kernels (moe_route_sigmoid in vmoe_route and moe_route_topk_cfg,
// ab_moe_route.metal): lfm_moe_route and lfm_route_topk were removed.

// Prefill MoE combine without shared expert, added to the fp32 residual stream: out[t] += Σ_k w·Yd[pos].
[[kernel]] void moe_combine_acc_ns(const device half*  yd  [[buffer(0)]],
                                   const device uint*  pos [[buffer(1)]],
                                   const device float* wts [[buffer(2)]],
                                   device float*       out [[buffer(5)]],
                                   constant uint2&     p   [[buffer(6)]],   // (D, topk)
                                   uint t   [[threadgroup_position_in_grid]],
                                   uint tid [[thread_index_in_threadgroup]],
                                   uint ntg [[threads_per_threadgroup]]) {
    for (uint c = tid * 4; c < p.x; c += ntg * 4) {
        float4 a = 0;
        for (uint k = 0; k < p.y; k++) {
            const float w = wts[t * p.y + k];
            a += w * float4(*(const device half4*)(yd + size_t(pos[t * p.y + k]) * p.x + c));
        }
        device float4* o = (device float4*)(out + size_t(t) * p.x + c);
        *o += a;
    }
}

#if LAB_MXFP4
// The same with an fp32 Yd (the MXFP4 experts' moe_down2 writes fp32, LAB_MXFP4).
[[kernel]] void moe_combine_acc_ns_f32(const device float* yd  [[buffer(0)]],
                                       const device uint*  pos [[buffer(1)]],
                                       const device float* wts [[buffer(2)]],
                                       device float*       out [[buffer(5)]],
                                       constant uint2&     p   [[buffer(6)]],   // (D, topk)
                                       uint t   [[threadgroup_position_in_grid]],
                                       uint tid [[thread_index_in_threadgroup]],
                                       uint ntg [[threads_per_threadgroup]]) {
    for (uint c = tid * 4; c < p.x; c += ntg * 4) {
        float4 a = 0;
        for (uint k = 0; k < p.y; k++) {
            const float w = wts[t * p.y + k];
            a += w * *(const device float4*)(yd + size_t(pos[t * p.y + k]) * p.x + c);
        }
        device float4* o = (device float4*)(out + size_t(t) * p.x + c);
        *o += a;
    }
}
#endif

// Y[M ≤ 16, N] (+)= X · Wᵀ with 6-bit W on NAX (like the 4-bit vnax_qmv, verify_nax.metal): the thread with fragment
// column fn takes the 16 contiguous values [16·(fn/4), +16) of the 64-block of its W row (96 bits: 3
// words) and in chunk c uses {c, c+4, 8+c, 12+c}; x with the same permutation. 6-bit head of Flash-Next with
// the 8–16 rows of the round (qmv6 stays with the split of 40 groups over 32 lanes).
struct VNaxW6 {
    uint w0[3], w1[3];
    half s0, b0, s1, b1;
    // r picks the row (w0 or w1). No reference to a thread array: with MSL 4.1 the members of a const method
    // are in the generic address space and `thread const uint (&)[3]` does not compile.
    inline half dq(int r, int i, half s, half b) const {
        const int bit = 6 * i, word = bit / 32, off = bit % 32;
        uint v = (r ? w1[word] : w0[word]) >> off;
        if (off > 26) v |= (r ? w1[word + 1] : w0[word + 1]) << (32 - off);
        return fma(half(v & 63u), s, b);
    }
    inline nax_hfrag frag(int c) const {
        return nax_hfrag(dq(0, c, s0, b0), dq(0, c + 4, s0, b0), dq(0, 8 + c, s0, b0), dq(0, 12 + c, s0, b0),
                         dq(1, c, s1, b1), dq(1, c + 4, s1, b1), dq(1, 8 + c, s1, b1), dq(1, 12 + c, s1, b1));
    }
};

// Its loader for vnax_qmv (verify_nax.metal): the 16 values at kk are the 3 words at byte 3·kk / 4 of the row.
struct VLoad6 {
    using W = uchar;
    using T = VNaxW6;
    const device uchar* w;
    const device bfloat *s, *b;
    uint bpr, gpr;
    static inline VLoad6 make(const device uchar* w, const device bfloat* s, const device bfloat* b, uint K) {
        return VLoad6{w, s, b, K * 3 / 4, K / 64};
    }
    inline T load(uint n, uint kk, uint kb) const {
        T r;
        const device uint* p0 = (const device uint*)(w + size_t(n) * bpr + kk * 3 / 4);
        const device uint* p1 = (const device uint*)(w + size_t(n + 8) * bpr + kk * 3 / 4);
        VUNROLL for (int i = 0; i < 3; i++) { r.w0[i] = p0[i]; r.w1[i] = p1[i]; }
        r.s0 = half(float(s[size_t(n) * gpr + kb])); r.b0 = half(float(b[size_t(n) * gpr + kb]));
        r.s1 = half(float(s[size_t(n + 8) * gpr + kb])); r.b1 = half(float(b[size_t(n + 8) * gpr + kb]));
        return r;
    }
};
#define VNQ6_INST(NF, SGS) VNQ_INST_WL("vnax_qmv6_nf" #NF "_sg" #SGS, NF, SGS, VLoad6)
VNQ6_INST(2, 8)
