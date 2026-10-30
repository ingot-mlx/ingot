// Verification (B <= 16 rows) on the matrix accelerators (NAX, matmul2d 16×32×16), as splash does in
// its linear decode kernels (Apache-2.0, incoai/splash 1.3.2: "8 rows per lane", tiles of 8/16/32 rows).
// With B >= 8 the scalar qmv becomes compute-bound (B FMAs per weight); here the products go to matmul2d and the
// core only dequantizes.
//
// Fragments of 16 rows (the B rows, repeated up to 16) × 16 of K. K is permuted inside each block of 64:
// the thread with fragment column fn carries the 16 contiguous values [16·(fn/4), +16) of the block (a 4-bit uint2
// per row of W), and in chunk c (0..3) of the block it uses the values {c, c+4, 8+c, 12+c} of those 16. This is exactly
// the order of the 0x6400 trick (a known half-precision bit trick): ((w >> 4c) & 0x000F000F) | 0x64006400 gives half2 (1024+q_c, 1024+q_c+4).
// x uses the same permutation, so the product does not change.
//
// Concatenated after qmm_nax.metal (nax_*) and verify.metal (VParams, VMoEParams).

// 16 contiguous halfs.
inline void vnax_load16(thread half (&d)[16], const device half* p) {
    const device half4* q = (const device half4*)p;
    VUNROLL for (int i = 0; i < 4; i++) { half4 v = q[i]; d[4 * i] = v.x; d[4 * i + 1] = v.y; d[4 * i + 2] = v.z; d[4 * i + 3] = v.w; }
}

inline nax_hfrag vnax_afrag(thread const half (&x0)[16], thread const half (&x1)[16], int c) {
    nax_hfrag a;
    a[0] = x0[c]; a[1] = x0[c + 4]; a[2] = x0[8 + c]; a[3] = x0[12 + c];
    a[4] = x1[c]; a[5] = x1[c + 4]; a[6] = x1[8 + c]; a[7] = x1[12 + c];
    return a;
}

inline half2 vnax_deq(uint w, int c, half2 s, half2 b) {
    return fma(as_type<half2>(((w >> (4 * c)) & 0x000F000Fu) | 0x64006400u) - half2(1024.0h), s, b);
}

// Fragment B (rows n0+fm and n0+fm+8 of W) of chunk c.
struct VNaxW {
    uint2 w0, w1;
    half2 s0, b0, s1, b1;
    inline nax_hfrag frag(int c) const {
        const half2 l0 = vnax_deq(w0.x, c, s0, b0), h0 = vnax_deq(w0.y, c, s0, b0);
        const half2 l1 = vnax_deq(w1.x, c, s1, b1), h1 = vnax_deq(w1.y, c, s1, b1);
        return nax_hfrag(l0.x, l0.y, h0.x, h0.y, l1.x, l1.y, h1.x, h1.y);
    }
};

// MXFP4 (LAB_MXFP4): the same 16 contiguous 4-bit codes per row in the same chunk order, but each code is an e2m1 float
// and each group of 32 has one e8m0 scale and no bias (lab_fp4x2 and lab_e8m0x, qmm_nax_moe.metal).
struct VNaxWF4 {
    uint2 w0, w1;
    float s0, s1;   // 2^(e − 127) · 2¹⁴
    inline nax_hfrag frag(int c) const {
        const half2 l0 = lab_fp4x2(w0.x >> (4 * c), s0), h0 = lab_fp4x2(w0.y >> (4 * c), s0);
        const half2 l1 = lab_fp4x2(w1.x >> (4 * c), s1), h1 = lab_fp4x2(w1.y >> (4 * c), s1);
        return nax_hfrag(l0.x, l0.y, h0.x, h0.y, l1.x, l1.y, h1.x, h1.y);
    }
};

// Rows n and n+8 of an MXFP4 W (16 codes in kk, all in the group kk / 32) with their scales; gpr = K / 32.
inline VNaxWF4 vnax_loadw_fp4(const device uint* w, const device uchar* s, uint n, uint wpr, uint gpr, uint kk) {
    VNaxWF4 r;
    r.w0 = *(const device uint2*)(w + size_t(n) * wpr + kk / 8);
    r.w1 = *(const device uint2*)(w + size_t(n + 8) * wpr + kk / 8);
    r.s0 = lab_e8m0x(s[size_t(n) * gpr + kk / 32]);
    r.s1 = lab_e8m0x(s[size_t(n + 8) * gpr + kk / 32]);
    return r;
}

// Weight fragment type of the expert kernels: affine (VNaxW) or MXFP4 (VNaxWF4).
template <bool FP4> struct VNaxWOf { using T = VNaxW; };
template <> struct VNaxWOf<true> { using T = VNaxWF4; };

// Loads rows n and n+8 of W (block kb, 16 values in kk) with their scale and bias. Groups of G: with 64,
// the group is block kb; with 32, kk / 32 (the 16 contiguous values always fall in a single group). gpr = K / G.
template <int G = 64>
inline VNaxW vnax_loadw(const device uint* w, const device bfloat* s, const device bfloat* b, uint n, uint wpr,
                        uint gpr, uint kk, uint kb) {
    static_assert(G == 64 || G == 32, "groups of 64 or 32");
    if constexpr (G != 64) kb = kk / G;
    VNaxW r;
    r.w0 = *(const device uint2*)(w + size_t(n) * wpr + kk / 8);
    r.w1 = *(const device uint2*)(w + size_t(n + 8) * wpr + kk / 8);
    r.s0 = half2(half(float(s[size_t(n) * gpr + kb]))); r.b0 = half2(half(float(b[size_t(n) * gpr + kb])));
    r.s1 = half2(half(float(s[size_t(n + 8) * gpr + kb]))); r.b1 = half2(half(float(b[size_t(n + 8) * gpr + kb])));
    return r;
}

// Split-K reduction of the round qmv on NAX: each simdgroup leaves its NF accumulator fragments in red
// ([SGS][16][NC]), then the threadgroup sums them in simdgroup order and writes rows < p.M of y[:, n0, +NC) (or adds
// to them with flags bit 0). One body for vnax_qmv_body (every weight format) and vnax_qmvt (xc_tern.metal).
template <int NF, int SGS, typename OT>
inline void vnax_reduce_store(thread const nax_ffrag (&acc)[NF], threadgroup float* red, device OT* y,
                              constant VParams& p, uint n0, uint tid, ushort sg, uint fm, uint fn) {
    constexpr int NC = 16 * NF;
    VUNROLL for (int f = 0; f < NF; f++) VUNROLL for (int e = 0; e < 8; e++)
        red[(sg * 16 + fm + 8 * (e / 4)) * NC + 16 * f + fn + e % 4] = acc[f][e];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < p.M * NC; i += SGS * 32) {
        const uint m = i / NC, n = i % NC;
        float v = 0;
        VUNROLL for (int q = 0; q < SGS; q++) v += red[(q * 16 + m) * NC + n];
        device OT* o = y + size_t(m) * p.ldy + n0 + n;
        *o = (p.flags & 1) ? OT(float(*o) + v) : OT(v);
    }
}

// Y[M <= 16, n0, +16·NF) (+)= X · Wᵀ for any weight format. The threadgroup carries 16·NF columns; its SGS simdgroups
// share the K blocks of 64 (split-K) and are summed by vnax_reduce_store. The weight loader WL (the format) gives, with
// `load(n, kk, kb)`, the fragment source (`frag(c)`) of rows n and n+8 for the 16 contiguous values at kk of block kb.
template <int NF, int SGS, class WL, typename OT>
inline void vnax_qmv_body(const device half* x, thread const WL& wl, device OT* y, constant VParams& p, uint n0,
                          threadgroup float* red, uint tid, ushort sg, ushort lane) {
    const short2 co = nax_coord(lane);
    const uint fm = co.x, fn = co.y;
    const uint r0 = min(fm, p.M - 1), r1 = min(fm + 8, p.M - 1);
    nax_ffrag acc[NF];
    VUNROLL for (int f = 0; f < NF; f++) acc[f] = 0;
    for (uint kb = sg; kb < p.K / 64; kb += SGS) {
        const uint kk = kb * 64 + (fn / 4) * 16;
        half x0[16], x1[16];
        vnax_load16(x0, x + size_t(r0) * p.ldx + kk);
        vnax_load16(x1, x + size_t(r1) * p.ldx + kk);
        typename WL::T wt[NF];
        VUNROLL for (int f = 0; f < NF; f++) wt[f] = wl.load(n0 + 16 * f + fm, kk, kb);
        nax_hfrag a[4];
        VUNROLL for (int c = 0; c < 4; c++) a[c] = vnax_afrag(x0, x1, c);
        if constexpr (NF == 1) {
            // 16 columns per threadgroup (twice the threadgroups: with N = 2048 and NF = 2 only 64 for 12 cores);
            // half of the matmul2d is thrown away, but here the weight read dominates.
            nax_ffrag dummy = 0;
            VUNROLL for (int c = 0; c < 4; c++) nax_mma2(acc[0], dummy, a[c], wt[0].frag(c), wt[0].frag(c));
        } else {
            VUNROLL for (int f = 0; f < NF; f += 2)
                VUNROLL for (int c = 0; c < 4; c++) nax_mma2(acc[f], acc[f + 1], a[c], wt[f].frag(c), wt[f + 1].frag(c));
        }
    }
    vnax_reduce_store<NF, SGS>(acc, red, y, p, n0, tid, sg, fm, fn);
}

// The round qmv of the formats with bf16 scale and bias (4 bits with groups of 64 or 32, 8 bits, and 6 bits in
// yw_lfm.metal): WL::W is the type of the weight buffer and WL::make(w, s, b, K) the loader.
template <int NF, int SGS, class WL>
[[kernel]] void vnax_qmv(const device half*              x [[buffer(0)]],
                         const device typename WL::W*    w [[buffer(1)]],
                         const device bfloat*            s [[buffer(2)]],
                         const device bfloat*            b [[buffer(3)]],
                         device float*                   y [[buffer(4)]],
                         constant VParams&               p [[buffer(5)]],
                         uint tg [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]],
                         ushort sg [[simdgroup_index_in_threadgroup]],
                         ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[SGS * 16 * 16 * NF];
    vnax_qmv_body<NF, SGS>(x, WL::make(w, s, b, p.K), y, p, tg * 16 * NF, red, tid, sg, lane);
}

// Instance NAME of vnax_qmv with the loader WL.
#define VNQ_INST_WL(NAME, NF, SGS, WL)                                                                             \
  template [[host_name(NAME)]] [[kernel]] void vnax_qmv<NF, SGS, WL>(                                              \
      const device half*, const device WL::W*, const device bfloat*, const device bfloat*, device float*,          \
      constant VParams&, uint, uint, ushort, ushort);

// 4 bits, groups of G (vnax_loadw).
template <int G> struct VLoad4 {
    using W = uint;
    using T = VNaxW;
    const device uint* w;
    const device bfloat *s, *b;
    uint wpr, gpr;
    static inline VLoad4 make(const device uint* w, const device bfloat* s, const device bfloat* b, uint K) {
        return VLoad4{w, s, b, K / 8, K / G};
    }
    inline T load(uint n, uint kk, uint kb) const { return vnax_loadw<G>(w, s, b, n, wpr, gpr, kk, kb); }
};

#define VNQ_INST(NF, SGS) VNQ_INST_WL("vnax_qmv_nf" #NF "_sg" #SGS, NF, SGS, VLoad4<64>)
VNQ_INST(1, 32)
#if LAB_G32
#define VNQ_INST_G32(NF, SGS) VNQ_INST_WL("vnax_qmv_g32_nf" #NF "_sg" #SGS, NF, SGS, VLoad4<32>)
VNQ_INST_G32(1, 32)
#endif

// The same with 8-bit weights (the MoE router): the thread carries the 16 contiguous bytes [16·(fn/4), +16) of its row
// (a uint4) and in chunk c it uses the values {c, c+4, 8+c, 12+c}, i.e. byte c of each of the 4 words.
struct VNaxW8 {
    uint4 w0, w1;
    half s0, b0, s1, b1;
    static inline half dq(uint w, int c, half s, half b) { return fma(half((w >> (8 * c)) & 0xFFu), s, b); }
    inline nax_hfrag frag(int c) const {
        return nax_hfrag(dq(w0.x, c, s0, b0), dq(w0.y, c, s0, b0), dq(w0.z, c, s0, b0), dq(w0.w, c, s0, b0),
                         dq(w1.x, c, s1, b1), dq(w1.y, c, s1, b1), dq(w1.z, c, s1, b1), dq(w1.w, c, s1, b1));
    }
};

struct VLoad8 {
    using W = uint;
    using T = VNaxW8;
    const device uint* w;
    const device bfloat *s, *b;
    uint wpr, gpr;
    static inline VLoad8 make(const device uint* w, const device bfloat* s, const device bfloat* b, uint K) {
        return VLoad8{w, s, b, K / 4, K / 64};
    }
    inline T load(uint n, uint kk, uint kb) const {
        T r;
        r.w0 = *(const device uint4*)(w + size_t(n) * wpr + kk / 4);
        r.w1 = *(const device uint4*)(w + size_t(n + 8) * wpr + kk / 4);
        r.s0 = half(float(s[size_t(n) * gpr + kb])); r.b0 = half(float(b[size_t(n) * gpr + kb]));
        r.s1 = half(float(s[size_t(n + 8) * gpr + kb])); r.b1 = half(float(b[size_t(n + 8) * gpr + kb]));
        return r;
    }
};
#define VNQ8_INST(NF, SGS) VNQ_INST_WL("vnax_qmv8_nf" #NF "_sg" #SGS, NF, SGS, VLoad8)
VNQ8_INST(2, 16)

// The same with unquantized bf16 weights (the MoE router that Nemotron-H and GLM ship in bf16): the thread
// carries the 16 contiguous values [16·(fn/4), +16) of its row converted to half (exact for |w| in half's normal range;
// below 2^-14 it rounds to a half subnormal, an absolute error under 3e-8) and in chunk c uses {c, c+4, 8+c, 12+c}.
struct VNaxWbf {
    half w0[16], w1[16];
    inline nax_hfrag frag(int c) const {
        return nax_hfrag(w0[c], w0[c + 4], w0[8 + c], w0[12 + c], w1[c], w1[c + 4], w1[8 + c], w1[12 + c]);
    }
};
inline void vnax_load16bf(thread half (&d)[16], const device bfloat* p) {
    const device bfloat4* q = (const device bfloat4*)p;
    VUNROLL for (int i = 0; i < 4; i++) {
        const float4 v = float4(q[i]);
        d[4 * i] = half(v.x); d[4 * i + 1] = half(v.y); d[4 * i + 2] = half(v.z); d[4 * i + 3] = half(v.w);
    }
}

struct VLoadBf16 {
    using T = VNaxWbf;
    const device bfloat* w;
    uint K;
    inline T load(uint n, uint kk, uint) const {
        T r;
        vnax_load16bf(r.w0, w + size_t(n) * K + kk);
        vnax_load16bf(r.w1, w + size_t(n + 8) * K + kk);
        return r;
    }
};

// No scale or bias buffers (2 and 3 unbound), so its own entry point.
template <int NF, int SGS>
[[kernel]] void vnax_qmvbf16(const device half*   x [[buffer(0)]],
                             const device bfloat* w [[buffer(1)]],
                             device float*        y [[buffer(4)]],
                             constant VParams&    p [[buffer(5)]],
                             uint tg [[threadgroup_position_in_grid]],
                             uint tid [[thread_index_in_threadgroup]],
                             ushort sg [[simdgroup_index_in_threadgroup]],
                             ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[SGS * 16 * 16 * NF];
    vnax_qmv_body<NF, SGS>(x, VLoadBf16{w, p.K}, y, p, tg * 16 * NF, red, tid, sg, lane);
}
#define VNQBF_INST(NF, SGS)                                                                                        \
  template [[host_name("vnax_qmvbf16_nf" #NF "_sg" #SGS)]] [[kernel]] void vnax_qmvbf16<NF, SGS>(                  \
      const device half*, const device bfloat*, device float*, constant VParams&, uint, uint, ushort, ushort);
VNQBF_INST(2, 16)

// MoE gate+up+SwiGLU for verification on NAX. Indirect grid (tiles, U + 1): one expert of the list and
// 16·NF columns of the intermediate; the expert's rows (<= 16) go in one fragment; split-K across simdgroups.
// Shared expert with its own intermediate (p.sinter): the grid covers the larger of the two (the extra tiles of the
// other do nothing) and its rows of h go after those of the pairs, with stride sinter (with sinter = inter, the
// same positions as before).
// ACT: activation (lab_glu; `vnax_moe_gateup_gelu_*` instances with LAB_FFN_GELU, `_oai_` with LAB_FFN_OAI, `_relu2_`
// with LAB_FFN_RELU2: no gate, its buffers are bound but not read).
// LAB_MOE_EBIAS: the routed expert's gate and up biases (fp32 [E, inter], buffers 18 and 19) before the activation.
// FP4 (`_mxfp4` instances, LAB_MXFP4; G = 32): MXFP4 experts, their uint8 e8m0 scales bound where the bf16 scales go and
// the bias buffers unread; no shared expert.
template <int NF, int SGS, int G = 64, int ACT = 0, bool FP4 = false>
[[kernel]] void vnax_moe_gateup(const device half*   x       [[buffer(0)]],
                                const device uint*   egw     [[buffer(1)]],
                                const device bfloat* egs     [[buffer(2)]],
                                const device bfloat* egb     [[buffer(3)]],
                                const device uint*   euw     [[buffer(4)]],
                                const device bfloat* eus     [[buffer(5)]],
                                const device bfloat* eub     [[buffer(6)]],
                                const device uint*   shgw    [[buffer(7)]],
                                const device bfloat* shgs    [[buffer(8)]],
                                const device bfloat* shgb    [[buffer(9)]],
                                const device uint*   shuw    [[buffer(10)]],
                                const device bfloat* shus    [[buffer(11)]],
                                const device bfloat* shub    [[buffer(12)]],
                                const device uint4*  entries [[buffer(13)]],
                                const device uint2*  pairs   [[buffer(14)]],
                                device half*         h       [[buffer(15)]],   // [pairs, inter]
                                constant VMoEParams& p       [[buffer(16)]],
                                const device uint*   smap    [[buffer(17)]],   // expert -> slot
                                const device float*  ebg     [[buffer(18)]],   // gate and up biases (LAB_MOE_EBIAS)
                                const device float*  ebu     [[buffer(19)]],
                                uint2 tg [[threadgroup_position_in_grid]],
                                uint tid [[thread_index_in_threadgroup]],
                                ushort sg [[simdgroup_index_in_threadgroup]],
                                ushort lane [[thread_index_in_simdgroup]]) {
    constexpr int NC = 16 * NF;
    threadgroup float red[SGS * 16 * 2 * NC];
    const uint4 en = entries[tg.y];
    const uint e = en.x, start = en.y, cnt = en.z;
    const device uint *wg = shgw, *wu = shuw;
    const device bfloat *sg_ = shgs, *bg = shgb, *su = shus, *bu = shub;
    const device uchar *sgq = nullptr, *suq = nullptr;   // MXFP4 scales (FP4)
    if (e < p.E) {
        const size_t se = smap[e];
        const size_t ow = se * p.inter * (p.hidden / 8), os = se * p.inter * (p.hidden / G);
        wg = egw + ow; wu = euw + ow; sg_ = egs + os; bg = egb + os; su = eus + os; bu = eub + os;
        if constexpr (FP4) { sgq = (const device uchar*)egs + os; suq = (const device uchar*)eus + os; }
    }
    const short2 co = nax_coord(lane);
    const uint fm = co.x, fn = co.y;
    const uint n0 = tg.x * NC, wpr = p.hidden / 8, gpr = p.hidden / G;
    const uint I = e < p.E ? p.inter : p.sinter;   // width of this expert's output
    if (n0 >= I) return;
    const uint xr0 = pairs[start + min(fm, cnt - 1)].x, xr1 = pairs[start + min(fm + 8, cnt - 1)].x;
    nax_ffrag ag[NF], au[NF];
    VUNROLL for (int f = 0; f < NF; f++) { ag[f] = 0; au[f] = 0; }
    for (uint kb = sg; kb < p.hidden / 64; kb += SGS) {
        const uint kk = kb * 64 + (fn / 4) * 16;
        half x0[16], x1[16];
        vnax_load16(x0, x + size_t(xr0) * p.hidden + kk);
        vnax_load16(x1, x + size_t(xr1) * p.hidden + kk);
        typename VNaxWOf<FP4>::T wgt[NF], wut[NF];
        VUNROLL for (int f = 0; f < NF; f++) {
            if constexpr (FP4) {
                wgt[f] = vnax_loadw_fp4(wg, sgq, n0 + 16 * f + fm, wpr, gpr, kk);
                wut[f] = vnax_loadw_fp4(wu, suq, n0 + 16 * f + fm, wpr, gpr, kk);
            } else {
                if constexpr (ACT != 3) wgt[f] = vnax_loadw<G>(wg, sg_, bg, n0 + 16 * f + fm, wpr, gpr, kk, kb);
                wut[f] = vnax_loadw<G>(wu, su, bu, n0 + 16 * f + fm, wpr, gpr, kk, kb);
            }
        }
        nax_hfrag a[4];
        VUNROLL for (int c = 0; c < 4; c++) a[c] = vnax_afrag(x0, x1, c);
        if constexpr (ACT == 3 && NF % 2 == 0) {
            // relu² without gate: up only, two column fragments per matmul.
            VUNROLL for (int f = 0; f < NF; f += 2)
                VUNROLL for (int c = 0; c < 4; c++) nax_mma2(au[f], au[f + 1], a[c], wut[f].frag(c), wut[f + 1].frag(c));
        } else if constexpr (ACT == 3) {
            // relu² with one fragment: the second half of the matmul goes to ag, which nobody reads.
            VUNROLL for (int f = 0; f < NF; f++)
                VUNROLL for (int c = 0; c < 4; c++) nax_mma2(ag[f], au[f], a[c], wut[f].frag(c), wut[f].frag(c));
        } else {
            VUNROLL for (int f = 0; f < NF; f++)
                VUNROLL for (int c = 0; c < 4; c++) nax_mma2(ag[f], au[f], a[c], wgt[f].frag(c), wut[f].frag(c));
        }
    }
    VUNROLL for (int f = 0; f < NF; f++) VUNROLL for (int q = 0; q < 8; q++) {
        const uint m = fm + 8 * (q / 4), n = 16 * f + fn + q % 4;
        red[(sg * 16 + m) * 2 * NC + n] = ag[f][q];
        red[(sg * 16 + m) * 2 * NC + NC + n] = au[f][q];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < cnt * NC; i += SGS * 32) {
        const uint m = i / NC, n = i % NC;
        float g = 0, u = 0;
        VUNROLL for (int q = 0; q < SGS; q++) { g += red[(q * 16 + m) * 2 * NC + n]; u += red[(q * 16 + m) * 2 * NC + NC + n]; }
#if LAB_MOE_EBIAS
        if (e < p.E) { g += ebg[size_t(e) * p.inter + n0 + n]; u += ebu[size_t(e) * p.inter + n0 + n]; }
#endif
        h[size_t(start) * p.inter + size_t(m) * I + n0 + n] = half(lab_glu<ACT>(g, u));
    }
}

#define VNGU_INST(NF, SGS)                                                                                         \
  template [[host_name("vnax_moe_gateup_nf" #NF "_sg" #SGS)]] [[kernel]] void vnax_moe_gateup<NF, SGS>(            \
      const device half*, const device uint*, const device bfloat*, const device bfloat*, const device uint*,      \
      const device bfloat*, const device bfloat*, const device uint*, const device bfloat*, const device bfloat*,  \
      const device uint*, const device bfloat*, const device bfloat*, const device uint4*, const device uint2*,    \
      device half*, constant VMoEParams&, const device uint*, const device float*, const device float*, uint2, uint,   \
      ushort, ushort);
VNGU_INST(2, 8) VNGU_INST(1, 16)
#if LAB_FFN_GELU
#define VNGU_INST_GELU(NF, SGS)                                                                                    \
  template [[host_name("vnax_moe_gateup_gelu_nf" #NF "_sg" #SGS)]] [[kernel]] void                                 \
  vnax_moe_gateup<NF, SGS, 64, 1>(                                                                               \
      const device half*, const device uint*, const device bfloat*, const device bfloat*, const device uint*,      \
      const device bfloat*, const device bfloat*, const device uint*, const device bfloat*, const device bfloat*,  \
      const device uint*, const device bfloat*, const device bfloat*, const device uint4*, const device uint2*,    \
      device half*, constant VMoEParams&, const device uint*, const device float*, const device float*, uint2, uint,   \
      ushort, ushort);
VNGU_INST_GELU(2, 8) VNGU_INST_GELU(1, 16)
#endif
#if LAB_FFN_OAI
#define VNGU_INST_OAI(NF, SGS)                                                                                     \
  template [[host_name("vnax_moe_gateup_oai_nf" #NF "_sg" #SGS)]] [[kernel]] void                                  \
  vnax_moe_gateup<NF, SGS, 64, 2>(                                                                               \
      const device half*, const device uint*, const device bfloat*, const device bfloat*, const device uint*,      \
      const device bfloat*, const device bfloat*, const device uint*, const device bfloat*, const device bfloat*,  \
      const device uint*, const device bfloat*, const device bfloat*, const device uint4*, const device uint2*,    \
      device half*, constant VMoEParams&, const device uint*, const device float*, const device float*, uint2, uint,   \
      ushort, ushort);
VNGU_INST_OAI(2, 8) VNGU_INST_OAI(1, 16)
#endif
#if LAB_FFN_RELU2
#define VNGU_INST_RELU2(NF, SGS)                                                                                   \
  template [[host_name("vnax_moe_gateup_relu2_nf" #NF "_sg" #SGS)]] [[kernel]] void                                \
  vnax_moe_gateup<NF, SGS, 64, 3>(                                                                               \
      const device half*, const device uint*, const device bfloat*, const device bfloat*, const device uint*,      \
      const device bfloat*, const device bfloat*, const device uint*, const device bfloat*, const device bfloat*,  \
      const device uint*, const device bfloat*, const device bfloat*, const device uint4*, const device uint2*,    \
      device half*, constant VMoEParams&, const device uint*, const device float*, const device float*, uint2, uint,   \
      ushort, ushort);
VNGU_INST_RELU2(2, 8) VNGU_INST_RELU2(1, 16)
#endif
#if LAB_MXFP4
#define VNGU_INST_FP4(ACTN, ACT, NF, SGS)                                                                         \
  template [[host_name("vnax_moe_gateup" ACTN "_mxfp4_nf" #NF "_sg" #SGS)]] [[kernel]] void                       \
  vnax_moe_gateup<NF, SGS, 32, ACT, true>(                                                                       \
      const device half*, const device uint*, const device bfloat*, const device bfloat*, const device uint*,      \
      const device bfloat*, const device bfloat*, const device uint*, const device bfloat*, const device bfloat*,  \
      const device uint*, const device bfloat*, const device bfloat*, const device uint4*, const device uint2*,    \
      device half*, constant VMoEParams&, const device uint*, const device float*, const device float*, uint2, uint,   \
      ushort, ushort);
#define VNGU_INST_FP4_ALL(ACTN, ACT)                                                                              \
  VNGU_INST_FP4(ACTN, ACT, 2, 8) VNGU_INST_FP4(ACTN, ACT, 1, 16)
VNGU_INST_FP4_ALL("", 0)
#if LAB_FFN_OAI
VNGU_INST_FP4_ALL("_oai", 2)
#endif
#endif

// down per (expert, 16·NF columns) with split-K across SGS simdgroups (the gate+up shape, which hits the memory
// floor), without combining: each pair leaves weight · (h · Wdᵀ) in part[slot][row][column] (slot = expert rank in
// the row's top-8, 8 for the shared one) and vmoe_combine9 sums the 9 slots of each row in that order. The value of
// each pair does not depend on the other rows of the block (B = 1 and B = 8 give the same: greedy speculation must
// match plain decoding bit for bit). The previous kernel (removed) combined inside: each threadgroup
// walked all the experts (about 30 × 8 K blocks in a chain per simdgroup) and stayed at 80 % of the floor.
// LAB_MOE_EBIAS: the routed expert's down bias (fp32 [E, hidden], buffer 14) is added before the weight.
// FP4 (`_mxfp4` instances, LAB_MXFP4; G = 32): MXFP4 experts as in vnax_moe_gateup.
template <int NF, int SGS, int G = 64, bool FP4 = false>
[[kernel, max_total_threads_per_threadgroup(32 * SGS)]] void vnax_moe_down_split(
    const device half*   h       [[buffer(0)]],
    const device uint*   edw     [[buffer(1)]],
    const device bfloat* eds     [[buffer(2)]],
    const device bfloat* edb     [[buffer(3)]],
    const device uint*   shdw    [[buffer(4)]],
    const device bfloat* shds    [[buffer(5)]],
    const device bfloat* shdb    [[buffer(6)]],
    const device uint4*  entries [[buffer(7)]],
    const device uint2*  pairs   [[buffer(8)]],
    device float*        part    [[buffer(10)]],   // [9, 16, hidden]
    constant VMoEParams& p       [[buffer(11)]],
    const device uint*   inds    [[buffer(12)]],
    const device uint*   smap    [[buffer(13)]],   // expert -> slot
    const device float*  ebd     [[buffer(14)]],   // down biases (LAB_MOE_EBIAS)
    uint2 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    constexpr int NC = 16 * NF;
    threadgroup float red[SGS * 16 * NC];
    const uint4 en = entries[tg.y];
    const uint e = en.x, start = en.y, cnt = en.z;
    const uint I = e < p.E ? p.inter : p.sinter;   // K of this expert (the shared one, with its intermediate)
    const uint wpr = I / 8, gpr = I / G, nkb = I / 64;   // groups per row and blocks of 64 of K
    const device uint* w = shdw;
    const device bfloat *s = shds, *b = shdb;
    const device uchar* sq = nullptr;   // MXFP4 scales (FP4)
    if (e < p.E) {
        const size_t se = smap[e];
        w = edw + se * p.hidden * wpr;
        s = eds + se * p.hidden * gpr;
        b = edb + se * p.hidden * gpr;
        if constexpr (FP4) sq = (const device uchar*)eds + se * p.hidden * gpr;
    }
    const short2 co = nax_coord(lane);
    const uint fm = co.x, fn = co.y;
    const uint n0 = tg.x * NC;
    const device half* hb = h + size_t(start) * p.inter;   // rows of h of this expert, with stride I
    const uint hr0 = min(fm, cnt - 1), hr1 = min(fm + 8, cnt - 1);
    nax_ffrag c[NF];
    VUNROLL for (int f = 0; f < NF; f++) c[f] = 0;
    for (uint kb = sg; kb < nkb; kb += SGS) {
        const uint kk = kb * 64 + (fn / 4) * 16;
        half x0[16], x1[16];
        vnax_load16(x0, hb + size_t(hr0) * I + kk);
        vnax_load16(x1, hb + size_t(hr1) * I + kk);
        typename VNaxWOf<FP4>::T wt[NF];
        VUNROLL for (int f = 0; f < NF; f++) {
            if constexpr (FP4) wt[f] = vnax_loadw_fp4(w, sq, n0 + 16 * f + fm, wpr, gpr, kk);
            else wt[f] = vnax_loadw<G>(w, s, b, n0 + 16 * f + fm, wpr, gpr, kk, kb);
        }
        nax_hfrag a[4];
        VUNROLL for (int q = 0; q < 4; q++) a[q] = vnax_afrag(x0, x1, q);
        if constexpr (NF == 1) {
            nax_ffrag dummy = 0;
            VUNROLL for (int q = 0; q < 4; q++) nax_mma2(c[0], dummy, a[q], wt[0].frag(q), wt[0].frag(q));
        } else {
            VUNROLL for (int f = 0; f < NF; f += 2)
                VUNROLL for (int q = 0; q < 4; q++) nax_mma2(c[f], c[f + 1], a[q], wt[f].frag(q), wt[f + 1].frag(q));
        }
    }
    VUNROLL for (int f = 0; f < NF; f++) VUNROLL for (int q = 0; q < 8; q++)
        red[(sg * 16 + fm + 8 * (q / 4)) * NC + 16 * f + fn + q % 4] = c[f][q];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < cnt * NC; i += SGS * 32) {
        const uint m = i / NC, n = i % NC;
        float v = 0;
        VUNROLL for (int q = 0; q < SGS; q++) v += red[(q * 16 + m) * NC + n];
        const uint2 pr = pairs[start + m];
        const uint topk = (p.flags >> 16) & 0xffu ? (p.flags >> 16) & 0xffu : 8u;
        uint slot = topk;
        if (e < p.E) {
            slot = 0;
            for (uint k = 0; k < topk; k++) slot += inds[pr.x * topk + k] < e ? 1 : 0;
#if LAB_MOE_EBIAS
            v += ebd[size_t(e) * p.hidden + n0 + n];
#endif
        }
        part[(slot * 16 + pr.x) * p.hidden + n0 + n] = as_type<float>(pr.y) * v;
    }
}

#define VNDS_INST(NF, SGS)                                                                                         \
  template [[host_name("vnax_moe_down_split_nf" #NF "_sg" #SGS)]] [[kernel]] void vnax_moe_down_split<NF, SGS>(    \
      const device half*, const device uint*, const device bfloat*, const device bfloat*, const device uint*,      \
      const device bfloat*, const device bfloat*, const device uint4*, const device uint2*, device float*,        \
      constant VMoEParams&, const device uint*, const device uint*, const device float*, uint2, uint, ushort, ushort);
VNDS_INST(2, 8) VNDS_INST(1, 16)
#if LAB_MXFP4
#define VNDS_INST_FP4(NF, SGS)                                                                                     \
  template [[host_name("vnax_moe_down_split_mxfp4_nf" #NF "_sg" #SGS)]] [[kernel]] void                           \
  vnax_moe_down_split<NF, SGS, 32, true>(                                                                        \
      const device half*, const device uint*, const device bfloat*, const device bfloat*, const device uint*,      \
      const device bfloat*, const device bfloat*, const device uint4*, const device uint2*, device float*,        \
      constant VMoEParams&, const device uint*, const device uint*, const device float*, uint2, uint, ushort, ushort);
VNDS_INST_FP4(2, 8) VNDS_INST_FP4(1, 16)
#endif

// out[r, c] (+)= Σ_{slot 0..8} part[slot][r][c], in that order (4 columns per thread).
[[kernel]] void vmoe_combine9(const device float* part [[buffer(0)]],
                              device float*       out  [[buffer(1)]],
                              constant VMoEParams& p   [[buffer(2)]],
                              const device uint*  rst  [[buffer(3)]],   // flags bit 1: only the first st[6] rows
                              uint gid [[thread_position_in_grid]]) {
    const uint r = gid / (p.hidden / 4), c4 = gid % (p.hidden / 4);
    if (r >= ((p.flags & 2) ? min(p.B, rst[6]) : p.B)) return;
    float4 v = 0;
    const uint ns = p.flags >> 24 ? p.flags >> 24 : 9u;
    for (uint q = 0; q < ns; q++) v += ((const device float4*)(part + (q * 16 + r) * p.hidden))[c4];
    device float4* o = (device float4*)(out + size_t(r) * p.hidden) + c4;
    *o = (p.flags & 1) ? *o + v : v;
}
