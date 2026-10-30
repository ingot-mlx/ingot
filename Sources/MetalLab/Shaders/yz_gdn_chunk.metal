// GatedDeltaNet chunked prefill, 64-token chunks (WY/UT form, like chunk_gated_delta_rule of
// flash-linear-attention, MIT) with products on NAX. Concatenated after verify.metal (GDN_*) and qmm_nax.metal (nax_*).
//
// Per head v (state S: 128 dv × 128 dk, fp32) and chunk of C = 64 tokens, with γ_i = Π_{j≤i} g_j inside the chunk:
//   S_i = g_i S_{i-1} + β_i (v_i − g_i S_{i-1} k_i) k_iᵀ        (the gdn_recur recurrence)
// is written S_i = γ_i S_0 + Σ_{j≤i} (γ_i/γ_j) u_j k_jᵀ with
//   (I + A) U = diag(β) (V − diag(γ) K S_0ᵀ),   A_ij = β_i (γ_i/γ_j) k_i·k_j  (j < i)
// so, with T = (I + A)⁻¹, W = T diag(βγ) K and Ũ = T diag(β) V (independent of S_0):
//   U = Ũ − W S_0ᵀ
//   O = diag(γ) Q S_0ᵀ + P U,          P_ij = (γ_i/γ_j) q_i·k_j  (j ≤ i)
//   S_C = γ_C S_0 + Uᵀ diag(γ_C/γ) K
// gdn_cprep computes in parallel for all chunks W, Ũᵀ, Q̃ = diag(γ) Q, P and K̃ᵀ = (diag(γ_C/γ) K)ᵀ; gdn_cscan
// walks the chunks in order with S in threadgroup memory (fp32) and does only 4 products of 64×128 per chunk.
// All the γ_i/γ_j factors with j ≤ i are computed as exp(lγ_i − lγ_j) ≤ 1.
// Precision: NAX operands in half with fp32 accumulation; T is split into high half + low half (almost fp32) and the
// state is accumulated in fp32. Versus the fp32 recurrence (2048 rows): rel L2 4.7e-4 in the output and
// 4.2e-4 in the state (splitting S_0 into high + low too does not change the error and costs 18% more scan time).
// Measured (2048 rows, 32 heads): sequential recurrence 4.86 ms; chunked 1.36 (prep) + 0.93 (scan).
// Pitfalls: (1) a matmul2d product takes ~1000 cycles to be ready, so with few simdgroups (the
// scan has 32 heads × 8 tiles) each sum must be split across several chains; (2) substituting T with a whole
// column per thread (64 registers) lowers occupancy: 4 threads per column with 16 rows each.

constant constexpr uint GC = 64;   // tokens per chunk

struct GdnChunkParams {
    uint rows;    // real rows of the prefill chunk
    uint nch;     // chunks of 64
};

// 16×16 of float → half fragment (rows c.x, c.x + 8; columns c.y..c.y+3).
METAL_FUNC void nax_load_f(thread nax_hfrag& d, const device float* src, uint ld, short2 c) {
    const float4 v0 = *(const device float4*)(src + size_t(c.x) * ld + c.y);
    const float4 v1 = *(const device float4*)(src + size_t(c.x + 8) * ld + c.y);
    d = nax_hfrag(half4(v0), half4(v1));
}
METAL_FUNC void nax_load_ff(thread nax_ffrag& d, const device float* src, uint ld, short2 c) {
    const float4 v0 = *(const device float4*)(src + size_t(c.x) * ld + c.y);
    const float4 v1 = *(const device float4*)(src + size_t(c.x + 8) * ld + c.y);
    d = nax_ffrag(v0, v1);
}
// High and low half parts of an fp32 fragment in threadgroup memory (x ≈ high + low with ~22 bits).
METAL_FUNC void nax_load_split(thread nax_hfrag& hi, thread nax_hfrag& lo, const threadgroup float* src, uint ld, short2 c) {
    const float4 v0 = *(const threadgroup float4*)(src + c.x * ld + c.y);
    const float4 v1 = *(const threadgroup float4*)(src + (c.x + 8) * ld + c.y);
    const half4 h0 = half4(v0), h1 = half4(v1);
    hi = nax_hfrag(h0, h1);
    lo = nax_hfrag(half4(v0 - float4(h0)), half4(v1 - float4(h1)));
}

template <typename IT> METAL_FUNC float4 ld4(const device IT* p) { return float4(*(const device vec<IT, 4>*)p); }

// Grid (chunks, 32 v heads), 256 threads (8 simdgroups). IT: type of q, k, v (float or half).
template <typename IT>
[[kernel]] void gdn_cprep(const device IT*    q    [[buffer(0)]],   // [r, 2048]
                          const device IT*    k    [[buffer(1)]],   // [r, 2048]
                          const device IT*    v    [[buffer(2)]],   // [r, 4096]
                          const device float* g    [[buffer(3)]],   // [r, 32]
                          const device float* beta [[buffer(4)]],   // [r, 32]
                          device half*        Wb   [[buffer(5)]],   // [32, nch, 64, 128]   W
                          device float*       Ub   [[buffer(6)]],   // [32, nch, 128, 64]   Ũᵀ
                          device half*        Qb   [[buffer(7)]],   // [32, nch, 64, 128]   Q̃
                          device half*        Pb   [[buffer(8)]],   // [32, nch, 64, 64]    P
                          device half*        Kb   [[buffer(9)]],   // [32, nch, 128, 64]   K̃ᵀ
                          device float*       gC   [[buffer(10)]],  // [32, nch]
                          constant GdnChunkParams& p [[buffer(11)]],
                          uint2 tg [[threadgroup_position_in_grid]],
                          uint tid [[thread_index_in_threadgroup]],
                          ushort sg [[simdgroup_index_in_threadgroup]],
                          ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float As[GC * GC];     // A and then T (in place)
    threadgroup float lgs[GC], bs[GC], carry[1];
    constexpr uint TL = 24;            // transposed blocks [128][16] (+8 padding)
    threadgroup half kt[128 * TL];     // block of K and then of β·V
    threadgroup half* vt = kt;
    const uint ch = tg.x, hv = tg.y, hk = hv / GDN_R, t0 = ch * GC;
    const uint last = p.rows - 1;
    auto row = [&](uint i) { return min(t0 + i, last); };   // padding rows: data of the last row, β = 0
    // 1. lγ (prefix sum of log g) and β.
    if (tid < GC) {
        const bool ok = t0 + tid < p.rows;
        const float lg = ok ? log(max(g[(t0 + tid) * GDN_HV + hv], 1e-30f)) : 0.0f;
        float s = simd_prefix_inclusive_sum(lg);
        if (tid == 31) carry[0] = s;
        bs[tid] = ok ? beta[(t0 + tid) * GDN_HV + hv] : 0.0f;
        lgs[tid] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid >= 32 && tid < GC) lgs[tid] += carry[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const short2 co = nax_coord(lane);
    const uint fm = co.x, fn = co.y;
    const size_t hb = size_t(hv) * p.nch + ch;
    // 2. K Kᵀ and Q Kᵀ (64×64): simdgroup s → rows 16·(s/2), columns 32·(s%2) .. +32.
    {
        const uint rm = sg / 2, cp = sg % 2;
        nax_ffrag kk[2], qk[2];
        kk[0] = 0; kk[1] = 0; qk[0] = 0; qk[1] = 0;
        const uint ra0 = row(16 * rm + fm), ra1 = row(16 * rm + fm + 8);
        const uint rb0 = row(32 * cp + fm), rb1 = row(32 * cp + fm + 8), rb2 = row(32 * cp + 16 + fm), rb3 = row(32 * cp + 24 + fm);
        NAX_UNROLL for (int kb = 0; kb < 8; kb++) {
            const uint d = hk * 128 + 16 * kb + fn;
            const nax_hfrag ak = nax_hfrag(half4(ld4(k + size_t(ra0) * GDN_QK + d)), half4(ld4(k + size_t(ra1) * GDN_QK + d)));
            const nax_hfrag aq = nax_hfrag(half4(ld4(q + size_t(ra0) * GDN_QK + d)), half4(ld4(q + size_t(ra1) * GDN_QK + d)));
            const nax_hfrag b0 = nax_hfrag(half4(ld4(k + size_t(rb0) * GDN_QK + d)), half4(ld4(k + size_t(rb1) * GDN_QK + d)));
            const nax_hfrag b1 = nax_hfrag(half4(ld4(k + size_t(rb2) * GDN_QK + d)), half4(ld4(k + size_t(rb3) * GDN_QK + d)));
            nax_mma2(kk[0], kk[1], ak, b0, b1);
            nax_mma2(qk[0], qk[1], aq, b0, b1);
        }
        NAX_UNROLL for (int f = 0; f < 2; f++) {
            const nax_ffrag kkf = kk[f], qkf = qk[f];
            NAX_UNROLL for (int h = 0; h < 2; h++) {
                const uint i = 16 * rm + fm + 8 * h, j0 = 32 * cp + 16 * f + fn;
                float4 av, pv;
                NAX_UNROLL for (int e = 0; e < 4; e++) {
                    const uint j = j0 + e;
                    const float dec = j <= i ? exp(lgs[i] - lgs[j]) : 0.0f;
                    av[e] = j < i ? bs[i] * dec * kkf[4 * h + e] : 0.0f;
                    pv[e] = dec * qkf[4 * h + e];
                }
                *(threadgroup float4*)(As + i * GC + j0) = av;
                *(device half4*)(Pb + hb * GC * GC + i * GC + j0) = half4(pv);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // 3. T = (I + A)⁻¹ by forward substitution, T_i = e_i − Σ_{j<i} A_ij T_j: 4 threads per column c of T, each
    //    with 16 rows in registers; each step adds its part and reduces it within the quad (two shuffles), no barriers.
    const float lC = lgs[GC - 1];
    {
        const uint c = tid / 4, qq = tid % 4;
        float t[16];
        NAX_UNROLL for (uint jj = 0; jj < 16; jj++) t[jj] = 16 * qq + jj == c ? 1.0f : 0.0f;
        NAX_UNROLL for (uint i = 1; i < GC; i++) {
            float part = 0;
            if (16 * qq < i) {
                NAX_UNROLL for (uint jj = 0; jj < 16; jj += 4) {
                    const float4 a = *(const threadgroup float4*)(As + i * GC + 16 * qq + jj);
                    part += a.x * t[jj] + a.y * t[jj + 1] + a.z * t[jj + 2] + a.w * t[jj + 3];
                }
            }
            part += simd_shuffle_xor(part, 1);
            part += simd_shuffle_xor(part, 2);
            if (qq == i / 16 && i > c) t[i % 16] = -part;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);   // all finished reading A
        NAX_UNROLL for (uint jj = 0; jj < 16; jj++) As[(16 * qq + jj) * GC + c] = t[jj];
    }
    // 5. Q̃ = diag(γ) Q (64×128) and K̃ᵀ = (diag(γ_C/γ) K)ᵀ (128×64).
    for (uint idx = tid; idx < GC * 32; idx += 256) {
        const uint i = idx / 32, c4 = (idx % 32) * 4;
        const float4 qv = ld4(q + size_t(row(i)) * GDN_QK + hk * 128 + c4);
        *(device half4*)(Qb + hb * GC * 128 + i * 128 + c4) = half4(qv * exp(lgs[i]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // 4. W = T · diag(βγ) K (64×128), Ũᵀ = (T · diag(β) V)ᵀ (128×64) and K̃ᵀ, in blocks of 16 tokens j: each block of
    //    K and of β·V is copied transposed ([dim][j], half) to threadgroup memory, so operands are read as fragments.
    //    W: simdgroup s → rows i 16·(s/2), dk 64·(s%2) .. +64. Ũᵀ: simdgroup s → rows dv 16·s.
    {
        const uint rm = sg / 2, dh = sg % 2;
        nax_ffrag w[4], u[4];
        NAX_UNROLL for (int f = 0; f < 4; f++) { w[f] = 0; u[f] = 0; }
        NAX_UNROLL for (int kb = 0; kb < 4; kb++) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint idx = tid; idx < 16 * 32; idx += 256) {          // 16 tokens × 32 float4 of dk
                const uint jj = idx / 32, c4 = (idx % 32) * 4, rj = row(16 * kb + jj);
                const float4 kv = ld4(k + size_t(rj) * GDN_QK + hk * 128 + c4);
                NAX_UNROLL for (int e = 0; e < 4; e++) kt[(c4 + e) * TL + jj] = half(kv[e]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (kb <= int(rm)) {
                nax_hfrag th, tl;
                nax_load_split(th, tl, As + (16 * rm) * GC + 16 * kb, GC, co);
                half4 sc;
                NAX_UNROLL for (int e = 0; e < 4; e++) { const uint j = 16 * kb + fn + e; sc[e] = half(bs[j] * exp(lgs[j])); }
                NAX_UNROLL for (int f = 0; f < 4; f += 2) {
                    nax_hfrag b0, b1;
                    nax_load(b0, kt + (64 * dh + 16 * f) * TL, TL, co);
                    nax_load(b1, kt + (64 * dh + 16 * f + 16) * TL, TL, co);
                    b0 *= nax_hfrag(sc, sc); b1 *= nax_hfrag(sc, sc);
                    nax_mma2(w[f], w[f + 1], th, b0, b1);
                    nax_mma2(w[f], w[f + 1], tl, b0, b1);
                }
            }
            for (uint idx = tid; idx < 128 * 4; idx += 256) {   // K̃ᵀ of the block: 128 dk × 4 half4
                const uint dk = idx / 4, j4 = (idx % 4) * 4;
                half4 kk = *(const threadgroup half4*)(kt + dk * TL + j4);
                float4 f4;
                NAX_UNROLL for (int e = 0; e < 4; e++) f4[e] = exp(lC - lgs[16 * kb + j4 + e]) * float(kk[e]);
                *(device half4*)(Kb + hb * 128 * GC + dk * GC + 16 * kb + j4) = half4(f4);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint idx = tid; idx < 16 * 32; idx += 256) {          // β·V of the block: 16 tokens × 32 float4 of dv
                const uint jj = idx / 32, c4 = (idx % 32) * 4, rj = row(16 * kb + jj);
                const float4 vv = ld4(v + size_t(rj) * GDN_V + hv * 128 + c4) * bs[16 * kb + jj];
                NAX_UNROLL for (int e = 0; e < 4; e++) vt[(c4 + e) * TL + jj] = half(vv[e]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            {
                nax_hfrag a;
                nax_load(a, vt + (16 * sg) * TL, TL, co);
                NAX_UNROLL for (int f = 0; f < 4; f += 2) {
                    if (f + 1 < kb) continue;                              // T[i, j] = 0 if block of i < block of j
                    nax_hfrag b0h, b0l, b1h, b1l;
                    nax_load_split(b0h, b0l, As + (16 * f) * GC + 16 * kb, GC, co);
                    nax_load_split(b1h, b1l, As + (16 * f + 16) * GC + 16 * kb, GC, co);
                    nax_mma2(u[f], u[f + 1], a, b0h, b1h);
                    nax_mma2(u[f], u[f + 1], a, b0l, b1l);
                }
            }
        }
        device half* wo = Wb + hb * GC * 128;
        NAX_UNROLL for (int f = 0; f < 4; f++) NAX_UNROLL for (int i = 0; i < 2; i++) {
            const uint r = 16 * rm + fm + 8 * i;
            *(device half4*)(wo + r * 128 + 64 * dh + 16 * f + fn) = half4(w[f][4 * i], w[f][4 * i + 1], w[f][4 * i + 2], w[f][4 * i + 3]);
        }
        device float* uo = Ub + hb * 128 * GC;
        NAX_UNROLL for (int f = 0; f < 4; f++) NAX_UNROLL for (int i = 0; i < 2; i++) {
            const uint r = 16 * sg + fm + 8 * i;
            *(device float4*)(uo + r * GC + 16 * f + fn) = float4(u[f][4 * i], u[f][4 * i + 1], u[f][4 * i + 2], u[f][4 * i + 3]);
        }
    }
    if (tid == 0) gC[hb] = exp(lC);
}

#define GDN_CPREP_INST(IT, NAME)                                                                                   \
  template [[host_name(NAME)]] [[kernel]] void gdn_cprep<IT>(const device IT*, const device IT*, const device IT*,  \
      const device float*, const device float*, device half*, device float*, device half*, device half*,          \
      device half*, device float*, constant GdnChunkParams&, uint2, uint, ushort, ushort);
GDN_CPREP_INST(float, "gdn_cprep")
GDN_CPREP_INST(half, "gdn_cprep_h")

// Scan with short chains: a matmul2d product takes ~1000 cycles to be ready (microbenchmark: with 256
// simdgroups and a single chain per simdgroup, 5.3 TFLOPS; with 4 chains, 17.7), so the scan needs to have
// many independent chains. Each tile of 16 dv rows is handled by 2 simdgroups (half of the tokens for U and O,
// half of dk to update S), each sum goes in 2 accumulators and S lives in threadgroup memory.
// Threadgroup = 2 tiles (4 simdgroups); grid 32 heads × 4.
// CAP (in-pass checkpoint): additionally, in the chunk of each capture q (cc.c[q]), the state after
// row cc.j[q] of the chunk, S' = γ' S_0 + Uᵀ K̃' with K̃' and γ' from gdn_capprep (K̃' = 0 in the later rows), to
// cap[q]. These are the same operations, in the same order, as the last chunk of a pass ending at that row (there
// U = 0 in the padding rows; here K̃' = 0): the state comes out bit-exact with that pass's.
// Up to 3 captures (points (a) and (b) and the conversation cut of the prefix cache).
struct GdnCapParams { uint n; uint c[4]; uint j[4]; };

template <bool CAP>
[[kernel]] void gdn_cscan(const device float* sin  [[buffer(0)]],
                           device float*       sout [[buffer(1)]],
                           const device half*  Wb   [[buffer(2)]],
                           const device float* Ub   [[buffer(3)]],
                           const device half*  Qb   [[buffer(4)]],
                           const device half*  Pb   [[buffer(5)]],
                           const device half*  Kb   [[buffer(6)]],
                           const device float* gC   [[buffer(7)]],
                           device float*       y    [[buffer(8)]],
                           constant GdnChunkParams& p [[buffer(9)]],
                           constant GdnCapParams& cc  [[buffer(10)]],
                           const device half*  Kcap [[buffer(11)]],   // [4, 32, 128, 64]
                           const device float* gcap [[buffer(12)]],   // [4, 32]
                           device float*       cap0 [[buffer(13)]],   // [32, 128, 128]
                           device float*       cap1 [[buffer(14)]],
                           device float*       cap2 [[buffer(15)]],
                           device float*       cap3 [[buffer(16)]],
                           uint tg [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]],
                           ushort sg [[simdgroup_index_in_threadgroup]],
                           ushort lane [[thread_index_in_simdgroup]]) {
    constexpr uint SL = 132, XL = 72;
    threadgroup float ss[2][16 * SL];
    threadgroup half xs[2][16 * XL];
    const uint hv = tg / 4, t = sg / 2, hs = sg % 2;
    const uint dv0 = 16 * ((tg % 4) * 2 + t);
    const short2 co = nax_coord(lane);
    const uint fm = co.x, fn = co.y;
    threadgroup float* S = ss[t];
    threadgroup half* X = xs[t];
    for (uint i = tid; i < 2 * 16 * 32; i += 128) {
        const uint tt = i / 512, r = (i / 32) % 16, c4 = (i % 32) * 4;
        const uint dvr = 16 * ((tg % 4) * 2 + tt) + r;
        *(threadgroup float4*)(ss[tt] + r * SL + c4) = *(const device float4*)(sin + (size_t(hv) * 128 + dvr) * 128 + c4);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto lds = [&](int f) {
        return nax_ffrag(*(const threadgroup float4*)(S + fm * SL + 16 * f + fn), *(const threadgroup float4*)(S + (fm + 8) * SL + 16 * f + fn));
    };
    const uint tk0 = 32 * hs;   // tokens of U and O for this simdgroup
    for (uint ch = 0; ch < p.nch; ch++) {
        const size_t hb = size_t(hv) * p.nch + ch;
        const device half* W = Wb + hb * GC * 128;
        const device half* Qt = Qb + hb * GC * 128;
        const device half* P = Pb + hb * GC * GC;
        const device half* Kt = Kb + hb * 128 * GC;
        // Uᵀ (16 dv × 32 tokens) = Ũᵀ − S_0 Wᵀ, in 2 chains (even/odd f).
        nax_ffrag ua[2], ub[2];
        NAX_UNROLL for (int c = 0; c < 2; c++) { ua[c] = 0; ub[c] = 0; }
        NAX_UNROLL for (int f = 0; f < 8; f++) {
            const nax_ffrag sf = lds(f);
            const nax_hfrag sh = nax_hfrag(sf);
            nax_hfrag b0, b1;
            nax_load(b0, W + tk0 * 128 + 16 * f, 128, co); nax_load(b1, W + (tk0 + 16) * 128 + 16 * f, 128, co);
            nax_mma2(ua[f % 2], ub[f % 2], sh, b0, b1);
        }
        const device float* Ut = Ub + hb * 128 * GC + dv0 * GC + tk0;
        {
            nax_ffrag u0, u1;
            nax_load_ff(u0, Ut, GC, co); nax_load_ff(u1, Ut + 16, GC, co);
            u0 -= ua[0] + ua[1];
            u1 -= ub[0] + ub[1];
            const nax_hfrag h0 = nax_hfrag(u0), h1 = nax_hfrag(u1);
            *(threadgroup half4*)(X + fm * XL + tk0 + fn) = half4(h0[0], h0[1], h0[2], h0[3]);
            *(threadgroup half4*)(X + (fm + 8) * XL + tk0 + fn) = half4(h0[4], h0[5], h0[6], h0[7]);
            *(threadgroup half4*)(X + fm * XL + tk0 + 16 + fn) = half4(h1[0], h1[1], h1[2], h1[3]);
            *(threadgroup half4*)(X + (fm + 8) * XL + tk0 + 16 + fn) = half4(h1[4], h1[5], h1[6], h1[7]);
        }
        // Oᵀ (16 dv × 32 tokens) = S_0 Q̃ᵀ in 2 chains.
        NAX_UNROLL for (int c = 0; c < 2; c++) { ua[c] = 0; ub[c] = 0; }
        NAX_UNROLL for (int f = 0; f < 8; f++) {
            const nax_ffrag sf = lds(f);
            const nax_hfrag sh = nax_hfrag(sf);
            nax_hfrag b0, b1;
            nax_load(b0, Qt + tk0 * 128 + 16 * f, 128, co); nax_load(b1, Qt + (tk0 + 16) * 128 + 16 * f, 128, co);
            nax_mma2(ua[f % 2], ub[f % 2], sh, b0, b1);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);   // X complete; nobody reads S again in this chunk
        // Oᵀ += Uᵀ Pᵀ (token blocks j ≤ i).
        nax_ffrag o0 = ua[0] + ua[1], o1 = ub[0] + ub[1];
        NAX_UNROLL for (int kb = 0; kb < 4; kb++) {
            if (kb > 2 * int(hs) + 1) continue;
            nax_hfrag a, b0, b1;
            nax_load(a, X + 16 * kb, XL, co);
            nax_load(b0, P + tk0 * GC + 16 * kb, GC, co); nax_load(b1, P + (tk0 + 16) * GC + 16 * kb, GC, co);
            nax_mma2(kb % 2 ? ua[0] : o0, kb % 2 ? ub[0] : o1, a, b0, b1);
            if (kb == 0) { ua[0] = 0; ub[0] = 0; }
        }
        // (the split into two chains: even blocks in o, odd in ua[0]/ub[0])
        if (true) { o0 += ua[0]; o1 += ub[0]; }
        NAX_UNROLL for (int f = 0; f < 2; f++) NAX_UNROLL for (int e = 0; e < 8; e++) {
            const uint tok = ch * GC + tk0 + 16 * f + fn + e % 4;
            if (tok < p.rows) y[size_t(tok) * GDN_V + hv * 128 + dv0 + fm + 8 * (e / 4)] = (f == 0 ? o0 : o1)[e];
        }
        // S[:, dk of this half] = γ_C S + Uᵀ K̃ (2 column pairs, 2 chains each).
        const float gc = gC[hb];
        NAX_UNROLL for (int f = 0; f < 4; f += 2) {
            const int fk = 4 * hs + f;
            if (CAP) {
                for (uint q = 0; q < cc.n; q++) {
                    if (cc.c[q] != ch) continue;
                    const device half* Kq = Kcap + (size_t(q) * GDN_HV + hv) * 128 * GC;
                    const float gq = gcap[q * GDN_HV + hv];
                    nax_ffrag c0 = 0, c1 = 0, d0 = 0, d1 = 0;
                    NAX_UNROLL for (int kb = 0; kb < 4; kb++) {
                        nax_hfrag a, b0, b1;
                        nax_load(a, X + 16 * kb, XL, co);
                        nax_load(b0, Kq + (16 * fk) * GC + 16 * kb, GC, co);
                        nax_load(b1, Kq + (16 * fk + 16) * GC + 16 * kb, GC, co);
                        if (kb % 2) nax_mma2(d0, d1, a, b0, b1); else nax_mma2(c0, c1, a, b0, b1);
                    }
                    const nax_ffrag n0 = lds(fk) * gq + (c0 + d0), n1 = lds(fk + 1) * gq + (c1 + d1);
                    device float* o = (q == 0 ? cap0 : q == 1 ? cap1 : q == 2 ? cap2 : cap3) + (size_t(hv) * 128 + dv0) * 128;
                    *(device float4*)(o + fm * 128 + 16 * fk + fn) = float4(n0[0], n0[1], n0[2], n0[3]);
                    *(device float4*)(o + (fm + 8) * 128 + 16 * fk + fn) = float4(n0[4], n0[5], n0[6], n0[7]);
                    *(device float4*)(o + fm * 128 + 16 * fk + 16 + fn) = float4(n1[0], n1[1], n1[2], n1[3]);
                    *(device float4*)(o + (fm + 8) * 128 + 16 * fk + 16 + fn) = float4(n1[4], n1[5], n1[6], n1[7]);
                }
            }
            nax_ffrag c0 = 0, c1 = 0, d0 = 0, d1 = 0;
            NAX_UNROLL for (int kb = 0; kb < 4; kb++) {
                nax_hfrag a, b0, b1;
                nax_load(a, X + 16 * kb, XL, co);
                nax_load(b0, Kt + (16 * fk) * GC + 16 * kb, GC, co);
                nax_load(b1, Kt + (16 * fk + 16) * GC + 16 * kb, GC, co);
                if (kb % 2) nax_mma2(d0, d1, a, b0, b1); else nax_mma2(c0, c1, a, b0, b1);
            }
            const nax_ffrag n0 = lds(fk) * gc + (c0 + d0), n1 = lds(fk + 1) * gc + (c1 + d1);
            *(threadgroup float4*)(S + fm * SL + 16 * fk + fn) = float4(n0[0], n0[1], n0[2], n0[3]);
            *(threadgroup float4*)(S + (fm + 8) * SL + 16 * fk + fn) = float4(n0[4], n0[5], n0[6], n0[7]);
            *(threadgroup float4*)(S + fm * SL + 16 * fk + 16 + fn) = float4(n1[0], n1[1], n1[2], n1[3]);
            *(threadgroup float4*)(S + (fm + 8) * SL + 16 * fk + 16 + fn) = float4(n1[4], n1[5], n1[6], n1[7]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = tid; i < 2 * 16 * 32; i += 128) {
        const uint tt = i / 512, r = (i / 32) % 16, c4 = (i % 32) * 4;
        const uint dvr = 16 * ((tg % 4) * 2 + tt) + r;
        *(device float4*)(sout + (size_t(hv) * 128 + dvr) * 128 + c4) = *(const threadgroup float4*)(ss[tt] + r * SL + c4);
    }
}

template [[host_name("gdn_cscan")]] [[kernel]] void gdn_cscan<false>(const device float*, device float*, const device half*,
    const device float*, const device half*, const device half*, const device half*, const device float*, device float*,
    constant GdnChunkParams&, constant GdnCapParams&, const device half*, const device float*, device float*, device float*,
    device float*, device float*, uint, uint, ushort, ushort);
template [[host_name("gdn_cscan_cap")]] [[kernel]] void gdn_cscan<true>(const device float*, device float*, const device half*,
    const device float*, const device half*, const device half*, const device half*, const device float*, device float*,
    constant GdnChunkParams&, constant GdnCapParams&, const device half*, const device float*, device float*, device float*,
    device float*, device float*, uint, uint, ushort, ushort);

// K̃' and γ' of capture q for gdn_cscan_cap: those of gdn_cprep for the chunk of row cc.j[q] as if the pass
// ended at it (later rows with lg = 0; K̃' = 0 in them). lγ is computed with the same code as gdn_cprep.
// Grid (32 v heads, captures), 256 threads.
template <typename IT>
[[kernel]] void gdn_capprep(const device IT*    k    [[buffer(0)]],   // [r, 2048]
                            const device float* g    [[buffer(1)]],   // [r, 32]
                            device half*        Kcap [[buffer(2)]],   // [4, 32, 128, 64]
                            device float*       gcap [[buffer(3)]],   // [4, 32]
                            constant GdnCapParams& cc [[buffer(4)]],
                            uint2 tg [[threadgroup_position_in_grid]],
                            uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float lgs[GC], carry[1];
    const uint hv = tg.x, q = tg.y, hk = hv / GDN_R, t0 = cc.c[q] * GC, jj = cc.j[q];
    const uint rows = t0 + jj + 1, last = rows - 1;
    if (tid < GC) {
        const bool ok = t0 + tid < rows;
        const float lg = ok ? log(max(g[(t0 + tid) * GDN_HV + hv], 1e-30f)) : 0.0f;
        float s = simd_prefix_inclusive_sum(lg);
        if (tid == 31) carry[0] = s;
        lgs[tid] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid >= 32 && tid < GC) lgs[tid] += carry[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float lC = lgs[GC - 1];
    device half* Ko = Kcap + (size_t(q) * GDN_HV + hv) * 128 * GC;
    for (uint idx = tid; idx < 128 * GC; idx += 256) {
        const uint dk = idx / GC, i = idx % GC;
        const half kk = half(float(k[size_t(min(t0 + i, last)) * GDN_QK + hk * 128 + dk]));
        Ko[dk * GC + i] = i <= jj ? half(exp(lC - lgs[i]) * float(kk)) : half(0.0h);
    }
    if (tid == 0) gcap[q * GDN_HV + hv] = exp(lC);
}
template [[host_name("gdn_capprep")]] [[kernel]] void gdn_capprep<float>(const device float*, const device float*, device half*,
    device float*, constant GdnCapParams&, uint2, uint);
template [[host_name("gdn_capprep_h")]] [[kernel]] void gdn_capprep<half>(const device half*, const device float*, device half*,
    device float*, constant GdnCapParams&, uint2, uint);
