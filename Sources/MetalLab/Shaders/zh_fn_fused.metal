// Flash-Next: exact fusions of the hyper-connections. Each kernel does the same
// operations, in the same per-element order, as the kernel sequence it replaces (zf_flashnext.metal), so
// the results are bit-exact; only launches and memory passes are saved.
//   - fn_hc_inj_gnorm(_rr): injection of the previous block (x += br·inj) + the norm of the next HC
//     (fn_hc_inject + fn_gnorm / fn_gnorm_rr);
//   - fn_hc_upmix_v: up projection of the round (vnax_qmv8 with reordered rows) with the input activation
//     (fn_hc_act) in the prologue and the mix (fn_hc_mix) in the epilogue, and br to zero (fn_zero) if no fp32 needed;
//   - fn_moe_combine_inj: MoE combine + injection (combine with the shared expert + fn_hc_inject), skipping br.
// Concatenated after zf_flashnext.metal (FN_*, fn_sigmoid, FnNormParams) and verify_nax.metal (vnax_*, VNaxW8).

// Injection + RMSNorm of the 4 streams. Grid (4, rows), 256 threads. With inj == nullptr (flags bit 0 = 0) no inject.
// Output as fn_gnorm (or fp32 + half oh) or as fn_gnorm_rr (rr + half oh) depending on bit 1 of flags.
struct FnInjNormParams { uint D; float eps; uint ldx; uint ldo; uint flags; };   // bit 0: inject; bit 1: rr

[[kernel]] void fn_hc_inj_gnorm(device float*              x   [[buffer(0)]],   // streams [rows, 4·D]
                                const device float*        w   [[buffer(1)]],
                                device float*              o   [[buffer(2)]],   // nrm [rows, 4·D] or rr [rows, 4]
                                constant FnInjNormParams&  p   [[buffer(3)]],
                                device half*               oh  [[buffer(4)]],
                                const device float*        br  [[buffer(5)]],   // [rows, D]
                                const device float*        inj [[buffer(6)]],   // [rows, 4]
                                uint2 tg [[threadgroup_position_in_grid]],
                                uint tid [[thread_index_in_threadgroup]],
                                ushort sg [[simdgroup_index_in_threadgroup]],
                                ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[8];
    device float* xr = x + size_t(tg.y) * p.ldx + tg.x * p.D;
    float ss = 0;
    if (p.flags & 1) {
        const device float* brr = br + size_t(tg.y) * FN_D;
        const float iv = inj[tg.y * FN_S + tg.x];
        for (uint i = tid; i < p.D; i += 256) {
            const float v = xr[i] + brr[i] * iv;
            xr[i] = v;
            ss += v * v;
        }
    } else {
        for (uint i = tid; i < p.D; i += 256) ss += xr[i] * xr[i];
    }
    ss = simd_sum(ss);
    if (lane == 0) part[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0;
    VUNROLL for (uint i = 0; i < 8; i++) tot += part[i];
    const float r = rsqrt(tot / float(p.D) + p.eps);
    device half* hrow = oh + size_t(tg.y) * p.ldo + tg.x * p.D;
    if (p.flags & 2) {
        if (tid == 0) o[tg.y * 4 + tg.x] = r;
        for (uint i = tid; i < p.D; i += 256) hrow[i] = half(xr[i] * r * w[tg.x * p.D + i]);
    } else {
        device float* orow = o + size_t(tg.y) * p.ldo + tg.x * p.D;
        for (uint i = tid; i < p.D; i += 256) {
            const float v = xr[i] * r * w[tg.x * p.D + i];
            orow[i] = v;
            hrow[i] = half(v);
        }
    }
}

// up of the round (8 bits, reordered rows d·4 + s) with the activation and the mix. Like vnax_qmv8<NF, SGS> with
// x = half(SiLU(dp / 4)) read from dp [rows, ldd] (fn_hc_act) and, instead of writing g, the mix of fn_hc_mix
// (xh, br). Threadgroup 0 also writes inj (fn_hc_act). Grid (N / (16·NF)), 32·SGS threads.
struct FnUpMixVParams { uint M; uint N; uint K; uint ldd; uint flags; };
// flags: bit 0 writes br = mix (fp32); bit 1 sets br to 0; bit 2 writes inj (the HC injects)

template <int NF, int SGS>
[[kernel]] void fn_hc_upmix_v(const device float*  dp  [[buffer(0)]],
                              const device uint*   w   [[buffer(1)]],
                              const device bfloat* s   [[buffer(2)]],
                              const device bfloat* b   [[buffer(3)]],
                              const device float*  nrm [[buffer(4)]],   // [rows, 4·D] (fn_gnorm)
                              device half*         oh  [[buffer(5)]],   // xh [rows, D]
                              device float*        of  [[buffer(6)]],   // br [rows, D]
                              device float*        inj [[buffer(7)]],   // [rows, 4]
                              constant FnUpMixVParams& p [[buffer(8)]],
                              uint tg [[threadgroup_position_in_grid]],
                              uint tid [[thread_index_in_threadgroup]],
                              ushort sg [[simdgroup_index_in_threadgroup]],
                              ushort lane [[thread_index_in_simdgroup]]) {
    constexpr int NC = 16 * NF;
    threadgroup float red[SGS * 16 * NC];
    const short2 co = nax_coord(lane);
    const uint fm = co.x, fn = co.y;
    const uint n0 = tg * NC, wpr = p.K / 4, gpr = p.K / 64;
    const uint r0 = min(fm, p.M - 1), r1 = min(fm + 8, p.M - 1);
    if (tg == 0 && (p.flags & 4)) {
        for (uint i = tid; i < p.M * FN_S; i += SGS * 32) {
            const float v = dp[size_t(i / FN_S) * p.ldd + p.K + i % FN_S] / float(FN_S);
            inj[i] = 2.0f * fn_sigmoid(v);
        }
    }
    nax_ffrag acc[NF];
    VUNROLL for (int f = 0; f < NF; f++) acc[f] = 0;
    for (uint kb = sg; kb < p.K / 64; kb += SGS) {
        const uint kk = kb * 64 + (fn / 4) * 16;
        half x0[16], x1[16];
        VUNROLL for (int i = 0; i < 16; i++) {
            const float v0 = dp[size_t(r0) * p.ldd + kk + i] / float(FN_S);
            const float v1 = dp[size_t(r1) * p.ldd + kk + i] / float(FN_S);
            x0[i] = half(v0 * fn_sigmoid(v0));
            x1[i] = half(v1 * fn_sigmoid(v1));
        }
        VNaxW8 wt[NF];
        VUNROLL for (int f = 0; f < NF; f++) {
            const uint n = n0 + 16 * f + fm;
            wt[f].w0 = *(const device uint4*)(w + size_t(n) * wpr + kk / 4);
            wt[f].w1 = *(const device uint4*)(w + size_t(n + 8) * wpr + kk / 4);
            wt[f].s0 = half(float(s[size_t(n) * gpr + kb])); wt[f].b0 = half(float(b[size_t(n) * gpr + kb]));
            wt[f].s1 = half(float(s[size_t(n + 8) * gpr + kb])); wt[f].b1 = half(float(b[size_t(n + 8) * gpr + kb]));
        }
        nax_hfrag a[4];
        VUNROLL for (int c = 0; c < 4; c++) a[c] = vnax_afrag(x0, x1, c);
        VUNROLL for (int f = 0; f < NF; f += 2)
            VUNROLL for (int c = 0; c < 4; c++) nax_mma2(acc[f], acc[f + 1], a[c], wt[f].frag(c), wt[f + 1].frag(c));
    }
    VUNROLL for (int f = 0; f < NF; f++) VUNROLL for (int e = 0; e < 8; e++)
        red[(sg * 16 + fm + 8 * (e / 4)) * NC + 16 * f + fn + e % 4] = acc[f][e];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // g of the tile (like vnax_qmv8: sum over the simdgroups in order) into red[0 …] and the mix per (row, d).
    threadgroup float gt[16 * NC];
    for (uint i = tid; i < p.M * NC; i += SGS * 32) {
        const uint m = i / NC, n = i % NC;
        float v = 0;
        VUNROLL for (int q = 0; q < SGS; q++) v += red[(q * 16 + m) * NC + n];
        gt[m * NC + n] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    constexpr uint ND = NC / FN_S;
    const uint d0 = n0 / FN_S;
    for (uint i = tid; i < p.M * ND; i += SGS * 32) {
        const uint m = i / ND, dl = i % ND, d = d0 + dl;
        const device float* nr = nrm + size_t(m) * FN_W;
        float a = 0;
        VUNROLL for (uint q = 0; q < FN_S; q++) a += nr[q * FN_D + d] * fn_sigmoid(gt[m * NC + dl * FN_S + q]);
        a = a / float(FN_S);
        oh[size_t(m) * FN_D + d] = half(a);
        if (p.flags & 1) of[size_t(m) * FN_D + d] = a;
        else if (p.flags & 2) of[size_t(m) * FN_D + d] = 0;
    }
}

#define FN_UPMIX_INST(NF, SGS)                                                                                     \
  template [[host_name("fn_hc_upmix_v_nf" #NF "_sg" #SGS)]] [[kernel]] void fn_hc_upmix_v<NF, SGS>(                \
      const device float*, const device uint*, const device bfloat*, const device bfloat*, const device float*,  \
      device half*, device float*, device float*, constant FnUpMixVParams&, uint, uint, ushort, ushort);
FN_UPMIX_INST(2, 5)

// MoE combine + injection: S[r, s, n] += (Σ_k w·y + σ(gate)·sh)·inj[r, s]. One thread per (row, n).
[[kernel]] void fn_moe_combine_inj(const device float* y     [[buffer(0)]],
                                   const device uint*  pos   [[buffer(1)]],
                                   const device float* wts   [[buffer(2)]],
                                   const device float* sh    [[buffer(3)]],
                                   const device float* sgate [[buffer(4)]],
                                   device float*       S     [[buffer(5)]],
                                   const device float* inj   [[buffer(6)]],
                                   constant uint&      T     [[buffer(7)]],
                                   uint gid [[thread_position_in_grid]]) {
    const uint r = gid / FN_D, n = gid % FN_D;
    if (r >= T) return;
    float acc = 0;
    VUNROLL for (uint k = 0; k < FN_K; k++) acc += wts[r * FN_K + k] * y[size_t(pos[r * FN_K + k]) * FN_D + n];
    acc += sgate[r] * sh[size_t(r) * FN_D + n];
    device float* xr = S + size_t(r) * FN_W;
    VUNROLL for (uint q = 0; q < FN_S; q++) xr[q * FN_D + n] += acc * inj[r * FN_S + q];
}

// vnax_qmv8 with 16 columns per threadgroup (NF = 1: twice the threadgroups for the HC down projection,
// N = 384; half of the matmul2d is thrown away). Same summation order as vnax_qmv8<2, SGS> per column. Also VQMV's
// 8-bit branch for every family (half the threadgroup memory, faster in every shape measured).
template <int SGS>
[[kernel]] void fn_qmv8_nf1(const device half*   x [[buffer(0)]],
                            const device uint*   w [[buffer(1)]],
                            const device bfloat* s [[buffer(2)]],
                            const device bfloat* b [[buffer(3)]],
                            device float*        y [[buffer(4)]],
                            constant VParams&    p [[buffer(5)]],
                            uint tg [[threadgroup_position_in_grid]],
                            uint tid [[thread_index_in_threadgroup]],
                            ushort sg [[simdgroup_index_in_threadgroup]],
                            ushort lane [[thread_index_in_simdgroup]]) {
    constexpr int NC = 16;
    threadgroup float red[SGS * 16 * NC];
    const short2 co = nax_coord(lane);
    const uint fm = co.x, fn = co.y;
    const uint n0 = tg * NC, wpr = p.K / 4, gpr = p.K / 64;
    const uint r0 = min(fm, p.M - 1), r1 = min(fm + 8, p.M - 1);
    nax_ffrag acc = 0, dummy = 0;
    for (uint kb = sg; kb < p.K / 64; kb += SGS) {
        const uint kk = kb * 64 + (fn / 4) * 16;
        half x0[16], x1[16];
        vnax_load16(x0, x + size_t(r0) * p.ldx + kk);
        vnax_load16(x1, x + size_t(r1) * p.ldx + kk);
        VNaxW8 wt;
        const uint n = n0 + fm;
        wt.w0 = *(const device uint4*)(w + size_t(n) * wpr + kk / 4);
        wt.w1 = *(const device uint4*)(w + size_t(n + 8) * wpr + kk / 4);
        wt.s0 = half(float(s[size_t(n) * gpr + kb])); wt.b0 = half(float(b[size_t(n) * gpr + kb]));
        wt.s1 = half(float(s[size_t(n + 8) * gpr + kb])); wt.b1 = half(float(b[size_t(n + 8) * gpr + kb]));
        nax_hfrag a[4];
        VUNROLL for (int c = 0; c < 4; c++) a[c] = vnax_afrag(x0, x1, c);
        VUNROLL for (int c = 0; c < 4; c++) nax_mma2(acc, dummy, a[c], wt.frag(c), wt.frag(c));
    }
    VUNROLL for (int e = 0; e < 8; e++) red[(sg * 16 + fm + 8 * (e / 4)) * NC + fn + e % 4] = acc[e];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < p.M * NC; i += SGS * 32) {
        const uint m = i / NC, n = i % NC;
        float v = 0;
        VUNROLL for (int q = 0; q < SGS; q++) v += red[(q * 16 + m) * NC + n];
        device float* o = y + size_t(m) * p.ldy + n0 + n;
        *o = (p.flags & 1) ? *o + v : v;
    }
}
template [[host_name("fn_qmv8_nf1_sg16")]] [[kernel]] void fn_qmv8_nf1<16>(const device half*, const device uint*,
    const device bfloat*, const device bfloat*, device float*, constant VParams&, uint, uint, ushort, ushort);
