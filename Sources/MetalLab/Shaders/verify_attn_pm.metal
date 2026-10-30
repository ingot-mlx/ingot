// Decode/verification attention for MLA's latent head (576 = 64 RoPE + 512 latent dims; V = c, read from the latent
// part of K) with 8 simdgroups per group of 16 queries instead of pattn_i8's 4. Concatenated after verify_attn_nax.metal
// (AttnParams, nax_*, nax_load_i8, nax_mma2_kn).
//
// A step covers KB = 256 keys: simdgroup j takes keys [32·j, 32·j + 32) of the step for S (all 576 dimensions, two
// fragments of 16 keys; k_pe's 64 and c's 512 summed apart, each with its scale: ks and vs) and runs the softmax of
// its keys (maxima through threadgroup memory, P·vs in half); for P·V it takes 64 of the 512 latent dimensions (dims
// 64 + 64·j…, 4 fragments) over the 256 keys, read key-major from kc (H₅₁₂ c: vattn_reduce rotates the sum back).
// V has no RoPE part: dims 0…63 of opart are not written (vattn_reduce writes them as zero). Twice the simdgroups and twice
// the keys per step of pattn_i8 give the GPU more work in flight per chunk at the same number of threadgroups.
// Microbench (47 layers, attn + reduce, random int8 KV; swift test, measurement branch): with 1 row
// 3.75 -> 2.28 ms at 4K, 7.33 -> 4.28 at 8K, 10.45 -> 8.57 at 16K; with 16 rows 16.5 -> 11.5 at 4K, 57.0 -> 38.6 at
// 16K. Tried and slower: 8 simdgroups with 512-key steps (equal in the model), 16 with 512, and S split over the
// dimensions in 2 or 3 parts (8/12/16 simdgroups); one query group per threadgroup beats two.
// Exactness between rows: a row's result depends only on that row (fragment rows are independent, steps start at
// the chunk start, maxima and sums in fixed order), so B = 1 and B = 16 give the same bits with the same chunk C.
// The host only uses it with C a multiple of 256 (the latent chunks are 256, 512 or 1024): a step never crosses
// a chunk end.
#if LAB_AT_HD == 576
[[kernel]] void mattn_i8(const device half*  q     [[buffer(0)]],   // [1, AT_G·B, 576]
                         const device char*  kc    [[buffer(1)]],   // [1, ldkv, 576]
                         const device char*  vt    [[buffer(2)]],   // unused (V is read from kc)
                         const device float* ks    [[buffer(3)]],   // [1, ldkv] scale of k_pe
                         const device float* vs    [[buffer(4)]],   // [1, ldkv] scale of c (K's latent part and V)
                         constant AttnParams& p    [[buffer(7)]],
                         device float*       opart [[buffer(8)]],   // [1, nsplit, AT_G·B, 576]
                         device float2*      ml    [[buffer(9)]],   // [1, nsplit, AT_G·B]
                         const device uint*  st    [[buffer(10)]],  // chained decode: st[0] = position
                         uint3 tg [[threadgroup_position_in_grid]],
                         ushort sg [[simdgroup_index_in_threadgroup]],
                         ushort lane [[thread_index_in_simdgroup]]) {
    constexpr int NSG = 8, KB = 256, KS = KB / NSG, PVF = 4, PL = KB + 8;
    threadgroup half pb[16 * PL];
    threadgroup float redm[NSG][16];
    const uint Q = AT_G * p.B;
    const uint ctx = (p.dyn & AT_DYN_ST) ? st[0] : p.ctx;
    const uint j = sg;
    const uint row0 = tg.z * 16;
    const uint split = tg.x;
    const uint key0 = split * p.C;
    const uint kend = min(key0 + p.C, ctx + p.B);
    // from here on some rows do not see all the keys
    const uint kdiag = ctx + row0 / AT_G;
    const short2 co = nax_coord(lane);
    const uint fm = co.x, fn = co.y;
    const uint m0 = row0 + fm, m1 = m0 + 8;
    const bool val0 = m0 < Q, val1 = m1 < Q;
    const device half* q0 = q + size_t(min(m0, Q - 1)) * AT_HD + fn;
    const device half* q1 = q + size_t(min(m1, Q - 1)) * AT_HD + fn;
    const uint lim0 = ctx + m0 / AT_G, lim1 = ctx + m1 / AT_G;
    nax_ffrag o[PVF];
    NAX_UNROLL for (int f = 0; f < PVF; f++) o[f] = 0;
    float mr0 = -INFINITY, mr1 = -INFINITY, l0 = 0, l1 = 0;
    const uint d0 = 64 + 64 * j;
    for (uint k0 = key0; k0 < kend; k0 += KB) {
        const uint kj = k0 + KS * j;
        // k_pe (dims 0…63) and c (64…575) apart: each has its own scale.
        nax_ffrag e0 = 0, e1 = 0, s0 = 0, s1 = 0;
        NAX_UNROLL for (int c = 0; c < 4; c++) {
            const nax_hfrag qc = nax_hfrag(*(const device half4*)(q0 + 16 * c), *(const device half4*)(q1 + 16 * c));
            nax_hfrag kb0, kb1;
            nax_load_i8(kb0, kc + size_t(kj) * AT_HD + 16 * c, AT_HD, co);
            nax_load_i8(kb1, kc + size_t(kj + 16) * AT_HD + 16 * c, AT_HD, co);
            nax_mma2(e0, e1, qc, kb0, kb1);
        }
        PF_UNROLL2 for (int c = 4; c < AT_HD / 16; c++) {
            const nax_hfrag qc = nax_hfrag(*(const device half4*)(q0 + 16 * c), *(const device half4*)(q1 + 16 * c));
            nax_hfrag kb0, kb1;
            nax_load_i8(kb0, kc + size_t(kj) * AT_HD + 16 * c, AT_HD, co);
            nax_load_i8(kb1, kc + size_t(kj + 16) * AT_HD + 16 * c, AT_HD, co);
            nax_mma2(s0, s1, qc, kb0, kb1);
        }
        const bool diag = k0 + KB > min(kdiag, kend);
        float sv[16];
        float mx0 = -INFINITY, mx1 = -INFINITY;
        NAX_UNROLL for (int f = 0; f < 2; f++) {
            const uint kb = kj + 16 * f + fn;
            const float4 ksv = *(const device float4*)(ks + kb) * p.scale;
            const float4 csv = *(const device float4*)(vs + kb) * p.scale;
            NAX_UNROLL for (int e = 0; e < 8; e++) {
                float v = (f == 0 ? e0 : e1)[e] * ksv[e % 4] + (f == 0 ? s0 : s1)[e] * csv[e % 4];
                if (diag) {
                    const uint key = kb + e % 4;
                    v = key <= (e < 4 ? lim0 : lim1) && key < kend ? v : -INFINITY;
                }
                sv[f * 8 + e] = v;
                if (e < 4) mx0 = max(mx0, v); else mx1 = max(mx1, v);
            }
        }
        mx0 = max(mx0, simd_shuffle_xor(mx0, 1)); mx0 = max(mx0, simd_shuffle_xor(mx0, 8));
        mx1 = max(mx1, simd_shuffle_xor(mx1, 1)); mx1 = max(mx1, simd_shuffle_xor(mx1, 8));
        if ((lane & 9) == 0) { redm[j][fm] = mx0; redm[j][fm + 8] = mx1; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float t0 = redm[0][fm], t1 = redm[0][fm + 8];
        NAX_UNROLL for (int x = 1; x < NSG; x++) { t0 = max(t0, redm[x][fm]); t1 = max(t1, redm[x][fm + 8]); }
        const float mn0 = max(mr0, t0), mn1 = max(mr1, t1);
        const float a0 = mn0 == -INFINITY ? 1.0f : exp(mr0 - mn0), a1 = mn1 == -INFINITY ? 1.0f : exp(mr1 - mn1);
        float rs0 = 0, rs1 = 0;
        NAX_UNROLL for (int f = 0; f < 2; f++) {
            const float4 vsv = *(const device float4*)(vs + kj + 16 * f + fn);
            half4 h0, h1;
            NAX_UNROLL for (int e = 0; e < 4; e++) {
                const float p0 = mn0 == -INFINITY ? 0.0f : exp(sv[f * 8 + e] - mn0);
                const float p1 = mn1 == -INFINITY ? 0.0f : exp(sv[f * 8 + 4 + e] - mn1);
                rs0 += p0; rs1 += p1;
                h0[e] = half(p0 * vsv[e]); h1[e] = half(p1 * vsv[e]);
            }
            *(threadgroup half4*)(pb + fm * PL + KS * j + 16 * f + fn) = h0;
            *(threadgroup half4*)(pb + (fm + 8) * PL + KS * j + 16 * f + fn) = h1;
        }
        rs0 += simd_shuffle_xor(rs0, 1); rs0 += simd_shuffle_xor(rs0, 8);
        rs1 += simd_shuffle_xor(rs1, 1); rs1 += simd_shuffle_xor(rs1, 8);
        l0 = l0 * a0 + rs0; l1 = l1 * a1 + rs1;
        mr0 = mn0; mr1 = mn1;
        if (!simd_all(a0 == 1.0f && a1 == 1.0f)) {
            NAX_UNROLL for (int f = 0; f < PVF; f++) NAX_UNROLL for (int e = 0; e < 8; e++) o[f][e] *= (e < 4 ? a0 : a1);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        PF_UNROLL2 for (int kc2 = 0; kc2 < KB / 16; kc2++) {
            nax_hfrag pa;
            nax_load(pa, pb + 16 * kc2, PL, co);
            NAX_UNROLL for (int f = 0; f < PVF; f += 2) {
                // 16 keys × 32 latent dims of kc (key-major).
                const device char* kr = kc + size_t(k0 + 16 * kc2) * AT_HD + d0 + 16 * f;
                nax_hfrag vb0, vb1;
                nax_load_i8(vb0, kr, AT_HD, co);
                nax_load_i8(vb1, kr + 16, AT_HD, co);
                nax_mma2_kn(o[f], o[f + 1], pa, vb0, vb1);
            }
        }
    }
    // l total = sum of the 8 parts (each simdgroup summed its keys), in fixed order.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if ((lane & 9) == 0) { redm[j][fm] = l0; redm[j][fm + 8] = l1; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float L0 = redm[0][fm], L1 = redm[0][fm + 8];
    NAX_UNROLL for (int x = 1; x < NSG; x++) { L0 += redm[x][fm]; L1 += redm[x][fm + 8]; }
    device float* op = opart + size_t(split) * Q * AT_HD;
    NAX_UNROLL for (int f = 0; f < PVF; f++) NAX_UNROLL for (int i = 0; i < 2; i++) {
        const uint m = i == 0 ? m0 : m1;
        if (i == 0 ? val0 : val1) *(device float4*)(op + size_t(m) * AT_HD + d0 + 16 * f + fn) =
            float4(o[f][4 * i], o[f][4 * i + 1], o[f][4 * i + 2], o[f][4 * i + 3]);
    }
    if (j == 0 && (lane & 9) == 0) {
        device float2* mp = ml + size_t(split) * Q;
        if (val0) mp[m0] = float2(mr0, L0);
        if (val1) mp[m1] = float2(mr1, L1);
    }
}
#endif
