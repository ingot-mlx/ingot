
// int8 KV write and loads for the NAX attention kernels (verify_attn_pf.metal, verify_attn_pm.metal). Concatenated after
// qmm_nax.metal and verify_attn.metal. V is stored transposed in the cache ([head][dim][position], like splash
// `paged_store_row.h`: "V dimension-major"), so P*V is A*B^T with B = V^T in its natural order.
//
// int8 KV (like splash: symmetric, one fp32 scale per token and head, V by dimensions). K and V^T are loaded as
// unscaled half (exactly -127..127) and each key's scale is applied outside: S = scale_k*(q . k8) and P*V = sum (p *
// scale_v) v8. Half the KV bytes per round.
//
// MLA's latent head (LAB_AT_HD 576, GLM-4.7-Flash): one KV head [H₆₄ k_pe | H₅₁₂ c] in kc and no vt. ks holds the
// scale of k_pe and vs that of c, so S = scale·(ks·(q_pe · k8_pe) + vs·(q_c · k8_c)), and V = c is read from the
// latent part of kc (key-major, not transposed) with its scale vs: P·V = Σ (p · vs) (H₅₁₂ c)8, rotated back by H₅₁₂
// where the output is normalized (vattn_reduce in decode and verification, mla_out_unrotate after the prefill).
// 584 bytes per token and layer instead of 1,160 (no second copy of c).

// MLA's latent head: [k_pe | c] in one KV head, with c also V (see above); AT_CPE = the 16-dim chunks of k_pe.
constant constexpr bool AT_MLA = LAB_AT_HD == 576;
constant constexpr int AT_CPE = AT_MLA ? 4 : 0;

METAL_FUNC void nax_load_i8(thread nax_hfrag& d, const device char* src, uint ld, short2 c) {
    const char4 v0 = *(const device char4*)(src + size_t(c.x) * ld + c.y);
    const char4 v1 = *(const device char4*)(src + size_t(c.x + 8) * ld + c.y);
    d = nax_hfrag(half4(v0), half4(v1));
}

// C0 += A · B0, C1 += A · B1 (A: 16×16 [m,k]; B0, B1: 16×16 [k,n], not transposed); one 16×32×16 matmul2d. With
// nax_load_i8 over a key-major source (rows = keys, columns = dims), B is V read straight from K (MLA's latent part
// of kc). The fragment layout of the right operand is the same as A's (KernelCase mla-mma-kn checks it).
METAL_FUNC void nax_mma2_kn(thread nax_ffrag& c0, thread nax_ffrag& c1, const thread nax_hfrag& a,
                            const thread nax_hfrag& b0, const thread nax_hfrag& b1) {
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        16, 32, 16, false, false, true, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    auto ca = op.template get_left_input_cooperative_tensor<half, half, float>();
    auto cb = op.template get_right_input_cooperative_tensor<half, half, float>();
    auto cc = op.template get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(ca)>, metal::remove_addrspace_t<decltype(cb)>, float>();
    NAX_UNROLL for (short i = 0; i < 8; i++) {
        ca[i] = a[i];
        cb[i] = b0[i]; cb[8 + i] = b1[i];
        cc[i] = c0[i]; cc[8 + i] = c1[i];
    }
    op.run(ca, cb, cc);
    NAX_UNROLL for (short i = 0; i < 8; i++) { c0[i] = cc[i]; c1[i] = cc[8 + i]; }
}

// Norms, RoPE and K/V write, with K and V stored in int8 with their scale. Grid (AT_QH + 2*AT_KVH, B), AT_HD threads.
// Q and K leave rotated by the orthonormal Hadamard H after the RoPE (the cache holds H k): (H q) . (H k) = q . k, so
// the readers are unchanged, and the rotation spreads outlier channels over the head before the per-token scale is
// taken, which lowers the int8 error of K. V is stored as it is (MLA: no V slot, see above).
template <bool TREE>
[[kernel]] void vattn_prework_i8(const device float*  proj [[buffer(0)]],
                                 const device LAB_NWT* qnw  [[buffer(1)]],
                                 const device LAB_NWT* knw  [[buffer(2)]],
                                 device half*         q    [[buffer(3)]],
                                 device char*         kc   [[buffer(4)]],   // [AT_KVH, ldkv, AT_HD]
                                 device float*        ks   [[buffer(5)]],   // [AT_KVH, ldkv]
                                 constant AttnParams& p    [[buffer(6)]],
                                 device char*         vt   [[buffer(7)]],   // [AT_KVH, AT_HD, ldkv]
                                 device float*        vs   [[buffer(8)]],   // [AT_KVH, ldkv]
                                 const device uint*   st   [[buffer(9)]],
                                 const device uint*   tree [[buffer(10)]],  // (TREE): depth per row in tree[32 + row]
                                 const device float*  qkvb [[buffer(12)]]   // [AT_QP] bias (AT_QKVB)
                                 AT_RTAB_ARG(11),
                                 uint2 tg [[threadgroup_position_in_grid]],
                                 uint d [[thread_index_in_threadgroup]],
                                 ushort sg [[simdgroup_index_in_threadgroup]],
                                 ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float xs[AT_HD];
    threadgroup float part[AT_NSG];
    const uint slot = tg.x, t = tg.y;
    const device float* row = proj + size_t(t) * AT_QP;
    const uint col = at_col(slot, d);
    float x = row[col];
    if (AT_QKVB) x += qkvb[col];
    // Tree: row t goes in slot ctx + t of the cache, but its RoPE is that of its depth (ctx + dep).
    const uint ctx = (p.dyn & AT_DYN_ST) ? st[0] : p.ctx;
    const uint pos = ctx + t;
    const uint rpos = TREE ? ctx + tree[32 + t] : pos;
    // Ring KV of a sliding-window layer: the key of position pos goes to slot pos % ring.
    const uint kpos = LAB_AT_SWA && p.ring ? pos % p.ring : pos;
    if (slot < AT_QH + AT_KVH) {
        if (AT_QKN) {
            const float ss = simd_sum(x * x);
            if (lane == 0) part[sg] = ss;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float tot = 0;
            VUNROLL for (uint i = 0; i < AT_NSG; i++) tot += part[i];
            x = x * rsqrt(tot / float(AT_HD) + p.eps) * float(slot < AT_QH ? qnw[d] : knw[d]);
        }
        xs[d] = x;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (d < AT_ROT && !(p.dyn & AT_DYN_NOPE)) {
            const uint i = d % AT_RH;
            const float ang = AT_ROPE_ANG(rpos, i);
            const float c = cos(ang), s = sin(ang);
            x = AT_ROPE_SCALE(d < AT_RH ? x * c - xs[d + AT_RH] * s : x * c + xs[d - AT_RH] * s);
        }
        // The branch is uniform per threadgroup (slot = tg.x), so the barriers inside at_fwht are legal; its first
        // barrier also orders the RoPE reads of xs before it is overwritten.
        x = at_fwht(x, xs, d, lane);
        if (slot < AT_QH) {
            q[(size_t(slot / AT_G) * AT_G * p.B + t * AT_G + slot % AT_G) * AT_HD + d] = half(x);
            return;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    } else if (AT_VN) {
        // Weightless v_norm (uniform per threadgroup); the barrier keeps the reads of `part` before the max below.
        x *= rsqrt(at_head_ss(x, part, sg, lane) / float(AT_HD) + p.eps);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // K (AT_QH...) or V (AT_QH + AT_KVH...): max of |x| in the head and symmetric quantization.
    const float mx = simd_max(fabs(x));
    if (lane == 0) part[sg] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
#if LAB_AT_HD == 576
    // MLA's latent head: K only (the host launches no V slot; V is read from K) with two scales, one for [k_pe]
    // (simdgroups 0-1, dims 0…63) in ks and one for [c] (simdgroups 2…17) in vs: c is also V, and its own scale keeps
    // k_pe's range out of it.
    float ape = max(part[0], part[1]), ac = 0;
    VUNROLL for (uint i = 2; i < AT_NSG; i++) ac = max(ac, part[i]);
    const float amax = d < 64 ? ape : ac;
#else
    float amax = 0;
    VUNROLL for (uint i = 0; i < AT_NSG; i++) amax = max(amax, part[i]);
#endif
    const float sc = amax > 0 ? amax / 127.0f : 1.0f;
    const char qv = char(clamp(rint(x / sc), -127.0f, 127.0f));
    const uint h = slot < AT_QH + AT_KVH ? slot - AT_QH : slot - AT_QH - AT_KVH;
    if (slot < AT_QH + AT_KVH) {
        kc[(size_t(h) * p.ldkv + kpos) * AT_HD + d] = qv;
        if (d == 0) ks[h * p.ldkv + kpos] = sc;
#if LAB_AT_HD == 576
        if (d == 64) vs[h * p.ldkv + kpos] = sc;
#endif
    } else {
        vt[(size_t(h) * AT_HD + d) * p.ldkv + kpos] = qv;
        if (d == 0) vs[h * p.ldkv + kpos] = sc;
    }
}
template [[host_name("vattn_prework_i8")]] [[kernel]] void vattn_prework_i8<false>(const device float*, const device LAB_NWT*,
    const device LAB_NWT*, device half*, device char*, device float*, constant AttnParams&, device char*, device float*,
    const device uint*, const device uint*, const device float* AT_RTAB_T, uint2, uint, ushort, ushort);
template [[host_name("vattn_prework_i8t")]] [[kernel]] void vattn_prework_i8<true>(const device float*, const device LAB_NWT*,
    const device LAB_NWT*, device half*, device char*, device float*, constant AttnParams&, device char*, device float*,
    const device uint*, const device uint*, const device float* AT_RTAB_T, uint2, uint, ushort, ushort);
