// Common pieces of the int8 kernels on NAX (matmul2d int8×int8 → int32, 42 TOPS measured on the M6 vs 21 for
// fp16): int8/int32 fragments and the Walsh-Hadamard of 32 in registers. Used by the rotated W8A8 GEMM of the prefill
// (xa_dense_i8.metal) and by the prefill attention with int8 Q·Kᵀ (verify_attn_pf.metal).
// Concatenated after qmm_nax.metal (NAX_UNROLL, mpp).

typedef vec<char, 8> nax_cfrag;
typedef vec<int, 8>  nax_ifrag;

METAL_FUNC void nax_load_c(thread nax_cfrag& d, const threadgroup char* src, uint ld, short2 c) {
    const char4 v0 = *(const threadgroup char4*)(src + c.x * ld + c.y);
    const char4 v1 = *(const threadgroup char4*)(src + (c.x + 8) * ld + c.y);
    d = nax_cfrag(v0, v1);
}

// C0 = A·B0ᵀ, C1 = A·B1ᵀ in int32 (multiply-accumulate onto C).
METAL_FUNC void nax_mma2_i8(thread nax_ifrag& c0, thread nax_ifrag& c1, const thread nax_cfrag& a,
                            const thread nax_cfrag& b0, const thread nax_cfrag& b1) {
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        16, 32, 16, false, true, true, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    auto ca = op.template get_left_input_cooperative_tensor<int8_t, int8_t, int32_t>();
    auto cb = op.template get_right_input_cooperative_tensor<int8_t, int8_t, int32_t>();
    auto cc = op.template get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(ca)>, metal::remove_addrspace_t<decltype(cb)>, int32_t>();
    NAX_UNROLL for (short i = 0; i < 8; i++) {
        ca[i] = a[i];
        cb[i] = b0[i]; cb[8 + i] = b1[i];
        cc[i] = c0[i]; cc[8 + i] = c1[i];
    }
    op.run(ca, cb, cc);
    NAX_UNROLL for (short i = 0; i < 8; i++) { c0[i] = cc[i]; c1[i] = cc[8 + i]; }
}

// Walsh-Hadamard of 32, not normalized (H·Hᵀ = 32·I), in registers (constant indices in each stage: with the
// triple loop the compiler did not unroll it and the array went to the stack, 30× slower).
template <int H>
METAL_FUNC void fwht_stage(thread float (&f)[32]) {
    NAX_UNROLL for (int k = 0; k < 16; k++) {
        const int i = (k / H) * 2 * H + (k % H);
        const float a = f[i], b = f[i + H];
        f[i] = a + b;
        f[i + H] = a - b;
    }
}
METAL_FUNC void fwht32(thread float (&f)[32]) {
    fwht_stage<1>(f); fwht_stage<2>(f); fwht_stage<4>(f); fwht_stage<8>(f); fwht_stage<16>(f);
}

// Half rows [T, D] → rotated in blocks of 32 and quantized to int8 per row: xq = rint(Hx / s), s = max|Hx| / 127.
