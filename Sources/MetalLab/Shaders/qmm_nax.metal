// Matmul with quantized weights on the GPU matrix accelerators (NAX) with
// MetalPerformancePrimitives `matmul2d` (public API, macOS 26+).
//
// Scheme taken from MLX (MIT, Copyright © 2025 Apple Inc.): mlx/backend/metal/kernels/steel/gemm/nax.h
// (BaseNAXFrag: 16×16 fragments, 8 elements per thread, get_coord coordinates, mma 16×32×16) and
// quantized_nax.h (qmm_t_nax_tgp_impl: X straight from memory to registers, W dequantized in threadgroup).
// `nax_coord` and `nax_mma2` are copied from nax.h (get_coord and the mma); see THIRD_PARTY_NOTICES.md.
// This file is concatenated after qmm.metal (it uses QmmParams from there).

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

// All loops over fragments must be unrolled: otherwise the fragment arrays are indexed at
// run time, go to private memory and matmul2d drops from ~21 to ~3 TFLOPS (measured with a microbenchmark).
#define NAX_UNROLL _Pragma("clang loop unroll(full)")

typedef vec<half, 8>  nax_hfrag;
typedef vec<float, 8> nax_ffrag;

// (row, column) of the thread's first element in a 16×16 fragment. Element e = i*4 + j is at
// (row + 8*i, column + j), i ∈ {0,1}, j ∈ {0..3}.
METAL_FUNC short2 nax_coord(ushort lane) {
    const short qid = lane >> 2;
    const short fm = (qid & 4) | ((lane >> 1) & 3);
    const short fn = ((qid & 2) | (lane & 1)) * 4;
    return short2(fm, fn);
}

// Vector load: each thread reads 2 contiguous half4 (rows c.x and c.x + 8).
METAL_FUNC void nax_load(thread nax_hfrag& d, const device half* src, uint ld, short2 c) {
    const half4 v0 = *(const device half4*)(src + size_t(c.x) * ld + c.y);
    const half4 v1 = *(const device half4*)(src + size_t(c.x + 8) * ld + c.y);
    d = nax_hfrag(v0, v1);
}
METAL_FUNC void nax_load(thread nax_hfrag& d, const threadgroup half* src, uint ld, short2 c) {
    const half4 v0 = *(const threadgroup half4*)(src + c.x * ld + c.y);
    const half4 v1 = *(const threadgroup half4*)(src + (c.x + 8) * ld + c.y);
    d = nax_hfrag(v0, v1);
}

// C0 += A · B0ᵀ, C1 += A · B1ᵀ  (A: 16×16 [m,k]; B0, B1: 16×16 [n,k]); one 16×32×16 matmul2d operation.
METAL_FUNC void nax_mma2(thread nax_ffrag& c0, thread nax_ffrag& c1, const thread nax_hfrag& a,
                         const thread nax_hfrag& b0, const thread nax_hfrag& b1) {
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        16, 32, 16, false, true, true, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
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

constant constexpr int NX_BM = 64, NX_BK = 64, NX_LD = NX_BK + 8, NX_THREADS = 128;

// Dequantizes rows [n0, n0+rows) × columns [k0, k0+NX_BK) of W into ws (row r at ws + r*NX_LD).
template <int bits, int G, int rows>
METAL_FUNC void nax_load_w(const device uint* w, const device bfloat* s, const device bfloat* b,
                           threadgroup half* ws, uint K, uint n0, uint k0, uint tid) {
    constexpr uint per = 32 / bits, upr = NX_BK / per;
    const uint packedPerRow = K / per;
    if constexpr (bits == 16) {
        // Unquantized bf16 (the bf16 MoE router): two values per uint32, to half (s and b are not read).
        for (uint i = 0; i < (rows * upr) / NX_THREADS; i++) {
            const uint idx = tid + i * NX_THREADS;
            const uint r = idx / upr, u = idx % upr;
            const float2 v = float2(as_type<bfloat2>(w[size_t(n0 + r) * packedPerRow + (k0 + u * per) / per]));
            *(threadgroup half2*)(ws + r * NX_LD + u * per) = half2(v);
        }
        return;
    }
    if constexpr (bits == 4) {
        // 4 bits: (v & 0x000F000F) | 0x6400 gives the half 1024 + q for elements j and j+4 at once (a known
        // half-precision bit trick); subtracting 1024 is exact. Result in half2 with fma and two half4 stores per uint32.
        for (uint i = 0; i < (rows * upr) / NX_THREADS; i++) {
            const uint idx = tid + i * NX_THREADS;
            const uint r = idx / upr, u = idx % upr;
            const uint k = k0 + u * per;
            const uint g = (n0 + r) * (K / G) + k / G;
            const half2 sc = half2(half(float(s[g]))), bi = half2(half(float(b[g])));
            const uint v = w[size_t(n0 + r) * packedPerRow + k / per];
            const half2 q0 = fma(as_type<half2>((v & 0x000F000Fu) | 0x64006400u) - half2(1024.0h), sc, bi);
            const half2 q1 = fma(as_type<half2>(((v >> 4) & 0x000F000Fu) | 0x64006400u) - half2(1024.0h), sc, bi);
            const half2 q2 = fma(as_type<half2>(((v >> 8) & 0x000F000Fu) | 0x64006400u) - half2(1024.0h), sc, bi);
            const half2 q3 = fma(as_type<half2>(((v >> 12) & 0x000F000Fu) | 0x64006400u) - half2(1024.0h), sc, bi);
            threadgroup half* o = ws + r * NX_LD + u * per;
            *(threadgroup half4*)o = half4(q0.x, q1.x, q2.x, q3.x);
            *(threadgroup half4*)(o + 4) = half4(q0.y, q1.y, q2.y, q3.y);
        }
        return;
    }
    for (uint i = 0; i < (rows * upr) / NX_THREADS; i++) {
        const uint idx = tid + i * NX_THREADS;
        const uint r = idx / upr, u = idx % upr;
        const uint k = k0 + u * per;
        const uint g = (n0 + r) * (K / G) + k / G;
        const half sc = half(float(s[g])), bi = half(float(b[g]));
        const uint v = w[size_t(n0 + r) * packedPerRow + k / per];
        threadgroup half* o = ws + r * NX_LD + u * per;
        for (uint j = 0; j < per; j++) o[j] = sc * half((v >> (bits * j)) & ((1u << bits) - 1)) + bi;
    }
}

// Y[M,N] = X[M,K] · W[N,K]ᵀ. 64×64 tile, BK = 64, 4 simdgroups 2×2 of 32×32 (2×2 fragments of 16×16).
// OT: output type (half, or float for the MoE router logits).
template <int bits, int G, typename OT = half, bool ACC = false>
[[kernel, max_total_threads_per_threadgroup(NX_THREADS)]] void qmm_t_nax(
    const device half*   x   [[buffer(0)]],
    const device uint*   w   [[buffer(1)]],
    const device bfloat* s   [[buffer(2)]],
    const device bfloat* b   [[buffer(3)]],
    device OT*           y   [[buffer(4)]],
    constant QmmParams&  p   [[buffer(5)]],
    uint3  tgid [[threadgroup_position_in_grid]],
    uint   tid  [[thread_index_in_threadgroup]],
    ushort sgid [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]])
{
    constexpr int BN = 64;
    threadgroup half ws[BN * NX_LD];
    const uint m0 = tgid.y * NX_BM, n0 = tgid.x * BN;
    const uint tm = (sgid / 2) * 32, tn = (sgid % 2) * 32;
    const short2 c = nax_coord(lane);
    const device half* xs = x + size_t(m0 + tm) * p.K;

    nax_ffrag acc[2][2];
    NAX_UNROLL for (int i = 0; i < 2; i++) NAX_UNROLL for (int j = 0; j < 2; j++) acc[i][j] = 0;

    for (uint k0 = 0; k0 < p.K; k0 += NX_BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        nax_load_w<bits, G, BN>(w, s, b, ws, p.K, n0, k0, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // First all the loads of the K block, then, for each row of fragments, the 4 calls back to back
        // on the same accumulator: the accelerators only run at full speed if each matmul2d accumulates onto the
        // destination of the previous call (microbenchmark: 21 TFLOPS with one chain, 3.3 alternating two).
        nax_hfrag a[2][4], bt[2][4];
        NAX_UNROLL for (int kk = 0; kk < 4; kk++) {
            NAX_UNROLL for (int i = 0; i < 2; i++) nax_load(a[i][kk], xs + size_t(i * 16) * p.K + k0 + kk * 16, p.K, c);
            NAX_UNROLL for (int j = 0; j < 2; j++) nax_load(bt[j][kk], ws + (tn + j * 16) * NX_LD + kk * 16, NX_LD, c);
        }
        NAX_UNROLL for (int i = 0; i < 2; i++)
            NAX_UNROLL for (int kk = 0; kk < 4; kk++) nax_mma2(acc[i][0], acc[i][1], a[i][kk], bt[0][kk], bt[1][kk]);
    }

    NAX_UNROLL for (int i = 0; i < 2; i++)
        NAX_UNROLL for (int j = 0; j < 2; j++)
            NAX_UNROLL for (int e = 0; e < 2; e++) {
                const uint r = m0 + tm + i * 16 + c.x + e * 8, col = n0 + tn + j * 16 + c.y;
                const vec<OT, 4> v(acc[i][j][e * 4], acc[i][j][e * 4 + 1], acc[i][j][e * 4 + 2], acc[i][j][e * 4 + 3]);
                device vec<OT, 4>* yp = (device vec<OT, 4>*)(y + size_t(r) * p.N + col);
                *yp = ACC ? *yp + v : v;
            }
}

// MoE router logits in fp32 (8 bits).
template [[host_name("qmm_t_nax_f32_b8_g64")]] [[kernel]] void qmm_t_nax<8, 64, float>(
    const device half*, const device uint*, const device bfloat*, const device bfloat*, device float*,
    constant QmmParams&, uint3, uint, ushort, ushort);
// The same with the bf16 router (Nemotron-H, GLM).
template [[host_name("qmm_t_nax_f32_bf16")]] [[kernel]] void qmm_t_nax<16, 64, float>(
    const device half*, const device uint*, const device bfloat*, const device bfloat*, device float*,
    constant QmmParams&, uint3, uint, ushort, ushort);
