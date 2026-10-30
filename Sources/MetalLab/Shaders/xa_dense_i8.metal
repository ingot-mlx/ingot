// Dense prefill GEMMs in rotated W8A8 (lossy option, not the default): dense Qwen3.8-27B.
//
// Per call (no permanent copy of the weights):
//   1. w4_i8rot: 4-bit weights (affine, groups of 64) → per-row int8 after a 32-wide Hadamard along K, with
//      its per-row scale (two passes per row: maximum and write). They are written to a work buffer [N, K].
//   2. quant_rows_rot_i8g: half activations [M, K] → per-row int8 with the same rotation.
//   3. qmm_i8: Y = (Xq · Wqᵀ) · sx · sw / 32 with matmul2d int8×int8 → int32 (42 TOPS measured versus 21 for fp16),
//      the body of qmm_t_nax3 (X and W through threadgroup memory, grid WM × WN of 32 × 32 simdgroups).
// The conversion is paid once per prefill chunk (2048 rows), not per tile as in the earlier MoE attempt, where it
// went inside the GEMM and cost more than it saved. Idea of converting per call: our mlx-serve PR #885
// (`w8a8.zig`, `convert4`); the 32-wide Hadamard (fwht32) is in qmm_nax_i8.metal.
// Concatenated after qmm_nax_i8.metal (nax_cfrag, nax_ifrag, nax_load_c, nax_mma2_i8, fwht32).

// Hadamard of 32·2^LB: the 32-wide one of each block (fwht32) and LB more stages between neighboring blocks. The blocks of a row are
// distributed blk = lane + 32·i, so neighboring blocks sit in neighboring lanes: the stages between blocks are
// simd_shuffle_xor (requires K/32 to be a multiple of 32, as in the 27B: 5120, 6144, 17408).
template <int LB>
METAL_FUNC void fwht_blocks(thread float (&f)[32], ushort lane) {
    NAX_UNROLL for (int st = 0; st < LB; st++) {
        const ushort m = ushort(1) << st;
        const bool hi = (lane & m) != 0;
        NAX_UNROLL for (int i = 0; i < 32; i++) {
            const float o = simd_shuffle_xor(f[i], m);
            f[i] = hi ? o - f[i] : f[i] + o;
        }
    }
}

// 4-bit weights [N, K] → per-row rotated int8. One simdgroup per row (4 per threadgroup); each lane carries blocks of
// 32 values (one uint4 of weights, half a scale group).
// MODE 0: both passes; 1: only the per-row scale (at load time); 2: only the write, with the scale of sw (precomputed).
template <int LB, int MODE, int G = 64>
[[kernel]] void w4_i8rot(const device uint*   w   [[buffer(0)]],
                         const device bfloat* s   [[buffer(1)]],
                         const device bfloat* b   [[buffer(2)]],
                         device char*         wq  [[buffer(3)]],
                         device float*        sw  [[buffer(4)]],
                         constant uint2&      p   [[buffer(5)]],   // (N, K)
                         uint   tg   [[threadgroup_position_in_grid]],
                         ushort sg   [[simdgroup_index_in_threadgroup]],
                         ushort lane [[thread_index_in_simdgroup]]) {
    const uint r = tg * 4 + sg;
    if (r >= p.x) return;
    const uint K = p.y, nb = K / 32;
    auto block = [&](uint blk, thread float (&f)[32]) {
        const uint4 v = *(const device uint4*)(w + size_t(r) * (K / 8) + blk * 4);
        const uint g = G == 64 ? r * (K / 64) + blk / 2 : r * (K / 32) + blk;   // block of 32: half a group or a whole one
        const float sc = float(s[g]), bi = float(b[g]);
        NAX_UNROLL for (int u = 0; u < 4; u++) NAX_UNROLL for (int j = 0; j < 8; j++) f[u * 8 + j] = sc * float((v[u] >> (4 * j)) & 15u) + bi;
        fwht32(f);
        fwht_blocks<LB>(f, lane);
    };
    float sc;
    if (MODE == 2) {
        sc = sw[r];
    } else {
        float m = 0;
        for (uint blk = lane; blk < nb; blk += 32) {
            float f[32];
            block(blk, f);
            NAX_UNROLL for (int i = 0; i < 32; i++) m = max(m, abs(f[i]));
        }
        m = simd_max(m);
        sc = m > 0 ? m / 127.0f : 1.0f;
        if (lane == 0) sw[r] = sc;
        if (MODE == 1) return;
    }
    const float inv = 1.0f / sc;
    for (uint blk = lane; blk < nb; blk += 32) {
        float f[32];
        block(blk, f);
        device char* o = wq + size_t(r) * K + blk * 32;
        NAX_UNROLL for (int i = 0; i < 8; i++)
            *(device char4*)(o + 4 * i) = char4(clamp(rint(float4(f[4 * i], f[4 * i + 1], f[4 * i + 2], f[4 * i + 3]) * inv), -127.0f, 127.0f));
    }
}

// Half rows [M, K] → per-row rotated int8 (K a multiple of 32, any size). One simdgroup per row.
// SW: x is gate | up [M, 2K] (half output of the gate|up GEMM) and h = silu(gate)·up is quantized (rounded to half as
// in dswiglu_hh): the SwiGLU is fused with the quantization of the input of down.
// ACT: activation of the fused gated FFN (lab_glu; `swiglu_quant_rot_i8_gelu` with LAB_FFN_GELU).
template <int LB, bool SW, int ACT = 0>
[[kernel]] void quant_rows_rot_i8g(const device half* x  [[buffer(0)]],
                                   device char*       xq [[buffer(1)]],
                                   device float*      sx [[buffer(2)]],
                                   constant uint2&    p  [[buffer(3)]],   // (rows, K)
                                   uint   tg   [[threadgroup_position_in_grid]],
                                   ushort sg   [[simdgroup_index_in_threadgroup]],
                                   ushort lane [[thread_index_in_simdgroup]]) {
    const uint r = tg * 4 + sg;
    if (r >= p.x) return;
    const uint K = p.y, nb = K / 32;
    const device half* xr = x + size_t(r) * (SW ? 2 * K : K);
    auto block = [&](uint blk, thread float (&f)[32]) {
        NAX_UNROLL for (int i = 0; i < 8; i++) {
            half4 v = *(const device half4*)(xr + blk * 32 + 4 * i);
            if (SW) {
                const float4 g = float4(v), u = float4(*(const device half4*)(xr + K + blk * 32 + 4 * i));
                v = half4(lab_glu<ACT>(g, u));
            }
            f[4 * i] = v.x; f[4 * i + 1] = v.y; f[4 * i + 2] = v.z; f[4 * i + 3] = v.w;
        }
        fwht32(f);
        fwht_blocks<LB>(f, lane);
    };
    float m = 0;
    for (uint blk = lane; blk < nb; blk += 32) {
        float f[32];
        block(blk, f);
        NAX_UNROLL for (int i = 0; i < 32; i++) m = max(m, abs(f[i]));
    }
    m = simd_max(m);
    const float sc = m > 0 ? m / 127.0f : 1.0f, inv = 1.0f / sc;
    if (lane == 0) sx[r] = sc;
    for (uint blk = lane; blk < nb; blk += 32) {
        float f[32];
        block(blk, f);
        device char* o = xq + size_t(r) * K + blk * 32;
        NAX_UNROLL for (int i = 0; i < 8; i++)
            *(device char4*)(o + 4 * i) = char4(clamp(rint(float4(f[4 * i], f[4 * i + 1], f[4 * i + 2], f[4 * i + 3]) * inv), -127.0f, 127.0f));
    }
}

// `rows` rows × BK bytes of a K step, per uint4 (BK/16 per row); row stride LD in threadgroup memory.
template <int rows, int NT, int BK = 64>
struct I8Stage {
    static constant constexpr int CPR = BK / 16, LD = BK + 16, NI = (rows * CPR + NT - 1) / NT;
    uint4 v[NI];
    METAL_FUNC void load(const device char* src, uint ld, uint r0, uint k0, uint tid) {
        NAX_UNROLL for (int i = 0; i < NI; i++) {
            const uint ch = tid + i * NT;
            if (NI * NT != rows * CPR && ch >= uint(rows * CPR)) break;
            v[i] = *(const device uint4*)(src + size_t(r0 + ch / CPR) * ld + k0 + (ch % CPR) * 16);
        }
    }
    METAL_FUNC void store(threadgroup char* dst, uint tid) const {
        NAX_UNROLL for (int i = 0; i < NI; i++) {
            const uint ch = tid + i * NT;
            if (NI * NT != rows * CPR && ch >= uint(rows * CPR)) break;
            *(threadgroup uint4*)(dst + (ch / CPR) * LD + (ch % CPR) * 16) = v[i];
        }
    }
};

// Y[M, N] (+)= (Xq[M, K] · Wq[N, K]ᵀ) · sx[m] · sw[n] · SC. Tile 32·WM × 32·WN; M a multiple of 32·WM, N of 32·WN.
template <int WM, int WN, typename OT, bool ACC, int BK = 64>
[[kernel, max_total_threads_per_threadgroup(32 * WM * WN)]] void qmm_i8(
    const device char*  xq  [[buffer(0)]],
    const device char*  wq  [[buffer(1)]],
    const device float* sx  [[buffer(2)]],
    const device float* sw  [[buffer(3)]],
    device OT*          y   [[buffer(4)]],
    constant QmmParams& p   [[buffer(5)]],
    constant float&     SC  [[buffer(6)]],
    uint3  tgid [[threadgroup_position_in_grid]],
    uint   tid  [[thread_index_in_threadgroup]],
    ushort sgid [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]])
{
    constexpr int NT = 32 * WM * WN, BM = 32 * WM, BN = 32 * WN, LD = BK + 16, KF = BK / 16;
    threadgroup char ws[BN * LD];
    threadgroup char xs[BM * LD];
    // (With the row tiles along x, so that simultaneous threadgroups share the weight tile, it was worse:
    // gate|up 25.7 → 32 ms, down 19.3 → 17.5 ms.)
    const uint r0 = tgid.y * BM, n0 = tgid.x * BN;
    const int tm = (sgid / WN) * 32, tn = (sgid % WN) * 32;
    const short2 c = nax_coord(lane);
    I8Stage<BM, NT, BK> sa;
    I8Stage<BN, NT, BK> sb;
    nax_ifrag a00 = 0, a01 = 0, a10 = 0, a11 = 0;
    sa.load(xq, p.K, r0, 0, tid);
    sb.load(wq, p.K, n0, 0, tid);
    for (uint k0 = 0; k0 < p.K; k0 += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sa.store(xs, tid);
        sb.store(ws, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (k0 + BK < p.K) {
            sa.load(xq, p.K, r0, k0 + BK, tid);
            sb.load(wq, p.K, n0, k0 + BK, tid);
        }
        nax_cfrag b0[KF], b1[KF];
        NAX_UNROLL for (int kk = 0; kk < KF; kk++) {
            nax_load_c(b0[kk], ws + tn * LD + kk * 16, LD, c);
            nax_load_c(b1[kk], ws + (tn + 16) * LD + kk * 16, LD, c);
        }
        NAX_UNROLL for (int kk = 0; kk < KF; kk++) {
            nax_cfrag a, a1;
            nax_load_c(a, xs + tm * LD + kk * 16, LD, c);
            nax_load_c(a1, xs + (tm + 16) * LD + kk * 16, LD, c);
            nax_mma2_i8(a00, a01, a, b0[kk], b1[kk]);
            nax_mma2_i8(a10, a11, a1, b0[kk], b1[kk]);
        }
    }
    NAX_UNROLL for (int j = 0; j < 2; j++) {
        const float4 s4 = *(const device float4*)(sw + n0 + tn + j * 16 + c.y) * SC;
        NAX_UNROLL for (int i = 0; i < 2; i++) {
            const nax_ifrag A8 = i == 0 ? (j == 0 ? a00 : a01) : (j == 0 ? a10 : a11);
            NAX_UNROLL for (int q = 0; q < 2; q++) {
                const uint r = r0 + tm + i * 16 + c.x + q * 8;
                const float4 v = float4(A8[q * 4], A8[q * 4 + 1], A8[q * 4 + 2], A8[q * 4 + 3]) * sx[r] * s4;
                device vec<OT, 4>* yp = (device vec<OT, 4>*)(y + size_t(r) * p.N + n0 + tn + j * 16 + c.y);
                *yp = ACC ? *yp + vec<OT, 4>(v) : vec<OT, 4>(v);
            }
        }
    }
}

#define INST_QI8(WM, WN, BK)                                                                                       \
  template [[host_name("qmm_i8_w" #WM "x" #WN "_bk" #BK "_f32acc")]] [[kernel]] void qmm_i8<WM, WN, float, true, BK>( \
      const device char*, const device char*, const device float*, const device float*, device float*,            \
      constant QmmParams&, constant float&, uint3, uint, ushort, ushort);                                         \
  template [[host_name("qmm_i8_w" #WM "x" #WN "_bk" #BK "_f32")]] [[kernel]] void qmm_i8<WM, WN, float, false, BK>( \
      const device char*, const device char*, const device float*, const device float*, device float*,            \
      constant QmmParams&, constant float&, uint3, uint, ushort, ushort);                                         \
  template [[host_name("qmm_i8_w" #WM "x" #WN "_bk" #BK "_h")]] [[kernel]] void qmm_i8<WM, WN, half, false, BK>(    \
      const device char*, const device char*, const device float*, const device float*, device half*,             \
      constant QmmParams&, constant float&, uint3, uint, ushort, ushort);
INST_QI8(2, 2, 64) INST_QI8(2, 4, 64) INST_QI8(2, 2, 128) INST_QI8(2, 4, 128)

#define INST_ROT(LB)                                                                                               \
  template [[host_name("w4_i8rot_l" #LB)]] [[kernel]] void w4_i8rot<LB, 0>(const device uint*, const device bfloat*, \
      const device bfloat*, device char*, device float*, constant uint2&, uint, ushort, ushort);                  \
  template [[host_name("w4_rowscale_l" #LB)]] [[kernel]] void w4_i8rot<LB, 1>(const device uint*, const device bfloat*, \
      const device bfloat*, device char*, device float*, constant uint2&, uint, ushort, ushort);                  \
  template [[host_name("w4_i8rot_pre_l" #LB)]] [[kernel]] void w4_i8rot<LB, 2>(const device uint*, const device bfloat*, \
      const device bfloat*, device char*, device float*, constant uint2&, uint, ushort, ushort);                  \
  template [[host_name("quant_rows_rot_i8g_l" #LB)]] [[kernel]] void quant_rows_rot_i8g<LB, false>(const device half*, \
      device char*, device float*, constant uint2&, uint, ushort, ushort);                                        \
  template [[host_name("swiglu_quant_rot_i8_l" #LB)]] [[kernel]] void quant_rows_rot_i8g<LB, true>(const device half*, \
      device char*, device float*, constant uint2&, uint, ushort, ushort);
INST_ROT(0) INST_ROT(3)
#if LAB_FFN_GELU
#define INST_ROT_GELU(LB)                                                                                          \
  template [[host_name("swiglu_quant_rot_i8_gelu_l" #LB)]] [[kernel]] void quant_rows_rot_i8g<LB, true, 1>(        \
      const device half*, device char*, device float*, constant uint2&, uint, ushort, ushort);
INST_ROT_GELU(0) INST_ROT_GELU(3)
#endif
// 4-bit weights with groups of 32 (variant j): the same conversion (each block of 32 is a whole group).
#if LAB_G32
#define INST_ROT_G32(LB)                                                                                           \
  template [[host_name("w4_i8rot_g32_l" #LB)]] [[kernel]] void w4_i8rot<LB, 0, 32>(const device uint*,            \
      const device bfloat*, const device bfloat*, device char*, device float*, constant uint2&, uint, ushort, ushort); \
  template [[host_name("w4_rowscale_g32_l" #LB)]] [[kernel]] void w4_i8rot<LB, 1, 32>(const device uint*,         \
      const device bfloat*, const device bfloat*, device char*, device float*, constant uint2&, uint, ushort, ushort); \
  template [[host_name("w4_i8rot_pre_g32_l" #LB)]] [[kernel]] void w4_i8rot<LB, 2, 32>(const device uint*,        \
      const device bfloat*, const device bfloat*, device char*, device float*, constant uint2&, uint, ushort, ushort);
INST_ROT_G32(0) INST_ROT_G32(3)
#endif
