// GLM-4.7-Flash MLA in the absorbed form over the latent cache (Layers/MLA.swift). Compiled only with the
// latent attention head (AT_HD 576 = qk_rope 64 + kv_lora_rank 512): the other shapes' libraries do not change.
//
// The attention reads one KV head [k_pe | c] (AT_KVH 1) with every query head (AT_G = AT_QH): the query of head h is
// [q_pe_h | embed_q_h · q_nope_h] and the output of each head (576) goes through unembed_out_h (its dims 64…575)
// before o_proj. Here:
// - mla_split: from q = q_b(norm(q_a)) (each head [q_pe | q_nope]), a = [q_a | c | k_pe] and the normed latent cn,
//   writes q_pe and the k head of the projection row the prework reads, and q_nope in half for embed_q;
// - mla_out_unrotate: H₅₁₂ on the latent output of the prefill attention (V is read rotated from K);
// - vnax_qmv_heads / qmm_t_nax3_heads: Y_h = X_h · W_hᵀ for every head h in one dispatch (grid dimension per head,
//   with per-head offsets of the weights, scales, input and output): embed_q (K nope → N kv_lora_rank) writes the
//   q_lat part of each q head straight into the projection row, unembed_out (K kv_lora_rank → N v_head_dim) reads
//   dims 64…575 of each head of the attention output.
#if LAB_AT_HD == 576

struct MLASplitParams {
    uint qhd;    // q head of q_b: rope + nope (256)
    uint rope;   // rotary dims at the head of each q and k head (64)
    uint nope;   // q_nope dims per head (192)
    uint aLd;    // stride of a (q_lora_rank + kv_lora_rank + rope)
    uint kpeOff; // column of k_pe in a
};

// Grid (ceil(cols / 256), rows), 256 threads, one thread per column of the virtual row
// [q heads (AT_QH × qhd) | k head (AT_HD)]:
// - q column (h, d): d < rope → proj[q head h, d] (q_pe); otherwise qn[h, d − rope] = half(q_nope);
// - k head: [k_pe (from a) | c (normed latent)]. The v head of the projection row is not written: the cache keeps c
//   once, in K, and V is read from there (verify_attn_nax.metal).
[[kernel]] void mla_split(const device float*  a    [[buffer(0)]],   // [rows, aLd]
                          const device float*  q    [[buffer(1)]],   // [rows, AT_QH · qhd]
                          const device half*   cn   [[buffer(2)]],   // [rows, AT_HD − rope]
                          device half*         qn   [[buffer(3)]],   // [rows, AT_QH · nope]
                          device float*        proj [[buffer(4)]],   // [rows, AT_QP]
                          constant MLASplitParams& p [[buffer(5)]],
                          uint2 tg [[threadgroup_position_in_grid]],
                          uint tid [[thread_index_in_threadgroup]]) {
    const uint col = tg.x * 256 + tid, t = tg.y, qN = AT_QH * p.qhd, R = AT_HD - p.rope;
    if (col >= qN + AT_HD) return;
    device float* row = proj + size_t(t) * AT_QP;
    if (col < qN) {
        const uint h = col / p.qhd, d = col % p.qhd;
        const float x = q[size_t(t) * qN + col];
        if (d < p.rope) row[h * AT_HD + d] = x;
        else qn[size_t(t) * AT_QH * p.nope + h * p.nope + d - p.rope] = half(x);
        return;
    }
    const uint c = col - qN;
    row[AT_QK + c] = c < p.rope ? a[size_t(t) * p.aLd + p.kpeOff + c] : float(cn[size_t(t) * R + c - p.rope]);
}

// The prefill attention (pattn_i8) leaves each head's latent output as P·(H₅₁₂ c), V being read rotated from K;
// this rotates it back in place (H₅₁₂ is orthonormal and its own inverse), as vattn_reduce does in decode. Grid
// (AT_QH, rows), AT_HD threads (at_fwht's shape: the 64 RoPE threads carry zeros, and write them, as the decode).
[[kernel]] void mla_out_unrotate(device half* out [[buffer(0)]],   // [rows, AT_QH · AT_HD], dims 64… of each head
                                 uint2 tg [[threadgroup_position_in_grid]],
                                 uint d [[thread_index_in_threadgroup]],
                                 ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float xs[AT_HD];
    device half* o = out + (size_t(tg.y) * AT_QH + tg.x) * AT_HD;
    const float v = at_fwht(d < 64 ? 0.0f : float(o[d]), xs, d, lane);
    o[d] = half(v);
}

// Per-head strides of vnax_qmv_heads (elements of x and y between two heads; the weights of head h start at row h·N of
// the [heads · N, K] matrix).
struct MLAHeadParams {
    uint xh;
    uint yh;
};

// vnax_qmv (verify_nax.metal) with a head per tg.y: Y_h[M ≤ 16, N] (+)= X_h · W_hᵀ (4 bits, groups of 64), output fp32
// or half (OT). The same body (vnax_qmv_body) per head, so each head gives the same bits as a vnax_qmv over its own
// matrix.
template <int NF, int SGS, typename OT>
[[kernel]] void vnax_qmv_heads(const device half*   x  [[buffer(0)]],
                               const device uint*   w  [[buffer(1)]],
                               const device bfloat* s  [[buffer(2)]],
                               const device bfloat* b  [[buffer(3)]],
                               device OT*           y  [[buffer(4)]],
                               constant VParams&    p  [[buffer(5)]],
                               constant MLAHeadParams& hp [[buffer(6)]],
                               uint2 tg [[threadgroup_position_in_grid]],
                               uint tid [[thread_index_in_threadgroup]],
                               ushort sg [[simdgroup_index_in_threadgroup]],
                               ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[SGS * 16 * 16 * NF];
    const uint h = tg.y, wpr = p.K / 8, gpr = p.K / 64;
    const VLoad4<64> wl{w + size_t(h) * p.N * wpr, s + size_t(h) * p.N * gpr, b + size_t(h) * p.N * gpr, wpr, gpr};
    vnax_qmv_body<NF, SGS>(x + size_t(h) * hp.xh, wl, y + size_t(h) * hp.yh, p, tg.x * 16 * NF, red, tid, sg, lane);
}

#define VNQH_INST(NF, SGS)                                                                                         \
  template [[host_name("vnax_qmv_heads_nf" #NF "_sg" #SGS "_f32")]] [[kernel]] void vnax_qmv_heads<NF, SGS, float>( \
      const device half*, const device uint*, const device bfloat*, const device bfloat*, device float*,           \
      constant VParams&, constant MLAHeadParams&, uint2, uint, ushort, ushort);                                    \
  template [[host_name("vnax_qmv_heads_nf" #NF "_sg" #SGS "_h")]] [[kernel]] void vnax_qmv_heads<NF, SGS, half>(    \
      const device half*, const device uint*, const device bfloat*, const device bfloat*, device half*,            \
      constant VParams&, constant MLAHeadParams&, uint2, uint, ushort, ushort);
VNQH_INST(2, 4) VNQH_INST(2, 8)

struct QmmHeadParams {
    uint M, N, K;
    uint ldx;    // row stride of x
    uint ldy;    // row stride of y
    uint xh, yh; // offsets (elements) between two heads in x and y
};

// qmm_t_nax3 (qmm_nax_v3.metal) with a head per tg.z: Y_h[M, N] = X_h[M, K] · W_hᵀ (4 bits, groups of 64), for prefill
// (M a multiple of 32·WM), with row strides ldx / ldy. The same body, so each head gives the bits of a qmm_t_nax3 over
// its own matrix.
template <int WM, int WN, int BK, typename OT>
[[kernel, max_total_threads_per_threadgroup(32 * WM * WN)]] void qmm_t_nax3_heads(
    const device half*   x   [[buffer(0)]],
    const device uint*   w   [[buffer(1)]],
    const device bfloat* s   [[buffer(2)]],
    const device bfloat* b   [[buffer(3)]],
    device OT*           y   [[buffer(4)]],
    constant QmmHeadParams& p [[buffer(5)]],
    uint3  tgid [[threadgroup_position_in_grid]],
    uint   tid  [[thread_index_in_threadgroup]],
    ushort sgid [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]])
{
    constexpr int NT = 32 * WM * WN, BM = 32 * WM, BN = 32 * WN, LD = BK + 8;
    using AS = NaxAStageN<BM, BK, NT>;
    threadgroup half ws[BN * LD];
    threadgroup half xs[BM * LD];
    const uint h = tgid.z;
    x += size_t(h) * p.xh;
    y += size_t(h) * p.yh;
    w += size_t(h) * p.N * (p.K / 8);
    s += size_t(h) * p.N * (p.K / 64);
    b += size_t(h) * p.N * (p.K / 64);
    const uint r0 = tgid.y * BM, n0 = tgid.x * BN;
    array<const device half*, AS::NI> rp;
    NAX_UNROLL for (int i = 0; i < AS::NI; i++) rp[i] = x + size_t(r0 + (tid + i * NT) / AS::CPR) * p.ldx;
    // down3_body takes the row stride of y as its N.
    down3_body<4, 64, WM, WN, 1, true, BK, OT, false>(w, s, b, y, rp, ws, xs, p.K, p.ldy, n0, r0, BM,
                                                      (sgid / WN) * 32, (sgid % WN) * 32, nax_coord(lane), tid);
}

#define QMMH_INST(WM, WN, BK)                                                                                      \
  template [[host_name("qmm_t_nax3_heads_w" #WM "x" #WN "_bk" #BK "_f32")]] [[kernel]] void                       \
  qmm_t_nax3_heads<WM, WN, BK, float>(const device half*, const device uint*, const device bfloat*,               \
      const device bfloat*, device float*, constant QmmHeadParams&, uint3, uint, ushort, ushort);                  \
  template [[host_name("qmm_t_nax3_heads_w" #WM "x" #WN "_bk" #BK "_h")]] [[kernel]] void                         \
  qmm_t_nax3_heads<WM, WN, BK, half>(const device half*, const device uint*, const device bfloat*,                \
      const device bfloat*, device half*, constant QmmHeadParams&, uint3, uint, ushort, ushort);
QMMH_INST(2, 2, 32) QMMH_INST(2, 2, 64)

#endif  // LAB_AT_HD == 576
