// GLM-4.7-Flash MLA, decompressed prefill attention for long chunks (Layers/MLA.swift, `MLAProjection.decompressed`).
// Compiled only with the latent attention head (AT_HD 576). Concatenated after verify_attn*.metal and zm_mla.metal.
//
// The absorbed form costs 576 + 512 = 1,088 MAC per row, key and head. With many rows per chunk it is cheaper to
// decompress the latent keys once per chunk (k_nope_h = embed_qᵀ_h c, v_h = unembed_out_h c: 4.59 M MAC per key) and
// run a 20-head attention with a 256 head ([q_nope | q_pe] · [k_nope | k_pe]: 192 + 64; P·V over 256): 512 MAC.
// - mla_dec_weights: W' = [embed_qᵀ_h ; unembed_out_h] · H₅₁₂ per head ([H · 448, 512] half), dequantized from the
//   4-bit checkpoint matrices and rotated so it multiplies the cache's rotated latent (H₅₁₂ c) directly
//   (H₅₁₂ is orthonormal and symmetric);
// - mla_dec_kv: one block of keys [kb0, kb0 + n): Y = vs · (c8 · W'ᵀ) → k_nope [H, blk, 192] and v [H, blk, 256]
//   in half (the block is a multiple of 128 keys; keys past the written ones read the last one, so they are finite);
// - pattn_mla_dec: flash attention of one key block for every row of the chunk, one head per tg.y (G = 1); k_pe is
//   read int8 from kc (dims 0…63, H₆₄-rotated as q_pe, scale ks). Blocks after the first resume from the running
//   state (O fp32, max and sum per row and head) and the last one writes the normalized output straight into
//   o_proj's input ([rows, H · 256]), with no unembed_out.
// The block of keys bounds the working memory (the whole decompressed cache would be 17.9 KB per token and layer).
#if LAB_AT_HD == 576

struct MLADecParams {
    uint B;      // rows of the chunk
    uint ctx;    // keys before the chunk
    uint kb0;    // first key of the block (multiple of 128)
    uint kb1;    // end of the block (multiple of 128; may pass ctx + B)
    uint blk;    // key stride of the k_nope / v scratch (keys of a block)
    uint flags;  // 1: first block (no state to resume), 2: last block (write the output)
    uint ldkv;   // stride of kc
    float scale; // score scale
};

constant constexpr uint MD_H = AT_QH, MD_NOPE = 192, MD_V = 256, MD_R = 512, MD_W = MD_NOPE + MD_V;

// Grid (MD_W, AT_QH), AT_HD threads (at_fwht's shape; threads 0…63 carry zeros). Row r of head h of W' is
// H₅₁₂ · (column r of embed_q_h) for r < 192 and H₅₁₂ · (row r − 192 of unembed_out_h) otherwise.
[[kernel]] void mla_dec_weights(const device uint*   ew [[buffer(0)]],   // embed_q [H · 512, 192], 4 bits g64
                                const device bfloat* es [[buffer(1)]],
                                const device bfloat* eb [[buffer(2)]],
                                const device uint*   uw [[buffer(3)]],   // unembed_out [H · 256, 512], 4 bits g64
                                const device bfloat* us [[buffer(4)]],
                                const device bfloat* ub [[buffer(5)]],
                                device half*         wp [[buffer(6)]],   // [H · 448, 512]
                                uint2 tg [[threadgroup_position_in_grid]],
                                uint d [[thread_index_in_threadgroup]],
                                ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float xs[AT_HD];
    const uint r = tg.x, h = tg.y;
    float x = 0;
    if (d >= 64) {
        const uint n = d - 64;
        if (r < MD_NOPE) {
            const uint row = h * MD_R + n, i = row * MD_NOPE + r, gi = row * (MD_NOPE / 64) + r / 64;
            x = float((ew[i / 8] >> (4 * (i % 8))) & 15u) * float(es[gi]) + float(eb[gi]);
        } else {
            const uint row = h * MD_V + r - MD_NOPE, i = row * MD_R + n, gi = row * (MD_R / 64) + n / 64;
            x = float((uw[i / 8] >> (4 * (i % 8))) & 15u) * float(us[gi]) + float(ub[gi]);
        }
    }
    const float v = at_fwht(x, xs, d, lane);
    if (d >= 64) wp[(size_t(h) * MD_W + r) * MD_R + d - 64] = half(v);
}

// Grid (MD_W / 64, blk / 64, AT_QH), 128 threads: a 64 keys × 64 outputs tile, each simdgroup 32 × 32, K = 512.
[[kernel]] void mla_dec_kv(const device char*  kc [[buffer(0)]],   // [ldkv, 576]: dims 64… = H₅₁₂ c (int8)
                           const device float* vs [[buffer(1)]],   // [ldkv] scale of c
                           const device half*  wp [[buffer(2)]],   // [H · 448, 512]
                           device half*        kn [[buffer(3)]],   // [H, blk, 192]
                           device half*        vv [[buffer(4)]],   // [H, blk, 256]
                           constant MLADecParams& p [[buffer(5)]],
                           uint3 tg [[threadgroup_position_in_grid]],
                           ushort sg [[simdgroup_index_in_threadgroup]],
                           ushort lane [[thread_index_in_simdgroup]]) {
    const short2 co = nax_coord(lane);
    const uint h = tg.z, n0 = tg.x * 64 + (sg % 2) * 32, kr0 = tg.y * 64 + (sg / 2) * 32;   // key rows in the block
    const uint klast = p.ctx + p.B - 1;
    // Rows of the A fragments (int8 latent of the keys), clamped to the last written key.
    const device char* a[2][2];
    NAX_UNROLL for (int i = 0; i < 2; i++) NAX_UNROLL for (int e = 0; e < 2; e++) {
        const uint key = min(p.kb0 + kr0 + 16 * i + co.x + 8 * e, klast);
        a[i][e] = kc + size_t(key) * AT_HD + 64 + co.y;
    }
    const device half* w = wp + (size_t(h) * MD_W + n0) * MD_R;
    nax_ffrag acc[2][2];
    NAX_UNROLL for (int i = 0; i < 2; i++) NAX_UNROLL for (int j = 0; j < 2; j++) acc[i][j] = 0;
    PF_UNROLL2 for (uint k = 0; k < MD_R; k += 16) {
        nax_hfrag fa[2], fb[2];
        NAX_UNROLL for (int i = 0; i < 2; i++)
            fa[i] = nax_hfrag(half4(*(const device char4*)(a[i][0] + k)), half4(*(const device char4*)(a[i][1] + k)));
        NAX_UNROLL for (int j = 0; j < 2; j++) nax_load(fb[j], w + size_t(16 * j) * MD_R + k, MD_R, co);
        NAX_UNROLL for (int i = 0; i < 2; i++) nax_mma2(acc[i][0], acc[i][1], fa[i], fb[0], fb[1]);
    }
    NAX_UNROLL for (int i = 0; i < 2; i++) NAX_UNROLL for (int e = 0; e < 2; e++) {
        const uint kr = kr0 + 16 * i + co.x + 8 * e;
        const float sc = vs[min(p.kb0 + kr, klast)];
        NAX_UNROLL for (int j = 0; j < 2; j++) {
            const uint c = n0 + 16 * j + co.y;   // output column in the head (0…447), 4 of them
            const half4 y = half4(float4(acc[i][j][4 * e], acc[i][j][4 * e + 1], acc[i][j][4 * e + 2],
                                         acc[i][j][4 * e + 3]) * sc);
            if (c < MD_NOPE) *(device half4*)(kn + (size_t(h) * p.blk + kr) * MD_NOPE + c) = y;
            else *(device half4*)(vv + (size_t(h) * p.blk + kr) * MD_V + c - MD_NOPE) = y;
        }
    }
}

// Flash attention over one block of decompressed keys (layout of pattn_i8 with G = 1: RG groups of 16·RF rows per
// threadgroup, the 4 simdgroups of a group split the keys of each 128 step for S and the 256 dims for P·V). Each
// simdgroup carries RF fragments of 16 rows, so every K and V fragment it loads feeds RF matmuls: with one KV head per
// query head nothing else shares them, and the kernel is bound by those loads. 2,048 rows over 16K and 32K keys
// (RG, RF): (2, 1) 4.4 TFLOPS, (4, 1) 8.3, (2, 2) 9.0, (2, 3) 8.7, (1, 4) 6.9, (3, 2) 10.3-10.9 (threadgroup memory
// caps RG · RF at 6).
// Grid (ceil(B / (16·RF·RG)), AT_QH), 128·RG threads.
template <int RG, int RF>
[[kernel]] void pattn_mla_dec(const device half*  qn    [[buffer(0)]],   // q_nope [B, H · 192]
                              const device half*  q     [[buffer(1)]],   // prework q [B · H, 576]: dims 0…63 = H₆₄ q_pe
                              const device char*  kc    [[buffer(2)]],   // [ldkv, 576]: dims 0…63 = H₆₄ k_pe (int8)
                              const device float* ks    [[buffer(3)]],   // [ldkv] scale of k_pe
                              const device half*  kn    [[buffer(4)]],   // [H, blk, 192]
                              const device half*  vv    [[buffer(5)]],   // [H, blk, 256]
                              device float*       st    [[buffer(6)]],   // running O [H, B, 256]
                              device float2*      ml    [[buffer(7)]],   // running (max, sum) [H, B]
                              device half*        out   [[buffer(8)]],   // [B, H · 256]
                              constant MLADecParams& p  [[buffer(9)]],
                              uint3 tg [[threadgroup_position_in_grid]],
                              ushort sg [[simdgroup_index_in_threadgroup]],
                              ushort lane [[thread_index_in_simdgroup]]) {
    constexpr int KB = 128, PL = KB + 8, PV_F = 4, GR = 16 * RF;
    threadgroup half pbuf[RG][GR * PL];
    threadgroup float redm[RG][4][GR];
    const uint h = tg.y, Q = p.B, ctx = p.ctx;
    const uint g = sg / 4, j = sg % 4;
    const uint row0 = tg.x * (GR * RG);
    const uint rlast = min(row0 + GR * RG - 1, Q - 1);
    const uint kend = min(ctx + rlast + 1, p.kb1);
    const uint kdiag = ctx + row0;
    const short2 co = nax_coord(lane);
    const uint fm = co.x, fn = co.y;
    const device half* knh = kn + size_t(h) * p.blk * MD_NOPE;
    const device half* vh = vv + size_t(h) * p.blk * MD_V;
    const uint d0 = 64 * j;
    // Rows of the thread: fragment r, half i → row m[r][i] (fm or fm + 8 of the fragment).
    uint m[RF][2], cr[RF][2];
    bool val[RF][2];
    nax_ffrag o[RF][PV_F];
    float mr[RF][2], l[RF][2];
    NAX_UNROLL for (int r = 0; r < RF; r++) NAX_UNROLL for (int i = 0; i < 2; i++) {
        m[r][i] = row0 + GR * g + 16 * r + fm + 8 * i;
        val[r][i] = m[r][i] < Q;
        cr[r][i] = min(m[r][i], Q - 1);
        mr[r][i] = -INFINITY; l[r][i] = 0;
    }
    if (p.flags & 1u) {
        NAX_UNROLL for (int r = 0; r < RF; r++) NAX_UNROLL for (int f = 0; f < PV_F; f++) o[r][f] = 0;
    } else {
        NAX_UNROLL for (int r = 0; r < RF; r++) {
            const device float* s0p = st + (size_t(h) * Q + cr[r][0]) * MD_V + d0 + fn;
            const device float* s1p = st + (size_t(h) * Q + cr[r][1]) * MD_V + d0 + fn;
            NAX_UNROLL for (int f = 0; f < PV_F; f++)
                o[r][f] = nax_ffrag(*(const device float4*)(s0p + 16 * f), *(const device float4*)(s1p + 16 * f));
            NAX_UNROLL for (int i = 0; i < 2; i++) {
                const float2 sv = ml[size_t(h) * Q + cr[r][i]];
                // The sum is carried whole by simdgroup 0 (all share the running max; the parts add up at the end).
                mr[r][i] = sv.x; l[r][i] = j == 0 ? sv.y : 0;
            }
        }
    }
    threadgroup half* pb = pbuf[g];
    for (uint k0 = p.kb0; k0 < kend; k0 += KB) {
        const uint kj = k0 + 32 * j, bj = kj - p.kb0;   // S keys of this simdgroup; their row in the block
        nax_ffrag s[RF][2], e[RF][2];
        NAX_UNROLL for (int r = 0; r < RF; r++) { s[r][0] = 0; s[r][1] = 0; e[r][0] = 0; e[r][1] = 0; }
        PF_UNROLL2 for (int c = 0; c < int(MD_NOPE / 16); c++) {
            nax_hfrag kb0, kb1;
            nax_load(kb0, knh + size_t(bj) * MD_NOPE + 16 * c, MD_NOPE, co);
            nax_load(kb1, knh + size_t(bj + 16) * MD_NOPE + 16 * c, MD_NOPE, co);
            NAX_UNROLL for (int r = 0; r < RF; r++) {
                const device half* q0 = qn + size_t(cr[r][0]) * MD_H * MD_NOPE + h * MD_NOPE + fn + 16 * c;
                const device half* q1 = qn + size_t(cr[r][1]) * MD_H * MD_NOPE + h * MD_NOPE + fn + 16 * c;
                const nax_hfrag qc = nax_hfrag(*(const device half4*)q0, *(const device half4*)q1);
                nax_mma2(s[r][0], s[r][1], qc, kb0, kb1);
            }
        }
        NAX_UNROLL for (int c = 0; c < 4; c++) {
            nax_hfrag kb0, kb1;
            nax_load_i8(kb0, kc + size_t(kj) * AT_HD + 16 * c, AT_HD, co);
            nax_load_i8(kb1, kc + size_t(kj + 16) * AT_HD + 16 * c, AT_HD, co);
            NAX_UNROLL for (int r = 0; r < RF; r++) {
                const device half* q0 = q + (size_t(cr[r][0]) * MD_H + h) * AT_HD + fn + 16 * c;
                const device half* q1 = q + (size_t(cr[r][1]) * MD_H + h) * AT_HD + fn + 16 * c;
                const nax_hfrag qc = nax_hfrag(*(const device half4*)q0, *(const device half4*)q1);
                nax_mma2(e[r][0], e[r][1], qc, kb0, kb1);
            }
        }
        const bool diag = k0 + KB > kdiag;
        float sv[RF][16];
        float mx[RF][2];
        float4 ksv[2];
        NAX_UNROLL for (int f = 0; f < 2; f++) ksv[f] = *(const device float4*)(ks + kj + 16 * f + fn);
        NAX_UNROLL for (int r = 0; r < RF; r++) {
            mx[r][0] = -INFINITY; mx[r][1] = -INFINITY;
            NAX_UNROLL for (int f = 0; f < 2; f++) NAX_UNROLL for (int x = 0; x < 8; x++) {
                float v = (s[r][f][x] + e[r][f][x] * ksv[f][x % 4]) * p.scale;
                if (diag) {
                    const uint key = kj + 16 * f + fn + x % 4;
                    v = key <= ctx + m[r][x / 4] && key < kend ? v : -INFINITY;
                }
                sv[r][f * 8 + x] = v;
                mx[r][x / 4] = max(mx[r][x / 4], v);
            }
            NAX_UNROLL for (int i = 0; i < 2; i++) {
                mx[r][i] = max(mx[r][i], simd_shuffle_xor(mx[r][i], 1));
                mx[r][i] = max(mx[r][i], simd_shuffle_xor(mx[r][i], 8));
            }
            if ((lane & 9) == 0) { redm[g][j][16 * r + fm] = mx[r][0]; redm[g][j][16 * r + fm + 8] = mx[r][1]; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        bool same = true;
        float a[RF][2];
        NAX_UNROLL for (int r = 0; r < RF; r++) {
            float mn[2];
            NAX_UNROLL for (int i = 0; i < 2; i++) {
                const uint x = 16 * r + fm + 8 * i;
                const float t = max(max(redm[g][0][x], redm[g][1][x]), max(redm[g][2][x], redm[g][3][x]));
                mn[i] = max(mr[r][i], t);
                a[r][i] = mn[i] == -INFINITY ? 1.0f : exp(mr[r][i] - mn[i]);
                same = same && a[r][i] == 1.0f;
            }
            float rs[2] = {0, 0};
            NAX_UNROLL for (int f = 0; f < 2; f++) {
                half4 h0, h1;
                NAX_UNROLL for (int x = 0; x < 4; x++) {
                    const float p0 = mn[0] == -INFINITY ? 0.0f : exp(sv[r][f * 8 + x] - mn[0]);
                    const float p1 = mn[1] == -INFINITY ? 0.0f : exp(sv[r][f * 8 + 4 + x] - mn[1]);
                    rs[0] += p0; rs[1] += p1;
                    h0[x] = half(p0); h1[x] = half(p1);
                }
                *(threadgroup half4*)(pb + (16 * r + fm) * PL + 32 * j + 16 * f + fn) = h0;
                *(threadgroup half4*)(pb + (16 * r + fm + 8) * PL + 32 * j + 16 * f + fn) = h1;
            }
            NAX_UNROLL for (int i = 0; i < 2; i++) {
                rs[i] += simd_shuffle_xor(rs[i], 1); rs[i] += simd_shuffle_xor(rs[i], 8);
                l[r][i] = l[r][i] * a[r][i] + rs[i];
                mr[r][i] = mn[i];
            }
        }
        if (!simd_all(same)) {
            NAX_UNROLL for (int r = 0; r < RF; r++) NAX_UNROLL for (int f = 0; f < PV_F; f++)
                NAX_UNROLL for (int x = 0; x < 8; x++) o[r][f][x] *= a[r][x / 4];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        PF_UNROLL2 for (int kc2 = 0; kc2 < KB / 16; kc2++) {
            const device half* vk = vh + size_t(k0 - p.kb0 + 16 * kc2) * MD_V + d0;
            nax_hfrag vb[PV_F];
            NAX_UNROLL for (int f = 0; f < PV_F; f++) nax_load(vb[f], vk + 16 * f, MD_V, co);
            NAX_UNROLL for (int r = 0; r < RF; r++) {
                nax_hfrag pa;
                nax_load(pa, pb + 16 * r * PL + 16 * kc2, PL, co);
                NAX_UNROLL for (int f = 0; f < PV_F; f += 2) nax_mma2_kn(o[r][f], o[r][f + 1], pa, vb[f], vb[f + 1]);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if ((lane & 9) == 0) {
        NAX_UNROLL for (int r = 0; r < RF; r++) {
            redm[g][j][16 * r + fm] = l[r][0]; redm[g][j][16 * r + fm + 8] = l[r][1];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    NAX_UNROLL for (int r = 0; r < RF; r++) NAX_UNROLL for (int i = 0; i < 2; i++) {
        const uint x = 16 * r + fm + 8 * i;
        const float L = (redm[g][0][x] + redm[g][1][x]) + (redm[g][2][x] + redm[g][3][x]);
        if (!val[r][i]) continue;
        const uint t = m[r][i];
        if (!(p.flags & 2u)) {
            device float* sp = st + (size_t(h) * Q + t) * MD_V + d0 + fn;
            NAX_UNROLL for (int f = 0; f < PV_F; f++)
                *(device float4*)(sp + 16 * f) =
                    float4(o[r][f][4 * i], o[r][f][4 * i + 1], o[r][f][4 * i + 2], o[r][f][4 * i + 3]);
            if (j == 0 && (lane & 9) == 0) ml[size_t(h) * Q + t] = float2(mr[r][i], L);
        } else {
            device half* op = out + size_t(t) * MD_H * MD_V + h * MD_V + d0 + fn;
            NAX_UNROLL for (int f = 0; f < PV_F; f++)
                *(device half4*)(op + 16 * f) =
                    half4(float4(o[r][f][4 * i], o[r][f][4 * i + 1], o[r][f][4 * i + 2], o[r][f][4 * i + 3]) / L);
        }
    }
}

#define PMLA_DEC_INST(RG, RF) \
    template [[host_name("pattn_mla_dec_" #RG "_" #RF)]] [[kernel]] void pattn_mla_dec<RG, RF>( \
        const device half*, const device half*, const device char*, const device float*, const device half*, \
        const device half*, device float*, device float2*, device half*, constant MLADecParams&, uint3, ushort, ushort);
PMLA_DEC_INST(3, 2)

#endif
