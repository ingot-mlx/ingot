// Sparse attention over block lists. Concatenated after verify_attn_pf.metal (uses AttnParams, nax_*,
// PF_UNROLL2 and the split scheme of pattn_i8).
//
// Contract between the selector and the attention (BlockList in Swift, BlockList.swift):
// - one list per tile of T consecutive query rows (all the heads of its GQA group) and, with perKVHead, per
//   KV head: lists [tiles · (perKVHead ? AT_KVH : 1)][stride] uint32, counts [same index] uint32;
// - entry = block << T | T-bit mask (bit r: row r of the tile attends the block), blocks of BT tokens in
//   ascending order and without repeats;
// - the list includes the open block of each row (the one at its position): the kernel only adds the causal mask
//   (key ≤ position of the row);
// - the split into chunks depends only on the list: chunk s walks the entries [s·E, (s+1)·E) (E = fixed perSplit).
// The selector knows nothing about the kernel nor the kernel about the selector: the QSA writes its bitmap and
// blocklist_from_bitmap converts it; another selector (DSA, NSA, MoBA) would write lists, same kernel and reduction.
struct SparseListParams {
    uint stride;     // maximum entries per list
    uint tiles;      // tiles (verification with T = 1: rows of the round)
    uint perKVHead;  // 0: one list per tile for all KV heads; 1: [tile][KV head]
    uint perSplit;   // E: entries per chunk, fixed (best a multiple of 128 / BT, the entries of one step)
};

// Per-row bitmap ([rows, nw]; bit b = complete block b of 4 tokens, the one of qsa_m_select) → lists with T = 1.
struct BlockListBuild {
    uint nw;      // words per row of the bitmap
    uint pos0;    // position of row 0 (without AT_DYN_ST in flags)
    uint stride;  // entries per list
    uint flags;   // bit 0: the position of row 0 is read from st[0] (chained rounds)
};

// One threadgroup of 256 threads per row: each thread counts the bits of its span of words (only the complete blocks
// of the row, < (pos + 1) / 4), prefix sum across threads and write in ascending order; at the end, the open block
// (pos + 1) / 4 if it has some key ≤ pos ((pos + 1) % 4 ≠ 0). Same semantics as the pattn_i8 SPARSE mask: a
// row sees the marked complete blocks and the keys ≤ pos of its open block.
[[kernel]] void blocklist_from_bitmap(const device uint*     bm  [[buffer(0)]],
                                      device uint*           lst [[buffer(1)]],
                                      device uint*           cnt [[buffer(2)]],
                                      constant BlockListBuild& bp [[buffer(3)]],
                                      const device uint*     st  [[buffer(4)]],
                                      uint   r    [[threadgroup_position_in_grid]],
                                      uint   t    [[thread_index_in_threadgroup]],
                                      ushort sg   [[simdgroup_index_in_threadgroup]],
                                      ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup uint tot[8];
    const uint pos = ((bp.flags & 1u) ? st[0] : bp.pos0) + r;
    const uint nfull = (pos + 1) / 4;
    const uint per = (bp.nw + 255) / 256, w0 = min(t * per, bp.nw), w1 = min(w0 + per, bp.nw);
    const device uint* row = bm + size_t(r) * bp.nw;
    auto word = [&](uint w) -> uint {
        const uint lo = w * 32;
        if (lo >= nfull) return 0u;
        const uint x = row[w];
        return nfull - lo < 32 ? x & ((1u << (nfull - lo)) - 1u) : x;
    };
    uint c = 0;
    for (uint w = w0; w < w1; w++) c += popcount(word(w));
    const uint ex = simd_prefix_exclusive_sum(c);
    if (lane == 31) tot[sg] = ex + c;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint base = ex, total = 0;
    for (uint s = 0; s < 8; s++) {
        base += s < sg ? tot[s] : 0u;
        total += tot[s];
    }
    device uint* L = lst + size_t(r) * bp.stride;
    uint i = base;
    for (uint w = w0; w < w1; w++) {
        uint x = word(w);
        while (x != 0u) {
            const uint b = ctz(x);
            if (i < bp.stride) L[i] = ((w * 32 + b) << 1) | 1u;
            i++;
            x &= x - 1u;
        }
    }
    if (t == 0) {
        uint n = total;
        if ((pos + 1) % 4 != 0 && n < bp.stride) L[n++] = (nfull << 1) | 1u;
        cnt[r] = min(n, bp.stride);
    }
}

// Verification over lists with T = 1 (one list per row) and blocks of BT = 4 tokens. Grid (chunk, KV head,
// row), 128 threads: a group of 16 queries with the AT_G heads of the row (the padding lanes repeat the last
// head and do not write). Like pattn_i8 DEC, but each step is 32 list entries (128 chosen keys) instead of
// a window of 128 positions:
// - S: simdgroup j does entries 8j…8j + 7 of the step (32 keys); each lane loads its 4 key rows with a
//   pointer per row (like the ROWG path of the tree), and at the output each lane has 4 consecutive keys = one block,
//   with their scales in a float4;
// - P·V: each fragment of 16 keys is 4 entries; each lane reads a char4 per dimension row of the transposed V
//   (vh + dim·ldkv + block·4), which is a whole block;
// - the entries missing at the end of the chunk (≥ e1) are read from block 0 and left at −inf (P = 0); the simdgroup
//   whose 8 entries are all missing skips its Q·Kᵀ, and the missing 16-key fragments skip their P·V (adds +0 to O).
// What a row computes depends only on its list and on E: B = 1 and B = 16 give bit-identical results. The sum order
// differs from pattn_i8 SPARSE (the steps are different): in the sparse regime the logits change by rounding.
template <int BT>
[[kernel]] void dattn_i8l(const device half*  q     [[buffer(0)]],   // [AT_KVH, AT_G·B, AT_HD]
                          const device char*  kc    [[buffer(1)]],   // [AT_KVH, ldkv, AT_HD]
                          const device char*  vt    [[buffer(2)]],   // [AT_KVH, AT_HD, ldkv]
                          const device float* ks    [[buffer(3)]],   // [AT_KVH, ldkv]
                          const device float* vs    [[buffer(4)]],   // [AT_KVH, ldkv]
                          constant AttnParams& p    [[buffer(7)]],
                          device float*       opart [[buffer(8)]],   // [AT_KVH, nsplit, AT_G·B, AT_HD]
                          device float2*      ml    [[buffer(9)]],   // [AT_KVH, nsplit, AT_G·B]
                          const device uint*  st    [[buffer(10)]],  // chained decode: st[0] = position
                          const device uint*  lst   [[buffer(14)]],
                          const device uint*  cnt   [[buffer(15)]],
                          constant SparseListParams& lp [[buffer(16)]],
                          uint3 tg [[threadgroup_position_in_grid]],
                          ushort sg [[simdgroup_index_in_threadgroup]],
                          ushort lane [[thread_index_in_simdgroup]]) {
    static_assert(BT == 4, "only 4-token blocks: one lane = one block in S and in P·V");
    constexpr int KB = 128, PL = KB + 8, EP = KB / BT;   // EP: entries per step
    constexpr uint LT = 1;                               // T: mask bits in the entry
    threadgroup half pb[16 * PL];
    threadgroup float redm[4][16];
    const uint split = tg.x, kvh = tg.y, row = tg.z, Q = AT_G * p.B, j = sg;
    const uint ctx = (p.dyn & AT_DYN_ST) ? st[0] : p.ctx;
    const uint lim = ctx + row;   // causal: the row sees the keys ≤ its position
    const short2 co = nax_coord(lane);
    const uint fm = co.x, fn = co.y;
    const uint m0 = row * AT_G + fm, m1 = m0 + 8;
    const bool val0 = fm < AT_G, val1 = fm + 8 < AT_G;
    const uint mq0 = row * AT_G + min(uint(fm), AT_G - 1), mq1 = row * AT_G + min(uint(fm) + 8, AT_G - 1);
    const uint li = lp.perKVHead ? row * AT_KVH + kvh : row;
    const device uint* L = lst + size_t(li) * lp.stride;
    const uint n = min(cnt[li], lp.stride);
    const uint e0 = split * lp.perSplit, e1 = min(e0 + lp.perSplit, n);
    const device half* qh = q + size_t(kvh) * Q * AT_HD;
    const device char* kh = kc + size_t(kvh) * p.ldkv * AT_HD;
    const device char* vh = vt + size_t(kvh) * AT_HD * p.ldkv;
    const device float* ksh = ks + size_t(kvh) * p.ldkv;
    const device float* vsh = vs + size_t(kvh) * p.ldkv;
    const device half* q0 = qh + size_t(mq0) * AT_HD + fn;
    const device half* q1 = qh + size_t(mq1) * AT_HD + fn;
    // First key of the block of entry e (the missing ones, block 0: they are read and left masked).
    auto kbase = [&](uint e) -> uint { return e < e1 ? (L[e] >> LT) * BT : 0u; };
    constexpr int PV_F = AT_HD / 64;
    nax_ffrag o[PV_F];
    NAX_UNROLL for (int f = 0; f < PV_F; f++) o[f] = 0;
    float mr0 = -INFINITY, mr1 = -INFINITY, l0 = 0, l1 = 0;
    const uint d0 = (AT_HD / 4) * j;
    for (uint eb = e0; eb < e1; eb += EP) {
        const uint ej = eb + 8 * j;   // S entries of this simdgroup
        nax_ffrag s0 = 0, s1 = 0;
        if (ej < e1) {
            // Keys of rows fm, fm + 8, 16 + fm and 24 + fm of the K fragments: entries ej + fm/4 + {0, 2, 4, 6}.
            const uint fo = fm % 4, fe = ej + fm / 4;
            const device char* r0 = kh + size_t(kbase(fe) + fo) * AT_HD + fn;
            const device char* r1 = kh + size_t(kbase(fe + 2) + fo) * AT_HD + fn;
            const device char* r2 = kh + size_t(kbase(fe + 4) + fo) * AT_HD + fn;
            const device char* r3 = kh + size_t(kbase(fe + 6) + fo) * AT_HD + fn;
            PF_UNROLL2 for (int c = 0; c < int(AT_HD / 16); c++) {
                const nax_hfrag qc = nax_hfrag(*(const device half4*)(q0 + 16 * c), *(const device half4*)(q1 + 16 * c));
                const nax_hfrag kb0 = nax_hfrag(half4(*(const device char4*)(r0 + 16 * c)), half4(*(const device char4*)(r1 + 16 * c)));
                const nax_hfrag kb1 = nax_hfrag(half4(*(const device char4*)(r2 + 16 * c)), half4(*(const device char4*)(r3 + 16 * c)));
                nax_mma2(s0, s1, qc, kb0, kb1);
            }
        }
        // S output: the lane has keys 16f + fn … + 3 of its simdgroup = entry ej + 4f + fn/4.
        uint kbs[2];
        float sv[16];
        float mx0 = -INFINITY, mx1 = -INFINITY;
        NAX_UNROLL for (int f = 0; f < 2; f++) {
            const uint ee = ej + 4 * f + fn / 4;
            const bool ok = ee < e1;
            const uint kb = kbase(ee);
            kbs[f] = kb;
            const float4 ksv = *(const device float4*)(ksh + kb) * p.scale;
            NAX_UNROLL for (int e = 0; e < 8; e++) {
                float v = (f == 0 ? s0 : s1)[e] * ksv[e % 4];
                v = ok && kb + e % 4 <= lim ? v : -INFINITY;
                sv[f * 8 + e] = v;
                if (e < 4) mx0 = max(mx0, v); else mx1 = max(mx1, v);
            }
        }
        mx0 = max(mx0, simd_shuffle_xor(mx0, 1)); mx0 = max(mx0, simd_shuffle_xor(mx0, 8));
        mx1 = max(mx1, simd_shuffle_xor(mx1, 1)); mx1 = max(mx1, simd_shuffle_xor(mx1, 8));
        if ((lane & 9) == 0) { redm[j][fm] = mx0; redm[j][fm + 8] = mx1; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float t0 = max(max(redm[0][fm], redm[1][fm]), max(redm[2][fm], redm[3][fm]));
        const float t1 = max(max(redm[0][fm + 8], redm[1][fm + 8]), max(redm[2][fm + 8], redm[3][fm + 8]));
        const float mn0 = max(mr0, t0), mn1 = max(mr1, t1);
        const float a0 = mn0 == -INFINITY ? 1.0f : exp(mr0 - mn0), a1 = mn1 == -INFINITY ? 1.0f : exp(mr1 - mn1);
        float rs0 = 0, rs1 = 0;
        NAX_UNROLL for (int f = 0; f < 2; f++) {
            const float4 vsv = *(const device float4*)(vsh + kbs[f]);
            half4 h0, h1;
            NAX_UNROLL for (int e = 0; e < 4; e++) {
                const float p0 = mn0 == -INFINITY ? 0.0f : exp(sv[f * 8 + e] - mn0);
                const float p1 = mn1 == -INFINITY ? 0.0f : exp(sv[f * 8 + 4 + e] - mn1);
                rs0 += p0; rs1 += p1;
                h0[e] = half(p0 * vsv[e]); h1[e] = half(p1 * vsv[e]);
            }
            *(threadgroup half4*)(pb + fm * PL + 32 * j + 16 * f + fn) = h0;
            *(threadgroup half4*)(pb + (fm + 8) * PL + 32 * j + 16 * f + fn) = h1;
        }
        rs0 += simd_shuffle_xor(rs0, 1); rs0 += simd_shuffle_xor(rs0, 8);
        rs1 += simd_shuffle_xor(rs1, 1); rs1 += simd_shuffle_xor(rs1, 8);
        l0 = l0 * a0 + rs0; l1 = l1 * a1 + rs1;
        mr0 = mn0; mr1 = mn1;
        if (!simd_all(a0 == 1.0f && a1 == 1.0f)) {
            NAX_UNROLL for (int f = 0; f < PV_F; f++) NAX_UNROLL for (int e = 0; e < 8; e++) o[f][e] *= (e < 4 ? a0 : a1);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // P·V: fragment kc2 is entries eb + 4·kc2 … + 3; the lane reads the 4 keys of block eb + 4·kc2 + fn/4.
        PF_UNROLL2 for (int kc2 = 0; kc2 < KB / 16; kc2++) {
            if (eb + 4 * kc2 >= e1) continue;   // missing entries: P = 0
            const uint kb = kbase(eb + 4 * kc2 + fn / 4);
            nax_hfrag pa;
            nax_load(pa, pb + 16 * kc2, PL, co);
            auto loadv = [&](thread nax_hfrag& d, uint dim) {
                d = nax_hfrag(half4(*(const device char4*)(vh + size_t(dim + fm) * p.ldkv + kb)),
                              half4(*(const device char4*)(vh + size_t(dim + fm + 8) * p.ldkv + kb)));
            };
            if constexpr (PV_F == 1) {
                nax_hfrag vb0;
                loadv(vb0, d0);
                nax_ffrag dummy = 0;
                nax_mma2(o[0], dummy, pa, vb0, vb0);
            } else {
#if LAB_AT_HD == 576
                // MLA's latent head (9 fragments): four pairs and the last one alone.
                NAX_UNROLL for (int f = 0; f + 1 < PV_F; f += 2) {
                    nax_hfrag vb0, vb1;
                    loadv(vb0, d0 + 16 * f);
                    loadv(vb1, d0 + 16 * f + 16);
                    nax_mma2(o[f], o[f + 1], pa, vb0, vb1);
                }
                nax_hfrag vt0;
                loadv(vt0, d0 + 16 * (PV_F - 1));
                nax_ffrag dummy = 0;
                nax_mma2(o[PV_F - 1], dummy, pa, vt0, vt0);
#else
                NAX_UNROLL for (int f = 0; f < PV_F; f += 2) {
                    nax_hfrag vb0, vb1;
                    loadv(vb0, d0 + 16 * f);
                    loadv(vb1, d0 + 16 * f + 16);
                    nax_mma2(o[f], o[f + 1], pa, vb0, vb1);
                }
#endif
            }
        }
    }
    // total l = sum of the 4 parts, in a fixed order (like pattn_i8).
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if ((lane & 9) == 0) { redm[j][fm] = l0; redm[j][fm + 8] = l1; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float L0 = (redm[0][fm] + redm[1][fm]) + (redm[2][fm] + redm[3][fm]);
    const float L1 = (redm[0][fm + 8] + redm[1][fm + 8]) + (redm[2][fm + 8] + redm[3][fm + 8]);
    device float* op = opart + (size_t(kvh) * p.nsplit + split) * Q * AT_HD;
    NAX_UNROLL for (int f = 0; f < PV_F; f++) NAX_UNROLL for (int i = 0; i < 2; i++) {
        const uint m = i == 0 ? m0 : m1;
        if (i == 0 ? val0 : val1) *(device float4*)(op + size_t(m) * AT_HD + d0 + 16 * f + fn) =
            float4(o[f][4 * i], o[f][4 * i + 1], o[f][4 * i + 2], o[f][4 * i + 3]);
    }
    if (j == 0 && (lane & 9) == 0) {
        device float2* mp = ml + (size_t(kvh) * p.nsplit + split) * Q;
        if (val0) mp[m0] = float2(mr0, L0);
        if (val1) mp[m1] = float2(mr1, L1);
    }
}

template [[host_name("dattn_i8l_1_4")]] [[kernel]] void dattn_i8l<4>(
    const device half*, const device char*, const device char*, const device float*, const device float*,
    constant AttnParams&, device float*, device float2*, const device uint*, const device uint*, const device uint*,
    constant SparseListParams&, uint3, ushort, ushort);
