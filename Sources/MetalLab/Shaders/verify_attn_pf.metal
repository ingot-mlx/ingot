// Flash attention on NAX with the decode int8 KV, for prefill (causal, one pass) and for decode/verification
// (key chunks + vattn_reduce). Concatenated after verify_attn_nax.metal (uses AttnParams, nax_* and nax_load_i8).
//
// Layout: a threadgroup carries RG groups of 16 queries (16 / AT_G tokens × the AT_G heads of its GQA group; GQA 8:
// 2 tokens); the 4 simdgroups of a group split the KEYS for S (each 32 keys of a step of 128, with all AT_HD
// dimensions, so S comes out complete without summing parts) and the DIMENSIONS for P·V (each AT_HD / 4 dimensions of
// O with the 128 keys: 64 with the 256 head, 32 with the 128 one, 16 with the 64 one).
// Per step only the per-row maxima (4 floats) and P in half (16×128) are exchanged through threadgroup memory, and each
// exponential is computed once. The sum l of each simdgroup covers its keys; the 4 are added at the end (all use the
// same maximum at each step).
//
// Measured (ctx 20480 + 2048 rows, 722 GFLOP): vattn_nax_i8 (since removed; 16 queries per threadgroup, 4
// simdgroups that split the dimensions and sum S through threadgroup memory, softmax repeated 4 times) 92 ms = 7.9
// TFLOPS; with that same split but one pass, mask only on the diagonal and lazy rescaling, 60 ms = 12 TFLOPS; this
// layout, 46 ms = 15.6 TFLOPS. Pitfall: with the loops of 16 (S) and 8 (P·V) steps fully unrolled, the compiler hoists
// all the loads, runs out of registers and the kernel is 1.5-4× slower: they are unrolled by 2. Q is re-read from
// memory at each step (it does not fit in registers next to O; from threadgroup memory it does not win). Neither more
// groups per threadgroup, nor the KV in fp16, nor a direct half×int8 matmul2d change anything: ALU work per step rules.
// The idea of reading each K/V chunk once for many queries (8 rows × 8 heads) and of int8 KV with the scale
// outside the matmuls: splash `paged_attention_tile.h` (Apache-2.0).
//
// Exactness of speculative decode: what a row computes does not depend on the other rows of the block (the steps of 128
// keys start at the chunk start, the maxima and sums are combined in fixed order, rescaling by 1 is exact),
// so B = 1 (RG = 1) and B = 8 (RG = 4) give bit for bit the same with the same chunk C.

#define PF_UNROLL2 _Pragma("clang loop unroll_count(2)")

// QI8 (prefill, lossy option): Q·Kᵀ with matmul2d int8×int8 -> int32; Q quantized to int8 per row (query and
// head) in q_quant_i8, with its scale qsc multiplied together with K's.
// TREE (draft tree): in the same dispatch, the threadgroups with tg.z < nr (p.dyn >> 8; "per-row", the first ones so
// they do not land in the last wave) do the chunks that touch the block with one query group per tree row
// (row tg.z·RG + g; only the first AT_G queries of the group count) and
// read the block keys by logical position: key ctx + i of row r is that of its ancestor at depth i
// (tree[176 + 8r + i]); the others (tg.z − nr) do the chunks that end before the block, as in the chain.
// This way each row sums the same and in the same order as the row of that depth in a chain
// (or as decoding one token): bit-for-bit equal.
// SPARSE (Flash-Next QSA): besides the causal mask, the row at position pos only sees the keys of the blocks
// (of 4) marked in its row of qbm ([rows, qnw]) and those of its open block (block >= (pos + 1) / 4). With <= 512
// visible blocks qsa_m_select marks them all: the same as dense attention.
// SWA (sliding window of p.window keys counting its own): row at position r only sees keys k with r − k < W (mlx-lm
// create_attention_mask(window_size)). The 128-key steps keep their absolute alignment and start at the first step that
// holds the window start of the threadgroup's first row: the skipped steps are fully masked for every row (they would
// leave m, l and O unchanged: a = 1, p = 0), so the result is bit for bit the one of the full loop with the mask, and a
// row gives the same bits whatever the block or prefill chunk it is in. A decode split entirely before the window
// writes (−inf, 0) and the reduction skips it. Not combined with TREE or SPARSE (the host refuses it).
// Ring KV (p.ring > 0): logical key k lives in slot k % ring. ring is a multiple of 128, so a 128-key step is
// one contiguous block of slots; the masks keep using logical positions. Keys of a step below the window may already
// hold newer positions: they are masked (v = −inf, P = 0), so their content does not matter.
template <int RG, bool DEC, bool QI8 = false, bool TREE = false, bool SPARSE = false, bool SWA = false>
[[kernel]] void pattn_i8(const device half*  q     [[buffer(0)]],   // [AT_KVH, AT_G·B, AT_HD] (m = row·AT_G + h)
                         const device char*  kc    [[buffer(1)]],   // [AT_KVH, ldkv, AT_HD]
                         const device char*  vt    [[buffer(2)]],   // [AT_KVH, AT_HD, ldkv]
                         const device float* ks    [[buffer(3)]],   // [AT_KVH, ldkv]
                         const device float* vs    [[buffer(4)]],   // [AT_KVH, ldkv]
                         const device float* proj  [[buffer(5)]],   // prefill: [B, AT_QP] (gate)
                         device half*        out   [[buffer(6)]],   // prefill: [B, AT_O]
                         constant AttnParams& p    [[buffer(7)]],
                         device float*       opart [[buffer(8)]],   // decode: [AT_KVH, nsplit, AT_G·B, AT_HD]
                         device float2*      ml    [[buffer(9)]],   // decode: [AT_KVH, nsplit, AT_G·B]
                         const device uint*  st    [[buffer(10)]],  // chained decode: st[0] = position
                         const device char*  qi    [[buffer(11)]],  // QI8: [AT_KVH, AT_G·B, AT_HD]
                         const device float* qsc   [[buffer(12)]],  // QI8: [AT_KVH, AT_G·B]
                         const device uint*  tree  [[buffer(13)]],  // (TREE): per-row ancestor mask in tree[48 + row]
                         const device uint*  qbm   [[buffer(14)]],  // SPARSE: attended blocks per row
                         constant uint&      qnw   [[buffer(15)]]   // SPARSE: words per row of qbm
                         AT_SINK_ARG(16),                            // LAB_AT_SINKS: [AT_QH] (prefill epilogue)
                         uint3 tg [[threadgroup_position_in_grid]],
                         ushort sg [[simdgroup_index_in_threadgroup]],
                         ushort lane [[thread_index_in_simdgroup]]) {
    constexpr int KB = 128, PL = KB + 8;
    threadgroup half pbuf[RG][16 * PL];
    threadgroup float redm[RG][4][16];
    const uint kvh = tg.y, Q = AT_G * p.B;
    const uint ctx = DEC && (p.dyn & AT_DYN_ST) ? st[0] : p.ctx;
    const uint g = sg / 4, j = sg % 4;
    // Prefill: threadgroup tg.x carries the queries [16·RG·tg.x, ...) and all their visible keys.
    // Decode: all the queries and the key chunk tg.x.
    // Decode: the groups of 16 queries can also be split over tg.z (RG groups per threadgroup).
    const uint nzr = (p.dyn >> 8) & 0xFFu;
    const uint row0 = DEC ? (TREE ? (tg.z >= nzr ? tg.z - nzr : 0) : tg.z) * (16 * RG) : tg.x * (16 * RG);
    const uint rlast = min(row0 + 16 * RG - 1, Q - 1);
    const bool ROWG = TREE && tg.z < nzr;                       // per-row threadgroup
    const uint trow = ROWG ? min(tg.z * RG + g, p.B - 1) : 0;
    const bool rowok = !ROWG || tg.z * RG + g < p.B;
    const uint tdep = ROWG ? tree[32 + trow] : 0;
    // SWA decode: the host does not launch the splits before the window; dyn >> 16 is the first launched one.
    const uint s0 = SWA && DEC ? p.dyn >> 16 : 0;
    const uint split = ROWG ? ctx / p.C + tg.x : tg.x + s0;
    if (TREE && !ROWG && split * p.C + p.C > ctx) return;       // chunk that touches the block: the per-row ones do it
    if (ROWG && split >= p.nsplit) return;
    const uint key0 = DEC ? split * p.C : 0;
    // Per row, the extra steps (all their keys masked) change nothing: the same kend for the whole threadgroup.
    const uint kend = DEC ? min(key0 + p.C, ctx + p.B) : ctx + rlast / AT_G + 1;
    // from here on some rows do not see all the keys
    const uint kdiag = TREE ? ctx : ctx + row0 / AT_G;
    // SWA: window start of the first and of the last row of the threadgroup (keys below wlast need the low mask).
    const uint rp0 = ctx + min(row0, Q - 1) / AT_G, rpl = ctx + rlast / AT_G;
    const uint wfirst = SWA && rp0 + 1 > p.window ? rp0 + 1 - p.window : 0;
    const uint wlast = SWA && rpl + 1 > p.window ? rpl + 1 - p.window : 0;
    const uint kstart = SWA ? max(key0, (wfirst / KB) * KB) : key0;
    const short2 co = nax_coord(lane);
    const uint fm = co.x, fn = co.y;
    const uint m0 = ROWG ? trow * AT_G + min(uint(fm), AT_G - 1) : row0 + 16 * g + fm;
    const uint m1 = ROWG ? trow * AT_G + min(uint(fm) + 8, AT_G - 1) : m0 + 8;
    const bool val0 = ROWG ? rowok && fm < AT_G : m0 < Q, val1 = ROWG ? rowok && fm + 8 < AT_G : m1 < Q;
    // ROWG: physical position of the logical key (the block ones, through the ancestor of that depth).
    auto pk = [&](uint key) -> uint { return key < ctx ? key : ctx + tree[176 + 8 * trow + min(key - ctx, 7u)]; };
    const device half* qh = q + size_t(kvh) * Q * AT_HD;
    const device char* kh = kc + size_t(kvh) * p.ldkv * AT_HD;
    const device char* vh = vt + size_t(kvh) * AT_HD * p.ldkv;
    const device float* ksh = ks + size_t(kvh) * p.ldkv;
    const device float* vsh = vs + size_t(kvh) * p.ldkv;
    const device half* q0 = qh + size_t(min(m0, Q - 1)) * AT_HD + fn;
    const device half* q1 = qh + size_t(min(m1, Q - 1)) * AT_HD + fn;
    const uint lim0 = ROWG ? ctx + tdep : ctx + m0 / AT_G, lim1 = ROWG ? ctx + tdep : ctx + m1 / AT_G;
    // P·V: each simdgroup carries AT_HD / 4 dimensions of O (PV_F fragments of 16).
    // MLA's latent head: V = c (no RoPE part), so P·V only covers the 512 latent dimensions, 128 per simdgroup
    // (8 fragments); dims 0…63 of the output are not written (vattn_reduce writes them as zero in decode; in prefill
    // unembed_out only reads 64…575).
    constexpr int PV_F = AT_MLA ? 8 : AT_HD / 64;
    nax_ffrag o[PV_F];
    NAX_UNROLL for (int f = 0; f < PV_F; f++) o[f] = 0;
    float mr0 = -INFINITY, mr1 = -INFINITY, l0 = 0, l1 = 0;
    threadgroup half* pb = pbuf[g];
    const uint d0 = AT_MLA ? 64 + 128 * j : (AT_HD / 4) * j;
    // SPARSE: each step of 128 keys is 32 blocks = one word of qbm (key0 is a multiple of 128). If the step lies
    // entirely below the open block of all the threadgroup's rows and no row marks a block of it, all its
    // scores would be −inf: the step changes neither m, l nor O (a = 1, p = 0), so skipping it is exact.
    const uint sr0 = min(row0, Q - 1) / AT_G, sr1 = rlast / AT_G;
    const uint sclosed = SPARSE ? ((ctx + sr0 + 1) / 4) * 4 : 0;
    // Inside a closed step, the 16-key fragments (4 blocks, one nibble of the word) with no bit in the OR of the rows
    // have all scores at −inf and P = 0: their Q·Kᵀ is skipped (the simdgroup whose 32 keys fall entirely in empty
    // fragments; the mask sets them to −inf anyway) and so is their P·V (adds +0 to O). Bit-exact (except the
    // sign of an exact zero of O, not reaching the output). Uniform in the threadgroup; the tail is untouched.
    for (uint k0 = kstart; k0 < kend; k0 += KB) {
        uint qw0 = 0, qw1 = 0, skip16 = 0;
        if (SPARSE) {
            const uint w = k0 / KB;
            if (k0 + KB <= sclosed) {
                uint any = 0;
                for (uint r = sr0; r <= sr1; r++) any |= qbm[r * qnw + w];
                if (any == 0) continue;
                NAX_UNROLL for (uint f = 0; f < 8; f++) skip16 |= ((any >> (4 * f)) & 0xFu) == 0 ? 1u << f : 0u;
            }
            if (w < qnw) {   // outside qbm there are only keys of the open block (no mask)
                qw0 = qbm[(min(m0, Q - 1) / AT_G) * qnw + w];
                qw1 = qbm[(min(m1, Q - 1) / AT_G) * qnw + w];
            }
        }
        const uint kj = k0 + 32 * j;   // S keys of this simdgroup
        const uint ps0 = SWA && p.ring ? k0 % p.ring : k0;   // first slot of the step (ring KV)
        const uint pj = ps0 + 32 * j;
        // MLA: S of k_pe (e0, e1, scale ks) and of c (s0, s1, scale vs) apart.
        nax_ffrag s0 = 0, s1 = 0, e0 = 0, e1 = 0;
        if (SPARSE && ((skip16 >> (2 * j)) & 3u) == 3u) {
            // the 32 keys of this simdgroup in empty fragments (S = 0; the mask leaves them at −inf)
        } else if (QI8) {
            nax_ifrag si0 = 0, si1 = 0;
            const device char* qa0 = qi + size_t(kvh) * Q * AT_HD + size_t(min(m0, Q - 1)) * AT_HD + fn;
            const device char* qa1 = qi + size_t(kvh) * Q * AT_HD + size_t(min(m1, Q - 1)) * AT_HD + fn;
            const device char* kr0 = kh + size_t(pj + fm) * AT_HD + fn;
            const device char* kr1 = kh + size_t(pj + 16 + fm) * AT_HD + fn;
            nax_ifrag sp0 = 0, sp1 = 0;
            NAX_UNROLL for (int c = 0; c < AT_CPE; c++) {
                const nax_cfrag qa = nax_cfrag(*(const device char4*)(qa0 + 16 * c), *(const device char4*)(qa1 + 16 * c));
                const nax_cfrag kb0 = nax_cfrag(*(const device char4*)(kr0 + 16 * c), *(const device char4*)(kr0 + 8 * AT_HD + 16 * c));
                const nax_cfrag kb1 = nax_cfrag(*(const device char4*)(kr1 + 16 * c), *(const device char4*)(kr1 + 8 * AT_HD + 16 * c));
                nax_mma2_i8(sp0, sp1, qa, kb0, kb1);
            }
            PF_UNROLL2 for (int c = AT_CPE; c < int(AT_HD / 16); c++) {
                const nax_cfrag qa = nax_cfrag(*(const device char4*)(qa0 + 16 * c), *(const device char4*)(qa1 + 16 * c));
                const nax_cfrag kb0 = nax_cfrag(*(const device char4*)(kr0 + 16 * c), *(const device char4*)(kr0 + 8 * AT_HD + 16 * c));
                const nax_cfrag kb1 = nax_cfrag(*(const device char4*)(kr1 + 16 * c), *(const device char4*)(kr1 + 8 * AT_HD + 16 * c));
                nax_mma2_i8(si0, si1, qa, kb0, kb1);
            }
            const float qs0 = qsc[size_t(kvh) * Q + min(m0, Q - 1)], qs1 = qsc[size_t(kvh) * Q + min(m1, Q - 1)];
            NAX_UNROLL for (int e = 0; e < 8; e++) { s0[e] = float(si0[e]) * (e < 4 ? qs0 : qs1); s1[e] = float(si1[e]) * (e < 4 ? qs0 : qs1); }
            if (AT_MLA) {
                NAX_UNROLL for (int e = 0; e < 8; e++) { e0[e] = float(sp0[e]) * (e < 4 ? qs0 : qs1); e1[e] = float(sp1[e]) * (e < 4 ? qs0 : qs1); }
            }
        } else {
        if (ROWG && k0 + KB > ctx) {
            const device char* r0 = kh + size_t(pk(kj + fm)) * AT_HD + fn;
            const device char* r1 = kh + size_t(pk(kj + fm + 8)) * AT_HD + fn;
            const device char* r2 = kh + size_t(pk(kj + 16 + fm)) * AT_HD + fn;
            const device char* r3 = kh + size_t(pk(kj + 24 + fm)) * AT_HD + fn;
            NAX_UNROLL for (int c = 0; c < AT_CPE; c++) {
                const nax_hfrag qc = nax_hfrag(*(const device half4*)(q0 + 16 * c), *(const device half4*)(q1 + 16 * c));
                const nax_hfrag kb0 = nax_hfrag(half4(*(const device char4*)(r0 + 16 * c)), half4(*(const device char4*)(r1 + 16 * c)));
                const nax_hfrag kb1 = nax_hfrag(half4(*(const device char4*)(r2 + 16 * c)), half4(*(const device char4*)(r3 + 16 * c)));
                nax_mma2(e0, e1, qc, kb0, kb1);
            }
            PF_UNROLL2 for (int c = AT_CPE; c < int(AT_HD / 16); c++) {
                const nax_hfrag qc = nax_hfrag(*(const device half4*)(q0 + 16 * c), *(const device half4*)(q1 + 16 * c));
                const nax_hfrag kb0 = nax_hfrag(half4(*(const device char4*)(r0 + 16 * c)), half4(*(const device char4*)(r1 + 16 * c)));
                const nax_hfrag kb1 = nax_hfrag(half4(*(const device char4*)(r2 + 16 * c)), half4(*(const device char4*)(r3 + 16 * c)));
                nax_mma2(s0, s1, qc, kb0, kb1);
            }
        } else {
        NAX_UNROLL for (int c = 0; c < AT_CPE; c++) {
            const nax_hfrag qc = nax_hfrag(*(const device half4*)(q0 + 16 * c), *(const device half4*)(q1 + 16 * c));
            nax_hfrag kb0, kb1;
            nax_load_i8(kb0, kh + size_t(pj) * AT_HD + 16 * c, AT_HD, co);
            nax_load_i8(kb1, kh + size_t(pj + 16) * AT_HD + 16 * c, AT_HD, co);
            nax_mma2(e0, e1, qc, kb0, kb1);
        }
        PF_UNROLL2 for (int c = AT_CPE; c < int(AT_HD / 16); c++) {
            const nax_hfrag qc = nax_hfrag(*(const device half4*)(q0 + 16 * c), *(const device half4*)(q1 + 16 * c));
            nax_hfrag kb0, kb1;
            nax_load_i8(kb0, kh + size_t(pj) * AT_HD + 16 * c, AT_HD, co);
            nax_load_i8(kb1, kh + size_t(pj + 16) * AT_HD + 16 * c, AT_HD, co);
            nax_mma2(s0, s1, qc, kb0, kb1);
        }
        }
        }
        const bool diag = k0 + KB > kdiag;
        const bool low = SWA && k0 < wlast;   // step with keys below the window of some row
        const bool gat = ROWG && k0 + KB > ctx;   // step with block keys: read by logical position
        float sv[16];
        float mx0 = -INFINITY, mx1 = -INFINITY;
        NAX_UNROLL for (int f = 0; f < 2; f++) {
            const uint kb = kj + 16 * f + fn, pb4 = pj + 16 * f + fn;
            const float4 ksv = (gat ? float4(ksh[pk(kb)], ksh[pk(kb + 1)], ksh[pk(kb + 2)], ksh[pk(kb + 3)])
                                    : *(const device float4*)(ksh + pb4)) * p.scale;
            float4 csv = 0;
            if (AT_MLA) {
                csv = (gat ? float4(vsh[pk(kb)], vsh[pk(kb + 1)], vsh[pk(kb + 2)], vsh[pk(kb + 3)])
                           : *(const device float4*)(vsh + pb4)) * p.scale;
            }
            NAX_UNROLL for (int e = 0; e < 8; e++) {
                float v = AT_MLA ? (f == 0 ? e0 : e1)[e] * ksv[e % 4] + (f == 0 ? s0 : s1)[e] * csv[e % 4]
                                 : (f == 0 ? s0 : s1)[e] * ksv[e % 4];
                if (SPARSE) {
                    const uint key = kj + 16 * f + fn + e % 4, kb4 = key / 4;
                    const uint row = min(e < 4 ? m0 : m1, Q - 1) / AT_G;
                    if (kb4 < (ctx + row + 1) / 4 && !(((e < 4 ? qw0 : qw1) >> (kb4 % 32)) & 1u)) v = -INFINITY;
                }
                if (low) {
                    const uint key = kj + 16 * f + fn + e % 4;
                    if (key + p.window <= (e < 4 ? lim0 : lim1)) v = -INFINITY;
                }
                if (diag) {
                    const uint key = kj + 16 * f + fn + e % 4;
                    if (TREE && !ROWG) {
                        const uint am = tree[48 + min((e < 4 ? m0 : m1) / AT_G, 15u)];
                        v = (key < ctx || (key - ctx < 16 && ((am >> (key - ctx)) & 1u))) && key < kend ? v : -INFINITY;
                    } else {
                        v = key <= (e < 4 ? lim0 : lim1) && key < kend ? v : -INFINITY;
                    }
                }
                sv[f * 8 + e] = v;
                if (e < 4) mx0 = max(mx0, v); else mx1 = max(mx1, v);
            }
        }
        mx0 = max(mx0, simd_shuffle_xor(mx0, 1)); mx0 = max(mx0, simd_shuffle_xor(mx0, 8));
        mx1 = max(mx1, simd_shuffle_xor(mx1, 1)); mx1 = max(mx1, simd_shuffle_xor(mx1, 8));
        if ((lane & 9) == 0) { redm[g][j][fm] = mx0; redm[g][j][fm + 8] = mx1; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float t0 = max(max(redm[g][0][fm], redm[g][1][fm]), max(redm[g][2][fm], redm[g][3][fm]));
        const float t1 = max(max(redm[g][0][fm + 8], redm[g][1][fm + 8]), max(redm[g][2][fm + 8], redm[g][3][fm + 8]));
        const float mn0 = max(mr0, t0), mn1 = max(mr1, t1);
        const float a0 = mn0 == -INFINITY ? 1.0f : exp(mr0 - mn0), a1 = mn1 == -INFINITY ? 1.0f : exp(mr1 - mn1);
        float rs0 = 0, rs1 = 0;
        NAX_UNROLL for (int f = 0; f < 2; f++) {
            const uint kb = kj + 16 * f + fn, pb4 = pj + 16 * f + fn;
            const float4 vsv = gat ? float4(vsh[pk(kb)], vsh[pk(kb + 1)], vsh[pk(kb + 2)], vsh[pk(kb + 3)])
                                   : *(const device float4*)(vsh + pb4);
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
        // ROWG: transposed V read by logical keys (4 loose bytes per row of dimensions).
        // MLA: V is the latent part of kc, key-major (rows = keys, columns = dims; P·V with nax_mma2_kn).
        auto loadv = [&](thread nax_hfrag& d, uint dim, uint key) {
            if (AT_MLA) {
                if (gat) {
                    const device char* x0 = kh + size_t(pk(key + fm)) * AT_HD + dim + fn;
                    const device char* x1 = kh + size_t(pk(key + fm + 8)) * AT_HD + dim + fn;
                    d = nax_hfrag(half4(*(const device char4*)x0), half4(*(const device char4*)x1));
                } else {
                    nax_load_i8(d, kh + size_t(key - k0 + ps0) * AT_HD + dim, AT_HD, co);
                }
            } else if (gat) {
                const uint a = pk(key + fn), b = pk(key + fn + 1), c2 = pk(key + fn + 2), e2 = pk(key + fn + 3);
                const device char* x0 = vh + size_t(dim + fm) * p.ldkv;
                const device char* x1 = vh + size_t(dim + fm + 8) * p.ldkv;
                d = nax_hfrag(half4(half(x0[a]), half(x0[b]), half(x0[c2]), half(x0[e2])),
                              half4(half(x1[a]), half(x1[b]), half(x1[c2]), half(x1[e2])));
            } else {
                nax_load_i8(d, vh + size_t(dim) * p.ldkv + (key - k0 + ps0), p.ldkv, co);
            }
        };
        PF_UNROLL2 for (int kc2 = 0; kc2 < KB / 16; kc2++) {
            if (SPARSE && ((skip16 >> kc2) & 1u)) continue;   // P = 0 in the 16 keys
            nax_hfrag pa;
            nax_load(pa, pb + 16 * kc2, PL, co);
            if constexpr (PV_F == 1) {
                nax_hfrag vb0;
                loadv(vb0, d0, k0 + 16 * kc2);
                nax_ffrag dummy = 0;
                nax_mma2(o[0], dummy, pa, vb0, vb0);
            } else {
#if LAB_AT_HD == 576
                // MLA's latent head: four pairs of 16 latent dims, V key-major from kc.
                NAX_UNROLL for (int f = 0; f < PV_F; f += 2) {
                    nax_hfrag vb0, vb1;
                    loadv(vb0, d0 + 16 * f, k0 + 16 * kc2);
                    loadv(vb1, d0 + 16 * f + 16, k0 + 16 * kc2);
                    nax_mma2_kn(o[f], o[f + 1], pa, vb0, vb1);
                }
#else
                NAX_UNROLL for (int f = 0; f < PV_F; f += 2) {
                    nax_hfrag vb0, vb1;
                    loadv(vb0, d0 + 16 * f, k0 + 16 * kc2);
                    loadv(vb1, d0 + 16 * f + 16, k0 + 16 * kc2);
                    nax_mma2(o[f], o[f + 1], pa, vb0, vb1);
                }
#endif
            }
        }
    }
    // l total = sum of the 4 parts (each simdgroup summed its keys), in fixed order.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if ((lane & 9) == 0) { redm[g][j][fm] = l0; redm[g][j][fm + 8] = l1; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float L0 = (redm[g][0][fm] + redm[g][1][fm]) + (redm[g][2][fm] + redm[g][3][fm]);
    const float L1 = (redm[g][0][fm + 8] + redm[g][1][fm + 8]) + (redm[g][2][fm + 8] + redm[g][3][fm + 8]);
    if (DEC) {
        device float* op = opart + (size_t(kvh) * p.nsplit + split - s0) * Q * AT_HD;
        NAX_UNROLL for (int f = 0; f < PV_F; f++) NAX_UNROLL for (int i = 0; i < 2; i++) {
            const uint m = i == 0 ? m0 : m1;
            if (i == 0 ? val0 : val1) *(device float4*)(op + size_t(m) * AT_HD + d0 + 16 * f + fn) =
                float4(o[f][4 * i], o[f][4 * i + 1], o[f][4 * i + 2], o[f][4 * i + 3]);
        }
        if (j == 0 && (lane & 9) == 0) {
            device float2* mp = ml + (size_t(kvh) * p.nsplit + split - s0) * Q;
            if (val0) mp[m0] = float2(mr0, L0);
            if (val1) mp[m1] = float2(mr1, L1);
        }
        return;
    }
    // Prefill: output normalized with the gate, out[t, h·AT_HD + d] = O / l · sigmoid(gate).
    NAX_UNROLL for (int i = 0; i < 2; i++) {
        const uint m = i == 0 ? m0 : m1;
        if (m >= Q) continue;
        const uint t = m / AT_G, h = kvh * AT_G + m % AT_G;
        float il = 1.0f / (i == 0 ? L0 : L1), sc = 1.0f;
#if LAB_AT_SINKS
        // Sink: with Mf = max(m, s), O·e^(m − Mf) / (L·e^(m − Mf) + e^(s − Mf)).
        const float mr = i == 0 ? mr0 : mr1, mf = max(mr, sinks[h]);
        sc = exp(mr - mf);
        il = 1.0f / ((i == 0 ? L0 : L1) * sc + exp(sinks[h] - mf));
#endif
        const device float* gp = proj + size_t(t) * AT_QP + h * AT_QS + AT_HD;
        device half* op = out + size_t(t) * AT_O + h * AT_HD;
        NAX_UNROLL for (int f = 0; f < PV_F; f++) {
            const uint d = d0 + 16 * f + fn;
            const float4 ov = float4(o[f][4 * i], o[f][4 * i + 1], o[f][4 * i + 2], o[f][4 * i + 3]) * (sc * il);
            if (AT_GATED) {
                const float4 gv = *(const device float4*)(gp + d);
                *(device half4*)(op + d) = half4(ov / (1.0f + exp(-gv)));
            } else {
                *(device half4*)(op + d) = half4(ov);
            }
        }
    }
}

#define PATTN_INST(RG, DEC, NAME) PATTN_INSTQ(RG, DEC, false, NAME)
#define PATTN_INSTQ(RG, DEC, QI8, NAME) PATTN_INSTT(RG, DEC, QI8, false, NAME)
#define PATTN_INSTT(RG, DEC, QI8, TREE, NAME) PATTN_INSTS(RG, DEC, QI8, TREE, false, NAME)
#define PATTN_INSTS(RG, DEC, QI8, TREE, SP, NAME) PATTN_INSTW(RG, DEC, QI8, TREE, SP, false, NAME)
#define PATTN_INSTW(RG, DEC, QI8, TREE, SP, SWA, NAME) \
    template [[host_name(NAME)]] [[kernel]] void pattn_i8<RG, DEC, QI8, TREE, SP, SWA>( \
        const device half*, const device char*, const device char*, const device float*, const device float*, \
        const device float*, device half*, constant AttnParams&, device float*, device float2*, const device uint*, \
        const device char*, const device float*, const device uint*, const device uint*, constant uint& AT_SINK_T, \
        uint3, ushort, ushort);
PATTN_INST(1, false, "pattn_i8_1")
PATTN_INST(2, false, "pattn_i8_2")
PATTN_INST(4, false, "pattn_i8_4")
PATTN_INST(1, true, "dattn_i8_1")
PATTN_INST(2, true, "dattn_i8_2")
PATTN_INST(4, true, "dattn_i8_4")
// Verification of a draft tree (ancestor mask instead of causal).
PATTN_INSTT(1, true, false, true, "dattn_i8t_1")
PATTN_INSTT(2, true, false, true, "dattn_i8t_2")
// Flash-Next QSA (per-row block mask).
PATTN_INSTS(2, false, false, false, true, "pattn_i8s_2")
PATTN_INSTS(1, true, false, false, true, "dattn_i8s_1")
PATTN_INSTS(2, true, false, false, true, "dattn_i8s_2")
PATTN_INSTS(4, true, false, false, true, "dattn_i8s_4")
PATTN_INSTQ(1, false, true, "pattn_q8_1")
PATTN_INSTQ(2, false, true, "pattn_q8_2")
PATTN_INSTQ(4, false, true, "pattn_q8_4")
// Q·Kᵀ in int8 with the QSA mask (lossy prefill of Flash-Next from 8K).
PATTN_INSTS(2, false, true, false, true, "pattn_q8s_2")
#if LAB_AT_SWA
// Sliding-window layers (prefill, Q·Kᵀ int8 prefill, verification and decode).
PATTN_INSTW(2, false, false, false, false, true, "pattn_i8w_2")
PATTN_INSTW(2, false, true, false, false, true, "pattn_q8w_2")
PATTN_INSTW(1, true, false, false, false, true, "dattn_i8w_1")
PATTN_INSTW(2, true, false, false, false, true, "dattn_i8w_2")
PATTN_INSTW(4, true, false, false, false, true, "dattn_i8w_4")
#endif

#if LAB_AT_HD == 64
// Causal prefill attention for 64-wide heads (gpt-oss, LFM2, Granite 4.1), int8 KV, full layers only (no window,
// no QSA mask, Q in half). pattn_i8's layout (4 simdgroups splitting the keys of a 128-key step for S and the dimensions
// for P·V) pays the same per-step exchanges for half the work with a 64 head, and its P·V runs 16×32×16 matmuls with
// half of the output thrown away (one 16-dimension fragment per simdgroup): 55.6 ms per gpt-oss global layer for a
// 1,920-row chunk at 19,200 of context (11.4 TFLOP/s; MLX's NAX attention, 31 ms).
// Here, as in MLX's steel_attention_nax (bq 64, bk 32, 4 simdgroups; MIT, Copyright © 2024-26 Apple Inc.), each
// simdgroup is an independent flash attention over TQ fragments of 16 queries (16 / AT_G tokens × the AT_G heads of the
// GQA group) and all their keys, in steps of 32: Q stays in registers, S is two 16×16 fragments per query fragment,
// P goes from the S accumulator straight to the left operand of P·V (same element layout, no threadgroup memory) and O
// is 4 fragments (all 64 dimensions). No barriers: a simdgroup stops at the last key of its own rows.
// Wide int8 loads: the head dimensions and the keys of a step are permuted the same way on both sides of each product
// (the sums are the same up to their order), so each thread reads a K row with one 16-byte load and a V row with one
// 8-byte load per step instead of 4-byte ones: column fn + j of fragment c of Q·Kᵀ is dimension 4·fn + 4c + j; column
// fn + j of key fragment f is key k0 + 2·fn + 4f + j.
// Measured (microbenchmark, gpt-oss shape, 1,920 rows, M6): 1.63 / 9.54 / 18.3 / 29.4 / 41.8 ms at 0 / 4,800 /
// 9,600 / 14,400 / 19,200 of context with TQ 1 (pattn_i8_2: 2.47 / 13.9 / 26.6 / 40.6 / 55.6); with TQ 2, 28.2 and
// 37.2 at 14,400 and 19,200 (fewer threadgroups walking the cache: past ~2 MB of KV per head the reads miss; with the
// loads kept in cache the kernel runs at 19.6 TFLOP/s, the matmuls alone at 21). Neither exp2, nor loading V before
// the softmax, nor prefetching the next step (registers), nor TQ 4 (spills) helped.
// Exactness: what a row computes depends only on its keys and their order (32-key steps from key 0; masked keys of an
// extra step give a = 1, p = 0 and change neither m, l nor O; the products of a fragment do not depend on TQ or NSG),
// so a row gives the same bits in any prefill chunk, block, TQ or NSG; not the same bits as pattn_i8 (another
// summation order).
template <int NSG, int TQ>
[[kernel]] void pattn64_i8(const device half*  q    [[buffer(0)]],   // [AT_KVH, AT_G·B, 64] (m = row·AT_G + h)
                           const device char*  kc   [[buffer(1)]],   // [AT_KVH, ldkv, 64]
                           const device char*  vt   [[buffer(2)]],   // [AT_KVH, 64, ldkv]
                           const device float* ks   [[buffer(3)]],   // [AT_KVH, ldkv]
                           const device float* vs   [[buffer(4)]],   // [AT_KVH, ldkv]
                           const device float* proj [[buffer(5)]],   // [B, AT_QP] (gate)
                           device half*        out  [[buffer(6)]],   // [B, AT_O]
                           constant AttnParams& p   [[buffer(7)]]
                           AT_SINK_ARG(16),                          // LAB_AT_SINKS: [AT_QH]
                           uint3 tg [[threadgroup_position_in_grid]],
                           ushort sg [[simdgroup_index_in_threadgroup]],
                           ushort lane [[thread_index_in_simdgroup]]) {
    constexpr uint KB = 32;
    const uint kvh = tg.y, Q = AT_G * p.B, ctx = p.ctx;
    const uint row0 = (tg.x * NSG + sg) * 16 * TQ;
    if (row0 >= Q) return;   // no threadgroup memory nor barriers in this kernel
    const uint rlast = min(row0 + 16 * TQ - 1, Q - 1);
    const uint kend = ctx + rlast / AT_G + 1;   // keys of the last row of this simdgroup
    const uint kdiag = ctx + row0 / AT_G;       // from here on some rows do not see every key
    const short2 co = nax_coord(lane);
    const uint fm = co.x, fn = co.y;
    const device char* kh = kc + size_t(kvh) * p.ldkv * 64;
    const device char* vh = vt + size_t(kvh) * 64 * p.ldkv;
    const device float* ksh = ks + size_t(kvh) * p.ldkv;
    const device float* vsh = vs + size_t(kvh) * p.ldkv;
    // Fragment iq holds the queries row0 + 16·iq + [0, 16): this thread's rows m0 (elements 0-3) and m0 + 8 (4-7).
    nax_hfrag qf[TQ][4];
    uint lim[TQ][2];
    NAX_UNROLL for (int iq = 0; iq < TQ; iq++) {
        const uint m0 = row0 + 16 * iq + fm;
        const device half* qh = q + size_t(kvh) * Q * 64;
        const device half* q0 = qh + size_t(min(m0, Q - 1)) * 64 + 4 * fn;
        const device half* q1 = qh + size_t(min(m0 + 8, Q - 1)) * 64 + 4 * fn;
        NAX_UNROLL for (int c = 0; c < 4; c++)
            qf[iq][c] = nax_hfrag(*(const device half4*)(q0 + 4 * c), *(const device half4*)(q1 + 4 * c));
        lim[iq][0] = ctx + m0 / AT_G; lim[iq][1] = ctx + (m0 + 8) / AT_G;
    }
    nax_ffrag o[TQ][4];
    float mr[TQ][2], l[TQ][2];
    NAX_UNROLL for (int iq = 0; iq < TQ; iq++) {
        NAX_UNROLL for (int f = 0; f < 4; f++) o[iq][f] = 0;
        mr[iq][0] = mr[iq][1] = -INFINITY; l[iq][0] = l[iq][1] = 0;
    }
    for (uint k0 = 0; k0 < kend; k0 += KB) {
        // K: row fm + 8i of key fragment f (key k0 + 8·(r / 4) + 4f + r % 4), dimensions 4·fn … 4·fn + 15.
        uint4 kw[2][2];
        NAX_UNROLL for (int f = 0; f < 2; f++) NAX_UNROLL for (int i = 0; i < 2; i++) {
            const uint r = fm + 8 * i;
            kw[f][i] = *(const device uint4*)(kh + size_t(k0 + 8 * (r / 4) + 4 * f + r % 4) * 64 + 4 * fn);
        }
        nax_ffrag s[TQ][2];
        NAX_UNROLL for (int iq = 0; iq < TQ; iq++) s[iq][0] = s[iq][1] = 0;
        NAX_UNROLL for (int c = 0; c < 4; c++) {
            const nax_hfrag kb0 = nax_hfrag(half4(as_type<char4>(kw[0][0][c])), half4(as_type<char4>(kw[0][1][c])));
            const nax_hfrag kb1 = nax_hfrag(half4(as_type<char4>(kw[1][0][c])), half4(as_type<char4>(kw[1][1][c])));
            NAX_UNROLL for (int iq = 0; iq < TQ; iq++) nax_mma2(s[iq][0], s[iq][1], qf[iq][c], kb0, kb1);
        }
        const bool diag = k0 + KB > kdiag;
        const uint kf[2] = {k0 + 2 * fn, k0 + 2 * fn + 4};   // key of column fn of each key fragment
        const float4 ksv[2] = {*(const device float4*)(ksh + kf[0]) * p.scale,
                               *(const device float4*)(ksh + kf[1]) * p.scale};
        const float4 vsv[2] = {*(const device float4*)(vsh + kf[0]), *(const device float4*)(vsh + kf[1])};
        nax_hfrag pa[TQ][2];
        NAX_UNROLL for (int iq = 0; iq < TQ; iq++) {
            float mx[2] = {-INFINITY, -INFINITY};
            NAX_UNROLL for (int f = 0; f < 2; f++) NAX_UNROLL for (int e = 0; e < 8; e++) {
                float v = s[iq][f][e] * ksv[f][e % 4];
                if (diag) {
                    const uint key = kf[f] + e % 4;
                    v = key <= lim[iq][e / 4] && key < kend ? v : -INFINITY;
                }
                s[iq][f][e] = v;
                mx[e / 4] = max(mx[e / 4], v);
            }
            float a[2], rs[2] = {0, 0}, mn[2];
            NAX_UNROLL for (int i = 0; i < 2; i++) {
                mx[i] = max(mx[i], simd_shuffle_xor(mx[i], 1)); mx[i] = max(mx[i], simd_shuffle_xor(mx[i], 8));
                mn[i] = max(mr[iq][i], mx[i]);
                a[i] = mn[i] == -INFINITY ? 1.0f : exp(mr[iq][i] - mn[i]);
            }
            NAX_UNROLL for (int f = 0; f < 2; f++) NAX_UNROLL for (int e = 0; e < 8; e++) {
                const float pv = mn[e / 4] == -INFINITY ? 0.0f : exp(s[iq][f][e] - mn[e / 4]);
                rs[e / 4] += pv;
                pa[iq][f][e] = half(pv * vsv[f][e % 4]);
            }
            NAX_UNROLL for (int i = 0; i < 2; i++) {
                rs[i] += simd_shuffle_xor(rs[i], 1); rs[i] += simd_shuffle_xor(rs[i], 8);
                l[iq][i] = l[iq][i] * a[i] + rs[i];
                mr[iq][i] = mn[i];
            }
            if (!simd_all(a[0] == 1.0f && a[1] == 1.0f)) {
                NAX_UNROLL for (int f = 0; f < 4; f++) NAX_UNROLL for (int e = 0; e < 8; e++) o[iq][f][e] *= a[e / 4];
            }
        }
        // V: dimension row 16d + fm + 8i, keys k0 + 2·fn … + 7 (the first 4 of key fragment 0, the next 4 of fragment 1).
        uint2 vw[4][2];
        NAX_UNROLL for (int d = 0; d < 4; d++) NAX_UNROLL for (int i = 0; i < 2; i++)
            vw[d][i] = *(const device uint2*)(vh + size_t(16 * d + fm + 8 * i) * p.ldkv + k0 + 2 * fn);
        NAX_UNROLL for (int f = 0; f < 2; f++) {
            NAX_UNROLL for (int d = 0; d < 4; d += 2) {
                const nax_hfrag vb0 = nax_hfrag(half4(as_type<char4>(vw[d][0][f])), half4(as_type<char4>(vw[d][1][f])));
                const nax_hfrag vb1 =
                    nax_hfrag(half4(as_type<char4>(vw[d + 1][0][f])), half4(as_type<char4>(vw[d + 1][1][f])));
                NAX_UNROLL for (int iq = 0; iq < TQ; iq++) nax_mma2(o[iq][d], o[iq][d + 1], pa[iq][f], vb0, vb1);
            }
        }
    }
    // Output normalized with the gate (and the sink), as pattn_i8's prefill epilogue.
    NAX_UNROLL for (int iq = 0; iq < TQ; iq++) NAX_UNROLL for (int i = 0; i < 2; i++) {
        const uint m = row0 + 16 * iq + fm + 8 * i;
        if (m >= Q) continue;
        const uint t = m / AT_G, h = kvh * AT_G + m % AT_G;
        float il = 1.0f / l[iq][i], sc = 1.0f;
#if LAB_AT_SINKS
        // Sink: with Mf = max(m, s), O·e^(m − Mf) / (L·e^(m − Mf) + e^(s − Mf)).
        const float mf = max(mr[iq][i], sinks[h]);
        sc = exp(mr[iq][i] - mf);
        il = 1.0f / (l[iq][i] * sc + exp(sinks[h] - mf));
#endif
        const device float* gp = proj + size_t(t) * AT_QP + h * AT_QS + 64;
        device half* op = out + size_t(t) * AT_O + h * 64;
        NAX_UNROLL for (int f = 0; f < 4; f++) {
            const uint d = 16 * f + fn;
            const float4 ov =
                float4(o[iq][f][4 * i], o[iq][f][4 * i + 1], o[iq][f][4 * i + 2], o[iq][f][4 * i + 3]) * (sc * il);
            if (AT_GATED) {
                const float4 gv = *(const device float4*)(gp + d);
                *(device half4*)(op + d) = half4(ov / (1.0f + exp(-gv)));
            } else {
                *(device half4*)(op + d) = half4(ov);
            }
        }
    }
}
#define PATTN64_INST(NSG, TQ, NAME) \
    template [[host_name(NAME)]] [[kernel]] void pattn64_i8<NSG, TQ>( \
        const device half*, const device char*, const device char*, const device float*, const device float*, \
        const device float*, device half*, constant AttnParams& AT_SINK_T, uint3, ushort, ushort);
PATTN64_INST(4, 1, "pattn64_i8_t1")
PATTN64_INST(4, 2, "pattn64_i8_t2")
#endif

// Q half [rows, AT_HD] -> int8 per row (scale absmax / 127). One simdgroup per row, AT_HD / 32 values per lane (the
// 256 head, 8 in two half4; the 128 and 64 ones, 4 and 2).
[[kernel]] void q_quant_i8(const device half* q   [[buffer(0)]],
                           device char*       qi  [[buffer(1)]],
                           device float*      qs  [[buffer(2)]],
                           constant uint&     n   [[buffer(3)]],
                           uint   tg   [[threadgroup_position_in_grid]],
                           ushort sg   [[simdgroup_index_in_threadgroup]],
                           ushort lane [[thread_index_in_simdgroup]]) {
    const uint r = tg * 4 + sg;
    if (r >= n) return;
    if (AT_HD != 256) {
        constexpr uint QV = AT_HD / 32;
        const device half* src = q + size_t(r) * AT_HD + lane * QV;
        float v[QV], mx = 0;
        VUNROLL for (uint i = 0; i < QV; i++) { v[i] = float(src[i]); mx = max(mx, abs(v[i])); }
        const float m = simd_max(mx);
        const float sc = m > 0 ? m / 127.0f : 1.0f, inv = 1.0f / sc;
        VUNROLL for (uint i = 0; i < QV; i++) qi[size_t(r) * AT_HD + lane * QV + i] = char(clamp(rint(v[i] * inv), -127.0f, 127.0f));
        if (lane == 0) qs[r] = sc;
        return;
    }
    const device half* src = q + size_t(r) * 256 + lane * 8;
    const float4 a = float4(*(const device half4*)src), b = float4(*(const device half4*)(src + 4));
    const float m = simd_max(max(max(abs(a.x), abs(a.y)), max(max(abs(a.z), abs(a.w)), max(max(abs(b.x), abs(b.y)), max(abs(b.z), abs(b.w))))));
    const float sc = m > 0 ? m / 127.0f : 1.0f, inv = 1.0f / sc;
    *(device char4*)(qi + size_t(r) * 256 + lane * 8) = char4(clamp(rint(a * inv), -127.0f, 127.0f));
    *(device char4*)(qi + size_t(r) * 256 + lane * 8 + 4) = char4(clamp(rint(b * inv), -127.0f, 127.0f));
    if (lane == 0) qs[r] = sc;
}
