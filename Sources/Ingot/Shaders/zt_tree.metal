// Portions adapted from TensorFold (Apache-2.0; see THIRD_PARTY_NOTICES.md); changed: the slot stack is rewritten
// for the draft tree.
// Draft-tree verification on dense models (`--tree`). Goes after verify*.metal and yy_model.metal
// (uses GdnParams, GDN_*, TK and VUNROLL).
//
// Tree buffer (uint32), one round:
//   [0] rows B (nodes, anchor included)        [16 + r] parent of row r (0xffffffff in row 0, the anchor)
//   [32 + r] depth (anchor 0)                   [48 + r] ancestor mask (bit i = row i is an ancestor or itself)
//   [64 + 3r + j] conv inputs of row r (j = 0..2, from oldest to newest): "extended" index,
//                 0..2 = rows of the history window, 3 + i = row i of the block
//   [112 + r] after row r, save the state in the slot (value − 1; 0 = no)
//   [128 + r] before row r, restore the state from the slot (value − 1; 0 = no)
//   [144 + i] accepted path (rows of the block), [160] slots used, [165] m (accepted drafts), [166] P from
//   before the round
//   [176 + 8r + i] row of the ancestor at depth i of row r (i ≤ depth; after that, r): exact attention
// On accept, the path is compacted into contiguous rows (KV, drafter captures, GDN tape and history), so that
// the next round, the undo and the prefix cache see the same as after a chain.
// Rows go in preorder (parent before children and the first child right after the parent); the children with more
// descendants go last, so a stack of slots of depth ≤ log2(B) is enough.
constant constexpr uint TR_PAR = 16, TR_DEP = 32, TR_ANC = 48, TR_CONV = 64, TR_SAVE = 112, TR_REST = 128, TR_PATH = 144,
                        TR_NS = 160, TR_M = 165, TR_P0 = 166, TR_AROW = 176, TR_WORDS = 304;

// causal conv1d + SiLU, q and k norms, g and beta, like gdn_prework<float, float>, but each row takes as its previous
// inputs those of its ancestors (indices of the tree buffer) and the history window along the accepted path.
[[kernel]] void gdn_prework_tree(const device float*  proj  [[buffer(0)]],
                                 const device LAB_NWT* convw [[buffer(1)]],
                                 const device float*  hprev [[buffer(2)]],
                                 const device float*   alog  [[buffer(3)]],
                                 const device LAB_NWT* dtb   [[buffer(4)]],
                                 device float*        hcur  [[buffer(5)]],
                                 device float*        q     [[buffer(6)]],
                                 device float*        k     [[buffer(7)]],
                                 device float*        v     [[buffer(8)]],
                                 device float*        g     [[buffer(9)]],
                                 device float*        beta  [[buffer(10)]],
                                 constant GdnParams&  p     [[buffer(11)]],
                                 const device uint*   tree  [[buffer(12)]],
                                 const device uint*   rst   [[buffer(13)]],
                                 uint2 tg [[threadgroup_position_in_grid]],
                                 uint tid [[thread_index_in_threadgroup]],
                                 ushort sg [[simdgroup_index_in_threadgroup]],
                                 ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[4];
    const uint t = tg.y, slot = tg.x, c = slot * 128 + tid;
    const uint nacc = (p.dyn & 1) ? rst[1] : p.nacc;
    float acc = 0;
    VUNROLL for (uint j = 0; j < 4; j++) {
        float in;
        if (j < 3) {
            const uint e = tree[TR_CONV + 3 * t + j];
            in = e < 3 ? hprev[(nacc + e) * GDN_C + c] : proj[(e - 3) * p.ldp + c];
        } else {
            in = proj[t * p.ldp + c];
        }
        acc = fma(float(convw[c * 4 + j]), in, acc);
    }
    const float o = acc / (1.0f + exp(-acc));
    if (t == 0) VUNROLL for (uint i = 0; i < 3; i++) hcur[i * GDN_C + c] = hprev[(nacc + i) * GDN_C + c];
    hcur[(3 + t) * GDN_C + c] = proj[t * p.ldp + c];
    if (slot < 32) {
        const float ss = simd_sum(o * o);
        if (lane == 0) part[sg] = ss;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float tot = part[0] + part[1] + part[2] + part[3];
        constexpr float inv = 0.08838834764831845f;
        const float r = rsqrt(tot / 128.0f + 1e-6f * inv * inv);
        if (slot < 16) q[t * GDN_QK + c] = o * r * (inv * inv);
        else k[t * GDN_QK + c - GDN_QK] = o * r * inv;
    } else {
        v[t * GDN_V + c - 2 * GDN_QK] = o;
    }
    if (slot == 0 && tid < GDN_HV) {
        const float bb = float(proj[t * p.ldp + GDN_PB + tid]), aa = float(proj[t * p.ldp + GDN_PA + tid]);
        beta[t * GDN_HV + tid] = 1.0f / (1.0f + exp(-bb));
        const float xa = aa + float(dtb[tid]);
        const float sp = xa > 20.0f ? xa : log(1.0f + exp(xa));
        g[t * GDN_HV + tid] = exp(-exp(float(alog[tid])) * sp);
    }
}

// Delta rule recurrence over a tree (like gdn_recur_g<4, 2>): folds the nacc rows of the tape (the accepted path
// of the previous round, already compacted), writes that state and walks the B rows of the tree in preorder; before
// a row whose parent is not the previous one it restores the parent's state from a slot into registers and, after a row
// with several children, saves it (stack of S slots, idea from TensorFold `deltanet_slots.metal`). With the child with
// the most descendants last, k slots open at once need ≥ 3·2^(k−1) rows: S = 1 up to 5 rows, 2 up to 11 and 3
// up to 23. Each slot is 16 registers per thread: with fewer, the recurrence runs faster.
template <int TREE_SLOTS>
[[kernel]] void gdn_recur_tree(const device float* sin   [[buffer(0)]],
                               device float*       sout  [[buffer(1)]],
                               const device float* kp    [[buffer(2)]],
                               const device float* vp    [[buffer(3)]],
                               const device float* gp    [[buffer(4)]],
                               const device float* bp    [[buffer(5)]],
                               const device float* q     [[buffer(6)]],
                               const device float* k     [[buffer(7)]],
                               const device float* v     [[buffer(8)]],
                               const device float* g     [[buffer(9)]],
                               const device float* beta  [[buffer(10)]],
                               device float*       y     [[buffer(11)]],
                               constant GdnParams& p     [[buffer(12)]],
                               const device uint*  tree  [[buffer(13)]],
                               const device uint*  rst   [[buffer(14)]],
                               uint tg [[threadgroup_position_in_grid]],
                               ushort sg [[simdgroup_index_in_threadgroup]],
                               ushort lane [[thread_index_in_simdgroup]]) {
    constexpr int DVS = 4, G = 2;
    constexpr uint per_head = 128 / (4 * DVS);
    const uint hv = tg / per_head, dv0 = (tg % per_head) * 4 * DVS + sg * DVS, hk = hv / GDN_R;
    const uint nacc = (p.dyn & 1) ? rst[1] : p.nacc;
    const uint T = nacc + p.B;
    struct Grp { float4 k[G], q[G]; float v[G][DVS], g[G], b[G]; uint sv[G], rs[G]; };
    auto load = [&](uint t0, thread Grp& D) {
        VUNROLL for (int i = 0; i < G; i++) {
            const uint t = min(t0 + i, T - 1);
            const bool tape = t < nacc;
            const uint tc = tape ? t : t - nacc;
            D.k[i] = ((const device float4*)((tape ? kp : k) + tc * GDN_QK + hk * 128))[lane];
            D.q[i] = tape ? float4(0) : ((const device float4*)(q + tc * GDN_QK + hk * 128))[lane];
            VUNROLL for (int d = 0; d < DVS; d++) D.v[i][d] = (tape ? vp : v)[tc * GDN_V + hv * 128 + dv0 + d];
            D.g[i] = (tape ? gp : g)[tc * GDN_HV + hv];
            D.b[i] = (tape ? bp : beta)[tc * GDN_HV + hv];
            D.sv[i] = tape ? 0 : tree[TR_SAVE + tc];
            D.rs[i] = tape ? 0 : tree[TR_REST + tc];
        }
    };
    float4 st[DVS], sl[TREE_SLOTS][DVS];
    VUNROLL for (int d = 0; d < DVS; d++) st[d] = ((const device float4*)(sin + (hv * 128 + dv0 + d) * 128))[lane];
    VUNROLL for (int s = 0; s < TREE_SLOTS; s++) VUNROLL for (int d = 0; d < DVS; d++) sl[s][d] = 0;
    if (nacc == 0) VUNROLL for (int d = 0; d < DVS; d++) ((device float4*)(sout + (hv * 128 + dv0 + d) * 128))[lane] = st[d];
    Grp cur, nxt;
    if (T > 0) load(0, cur);
    for (uint t0 = 0; t0 < T; t0 += G) {
        if (t0 + G < T) load(t0 + G, nxt);
        VUNROLL for (int i = 0; i < G; i++) {
            const uint t = t0 + i;
            if (t < T) {
                if (cur.rs[i]) VUNROLL for (int s = 0; s < TREE_SLOTS; s++) if (cur.rs[i] == uint(s + 1)) VUNROLL for (int d = 0; d < DVS; d++) st[d] = sl[s][d];
                VUNROLL for (int d = 0; d < DVS; d++) {
                    st[d] *= cur.g[i];
                    const float kv = simd_sum(dot(st[d], cur.k[i]));
                    st[d] += cur.k[i] * ((cur.v[i][d] - kv) * cur.b[i]);
                    if (t >= nacc) {
                        const float yo = simd_sum(dot(st[d], cur.q[i]));
                        if (lane == 0) y[(t - nacc) * GDN_V + hv * 128 + dv0 + d] = yo;
                    }
                }
                if (cur.sv[i]) VUNROLL for (int s = 0; s < TREE_SLOTS; s++) if (cur.sv[i] == uint(s + 1)) VUNROLL for (int d = 0; d < DVS; d++) sl[s][d] = st[d];
                if (t + 1 == nacc)
                    VUNROLL for (int d = 0; d < DVS; d++) ((device float4*)(sout + (hv * 128 + dv0 + d) * 128))[lane] = st[d];
            }
        }
        cur = nxt;
    }
}
#define GDN_RECUR_TREE_INST(S)                                                                                     \
  template [[host_name("gdn_recur_tree_s" #S)]] [[kernel]] void gdn_recur_tree<S>(                                 \
      const device float*, device float*, const device float*, const device float*, const device float*,           \
      const device float*, const device float*, const device float*, const device float*, const device float*,     \
      const device float*, device float*, constant GdnParams&, const device uint*, const device uint*, uint, ushort, ushort);
GDN_RECUR_TREE_INST(1)
GDN_RECUR_TREE_INST(2)
GDN_RECUR_TREE_INST(3)

// Builds the tree on the GPU from the drafter's top-16: best-first by cumulative probability (softmax of
// s/τ over the top-16 of each node, s = selector score with that parent; ≤ kids children per node), like
// tools/sim/h7_tree.py build_tree. Writes the row tokens to tok[1..B−1] (tok[0] = anchor) and the tree buffer in
// preorder. One threadgroup of 256 threads (one per selector component).
struct TreeBuildParams { uint nodes, M, kids; float tau; };

[[kernel]] void dtree_build(const device uint*   cand [[buffer(0)]],   // [M, 16]
                            const device float*  un   [[buffer(1)]],   // [M, 16]
                            const device float*  hp   [[buffer(2)]],   // [M, 256]
                            const device bfloat* pred [[buffer(3)]],
                            const device bfloat* succ [[buffer(4)]],
                            device uint*         tok  [[buffer(5)]],
                            device uint*         tree [[buffer(6)]],
                            constant TreeBuildParams& bp [[buffer(7)]],
                            uint r [[thread_index_in_threadgroup]],
                            ushort sg [[simdgroup_index_in_threadgroup]],
                            ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[8][TK];
    // Nodes in extraction order (0 = anchor): token, parent, depth (anchor 0), cumulative log P.
    threadgroup uint ntok[16], npar[16], ndep[16];
    threadgroup float nlp[16];
    // Frontier: pending candidates (log P, token, parent, depth).
    threadgroup float flp[64];
    threadgroup uint ftok[64], fpar[64], fdep[64];
    threadgroup uint nf, nn, expand;
    if (r == 0) { ntok[0] = tok[0]; npar[0] = 0xffffffffu; ndep[0] = 0; nlp[0] = 0; nf = 0; nn = 1; expand = 0; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint N = min(bp.nodes, 16u);
    for (uint it = 0; it < N; it++) {
        const uint ex = expand;           // node expanded in this iteration
        const uint t = ndep[ex];          // drafter position of its children
        if (t < bp.M) {
            const float a = float(pred[size_t(ntok[ex]) * 256 + r]) * hp[t * 256 + r];
            VUNROLL for (int j = 0; j < TK; j++) {
                const float e = simd_sum(a * float(succ[size_t(cand[t * TK + j]) * 256 + r]));
                if (lane == 0) part[sg][j] = e;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (r == 0) {
            if (t < bp.M) {
                float scs[TK], mx = -INFINITY;
                for (int j = 0; j < TK; j++) {
                    float e = 0;
                    for (int qq = 0; qq < 8; qq++) e += part[qq][j];
                    scs[j] = un[t * TK + j] + e;
                    mx = max(mx, scs[j]);
                }
                float z = 0;
                for (int j = 0; j < TK; j++) z += exp((scs[j] - mx) / bp.tau);
                const float lz = log(z);
                // The kids best by s (ties to the lower index).
                uint used = 0;
                for (uint c = 0; c < bp.kids; c++) {
                    int bj = -1; float bs = -INFINITY;
                    for (int j = 0; j < TK; j++) if (!((used >> j) & 1u) && scs[j] > bs) { bs = scs[j]; bj = j; }
                    if (bj < 0 || nf >= 64) break;
                    used |= 1u << bj;
                    flp[nf] = nlp[ex] + (bs - mx) / bp.tau - lz;
                    ftok[nf] = cand[t * TK + bj]; fpar[nf] = ex; fdep[nf] = t + 1;
                    nf++;
                }
            }
            // Extract the best of the frontier.
            if (nn < N && nf > 0) {
                uint bi = 0;
                for (uint i = 1; i < nf; i++) if (flp[i] > flp[bi]) bi = i;
                ntok[nn] = ftok[bi]; npar[nn] = fpar[bi]; ndep[nn] = fdep[bi]; nlp[nn] = flp[bi];
                expand = nn; nn++;
                nf--; flp[bi] = flp[nf]; ftok[bi] = ftok[nf]; fpar[bi] = fpar[nf]; fdep[bi] = fdep[nf];
            } else {
                expand = 0xffffu;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (expand == 0xffffu || nn >= N) break;
    }
    if (r != 0) return;
    // Preorder with the children of most descendants last; slots as a stack.
    const uint n = nn;
    uint sz[16], kidsOf[16][16], nk[16];
    for (uint i = 0; i < n; i++) { sz[i] = 1; nk[i] = 0; }
    for (int i = int(n) - 1; i > 0; i--) sz[npar[i]] += sz[i];
    for (uint i = 1; i < n; i++) kidsOf[npar[i]][nk[npar[i]]++] = i;
    for (uint i = 0; i < n; i++)   // from fewest to most descendants (insertion, stable)
        for (uint a = 1; a < nk[i]; a++) { uint x = kidsOf[i][a]; int b = int(a) - 1; while (b >= 0 && sz[kidsOf[i][b]] > sz[x]) { kidsOf[i][b + 1] = kidsOf[i][b]; b--; } kidsOf[i][b + 1] = x; }
    uint order[16], rowOf[16], stk[16], sp = 0, no = 0;
    stk[sp++] = 0;
    while (sp > 0) {
        const uint x = stk[--sp];
        rowOf[x] = no; order[no++] = x;
        for (int a = int(nk[x]) - 1; a >= 0; a--) stk[sp++] = kidsOf[x][a];   // the first to come out is kidsOf[x][0]
    }
    tree[0] = n;
    uint slotOf[16], left[16], depthSlots = 0, maxSlots = 0;
    for (uint rr = 0; rr < n; rr++) {
        const uint x = order[rr];
        const uint pr = rr == 0 ? 0xffffffffu : rowOf[npar[x]];
        tok[rr] = ntok[x];
        tree[TR_PAR + rr] = pr;
        tree[TR_DEP + rr] = ndep[x];
        tree[TR_ANC + rr] = (rr == 0 ? 0u : tree[TR_ANC + pr]) | (1u << rr);
        // Conv: the 3 nearest ancestors (oldest to newest); above the anchor, the history.
        // Above the anchor (target depth < 0) go the rows of the history window: −1 → 2, −2 → 1,
        // −3 → 0, that is, index dep + j.
        uint ch[3]; uint a = rr;
        for (int j = 2; j >= 0; j--) {
            if (a != 0) { a = tree[TR_PAR + a]; ch[j] = 3 + a; }
            else ch[j] = ndep[x] + uint(j);
        }
        for (int j = 0; j < 3; j++) tree[TR_CONV + 3 * rr + j] = ch[j];
        for (uint i = 0; i < 8; i++) tree[TR_AROW + 8 * rr + i] = rr;
        for (uint b2 = rr; ; b2 = tree[TR_PAR + b2]) { tree[TR_AROW + 8 * rr + min(tree[TR_DEP + b2], 7u)] = b2; if (b2 == 0) break; }
        tree[TR_SAVE + rr] = 0; tree[TR_REST + rr] = 0;
        left[rr] = nk[x];
    }
    // Stack of slots: after a row with ≥ 2 children it is saved; before a child that does not follow the parent right
    // after it, it is restored; on restoring for the last child it is freed.
    for (uint rr = 1; rr < n; rr++) {
        const uint pr = tree[TR_PAR + rr];
        if (pr != rr - 1) {
            tree[TR_REST + rr] = slotOf[pr] + 1;
            if (--left[pr] == 1) depthSlots--;   // last child: slot free
        } else if (left[pr] >= 2) {
            slotOf[pr] = depthSlots++;
            maxSlots = max(maxSlots, depthSlots);
            tree[TR_SAVE + pr] = slotOf[pr] + 1;
        }
    }
    tree[TR_NS] = maxSlots;
}

// Greedy acceptance over the tree (like spec_accept): descends from the anchor while some child matches the argmax
// of its parent; stores the path, m and P from before in the tree buffer, the round record in rec and advances the
// round state st = [P, nacc, pending context rows, their position, anchor, round, rows].
[[kernel]] void spec_accept_tree(device uint*       st   [[buffer(0)]],
                                 device uint*       vtok [[buffer(1)]],
                                 const device uint* top1 [[buffer(2)]],
                                 device uint*       dtok [[buffer(3)]],
                                 device uint*       rec  [[buffer(4)]],
                                 constant uint&     ring [[buffer(5)]],
                                 device uint*       tree [[buffer(6)]],
                                 uint tid [[thread_position_in_grid]]) {
    if (tid != 0) return;
    const uint n = tree[0];
    uint cur = 0, m = 0;
    tree[TR_PATH] = 0;
    device uint* r = rec + (st[5] % ring) * REC_STRIDE;
    while (true) {
        uint nx = 0xffffffffu;
        for (uint q = cur + 1; q < n; q++) if (tree[TR_PAR + q] == cur && vtok[q] == top1[cur]) { nx = q; break; }
        if (nx == 0xffffffffu) break;
        m++; cur = nx; tree[TR_PATH + m] = cur; r[m] = vtok[cur];
    }
    r[0] = m; r[m + 1] = top1[cur]; r[REC_P] = st[0]; r[REC_ROWS] = n;
    tree[TR_M] = m; tree[TR_P0] = st[0];
    st[3] = st[0]; st[2] = m + 1; st[0] += m + 1; st[1] = m + 1; st[4] = top1[cur]; st[5] += 1;
    vtok[0] = top1[cur]; dtok[0] = top1[cur];
}

// Compaction of the accepted path into contiguous rows: row i ← row path[i] (i = 1..m; path[i] ≥ i and grows, so
// walking i in order inside each thread is enough). KV of an attention layer (slots P₀ + i), with its scales.
[[kernel]] void kv_compact_tree(device char*        kc   [[buffer(0)]],
                                device char*        vt   [[buffer(1)]],
                                device float*       ks   [[buffer(2)]],
                                device float*       vs   [[buffer(3)]],
                                const device uint*  tree [[buffer(4)]],
                                constant uint&      ldkv [[buffer(5)]],
                                uint2 gid [[thread_position_in_grid]]) {
    const uint d = gid.x, h = gid.y, m = tree[TR_M], p0 = tree[TR_P0];
    for (uint i = 1; i <= m; i++) {
        const uint src = p0 + tree[TR_PATH + i], dst = p0 + i;
        if (src == dst) continue;
        kc[(size_t(h) * ldkv + dst) * AT_HD + d] = kc[(size_t(h) * ldkv + src) * AT_HD + d];
        // MLA keeps no vt (V is read from kc; its buffer is a placeholder): only kc and the two scales move.
        if (!AT_MLA) vt[(size_t(h) * AT_HD + d) * ldkv + dst] = vt[(size_t(h) * AT_HD + d) * ldkv + src];
        if (d == 0) { ks[h * ldkv + dst] = ks[h * ldkv + src]; vs[h * ldkv + dst] = vs[h * ldkv + src]; }
    }
}

// Rows of a [rows, n] buffer of 4 or 2 bytes (drafter captures in half; GDN tape in float).
template <typename T>
[[kernel]] void rows_compact_tree(device T*          x    [[buffer(0)]],
                                  const device uint* tree [[buffer(1)]],
                                  constant uint2&    nb   [[buffer(2)]],   // (n per row, base row: 3 in the history)
                                  uint c [[thread_position_in_grid]]) {
    if (c >= nb.x) return;
    const uint m = tree[TR_M];
    for (uint i = 1; i <= m; i++) {
        const uint src = tree[TR_PATH + i];
        if (src != i) x[size_t(nb.y + i) * nb.x + c] = x[size_t(nb.y + src) * nb.x + c];
    }
}
template [[host_name("rows_compact_tree_h")]] [[kernel]] void rows_compact_tree<half>(device half*, const device uint*, constant uint2&, uint);

// Tape and history of the conv of a GDN layer (those of the side the round just wrote): k, v, g, beta (row i) and the
// raw conv input (row 3 + i). One thread per column of the five together.
[[kernel]] void gdn_compact_tree(device float*      tk   [[buffer(0)]],
                                 device float*      tv   [[buffer(1)]],
                                 device float*      tg   [[buffer(2)]],
                                 device float*      tb   [[buffer(3)]],
                                 device float*      hist [[buffer(4)]],
                                 const device uint* tree [[buffer(5)]],
                                 uint c0 [[thread_position_in_grid]]) {
    device float* x; uint n, base = 0, c = c0;
    if (c < GDN_QK) { x = tk; n = GDN_QK; }
    else if ((c -= GDN_QK) < GDN_V) { x = tv; n = GDN_V; }
    else if ((c -= GDN_V) < GDN_HV) { x = tg; n = GDN_HV; }
    else if ((c -= GDN_HV) < GDN_HV) { x = tb; n = GDN_HV; }
    else if ((c -= GDN_HV) < GDN_C) { x = hist; n = GDN_C; base = 3; }
    else return;
    const uint m = tree[TR_M];
    for (uint i = 1; i <= m; i++) {
        const uint src = tree[TR_PATH + i];
        if (src != i) x[size_t(base + i) * n + c] = x[size_t(base + src) * n + c];
    }
}
