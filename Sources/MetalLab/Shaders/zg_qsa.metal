// QSA (Qwen Sparse Attention) of Flash-Next, in isolation (phase F4; the model does not use it yet and runs dense
// attention up to 2048 of history). Semantics of mlx-vlm qwen4_exp (Qwen4ExpQSAIndexer) with what is pinned down by
// splash, TensorFold and deterministic SGLang:
//   - indexer: 4 query heads and 1 key head of 128; (1 + w) norm and RoPE on the first 64 dimensions;
//   - block key (4 tokens): fp32 mean of the 4 raw keys → norm → RoPE at the position of the first token;
//   - score: Σ_h relu(q_h · k_b) / √128 in fp32;
//   - the row at position p sees the complete blocks b < (p + 1) / 4; with more than 512 visible, it attends to the
//     512 top-scoring ones (tie: the lower-index block) and to its open block. Its own block, if complete,
//     competes in the top-512: if it loses, the token does not attend to itself.
//   - with `bf16` (bit 0 of flags), q and the block keys are rounded to bf16 after RoPE (like transformers).

constant constexpr uint QSA_H = 4, QSA_D = 128, QSA_TOPK = 512;

struct QsaParams {
    uint rows;    // rows (queries) of the block
    uint pos0;    // position of the first row
    uint nb;      // blocks with a key (complete in the whole history)
    uint ldk;     // stride of the raw keys
    float eps;
    float theta;
    uint flags;   // bit 0: bf16 rounding after RoPE
    uint pad;
};

inline float qsa_rope(float x, float xp, uint d, float pos, float theta) {
    // Non-interleaved RoPE over 64 dimensions: pairs (i, i + 32), frequency θ^(−i/32).
    if (d >= 64) return x;
    const uint i = d % 32;
    const float ang = pos * pow(theta, -float(i) / 32.0f);
    const float c = cos(ang), s = sin(ang);
    return d < 32 ? x * c - xp * s : x * c + xp * s;
}

// Indexer queries: qk [rows, 640] (4·128 of query | 128 of key) → q [rows, 4, 128] normalized and
// rotated; the raw key of each row is copied to rawk[pos]. Grid (5, rows), 128 threads (the 5th: the key).
[[kernel]] void qsa_prep(const device float*  qk   [[buffer(0)]],
                         const device float*  wq   [[buffer(1)]],   // 1 + w of q_layernorm [128]
                         device float*        q    [[buffer(2)]],
                         device float*        rawk [[buffer(3)]],   // [history, 128]
                         constant QsaParams&  p    [[buffer(4)]],
                         uint2 tg [[threadgroup_position_in_grid]],
                         uint d [[thread_index_in_threadgroup]],
                         ushort sg [[simdgroup_index_in_threadgroup]],
                         ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[4], xs[QSA_D];
    const uint r = tg.y, h = tg.x;
    const float x = qk[size_t(r) * 640 + h * QSA_D + d];
    if (h == QSA_H) { rawk[size_t(p.pos0 + r) * p.ldk + d] = x; return; }
    const float ss = simd_sum(x * x);
    if (lane == 0) part[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float rr = rsqrt((part[0] + part[1] + part[2] + part[3]) / float(QSA_D) + p.eps);
    xs[d] = x * rr * wq[d];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float o = qsa_rope(xs[d], xs[d < 32 ? d + 32 : (d < 64 ? d - 32 : d)], d, float(p.pos0 + r), p.theta);
    if (p.flags & 1) o = float(bfloat(o));
    q[(size_t(r) * QSA_H + h) * QSA_D + d] = o;
}

// New block keys: blocks [b0, b0 + n) from the raw keys. Grid (n), 128 threads.
[[kernel]] void qsa_blockkeys(const device float*  rawk [[buffer(0)]],
                              const device float*  wk   [[buffer(1)]],   // 1 + w of k_layernorm
                              device float*        bk   [[buffer(2)]],   // [blocks, 128]
                              constant QsaParams&  p    [[buffer(3)]],
                              constant uint&       b0   [[buffer(4)]],
                              uint tg [[threadgroup_position_in_grid]],
                              uint d [[thread_index_in_threadgroup]],
                              ushort sg [[simdgroup_index_in_threadgroup]],
                              ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[4], xs[QSA_D];
    const uint b = b0 + tg;
    float m = 0;
    VUNROLL for (uint j = 0; j < 4; j++) m += rawk[size_t(4 * b + j) * p.ldk + d];
    m = m / 4.0f;
    if (p.flags & 1) m = float(bfloat(m));   // the mean is stored in the type of the raw keys (bf16 in the reference)
    const float ss = simd_sum(m * m);
    if (lane == 0) part[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float rr = rsqrt((part[0] + part[1] + part[2] + part[3]) / float(QSA_D) + p.eps);
    xs[d] = m * rr * wk[d];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float o = qsa_rope(xs[d], xs[d < 32 ? d + 32 : (d < 64 ? d - 32 : d)], d, float(4 * b), p.theta);
    if (p.flags & 1) o = float(bfloat(o));
    bk[size_t(b) * QSA_D + d] = o;
}

// Scores: s[r, b] = Σ_h relu(q[r, h]·k_b) / √128 for the blocks visible to the row (−inf for the others).
// One simdgroup per (row, block), 8 per threadgroup: grid (⌈nb / 8⌉, rows).
[[kernel]] void qsa_score(const device float*  q  [[buffer(0)]],
                          const device float*  bk [[buffer(1)]],
                          device float*        s  [[buffer(2)]],   // [rows, nb]
                          constant QsaParams&  p  [[buffer(3)]],
                          uint2 tg [[threadgroup_position_in_grid]],
                          ushort sg [[simdgroup_index_in_threadgroup]],
                          ushort lane [[thread_index_in_simdgroup]]) {
    const uint r = tg.y, b = tg.x * 8 + sg;
    if (b >= p.nb) return;
    const uint vis = (p.pos0 + r + 1) / 4;
    if (b >= vis) { if (lane == 0) s[size_t(r) * p.nb + b] = -INFINITY; return; }
    const float4 kv = ((const device float4*)(bk + size_t(b) * QSA_D))[lane];
    float tot = 0;
    VUNROLL for (uint h = 0; h < QSA_H; h++) {
        const float4 qv = ((const device float4*)(q + (size_t(r) * QSA_H + h) * QSA_D))[lane];
        tot += max(simd_sum(dot(qv, kv)), 0.0f);
    }
    if (lane == 0) s[size_t(r) * p.nb + b] = tot / sqrt(float(QSA_D));
}

// Per-row selection (one threadgroup of 1024 threads): bitmap of attended blocks [rows, ⌈nb / 32⌉]. With ≤ 512
// visible blocks, all of them; otherwise the 512 top-scoring ones with ties going to the lower index: radix search of
// the threshold over the 32-bit sortable key (4 passes of 8 bits), then those equal to the threshold in index order.
inline uint qsa_key(float v) {
    const uint u = as_type<uint>(v);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

[[kernel]] void qsa_select(const device float* s   [[buffer(0)]],
                           device uint*        bm  [[buffer(1)]],
                           constant QsaParams& p   [[buffer(2)]],
                           uint r [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]]) {
    threadgroup atomic_uint hist[256];
    threadgroup uint sh[4];
    const uint nw = (p.nb + 31) / 32, vis = min((p.pos0 + r + 1) / 4, p.nb);
    const device float* sr = s + size_t(r) * p.nb;
    device uint* br = bm + size_t(r) * nw;
    for (uint w = tid; w < nw; w += 1024) br[w] = 0;
    threadgroup_barrier(mem_flags::mem_device);
    if (vis <= QSA_TOPK) {
        for (uint b = tid; b < vis; b += 1024) atomic_fetch_or_explicit((device atomic_uint*)&br[b / 32], 1u << (b % 32), memory_order_relaxed);
        return;
    }
    // Threshold: the key of the 512th score (from highest to lowest).
    uint prefix = 0, need = QSA_TOPK;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (uint i = tid; i < 256; i += 1024) atomic_store_explicit(&hist[i], 0, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint hmask = shift == 24 ? 0u : (0xFFFFFFFFu << (shift + 8));
        for (uint b = tid; b < vis; b += 1024) {
            const uint k = qsa_key(sr[b]);
            if ((k & hmask) == (prefix & hmask)) atomic_fetch_add_explicit(&hist[(k >> shift) & 255u], 1, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            uint acc = 0, digit = 0;
            for (int dgt = 255; dgt >= 0; dgt--) {
                const uint c = atomic_load_explicit(&hist[dgt], memory_order_relaxed);
                if (acc + c >= need) { digit = uint(dgt); break; }
                acc += c;
            }
            sh[0] = prefix | (digit << shift);
            sh[1] = need - acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        prefix = sh[0]; need = sh[1];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // prefix = threshold key; `need` = how many equal to the threshold get in (the lower-index ones).
    for (uint b = tid; b < vis; b += 1024) {
        if (qsa_key(sr[b]) > prefix) atomic_fetch_or_explicit((device atomic_uint*)&br[b / 32], 1u << (b % 32), memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (tid == 0) {
        uint left = need;
        for (uint b = 0; b < vis && left > 0; b++) {
            if (qsa_key(sr[b]) == prefix) { br[b / 32] |= 1u << (b % 32); left--; }
        }
    }
}

// Reference sparse attention (fp32, fp32 KV [history, kvH, 256]; the engine's will run on the int8 KV in F4): row r
// attends to the tokens of the blocks in the bitmap and to its open block [4·vis, p + 1); with ≤ 512 visible
// blocks, to everything causal. One threadgroup of 256 threads (one per dimension) per (row, head).
struct QsaAttnParams { uint rows; uint pos0; uint nb; uint qh; uint kvh; float scale; uint pad0; uint pad1; };

[[kernel]] void qsa_attn_ref(const device float* q   [[buffer(0)]],   // [rows, qh, 256]
                             const device float* k   [[buffer(1)]],   // [history, kvh, 256]
                             const device float* v   [[buffer(2)]],
                             const device uint*  bm  [[buffer(3)]],
                             device float*       o   [[buffer(4)]],   // [rows, qh, 256]
                             constant QsaAttnParams& p [[buffer(5)]],
                             uint2 tg [[threadgroup_position_in_grid]],
                             uint d [[thread_index_in_threadgroup]],
                             ushort sg [[simdgroup_index_in_threadgroup]],
                             ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[8];
    const uint r = tg.y, h = tg.x, kh = h / (p.qh / p.kvh), pos = p.pos0 + r;
    const uint vis = (pos + 1) / 4, nw = (p.nb + 31) / 32;
    const bool sparse = vis > QSA_TOPK;
    const float qd = q[(size_t(r) * p.qh + h) * 256 + d];
    float m = -INFINITY, l = 0, acc = 0;
    for (uint t = 0; t <= pos; t++) {
        const uint b = t / 4;
        if (sparse && b < vis && !((bm[size_t(r) * nw + b / 32] >> (b % 32)) & 1u)) continue;
        const float pd = simd_sum(qd * k[(size_t(t) * p.kvh + kh) * 256 + d]);
        if (lane == 0) part[sg] = pd;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float sc = 0;
        VUNROLL for (uint i = 0; i < 8; i++) sc += part[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sc *= p.scale;
        const float mn = max(m, sc), a = exp(m - mn), e = exp(sc - mn);
        l = l * a + e;
        acc = acc * a + e * v[(size_t(t) * p.kvh + kh) * 256 + d];
        m = mn;
    }
    o[(size_t(r) * p.qh + h) * 256 + d] = acc / l;
}

// ---------------------------------------------------------------------------------------------
// QSA inside the model (FlashQSA, FlashParts.swift). State per attention layer: the raw keys of the open block
// (window of the last 3 positions before P, with the round tape and ping-pong, like the PLE conv:
// the commit after verification comes from `nacc`) and the block keys per block (like the KV: each round recomputes
// those of the blocks it touches). With `dyn` (bit 1 of flags), P = st[0] and nacc = st[1] (chained rounds).
struct QsaMParams {
    uint rows;    // rows of the round or of the chunk
    uint pos0;    // P (without dyn)
    uint nb;      // blocks (upper bound; stride of the scores)
    uint nacc;    // rows accepted in the previous round (without dyn)
    float eps;
    float theta;
    uint flags;   // bit 0: bf16 after RoPE; bit 1: dyn; bit 2: tape (round)
    uint pad;
};

inline uint qsa_P(constant QsaMParams& p, const device uint* st) { return (p.flags & 2) ? st[0] : p.pos0; }
inline uint qsa_nacc(constant QsaMParams& p, const device uint* st) { return (p.flags & 2) ? st[1] : p.nacc; }

// Normalized and rotated queries (position P + r) and raw keys to rk. Grid (5, rows), 128 threads.
[[kernel]] void qsa_m_prep(const device float*  qk  [[buffer(0)]],   // [rows, 640]
                           const device float*  wq  [[buffer(1)]],
                           device float*        q   [[buffer(2)]],   // [rows, 4, 128]
                           device float*        rk  [[buffer(3)]],   // [rows, 128]
                           constant QsaMParams& p   [[buffer(4)]],
                           const device uint*   st  [[buffer(5)]],
                           uint2 tg [[threadgroup_position_in_grid]],
                           uint d [[thread_index_in_threadgroup]],
                           ushort sg [[simdgroup_index_in_threadgroup]],
                           ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[4], xs[QSA_D];
    const uint r = tg.y, h = tg.x;
    const float x = qk[size_t(r) * 640 + h * QSA_D + d];
    if (h == QSA_H) { rk[size_t(r) * QSA_D + d] = x; return; }
    const float ss = simd_sum(x * x);
    if (lane == 0) part[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float rr = rsqrt((part[0] + part[1] + part[2] + part[3]) / float(QSA_D) + p.eps);
    xs[d] = x * rr * wq[d];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float o = qsa_rope(xs[d], xs[d < 32 ? d + 32 : (d < 64 ? d - 32 : d)], d, float(qsa_P(p, st) + r), p.theta);
    if (p.flags & 1) o = float(bfloat(o));
    q[(size_t(r) * QSA_H + h) * QSA_D + d] = o;
}

// Keys of the blocks completed by rows [P, P + rows): b in [P/4, (P + rows)/4); position q < P comes from the
// window (row nacc + 3 − (P − q) of hprev), the others from rk. Grid (upper bound of blocks), 128 threads.
[[kernel]] void qsa_m_blocks(const device float*  rk    [[buffer(0)]],
                             const device float*  hprev [[buffer(1)]],   // [3 + B_prev, 128]
                             const device float*  wk    [[buffer(2)]],
                             device float*        bk    [[buffer(3)]],   // [blocks, 128]
                             constant QsaMParams& p     [[buffer(4)]],
                             const device uint*   st    [[buffer(5)]],
                             uint tg [[threadgroup_position_in_grid]],
                             uint d [[thread_index_in_threadgroup]],
                             ushort sg [[simdgroup_index_in_threadgroup]],
                             ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[4], xs[QSA_D];
    const uint P = qsa_P(p, st), nacc = qsa_nacc(p, st);
    const uint b = P / 4 + tg;
    if (b >= (P + p.rows) / 4) return;
    float m = 0;
    VUNROLL for (uint j = 0; j < 4; j++) {
        const uint qp = 4 * b + j;
        m += qp < P ? hprev[size_t(nacc + 3 - (P - qp)) * QSA_D + d] : rk[size_t(qp - P) * QSA_D + d];
    }
    m = m / 4.0f;
    if (p.flags & 1) m = float(bfloat(m));
    const float ss = simd_sum(m * m);
    if (lane == 0) part[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float rr = rsqrt((part[0] + part[1] + part[2] + part[3]) / float(QSA_D) + p.eps);
    xs[d] = m * rr * wk[d];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float o = qsa_rope(xs[d], xs[d < 32 ? d + 32 : (d < 64 ? d - 32 : d)], d, float(4 * b), p.theta);
    if (p.flags & 1) o = float(bfloat(o));
    bk[size_t(b) * QSA_D + d] = o;
}

// New window: with tape (round), hcur = [hprev rows nacc..nacc + 2 | rk of the rows]; without tape (prefill), the 3
// last of [hprev rows 0..2 | rk]. One thread per value: (3 + rows) × 128 with tape, 3 × 128 without.
[[kernel]] void qsa_m_hist(const device float*  rk    [[buffer(0)]],
                           const device float*  hprev [[buffer(1)]],
                           device float*        hcur  [[buffer(2)]],
                           constant QsaMParams& p     [[buffer(3)]],
                           const device uint*   st    [[buffer(4)]],
                           uint gid [[thread_position_in_grid]]) {
    const uint row = gid / QSA_D, d = gid % QSA_D;
    const uint nacc = qsa_nacc(p, st);
    if (p.flags & 4) {
        if (row >= 3 + p.rows) return;
        hcur[gid] = row < 3 ? hprev[size_t(nacc + row) * QSA_D + d] : rk[size_t(row - 3) * QSA_D + d];
    } else {
        if (row >= 3) return;
        const int i = int(p.rows) - 3 + int(row);   // index in rk (negative: previous window)
        hcur[gid] = i >= 0 ? rk[size_t(i) * QSA_D + d] : hprev[size_t(3 + i) * QSA_D + d];
    }
}

// Scores like qsa_score, with the position of row r = P + r (P from st[0] with dyn) and stride p.nb. One
// simdgroup per block walks up to QSA_RT rows of the batch, so the block key is read once and not once per row;
// each (row, block) is computed as before (dot per lane, simd_sum, per-head ReLU and sum in order): bit-identical
// scores. Grid (⌈nb / 8⌉, ⌈rows / QSA_RT⌉), 256 threads.
constant constexpr uint QSA_RT = 16;

[[kernel]] void qsa_m_score(const device float*  q  [[buffer(0)]],
                            const device float*  bk [[buffer(1)]],
                            device float*        s  [[buffer(2)]],
                            constant QsaMParams& p  [[buffer(3)]],
                            const device uint*   st [[buffer(4)]],
                            uint2 tg [[threadgroup_position_in_grid]],
                            ushort sg [[simdgroup_index_in_threadgroup]],
                            ushort lane [[thread_index_in_simdgroup]]) {
    const uint b = tg.x * 8 + sg;
    if (b >= p.nb) return;
    const uint P = qsa_P(p, st), r0 = tg.y * QSA_RT, r1 = min(r0 + QSA_RT, p.rows);
    // Visible to some row of the span (the last one sees the most): only then is the key read.
    const float4 kv = b < (P + r1) / 4 ? ((const device float4*)(bk + size_t(b) * QSA_D))[lane] : float4(0);
    for (uint r = r0; r < r1; r++) {
        if (b >= (P + r + 1) / 4) { if (lane == 0) s[size_t(r) * p.nb + b] = -INFINITY; continue; }
        float tot = 0;
        VUNROLL for (uint h = 0; h < QSA_H; h++) {
            const float4 qv = ((const device float4*)(q + (size_t(r) * QSA_H + h) * QSA_D))[lane];
            tot += max(simd_sum(dot(qv, kv)), 0.0f);
        }
        if (lane == 0) s[size_t(r) * p.nb + b] = tot / sqrt(float(QSA_D));
    }
}

// Selection like qsa_select (top-512, ties to the lower index; all with ≤ 512 visible), with P from st[0] with dyn.
// No single-thread loops (bit-identical bitmap):
//   - the digit of each radix pass comes from the suffix sum of the histogram (from 255 down) with prefix sums
//     per simdgroup: it is the only d with S(d) ≥ need > S(d) − c(d), the same as the serial walk;
//   - the final bitmap goes by words: each simdgroup takes 32 consecutive blocks (one word) with ballots of "above
//     the threshold" and of "equal"; those equal get in in index order while they fit in `need` (prefix sum across
//     simdgroups plus what earlier iterations carry), like the single-thread loop that walked from block 0.
[[kernel]] void qsa_m_select(const device float*  s   [[buffer(0)]],
                             device uint*         bm  [[buffer(1)]],
                             constant QsaMParams& p   [[buffer(2)]],
                             const device uint*   st  [[buffer(3)]],
                             uint r [[threadgroup_position_in_grid]],
                             uint tid [[thread_index_in_threadgroup]],
                             ushort sgi [[simdgroup_index_in_threadgroup]],
                             ushort ln [[thread_index_in_simdgroup]]) {
    threadgroup atomic_uint hist[256];
    threadgroup uint sh[4];
    threadgroup uint part[32];
    const uint nw = (p.nb + 31) / 32, vis = min((qsa_P(p, st) + r + 1) / 4, p.nb);
    const device float* sr = s + size_t(r) * p.nb;
    device uint* br = bm + size_t(r) * nw;
    for (uint w = tid; w < nw; w += 1024) br[w] = 0;
    threadgroup_barrier(mem_flags::mem_device);
    if (vis <= QSA_TOPK) {
        for (uint b = tid; b < vis; b += 1024) atomic_fetch_or_explicit((device atomic_uint*)&br[b / 32], 1u << (b % 32), memory_order_relaxed);
        return;
    }
    uint prefix = 0, need = QSA_TOPK;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (uint i = tid; i < 256; i += 1024) atomic_store_explicit(&hist[i], 0, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint hmask = shift == 24 ? 0u : (0xFFFFFFFFu << (shift + 8));
        for (uint b = tid; b < vis; b += 1024) {
            const uint k = qsa_key(sr[b]);
            if ((k & hmask) == (prefix & hmask)) atomic_fetch_add_explicit(&hist[(k >> shift) & 255u], 1, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Thread t ↔ digit 255 − t: the inclusive prefix sum at t is S(255 − t), count of that digit and above.
        uint c = 0, incl = 0;
        if (tid < 256) {
            c = atomic_load_explicit(&hist[255 - tid], memory_order_relaxed);
            incl = simd_prefix_inclusive_sum(c);
            if (ln == 31) part[sgi] = incl;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 256) {
            uint S = incl;
            for (uint k = 0; k < sgi; k++) S += part[k];
            if (S >= need && S - c < need) { sh[0] = prefix | ((255 - tid) << shift); sh[1] = need - (S - c); }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        prefix = sh[0]; need = sh[1];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // prefix = threshold key; `need` (≥ 1) = how many equal to the threshold get in (the lower-index ones).
    uint taken = 0;   // those equal that already got in during earlier iterations (uniform across the threadgroup)
    for (uint base = 0; base < vis; base += 1024) {
        const uint b = base + tid;
        const uint k = b < vis ? qsa_key(sr[b]) : 0u;
        const bool gt = b < vis && k > prefix, eq = b < vis && k == prefix;
        const uint gm = uint(simd_vote::vote_t(simd_ballot(gt)));
        const uint em = uint(simd_vote::vote_t(simd_ballot(eq)));
        if (ln == 0) part[sgi] = popcount(em);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint before = taken, tot = taken;
        for (uint j = 0; j < 32; j++) { const uint n = part[j]; if (j < sgi) before += n; tot += n; }
        const bool take = eq && before + popcount(em & ((1u << ln) - 1u)) < need;
        const uint sel = uint(simd_vote::vote_t(simd_ballot(take)));
        if (ln == 0 && base + 32u * sgi < vis) br[base / 32 + sgi] = gm | sel;
        taken = tot;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ---------------------------------------------------------------------------------------------
// The QSA indexer (Y fp32 [M, N] = X fp32 [M, K] · Wᵀ, W bf16) on NAX. X is split into 3 bf16 chunks
// (hi + mid + lo = x exact: 8 + 8 + 8 bits of mantissa) and W is already bf16, so each product is exact in fp32; only
// the summation order changes versus fn_bf16_mm (scalar fp32), on the order of 1 ulp. 64 × 64 tiles, 4 simdgroups
// of 32 × 32, BK = 32 through threadgroup memory. Any M (rows beyond it zeroed, not written). Default in the prefill
// (it meets fidelity versus mlx-vlm at 16K and 32K; NP = 3).
typedef vec<bfloat, 8> nax_bfrag;

METAL_FUNC void nax_load_bf(thread nax_bfrag& d, const threadgroup bfloat* src, uint ld, short2 c) {
    const bfloat4 v0 = *(const threadgroup bfloat4*)(src + c.x * ld + c.y);
    const bfloat4 v1 = *(const threadgroup bfloat4*)(src + (c.x + 8) * ld + c.y);
    d = nax_bfrag(v0, v1);
}

METAL_FUNC void nax_mma2_bf(thread nax_ffrag& c0, thread nax_ffrag& c1, const thread nax_bfrag& a,
                            const thread nax_bfrag& b0, const thread nax_bfrag& b1) {
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        16, 32, 16, false, true, true, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    auto ca = op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
    auto cb = op.template get_right_input_cooperative_tensor<bfloat, bfloat, float>();
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

template <int NP>
[[kernel, max_total_threads_per_threadgroup(128)]] void fn_f32bf16_mm_nax(
    const device float*  x [[buffer(0)]],
    const device bfloat* w [[buffer(1)]],
    device float*        y [[buffer(2)]],
    constant FnMMParams& p [[buffer(3)]],
    uint3  tgid [[threadgroup_position_in_grid]],
    uint   tid  [[thread_index_in_threadgroup]],
    ushort sgid [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]])
{
    constexpr int BM = 64, BN = 64, BK = 32, LD = BK + 8;
    threadgroup bfloat xs[NP][BM * LD];
    threadgroup bfloat ws[BN * LD];
    const uint m0 = tgid.y * BM, n0 = tgid.x * BN;
    const uint tm = (sgid / 2) * 32, tn = (sgid % 2) * 32;
    const short2 c = nax_coord(lane);
    nax_ffrag acc[2][2];
    NAX_UNROLL for (int i = 0; i < 2; i++) NAX_UNROLL for (int j = 0; j < 2; j++) acc[i][j] = 0;
    for (uint k0 = 0; k0 < p.K; k0 += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // X: 64 rows × 32 = 512 float4, 4 per thread; W: 64 × 32 bf16 = 512 bfloat4, 4 per thread.
        NAX_UNROLL for (int i = 0; i < 4; i++) {
            const uint idx = tid + i * 128, r = idx / 8, u = (idx % 8) * 4;
            float4 v = 0;
            if (m0 + r < p.M) v = *(const device float4*)(x + size_t(m0 + r) * p.ldx + k0 + u);
            NAX_UNROLL for (int q = 0; q < NP; q++) {
                const bfloat4 h = bfloat4(v);
                *(threadgroup bfloat4*)(xs[q] + r * LD + u) = h;
                v -= float4(h);
            }
            const uint n = n0 + r;
            bfloat4 wv = bfloat4(bfloat(0.0f));
            if (n < p.N) wv = *(const device bfloat4*)(w + size_t(n) * p.K + k0 + u);
            *(threadgroup bfloat4*)(ws + r * LD + u) = wv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        NAX_UNROLL for (int kk = 0; kk < BK / 16; kk++) {
            nax_bfrag b0, b1;
            nax_load_bf(b0, ws + tn * LD + kk * 16, LD, c);
            nax_load_bf(b1, ws + (tn + 16) * LD + kk * 16, LD, c);
            NAX_UNROLL for (int i = 0; i < 2; i++) {
                // From the small chunk to the big one: the rounding error of the sum stays in the low bits.
                NAX_UNROLL for (int q = NP - 1; q >= 0; q--) {
                    nax_bfrag a;
                    nax_load_bf(a, xs[q] + (tm + i * 16) * LD + kk * 16, LD, c);
                    nax_mma2_bf(acc[i][0], acc[i][1], a, b0, b1);
                }
            }
        }
    }
    NAX_UNROLL for (int i = 0; i < 2; i++) NAX_UNROLL for (int j = 0; j < 2; j++) NAX_UNROLL for (int e = 0; e < 2; e++) {
        const uint r = m0 + tm + i * 16 + c.x + e * 8, col = n0 + tn + j * 16 + c.y;
        if (r >= p.M || col >= p.N) continue;
        *(device float4*)(y + size_t(r) * p.ldy + col) =
            float4(acc[i][j][e * 4], acc[i][j][e * 4 + 1], acc[i][j][e * 4 + 2], acc[i][j][e * 4 + 3]);
    }
}
template [[host_name("fn_f32bf16_mm_nax3")]] [[kernel]] void fn_f32bf16_mm_nax<3>(const device float*,
    const device bfloat*, device float*, constant FnMMParams&, uint3, uint, ushort, ushort);
