// Temperature sampling in the chained round (goes after yy_model: uses tk_better).
//
// topk64_part + topk64_merge: per row, the 64 largest logits (sorted, ties by lower id), the max and
// Σ exp((x − max)/T) over the whole vocabulary (with top_k = 0 the top-p probabilities are the true ones, not those
// of the 64).
// spec_accept_sample: speculative verification by rejection with deterministic drafts (those of DFlash2): the
// draft d of row i is accepted with probability p_i(d); if rejected, the token comes from p_i without d
// (renormalized), i.e. max(0, p − q) normalized with q = δ_d. So the output follows exactly the distribution of the
// large model with temperature, top-k, min-p and top-p (temperature → top-k → min-p → top-p, the same as vLLM's
// temperature → min-p → top-k → top-p because min-p and top-k commute). Limit: the nucleus and the min-p set are
// searched among the 64 candidates (exact with 1 ≤ top_k ≤ 64; with top_k = 0 and a set over 64 tokens it stays
// at 64).

constant constexpr int SK = 64;

struct SampleParams {
    float temp;
    float top_p;
    uint top_k;
    uint V;
    float min_p;   // 0 = off
    float rep;     // repetition penalty (1 = off)
    float pres;    // presence penalty (0 = off)
    float freq;    // frequency penalty (0 = off)
};

// Two-pass top-64 (the same candidates as a single pass over the whole row in one threadgroup, which took ~0.9 ms
// with a 262K vocabulary): topk64_part splits each row into SPG chunks, one threadgroup each, and writes the sorted
// top-64 of its chunk and its (max, Σ exp((x − max)/T)); topk64_merge merges the SPG lists of a row and combines the
// partial sums in chunk order. The candidates (ids and values) are those of the whole row; Z can differ in the last
// bits from a single pass (another summation order).
constant constexpr uint SPG = 32;    // chunks (threadgroups) per row; topk64_merge gives one lane to each chunk
constant constexpr uint SPNT = 256;  // threads per threadgroup of topk64_part

/// Inserts (cv, cid) into the sorted list of 64 held by the simdgroup (entry l in lane l of (av, ai), entry 32 + l
/// in lane l of (bv, bi)), dropping the last; cv and cid are the same in all lanes.
inline void sg_insert64(float cv, uint cid, ushort lane, thread float& av, thread uint& ai, thread float& bv,
                        thread uint& bi) {
    // The list is sorted, so ga (and gb) is false up to the insertion point and true from there on.
    const ushort ga = tk_better(cv, cid, av, ai), gb = tk_better(cv, cid, bv, bi);
    const float upa = simd_shuffle_up(av, 1), upb = simd_shuffle_up(bv, 1);
    const uint upai = simd_shuffle_up(ai, 1), upbi = simd_shuffle_up(bi, 1);
    const ushort upga = simd_shuffle_up(ga, 1), upgb = simd_shuffle_up(gb, 1);
    const float a31 = simd_broadcast(av, 31);
    const uint a31i = simd_broadcast(ai, 31);
    const ushort ga31 = simd_broadcast(ga, 31);
    const bool pga = lane > 0 && upga != 0;
    const float pbv = lane > 0 ? upb : a31;
    const uint pbi = lane > 0 ? upbi : a31i;
    const bool pgb = (lane > 0 ? upgb : ga31) != 0;
    if (gb) { bv = pgb ? pbv : cv; bi = pgb ? pbi : cid; }
    if (ga) { av = pga ? upa : cv; ai = pga ? upai : cid; }
}

template <bool EACH>
[[kernel]] void topk64_part_t(const device float* x    [[buffer(0)]],
                              device uint*        pidx [[buffer(1)]],   // [rows, SPG, 64]
                              device float*       pval [[buffer(2)]],   // [rows, SPG, 64]
                              device float*       pz   [[buffer(3)]],   // [rows, SPG, 2]: max, Σ exp((x − max)/T)
                              constant SampleParams* ps [[buffer(4)]],  // [1] or [rows]
                              uint2 tg [[threadgroup_position_in_grid]], // (chunk, row)
                              uint tid [[thread_index_in_threadgroup]],
                              ushort sg [[simdgroup_index_in_threadgroup]],
                              ushort lane [[thread_index_in_simdgroup]]) {
    const uint g = tg.x, row = tg.y;
    constant SampleParams& p = ps[EACH ? row : 0];
    constexpr uint NSG = SPNT / 32;
    threadgroup float sv[NSG * SK];
    threadgroup uint si[NSG * SK];
    threadgroup float rm[NSG], rs[NSG];
    const device float* xr = x + size_t(row) * p.V;
    const float it = 1.0f / p.temp;
    const uint C = (p.V + SPG - 1) / SPG, lo = g * C, hi = min(p.V, lo + C);
    // The simdgroup's sorted top-64 (two entries per lane) and the lane's online max and sum.
    float av = -FLT_MAX, bv = -FLT_MAX;
    uint ai = 0xffffffffu, bi = 0xffffffffu;
    float mt = -FLT_MAX, s = 0;
    for (uint base = lo + sg * 32; base < hi; base += SPNT) {
        const uint ci = base + lane;
        const bool ok = ci < hi;
        const float c = ok ? xr[ci] : -FLT_MAX;
        if (ok) {
            if (c > mt) { s = s * exp((mt - c) * it) + 1.0f; mt = c; } else s += exp((c - mt) * it);
        }
        // The simd broadcasts outside the && (all lanes must take part).
        const float tv = simd_broadcast(bv, 31);
        const uint ti = simd_broadcast(bi, 31);
        uint want = uint(simd_vote::vote_t(simd_ballot(ok && tk_better(c, ci, tv, ti))));
        while (want != 0) {
            const uint k = ctz(want);
            want &= want - 1;
            const float cv = simd_broadcast(c, ushort(k));
            const uint cid = simd_broadcast(ci, ushort(k));
            // The list may have grown past it since the ballot.
            if (tk_better(cv, cid, simd_broadcast(bv, 31), simd_broadcast(bi, 31))) {
                sg_insert64(cv, cid, lane, av, ai, bv, bi);
            }
        }
    }
    const float m = simd_max(mt);
    const float ss = simd_sum(s * exp((mt - m) * it));
    sv[sg * SK + lane] = av; si[sg * SK + lane] = ai;
    sv[sg * SK + 32 + lane] = bv; si[sg * SK + 32 + lane] = bi;
    if (lane == 0) { rm[sg] = m; rs[sg] = ss; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg != 0) return;
    // Merge of the NSG sorted lists: 64 rounds taking the best list head (lane l < NSG holds list l).
    const size_t o = (size_t(row) * SPG + g) * SK;
    uint h = 0;
    for (int r = 0; r < SK; r++) {
        const bool live = lane < NSG && h < uint(SK);
        const float hv = live ? sv[lane * SK + h] : -FLT_MAX;
        const uint hid = live ? si[lane * SK + h] : 0xffffffffu;
        const float best = simd_max(hv);
        const uint bid = simd_min(hv == best ? hid : 0xffffffffu);
        if (lane == 0) { pval[o + r] = best; pidx[o + r] = bid; }
        if (hv == best && hid == bid) h++;
    }
    if (lane == 0) {
        float M = rm[0];
        for (uint i = 1; i < NSG; i++) M = max(M, rm[i]);
        float Z = 0;
        for (uint i = 0; i < NSG; i++) Z += rs[i] * exp((rm[i] - M) * it);
        pz[(size_t(row) * SPG + g) * 2] = M;
        pz[(size_t(row) * SPG + g) * 2 + 1] = Z;
    }
}

/// One simdgroup per row: lane l holds the head of chunk l's list (SPG = 32 lists).
template <bool EACH>
[[kernel]] void topk64_merge_t(const device uint*  pidx [[buffer(0)]],
                               const device float* pval [[buffer(1)]],
                               const device float* pz   [[buffer(2)]],
                               device uint*        idx  [[buffer(3)]],   // [rows, 64]
                               device float*       val  [[buffer(4)]],   // [rows, 64]
                               device float*       zout [[buffer(5)]],   // [rows, 2]
                               constant SampleParams* ps [[buffer(6)]],
                               uint row [[threadgroup_position_in_grid]],
                               ushort lane [[thread_index_in_simdgroup]]) {
    static_assert(SPG == 32, "one lane per chunk");
    constant SampleParams& p = ps[EACH ? row : 0];
    const float it = 1.0f / p.temp;
    const size_t o = (size_t(row) * SPG + lane) * SK;
    uint h = 0;
    float hv = pval[o];
    uint hid = pidx[o];
    float nv = pval[o + 1];
    uint nid = pidx[o + 1];
    for (int r = 0; r < SK; r++) {
        const float best = simd_max(hv);
        const uint bid = simd_min(hv == best ? hid : 0xffffffffu);
        if (lane == 0) { val[row * SK + r] = best; idx[row * SK + r] = bid; }
        if (hv == best && hid == bid) {
            h++;
            hv = nv; hid = nid;
            const bool more = h + 1 < uint(SK);
            nv = more ? pval[o + h + 1] : -FLT_MAX;
            nid = more ? pidx[o + h + 1] : 0xffffffffu;
        }
    }
    const float m = pz[(size_t(row) * SPG + lane) * 2];
    const float M = simd_max(m);
    if (lane == 0) {
        float Z = 0;
        for (uint i = 0; i < SPG; i++) {
            const size_t q = (size_t(row) * SPG + i) * 2;
            Z += pz[q + 1] * exp((pz[q] - M) * it);
        }
        zout[row * 2] = M;
        zout[row * 2 + 1] = Z;
    }
}

#define TOPK64_PART(NAME, EACH)                                                                                      \
    template [[host_name(NAME)]] [[kernel]] void topk64_part_t<EACH>(                                                \
        const device float* x [[buffer(0)]], device uint* pidx [[buffer(1)]], device float* pval [[buffer(2)]],      \
        device float* pz [[buffer(3)]], constant SampleParams* ps [[buffer(4)]],                                     \
        uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],                         \
        ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]);
TOPK64_PART("topk64_part", false)
TOPK64_PART("topk64_part_each", true)
#undef TOPK64_PART
#define TOPK64_MERGE(NAME, EACH)                                                                                     \
    template [[host_name(NAME)]] [[kernel]] void topk64_merge_t<EACH>(                                               \
        const device uint* pidx [[buffer(0)]], const device float* pval [[buffer(1)]],                               \
        const device float* pz [[buffer(2)]], device uint* idx [[buffer(3)]], device float* val [[buffer(4)]],       \
        device float* zout [[buffer(5)]], constant SampleParams* ps [[buffer(6)]],                                   \
        uint row [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]]);
TOPK64_MERGE("topk64_merge", false)
TOPK64_MERGE("topk64_merge_each", true)
#undef TOPK64_MERGE

/// Probabilities of a row's candidates after temperature, top-k, min-p and top-p (renormalized); returns how many
/// remain.
inline uint sample_dist(const device uint* ci, const device float* cv, float M, float Z, constant SampleParams& p,
                        thread float* pr) {
    const float it = 1.0f / p.temp;
    uint n = 0;
    while (n < uint(SK) && ci[n] != 0xffffffffu) n++;
    float tot = Z;
    if (p.top_k > 0 && p.top_k < n) n = p.top_k;
    if (p.top_k > 0) { tot = 0; for (uint j = 0; j < n; j++) tot += exp((cv[j] - M) * it); }
    for (uint j = 0; j < n; j++) pr[j] = exp((cv[j] - M) * it) / tot;
    if (p.min_p > 0.0f) {
        // Candidates are sorted: keep the prefix with p ≥ min_p · p_max and renormalize (top-p sees the kept set).
        const float thr = p.min_p * pr[0];
        uint k = 1;
        while (k < n && pr[k] >= thr) k++;
        n = k;
        float s = 0;
        for (uint j = 0; j < n; j++) s += pr[j];
        for (uint j = 0; j < n; j++) pr[j] /= s;
    }
    if (p.top_p < 1.0f) {
        float c = 0;
        uint k = n;
        for (uint j = 0; j < n; j++) { c += pr[j]; if (c >= p.top_p) { k = j + 1; break; } }
        n = max(k, 1u);
    }
    float s = 0;
    for (uint j = 0; j < n; j++) s += pr[j];
    for (uint j = 0; j < n; j++) pr[j] /= s;
    return n;
}

/// Token from pr[0..n) (without `skip`, whose mass is q) with the uniform u.
inline uint sample_pick(const device uint* ci, thread const float* pr, uint n, uint skip, float q, float u) {
    const float target = u * (1.0f - q);
    float acc = 0;
    uint last = 0xffffffffu;
    for (uint j = 0; j < n; j++) {
        if (ci[j] == skip) continue;
        acc += pr[j];
        last = ci[j];
        if (acc > target) return ci[j];
    }
    return last == 0xffffffffu ? ci[0] : last;
}

// Like spec_accept, but sampling: rnd has RND_PER_ROUND = 32 uniforms per round (2 per row: accept and
// resample), at position st[5] (round counter). Writes the same into rec and the state.
[[kernel]] void spec_accept_sample(device uint*        st   [[buffer(0)]],
                                   device uint*        vtok [[buffer(1)]],
                                   const device uint*  cidx [[buffer(2)]],
                                   const device float* cval [[buffer(3)]],
                                   const device float* z    [[buffer(4)]],
                                   device uint*        dtok [[buffer(5)]],
                                   device uint*        rec  [[buffer(6)]],
                                   const device float* rnd  [[buffer(7)]],
                                   constant SampleParams& p [[buffer(8)]],
                                   constant uint&      ring [[buffer(9)]],
                                   constant uint&      rows [[buffer(10)]],   // verified rows (≤ 16; 0 = st[6])
                                   uint tid [[thread_position_in_grid]]) {
    if (tid != 0) return;
    const uint nr = rows ? rows : st[6];   // 0: rows chosen by spec_rows_choose
    const device float* u = rnd + size_t(st[5]) * RND_PER_ROUND;
    float pr[SK];
    uint m = 0, next = 0;
    for (uint i = 0; i < nr; i++) {
        const device uint* ci = cidx + i * SK;
        const uint n = sample_dist(ci, cval + i * SK, z[2 * i], z[2 * i + 1], p, pr);
        if (i + 1 < nr) {
            const uint d = vtok[i + 1];
            float q = 0;
            for (uint j = 0; j < n; j++) if (ci[j] == d) q = pr[j];
            if (u[2 * i] < q) { m++; continue; }
            next = sample_pick(ci, pr, n, d, q, u[2 * i + 1]);
        } else {
            next = sample_pick(ci, pr, n, 0xffffffffu, 0, u[2 * i + 1]);
        }
        break;
    }
    device uint* r = rec + (st[5] % ring) * REC_STRIDE;
    r[0] = m;
    r[REC_ROWS] = nr;
    for (uint i = 1; i <= m; i++) r[i] = vtok[i];
    r[m + 1] = next;
    r[REC_P] = st[0];
    st[3] = st[0]; st[2] = m + 1; st[0] += m + 1; st[1] = m + 1; st[4] = next; st[5] += 1;
    vtok[0] = next; dtok[0] = next;
}

// Penalties (vLLM/OpenAI semantics, no window), only when a request asks for them. cnt[V]: occurrences of each token
// among the generated ones, bit 31 = in the prompt (only with repetition); touched = [n, the n tokens with
// cnt ≠ 0]. penalize_rows goes after the verification and before topk64_part, so the 64 candidates and the
// normalizer of the whole vocabulary are already penalized and the rejection sampling stays exact (the distribution
// of each row is a function of its prefix). count_emitted adds the round's emitted tokens after the accept kernel.

constant constexpr uint CNT_PROMPT = 0x80000000u;

/// Penalty of one token with c occurrences among the generated ones (vLLM order: repetition, frequency, presence).
/// precise::divide and the explicit fma keep it bit-identical to the CPU reference under fast math.
inline float penalize_one(float x, uint c, bool in_prompt, constant SampleParams& p) {
    if (in_prompt || c > 0) x = x > 0.0f ? precise::divide(x, p.rep) : x * p.rep;
    if (c > 0) { x = fma(-p.freq, float(c), x); x -= p.pres; }
    return x;
}

// One threadgroup per row: row i follows the history plus vtok[1...i] (vtok[0], the anchor, is already counted).
// Each token is touched once per row: the counted ones through `touched`, the others at their first draft.
[[kernel]] void penalize_rows(device float*       x       [[buffer(0)]],   // [rows, V], in place
                              const device uint*  cnt     [[buffer(1)]],
                              const device uint*  touched [[buffer(2)]],
                              const device uint*  vtok    [[buffer(3)]],
                              constant SampleParams& p    [[buffer(4)]],
                              uint row [[threadgroup_position_in_grid]],
                              uint tid [[thread_index_in_threadgroup]],
                              uint ntg [[threads_per_threadgroup]]) {
    device float* xr = x + size_t(row) * p.V;
    const uint nt = touched[0];
    for (uint j = tid; j < nt; j += ntg) {
        const uint t = touched[1 + j];
        uint c = cnt[t] & ~CNT_PROMPT;
        for (uint k = 1; k <= row; k++) c += vtok[k] == t ? 1u : 0u;
        xr[t] = penalize_one(xr[t], c, (cnt[t] & CNT_PROMPT) != 0, p);
    }
    for (uint k = 1 + tid; k <= row; k += ntg) {
        const uint t = vtok[k];
        if (t >= p.V || cnt[t] != 0) continue;
        bool first = true;
        for (uint q = 1; q < k; q++) first = first && vtok[q] != t;
        if (!first) continue;
        uint c = 0;
        for (uint q = k; q <= row; q++) c += vtok[q] == t ? 1u : 0u;
        xr[t] = penalize_one(xr[t], c, false, p);
    }
}

// Adds the tokens of the last round's record (rec[(st[5] − 1) % ring]: m, tokens 1...m + 1) to the counters.
[[kernel]] void count_emitted(const device uint* st      [[buffer(0)]],
                              const device uint* rec     [[buffer(1)]],
                              device uint*       cnt     [[buffer(2)]],
                              device uint*       touched [[buffer(3)]],
                              constant uint&     ring    [[buffer(4)]],
                              constant uint&     V       [[buffer(5)]],
                              uint tid [[thread_position_in_grid]]) {
    if (tid != 0) return;
    const device uint* r = rec + ((st[5] - 1) % ring) * REC_STRIDE;
    for (uint i = 1; i <= r[0] + 1; i++) {
        const uint t = r[i];
        if (t >= V) continue;
        if (cnt[t] == 0) { touched[0] += 1; touched[touched[0]] = t; }
        cnt[t] += 1;
    }
}
