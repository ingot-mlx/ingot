import Foundation

// Copy-from-context speculation chained with DFlash2 (Strata-style policy). Translated and adapted from Strata
// (Niko1221/Strata, `src/spec/suffix_drafter.cpp` and `src/spec/draft_policy.cpp` at e88ee39; MIT, Copyright (c) 2026
// Niko1221 and the Strata contributors; see THIRD_PARTY_NOTICES.md).
//
// Each round, if the history (prompt + generated) ends in a trigram that appeared before, propose what followed the
// occurrence with the longest match (min 3, max 64 tokens). A policy estimates the hit rate of each copied token by
// match-length bucket (< 6, < 12, < 24, ≥ 24; priors 0.75/0.88/0.93/0.96) and picks
// the copy round (anchor + T − 1 copied tokens, T ≤ 16) only if it gives more expected tokens per ms than the measured
// DFlash2 round (3% margin). The copy round does not pay for the drafter: only the verification of T rows (and writing
// the previous round's context into the drafter). Verification is the same as DFlash2's (greedy: argmax;
// with temperature: rejection against a deterministic draft), so the output does not change.

/// Trigram index of the history (4-way, most recent occurrences) with longest-suffix-match lookup.
/// Like Strata's `SuffixDrafter`.
public final class CopyIndex {
    public static let ways = 4
    public let minMatch: Int, maxMatch: Int
    public private(set) var hist: [UInt32] = []
    private var keys: [UInt64] = []
    private var pos: [UInt32] = []  // [slot · ways + w]
    private var cnt: [UInt8] = []
    private var mask = 0
    private var used = 0
    public private(set) var lastMatch = 0

    public init(minMatch: Int = 3, maxMatch: Int = 64, capacity: Int = 1 << 16) {
        self.minMatch = max(3, minMatch)
        self.maxMatch = max(self.minMatch, maxMatch)
        allocate(capacity)
    }

    private func allocate(_ capacityTokens: Int) {
        var cap = 1
        while cap < capacityTokens * 2 { cap <<= 1 }  // occupancy ≤ 0.5 at nominal capacity
        keys = [UInt64](repeating: 0, count: cap)
        pos = [UInt32](repeating: 0, count: cap * Self.ways)
        cnt = [UInt8](repeating: 0, count: cap)
        mask = cap - 1
        used = 0
    }

    @inline(__always) static func mix(_ x0: UInt64) -> UInt64 {
        var x = x0
        x ^= x >> 33; x = x &* 0xff51afd7ed558ccd
        x ^= x >> 33; x = x &* 0xc4ceb9fe1a85ec53
        return x ^ (x >> 33)
    }

    @inline(__always) func keyAt(_ end: Int) -> UInt64 {
        let a = UInt64(hist[end - 2]), b = UInt64(hist[end - 1]), c = UInt64(hist[end])
        return Self.mix((a &* 0x9E3779B97F4A7C15) ^ Self.mix(b &+ 0x632BE59BD9B4E019) ^ (c << 1)) | 1
    }

    @inline(__always) func find(_ key: UInt64) -> Int? {
        var i = Int(key & UInt64(mask))
        while true {
            let k = keys[i]
            if k == key { return i }
            if k == 0 { return nil }
            i = (i + 1) & mask
        }
    }

    private func index(_ end: Int) {
        if used * 2 > keys.count { rebuild(capacity: hist.count * 2); return }
        let key = keyAt(end)
        var i = Int(key & UInt64(mask))
        while keys[i] != 0 && keys[i] != key { i = (i + 1) & mask }
        if keys[i] == 0 { keys[i] = key; used += 1 }
        let b = i * Self.ways
        for w in stride(from: Self.ways - 1, to: 0, by: -1) { pos[b + w] = pos[b + w - 1] }
        pos[b] = UInt32(end)
        if cnt[i] < UInt8(Self.ways) { cnt[i] += 1 }
    }

    private func rebuild(capacity: Int) {
        allocate(max(capacity, 1 << 16))
        let h = hist
        hist = []
        hist.reserveCapacity(h.count)
        append(h)
    }

    public func reset() {
        hist.removeAll(keepingCapacity: true)
        if used > 0 { allocate((keys.count / 2)) }
        lastMatch = 0
    }

    public func append(_ toks: [UInt32]) { for t in toks { append(t) } }

    @inline(__always) public func append(_ t: UInt32) {
        hist.append(t)
        let end = hist.count - 1
        if end >= 2 { index(end) }
    }

    /// Sets the history to `tokens`: if what is indexed is a prefix, only appends what is missing; otherwise starts over.
    public func sync(to tokens: [UInt32]) {
        let n = hist.count
        let isPrefix =
            n <= tokens.count
            && hist.withUnsafeBufferPointer { a in
                tokens.withUnsafeBufferPointer { b in memcmp(a.baseAddress, b.baseAddress, n * 4) == 0 }
            }
        if !isPrefix {
            reset()
            if keys.count < tokens.count * 2 { allocate(tokens.count + 8192) }
        }
        hist.reserveCapacity(tokens.count + 8192)
        for i in hist.count..<tokens.count { append(tokens[i]) }
    }

    /// Continuation of the previous occurrence of the current suffix with the longest match (≥ minMatch), up to
    /// `maxK` tokens; empty if there is none. Leaves the match length in `lastMatch`.
    public func propose(maxK: Int) -> [UInt32] {
        lastMatch = 0
        let n = hist.count
        guard n >= 4, maxK > 0, let s = find(keyAt(n - 1)) else { return [] }
        let cur = n - 1
        var bestEnd = 0, bestLen = 0
        hist.withUnsafeBufferPointer { h in
            for w in 0..<Int(cnt[s]) {
                let p = Int(pos[s * Self.ways + w])
                if p >= cur { continue }
                var len = 0
                while len < maxMatch && len <= p && h[p - len] == h[cur - len] { len += 1 }
                if len > bestLen { bestLen = len; bestEnd = p }  // most recent first: on ties, the newer one
            }
        }
        guard bestLen >= minMatch else { return [] }
        lastMatch = bestLen
        // The continuation may run into the current suffix (periodic text): reading up to `cur` is fine.
        let end = min(cur, bestEnd + maxK)
        return Array(hist[(bestEnd + 1)...end])
    }
}

/// Copy policy (Strata's DraftPolicy adapted to DFlash2): measured cost of the copy round by number of rows
/// (forgetting average; before measuring, the relative `shape`), tokens per ms of the measured DFlash2 rounds and
/// hit rate per match-length bucket.
public struct CopyPolicy {
    /// Max rows of the copy round (anchor included) and minimum advantage over DFlash2 in tokens per ms.
    public static let maxRows = 16, defaultMargin = 0.03
    static let priorQ: [Double] = [0.75, 0.88, 0.93, 0.96]
    static let priorN = 4.0, decay = 0.97, costAlpha = 0.1, baseAlpha = 0.05, probes = 3.0, staleRounds = 300.0
    /// Relative cost of verifying T rows (T = 1...16): for unmeasured sizes, or for all of them when untimed
    /// (`defaultShape`).
    public var shape: [Double]
    public let maxT: Int
    public let margin: Double
    var cost = [Double](repeating: 0, count: 17), costN = [Double](repeating: 0, count: 17),
        last = [Double](repeating: 0, count: 17)
    var ok = [Double](repeating: 0, count: 4), bad = [Double](repeating: 0, count: 4)
    var baseTok = 0.0, baseMs = 0.0, baseN = 0.0
    var rounds = 0.0

    /// true: the reference round is the normal one-row round (no drafter, LFM2); false: DFlash2 (drafter + ≤ 8 rows).
    public let plainBase: Bool

    public init(maxT: Int, margin: Double = 0.03, shape: [Double] = CopyPolicy.defaultShape, plainBase: Bool = false) {
        self.maxT = min(max(maxT, 2), 16)
        self.margin = margin
        self.shape = shape
        self.plainBase = plainBase
    }

    /// Verification of T rows relative to 8 rows (index T; 0 unused). Qwen3.6 Heretic calibrated, P = 3059, GPU per
    /// command buffer with consecutive prompt tokens: 13.4 ms
    /// with 1 row, 25.6 with 8 and 35.6 with 16 (+39%); at 22.7K the shape is almost the same. The drafter block costs
    /// 4.0 ms.
    nonisolated(unsafe) public static var defaultShape: [Double] = [
        0, 0.52, 0.61, 0.69, 0.76, 0.82, 0.87, 0.96, 1.0, 1.03, 1.07, 1.13, 1.17, 1.26, 1.32, 1.36, 1.39,
    ]

    public static func bucket(_ m: Int) -> Int { m < 6 ? 0 : m < 12 ? 1 : m < 24 ? 2 : 3 }

    public func rate(_ match: Int) -> Double {
        let b = Self.bucket(match)
        return (ok[b] + Self.priorN * Self.priorQ[b]) / (ok[b] + bad[b] + Self.priorN)
    }

    /// ms of a T-row copy round: the measured one, or the shape scaled from the measurements (or from the DFlash2 round,
    /// which is drafter + verification of ≤ 8 rows, if there is none yet).
    public func costMs(_ t: Int) -> Double {
        let t = min(max(t, 1), 16)
        if costN[t] > 0 { return cost[t] }
        var num = 0.0, den = 0.0
        for u in 1...16 where costN[u] > 0 {
            let w = min(costN[u], 20)
            num += w * cost[u] * shape[t] / shape[u]
            den += w
        }
        if den > 0 { return num / den }
        // DFlash2: verifying 8 rows is ~86% of the round; without a drafter, the normal round is the 1-row one.
        return plainBase ? (baseN > 0 ? baseMs : 10) * shape[t] / shape[1] : (baseN > 0 ? baseMs : 30) * 0.86 * shape[t]
    }

    func stale(_ t: Int) -> Bool { costN[t] > 0 && rounds - last[t] > Self.staleRounds }

    /// Tokens per ms of DFlash2 (what has to be beaten).
    public var baseRate: Double { baseN > 0 && baseMs > 0 ? baseTok / baseMs : plainBase ? 0.1 : 3.6 / 28 }

    /// Copy round for a proposal of `k` tokens with match length `match`? Returns the rows (anchor + copied) or nil.
    public func choose(k: Int, match: Int) -> Int? {
        guard k > 0 else { return nil }
        let q = rate(match)
        var e = 1.0, qi = 1.0, best = 0.0, bestT = 0
        let kmax = min(k, maxT - 1)
        for kk in 1...kmax {
            qi *= q
            e += qi
            let r = e / costMs(kk + 1)
            if r > best { best = r; bestT = kk + 1 }
        }
        let tFull = kmax + 1
        if bestT > 0 && best > baseRate * (1 + margin) {
            if bestT < tFull && q >= 0.85 && stale(tFull) { return tFull }
            return bestT
        }
        // An unmeasured size is probed a few times if the copy looks safe (verification does not change the output).
        if (costN[tFull] < Self.probes || stale(tFull)) && q >= 0.85 { return tFull }
        return nil
    }

    /// T-row copy round with `accepted` copied tokens accepted (of T − 1) and its cost in ms.
    public mutating func observeCopy(rows t: Int, accepted: Int, match: Int, ms: Double) {
        let t = min(max(t, 1), 16)
        if ms > 0 {
            cost[t] = costN[t] > 0 ? (1 - Self.costAlpha) * cost[t] + Self.costAlpha * ms : ms
            costN[t] += 1
            last[t] = rounds
        }
        rounds += 1
        let b = Self.bucket(match)
        ok[b] = Self.decay * ok[b] + Double(accepted)
        bad[b] = Self.decay * bad[b] + (accepted < t - 1 ? 1 : 0)
    }

    /// DFlash2 round with `tokens` emitted in `ms`.
    public mutating func observeBase(tokens: Int, ms: Double) {
        rounds += 1
        guard ms > 0 else { return }
        if baseN == 0 {
            baseTok = Double(tokens); baseMs = ms
        } else {
            baseTok = (1 - Self.baseAlpha) * baseTok + Self.baseAlpha * Double(tokens)
            baseMs = (1 - Self.baseAlpha) * baseMs + Self.baseAlpha * ms
        }
        baseN += 1
    }
}

/// Policy kept across requests (InferenceEngine): measured costs and per-bucket rates do not go back to the priors
/// on every turn of an agent.
public final class CopyPolicyState {
    public var policy: CopyPolicy?
    public init() {}
}
