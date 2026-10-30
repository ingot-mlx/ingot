import Testing

@testable import Ingot

/// Context-copy index (CopySpec.swift) against a naive search with the same rule: among the 4 most recent
/// earlier occurrences of the last trigram, the one with the longest suffix match (ties: the most
/// recent), min 3 and max 64; proposes what followed.
struct CopySpecTests {
    static func naive(_ h: [UInt32], maxK: Int) -> ([UInt32], Int) {
        let n = h.count
        guard n >= 4 else { return ([], 0) }
        let cur = n - 1
        var occ: [Int] = []
        for p in stride(from: cur - 1, through: 2, by: -1)
        where h[p] == h[cur] && h[p - 1] == h[cur - 1] && h[p - 2] == h[cur - 2] {
            occ.append(p)
        }
        // The index keeps the 4 most recent counting the current position itself (which is skipped).
        let cand = Array(([cur] + occ).prefix(4).dropFirst())
        var bestEnd = 0, bestLen = 0
        for p in cand {
            var len = 0
            while len < 64 && len <= p && h[p - len] == h[cur - len] { len += 1 }
            if len > bestLen { bestLen = len; bestEnd = p }
        }
        guard bestLen >= 3 else { return ([], 0) }
        return (Array(h[(bestEnd + 1)...min(cur, bestEnd + maxK)]), bestLen)
    }

    @Test func proposeMatchesNaive() {
        var g = SplitMix64(seed: 3)
        for trial in 0..<20 {
            let ci = CopyIndex(capacity: 64)  // small: forces rebuilding the index as it grows
            var h: [UInt32] = []
            let vocab = UInt64(trial % 2 == 0 ? 6 : 40)
            for _ in 0..<1500 {
                // Spans copied from earlier (like an agent repeating code) and random tokens.
                if h.count > 50 && g.next() % 4 == 0 {
                    let s = Int(g.next() % UInt64(h.count - 20)), l = Int(g.next() % 20) + 1
                    for t in h[s..<(s + l)] { h.append(t); ci.append(t) }
                } else {
                    let t = UInt32(g.next() % vocab)
                    h.append(t); ci.append(t)
                }
                let k = Int(g.next() % 16) + 1
                let (exp, m) = Self.naive(h, maxK: k)
                let got = ci.propose(maxK: k)
                #expect(got == exp && ci.lastMatch == m, "trial \(trial) n \(h.count)")
                if got != exp { return }
            }
            // sync: prefix → appends; another history → starts from scratch.
            let ext = h + [1, 2, 3, 1, 2, 3]
            ci.sync(to: ext)
            #expect(ci.hist == ext && ci.propose(maxK: 8) == Self.naive(ext, maxK: 8).0)
            let other = Array(h.prefix(700)) + [5, 5, 5, 5]
            ci.sync(to: other)
            #expect(ci.hist == other && ci.propose(maxK: 8) == Self.naive(other, maxK: 8).0)
        }
    }

    /// Decisions of a policy without a drafter over a fixed token stream (proposal length, match, accepted), fed the
    /// round times the way `decodeSpec` does: `ms: 0` when untimed (greedy or a seed), the wall otherwise.
    static func decisions(untimed: Bool, wallSeed: UInt64) -> [Int] {
        var pol = CopyPolicy(maxT: CopyPolicy.maxRows, plainBase: true)
        var tokens = SplitMix64(seed: 11), wall = SplitMix64(seed: wallSeed)
        var out: [Int] = []
        for _ in 0..<2000 {
            let k = Int(tokens.next() % 16), match = Int(tokens.next() % 40), acc = Int(tokens.next() % 17)
            // Machine noise: 0.8x to 1.25x around the real cost of the round.
            let noise = 0.8 + 0.45 * Double(wall.next() % 1000) / 1000
            if let t = pol.choose(k: k, match: match) {
                out.append(t)
                let ms = 18 * CopyPolicy.defaultShape[t] / CopyPolicy.defaultShape[1] * noise
                pol.observeCopy(rows: t, accepted: min(acc, t - 1), match: match, ms: untimed ? 0 : ms)
            } else {
                out.append(1)
                pol.observeBase(tokens: 1, ms: untimed ? 0 : 18 * noise)
            }
        }
        return out
    }

    /// Untimed, the round structure depends only on the tokens: different machine noise gives the same decisions.
    /// Control: with timings it does not.
    @Test func untimedDecisionsIgnoreTimings() {
        #expect(Self.decisions(untimed: true, wallSeed: 1) == Self.decisions(untimed: true, wallSeed: 2))
        #expect(Self.decisions(untimed: false, wallSeed: 1) != Self.decisions(untimed: false, wallSeed: 2))
    }
}
