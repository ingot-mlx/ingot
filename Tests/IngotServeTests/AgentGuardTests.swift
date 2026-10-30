import Foundation
import Testing

@testable import IngotServe

/// Agent guards: the token-loop detector on synthetic loops, data that repeats on purpose, the real loops and
/// near misses of Resources/guard/loops.json and its cost per round.
@Suite(.serialized) struct AgentGuardTests {
    /// Pushes `ids` in rounds of `round` tokens; the output length at the first trip, if any.
    static func firstTrip(_ ids: [UInt32], round: Int = 1) -> (at: Int, trip: AgentGuard.Trip)? {
        var g = AgentGuard()
        var i = 0
        while i < ids.count {
            let j = min(i + round, ids.count)
            if let t = g.push(Array(ids[i..<j])) { return (j, t) }
            i = j
        }
        return nil
    }

    /// 50 distinct tokens and then a block of `p` distinct tokens repeated: trips at the smallest period, exactly
    /// when the repeats cover the span the thresholds ask for (12 repeats, at least 96 tokens, 256 below period 8).
    @Test func syntheticLoops() {
        for p in 1...AgentGuard.maxPeriod {
            let prefix = (0..<50).map { UInt32(10_000 + $0) }
            let block = (0..<p).map { UInt32($0) }
            var ids = prefix
            while ids.count < 50 + 12 * p + 300 { ids += block }
            let span = max(AgentGuard.reps * p, AgentGuard.minSpan, p < 8 ? AgentGuard.shortSpan : 0)
            let hit = Self.firstTrip(ids)
            #expect(hit?.at == 50 + span, "period \(p)")
            #expect(hit?.trip == .tokenLoop(period: p, span: span), "period \(p)")
        }
    }

    /// With rounds of several tokens (speculative decoding) it trips at the end of the round that completes the span.
    @Test func roundsOfSeveralTokens() {
        let ids =
            (0..<20).map { UInt32(500 + $0) }
            + Array(repeating: [UInt32(1), 2, 3, 4, 5, 6, 7, 8, 9], count: 40).joined()
        let at = 20 + 12 * 9
        for round in [2, 5, 16] {
            let hit = Self.firstTrip(ids, round: round)
            #expect(hit?.at == (at + round - 1) / round * round, "round \(round)")
        }
    }

    /// Data repeats a few tokens on purpose: 200 zeros, a 40-row table whose rows differ in one token, and 11 repeats
    /// of a 20-token line do not trip.
    @Test func intendedRepetition() {
        #expect(Self.firstTrip(Array(repeating: 0, count: 200)) == nil)
        var table: [UInt32] = [1, 2, 3]
        for r in 0..<40 { table += [7, 8, UInt32(100 + r), 9, 10, 11] }
        #expect(Self.firstTrip(table) == nil)
        let line = (0..<20).map { UInt32(300 + $0) }
        #expect(Self.firstTrip(Array(Array(repeating: line, count: 11).joined())) == nil)
        #expect(Self.firstTrip(Array(Array(repeating: line, count: 12).joined()))?.at == 240)
    }

    static func fixtures() throws -> JSONValue {
        let url = Bundle.module.url(forResource: "Resources", withExtension: nil)!
            .appendingPathComponent("guard/loops.json")
        return try JSONValue.parse(Data(contentsOf: url))
    }

    static func ids(_ v: JSONValue) -> [UInt32] { (v["ids"]?.array ?? []).compactMap { $0.intValue.map(UInt32.init) } }

    /// The Muse loop (reasoning, period 130) and three GLM replays with the 4-bit head (periods 3 and 4) trip at the
    /// token the Python port found.
    @Test func realLoops() throws {
        let loops = try #require(try Self.fixtures()["loops"]?.array)
        #expect(loops.count == 4)
        for l in loops {
            let src = l["source"]?.string ?? "?"
            #expect(Self.firstTrip(Self.ids(l))?.at == l["trip_at"]?.intValue, "\(src)")
        }
    }

    /// The pi-accepted responses closest to the threshold (they trip with Locals' agent thresholds) never trip, in
    /// rounds of 1 to 16 tokens.
    @Test func normalResponses() throws {
        let normal = try #require(try Self.fixtures()["normal"]?.array)
        #expect(normal.count >= 20)
        for r in normal {
            for round in [1, 3, 16] {
                #expect(Self.firstTrip(Self.ids(r), round: round) == nil, "\(r["source"]?.string ?? "?")")
            }
        }
    }

    /// Cost per round with 16K tokens of output: under 20 µs (it runs on the engine queue after each round).
    @Test func costPerRound() {
        var rng = SystemRandomNumberGenerator()
        var g = AgentGuard()
        // A small alphabet: the last token comes back at many periods and each one must be compared.
        _ = g.push((0..<16_384).map { _ in UInt32.random(in: 0..<4, using: &rng) })
        // Best of 5 batches: the suite shares the machine with other tests.
        var best = Duration.seconds(1)
        var trips = 0
        for _ in 0..<5 {
            let toks = (0..<400).map { _ in UInt32.random(in: 0..<4, using: &rng) }
            let t0 = ContinuousClock.now
            for t in toks where g.push([t]) != nil { trips += 1 }
            best = min(best, (ContinuousClock.now - t0) / toks.count)
        }
        print("AgentGuard.push: \(best) per round with \(g.ids.count) tokens of output")
        #expect(trips == 0)
        #expect(best < .microseconds(20), "\(best) per push")
    }
}
