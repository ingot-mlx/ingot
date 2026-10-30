import Foundation
import Ingot
import IngotCLI
import Metal

/// ingot-dev batchcheck --serve [--prompts 512,4K,chat] [--rows 2,4] [--n 32] [--seed S]: ingot serve's engine loop
/// (`InferenceEngine.serve`, BatchServe.swift) with conversation slots, in this process and without HTTP. For each N of
/// --rows, N conversations take two turns each and all the requests are queued at once, alternating (A1 B1 … A2 B2 …;
/// turn 2 is turn 1's prompt, its answer and a new user turn), first with N slots (up to 4) and then, for N > 2, with
/// half as many. Each request must give the same ids as the same conversation alone in a fresh engine (turn 1, then
/// turn 2 from its live state), and, with N slots, every turn 2 must resume from its slot's live state (no answer read
/// again). Models with a drafter or MoE layers serve their slots one at a time: there each request also
/// takes the same number of rounds as alone (a wrong drafter context ring keeps the greedy ids, since the verification
/// is exact, but not the acceptance) and, with `--seed S`, the requests are then sampled too (conversation j with the
/// parameters and seed of `batchSampling(j, S)`) with the same check. Models that join rounds only compare
/// greedy ids: a conversation that joins takes new uniforms. With fewer slots, a turn that resumes from another point
/// than alone (a snapshot or a prefix cache entry instead of its live state) is only listed. Run it with --copy off: with the copy, the round
/// structure follows the policy's timings.
@MainActor func cmdBatchServeCheck() throws {
    let model = try args.required("model", modelHelp)
    let names = args.str("prompts", "512,4K,chat").split(separator: ",").map(String.init)
    let sizes = args.str("rows", "2,4").split(separator: ",").compactMap { Int($0) }.filter { $0 >= 2 }
    let n = args.int("n", 32)
    let seed = args.opts["seed"].flatMap(UInt64.init)
    let (dir, cfg, ctx) = try openModel(model)
    let base = try names.map { try promptIds($0) }
    guard let maxN = sizes.max(), !base.isEmpty, n >= 2 else {
        throw UsageError("batchcheck --serve: --rows (2 or more), --prompts, --n ≥ 2")
    }
    func prompt(_ j: Int) -> [UInt32] {
        let p = base[j % base.count]
        return Array(p.prefix(max(16, p.count - 37 * (j / base.count))))
    }
    // The new user turn of conversation j: 40 tokens of another prompt (only the ids matter).
    func userTurn(_ j: Int) -> [UInt32] { Array(base[(j + 1) % base.count].prefix(40)) }
    let firsts = (0..<maxN).map(prompt)
    let maxCtx = (firsts.map(\.count).max() ?? 0) + 2 * n + 40 + 64
    let m = try HybridModel(
        ctx: ctx, store: try SafetensorsStore(directory: dir), config: cfg, maxCtx: maxCtx,
        options: try engineOptions(cfg))
    let dr = try openModelDrafter(ctx, dir, cfg, m)
    func engine() -> InferenceEngine {
        let e = InferenceEngine(model: m, drafter: dr)
        e.generationMarker = genMarker
        return e
    }
    if let why = engine().batchUnsupported { throw UsageError("batchcheck --serve: this model has \(why)") }
    let serial = dr != nil || m.batchUnsupported != nil
    print(
        "model: \(dr == nil ? "no drafter" : "drafter"), "
            + (serial
                ? "slots served one at a time (\(dr != nil ? "drafter" : m.batchUnsupported!))"
                : "slots in joint rounds")
            + "; copy \(m.options.copyEnabled(moe: m.isMoE, drafter: dr != nil) ? "ON (use --copy off)" : "off")")
    struct Turn {
        var ids: [UInt32]
        var rounds: Int
        var source: String
    }
    // Alone: each conversation in a fresh engine, turn 1 then turn 2.
    func alone(_ j: Int, _ sp: SamplingParams) throws -> [Turn] {
        let e = engine()
        var turns: [Turn] = []
        var p = firsts[j]
        for t in 0..<2 {
            let r = try e.generate(prompt: p, maxTokens: n, sampling: sp, stopTokens: [])
            turns.append(Turn(ids: r.stats.tokens, rounds: r.stats.rounds, source: r.cacheSource))
            if t == 0 { p += r.stats.tokens + userTurn(j) }
        }
        return turns
    }
    // Served: all the requests of N conversations queued at once with `slots` slots; turn 2 of a conversation is
    // pushed when its turn 1 ends (its prompt needs the answer), behind whatever is queued then.
    func served(_ N: Int, slots: Int, _ sp: @escaping (Int) -> SamplingParams) throws -> [[Turn]] {
        let e = engine()
        e.enableBatching(slots: slots, positions: maxCtx)
        let q = EngineQueue()
        var out = [[Turn]](repeating: [], count: N)
        var failure: Error?
        func push(_ j: Int, _ p: [UInt32]) {
            q.push(
                EngineRequest(
                    prompt: p, maxTokens: n, sampling: sp(j), stopTokens: [], onStart: { true },
                    onTokens: { _ in true },
                    onDone: { res in
                        switch res {
                        case .success(let r):
                            out[j].append(Turn(ids: r.stats.tokens, rounds: r.stats.rounds, source: r.cacheSource))
                            if out[j].count == 1 { push(j, p + r.stats.tokens + userTurn(j)) }
                        case .failure(let err): failure = err
                        }
                    }))
        }
        for j in 0..<N { push(j, firsts[j]) }
        e.serve(q)
        if let failure { throw failure }
        return out
    }
    var bad = 0, total = 0
    func compare(_ tag: String, _ N: Int, slots: Int, ref: [[Turn]], got: [[Turn]]) {
        for j in 0..<N {
            for t in 0..<2 {
                total += 1
                guard t < got[j].count else {
                    bad += 1
                    print("\(tag) conversation \(j) turn \(t + 1): MISSING")
                    continue
                }
                let a = ref[j][t], b = got[j][t]
                // With fewer slots than conversations a turn 2 may find its slot taken and resume from a snapshot or a
                // prefix cache entry (the prompt read from another point): that path is not the one alone took (from
                // its live state) and may differ. Not compared.
                if slots < N && b.source != a.source {
                    total -= 1
                    print(
                        "\(tag) conversation \(j) turn \(t + 1): from \(b.source), alone from \(a.source): not compared"
                            + (a.ids == b.ids ? " (same ids)" : ""))
                    continue
                }
                var why: [String] = []
                if a.ids != b.ids {
                    why.append(
                        "DIFFERENT ids from token \(firstDifference(a.ids, b.ids) ?? min(a.ids.count, b.ids.count))")
                }
                if serial && a.rounds != b.rounds { why.append("rounds \(b.rounds), alone \(a.rounds)") }
                if t == 1 && slots >= N && b.source != "live" { why.append("turn 2 from \(b.source), not live") }
                if !why.isEmpty { bad += 1 }
                print(
                    "\(tag) conversation \(j) turn \(t + 1): "
                        + (why.isEmpty
                            ? "identical: same \(b.ids.count) ids, \(b.rounds) rounds, cache \(b.source)"
                            : why.joined(separator: "; ") + " (cache \(b.source))"))
            }
        }
    }
    let greedy = SamplingParams()
    let soloGreedy = try (0..<maxN).map { try alone($0, greedy) }
    for (j, s) in soloGreedy.enumerated() {
        print(
            "conversation \(j): \(firsts[j].count) prompt tokens; alone: \(s[0].rounds) and \(s[1].rounds) rounds, "
                + "turn 2 from \(s[1].source)")
    }
    let soloSampled = try seed.map { sd in try (0..<maxN).map { try alone($0, batchSampling($0, seed: sd)) } }
    for N in sizes {
        for slots in Set([min(N, 4), max(2, min(N, 4) / 2)]).sorted(by: >) {
            let tag = "N=\(N) slots=\(slots)"
            compare(tag, N, slots: slots, ref: soloGreedy, got: try served(N, slots: slots) { _ in greedy })
            if let sd = seed, let ref = soloSampled {
                if serial {
                    compare(
                        tag + " sampled", N, slots: slots, ref: ref,
                        got: try served(N, slots: slots) { batchSampling($0, seed: sd) })
                } else {
                    print("\(tag) sampled: skipped (joint rounds take new uniforms)")
                }
            }
        }
    }
    print("batchcheck --serve: \(total - bad) of \(total) turns identical")
    if bad > 0 {
        print("batchcheck --serve: FAILED")
        exit(1)
    }
}
