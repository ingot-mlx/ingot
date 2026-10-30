import Foundation
import LabCLI
import LabServe
import Metal
import MetalLab

/// lab-dev reuse [--n 45,46,...|40-72] [--cover N,N+1]: the state after a reply (chained round undone, tape folded) and the restore of
/// snapshots (a) and (b) must be bit-identical to the reference path (same prompt preparation and the reply step
/// by step with B = 1). Compares the logits of a chunk read afterwards and of 3 decode steps (a chunk read
/// after the reply plus three single steps).
@MainActor func cmdReuse() throws {
    let model = try args.required("model", modelHelp)
    // --n: greedy reply lengths, a list (45,46,47) or a range (40-72).
    var ns = try replyLengths(args.str("n", "45,46,47,48,49,50,51,52"))
    let (dir, cfg, ctx) = try openModel(model)
    // --prompt: another prompt (e.g. an agent one from prompts-es, with context-copy rounds).
    let A = try promptIds(args.str("prompt", "chat"))
    // S (the chunk read afterwards; the second turn uses up to 300 tokens): the 512 of the prompt set, or --prompt if
    // the set has none (prompts-es). Previously any --pdir picked --prompt: with --pdir prompts-lfm8b and the
    // 37-token chat, S[..<300] went out of bounds (LFM2.5-8B).
    let has512 = FileManager.default.fileExists(atPath: repoRoot.path + "/ref/out/\(promptSet)/512.json")
    let S = Array(try promptIds(has512 ? "512" : args.str("prompt", "chat")).prefix(300))
    guard S.count >= 300 else {
        throw UsageError("reuse: chunk S has \(S.count) tokens and the second turn needs 300 (use another --prompt)")
    }
    // Untimed rounds (like gen --golden-logits): each round's rows do not depend on machine load, and repeated
    // turns end with the same live state.
    var opts = try engineOptions(cfg)
    opts.bench.copyPolicyUntimed = true
    let m = try HybridModel(
        ctx: ctx, store: try SafetensorsStore(directory: dir), config: cfg, maxCtx: max(4096, A.count + 1024),
        options: opts)
    let dr = try openModelDrafter(ctx, dir, cfg, m)
    let eng = InferenceEngine(model: m, drafter: dr)
    eng.generationMarker = genMarker
    eng.log = { print("  " + $0) }
    // --short N: shortPassesMax (with short prompts, like the 38-token LFM2.5 chat, prompt preparation takes the
    // verification path and the snapshot reference, which does prefill, is no longer the same path).
    if let sp = args.opts["short"].flatMap({ Int($0) }) { m.shortPassesMax = sp }
    let V = m.vocab
    func logits(_ row: Int = 0) -> [Float] {
        Array(
            UnsafeBufferPointer(
                start: (m.logits.contents() + row * V * 4).bindMemory(to: Float.self, capacity: V), count: V))
    }
    let extra: [UInt32] = [198, 279, 3681]
    /// Reads S after the current state and takes 3 steps; returns the 4 logits vectors.
    func tail() throws -> [[Float]] {
        try m.foldPending(drafter: nil, context: nil)
        _ = try m.prefill(S, drafter: dr, continuing: true)
        var out = [logits()]
        for t in extra { _ = try m.stepLogits(t); out.append(logits()) }
        return out
    }
    /// Snapshot reference: the state after `ids[..<cut]` built without snapshots from the current state (P = c) by
    /// the same path as `preparePrompt(ids, from: c)`: one pass ending at the cut if capture happens inside the pass;
    /// otherwise batches of <= 16 folded rows or prefill up to s, and the rows of the final stretch up to the cut
    /// folded in place. (Before the dense fix, the second turn always assumed batches: with the 300-token turn without
    /// capture, on the 27B, the engine takes prefill and the reference did not follow its path.)
    func refState(_ ids: [UInt32], from c: Int, cut: Int, genStart g: Int?) throws {
        precondition(m.P == c && m.nacc == 0)
        if m.promptUsesCapture(count: ids.count, from: c, genStart: g) {
            _ = try m.prefill(Array(ids[c..<cut]), drafter: dr, continuing: c > 0, head: false)
            return
        }
        let plan = m.promptPlan(count: ids.count, from: c, genStart: g)
        if let b = plan.batches {
            for r in b { try m.stepRows(Array(ids[r]), drafter: dr); try m.foldPending(drafter: nil, context: nil) }
        } else if plan.s > c {
            _ = try m.prefill(Array(ids[c..<plan.s]), drafter: dr, continuing: c > 0, head: false)
        }
        if cut > plan.s - plan.head { try m.stepRows(Array(ids[(plan.s - plan.head)..<cut]), drafter: dr) }
    }
    // --cover N,N+1 (harness, for the CI): --n becomes the search domain. Greedy replies are generated (only the
    // reply, no reference) in --n order until every listed rewind case has come out, and only the first length of
    // each case is checked. A case that does not come out fails: a fixed --n list silently stopped covering the N
    // case when the toy numbers changed.
    let cover = try args.opts["cover"].map(rewindCases) ?? []
    if !cover.isEmpty { ns = try probeLengths(eng, A, ns, cover: cover) }
    var allOK = true
    var cases = ns.map { (Float(0), "greedy", $0, false) }
    cases += [(Float(0.7), "temperature 0.7", ns.last!, false)]
    // Penalties (repetition 1.3, presence 1.5, frequency 0.5): the state stays reusable; with greedy, the tokens
    // must also be the argmax of the step-by-step logits penalized on the CPU.
    cases += [(Float(0.7), "temperature 0.7 + penalties", ns.last!, true), (Float(0), "greedy + penalties", 64, true)]
    var branches = Set<String>()
    let gA = genStart(A)
    for (temp, label, n, pen) in cases {
        print("\(label): reply of \(n) tokens with the chained round")
        eng.clear()
        var sp = SamplingParams(temperature: temp, topP: 0.95, topK: 20, seed: 7)
        if pen { sp.repetitionPenalty = 1.3; sp.presencePenalty = 1.5; sp.frequencyPenalty = 0.5 }
        let r = try eng.generate(prompt: A, maxTokens: n, sampling: sp, stopTokens: m.eosSet)
        let gen = Array(eng.live[A.count...])
        print(
            "  \(r.stats.tokens.count) emitted, \(gen.count) in cache, \(r.stats.rounds) rounds, "
                + "finish \(r.stats.finish), block \(m.lastRewind); \(r.stats.copySummary)"
        )
        branches.insert(m.lastRewind)
        let live = try tail()
        // Reference: the same prompt preparation and the reply step by step (B = 1).
        eng.clear()
        let (_, row0, _) = try m.preparePrompt(A, from: 0, genStart: gA, drafter: nil)
        let expected = try stepAndPredict(m, gen, row0: row0, prompt: A, pen && temp == 0 ? sp : nil)
        if let expected { allOK = compareGreedyPenalties(r.stats.tokens, expected, inCache: gen.count) && allOK }
        let ref = try tail()
        allOK = compareLogitSteps(live, ref, "live state") && allOK
    }
    // Snapshots: after replying to A, a prompt A[..<k] + S must restore snapshot (a) or (b).
    // Point (a) only exists if the prompt continues after `<|im_start|>assistant\n` (Qwen3.6's `<think>\n`): with
    // LFM2.5-8B the chat ends there, the engine only takes (b) (Prompt.swift: genStart < n) and the (a) test asked for
    // a cut at A.count that no snapshot has. The second turn, which carries `<think>\n`, does cover (a).
    var snapCuts = [(A.count - 1, "(b) before the last token")]
    if let g = gA, g < A.count {
        snapCuts.insert((g, "(a) after <|im_start|>assistant\\n"), at: 0)
    } else {
        print("  no point (a) in the first turn (nothing after <|im_start|>assistant\\n, or absent): only (b)")
    }
    for (cut, name) in snapCuts {
        eng.clear()
        _ = try eng.generate(prompt: A, maxTokens: 48, sampling: .init(), stopTokens: m.eosSet)
        let P2 = Array(A[..<cut]) + S
        let r2 = try eng.generate(prompt: P2, maxTokens: 4, sampling: .init(), stopTokens: [])
        print("  snapshot \(name): source \(r2.cacheSource), \(r2.cachedTokens) tokens reused")
        allOK = allOK && r2.cacheSource == "snapshot" && r2.cachedTokens == cut
        // Bit-exact comparison: restore and read S, versus building the same state without snapshots.
        eng.clear()
        _ = try eng.generate(prompt: A, maxTokens: 48, sampling: .init(), stopTokens: m.eosSet)
        eng.restoreForTest(prefixOf: P2)
        precondition(m.P == cut)
        perturbForTest(m, "snapshot \(name) restored")
        let snapPath = try tail()
        // Reference: the same state built without snapshots and by the same path as preparePrompt (prefill up to
        // s, or batches of <= 16 folded rows if the chunk is short, like the 36-token chat) and the rows of the
        // final stretch up to the cut folded in place.
        eng.clear()
        m.reset()
        let plan = m.promptPlan(count: A.count, from: 0, genStart: gA)
        try refState(A, from: 0, cut: cut, genStart: gA)
        let ref = try tail()
        allOK = compareLogitSteps(snapPath, ref, "snapshot \(name)") && allOK
        if plan.batches != nil {
            // Informative: the prefix read by prefill (another path, need not be bit-identical).
            eng.clear()
            _ = try m.prefill(Array(A[..<plan.s]), drafter: dr, head: false)
            if cut > plan.s { try m.stepRows(Array(A[plan.s..<cut]), drafter: dr) }
            _ = compareLogitSteps(snapPath, try tail(), "  (informative) prefix by prefill, snapshot \(name)")
        }
    }
    // Short second turn: the new message takes the verification path and its last batch goes with the final
    // stretch, so snapshot (a) is also taken by folding tape rows. Reference: the same rows in batches of <= 16
    // folded one by one.
    // <think>\n (with the model's tokenizer: the toy ones have another vocabulary).
    let think = (try? BPETokenizer(directory: dir))?.encode("<think>\n").map { UInt32($0) } ?? [248068, 198]
    // k = 300: the turn takes prefill, with the snapshots captured inside the pass; reference: one pass from the
    // live state ending at the cut.
    for k in [5, 20, 300] {
        func turn2() throws -> [UInt32] {
            eng.clear()
            _ = try eng.generate(prompt: A, maxTokens: 48, sampling: .init(), stopTokens: m.eosSet)
            let P2 = eng.live + Array(S[..<k]) + genMarker + think
            let r = try eng.generate(prompt: P2, maxTokens: 4, sampling: .init(), stopTokens: [])
            allOK = allOK && r.cacheSource == "live"
            return P2
        }
        let P2 = try turn2()
        for (cut, name) in [(genStart(P2)!, "(a)"), (P2.count - 1, "(b)")] {
            let P3 = Array(P2[..<cut]) + S
            _ = try turn2()
            let r3 = try eng.generate(prompt: P3, maxTokens: 4, sampling: .init(), stopTokens: [])
            print(
                "  turn 2 (\(k) tokens), snapshot \(name): source \(r3.cacheSource), \(r3.cachedTokens) tokens reused"
            )
            allOK = allOK && r3.cacheSource == "snapshot" && r3.cachedTokens == cut
            _ = try turn2()
            eng.restoreForTest(prefixOf: P3)
            // If the repeated turn does not end with the same live state (P +- 1, e.g. with --perturb), there is no
            // snapshot at the cut: FAIL instead of crashing.
            guard m.P == cut else {
                print("  second turn of \(k) tokens, snapshot \(name): no snapshot at \(cut) (P \(m.P))")
                allOK = false
                continue
            }
            perturbForTest(m, "second turn of \(k) tokens, snapshot \(name) restored")
            let snapPath = try tail()
            eng.clear()
            _ = try eng.generate(prompt: A, maxTokens: 48, sampling: .init(), stopTokens: m.eosSet)
            try refState(P2, from: m.P, cut: cut, genStart: genStart(P2))
            let ref = try tail()
            allOK = compareLogitSteps(snapPath, ref, "second turn of \(k) tokens, snapshot \(name)") && allOK
        }
    }
    allOK = try checkRollback(eng, A, S, tail: tail) && allOK
    print("rewind cases covered: \(branches.sorted())")
    if !Set(cover).isSubset(of: branches) {
        print("  --cover \(cover): the checked replies did not repeat the probed rewind cases (non-deterministic?)")
        allOK = false
    }
    print(allOK ? "reuse: OK" : "reuse: FAILED")
    if !allOK { exit(1) }
}

/// A second turn that diverges inside the reply (the template wrote it back its own way). With a KV-only
/// state the live state goes back to the common prefix c and must give the same bits as a live state that ended
/// at c: a shorter greedy reply (24 tokens) leaves the reference state at c (copy rounds make c land where they
/// land), the 48-token reply goes on past it, and the second turn diverges right at c. With any other state the
/// turn takes a snapshot as before.
@MainActor private func checkRollback(
    _ eng: InferenceEngine, _ A: [UInt32], _ S: [UInt32], tail: () throws -> [[Float]]
) throws -> Bool {
    let kvOnly = eng.snapshotBytes == 0
    var allOK = true
    for k in [20, 300] {
        eng.clear()
        _ = try eng.generate(prompt: A, maxTokens: 24, sampling: .init(), stopTokens: [])
        let short = eng.live, c = short.count
        eng.clear()
        _ = try eng.generate(prompt: A, maxTokens: 48, sampling: .init(), stopTokens: [])
        let full = eng.live
        // The new part must differ from the reply at c.
        let o = full.count > c && S[0] == full[c] ? 1 : 0
        let P2 = short + Array(S[o..<(o + k - 1)])
        guard full.count > c, Array(full[..<c]) == short, full[c] != P2[c] else {
            print("  rollback: the 48-token reply (\(full.count)) does not go on from the 24-token one (\(c))")
            allOK = false
            continue
        }
        let r = try eng.generate(prompt: P2, maxTokens: 4, sampling: .init(), stopTokens: [])
        print(
            "  turn 2 diverging inside the reply (\(P2.count - c) new tokens, common prefix \(c) of \(full.count)): "
                + "source \(r.cacheSource), \(r.cachedTokens) tokens reused")
        guard kvOnly else {
            allOK = allOK && r.cacheSource != "live" && r.cachedTokens < c
            continue
        }
        allOK = allOK && r.cacheSource == "live" && r.cachedTokens == c
        let rolledBack = try tail()
        // Reference: the shorter reply, then the same second turn from it.
        eng.clear()
        _ = try eng.generate(prompt: A, maxTokens: 24, sampling: .init(), stopTokens: [])
        let r2 = try eng.generate(prompt: P2, maxTokens: 4, sampling: .init(), stopTokens: [])
        allOK = allOK && r2.cacheSource == "live" && r2.cachedTokens == c && r2.stats.tokens == r.stats.tokens
        allOK = compareLogitSteps(rolledBack, try tail(), "rollback to \(c), turn of \(P2.count - c) tokens") && allOK
    }
    return allOK
}

/// The block names `HybridModel.lastRewind` gives after a reply (Generate.swift, rewindExtraRound).
private let rewindNames: Set<String> = ["N", "N without extra round", "N+1"]

/// --cover: greedy replies of each length of `ns` in order (only the reply, no reference) until every case of
/// `cover` has come out; returns the first length of each case, or prints FAILED and exits if one does not come out.
@MainActor private func probeLengths(_ eng: InferenceEngine, _ prompt: [UInt32], _ ns: [Int], cover: [String]) throws
    -> [Int]
{
    let m = eng.model
    var found: [String: Int] = [:]
    for n in ns where !cover.allSatisfy({ found[$0] != nil }) {
        eng.clear()
        let r = try eng.generate(prompt: prompt, maxTokens: n, sampling: .init(), stopTokens: m.eosSet)
        print(
            "  probe: greedy reply of \(n) tokens (\(r.stats.tokens.count) emitted, \(r.stats.rounds) rounds) ends in "
                + "block \(m.lastRewind)")
        if found[m.lastRewind] == nil { found[m.lastRewind] = n }
    }
    let missing = cover.filter { found[$0] == nil }
    guard missing.isEmpty else {
        print("rewind cases \(missing) not reached with --n \(ns.first!)-\(ns.last!) (\(ns.count) lengths)")
        print("reuse: FAILED")
        exit(1)
    }
    let chosen = Set(cover.compactMap { found[$0] }).sorted()
    print("  --cover \(cover): checking lengths \(chosen)")
    return chosen
}

/// --cover: comma-separated rewind cases, each one of `rewindNames`.
private func rewindCases(_ s: String) throws -> [String] {
    let cs = s.split(separator: ",").map { String($0) }
    if let bad = cs.first(where: { !rewindNames.contains($0) }) {
        throw UsageError(
            "reuse --cover: unknown rewind case \"\(bad)\" (\(rewindNames.sorted().joined(separator: ", ")))")
    }
    return cs
}

/// --n: a comma-separated list of lengths or a range LO-HI.
private func replyLengths(_ s: String) throws -> [Int] {
    let range = s.split(separator: "-").map { Int($0) }
    var ns: [Int]
    if range.count == 2, let lo = range[0], let hi = range[1], lo <= hi {
        ns = Array(lo...hi)
    } else {
        ns = s.split(separator: ",").compactMap { Int($0) }
    }
    guard !ns.isEmpty, ns.allSatisfy({ $0 > 0 }) else {
        throw UsageError("reuse --n: a list of lengths (45,46) or a range (40-72), got \(s)")
    }
    return ns
}

/// Steps the reply `gen` with B = 1 from the prepared prompt (last row `row0`); with `penalized`, also returns the
/// penalized argmax before each step and after the last one (greedy + penalties reference).
private func stepAndPredict(
    _ m: HybridModel, _ gen: [UInt32], row0: Int, prompt: [UInt32], _ penalized: SamplingParams?
) throws -> [UInt32]? {
    func logits(_ row: Int) -> [Float] {
        Array(
            UnsafeBufferPointer(
                start: (m.logits.contents() + row * m.vocab * 4).bindMemory(to: Float.self, capacity: m.vocab),
                count: m.vocab))
    }
    var expected = penalized.map { [penalizedArgmax(logits(row0), prompt: prompt, generated: [], $0)] }
    for (i, t) in gen.enumerated() {
        _ = try m.stepLogits(t)
        if let p = penalized { expected?.append(penalizedArgmax(logits(0), prompt: prompt, generated: gen[...i], p)) }
    }
    return expected
}

private func compareGreedyPenalties(_ emitted: [UInt32], _ expected: [UInt32], inCache: Int) -> Bool {
    let k = min(emitted.count, expected.count)
    let same = Array(emitted.prefix(k)) == Array(expected.prefix(k)) && k >= inCache
    let firstDiff = (0..<k).first { emitted[$0] != expected[$0] } ?? -1
    print(
        "  greedy + penalties against the CPU step by step: "
            + (same ? "same \(k) tokens" : "DIFFERENT at \(firstDiff)"))
    return same
}

/// Argmax (ties by lower id, like tk_better) of `x` penalized on the CPU with vLLM semantics: repetition over prompt +
/// generated, then frequency and presence over the generated tokens (the same arithmetic as penalize_rows).
private func penalizedArgmax(_ x: [Float], prompt: [UInt32], generated: ArraySlice<UInt32>, _ p: SamplingParams)
    -> UInt32
{
    var y = x
    var c: [UInt32: Int] = [:]
    for t in generated { c[t, default: 0] += 1 }
    let inPrompt = p.repetitionPenalty != 1 ? Set(prompt) : []
    for t in Set(c.keys).union(inPrompt) where Int(t) < y.count {
        let n = c[t] ?? 0
        var v = y[Int(t)]
        if n > 0 || inPrompt.contains(t) { v = v > 0 ? v / p.repetitionPenalty : v * p.repetitionPenalty }
        if n > 0 {
            v = v.addingProduct(-p.frequencyPenalty, Float(n))
            v -= p.presencePenalty
        }
        y[Int(t)] = v
    }
    var best = 0
    for i in 1..<y.count where y[i] > y[best] { best = i }
    return UInt32(best)
}
