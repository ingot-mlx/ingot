import Foundation
import LabCLI
import LabServe
import Metal
import MetalLab

/// lab-dev prefixcache --body ref/agent_dump/0001-body.json [--n 128] [--no-cold]: prefix cache across sessions, with
/// lab serve's engine defaults (on MoE models the lossy prefill; --int8 0 --qk8 0 for the exact one).
/// --no-cold (ci.sh --toy and --real --validate) skips what only gives timings or informative comparisons and decides
/// nothing: the no-cache turns of step 1, step 3 and the comparison with the whole prompt in step 5.
/// First turns of three pi sessions (the same body with three different `<cwd>`, A, B and C):
/// 1. times of the first turn with and without cache (A cold; B resumes from the system point lowered to 1024; C from
///    the boundary of the common prefix that B left);
/// 2. exactness of the mechanism: restoring the entry and reading the rest has to give bit-identical logits to
///    building the same state without cache (prefill up to the cut and prefill of the rest);
/// 3. whether splitting the prefill at the cut changes anything against reading the prompt in one go (informative);
/// 4. logits of the last position of C (bit by bit) and greedy output of --n tokens with cache against without
///    (freshly started server); 4b. a turn ending a few tokens past C's entry, its last logits against without cache;
/// 5. returning to an earlier conversation: greedy with cache against the same uncached path, split at the same point;
/// 6. and 7. cuts across thresholds and at chosen points (below); 8. a sampled request repeated gives the same text.
@MainActor func cmdPrefixCache() throws {
    let model = try args.required("model", modelHelp)
    let bodyPath = args.str("body", repoRoot.path + "/ref/agent_dump/0001-body.json")
    let nGen = args.int("n", 128)
    let (dir, cfg, ctx) = try openModel(model)
    let info = try ChatModelInfo(directory: dir)
    let m = try HybridModel(
        ctx: ctx, store: try SafetensorsStore(directory: dir), config: cfg, maxCtx: 16384,
        options: try prefixCheckOptions(cfg))
    describeLossy(m)  // which lossy paths run (lab serve prints the same)
    let dr = try openModelDrafter(ctx, dir, cfg, m)
    let eng = InferenceEngine(model: m, drafter: dr)
    // The prefix cache points from the template's turns, as lab serve (ChatML: the markers it always used;
    // Gemma 3 and Granite have their own).
    eng.generationMarker = info.dialect.generationMarker.map { info.tokenizer.encode($0).map { UInt32($0) } } ?? []
    eng.userMarker = info.dialect.userMarker.map { info.tokenizer.encode($0).map { UInt32($0) } } ?? []
    eng.log = { print("  " + $0) }
    let server = ChatServer(engine: eng, info: info, modelId: "x")
    let raw = try String(contentsOfFile: bodyPath, encoding: .utf8)
    func session(_ tag: String, _ change: SessionChange = .cwd) throws -> [UInt32] {
        try server.promptIds(body: Data(sessionBody(raw, tag: tag, change: change).utf8))
    }
    let tags = ["Hk4tYw9aPq", "Qm81pZkLe0", "v7RnB2cYsd"]
    var (A, B, C) = (try session(tags[0]), try session(tags[1]), try session(tags[2]))
    var inUser = false
    // LFM2.5 puts the tools after the system message: the `<cwd>` lands around token 900 and the sessions share no
    // prefix of minEntryTokens or more (in lab serve the cache does not hit between pi sessions with that template).
    // To test the mechanism anyway, what changes moves to the end of the system block, as in Qwen. Gemma 3's template
    // drops the tools (the three prompts would be the same): there it moves to the end of the user message. Gemma 4's
    // template puts the tools at the end of its system turn, past the system message, so a changed tool lands inside
    // the turn whose end is the entry B would resume from (none is shared): the user message too (by the
    // template, so toy-gemma4, with the toys' ChatML one, keeps the tools change).
    let userChange = info.dialect.toolFormat == .gemma4
    // When the changed tool lands before A's system entry, B cannot resume from A (systemEntryCut): the user message.
    let toUser = { userChange || lcp(A, B) < systemEntryCut(eng, A) }
    for change in [SessionChange.tools, .user]
    where lcp(A, B) < eng.minEntryTokens + eng.cutAlign || A == B || (change == .user && !inUser && toUser()) {
        print(
            "the <cwd> falls at token \(lcp(A, B)) of \(A.count) (tools after the system message in the template, "
                + "or no tools): changing \(change == .tools ? "the last tool description" : "the user message")")
        (A, B, C) = (try session(tags[0], change), try session(tags[1], change), try session(tags[2], change))
        inUser = change == .user
    }
    print("prompts: A \(A.count), B \(B.count), C \(C.count) tokens; common prefix A–B \(lcp(A, B)), B–C \(lcp(B, C))")
    let V = m.vocab
    func logits() -> [Float] {
        Array(UnsafeBufferPointer(start: m.logits.contents().bindMemory(to: Float.self, capacity: V), count: V))
    }
    let greedy = SamplingParams()
    var allOK = true
    // Cache limit: 1 GB, or whatever fits 6 entries of the prompts' size if that is more. On the 27B (48 GDN layers
    // of 48 heads, 16 attention layers with 4 KV heads) an entry of ~4.6K tokens takes far more than on the MoE, and
    // with 1 GB the entries of 4288 and 4352 were evicted before step 2 ("entry missing").
    let cacheBytes = max(1_000_000_000, 6 * eng.prefixEntryBytes(tokens: max(A.count, B.count, C.count)))
    print(
        String(
            format: "prefix cache: limit %.2f GB (entry of %d tokens: %.0f MB)", Double(cacheBytes) / 1e9,
            C.count,
            Double(eng.prefixEntryBytes(tokens: C.count)) / 1e6))

    // Warm-up (pipelines) and timings.
    _ = try eng.generate(prompt: Array(A.prefix(300)), maxTokens: 4, sampling: greedy, stopTokens: [])
    eng.clear()
    eng.prefixCacheBytes = 0
    // --no-cold (ci.sh --toy and --real --validate): without the first three turns without cache, which only give
    // timings.
    let noCold = args.flags.contains("no-cold")
    var cold: [Double] = [.nan, .nan, .nan]
    if !noCold {
        cold = []
        for p in [A, B, C] {
            eng.clear()
            cold.append(try eng.generate(prompt: p, maxTokens: 1, sampling: greedy, stopTokens: []).prefillSeconds)
        }
    }
    eng.prefixCacheBytes = cacheBytes
    eng.clear()
    var warm: [(Double, Int, String)] = []
    for p in [A, B, C] {
        let r = try eng.generate(prompt: p, maxTokens: 1, sampling: greedy, stopTokens: [])
        warm.append((r.prefillSeconds, r.cachedTokens, r.cacheSource))
    }
    for (i, name) in ["A", "B", "C"].enumerated() {
        print(
            String(
                format: "first turn %@: without cache %.3f s; with cache %.3f s (%d tokens from %@)", name, cold[i],
                warm[i].0, warm[i].1, warm[i].2))
    }
    // Changing the user message, A leaves only its conversation entry (past the common prefix): B starts from scratch
    // and leaves the common entry that C resumes from (as in lab serve with that template).
    let firstFromEntry = warm[1].2 == "entry" || (inUser && warm[1].2 == "none")
    allOK = allOK && firstFromEntry && warm[2].2 == "entry"
    let cut = warm[2].1, cutB = warm[1].1

    // 2. Mechanism: restored entry against the same state built without cache.
    func tail(_ p: [UInt32], from c: Int) throws -> [[Float]] {
        _ = try m.prefill(Array(p[c...]), drafter: dr, continuing: true)
        var out = [logits()]
        for t: UInt32 in [198, 279, 3681] { _ = try m.stepLogits(t); out.append(logits()) }
        return out
    }
    for (c, p, name) in [(cut, C, "common boundary (C)"), (cutB, B, "first resumption (B)")] where c > 0 || !inUser {
        // Dirties the KV with another prompt, then restores.
        _ = try eng.generate(
            prompt: Array(A.prefix(2000)) + Array(C.prefix(2000)), maxTokens: 4, sampling: greedy, stopTokens: [])
        // And the KV and the token history past the cut: the generation above only reaches ~4000 positions
        // and the last ones of the entry kept their good values (a failure to copy the last positions went
        // unseen). It also dirties the token history of Flash-Next (ptok: the two tokens before the cut
        // that the PLE reads), which restoreState
        // restores.
        _ = try m.prefill(Array(p.reversed()), drafter: dr)
        guard eng.restoreEntryForTest(tokens: Array(p[..<c])) else {
            print("  entry of \(c) tokens missing"); allOK = false; continue
        }
        perturbForTest(m, "entry \(name) restored")
        let got = try tail(p, from: c)
        m.reset()
        _ = try m.prefill(Array(p[..<c]), drafter: dr, head: false)
        let ref = try tail(p, from: c)
        allOK = compareLogitSteps(got, ref, "entry \(name), \(c) tokens") && allOK
        eng.forgetLive()
    }

    // 3. Split the prefill at the cut against reading the prompt in one go (informative; not with --no-cold).
    if !noCold {
        m.reset(); _ = try m.prefill(C, drafter: dr); let whole = logits()
        m.reset(); _ = try m.prefill(Array(C[..<cut]), drafter: dr, head: false)
        _ = try m.prefill(Array(C[cut...]), drafter: dr, continuing: true)
        let split = logits()
        print(
            String(
                format: "prefill split at %d against whole: %@ (%@)", cut,
                whole == split ? "bit-identical" : "different", logitDistanceNote(split, whole)))
    }

    // 4. Greedy of C with cache (boundary entry) against without cache.
    eng.clear()
    eng.prefixCacheBytes = 0
    // And the logits of the last prompt position, bit by bit: a path difference in the preparation (prefill against
    // verification path in the last rows) shows up there with any --n; in the greedy only when it reaches a
    // near-tie (the toy dense model, at token 177).
    var promptLogits: [[Float]] = []
    eng.onPromptLogits = { row in
        let p = m.logits.contents().bindMemory(to: Float.self, capacity: (row + 1) * V)
        promptLogits.append(Array(UnsafeBufferPointer(start: p + row * V, count: V)))
    }
    defer { eng.onPromptLogits = nil }
    let ref = try eng.generate(prompt: C, maxTokens: nGen, sampling: greedy, stopTokens: m.eosSet).stats.tokens
    let refLogits = promptLogits.last!
    eng.prefixCacheBytes = cacheBytes
    eng.clear()
    for p in [A, B] { _ = try eng.generate(prompt: p, maxTokens: 8, sampling: greedy, stopTokens: m.eosSet) }
    let r = try eng.generate(prompt: C, maxTokens: nGen, sampling: greedy, stopTokens: m.eosSet)
    eng.onPromptLogits = nil
    let gotLogits = promptLogits.last!
    let sameLogits = gotLogits == refLogits
    print(
        "logits of the last position of C: with cache \(sameLogits ? "bit-identical" : "DIFFERENT") to without cache"
            + (sameLogits ? "" : " (" + logitDistanceNote(gotLogits, refLogits) + ")"))
    allOK = allOK && sameLogits
    let same = r.stats.tokens == ref
    let firstDiff = firstDifference(r.stats.tokens, ref)
    print(
        "greedy of C, \(ref.count) tokens: with cache (\(r.cacheSource), \(r.cachedTokens)) "
            + "\(same ? "identical" : "DIFFERENT from token \(firstDiff ?? min(ref.count, r.stats.tokens.count))") to without cache"
    )
    allOK = try shortTurnFromEntry(eng, m, C, entry: r.cachedTokens, cacheBytes: cacheBytes) && same && allOK
    allOK = try returnToConversation(eng, m, A, B, cacheBytes: cacheBytes, whole: !noCold) && allOK
    allOK = try convCutAcrossBound(eng, m, A, cacheBytes: cacheBytes) && allOK
    allOK = try entriesAtCuts(eng, m, cacheBytes: cacheBytes) && allOK
    allOK = try sampledRepeat(eng, m, C, B, n: nGen, cacheBytes: cacheBytes) && allOK
    print(allOK ? "prefixcache: OK" : "prefixcache: FAILED")
    if !allOK { exit(1) }
}

/// 4b. A short turn from an entry (a prompt that ends 12 tokens past C's entry, all of them before point (a)): the
/// logits of its last position bit by bit against reading it without cache. The few rows after the entry are few
/// enough for the verification path in every family (≤ 16 rows, `shortPassesMax`), which sums in another order than
/// the prefill that reads them without cache; they have to go through the prefill too (the real Muse, with 32
/// rows after its entry, gave max |Δlogit| 0.162 in step 4 while the toys, with longer tails, came out bit-identical).
@MainActor private func shortTurnFromEntry(
    _ eng: InferenceEngine, _ m: HybridModel, _ C: [UInt32], entry c: Int, cacheBytes: Int
) throws -> Bool {
    let k = min(12, C.count - 1 - c)
    guard c > 0, k >= 2 else {
        print("short turn from the entry: skipped (C resumed from \(c) of \(C.count) tokens)"); return true
    }
    let Q = Array(C[..<(c + k)]), V = m.vocab
    var pl: [[Float]] = []
    eng.onPromptLogits = { row in
        let p = m.logits.contents().bindMemory(to: Float.self, capacity: (row + 1) * V)
        pl.append(Array(UnsafeBufferPointer(start: p + row * V, count: V)))
    }
    defer { eng.onPromptLogits = nil }
    eng.forgetLive()
    let r = try eng.generate(prompt: Q, maxTokens: 1, sampling: SamplingParams(), stopTokens: [])
    eng.clear()
    eng.prefixCacheBytes = 0
    _ = try eng.generate(prompt: Q, maxTokens: 1, sampling: SamplingParams(), stopTokens: [])
    eng.prefixCacheBytes = cacheBytes
    eng.clear()
    let same = pl.count == 2 && pl[0] == pl[1]
    print(
        "short turn from the entry (\(Q.count) tokens, \(r.cacheSource) \(r.cachedTokens)): last-position logits "
            + (same ? "bit-identical" : "DIFFERENT (" + logitDistanceNote(pl[0], pl[pl.count - 1]) + ")")
            + " to without cache")
    return same && r.cacheSource == "entry" && r.cachedTokens == c
}

/// 5. Returning to an earlier conversation: A with its reply, then B (A becomes a "conversation" entry), then A
/// followed by a new message. Strict: greedy with cache against a reference without cache that splits the prefill
/// at the same point k (generate(A2[..<k + 1]) and then generate(A2), which resumes from snapshot (b) at k). With
/// cuts not aligned to 64 the chunked GDN rounds differently from the whole prompt: that comparison with the whole
/// prompt stays informative and is skipped when `whole` is false (--no-cold).
@MainActor private func returnToConversation(
    _ eng: InferenceEngine, _ m: HybridModel, _ A: [UInt32], _ B: [UInt32], cacheBytes: Int, whole: Bool
) throws -> Bool {
    let greedy = SamplingParams()
    eng.clear()
    let rA = try eng.generate(prompt: A, maxTokens: 48, sampling: greedy, stopTokens: m.eosSet)
    let A2 = A + rA.stats.tokens + Array(B.suffix(60))
    _ = try eng.generate(prompt: B, maxTokens: 8, sampling: greedy, stopTokens: m.eosSet)
    let r5 = try eng.generate(prompt: A2, maxTokens: 64, sampling: greedy, stopTokens: m.eosSet)
    let k5 = r5.cachedTokens
    eng.clear()
    eng.prefixCacheBytes = 0
    let whole5 = whole ? try eng.generate(prompt: A2, maxTokens: 64, sampling: greedy, stopTokens: m.eosSet) : nil
    eng.clear()
    var split5: [UInt32] = []
    var prefillSplit5 = 0.0
    if k5 > 0 && k5 < A2.count {
        _ = try eng.generate(prompt: Array(A2[..<(k5 + 1)]), maxTokens: 1, sampling: greedy, stopTokens: m.eosSet)
        eng.forgetLiveOnly()  // otherwise it would continue from the live state (k + 1), not from snapshot (b) at k
        let rs = try eng.generate(prompt: A2, maxTokens: 64, sampling: greedy, stopTokens: m.eosSet)
        split5 = rs.stats.tokens; prefillSplit5 = rs.prefillSeconds
    }
    eng.prefixCacheBytes = cacheBytes
    let same5 = !split5.isEmpty && r5.stats.tokens == split5
    var wholeTime = "", wholeNote = ""
    if let whole5 {
        wholeTime = String(format: "%.3f s whole, ", whole5.prefillSeconds)
        let sameWhole5 = r5.stats.tokens == whole5.stats.tokens
        let at = firstDifference(r5.stats.tokens, whole5.stats.tokens) ?? -1
        wholeNote = "; against whole (informative): " + (sameWhole5 ? "identical" : "different from token \(at)")
    }
    print(
        String(
            format:
                "return to conversation A (%d tokens): %@ %d tokens (%.3f s; without cache %@%.3f s split); "
                + "greedy against split at %d: %@%@",
            A2.count, r5.cacheSource, k5, r5.prefillSeconds, wholeTime, prefillSplit5, k5,
            same5 ? "identical" : "DIFFERENT from token \(firstDifference(r5.stats.tokens, split5) ?? -1)", wholeNote))
    return same5 && r5.cacheSource == "entry"
}

/// 8. Sampling with a seed: the same request repeated must give the same text as the first time. Two prompts: C read
/// from scratch, and a continuation of it (C, its reply and a new message, read from the live state through the
/// verification path; with presence penalty). Each one three times: the repetitions resume from snapshot (c), the whole
/// prompt (before, from (b) with the last token alone through the verification path: different logits, and on the MoE
/// toy a different text from token 23). Logits of the last prompt position bit by bit and the sampled tokens. Each run
/// starts from a different learned copy policy, which must be ignored.
@MainActor private func sampledRepeat(
    _ eng: InferenceEngine, _ m: HybridModel, _ C: [UInt32], _ B: [UInt32], n: Int, cacheBytes: Int
) throws -> Bool {
    let V = m.vocab
    var pl: [[Float]] = []
    eng.onPromptLogits = { row in
        let p = m.logits.contents().bindMemory(to: Float.self, capacity: (row + 1) * V)
        pl.append(Array(UnsafeBufferPointer(start: p + row * V, count: V)))
    }
    defer { eng.onPromptLogits = nil }
    eng.clear()
    eng.prefixCacheBytes = cacheBytes
    var ok = true
    // The copy policy must not carry over between requests. Run 0 starts with a policy that learned that every
    // copy is accepted and costs nothing (it copies whenever there is a proposal), the repetitions with one that learned
    // that none is (it never copies). Each request has to start from the priors and ignore them; if it did not, the
    // round structure would differ and with it the uniforms of each round and the text (Flash-Next, from token 74).
    // The toys seldom copy (the sampled text does not follow the proposals): this only bites on the real models.
    var alwaysCopy = CopyPolicy(maxT: CopyPolicy.maxRows), neverCopy = alwaysCopy
    for match in [3, 8, 16, 30] {
        for _ in 0..<200 {
            alwaysCopy.observeCopy(rows: CopyPolicy.maxRows, accepted: CopyPolicy.maxRows - 1, match: match, ms: 1)
            neverCopy.observeCopy(rows: CopyPolicy.maxRows, accepted: 0, match: match, ms: 1000)
        }
    }
    func repeated(_ name: String, _ prompt: [UInt32], _ s: SamplingParams) throws -> [UInt32] {
        pl = []
        var runs: [InferenceEngine.Result] = []
        for i in 0..<3 {
            eng.copyState.policy = i == 0 ? alwaysCopy : neverCopy  // must be ignored (above)
            runs.append(try eng.generate(prompt: prompt, maxTokens: n, sampling: s, stopTokens: []))
        }
        var line = "  \(name) (\(prompt.count) tokens):"
        for (i, r) in runs.enumerated() {
            let sameL = pl[i] == pl[0], sameT = r.stats.tokens == runs[0].stats.tokens
            ok = ok && sameL && sameT && (i == 0 || r.cachedTokens == prompt.count)
            let at = firstDifference(r.stats.tokens, runs[0].stats.tokens) ?? -1
            line +=
                " run \(i) \(r.cacheSource) \(r.cachedTokens): logits "
                + (sameL ? "bit-identical" : "DIFFERENT (" + logitDistanceNote(pl[i], pl[0]) + ")") + ", tokens "
                + (sameT ? "identical" : "DIFFERENT from \(at)")
                + ", copy rounds \(r.stats.copyRounds)/\(r.stats.rounds);"
        }
        print(line)
        return runs[0].stats.tokens
    }
    let s = SamplingParams(temperature: 1, topP: 0.95, topK: 20, seed: 1)
    let out = try repeated("from scratch", C, s)
    // The continuation: from the live state of the first run (decoded again from scratch, same seed).
    eng.clear()
    _ = try eng.generate(prompt: C, maxTokens: n, sampling: s, stopTokens: [])
    var sp = s
    sp.presencePenalty = 0.5
    _ = try repeated("continuation", C + out + Array(B.suffix(40)), sp)
    eng.clear()
    print("sampled repetition (T 1, top-k 20, top-p 0.95, seed 1): \(ok ? "OK" : "FAILED")")
    return ok
}

/// lab serve's defaults, the prefix cache pi uses: on MoE models the lossy prefill (--int8 1 --qk8 1, from the
/// family's threshold) with the family's int8 experts (Flash-Next's Q2 ones; the rest exact); --int8 0 --qk8 0 for
/// the exact prefill. Not its expert skipping (Flash-Next's SSD experts): it depends on read timing and is not
/// reproducible (--expert-skip-rank N turns it on). Rounds without timings (as in gen --golden-logits): with the measured copy
/// policy, the rows of each round depend on the machine load and step 6 repeats a history that has to end the same.
@MainActor private func prefixCheckOptions(_ cfg: ModelConfig) throws -> EngineOptions {
    var opts = try engineOptions(cfg, serve: true)
    if args.opts["expert-skip-rank"] == nil { opts.expertSkipRank = nil }
    opts.bench.copyPolicyUntimed = true
    return opts
}

func lcp(_ a: [UInt32], _ b: [UInt32]) -> Int {
    var i = 0
    while i < min(a.count, b.count) && a[i] == b[i] { i += 1 }
    return i
}

/// 6. "Conversation" cut of a turn that continues the conversation and crosses a threshold of the lossy prefill
/// (--int8 1: the family's `int8MinPos` position; in serve, also Q·Kᵀ from 8192). The threshold splits the last
/// span, and the cut, captured as a row of the last chunk, is counted from the new last chunk: the state was already
/// repositioned (fitCaptures) but the drafter's ring was not, and lab serve crashed on the precondition of `ringAt`
/// (Drafter.swift) or captured the ring of another row. It checks:
/// a) that the entry's ring is bit-identical to that of a pass ending at the cut (same history);
/// b) returning to the conversation from that entry (editing the turn after the cut): greedy equal to the reference
///    split at the same point with the same history (the drafter rounds, informative).
/// Without --int8 there is no threshold and the sequence goes through the normal case (also valid).
@MainActor func convCutAcrossBound(_ eng: InferenceEngine, _ m: HybridModel, _ A: [UInt32], cacheBytes: Int) throws
    -> Bool
{
    let greedy = SamplingParams()
    let T = (m.int8MinPos + 63) / 64 * 64
    // Every family has the threshold at 1024. With a lower one (only with a hand-changed threshold) the geometry does
    // not hold: with 512 the turn of T + 176 tokens left no entry ("none 0"), also without int8.
    guard A.count >= T + 600, T >= 1024 else {
        print("cut crossing threshold \(T): skipped (short prompt or threshold < 1024)"); return true
    }
    // Without the user marker or the generation marker: only the conversation cut (at the multiple of 64 before
    // n − 1). From the live state (~T − 128, with 8 decode tokens) to n = T + 176 (≥ expertBulkMinRows new rows): the
    // threshold falls 128 rows from the start of the span and the cut < 64 from the end.
    let filler = Array(A[100...])
    eng.clear()
    eng.prefixCacheBytes = cacheBytes
    let P1 = Array(A[..<(T - 136)])
    let r1 = try eng.generate(prompt: P1, maxTokens: 8, sampling: greedy, stopTokens: [])
    let p1 = m.P
    let head2 = P1 + r1.stats.tokens
    let P2 = head2 + Array(filler.prefix(T + 176 - head2.count))
    let r2 = try eng.generate(prompt: P2, maxTokens: 8, sampling: greedy, stopTokens: [])
    let conv = (P2.count - 1) / 64 * 64
    // a) The drafter ring of the entry against that of a pass of the same history ending at the cut.
    var ringOK = true
    if let dr = eng.drafter {
        // Only the positions of the window before the cut (drafterRing).
        func ring() -> [UInt16] { drafterRing(dr, before: conv) }
        eng.forgetLive()
        let again = try eng.generate(prompt: P1, maxTokens: 8, sampling: greedy, stopTokens: [])
        // After the decode, the last token may already be inside (P = head2) or not (P = head2 − 1, depending on where
        // the last drafter round ends; seen with a former MoE toy): the reference continues from the same P as r1.
        if again.stats.tokens == r1.stats.tokens && m.nacc == 0 && m.P == p1
            && (p1 == head2.count || p1 == head2.count - 1)
        {
            _ = try m.prefill(Array(P2[m.P..<conv]), drafter: dr, continuing: true, head: false)
            let ref = ring()
            eng.forgetLive()
            ringOK = eng.restoreEntryForTest(tokens: Array(P2[..<conv])) && ring() == ref
        } else {
            ringOK = false
            print("  ring: the reference does not reproduce the state (P \(m.P), nacc \(m.nacc))")
        }
        eng.forgetLive()
    }
    // b) It is abandoned (another prompt) and resumed editing the turn after the cut: the longest entry is the cut's.
    _ = try eng.generate(prompt: Array(A[..<(T + 76)]), maxTokens: 4, sampling: greedy, stopTokens: [])
    let P3 = Array(P2[..<(conv + 20)]) + Array(filler[300..<400])
    let r3 = try eng.generate(prompt: P3, maxTokens: 64, sampling: greedy, stopTokens: m.eosSet)
    let k = r3.cachedTokens
    // Reference without prefix cache and with the same history (P1 and its decode), split at k: P3[..<k + 1] from the
    // live state and P3 from snapshot (b) at k.
    eng.clear()
    eng.prefixCacheBytes = 0
    var ref: InferenceEngine.Result?
    if k > head2.count && k < P3.count {
        _ = try eng.generate(prompt: P1, maxTokens: 8, sampling: greedy, stopTokens: [])
        _ = try eng.generate(prompt: Array(P3[..<(k + 1)]), maxTokens: 1, sampling: greedy, stopTokens: m.eosSet)
        eng.forgetLiveOnly()
        ref = try eng.generate(prompt: P3, maxTokens: 64, sampling: greedy, stopTokens: m.eosSet)
    }
    eng.prefixCacheBytes = cacheBytes
    eng.clear()
    let same = ref.map { $0.stats.tokens == r3.stats.tokens } ?? false
    print(
        "conversation cut crossing threshold \(T): live \(r2.cachedTokens), prompt \(P2.count), cut \(conv); "
            + "entry ring against a pass up to the cut: "
            + (eng.drafter == nil ? "no drafter" : ringOK ? "bit-identical" : "DIFFERENT")
            + "; return (\(P3.count) tokens): \(r3.cacheSource) \(k); greedy against split at \(k): "
            + (same ? "identical" : "DIFFERENT")
            + "; drafter rounds \(r3.stats.rounds) against \(ref?.stats.rounds ?? -1) (informative)")
    return ringOK && same && r3.cacheSource == "entry" && k == conv
}

/// 7. Entries at chosen cuts: by default, with a ring KV, around the window W (W − 1, W, W + 1) and the ring
/// (ring ± 64: the window of the entry in one piece of the ring, and wrapped around its end); `--cuts` for others.
/// Without a ring and without `--cuts` it is skipped. For each cut c of the 4K prompt X, an entry at c ("common": two
/// prompts that share X[..<c + 8]; cutAlign 1 if the model has attention layers only, so that c need not be a multiple
/// of 64), then: a) restored onto a ring overwritten by other prompts, the logits of reading the rest and 3 decode steps
/// bit for bit equal to building the same state without cache; b) a turn X[..<c + 300] from the entry (drafter rounds
/// over the restored window): greedy identical to the same path without the prefix cache, split at c.
@MainActor private func entriesAtCuts(_ eng: InferenceEngine, _ m: HybridModel, cacheBytes: Int) throws -> Bool {
    let attn = m.layers.compactMap {
        if case .attn(let a) = $0.mixer { return a }; return nil
    }
    let X = try promptIds("4K")
    var cuts = args.str("cuts", "").split(separator: ",").compactMap { Int($0) }
    if cuts.isEmpty, let r = attn.first(where: { $0.ring > 0 }) {
        cuts = [r.window - 1, r.window, r.window + 1, r.ring - 64, r.ring + 64]
    }
    cuts = cuts.filter { $0 >= 64 && $0 + 400 <= X.count }
    guard !cuts.isEmpty else {
        print("entries at chosen cuts: skipped (no ring and no --cuts, or short prompt)"); return true
    }
    let attnOnly = attn.count == m.layers.count
    let dr = eng.drafter, greedy = SamplingParams(), V = m.vocab
    let (oldAlign, oldMin) = (eng.cutAlign, eng.minEntryTokens)
    defer { (eng.cutAlign, eng.minEntryTokens) = (oldAlign, oldMin) }
    if attnOnly { eng.cutAlign = 1 }
    eng.minEntryTokens = 64
    func logits() -> [Float] {
        Array(UnsafeBufferPointer(start: m.logits.contents().bindMemory(to: Float.self, capacity: V), count: V))
    }
    func tail(from c: Int) throws -> [[Float]] {
        _ = try m.prefill(Array(X[c..<(c + 300)]), drafter: dr, continuing: true)
        var out = [logits()]
        for t: UInt32 in [198, 279, 3681] { _ = try m.stepLogits(t); out.append(logits()) }
        return out
    }
    var ok = true
    for c in cuts where attnOnly || c % 64 == 0 {
        eng.clear()
        eng.prefixCacheBytes = cacheBytes
        let head = Array(X[..<(c + 8)])
        // Two 48-token endings from the far end of X (gpt-oss's o200k makes its 4K prompt 2889 tokens long).
        let (e1, e2) = (min(3000, X.count - 148), min(3100, X.count - 48))
        _ = try eng.generate(prompt: head + Array(X[e1..<(e1 + 48)]), maxTokens: 1, sampling: greedy, stopTokens: [])
        _ = try eng.generate(prompt: head + Array(X[e2..<(e2 + 48)]), maxTokens: 1, sampling: greedy, stopTokens: [])
        // a) Overwrite the whole ring (and the token history) with other prompts, then restore.
        _ = try eng.generate(prompt: Array(X[1000...]), maxTokens: 4, sampling: greedy, stopTokens: [])
        _ = try m.prefill(Array(X.reversed()), drafter: dr)
        guard eng.restoreEntryForTest(tokens: Array(X[..<c])) else {
            print("  cut \(c): entry missing (entries \(eng.prefixCacheSummary))"); ok = false; continue
        }
        perturbForTest(m, "entry of \(c) tokens restored")
        let got = try tail(from: c)
        m.reset()
        _ = try m.prefill(Array(X[..<c]), drafter: dr, head: false)
        let ref = try tail(from: c)
        let same = got == ref
        ok = ok && same
        // b) A turn from the entry against the same path without prefix cache, split at c.
        eng.forgetLive()
        let P = Array(X[..<(c + 300)])
        let r = try eng.generate(prompt: P, maxTokens: 64, sampling: greedy, stopTokens: m.eosSet)
        eng.clear()
        eng.prefixCacheBytes = 0
        _ = try eng.generate(prompt: Array(P[..<(c + 1)]), maxTokens: 1, sampling: greedy, stopTokens: m.eosSet)
        eng.forgetLiveOnly()
        let rs = try eng.generate(prompt: P, maxTokens: 64, sampling: greedy, stopTokens: m.eosSet)
        let sameGen =
            r.cacheSource == "entry" && r.cachedTokens == c && rs.cacheSource == "snapshot" && rs.cachedTokens == c
            && r.stats.tokens == rs.stats.tokens
        ok = ok && sameGen
        print(
            "  entry of \(c) tokens: logits after restoring \(same ? "bit-identical" : "DIFFERENT") to without cache"
                + (same ? "" : " (" + logitDistanceNote(got[0], ref[0]) + ")")
                + "; turn of \(P.count): \(r.cacheSource) \(r.cachedTokens) against \(rs.cacheSource) "
                + "\(rs.cachedTokens), greedy \(r.stats.tokens == rs.stats.tokens ? "identical" : "DIFFERENT")"
                + " (rounds \(r.stats.rounds) and \(rs.stats.rounds))")
    }
    eng.clear()
    eng.prefixCacheBytes = cacheBytes
    print("entries at chosen cuts \(cuts): \(ok ? "OK" : "FAILED")")
    return ok
}

/// What changes between the sessions of `prefixcache` (see sessionBody).
private enum SessionChange { case cwd, tools, user }

/// Body of session `tag` for `prefixcache`. What changes between sessions: the `<cwd>` at the end of pi's system
/// message; with `.tools`, the last tool description (the end of the system block if the template puts the tools
/// after the system message, as LFM2.5 does); with `.user`, the end of the last text of the body (the user message,
/// for templates without tools, as Gemma 3).
private func sessionBody(_ raw: String, tag: String, change: SessionChange) throws -> String {
    var s = raw
    switch change {
    case .tools:
        guard let r = raw.range(of: #""description":""#, options: .backwards) else {
            throw SafetensorsError.format("no tool description found in the body")
        }
        s.insert(contentsOf: "task.\(tag) ", at: r.upperBound)
    case .user:
        guard let r = raw.range(of: #""text":""#, options: .backwards) else {
            throw SafetensorsError.format("no user text found in the body")
        }
        // The closing quote of that JSON string: the first one not escaped.
        var i = r.upperBound
        while i < raw.endIndex, raw[i] != "\"" {
            if raw[i] == "\\" { i = raw.index(after: i) }
            i = raw.index(after: i)
        }
        s.insert(contentsOf: " (task.\(tag))", at: i)
    case .cwd:
        let re = try NSRegularExpression(pattern: #"<cwd>\\n[^\\]*?\\n</cwd>"#)
        s = re.stringByReplacingMatches(
            in: raw, range: NSRange(raw.startIndex..., in: raw),
            withTemplate:
                "<cwd>\\\\n/private/tmp/task.\(tag)/work\\\\n</cwd>")
        precondition(s != raw, "no <cwd> found in the body")
    }
    return s
}

/// The system entry the engine leaves for prompt `p` (PrefixCache.prefixCuts: 64-127 tokens before the first user
/// turn), 0 without a user marker. Nemotron-H's template puts the tools after the system message and ends its system
/// block with ~200 fixed tokens of instructions: a changed tool lands before this cut, so B cannot resume from A and
/// prefixcache changes the user message instead, as with Gemma 4.
@MainActor private func systemEntryCut(_ eng: InferenceEngine, _ p: [UInt32]) -> Int {
    let u = eng.userMarker
    guard !u.isEmpty, p.count >= u.count,
        let i = (0...(p.count - u.count)).first(where: { Array(p[$0..<($0 + u.count)]) == u })
    else { return 0 }
    return i / eng.cutAlign * eng.cutAlign - eng.cutAlign
}
