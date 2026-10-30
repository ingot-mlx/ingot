import Foundation
import Ingot
import IngotCLI
import Metal

/// ingot-dev rowscheck --pdir prompts-es --prompts es-chat,agent-t1-nt --n 64: with a drafter, verification trimmed to
/// k rows on the GPU gives the same logits (bit by bit) in rows < k as the 8-row one, and those do not depend on the
/// tokens of rows ≥ k. On the state after generating `n` tokens; real tokens that follow (the greedy text) and random
/// ones. Without a drafter, rounds of k rows through the verification path against k steps of one row
/// (`rowsStepsCheck`; `--n` and `--reps` do not apply).
@MainActor func cmdRowsCheck() throws {
    let model = try args.required("model", modelHelp)
    let names = args.str("prompts", "es-chat").split(separator: ",").map(String.init)
    let n = args.int("n", 64), reps = args.int("reps", 6)
    let (dir, cfg, ctx) = try openModel(model)
    let prompts = try names.map { ($0, try promptIds($0)) }
    let m = try HybridModel(
        ctx: ctx, store: try SafetensorsStore(directory: dir), config: cfg,
        maxCtx: (prompts.map { $0.1.count }.max() ?? 0) + max(n, RowsSteps.tokens) + 64,
        options: try engineOptions(cfg))
    guard let dr = try openModelDrafter(ctx, dir, cfg, m) else { return try rowsStepsCheck(m, prompts) }
    var g = SplitMix64(seed: 7)
    var bad = 0, total = 0
    for (name, ids) in prompts {
        // Reference greedy text (n + 16 tokens) and the state after the first n.
        let ref = try m.generatePipelined(
            prompt: ids, drafter: dr, maxTokens: n + 16, stopAtEOS: false, genStart: genStart(ids)
        ).tokens
        for rep in 0..<reps {
            let cut = n - 8 * rep
            _ = try m.generatePipelined(
                prompt: ids, drafter: dr, maxTokens: cut, stopAtEOS: false, genStart: genStart(ids))
            // decodeSpec leaves P behind the last emitted token except itself (which is the input of the next round).
            let start = m.P - ids.count
            let toks = Array(ref[start..<(start + 8)])
            let alt = (0..<8).map { _ in UInt32(g.uniform() * Float(m.vocab - 1)) }
            for k in 1...8 {
                let r = try m.rowLimitBitCheck(toks: toks, alt: alt, k: k)
                total += 1
                if !(r.masked && r.laterRows) { bad += 1 }
                print(
                    String(
                        format: "%@ P=%d k=%d: trimmed %@, rows ≥ k changed %@ (max |Δ| %.3g)", name, m.P, k,
                        r.masked ? "identical" : "DIFFERENT", r.laterRows ? "identical" : "DIFFERENT", r.maxDiff))
            }
        }
    }
    print("rowscheck: \(total - bad) of \(total) cases bit-identical")
    if bad > 0 { exit(1) }
}

/// Round sizes of the drafter-less check: every k from 1 to 16 (AttentionLayer.maxRows) over the greedy text, then
/// three rounds over random tokens (a full one and two odd sizes).
enum RowsSteps {
    static let greedy = Array(1...AttentionLayer.maxRows)
    static let rounds = greedy + [AttentionLayer.maxRows, 5, 11]
    static let tokens = rounds.reduce(0, +)
    static let greedyTokens = greedy.reduce(0, +)
}

/// Rowscheck without a drafter. After the prefill of each prompt the reference is the decode one row at a
/// time (`stepLogits`: B = 1, each row folded inside the next pass, as decodePlain); then, from the same prefill, the
/// same tokens in rounds of k rows through the verification path (B = k, as the copy rounds and the short prompt
/// tails, folded by the next round). Each row's logits and argmax have to be bit-identical to its step's: a kernel
/// whose result depends on the row's place in the round or on the number of rows (Mamba2's fma) fails
/// here, and since each side carries its own state, so does a fold of k rows that differs from k folds of one. The
/// tokens are the greedy text of the steps (rounds of 1…16 rows, 136 tokens) and then random ones.
/// Negative control (hidden `--perturb`): the rounds go through the prefill instead, which sums in another order,
/// and only their last row is compared; the check has to report FAILED.
@MainActor func rowsStepsCheck(_ m: HybridModel, _ prompts: [(String, [UInt32])]) throws {
    let V = m.vocab, viaPrefill = args.flags.contains("perturb")
    if viaPrefill {
        print("  [negative control] rounds of k rows through the prefill instead of the verification path")
    }
    func rowLogits(_ r: Int) -> [Float] {
        let p = m.logits.contents().bindMemory(to: Float.self, capacity: (r + 1) * V)
        return Array(UnsafeBufferPointer(start: p + r * V, count: V))
    }
    func top(_ r: Int) -> UInt32 { m.top1.contents().bindMemory(to: UInt32.self, capacity: r + 1)[r] }
    /// k rows at positions P… in one pass of the verification path.
    func round(_ toks: [UInt32]) throws {
        let tp = m.tok.contents().bindMemory(to: UInt32.self, capacity: AttentionLayer.maxRows)
        for (j, t) in toks.enumerated() { tp[j] = t }
        guard let cb = m.ctx.queue.makeCommandBuffer(), let e = cb.makeComputeCommandEncoder() else {
            throw SafetensorsError.io("cb")
        }
        try m.encodeVerify(e, rows: toks.count, drafter: nil, cb: cb).endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if let err = cb.error { throw err }
        m.commit(accepted: toks.count)
    }
    var g = SplitMix64(seed: 7)
    var bad = 0, total = 0
    for (name, ids) in prompts {
        // Reference: one row per step; the greedy text, then random tokens.
        var t = try m.prefill(ids, drafter: nil).token
        var seq: [UInt32] = [], refLogits: [[Float]] = [], refTop: [UInt32] = []
        for i in 0..<RowsSteps.tokens {
            if i >= RowsSteps.greedyTokens { t = UInt32(g.uniform() * Float(V - 1)) }
            seq.append(t)
            let next = try m.stepLogits(t)
            refLogits.append(rowLogits(0))
            refTop.append(next)
            t = next
        }
        // The same tokens in rounds of k rows from the same prefill.
        _ = try m.prefill(ids, drafter: nil)
        var i = 0
        for k in RowsSteps.rounds {
            let p0 = m.P, toks = Array(seq[i..<(i + k)])
            // (row of the round, row of the logits buffer): the prefill only leaves the last row, in row 0.
            var rows = (0..<k).map { ($0, $0) }
            if viaPrefill {
                try m.foldPending(drafter: nil, context: nil)
                _ = try m.prefill(toks, drafter: nil, continuing: true)
                rows = [(k - 1, 0)]
            } else {
                try round(toks)
            }
            var diffLogits = 0, diffTop = 0
            var first: String?
            for (r, b) in rows {
                let got = rowLogits(b), ref = refLogits[i + r]
                if !zip(got, ref).allSatisfy({ $0.bitPattern == $1.bitPattern }) {
                    diffLogits += 1
                    if first == nil { first = "first row \(r): " + logitDistanceNote(got, ref) }
                }
                if top(b) != refTop[i + r] { diffTop += 1 }
            }
            total += 1
            let same = diffLogits == 0 && diffTop == 0
            if !same { bad += 1 }
            let detail =
                same
                ? "bit-identical"
                : "DIFFERENT (logits in \(diffLogits) of \(rows.count) rows, argmax in \(diffTop)"
                    + (first.map { "; \($0)" } ?? "") + ")"
            print("\(name) P=\(p0) k=\(k) \(i >= RowsSteps.greedyTokens ? "random" : "greedy"): \(detail)")
            i += k
        }
    }
    print(
        bad == 0
            ? "rowscheck: OK (\(total) rounds of k rows = k steps of one row, bit by bit)"
            : "rowscheck: FAILED (\(bad) of \(total) rounds)")
    if bad > 0 { exit(1) }
}
