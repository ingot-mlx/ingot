import Foundation
import Ingot
import IngotCLI
import Metal

/// ingot-dev batchcheck --prompts 512,4K,chat --rows 2,4,8 --n 32 [--reps R]: batched decoding
/// (`HybridModel.generateBatch`) gives each conversation, bit for bit, the tokens and logits it gives alone. N
/// conversations per `--rows` value; conversation j takes prompt j % count, and the repeats lose 37·(j / count)
/// tokens from the end so that every conversation has its own position. With `--reps R > 0` it also times, in the
/// same process and alternating the order (A B C, then C B A), the N conversations one after another with
/// `generate` (one wait per step) and with `generatePipelined` (chained; with the copy that --copy says) against the
/// batch; greedy and `n` tokens per conversation in all three.
///
/// With `--seed S` it then checks sampling too: conversation j samples with its own seed S + j and the parameters
/// of row j % 4 of `batchSampling` (temperature, top-k, top-p, min-p and penalties differ between conversations, and
/// one is greedy with penalties), and must give the same ids as alone by the `ingot serve` path without copy
/// (`sampleFromLogits` for the first token, then `decode` → `decodePlain` with the same generator). The timing then
/// compares the batch against the N conversations one after another by that path, all sampled.
///
/// With `--serve`, ingot serve's engine loop with slots instead (`cmdBatchServeCheck`).
@MainActor func cmdBatchCheck() throws {
    if args.flags.contains("serve") { return try cmdBatchServeCheck() }
    func bitEqual(_ a: [Float], _ b: [Float]) -> Bool {
        a.count == b.count
            && a.withUnsafeBytes { x in b.withUnsafeBytes { memcmp(x.baseAddress!, $0.baseAddress!, x.count) == 0 } }
    }
    let model = try args.required("model", modelHelp)
    let names = args.str("prompts", "512,4K,chat").split(separator: ",").map(String.init)
    let sizes = args.str("rows", "2,4,8").split(separator: ",").compactMap { Int($0) }
    let n = args.int("n", 32), reps = args.int("reps", 0)
    let seed = args.opts["seed"].flatMap(UInt64.init)
    let (dir, cfg, ctx) = try openModel(model)
    let base = try names.map { try promptIds($0) }
    guard let maxN = sizes.max(), !base.isEmpty, n >= 2 else {
        throw UsageError("batchcheck: --rows, --prompts, --n ≥ 2")
    }
    func prompt(_ j: Int) -> [UInt32] {
        let p = base[j % base.count]
        return Array(p.prefix(max(16, p.count - 37 * (j / base.count))))
    }
    let prompts = (0..<maxN).map(prompt)
    let m = try HybridModel(
        ctx: ctx, store: try SafetensorsStore(directory: dir), config: cfg,
        maxCtx: (prompts.map(\.count).max() ?? 0) + n + 64, options: try engineOptions(cfg))
    if let why = m.batchUnsupported { throw UsageError("batchcheck: this model has \(why)") }
    let V = m.vocab
    func logitsRow(_ r: Int) -> [Float] {
        Array(
            UnsafeBufferPointer(
                start: m.logits.contents().bindMemory(to: Float.self, capacity: (r + 1) * V) + r * V, count: V))
    }
    // Alone: the step-by-step path of `generate` (B = 1), with the logits of each step.
    var solo: [(ids: [UInt32], logits: [[Float]])] = []
    for (j, p) in prompts.enumerated() {
        let (first, _, _) = try m.preparePrompt(p, from: 0, genStart: genStart(p), drafter: nil)
        var ids = [first], lg: [[Float]] = []
        while ids.count < n {
            ids.append(try m.stepLogits(ids.last!))
            lg.append(logitsRow(0))
        }
        solo.append((ids, lg))
        print("conversation \(j): \(p.count) prompt tokens, alone \(ids.count) tokens")
    }
    var bad = 0, total = 0
    for N in sizes where N >= 2 {
        var step = [Int](repeating: 0, count: N), diff = [String?](repeating: nil, count: N)
        let st = try m.generateBatch(
            prompts: Array(prompts[0..<N]), maxTokens: n, stopAtEOS: false, genStarts: prompts[0..<N].map(genStart)
        ) { convs in
            for (r, j) in convs.enumerated() {
                let k = step[j]
                step[j] += 1
                guard diff[j] == nil else { continue }
                let x = logitsRow(r), ref = solo[j].logits[k]
                if !bitEqual(x, ref) { diff[j] = "logits differ from step \(k) (\(logitDistanceNote(x, ref)))" }
            }
        }
        for j in 0..<N {
            total += 1
            let same = st[j].tokens == solo[j].ids && diff[j] == nil
            if !same { bad += 1 }
            let ids = st[j].tokens == solo[j].ids ? "same \(n) ids" : "DIFFERENT ids"
            print(
                "N=\(N) conversation \(j): \(same ? "identical" : "DIFFERENT"): \(ids), \(diff[j] ?? "logits identical")"
            )
        }
    }
    if let seed {
        // Sampling: the same ids as alone with the same seed (by decision), not only the same law.
        let sp = (0..<maxN).map { batchSampling($0, seed: seed) }
        var sampled: [[UInt32]] = []
        for (j, p) in prompts.enumerated() {
            sampled.append(try soloSampled(m, p, sp[j], n: n))
            let k = firstDifference(sampled[j], solo[j].ids)
            print(
                "conversation \(j) sampled (\(batchSamplingNote(sp[j]))): alone \(sampled[j].count) tokens, "
                    + (k.map { "leaves greedy at token \($0)" } ?? "the same as greedy"))
        }
        for N in sizes where N >= 2 {
            let st = try m.generateBatch(
                prompts: Array(prompts[0..<N]), maxTokens: n, stopAtEOS: false,
                genStarts: prompts[0..<N].map(genStart), sampling: Array(sp[0..<N]))
            for j in 0..<N {
                total += 1
                let k = firstDifference(st[j].tokens, sampled[j])
                let same = st[j].tokens == sampled[j]
                if !same { bad += 1 }
                print(
                    "N=\(N) conversation \(j) sampled: "
                        + (same
                            ? "identical: same \(n) ids"
                            : "DIFFERENT ids from token \(k ?? min(st[j].tokens.count, sampled[j].count))"))
            }
        }
    }
    print("batchcheck: \(total - bad) of \(total) conversations bit-identical")
    if bad > 0 {
        print("batchcheck: FAILED")
        exit(1)
    }
    guard reps > 0 else { return }
    // Timing: decode only (prefill excluded), the sum of the N conversations one after another against the batch.
    print("timing: copy \(m.options.copyEnabled(moe: m.isMoE, drafter: false) ? "on" : "off") in the chained base")
    if let seed {
        try timeSampled(m, prompts: prompts, sizes: sizes, n: n, reps: reps, seed: seed)
        return
    }
    for N in sizes where N >= 2 {
        let ps = Array(prompts[0..<N])
        var res: [String: [(tps: Double, ms: Double)]] = [:]
        func run(_ kind: String) throws {
            var toks = 0, secs = 0.0, rounds = 0
            switch kind {
            case "batch":
                let st = try m.generateBatch(prompts: ps, maxTokens: n, stopAtEOS: false, genStarts: ps.map(genStart))
                toks = st.reduce(0) { $0 + $1.tokens.count - 1 }
                secs = st[0].decodeSeconds
                rounds = st.map(\.rounds).max() ?? 0
            default:
                for p in ps {
                    let st =
                        kind == "normal"
                        ? try m.generate(prompt: p, maxTokens: n, stopAtEOS: false, genStart: genStart(p))
                        : try m.generatePipelined(
                            prompt: p, drafter: nil, maxTokens: n, stopAtEOS: false, genStart: genStart(p))
                    toks += st.tokens.count - 1
                    secs += st.decodeSeconds
                    rounds += st.rounds
                }
            }
            res[kind, default: []].append((Double(toks) / secs, secs * 1e3 / Double(max(1, rounds))))
        }
        let order = ["normal", "plain", "batch"]
        for rep in 0..<reps { for k in rep % 2 == 0 ? order : order.reversed() { try run(k) } }
        func med(_ k: String) -> (tps: Double, ms: Double) {
            let a = res[k] ?? []
            return (median(a.map(\.tps)), median(a.map(\.ms)))
        }
        let b = med("batch"), no = med("normal"), pl = med("plain")
        print(
            String(
                format:
                    "N=%d: batch %.1f tok/s (%.2f ms/round) | alone normal %.1f tok/s (%.2f ms/round): ×%.2f | "
                    + "alone chained %.1f tok/s (%.2f ms/round): ×%.2f | %d reps, medians",
                N, b.tps, b.ms, no.tps, no.ms, b.tps / no.tps, pl.tps, pl.ms, b.tps / pl.tps, reps))
    }
}

/// The sampling of conversation j in `batchcheck --seed`: four kinds that cover each knob, each with its own seed.
func batchSampling(_ j: Int, seed: UInt64) -> SamplingParams {
    let sd = seed &+ UInt64(j)
    switch j % 4 {
    case 0: return SamplingParams(temperature: 0.7, topP: 0.95, topK: 20, seed: sd)
    case 1: return SamplingParams(temperature: 1, seed: sd, minP: 0.05)
    case 2:
        return SamplingParams(
            temperature: 0.8, topP: 0.9, topK: 40, seed: sd, repetitionPenalty: 1.3, presencePenalty: 1.5,
            frequencyPenalty: 0.5)
    default: return SamplingParams(seed: sd, repetitionPenalty: 1.1, frequencyPenalty: 0.2)
    }
}

func batchSamplingNote(_ s: SamplingParams) -> String {
    String(
        format: "temp %.1f, top-k %d, top-p %.2f, min-p %.2f, rep %.1f, pres %.1f, freq %.1f, seed %llu",
        s.temperature, s.topK, s.topP, s.minP, s.repetitionPenalty, s.presencePenalty, s.frequencyPenalty,
        s.seed ?? 0)
}

/// One conversation alone, sampled by the `ingot serve` path without copy: the first token from the prompt's last row,
/// then the chained decode (`decodePlain`) with the same generator.
@MainActor func soloSampled(_ m: HybridModel, _ p: [UInt32], _ sp: SamplingParams, n: Int) throws -> [UInt32] {
    try soloSampledStats(m, p, sp, n: n).tokens
}

@MainActor func soloSampledStats(_ m: HybridModel, _ p: [UInt32], _ sp: SamplingParams, n: Int) throws
    -> HybridModel.GenStats
{
    let (argmax, row, _) = try m.preparePrompt(p, from: 0, genStart: genStart(p), drafter: nil)
    var rng = SplitMix64(seed: sp.seed ?? 0)
    m.preparePenalties(prompt: p, params: sp)
    let first = try m.sampleFromLogits(argmax: argmax, row: row, params: sp, rng: &rng)
    return try m.decode(first: first, drafter: nil, maxTokens: n, sampling: sp, rng: &rng, stopTokens: [])
}

/// `batchcheck --seed --reps`: decode only, the N conversations sampled one after another (chained, as `ingot serve`)
/// against the sampled batch, alternating the order.
@MainActor func timeSampled(_ m: HybridModel, prompts: [[UInt32]], sizes: [Int], n: Int, reps: Int, seed: UInt64) throws
{
    for N in sizes where N >= 2 {
        let ps = Array(prompts[0..<N]), sp = (0..<N).map { batchSampling($0, seed: seed) }
        var res: [String: [(tps: Double, ms: Double)]] = [:]
        func run(_ kind: String) throws {
            var toks = 0, secs = 0.0, rounds = 0
            if kind == "batch" {
                let st = try m.generateBatch(
                    prompts: ps, maxTokens: n, stopAtEOS: false, genStarts: ps.map(genStart), sampling: sp)
                toks = st.reduce(0) { $0 + $1.tokens.count - 1 }
                secs = st[0].decodeSeconds
                rounds = st.map(\.rounds).max() ?? 0
            } else {
                for (j, p) in ps.enumerated() {
                    let st = try soloSampledStats(m, p, sp[j], n: n)
                    toks += st.tokens.count - 1
                    secs += st.decodeSeconds
                    rounds += st.rounds
                }
            }
            res[kind, default: []].append((Double(toks) / secs, secs * 1e3 / Double(max(1, rounds))))
        }
        for rep in 0..<reps { for k in rep % 2 == 0 ? ["plain", "batch"] : ["batch", "plain"] { try run(k) } }
        func med(_ k: String) -> (tps: Double, ms: Double) {
            let a = res[k] ?? []
            return (median(a.map(\.tps)), median(a.map(\.ms)))
        }
        let b = med("batch"), pl = med("plain")
        print(
            String(
                format:
                    "N=%d sampled: batch %.1f tok/s (%.2f ms/round) | alone chained %.1f tok/s (%.2f ms/round): ×%.2f | "
                    + "%d reps, medians",
                N, b.tps, b.ms, pl.tps, pl.ms, b.tps / pl.tps, reps))
    }
}
