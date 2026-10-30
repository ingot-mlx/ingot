import Foundation
import LabBench
import LabCLI
import Metal
import MetalLab

// lab-dev gen: end-to-end decode with and without DFlash2. Prompts tokenized by ref/prompts.py (ids in
// ref/out/prompts/<name>.json); the output is written to ref/out/gen/<name>_lab_<mode>.json.

/// lab-dev gen --prompts chat,512,4K --n 512 --mode both|spec|normal [--nostop] [--jsonl path --label label]
///             [--check] [--golden f.json [--write-golden]] [--golden-logits [--logits-store dir]]
@MainActor func cmdGen() throws {
    let model = try args.required("model", modelHelp)
    let names = args.str("prompts", "chat").split(separator: ",").map(String.init)
    let n = args.int("n", 128)
    // --check: exactness regression test (temperature 0: with and without drafter, identical tokens); exits 1 if not.
    let check = args.flags.contains("check")
    let (dir, cfg, ctx) = try openModel(model)
    // LFM2 has no drafter: normal mode (one wait per token) and "plain" (chained decode, the one of lab serve);
    // --check compares the two (identical tokens) and the normal one with mlx-lm's greedy (ref/out/<genRefSet>,
    // gen_ref.py).
    // The same without drafter: with --drafter none, a family without one, or when the family's does not fit the model
    // (resolvedDrafterDir: Qwen3.5 0.8B and 2B). A toy whose family has no drafter by default (toy-gptoss) takes its
    // `drafter/` as lab serve does: its golden is the run with it (--drafter none for the run without).
    let mode = check ? "both" : args.str("mode", "both")
    let drafterDir = try genDrafterDir(dir, cfg, mode: mode)
    let noDrafter = drafterDir == nil
    var failures: [String] = []
    var mlxCompared = 0  // prompts whose normal-mode tokens passed the comparison with mlx-lm's greedy
    let stop = !args.flags.contains("nostop")
    let prompts = try names.map { ($0, try promptIds($0)) }
    let maxCtx = (prompts.map { $0.1.count }.max() ?? 0) + n + 64
    let store = try SafetensorsStore(directory: dir)
    // Memory plan (MemPlan.swift) with the context the prompts ask for: checks that it fits and picks the chunk.
    let plan = try memoryPlan(
        dir: dir, cfg: cfg, device: ctx.device,
        options: genMemOptions(maxCtx: maxCtx, drafter: mode != "normal" && !noDrafter))
    print("loading \(model) (maxCtx \(maxCtx))...")
    let m = try HybridModel(
        ctx: ctx, store: store, config: cfg, maxCtx: maxCtx, chunkRows: plan.chunkRows,
        options: try {
            var o = try engineOptions(cfg); applyPlan(&o, plan)
            o.bench.copyPolicyUntimed = args.flags.contains("golden-logits")  // reproducible last-round fingerprint
            return o
        }())
    print(String(format: "model: %.1f s, %.2f GB of weights, KV int8", m.loadSeconds, Double(m.weightBytes) / 1e9))
    describeLossy(m)
    if let p = m.kvPool { print(p.describe()) }
    defer {
        if let p = m.kvPool { print("at exit: \(p.describe()); grew \(p.grows) times, shrank \(p.shrinks)") }
    }
    var dr: DFlash2Drafter?
    if mode != "normal", let drafterDir {
        dr = try openDrafter(ctx, drafterDir, m)
        print(
            String(
                format: "drafter: %.1f s, %.2f GB (%@)", dr!.loadSeconds, Double(dr!.weightBytes) / 1e9,
                dr!.v1 ? "DFlash v1, 4 bits, block 8" : "4 bits + bf16 codebooks"))
    }
    print(String(format: "Metal memory allocated: %.2f GB", Double(ctx.device.currentAllocatedSize) / 1e9))
    checkPlanAfterLoad(plan, device: ctx.device)
    // Warm-up: compiles the pipelines.
    let warm = try promptIds("chat")
    _ = try m.generate(prompt: warm, maxTokens: 4, genStart: genStart(warm))
    if noDrafter { _ = try m.generatePipelined(prompt: warm, drafter: nil, maxTokens: 8, genStart: genStart(warm)) }
    if let dr {
        _ = try m.generateSpec(prompt: warm, drafter: dr, maxTokens: 12, genStart: genStart(warm))
        _ = try m.generatePipelined(prompt: warm, drafter: dr, maxTokens: 12, genStart: genStart(warm))
    }
    // Routing trace of every round (RouteTrace.swift), after the warm-up.
    if let p = args.opts["route-trace"] {
        m.routeTrace = try RouteTrace(path: p, model: m)
    }
    // GPU timestamps inside every round (RoundTrace.swift), after the warm-up.
    let traceLevel = args.int("round-trace", 0)
    if traceLevel > 0 {
        m.roundTrace = try RoundTrace(
            device: ctx.device, level: traceLevel, path: args.str("round-trace-out", "roundtrace.jsonl"))
    }
    var results: [GenResult] = []
    let lossyRecord = genLossyRecord(m, cfg)
    // --golden-logits (ci.sh --toy): besides the tokens, a fingerprint (FNV-1a) of the logits of the last row of the
    // prefill and of those of the last round of generation with a drafter. It catches any numeric change even when no
    // token changes (with the toy models, near-ties are rare).
    let goldenLogits = args.flags.contains("golden-logits")
    var logitHash: [String: [String]] = [:], logitRows: [String: [[Float]]] = [:]
    for (name, ids) in prompts {
        if goldenLogits {
            _ = try m.prefill(ids, drafter: nil)
            logitHash[name] = [logitsFingerprint(m)]
            logitRows[name] = [lastLogits(m)]
        }
        m.routeTrace?.nextLabel = name
        m.roundTrace?.label = name
        if mode != "spec" {
            let st = try m.generate(prompt: ids, maxTokens: n, stopAtEOS: stop, genStart: genStart(ids))
            let r = GenResult(name: name, mode: "normal", promptTokens: ids.count, st: st)
            print(r.line)
            printExpertStats(m, rounds: st.rounds)
            try writeGen(name, "normal", st, options: lossyRecord)
            results.append(r)
            if noDrafter, check {
                mlxCompared += mlxCheck(name, st.tokens, model: dir, isToy: cfg.raw["metal_lab_toy"] != nil, n: n)
                    .record(name, into: &failures)
            }
        }
        if let dr {
            let st = try m.generatePipelined(
                prompt: ids, drafter: dr, maxTokens: n, stopAtEOS: stop, genStart: genStart(ids))
            let r = GenResult(name: name, mode: "spec", promptTokens: ids.count, st: st)
            print(r.line)
            printExpertStats(m, rounds: st.rounds)
            printSpecStats(st)
            try writeGen(name, "spec", st, options: lossyRecord)
            results.append(r)
            if goldenLogits {
                logitHash[name, default: []].append(logitsFingerprint(m))
                logitRows[name, default: []].append(lastLogits(m))
            }
        }
        if noDrafter && mode != "normal" {
            let st = try m.generatePipelined(
                prompt: ids, drafter: nil, maxTokens: n, stopAtEOS: stop, genStart: genStart(ids))
            let r = GenResult(name: name, mode: "plain", promptTokens: ids.count, st: st)
            print(r.line)
            if st.copyRounds > 0 || st.deferred > 0 { print("  " + st.copySummary) }
            try writeGen(name, "plain", st, options: lossyRecord)
            results.append(r)
        }
        // --profile: without a drafter, the batching ceiling and the round by piece at the end of this prompt; the
        // prefill timeline of this prompt.
        try genProfileBatch(m, dr, ctx, label: name)
        try genPieceProfile(m, dr, ctx, label: name)
        try genPrefillTimeline(m, promptTokens: ids.count, label: name)
        if mode == "both", let a = results.dropLast().last, let b = results.last {
            let k = min(a.st.tokens.count, b.st.tokens.count)
            let div = (0..<k).first { a.st.tokens[$0] != b.st.tokens[$0] }
            let what = noDrafter ? "normal and chained" : "normal and speculative"
            print(div.map { "  \(what) DIVERGE at token \($0)!" } ?? "  \(what): \(k) identical tokens")
            if div != nil || a.st.tokens.count != b.st.tokens.count { failures.append(name) }
        }
    }
    if args.flags.contains("profile"), let dr { try genProfile(m, dr, ctx) }
    if let gpath = args.opts["golden"] {
        let mine = Dictionary(
            results.filter { $0.mode != "normal" || noDrafter }.map { ($0.name, $0.st.tokens.map { Int($0) }) },
            uniquingKeysWith: { a, _ in a })
        failures += try goldenCheck(
            gpath, model: dir.lastPathComponent, n: n, mine: mine, logits: goldenLogits ? logitHash : nil,
            rows: logitRows)
    }
    if !check && failures.contains(where: { $0.hasPrefix("golden") }) { exit(1) }
    if check {
        if failures.isEmpty {
            print(checkSummary(prompts: prompts.count, noDrafter: noDrafter, mlxCompared: mlxCompared))
        } else {
            print("check: FAILED in \(failures.joined(separator: ", "))"); exit(1)
        }
    }
    if let path = args.opts["jsonl"] { try writeGenJSONL(path, results, m) }
}

/// --golden f.json: the greedy tokens of each prompt against the reference ones of this model on this Mac (with
/// --write-golden, it writes them). The --check tests compare lab paths that share loading and prefill; this one
/// catches what changes all of them at once (loader, prefill, kernels, prompt tokenization).
/// With `logits` (--golden-logits), also the logits fingerprints. --golden-strict (ci.sh): without the file or
/// without a prompt in it, it fails (without the flag it only warns); the same for mlx-lm's reference (`mlxCheck`).
/// Returns the failures.
@MainActor private func goldenCheck(
    _ gpath: String, model: String, n: Int, mine: [String: [Int]], logits: [String: [String]]?,
    rows: [String: [[Float]]]
) throws -> [String] {
    let strict = args.flags.contains("golden-strict")
    let url = URL(fileURLWithPath: gpath)
    var failures: [String] = []
    if args.flags.contains("write-golden") {
        var obj: [String: Any] = ["model": model, "n": n, "ids": mine]
        if let logits { obj["logits"] = logits }
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try JSONSerialization.data(withJSONObject: obj, options: [.sortedKeys]).write(to: url)
        print("golden: wrote \(mine.count) prompts to \(gpath)")
        return []
    }
    guard let d = try? Data(contentsOf: url), let o = try? JSONSerialization.jsonObject(with: d) as? [String: Any],
        let ref = o["ids"] as? [String: [Int]]
    else {
        print("golden: no reference in \(gpath) (lab-dev gen … --golden \(gpath) --write-golden)")
        return strict ? ["golden:missing"] : []
    }
    for (name, ids) in mine.sorted(by: { $0.key < $1.key }) {
        guard let r = ref[name] else {
            print("golden: \(name) is not in \(gpath)")
            if strict { failures.append("golden:" + name) }
            continue
        }
        let k = min(r.count, ids.count)
        if let div = (0..<k).first(where: { r[$0] != ids[$0] }) {
            print("golden: \(name) DIVERGES from the reference at token \(div)!")
            failures.append("golden:" + name)
        } else {
            print("golden: \(name), \(k) tokens equal to the reference")
        }
    }
    let refL = o["logits"] as? [String: [String]] ?? [:]
    for (name, h) in (logits ?? [:]).sorted(by: { $0.key < $1.key }) {
        if refL[name] == h {
            print("golden: \(name), logits bit-identical to the reference (prefill and last round)")
        } else {
            let what = refL[name].map { "fingerprints \(h) against \($0)" } ?? "no fingerprint in the reference"
            print("golden: \(name): logits DIFFER from the reference (\(what))!")
            failures.append("golden-logits:" + name)
        }
        logitStore(name, mine: h, ref: refL[name], rows: rows[name] ?? [])
    }
    return failures
}

/// --logits-store DIR (ci.sh --toy): the logits of each matching fingerprint are saved to DIR/<fingerprint>.f32 (if
/// not already there); when one does not match and the reference's logits are there, the difference is printed (max
/// |Δlogit| and NMSE). The fingerprints remain the check; this only explains it.
@MainActor private func logitStore(_ name: String, mine: [String], ref: [String]?, rows: [[Float]]) {
    guard let dir = args.opts["logits-store"], let ref, rows.count == mine.count else { return }
    try? FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
    for (i, (h, row)) in zip(mine, rows).enumerated() where i < ref.count {
        let url = URL(fileURLWithPath: dir).appendingPathComponent(ref[i] + ".f32")
        let what = i == 0 ? "prefill" : "last round"
        if h == ref[i] {
            if !FileManager.default.fileExists(atPath: url.path) {
                try? row.withUnsafeBytes { Data($0) }.write(to: url)
            }
        } else if let d = try? Data(contentsOf: url), d.count == row.count * 4 {
            let r = d.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
            print("golden:   \(name), \(what): \(logitDistanceNote(row, r)) against the reference")
        } else {
            print("golden:   \(name), \(what): no logits for fingerprint \(ref[i]) in \(dir) (no distance)")
        }
    }
}

/// The logits of the last row (vocab floats).
@MainActor private func lastLogits(_ m: HybridModel) -> [Float] {
    Array(UnsafeBufferPointer(start: m.logits.contents().bindMemory(to: Float.self, capacity: m.vocab), count: m.vocab))
}

/// Summary of the SSD expert cache after each prompt; with Flash-Next, also the misses by rank.
@MainActor private func printExpertStats(_ m: HybridModel, rounds: Int) {
    guard let es = m.expertStream else { return }
    print("  " + es.summary(rounds: rounds))
    if let fn = m.fn {
        print("  misses by best rank (1..10): \(es.stats.missRank.prefix(10))")
        // The PLE of each round on the ExpertStream thread (customNs) and the cache of its rows.
        let R = Double(max(rounds, 1)), p = fn.pleRows.stats, s = es.stats
        print(
            String(
                format:
                    "  PLE: %.3f ms per round on the ExpertStream thread (%d jobs); rows: %.1f%% hits, "
                    + "%.1f reads per round (%.3f ms)",
                Double(s.customNs) / 1e6 / R, s.customJobs, p.hitRate * 100, Double(p.reads) / R,
                Double(p.readNs) / 1e6 / R))
        fn.pleRows.stats = .init()
    }
    es.resetStats()
}

/// FNV-1a fingerprint of the logits of row 0 (bit by bit).
@MainActor func logitsFingerprint(_ m: HybridModel) -> String {
    let n = m.vocab * 4
    let p = m.logits.contents().bindMemory(to: UInt8.self, capacity: n)
    var h: UInt64 = 0xCBF2_9CE4_8422_2325
    for i in 0..<n { h = (h ^ UInt64(p[i])) &* 0x100_0000_01B3 }
    return String(format: "%016llx", h)
}

/// mlx-lm's top1 − top2 margin with fp32 activations at step `pos` of a prompt: row `pos` of <base>_logits_f32.npy
/// (logits_ref.py --f32, after forcing `pos` tokens of mlx-lm's bf16 greedy, the same as the lab's up to a first
/// divergence there), or nil without the file or the row.
private func f32Margin(_ base: String, at pos: Int) -> Double? {
    guard let a = try? NPYArray(contentsOf: URL(fileURLWithPath: base + "_logits_f32.npy")), a.kind == .f4,
        a.shape.count == 2, pos < a.shape[0]
    else { return nil }
    let V = a.shape[1]
    return a.data.withUnsafeBytes { raw in
        let p = raw.bindMemory(to: Float.self).baseAddress! + pos * V
        var t1 = -Float.infinity, t2 = -Float.infinity
        for i in 0..<V {
            if p[i] > t1 { t2 = t1; t1 = p[i] } else if p[i] > t2 { t2 = p[i] }
        }
        return Double(t1 - t2)
    }
}

/// The last line of a green --check: what was compared (the CI log has to say whether mlx-lm was).
/// gen's drafter (resolvedDrafterDir with the toy's `drafter/`, as lab serve). In normal mode it is not opened: one
/// missing from the cache is not an error.
private func genDrafterDir(_ dir: URL, _ cfg: ModelConfig, mode: String) throws -> URL? {
    if mode == "normal" { return try? resolvedDrafterDir(dir, cfg, toyPack: true) }
    return try resolvedDrafterDir(dir, cfg, toyPack: true)
}

private func checkSummary(prompts: Int, noDrafter: Bool, mlxCompared: Int) -> String {
    if !noDrafter {
        return "check: OK (\(prompts) prompts, identical with and without drafter; not compared against mlx-lm: "
            + "model with a drafter)"
    }
    return mlxCompared == prompts
        ? "check: OK (\(prompts) prompts, normal = chained, and against mlx-lm)"
        : "check: OK (\(prompts) prompts, normal = chained; against mlx-lm only \(mlxCompared): see the lines above)"
}

enum MLXCheck {
    case passed, diverged, missing, skipped
    /// Appends the failure, if any, to `failures`; 1 if the prompt passed the comparison.
    func record(_ name: String, into failures: inout [String]) -> Int {
        switch self {
        case .diverged: failures.append(name)
        case .missing: failures.append("mlx-ref:" + name)
        case .passed: return 1
        case .skipped: break
        }
        return 0
    }
}

/// Models without a drafter (--check): tokens of the normal mode against mlx-lm's greedy (gen_ref.py) in
/// ref/out/<genRefSet> (the model's prompt set prompts-<x> → gen-<x>), equal up to the first
/// divergence and mlx-lm's top1 − top2 margin there. `.diverged` if they diverge before 32 tokens with a clear margin
/// (> 0.25; on the fp32 reference when the bf16 one looks clear and gen-<x> has the fp32 logits, `f32Margin`).
/// Without the reference: `.missing` with --golden-strict (ci.sh: a comparison skipped silently looks green), a
/// visible warning and `.skipped` without it. Toys have no mlx-lm reference here (their fidelity is in ci.sh --toy).
@MainActor private func mlxCheck(_ name: String, _ tokens: [UInt32], model: URL, isToy: Bool, n: Int) -> MLXCheck {
    if isToy {
        print("  against mlx-lm: not compared (toy model; its fidelity runs in ci.sh --toy)")
        return .skipped
    }
    let gen = genRefSet
    let refDir = repoRoot.path + "/ref/out/\(gen)"
    let refPath = refDir + "/\(name)_mlx.json"
    guard let d = try? Data(contentsOf: URL(fileURLWithPath: refPath)),
        let o = try? JSONSerialization.jsonObject(with: d) as? [String: Any],
        let ref = o["ids"] as? [NSNumber]
    else {
        let what =
            FileManager.default.fileExists(atPath: refDir)
            ? "no \(name)_mlx.json in ref/out/\(gen)" : "missing folder ref/out/\(gen)"
        let strict = args.flags.contains("golden-strict")
        print(
            "  against mlx-lm: \(strict ? "FAILED" : "warning, NOT compared"): \(what) (generate it with: cd ref && "
                + ".venv/bin/python gen_ref.py --model \(model.path) --pdir \(promptSet) --out \(gen) "
                + "--prompt \(name) --n \(max(n, 256)))")
        return strict ? .missing : .skipped
    }
    let rt = ref.map { UInt32($0.intValue) },
        margins = (o["margins"] as? [NSNumber])?.map(\.doubleValue) ?? []
    let k = min(rt.count, tokens.count)
    if let div = (0..<k).first(where: { rt[$0] != tokens[$0] }) {
        let mg = div < margins.count ? margins[div] : nil
        // The bf16 margin is rounded to the logits' bf16 step (0.125 between 16 and 32: Gemma 3's scale). If it looks
        // clear, it is judged on mlx-lm with fp32 activations at that step (logits_ref.py --f32), when present.
        let f32 = (mg ?? 1) > 0.25 && div < 32 ? f32Margin(refDir + "/\(name)", at: div) : nil
        print(
            "  against mlx-lm (ref/out/\(gen)): \(div) tokens equal out of \(k); diverges at \(div) (mlx-lm margin "
                + (mg.map { String(format: "%.3f", $0) } ?? "?")
                + (f32.map { String(format: ", %.3f with fp32 activations", $0) } ?? "") + ")")
        // Diverging at a near-tie is normal (a different summation order); before 32 tokens with a clear margin, it is not.
        if div < 32 && (f32 ?? mg ?? 1) > 0.25 { return .diverged }
    } else {
        print("  against mlx-lm (ref/out/\(gen)): \(k) identical tokens")
    }
    return .passed
}

/// --jsonl path --label label: one JSON line per result (those without a drafter, with the label ending in -nodraft).
@MainActor private func writeGenJSONL(_ path: String, _ results: [GenResult], _ m: HybridModel) throws {
    let label = args.str("label", "lab-q36-dflash2-think")
    let fm = FileManager.default
    if !fm.fileExists(atPath: path) { fm.createFile(atPath: path, contents: nil) }
    let fh = try FileHandle(forWritingTo: URL(fileURLWithPath: path))
    fh.seekToEndOfFile()
    for r in results {
        let lab = r.mode == "spec" ? label : label.replacingOccurrences(of: "-dflash2", with: "") + "-nodraft"
        fh.write(
            try JSONSerialization.data(
                withJSONObject: r.json(label: lab, rowsPolicy: rowsPolicyName(m)), options: [.sortedKeys]))
        fh.write(Data("\n".utf8))
    }
    try fh.close()
}

/// The lossy prefill options of this run next to lab serve's defaults for the model (EngineOptions.serve), written in
/// each ref/out/gen/<name>_lab_<mode>.json: ref/serve_check.py fails unless they match, so that it compares with what
/// lab serve serves by default (exact gen against lossy serve diverged at 4K on the Gemma 4 26B).
/// `ssd_experts`: experts read from the SSD (Flash-Next), where lab serve also skips missing experts by default
/// (lossy and timing-dependent: serve_check runs lab serve with --expert-skip-rank 0).
@MainActor private func genLossyRecord(_ m: HybridModel, _ cfg: ModelConfig) -> [String: Any] {
    let serve = EngineOptions.serve(moe: cfg.isMoE)
    return [
        "int8": m.options.prefillInt8, "qk8": m.options.prefillQK8, "serve_int8": serve.prefillInt8,
        "serve_qk8": serve.prefillQK8, "ssd_experts": m.expertStream != nil,
    ]
}

/// Accepted tokens and rows per round of generation with a drafter (with --profile, also the GPU per round).
@MainActor private func printSpecStats(_ st: HybridModel.GenStats) {
    print(
        "  accepted per round (0..15): \(st.acceptedHist.prefix(16).map(String.init).joined(separator: " "))")
    print(
        "  rows per round (1..16): \(st.rowsHist.dropFirst().prefix(16).map(String.init).joined(separator: " "))"
            + "; " + st.rowsSummary
    )
    if st.copyRounds > 0 || st.deferred > 0 { print("  " + st.copySummary) }
    if !st.wallMs.isEmpty { print("  " + st.roundSummary) }
    if args.flags.contains("profile") {
        let g = st.gpuMs.sorted()
        print(
            "  GPU per round: min \(g.first!) p10 \(g[g.count / 10]) p50 \(g[g.count / 2]) p90 \(g[g.count * 9 / 10]); "
                + "first: \(st.gpuMs.prefix(12).map { String(format: "%.1f", $0) })"
        )
    }
}
