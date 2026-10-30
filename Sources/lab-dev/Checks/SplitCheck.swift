import Foundation
import LabCLI
import MetalLab

/// lab-dev splitcheck [--prompt 4K] [--cuts 64,1024,2112,3008] [--n 3]: prefill split at cuts aligned to 64
/// (several calls with `continuing`, or `split` in a single one) has to give bit-identical logits to the whole
/// prompt, in the last row and in `--n` decode steps after it. It is the property the prefix cache and the snapshots
/// rely on (the lab rule: "prefill has to give bit-identical results split into chunks aligned to 64 and whole").
/// Exits 1 if any cut gives something else.
///
/// The one exception (docs/adr/0002): a family whose prefill has two algebraically different paths chosen by the
/// chunk's rows (today GLM's latent MLA: absorbed, or decompressed for chunks of ≥ 1,024 rows over ≥ 8K keys,
/// MLAProjection.decompressed). There the prompt is repeated up to `latentPromptTokens`, the default cuts add one that
/// stays on the decompressed path (it has to be bit-identical) and one that sends the tail through the absorbed path,
/// and a case that is not bit-identical passes if every step keeps the top-1 and KL(whole ‖ split) ≤ `latentMaxKL`.
@MainActor func cmdSplitCheck() throws {
    let model = try args.required("model", modelHelp)
    let (dir, cfg, ctx) = try openModel(model)
    var ids = try promptIds(args.str("prompt", "4K"))
    let opts = try engineOptions(cfg)
    // With the W8A8 prefill (--int8 1) the paths are not crossed: a row's int8 roundings amplify the paths'
    // difference through a toy's chaotic router (KL ~0.1 at single steps, top-1 flips), so that run keeps the
    // bit-exact check of the 4K prompt, which no long chunk reaches.
    let latent = try cfg.requiredSpec.mla != nil && !opts.prefillInt8
    if latent && ids.count < latentPromptTokens {
        let base = ids
        while ids.count < latentPromptTokens { ids += base }
        ids = Array(ids.prefix(latentPromptTokens))
    }
    let m = try HybridModel(
        ctx: ctx, store: try SafetensorsStore(directory: dir), config: cfg, maxCtx: ids.count + 256,
        options: opts)
    let dr = try openModelDrafter(ctx, dir, cfg, m)
    let V = m.vocab, steps: [UInt32] = [198, 279, 3681].prefix(args.int("n", 3)).map { $0 }
    func logits() -> [Float] {
        Array(UnsafeBufferPointer(start: m.logits.contents().bindMemory(to: Float.self, capacity: V), count: V))
    }
    /// Prefill of `ids` split at `cuts` (or in a single call with `split`) and `steps` decode steps.
    func run(_ cuts: [Int], split: Int? = nil) throws -> [[Float]] {
        m.reset()
        var a = 0
        for (k, c) in (cuts + [ids.count]).enumerated() {
            if k == 1 { perturbForTest(m, "between the first chunk and the second") }
            _ = try m.prefill(
                Array(ids[a..<c]), drafter: dr, continuing: k > 0, head: c == ids.count, split: split)
            a = c
        }
        var out = [logits()]
        for t in steps { _ = try m.stepLogits(t); out.append(logits()) }
        return out
    }
    let cuts = args.str("cuts", latent ? "64,1024,2112,3008,9216,9984" : "64,1024,2112,3008").split(separator: ",")
        .compactMap { Int($0) }
        .filter { $0 > 0 && $0 < ids.count }
    // The 64 alignment comes from the chunked GDN; a model with attention layers only (toy-gemma4: cuts at
    // W ± 1) takes any cut.
    let attnOnly = m.layers.allSatisfy {
        if case .attn = $0.mixer { return true }; return false
    }
    guard attnOnly || cuts.allSatisfy({ $0 % 64 == 0 }) else { throw UsageError("--cuts: multiples of 64") }
    print("prompt of \(ids.count) tokens, chunks of ≤ \(m.chunkRows) rows; cuts \(cuts)")
    _ = try run([])  // warm-up (pipelines)
    let whole = try run([])
    var cases: [(String, [Int], Int?)] = cuts.map { ("cut at \($0)", [$0], nil) }
    if cuts.count >= 2 { cases.append(("cuts at \(cuts[0]) and \(cuts[cuts.count - 1])", [cuts[0], cuts.last!], nil)) }
    if let s = cuts.last(where: { ids.count - $0 <= m.chunkRows }) {
        cases.append(("split \(s) in a single call", [], s))
    }
    var bad = 0
    for (name, cs, split) in cases {
        let got = try run(cs, split: split)
        let diffs = zip(got, whole).map { a, b in zip(a, b).map { abs($0 - $1) }.max()! }
        let same = got == whole
        // The latent MLA's two paths: within tolerance (top-1 kept, KL bounded) at every step.
        let tol = latent && !same ? zip(whole, got).map { latentStepDistance($0, $1) } : []
        let within = !tol.isEmpty && tol.allSatisfy { $0.top1 && $0.kl <= latentMaxKL }
        if !same && !within { bad += 1 }
        // If not bit-identical, the distance (NMSE and max |Δlogit|) of the first differing step.
        let note =
            zip(got, whole).first { $0 != $1 }.map { "; first differing step: " + logitDistanceNote($0, $1) } ?? ""
        let tnote =
            tol.isEmpty
            ? ""
            : "; KL per step "
                + tol.map { String(format: "%.2e%@", $0.kl, $0.top1 ? "" : " (top-1 DIFFERENT)") }
                .joined(separator: " ") + String(format: ", bound %.0e", latentMaxKL)
        print(
            "  \(name): \(same ? "bit-identical" : within ? "within tolerance (other path)" : "DIFFERENT") (max |Δ| per step "
                + diffs.map { String(format: "%.3g", $0) }.joined(separator: " ") + note + tnote + ")")
    }
    print(bad == 0 ? "splitcheck: OK" : "splitcheck: FAILED (\(bad) of \(cases.count))")
    if bad > 0 { exit(1) }
}

/// Latent MLA (GLM): prompt length of splitcheck, so that its last chunks take the decompressed prefill (8,192 + 2,048).
let latentPromptTokens = 10_240
/// Latent MLA: largest KL(whole ‖ split) in nats per step when a cut sends rows through the other prefill path.
let latentMaxKL = 1e-4

/// Top-1 equal and KL(a ‖ b) in nats of two logit rows.
func latentStepDistance(_ x: [Float], _ y: [Float]) -> (top1: Bool, kl: Double) {
    let lx = logSumExp(x), ly = logSumExp(y)
    var kl = 0.0
    for j in 0..<x.count {
        let a = Double(x[j]) - lx
        if a > -30 { kl += exp(a) * (a - (Double(y[j]) - ly)) }
    }
    return (argmaxIndex(x) == argmaxIndex(y), kl)
}
