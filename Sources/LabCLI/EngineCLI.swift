import Foundation
import MetalLab

// Engine options (EngineOptions) from the command line, and shared helpers to create the model and the drafter.

/// Engine options from the command line (EngineOptions; the same in serve and in the benchmarks):
///   --copy auto|on|off           copy-from-context speculation (auto: depends on the architecture)
///   --int8 0|1, --qk8 0|1        lossy prefill (W8A8; Q·Kᵀ in int8); in serve, 1 by default on MoE models
///   --tree N                     draft tree of N rows on dense models (12 by default; 0 = the chain)
///   --spec-rows N                (benchmarks) fixed rows per DFlash2 round
///   --prep-trace                 (benchmarks) ms of each prompt-preparation step
///   --expert-cache-gb X          X GB of experts in memory, the rest read from the SSD (Flash-Next: 8)
///   --expert-prefetch K          read-ahead of each row's top-K (4; 6 on Flash-Next; 0 = off)
///   --expert-pf-group N          prefill chunks per group, layer by layer (4 by default, 8 on Flash-Next; 1 = chunk by chunk)
///   --expert-pf-bulk N           group rows from which prefill fetches everything missing up front (256)
///   --expert-fn-bulk N           same on Flash-Next (below it, a chunk reads only what it routes; 900)
///   --ple-cache-mb N             (Flash-Next) CPU cache of PLE table rows (512)
///   --expert-wide K              (Flash-Next) wide read-ahead: top-K of each row into step buffers that
///                                evict nothing from the cache; what gets used enters it (10; 0 = that of --expert-prefetch)
///   --expert-skip-rank N         (Flash-Next; lossy) do not wait for missing experts whose best rank among
///                                the rows is N + 1 or worse (5 = ranks 6–10, the default in `lab serve`; 0 = off);
///                                the skipped ones are read in the background for later rounds
///   --exp key=value              (experiments, repeatable) EngineOptions.exp; none is read on main
package func engineOptions(_ cfg: ModelConfig, serve: Bool = false) throws -> EngineOptions {
    // Flash-Next too: W8A8 in the dense GEMMs of the GDN and attention, and Q·Kᵀ in int8.
    var o = serve ? EngineOptions.serve(moe: cfg.isMoE) : EngineOptions()
    if let v = args.opts["copy"] {
        guard let c = EngineOptions.CopyMode(rawValue: v) else { throw UsageError("--copy auto|on|off (not \(v))") }
        o.copy = c
    }
    if let v = args.opts["int8"] { o.prefillInt8 = (Int(v) ?? 0) > 0 }
    if let v = args.opts["qk8"] { o.prefillQK8 = (Int(v) ?? 0) > 0 }
    if let v = args.opts["tree"] {
        guard let k = Int(v), (0...16).contains(k) else { throw UsageError("--tree 0…16 (not \(v))") }
        o.treeNodes = k
    }
    if let v = args.opts["spec-rows"] {
        guard let k = Int(v), (1...16).contains(k) else { throw UsageError("--spec-rows 1…16 (not \(v))") }
        o.bench.specRowsFixed = k
    }
    o.exp = args.exp
    o.bench.prepTrace = args.flags.contains("prep-trace")
    try expertCacheOptions(&o)
    try expertFlashOptions(&o)
    if let v = args.opts["ple-cache-mb"] {
        guard let k = Int(v), k >= 0 else { throw UsageError("--ple-cache-mb N ≥ 0 (not \(v))") }
        o.pleCacheBytes = k * 1_000_000
    }
    return o
}

/// `--expert-cache-gb` and what tunes the SSD expert cache and the prefill that uses it.
private func expertCacheOptions(_ o: inout EngineOptions) throws {
    if let v = args.opts["expert-cache-gb"] {
        guard let g = Double(v), g > 0 else { throw UsageError("--expert-cache-gb X (GB > 0, not \(v))") }
        o.expertCacheBytes = Int(g * 1e9)
    }
    if let v = args.opts["expert-prefetch"] {
        guard let k = Int(v), (0...16).contains(k) else { throw UsageError("--expert-prefetch 0…16 (not \(v))") }
        o.expertPrefetch = k
    }
    if let v = args.opts["expert-pf-group"] {
        guard let k = Int(v), (1...16).contains(k) else { throw UsageError("--expert-pf-group 1…16 (not \(v))") }
        o.expertPrefillGroup = k
        o.flashPrefillGroup = k
    }
    if let v = args.opts["expert-pf-bulk"] {
        guard let k = Int(v), k >= 0 else { throw UsageError("--expert-pf-bulk N ≥ 0 (not \(v))") }
        o.expertBulkMinRows = k
    }
}

/// Flash-Next: wide read-ahead, expert skipping, and `--expert-fn-bulk`.
private func expertFlashOptions(_ o: inout EngineOptions) throws {
    if let v = args.opts["expert-wide"] {
        guard let k = Int(v), (0...16).contains(k) else { throw UsageError("--expert-wide 0…16 (not \(v))") }
        o.expertWide = k
    }
    if let v = args.opts["expert-skip-rank"] {
        guard let k = Int(v), (0...15).contains(k) else { throw UsageError("--expert-skip-rank 0…15 (not \(v))") }
        o.expertSkipRank = k == 0 ? nil : k
    }
    if let v = args.opts["expert-fn-bulk"] {
        guard let k = Int(v), k >= 0 else { throw UsageError("--expert-fn-bulk N ≥ 0 (not \(v))") }
        o.flashBulkMinRows = k
    }
}

/// What differs from the exact path, printed at load time.
package func describeLossy(_ m: HybridModel) {
    let o = m.options
    if o.prefillInt8 {
        print(
            "prefill in rotated W8A8 (lossy; "
                + "from position \(m.int8MinPos); "
                + (m.fn != nil
                    ? "projections of the GDN and attention"
                        + (m.int8Experts ? " and Q2 experts with int8 activations" : "")
                        + "; HC and PLE exact"
                    : m.isMoE ? "dense GEMMs; experts exact" : "all dense GEMMs")
                + "; --int8 0 turns it off)")
    }
    if o.prefillQK8 {
        print("prefill attention with Q·Kᵀ in int8 (from position \(o.int8.qk8MinCtx); --qk8 0 turns it off)")
    }
    if let r = m.expertStream?.skipRank {
        print(
            "expert skipping (lossy): rounds do not wait for missing experts with best rank ≥ \(r + 1)"
                + " (without renormalizing, read in background)")
    }
}

/// The drafter in `directory` (resolvedDrafterDir).
package func openDrafter(_ ctx: MetalContext, _ directory: URL, _ m: HybridModel) throws -> DFlash2Drafter {
    try DFlash2Drafter(
        ctx: ctx, directory: directory, vocab: m.vocab, embedScale: m.spec.block.embedScale, lmHead: m.lmHead)
}

/// The model's drafter (resolvedDrafterDir with the toy's `drafter/`) opened, or nil if it runs without one.
package func openModelDrafter(_ ctx: MetalContext, _ modelDir: URL, _ cfg: ModelConfig, _ m: HybridModel) throws
    -> DFlash2Drafter?
{
    try resolvedDrafterDir(modelDir, cfg, toyPack: true).map { try openDrafter(ctx, $0, m) }
}

/// openModelDrafter for the commands that need one (rowscheck, rowscal): an error if the model runs without drafter.
package func requiredModelDrafter(_ ctx: MetalContext, _ modelDir: URL, _ cfg: ModelConfig, _ m: HybridModel) throws
    -> DFlash2Drafter
{
    guard let dr = try openModelDrafter(ctx, modelDir, cfg, m) else {
        throw UsageError("this command needs a drafter and the model runs without one")
    }
    return dr
}

/// Row policy of the DFlash2 round, for the jsonl files: fixedN, conf (family calibration) or 8.
package func rowsPolicyName(_ m: HybridModel) -> String {
    m.options.bench.specRowsFixed.map { "fixed\($0)" } ?? (m.rowsConfig != nil ? "conf" : "fixed8")
}
