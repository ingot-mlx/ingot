import Foundation
import LabCLI
import LabServe
import MetalLab

/// lab-dev shapes [--models toy-qwen36-moe,toy-lfm2,... (default: all of ToyArch)] [--dir ~/.cache/metal-lab/toy]
/// [--prompts 512]: several models of different shapes in the same process (the shape goes through MetalContext,
/// no global). Loads them all (each with its context, library and drafter) and then, in load order and in reverse,
/// generates with each one like `gen --nostop --golden-logits` (same warmup, greedy with drafter or, without it,
/// normal decode) and compares tokens and logits fingerprints with ref/golden/<model>.json. If one model's shape
/// overwrote another's, their tokens or fingerprints would change. Exits with 1 on any mismatch.
@MainActor func cmdShapes() throws {
    let names = args.str("models", ToyArch.allCases.map(\.rawValue).joined(separator: ","))
        .split(separator: ",").map(String.init)
    let base = args.str("dir", NSHomeDirectory() + "/.cache/metal-lab/toy")
    let promptNames = args.str("prompts", "512").split(separator: ",").map(String.init)
    struct Loaded {
        let name: String
        let m: HybridModel
        let dr: DFlash2Drafter?
        let marker: [UInt32]
        let n: Int
        let ids: [String: [Int]], logits: [String: [String]]
    }
    // The contexts are created in parallel: each one compiles its shape's library (~1.5 s), and libraries of
    // different shapes compile at the same time (MetalContext locks per shape). Models are built afterwards, in order.
    let files = try loadModelFilesInParallel(names.map { $0.hasPrefix("/") ? $0 : base + "/" + $0 })
    var loaded: [Loaded] = []
    for (name, (dir, cfg, ctx)) in zip(names, files) {
        let gpath = repoRoot.path + "/ref/golden/\(name.split(separator: "/").last!).json"
        guard let o = try? jsonObject(at: URL(fileURLWithPath: gpath)), let ids = o["ids"] as? [String: [Int]],
            let n = (o["n"] as? NSNumber)?.intValue
        else { throw UsageError("missing or unreadable \(gpath)") }
        let prompts = try promptNames.map { try promptIds(repoRoot.path + "/ref/out/prompts-toy/\($0).json") }
        let maxCtx = (prompts.map(\.count).max() ?? 0) + n + 64
        // The toy's drafter as gen takes it (toy-gptoss's too, whose family has none by default).
        let drDir = try resolvedDrafterDir(dir, cfg, toyPack: true), hasDrafter = drDir != nil
        let plan = try memoryPlan(
            dir: dir, cfg: cfg, device: ctx.device, options: genMemOptions(maxCtx: maxCtx, drafter: hasDrafter))
        let m = try HybridModel(
            ctx: ctx, store: try SafetensorsStore(directory: dir), config: cfg, maxCtx: maxCtx,
            chunkRows: plan.chunkRows,
            options: try {
                var e = try engineOptions(cfg)
                applyPlan(&e, plan)
                e.bench.copyPolicyUntimed = true  // like gen --golden-logits
                return e
            }())
        let dr = try drDir.map { try openDrafter(ctx, $0, m) }
        let marker = (try? BPETokenizer(directory: dir))?.encode("<|im_start|>assistant\n").map { UInt32($0) } ?? []
        print(
            "loaded \(name): hidden \(ctx.dims.hidden), head \(ctx.dims.attnHD), \(ctx.dims.attnQH) q × "
                + "\(ctx.dims.attnKVH) KV, GDN \(ctx.dims.gdnHV) v\(dr == nil ? ", no drafter" : "")")
        loaded.append(
            Loaded(
                name: name, m: m, dr: dr, marker: marker, n: n, ids: ids,
                logits: o["logits"] as? [String: [String]] ?? [:]))
    }
    let distinct = Set(loaded.map { $0.m.ctx.dims }).count
    print("\(loaded.count) models in the process, \(distinct) distinct shapes")
    var failures: [String] = []
    var warmed = Set<String>()
    for (pass, order) in [loaded, loaded.reversed()].enumerated() {
        for l in order {
            let m = l.m
            func start(_ ids: [UInt32]) -> Int? { InferenceEngine.generationStart(ids, marker: l.marker) }
            if !warmed.contains(l.name) {
                // gen's warmup (the last round's fingerprints depend on what ran before).
                let warm = try promptIds(repoRoot.path + "/ref/out/prompts-toy/chat.json")
                _ = try m.generate(prompt: warm, maxTokens: 4, genStart: start(warm))
                if let dr = l.dr {
                    _ = try m.generateSpec(prompt: warm, drafter: dr, maxTokens: 12, genStart: start(warm))
                }
                _ = try m.generatePipelined(
                    prompt: warm, drafter: l.dr, maxTokens: l.dr == nil ? 8 : 12, genStart: start(warm))
                warmed.insert(l.name)
            }
            for p in promptNames {
                let ids = try promptIds(repoRoot.path + "/ref/out/prompts-toy/\(p).json")
                _ = try m.prefill(ids, drafter: nil)
                var hashes = [logitsFingerprint(m)]
                // Like gen --check: normal decode and then with drafter (or chained without one); the golden is the
                // drafter path's (without drafter, the normal one).
                var st = try m.generate(prompt: ids, maxTokens: l.n, stopAtEOS: false, genStart: start(ids))
                let spec = try m.generatePipelined(
                    prompt: ids, drafter: l.dr, maxTokens: l.n, stopAtEOS: false, genStart: start(ids))
                if l.dr != nil {
                    st = spec
                    hashes.append(logitsFingerprint(m))
                }
                let mine = st.tokens.map { Int($0) }
                // On the second pass the copy policy has seen more text: the last round's fingerprint may change
                // without error (it depends on the drafts); the tokens and the prefill's do not.
                let ref = l.logits[p] ?? []
                let tokOK = l.ids[p] == mine
                let logOK = pass == 0 ? ref == hashes : ref.first == hashes.first
                let tag = "pass \(pass + 1), \(l.name) \(p)"
                print(
                    "\(tag): \(tokOK ? "\(mine.count) tokens match the golden" : "tokens DIFFER from the golden!"); "
                        + (logOK
                            ? (pass == 0 ? "fingerprints match" : "prefill fingerprint matches")
                            : "fingerprints \(hashes) vs \(ref)!"))
                if !tokOK || !logOK { failures.append(tag) }
            }
        }
    }
    if failures.isEmpty {
        print("shapes: OK (\(loaded.count) models, \(distinct) shapes, in one process)")
    } else {
        print("shapes: FAILED in \(failures.joined(separator: "; "))")
        exit(1)
    }
}

/// `loadModelFiles` for each path at the same time; the results (or the first error, in path order) in path order.
private func loadModelFilesInParallel(_ paths: [String]) throws -> [(dir: URL, cfg: ModelConfig, ctx: MetalContext)] {
    final class Results: @unchecked Sendable {
        let lock = NSLock()
        var items: [Result<(dir: URL, cfg: ModelConfig, ctx: MetalContext), Error>?]
        init(_ n: Int) { items = Array(repeating: nil, count: n) }
    }
    let results = Results(paths.count)
    DispatchQueue.concurrentPerform(iterations: paths.count) { i in
        let r = Result { try loadModelFiles(paths[i]) }
        results.lock.withLock { results.items[i] = r }
    }
    return try results.items.map { try $0!.get() }
}
