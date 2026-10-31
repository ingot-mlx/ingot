import CryptoKit
import Foundation
import Ingot
import IngotCLI
import IngotServe
import Metal

/// ingot serve --model org/repo|dir [--port 8090] [--host 127.0.0.1] [--name id] [--max-ctx N] [--snapshots 3]
///           [--prefix-cache-gb 1] [--prefix-disk 0] [--chunk 2048] [--drafter none] [--mem-limit-gb N]
///           [--int8 0|1] [--qk8 0|1] [--copy auto|on|off]
///           [--dump-requests DIR] [--route-trace FILE] [--expert-preload 0|1] [--agent-guards on|off]
/// Without --max-ctx, --snapshots, --prefix-cache-gb, --chunk or --drafter, the memory plan decides them; the engine
/// options (engineOptions) depend on the architecture. --agent-guards on (off by default, so the server stays neutral)
/// stops a response that has fallen into a token loop with finish_reason "length" (IngotServe/AgentGuard.swift): for
/// agent sessions, where the client asks for it when it starts the server.
/// One model per process with its DFlash2 drafter (LFM2.5, no drafter); OpenAI-compatible API (IngotServe).
@MainActor func cmdServe() throws {
    let model = try args.required("model", modelHelp)
    let port = UInt16(args.int("port", 8090))
    let host = args.str("host", "127.0.0.1")
    let (dir, cfg, ctx) = try loadModelFiles(model)
    // First line of the log: the version, then the memory plan (with the context) and the rest of the loading.
    print("ingot \(IngotVersion.current): loading \(model)...")
    // Context, drafter, snapshots, prefix cache and prefill chunk: whatever the user sets and the rest according to
    // the Mac's memory (MemPlan.swift; --mem-limit-gb simulates another Mac). If not even the minimum fits, it is not loaded.
    let plan = try memoryPlan(dir: dir, cfg: cfg, device: ctx.device, options: memOptions(), loading: true)
    let useDrafter = plan.drafter
    let maxCtx = plan.context
    let t0 = Date()
    let info = try ChatModelInfo(directory: dir)
    if let w = info.dialect.warning { print("warning: \(w)") }
    let m = try HybridModel(
        ctx: ctx, store: try SafetensorsStore(directory: dir), config: cfg, maxCtx: maxCtx, chunkRows: plan.chunkRows,
        options: try {
            var o = try engineOptions(cfg, serve: true); applyPlan(&o, plan); return o
        }())
    // For MoE models, --int8 1 and --qk8 1 by default (EngineOptions.serve): W8A8 from the
    // family's position (ModelSpec.int8MinPos: 512 on Qwen3.6, 1024 on Flash-Next) and Q·Kᵀ in int8 from 8K. The
    // dense models (27B, Bonsai) stay exact: unmeasured. --int8 0 --qk8 0 returns to the exact path.
    describeLossy(m)
    if let p = m.kvPool { print(p.describe()) }
    // fnc: preload of the expert cache with the keys the previous session left, in the background (the
    // warm-up and requests go first; --expert-preload 0 removes it). ExpertPreload.swift.
    let preloadURL = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(
        ".cache/ingot/experts/" + dir.lastPathComponent + ".keys")
    let preloadOn = args.int("expert-preload", 1) != 0
    if preloadOn, let es = m.expertStream {
        let keys = es.loadPreloadKeys(from: preloadURL, model: dir.lastPathComponent)
        if !keys.isEmpty {
            es.preloadLog = { @Sendable s in print(s) }
            es.preload(keys)
            print(
                String(
                    format: "experts: preloading %d from the previous session in the background (%.2f GB)", keys.count,
                    Double(keys.count * es.expertBytes) / 1e9))
        }
    }
    let drafterDir = useDrafter ? try resolvedDrafterDir(dir, cfg, toyPack: true) : nil
    let dr = try drafterDir.map { try openDrafter(ctx, $0, m) }
    let eng = InferenceEngine(
        model: m, drafter: dr, maxSnapshots: plan.snapshots, prefixCacheBytes: plan.prefixCacheBytes)
    // Prefix cache points: the start of the assistant's and the user's turns in the model's template.
    eng.generationMarker = info.dialect.generationMarker.map { info.tokenizer.encode($0).map { UInt32($0) } } ?? []
    eng.userMarker = info.dialect.userMarker.map { info.tokenizer.encode($0).map { UInt32($0) } } ?? []
    let server = ChatServer(engine: eng, info: info, modelId: args.str("name", model))
    switch args.str("agent-guards", "off") {
    case "on": server.agentGuards = true
    case "off": server.agentGuards = false
    case let v: throw UsageError("--agent-guards must be on or off, not \(v)")
    }
    if let d = args.opts["dump-requests"] {
        // Saves the body and prompt of every request: they are whole conversations (for debugging only).
        let url = URL(fileURLWithPath: (d as NSString).expandingTildeInPath)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        server.dumpDirectory = url
        print("warning: --dump-requests saves every whole request (conversation included) in \(url.path)")
    }
    server.log = { @Sendable s in
        let f = DateFormatter()
        f.dateFormat = "HH:mm:ss"
        print("\(f.string(from: Date())) \(s)")
    }
    eng.log = { @Sendable s in server.log("  " + s) }
    // Warm-up: compiles the pipelines (greedy, sampling and penalties) and leaves the engine clean.
    let warm = try info.prompt(messages: [.object([("role", .string("user")), ("content", .string("Hola"))])]).ids.map {
        UInt32($0)
    }
    _ = try eng.generate(prompt: warm, maxTokens: 12, sampling: .init(), stopTokens: [])
    _ = try eng.generate(
        prompt: warm, maxTokens: 12, sampling: .init(temperature: 0.7, topP: 0.9, topK: 20, seed: 1), stopTokens: [])
    _ = try eng.generate(
        prompt: warm, maxTokens: 12,
        sampling: .init(
            temperature: 0.7, topP: 0.9, topK: 20, seed: 1, minP: 0.05, repetitionPenalty: 1.1,
            presencePenalty: 0.5, frequencyPenalty: 0.5), stopTokens: [])
    // SSD expert streaming: routing trace of every decode round (diagnostic; rounds stop being chained).
    if let p = args.opts["route-trace"] {
        m.routeTrace = try RouteTrace(path: (p as NSString).expandingTildeInPath, model: m)
        print("warning: --route-trace writes the routing of every round to \(p) (slower: rounds not chained)")
    }
    eng.clear()
    checkPlanAfterLoad(plan, device: ctx.device)
    try setUpBatching(eng, dir: dir, cfg: cfg, plan: plan, drafterDir: drafterDir)
    // Prefix cache on disk: loaded at startup and saved at shutdown (--prefix-disk 0 turns it off). The fingerprint
    // covers the weights, the drafter, the binary and --int8: if anything changes, the file is ignored.
    let diskURL = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(
        ".cache/ingot/prefix/" + dir.lastPathComponent + ".bin")
    let useDisk = args.int("prefix-disk", 1) != 0 && eng.prefixCacheBytes > 0
    let o = m.options
    let fp = fingerprint(
        dirs: [dir] + [drafterDir].compactMap { $0 },
        extra:
            "int8=\(o.prefillInt8 ? 1 : 0)\(o.prefillInt8 ? " i8pos=\(m.int8MinPos) qk8=\(o.prefillQK8)" : "") snapshots=\(eng.maxSnapshots)"
    )
    if useDisk {
        let t = Date()
        let k = (try? eng.loadPrefixCache(from: diskURL, fingerprint: fp)) ?? 0
        if k > 0 {
            print(
                String(
                    format: "prefix cache: %d entries loaded from disk in %.2f s (%@)", k,
                    Date().timeIntervalSince(t), eng.prefixCacheSummary))
        }
    }
    let gb = { (b: Int) in String(format: "%.2f GB", Double(b) / 1e9) }
    print(
        "model \(gb(m.weightBytes))\(dr.map { " + drafter " + gb($0.weightBytes) } ?? " (no drafter)"); "
            + "snapshots of \(gb(eng.snapshotBytes)) (up to \(eng.maxSnapshots)); "
            + "prefix cache up to \(gb(eng.prefixCacheBytes)); "
            + "Metal memory \(gb(ctx.device.currentAllocatedSize)) of \(gb(Int(ctx.device.recommendedMaxWorkingSetSize))) recommended"
    )
    print(
        "end tokens \(info.eosIds.sorted()); "
            + "default sampling T=\(info.defaultTemperature ?? 1) top_p=\(info.defaultTopP ?? 1) top_k=\(info.defaultTopK ?? 0)"
            + " min_p=\(info.defaultMinP ?? 0) repetition_penalty=\(info.defaultRepetitionPenalty ?? 1)"
            + " presence_penalty=\(info.defaultPresencePenalty ?? 0) frequency_penalty=\(info.defaultFrequencyPenalty ?? 0)"
    )
    let http = try HTTPServer(host: host, port: port) { @Sendable req, conn in server.handle(req, conn) }
    http.start { @Sendable err in
        if let err {
            FileHandle.standardError.write(Data("cannot listen on \(host):\(port): \(err)\n".utf8))
            exit(1)
        }
        print(
            String(
                format: "listening on http://%@:%d/v1 (ready in %.1f s)", host, Int(port), Date().timeIntervalSince(t0)
            ))
    }
    // SIGHUP too: closing the terminal window must save the prefix cache like Ctrl-C does.
    for s in [SIGINT, SIGTERM, SIGHUP] {
        signal(s, SIG_IGN)
        let src = DispatchSource.makeSignalSource(signal: s, queue: .main)
        src.setEventHandler {
            if useDisk {
                let t = Date()
                do {
                    let k = try server.withEngine { try eng.savePrefixCache(to: diskURL, fingerprint: fp) }
                    if k > 0 {
                        print(
                            String(
                                format: "prefix cache: %d entries saved to %@ (%.2f s)", k, diskURL.path,
                                Date().timeIntervalSince(t)))
                    }
                } catch { print("prefix cache: could not save it: \(error)") }
            }
            if preloadOn, let es = m.expertStream {
                do {
                    let k = try server.withEngine {
                        try es.savePreloadKeys(to: preloadURL, model: dir.lastPathComponent)
                    }
                    if k > 0 { print("experts: \(k) keys saved to preload at startup") }
                } catch { print("experts: could not save the keys: \(error)") }
            }
            if let path = args.opts["pipelines"] { try? recordPipelines(path) }
            print("stopped"); exit(0)
        }
        src.resume()
        signals.append(src)
    }
    dispatchMain()
}

/// Batching: models without a drafter that the joint round covers serve up to 4 conversations per round, as
/// many as the memory plan leaves room for. The models with a drafter or MoE layers keep as many slots but
/// serve them one at a time (each conversation's next turn resumes from its live state). The others, one
/// conversation at a time.
@MainActor func setUpBatching(
    _ eng: InferenceEngine, dir: URL, cfg: ModelConfig, plan: MemoryPlan, drafterDir: URL?
) throws {
    if let why = eng.batchUnsupported {
        print("one conversation at a time (\(why))")
        return
    }
    let (n, pos, snaps, bytes) = try batchPlan(dir: dir, cfg: cfg, plan: plan, drafterDir: drafterDir)
    eng.enableBatching(slots: n, positions: pos, snapshots: snaps)
    let serial = eng.drafter != nil ? "drafter" : eng.model.batchUnsupported
    let size =
        "extra slots of \(pos) positions and \(snaps) snapshots "
        + String(format: "(%.2f GB, allocated at the first ", Double(bytes) / 1e9)
    print(
        n == 1
            ? "one conversation at a time (no room for more slots)"
            : serial.map { "slots: up to \(n) conversations, one at a time (\($0)); " + size + "switch of slot)" }
                ?? "batching: up to \(n) conversations per round; " + size + "concurrent request)")
}

nonisolated(unsafe) var signals: [DispatchSourceSignal] = []

/// Model fingerprint for the on-disk prefix cache: config.json, name, size and date of each .safetensors in
/// `dirs`, the ingot binary and `extra`.
func fingerprint(dirs: [URL], extra: String) -> String {
    let fm = FileManager.default
    var h = SHA256()
    for d in dirs {
        if let c = try? Data(contentsOf: d.appendingPathComponent("config.json")) { h.update(data: c) }
        let files = ((try? fm.contentsOfDirectory(at: d, includingPropertiesForKeys: nil)) ?? []).filter {
            $0.pathExtension == "safetensors"
        }
        for f in files.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
            let a = try? fm.attributesOfItem(atPath: f.resolvingSymlinksInPath().path)
            h.update(
                data: Data(
                    "\(f.lastPathComponent) \(a?[.size] ?? 0) \((a?[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0)"
                        .utf8))
        }
    }
    let exe = URL(fileURLWithPath: Bundle.main.executablePath ?? CommandLine.arguments[0]).resolvingSymlinksInPath()
        .path
    let a = try? fm.attributesOfItem(atPath: exe)
    h.update(
        data: Data(
            "\(exe) \(a?[.size] ?? 0) \((a?[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0) \(extra)".utf8))
    return h.finalize().map { String(format: "%02x", $0) }.joined()
}
