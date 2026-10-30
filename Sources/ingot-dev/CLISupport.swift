import Foundation
import Ingot
import IngotCLI
import IngotServe
import Metal

// Shared pieces of the ingot-dev model commands: tokenized prompts from ref/out, model opening and generation
// results.

/// Subdirectory of ref/out with the tokenized prompts: "prompts" (Qwen3.6 template) or "prompts-q27" (the 27B one,
/// which adds a system message; `prompts.py --model <model-dir> --out prompts-q27`).
nonisolated(unsafe) var promptSet = "prompts"
/// Ids of the model's `<|im_start|>assistant\n`: point (a) of prompt preparation (same as ingot serve).
nonisolated(unsafe) var genMarker: [UInt32] = []
func genStart(_ ids: [UInt32]) -> Int? { InferenceEngine.generationStart(ids, marker: genMarker) }

func promptIds(_ name: String) throws -> [UInt32] {
    let path = name.hasSuffix(".json") ? name : repoRoot.path + "/ref/out/\(promptSet)/\(name).json"
    return try jsonUInt32s(at: URL(fileURLWithPath: path), key: "ids")
}

/// JSON object from a file (usage error if it is not an object).
func jsonObject(at url: URL) throws -> [String: Any] {
    guard let o = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any] else {
        throw UsageError("\(url.path): expected a JSON object")
    }
    return o
}

/// List of integers under key `key` of a JSON object, as token ids.
func jsonUInt32s(at url: URL, key: String) throws -> [UInt32] {
    guard let a = try jsonObject(at: url)[key] as? [NSNumber] else {
        throw UsageError("\(url.path): missing list `\(key)`")
    }
    return a.map { UInt32($0.intValue) }
}

/// ref/out/gen/<name>_ingot_<mode>.json: the tokens of a `ingot-dev gen` run (read by ref/serve_check.py). Each worktree
/// has its own ref/out/gen (never shared between checkouts); mlx-lm's references live in ref/out/gen-<x>.
func writeGen(_ name: String, _ mode: String, _ st: HybridModel.GenStats, options: [String: Any] = [:]) throws {
    let dir = repoRoot.path + "/ref/out/gen"
    try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
    var obj: [String: Any] = ["prompt": name, "mode": mode, "ids": st.tokens.map { Int($0) }]
    if !options.isEmpty { obj["options"] = options }
    try JSONSerialization.data(withJSONObject: obj).write(to: URL(fileURLWithPath: dir + "/\(name)_ingot_\(mode).json"))
}

/// Directory, config and a MetalContext with the model's dimensions; sets `promptSet` (--pdir or by model) and
/// `genMarker`.
func openModel(_ name: String) throws -> (dir: URL, cfg: ModelConfig, ctx: MetalContext) {
    let (dir, cfg, ctx) = try loadModelFiles(name)
    // The start of the assistant's turn from the template's dialect (ChatML: `<|im_start|>assistant\n`).
    genMarker =
        (try? ChatModelInfo(directory: dir)).flatMap { info in
            info.dialect.generationMarker.map { info.tokenizer.encode($0).map { UInt32($0) } }
        } ?? (try? BPETokenizer(directory: dir))?.encode("<|im_start|>assistant\n").map { UInt32($0) } ?? []
    promptSet = args.opts["pdir"] ?? defaultPromptSet(cfg)
    return (dir, cfg, ctx)
}

/// Dense Qwen3.5 packs with their own prompts and mlx-lm references, by hidden size (a gen-<x> holds the references of
/// one model's weights, so two sizes never share it): the small ones template without reasoning by default.
let qwen35DensePromptSets = [1024: "prompts-q35-08b", 2048: "prompts-q35-2b"]

/// Gemma 4 packs with their own prompts and mlx-lm references, by hidden size (E2B: the CI's; 26B-A4B and 31B: bench).
let gemma4PromptSets = [1536: "prompts-gemma4", 2816: "prompts-gemma4-26b", 5376: "prompts-gemma4-31b"]

/// The model's prompt set when there is no --pdir: one per family (and per model when the template or the weights
/// differ).
func defaultPromptSet(_ cfg: ModelConfig) -> String {
    if cfg.raw["ingot_toy"] != nil { return "prompts-toy" }
    switch cfg.family {
    case .flashNext: return "prompts-fn"
    // LFM2.5-8B-A1B (32 experts): its chat template does not append `<think>` after the assistant turn as the 24B's
    // does (the model writes it), so its own prompts and mlx-lm references (gen-lfm8b).
    case .lfm2: return cfg.numExperts == 32 ? "prompts-lfm8b" : "prompts-lfm"
    case .granite: return "prompts-granite41"  // Granite 4.1 3B; bench families have their own tokenizers
    // Gemma 4 by hidden size: each model has its own template (the 26B/31B one adds an empty thought block) and weights.
    case .gemma4: return gemma4PromptSets[cfg.hiddenSize] ?? "prompts-gemma4-\(cfg.hiddenSize)"
    case .gptOss: return "prompts-gptoss"  // gpt-oss-20b: template clock pinned (prompts.py --fixed-now)
    // Nemotron-H: the Nano 4B (hybrid dense, the CI's) and the Lightning 30B-A3B (MoE, bench only).
    case .nemotronH: return cfg.isMoE ? "prompts-nemotron-l" : "prompts-nemotron"
    case .glm4MoeLite: return "prompts-glm"  // GLM-4.7-Flash: [gMASK]<sop> ... <|assistant|><think>
    case .qwen35: return cfg.isMoE ? "prompts" : qwen35DensePromptSets[cfg.hiddenSize] ?? "prompts-q27"
    case .museGlimmer: return "prompts-muse"  // Muse Glimmer 30B: template clock pinned (--fixed-now)
    case .lfm3: return "prompts-lfm3"  // LFM3-1.7B
    default: return cfg.isMoE ? "prompts" : "prompts-q27"
    }
}

/// ref/out/gen-<x>: mlx-lm's greedy and logits (gen_ref.py, logits_ref.py) of the prompt set prompts-<x>
/// (gen-lfm of LFM2.5-24B, gen-lfm8b of LFM2.5-8B). The Qwen3.6 MoE's set `prompts` uses
/// gen-qwen36: ref/out/gen is where `writeGen` writes, one per checkout and never shared between them.
/// Valid after `openModel`.
var genRefSet: String {
    if promptSet == "prompts" { return "gen-qwen36" }
    return "gen-" + (promptSet.hasPrefix("prompts-") ? String(promptSet.dropFirst("prompts-".count)) : promptSet)
}

func median(_ a: [Double]) -> Double { a.isEmpty ? 0 : a.sorted()[a.count / 2] }
func mean(_ a: [Double]) -> Double { a.isEmpty ? 0 : a.reduce(0, +) / Double(a.count) }

struct GenResult {
    let name: String, mode: String, promptTokens: Int
    let st: HybridModel.GenStats
    var prefillTps: Double { Double(promptTokens) / st.prefillSeconds }
    var decodeTps: Double { Double(st.tokens.count - 1) / st.decodeSeconds }
    var msPerRound: Double { st.decodeSeconds * 1e3 / Double(max(1, st.rounds)) }
    var line: String {
        String(
            format:
                "%@ %@: prompt %d tok, prefill %.0f tok/s (%.2f s); %d tok, decode %.1f tok/s; "
                + "%d rounds, %.2f tok/round, %.2f ms/round (GPU %.2f, CPU encoding %.2f)",
            name, mode, promptTokens, prefillTps, st.prefillSeconds, st.tokens.count, decodeTps, st.rounds,
            st.tokensPerRound, msPerRound, median(st.gpuMs), median(st.encodeMs))
    }
    func json(label: String, rowsPolicy: String) -> [String: Any] {
        [
            "label": label, "ctx": name, "prompt_tokens": promptTokens, "prefill_tps": (prefillTps * 10).rounded() / 10,
            "ttft_s": (st.prefillSeconds * 100).rounded() / 100, "decode_tps": (decodeTps * 10).rounded() / 10,
            "gen_tokens": st.tokens.count, "rounds": st.rounds,
            "tokens_per_round": (st.tokensPerRound * 100).rounded() / 100,
            "ms_per_round": (msPerRound * 100).rounded() / 100,
            "gpu_ms_per_round_median": (median(st.gpuMs) * 100).rounded() / 100,
            "gpu_ms_per_round_mean": (mean(st.gpuMs) * 100).rounded() / 100,
            "wall_ms_per_round_mean": (mean(st.wallMs) * 100).rounded() / 100,
            "wall_ms_per_round_median": (median(st.wallMs) * 100).rounded() / 100,
            "cb_gap_ms_mean": (mean(st.gapMs) * 1000).rounded() / 1000,
            "encode_ms_per_round_median": (median(st.encodeMs) * 100).rounded() / 100, "accepted_hist": st.acceptedHist,
            "ms_per_token": st.tokens.count > 1
                ? (st.decodeSeconds * 1e5 / Double(st.tokens.count - 1)).rounded() / 100 : 0,
            "copy_rounds": st.copyRounds, "copy_tokens": st.copyTokens, "copy_rows": st.copyRows,
            "copy_gpu_ms": (st.copyGpuMs * 10).rounded() / 10, "copy_deferred": st.deferred,
            "copy_deferred_nocopy": st.deferredNoCopy,
            "copy_enabled": st.copyRounds > 0 || st.deferred > 0, "rows_hist": st.rowsHist,
            "rows_mean": st.rounds > 0
                ? (Double(st.rowsHist.enumerated().reduce(0) { $0 + $1.0 * $1.1 }) / Double(st.rounds) * 100).rounded()
                    / 100 : 0,
            "rows_policy": rowsPolicy,
            "date": ISO8601DateFormatter().string(from: Date()),
        ]
    }
}
