import Foundation

// Whether a DFlash2 drafter fits a model, and which drafter a model runs with. The family's default drafter
// (ModelSpec.defaultDrafter) is the one of its largest model: Qwen3.6-35B-A3B's for all of qwen35, which does not fit
// Qwen3.5 0.8B (hidden 1024) and asks Qwen3.5 2B (24 layers) for layers 27, 32 and 37. Gemma 4's (z-lab's DFlash v1
// of the 26B-A4B) is only the default for hidden 2816 (FamilyTraits.defaultDrafterHidden). Reading only config.json and
// the safetensors headers: no weights, no GPU.

extension DFlash2Drafter {
    /// Tensor present only in DFlash2 (the candidate selector); without it the drafter is DFlash v1.
    static let v2Tensor = "candidate_selector.hidden_projection.weight"

    /// Why the drafter with config `config` (its config.json) does not fit a model with `hidden` and `layers`, or nil if
    /// it fits: hidden size, target layers inside the model, `num_target_layers` (if present) equal to the model's
    /// layers, and the shape the kernels support (32 q / 8 KV heads of 128, or 64/8 of 64 with optional attention
    /// biases and YaRN; with `v1` known, the DFlash2 block, window, conv and selector). `layers` nil skips the layer
    /// checks; `v1` nil skips the DFlash2 shape check.
    public static func mismatch(config cj: [String: Any], hidden: Int, layers: Int?, v1: Bool?) -> String? {
        func int(_ d: [String: Any]?, _ k: String) -> Int? { (d?[k] as? NSNumber)?.intValue }
        let dc = cj["dflash_config"] as? [String: Any]
        guard let dc else { return "no dflash_config" }
        guard let h = int(cj, "hidden_size") else { return "no hidden_size" }
        if h != hidden { return "hidden \(h) vs \(hidden)" }
        if let layers {
            guard let ids = dc["target_layer_ids"] as? [NSNumber] else { return "no target_layer_ids" }
            if let bad = ids.map(\.intValue).first(where: { $0 < 0 || $0 >= layers }) {
                return "target layer \(bad) ≥ \(layers) layers"
            }
            if let n = int(cj, "num_target_layers"), n != layers { return "num_target_layers \(n) vs \(layers)" }
        }
        let hd = int(cj, "head_dim") ?? 128, qh = int(cj, "num_attention_heads") ?? 0
        guard (hd == 128 && qh == 32 || hd == 64 && qh == 64) && int(cj, "num_key_value_heads") == 8 else {
            return "heads \(qh)/\(int(cj, "num_key_value_heads") ?? 0) of \(hd) (32/8 of 128 or 64/8 of 64 supported)"
        }
        // The head-64 kernels (gpt-oss's) take the q/k/v/o biases and a scaled RoPE; the head-128 ones neither.
        let scaled: Bool
        do {
            scaled = try ropeScaling(cj) != nil
        } catch {
            return "RoPE: \(error)"
        }
        if hd == 128 && (cj["attention_bias"] as? Bool == true || scaled) {
            return "attention biases or a scaled RoPE with heads of 128 (only with heads of 64)"
        }
        if cj["mlp_bias"] as? Bool == true { return "MLP biases" }
        if v1 == false {
            let v2OK =
                int(dc, "block_size").map(blocks.contains) == true && int(cj, "sliding_window") == window
                && int(dc, "conv_kernel_size") == 2 && int(dc, "conv_group_size") == 16
                && int(dc, "selector_top_k") == 16 && int(dc, "selector_rank") == 256
            if !v2OK { return "DFlash2 block, window, conv or selector not supported" }
        }
        return nil
    }

    /// The drafter's RoPE scaling (`rope_scaling` or `rope_parameters`; nil for the default RoPE): the kernels take
    /// one per drafter, of the kinds RopeTable computes (yarn, linear).
    static func ropeScaling(_ cj: [String: Any]) throws -> RopeScaling? {
        let all = try (RopeSpec.allParameters(cj) + [cj["rope_scaling"]].compactMap { $0 as? [String: Any] })
            .compactMap { try RopeScaling.parse($0) }
        guard let s = all.first else { return nil }
        guard all.allSatisfy({ $0 == s }), s.kind != .proportional else {
            throw SafetensorsError.format("drafter with several or proportional RoPE scalings")
        }
        return s
    }

    /// `mismatch` for the drafter in `directory` (config.json and, if readable, the safetensors headers for v1/v2).
    public static func mismatch(directory: URL, hidden: Int, layers: Int) throws -> String? {
        let url = directory.appendingPathComponent("config.json")
        guard let cj = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any] else {
            throw SafetensorsError.format("\(url.path): not a JSON object")
        }
        let v1 = (try? ModelFootprint.headerTensors(directory: directory)).map { !$0.contains { $0.name == v2Tensor } }
        return mismatch(config: cj, hidden: hidden, layers: layers, v1: v1)
    }

    /// The drafter a model runs with, or nil for none. In order:
    /// 1. `option` (`--drafter`) "none": nil;
    /// 2. `option` with a name or path: that one (an error if it does not fit);
    /// 3. a family without drafter (LFM2, Granite, gpt-oss…): nil, except a toy (`metal_lab_toy`) with a `drafter/`
    ///    when `toyPack` (lab serve and every lab-dev check: toy-gptoss's); a real pack's `drafter/` is then left out
    ///    and `log` says so;
    /// 4. the `drafter/` of the model pack (the 27B's, the toys'): that one (an error if it does not fit);
    /// 5. none if the family's is not for this model (ModelSpec.defaultDrafter nil: Gemma 4 other than the 26B-A4B);
    ///    it is not looked up;
    /// 6. the family's (ModelSpec.defaultDrafter; Qwen3.6's with no known family), located with `locate` (an error if
    ///    it is not in the cache): that one if it fits; if not, nil and `log` gets the reason.
    public static func resolve(
        modelDir: URL, config cfg: ModelConfig, option: String?, toyPack: Bool,
        locate: (String) throws -> URL = ModelPaths.resolve, log: (String) -> Void
    ) throws -> URL? {
        func checked(_ d: URL, _ what: String) throws -> URL {
            if let why = try mismatch(directory: d, hidden: cfg.hiddenSize, layers: cfg.numLayers) {
                throw SafetensorsError.format("drafter \(what) does not fit this model (\(why))")
            }
            return d
        }
        if option == "none" { return nil }
        if let option { return try checked(try locate(option), option) }
        let inPack = modelDir.appendingPathComponent("drafter")
        let hasPack = FileManager.default.fileExists(atPath: inPack.appendingPathComponent("config.json").path)
        if cfg.spec?.hasDrafter == false {
            let toy = cfg.raw["metal_lab_toy"] != nil
            if hasPack && !toy { log("drafter/ left out: family \(cfg.family?.rawValue ?? "?") runs without drafter") }
            return toyPack && hasPack && toy ? try checked(inPack, "drafter/") : nil
        }
        if hasPack { return try checked(inPack, "drafter/") }
        let name: String
        if let spec = cfg.spec {
            guard let n = spec.defaultDrafter else { return nil }
            name = n
        } else {
            name = "incoai/Qwen3.6-35B-A3B-DFlash2"
        }
        let d = try locate(name)
        if let why = try mismatch(directory: d, hidden: cfg.hiddenSize, layers: cfg.numLayers) {
            log("default drafter \(name) does not fit this model (\(why)): running without drafter")
            return nil
        }
        return d
    }
}
