import Foundation

/// MLX affine quantization parameters (default or of a specific tensor).
public struct QuantParams: Sendable, Equatable, CustomStringConvertible {
    public var bits: Int
    public var groupSize: Int
    public var mode: String
    public init(bits: Int, groupSize: Int, mode: String = "affine") {
        self.bits = bits; self.groupSize = groupSize; self.mode = mode
    }
    public var description: String { "\(mode) \(bits)b g\(groupSize)" }
}

/// Reading of the `config.json` of MLX models (dense and MoE, with or without `text_config`).
/// It does not assume a specific model: keeps the common fields and leaves the whole JSON accessible.
public struct ModelConfig: @unchecked Sendable {
    public let raw: [String: Any]
    public let text: [String: Any]

    public let modelType: String
    /// Family (ModelSpec.swift), from `model_type`; nil if the engine does not know it (ModelSpec errors on load).
    public let family: ModelFamily?
    /// What the engine assumes about the model (ModelSpec.swift); nil with an unknown family (`requiredSpec` errors).
    public private(set) var spec: ModelSpec?
    /// `spec`, or the unsupported-family error.
    public var requiredSpec: ModelSpec {
        get throws { try spec ?? ModelSpec(self) }
    }
    public let hiddenSize: Int
    public let numLayers: Int
    public let intermediateSize: Int?  // dense
    public let numExperts: Int?  // MoE
    public let expertsPerToken: Int?
    public let moeIntermediateSize: Int?
    public let sharedExpertIntermediateSize: Int?
    public let layerTypes: [String]
    public let rmsNormEps: Double
    public let vocabSize: Int

    /// Default quantization and per-path overrides (e.g. `...layers.0.mlp.gate` at 8 bits).
    public let quantDefault: QuantParams?
    public let quantOverrides: [String: QuantParams]

    public var isMoE: Bool { (numExperts ?? 0) > 0 }

    public init(directory: URL) throws {
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        guard let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw SafetensorsError.format("config.json is not an object")
        }
        try self.init(json: obj)
    }

    public init(json obj: [String: Any]) throws {
        raw = obj
        let t = (obj["text_config"] as? [String: Any]) ?? obj
        text = t
        func int(_ k: String) -> Int? { (t[k] as? NSNumber)?.intValue }
        guard let h = int("hidden_size"), let n = int("num_hidden_layers") else {
            throw SafetensorsError.format("config without hidden_size/num_hidden_layers")
        }
        modelType = (t["model_type"] as? String) ?? (obj["model_type"] as? String) ?? "?"
        family = ModelFamily.detect(textType: modelType, rawType: obj["model_type"] as? String)
        hiddenSize = h
        numLayers = n
        intermediateSize = int("intermediate_size")
        // Gemma 4: `enable_moe_block` false is dense whatever `num_experts` says (mlx-lm builds no MoE then), and the
        // top-k is `top_k_experts`.
        // Nemotron-H: `n_routed_experts`.
        numExperts =
            (t["enable_moe_block"] as? Bool) == false
            ? nil : int("num_experts") ?? int("num_local_experts") ?? int("n_routed_experts")
        expertsPerToken = int("num_experts_per_tok") ?? int("top_k_experts")
        // gpt-oss has no MoE-specific key: its experts are `intermediate_size` wide (it has no dense FFN).
        moeIntermediateSize =
            int("moe_intermediate_size") ?? int("expert_intermediate_size")
            ?? (family == .gptOss ? int("intermediate_size") : nil)
        sharedExpertIntermediateSize =
            int("shared_expert_intermediate_size") ?? int("moe_shared_expert_intermediate_size")
        // LFM2 calls its short-convolution blocks "conv".
        // Nemotron-H: one block per layer, from its pattern (`nemotronBlocks`): Mamba2 `mamba`, attention
        // `full_attention`, and the MoE and MLP blocks `ffn` (a layer without a mixer; `ModelSpec.ffnKinds` tells them
        // apart). An unreadable pattern leaves no layer types, which ModelSpec reports.
        if family == .nemotronH {
            layerTypes = (Self.nemotronBlocks(t) ?? []).map(Self.nemotronLayerType)
        } else {
            layerTypes = (t["layer_types"] as? [String]) ?? Array(repeating: "full_attention", count: n)
        }
        // Nemotron-H: `layer_norm_epsilon` (mlx-lm reads only that one for every norm of the family).
        rmsNormEps =
            (family == .nemotronH ? (t["layer_norm_epsilon"] as? NSNumber)?.doubleValue : nil)
            ?? (t["rms_norm_eps"] as? NSNumber)?.doubleValue ?? (t["norm_eps"] as? NSNumber)?.doubleValue ?? 1e-6
        vocabSize = int("vocab_size") ?? 0

        let q = (obj["quantization"] as? [String: Any]) ?? (obj["quantization_config"] as? [String: Any])
        var def: QuantParams?
        var ov: [String: QuantParams] = [:]
        if let q {
            if let b = (q["bits"] as? NSNumber)?.intValue, let g = (q["group_size"] as? NSNumber)?.intValue {
                def = QuantParams(bits: b, groupSize: g, mode: (q["mode"] as? String) ?? "affine")
            }
            for (k, v) in q {
                guard let d = v as? [String: Any], let b = (d["bits"] as? NSNumber)?.intValue,
                    let g = (d["group_size"] as? NSNumber)?.intValue
                else { continue }
                // An override without `mode` is affine, as in mlx (`to_quantized(**params)` defaults to affine): it
                // does not inherit the default's mode (MXFP4-Q8: mxfp4 default, routers `{bits 8, group_size 64}`).
                ov[k] = QuantParams(bits: b, groupSize: g, mode: (d["mode"] as? String) ?? "affine")
            }
        }
        quantDefault = def
        quantOverrides = ov
        spec = nil
        spec = try? ModelSpec(self)
    }

    /// Quantization declared for module `path` (without `.weight`).
    public func quant(for path: String) -> QuantParams? { quantOverrides[path] ?? quantDefault }
}
