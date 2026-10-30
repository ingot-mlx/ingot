import Foundation

// The pieces of a ModelSpec read per layer or per family: RoPE, MoE router, weight names, expert layout and FFN kind
// (ModelSpec.swift reads them from the config; ModelSpecGaps.swift checks what the engine cannot run).

/// RoPE of an attention layer: base, rotated dimensions (head_dim · partial_rotary_factor; LFM2, the whole head)
/// and whether the layer has no RoPE (`nope`: bit of `AttnParams.dyn`; from `no_rope_layers`, a 0 in
/// `layer_rope_theta` (Muse Glimmer's global layers) or the family (Nemotron-H)). Scaling: linear and yarn per layer
/// type; other types are a load error.
public struct RopeSpec: Sendable, Equatable {
    public let theta: Float
    public let rotaryDims: Int
    public let nope: Bool
    /// Frequency scaling of the layer (linear, yarn; nil = default RoPE). With any scaled layer the
    /// attention kernels read a per-layer table (RopeTable, LAB_AT_ROPETAB).
    public let scaling: RopeScaling?

    public init(theta: Float, rotaryDims: Int, nope: Bool = false, scaling: RopeScaling? = nil) {
        self.theta = theta; self.rotaryDims = rotaryDims; self.nope = nope; self.scaling = scaling
    }

    /// Table of the layer for the kernels with LAB_AT_ROPETAB (inv_freq per pair + mscale).
    public var table: [Float] { RopeTable.make(theta: Double(theta), rotaryDims: rotaryDims, scaling: scaling) }

    /// Scaling of a layer type: its `rope_parameters` entry, or the flat `rope_scaling` (gpt-oss) when that entry is
    /// the default one. nil when the config asks for none or for one the engine cannot run (`unsupported` reports it).
    static func scaling(_ json: [String: Any], layerType: String, family: ModelFamily? = nil) -> RopeScaling? {
        if let p = parameters(json, layerType: layerType), RopeScaling.type(p) != "default" {
            return try? RopeScaling.parse(p)
        }
        // LFM3: a default entry with partial_rotary_factor < 1 is its zero-tail partial RoPE, the proportional table.
        if family == .lfm3, let z = ModelSpec.lfm3ZeroTail(json, layerType: layerType) { return z }
        return (json["rope_scaling"] as? [String: Any]).flatMap { try? RopeScaling.parse($0) }
    }

    /// `rope_parameters.rope_theta`, or `rope_theta`, or 1e7 (Qwen3.5/3.6's) of a config (model's or drafter's).
    /// With `rope_parameters` nested by layer type (transformers 5, Gemma 4: `full_attention` / `sliding_attention`), the
    /// one of `layerType`.
    public static func theta(_ json: [String: Any], layerType: String = "full_attention") -> Float {
        Float(
            (parameters(json, layerType: layerType)?["rope_theta"] as? NSNumber)?.doubleValue
                ?? (json["rope_theta"] as? NSNumber)?.doubleValue ?? 1e7)
    }

    /// `rope_parameters` of a layer type: the nested entry if the config keys them by layer type, else the flat dict.
    public static func parameters(_ json: [String: Any], layerType: String) -> [String: Any]? {
        guard let rp = json["rope_parameters"] as? [String: Any] else { return nil }
        return isNested(rp) ? rp[layerType] as? [String: Any] : rp
    }

    /// Every RoPE parameter dict of a config: the flat one, or one per layer type.
    static func allParameters(_ json: [String: Any]) -> [[String: Any]] {
        guard let rp = json["rope_parameters"] as? [String: Any] else { return [] }
        return isNested(rp) ? rp.values.compactMap { $0 as? [String: Any] } : [rp]
    }

    /// `rope_parameters` keyed by layer type (transformers 5: one entry per `layer_types` value).
    static func isNested(_ rp: [String: Any]) -> Bool {
        !rp.isEmpty && rp.allSatisfy { ModelSpec.attentionLayerTypes.contains($0.key) && $0.value is [String: Any] }
    }
}

/// MoE router.
public struct RouterSpec: Sendable, Equatable {
    /// Scoring function of the router logits (`scoring_func`): softmax (Qwen) or sigmoid (LFM2, GLM,
    /// Nemotron-H). The selection bias, the scale and the epsilon are separate fields.
    public enum Scoring: Sendable { case softmax, sigmoid }
    /// Shared expert: none (LFM2), gated with sigmoid (Qwen3.5/3.6, Flash-Next) or ungated (DeepSeek convention:
    /// `n_shared_experts` in the config).
    public enum Shared: Sendable { case none, gated, ungated }
    public let scoring: Scoring
    public let shared: Shared
    public let experts: Int, topK: Int, inter: Int
    /// Shared expert intermediate size (`shared_expert_intermediate_size`, or `n_shared_experts` × experts'; 0 without
    /// shared expert). The kernels carry it separately (`sinter` of VMoEParams and `sNK` of MoeGemmParams).
    public let sharedInter: Int
    /// Selection-only bias added to the scores before the top-k (`expert_bias` in LFM2, `e_score_correction_bias` in
    /// GLM and Nemotron-H); the weights are the unbiased scores.
    public var selectBias = false
    /// `routed_scaling_factor`: multiplies the routed weights after the renormalization, not the shared expert.
    public var scale: Float = 1
    /// Epsilon added to the sum of the top-k sigmoid scores when renormalizing: 1e-6 in LFM2 (mlx-lm lfm2_moe); none in
    /// GLM-4.7 (glm4_moe_lite's 1e-20 is the same in fp32 for any sum above ~1e-13).
    public var normEps: Float = 0
    /// Router linear with bias (gpt-oss `mlp.router.bias`): added to the logits, so it counts for the choice and for the
    /// weights (not a selection-only bias). Experts with biases (gpt-oss `mlp.experts.{gate,up,down}_proj.bias`).
    var logitBias = false, expertBiases = false
    /// Gemma 4's router: it reads the un-normalized residual stream through its own RMSNorm (weight `router.scale` ·
    /// hidden^-0.5, folded at load) and multiplies each chosen expert's weight by `per_expert_scale` after the
    /// renormalization (LAB_MOE_PES).
    public var ownNorm = false, perExpertScale = false
}

/// Names of the weights that change between families (prefixes relative to `layers.N.`).
public struct WeightNames: Sendable, Equatable {
    /// FFN or MoE block ("mlp." / "feed_forward.").
    public let ffn: String
    public let oProj: String, qNorm: String, kNorm: String
    public let inputNorm: String, postNorm: String
    /// Final norm, relative to the model root ("norm.weight" / "embedding_norm.weight").
    public let finalNorm: String
    /// MoE experts inside `ffn` ("switch_mlp.").
    public let experts: String
    /// Sandwich post-norms of the mixer and FFN branches (relative to `layers.N.`) and embedding norm (relative
    /// to the model root), for the families that have them; nil = none (a block that asks for one without a name is a
    /// load error, or ones for an embedding norm without weight).
    public var mixerPostNorm: String?, ffnPostNorm: String?, embedNorm: String?
    /// Learned attention sinks inside `self_attn.` (gpt-oss naming, "sinks"; Flint's `softmax_offset` comes with its
    /// family). Read only with ModelSpec.attnSinks.
    public var sinks = "sinks"
    /// MoE router inside `ffn` ("gate"; gpt-oss, "router").
    public var router = "gate"
    /// Attention projections inside the layer ("self_attn."; Nemotron-H, "mixer.").
    public var attn = "self_attn."
    /// Embedding table relative to the model root ("embed_tokens"; Nemotron-H, "embeddings").
    public var embedding = "embed_tokens"
    /// Shared expert inside `ffn` ("shared_expert."; Nemotron-H, "shared_experts.").
    public var sharedExpert = "shared_expert."
    /// Expert projections: gate (nil = none, Nemotron-H's relu², whose shared expert has none either), up, down (fc1, fc2).
    public var expertGate: String? = "gate_proj", expertUp = "up_proj", expertDown = "down_proj"
    /// Dense FFN projections inside `ffn`: gate, up, down (LFM3: w1, w3, w2).
    public var denseGate = "gate_proj", denseUp = "up_proj", denseDown = "down_proj"
    /// MoE block relative to `layers.N.` when it is not `ffn` (Gemma 4: "", its router and experts hang from the layer).
    public var moe: String?
    /// Gemma 4's router norm weight and per-expert scale inside the MoE block (read only with RouterSpec.ownNorm and
    /// perExpertScale), and the norms of the parallel FFN (FFNKind.denseAndMoE): the MoE's pre-norm and the post-norms
    /// of the dense MLP and of the MoE, relative to `layers.N.`.
    public var routerScale = "router.scale", perExpertScale = "router.per_expert_scale"
    public var moePreNorm = "pre_feedforward_layernorm_2.weight", densePostNorm = "post_feedforward_layernorm_1.weight"
    public var moePostNorm = "post_feedforward_layernorm_2.weight"
    /// The MLA projections inside `attn` (GLM-4.7-Flash; GLM.swift); nil in the families with a q|k|v attention.
    public var mla: MLANames?
    /// The attention output gate in its own projection inside `attn` (Muse Glimmer: "gate_proj", `out · σ(gate_proj(x))`
    /// before o_proj); nil = none, or inside q_proj (Qwen3.5/3.6, Flash-Next).
    public var attnGate: String?

    static let qwen = WeightNames(
        ffn: "mlp.", oProj: "o_proj", qNorm: "q_norm.weight", kNorm: "k_norm.weight",
        inputNorm: "input_layernorm.weight", postNorm: "post_attention_layernorm.weight", finalNorm: "norm.weight",
        experts: "switch_mlp.")
    /// Gemma 4: `post_attention_layernorm` is the post-norm of the attention branch and the norm before the FFN is
    /// `pre_feedforward_layernorm` (the Qwen name of the pre-FFN norm means something else here); the 26B-A4B's MoE
    /// sits beside the dense `mlp.`: `router.proj` and `experts.switch_glu.{gate,up,down}_proj` directly under the
    /// layer.
    static let gemma4: WeightNames = {
        var n = WeightNames(
            ffn: "mlp.", oProj: "o_proj", qNorm: "q_norm.weight", kNorm: "k_norm.weight",
            inputNorm: "input_layernorm.weight", postNorm: "pre_feedforward_layernorm.weight", finalNorm: "norm.weight",
            experts: "experts.switch_glu.")
        n.mixerPostNorm = "post_attention_layernorm.weight"
        n.ffnPostNorm = "post_feedforward_layernorm.weight"
        n.moe = ""
        n.router = "router.proj"
        return n
    }()
    static let lfm2 = WeightNames(
        ffn: "feed_forward.", oProj: "out_proj", qNorm: "q_layernorm.weight", kNorm: "k_layernorm.weight",
        inputNorm: "operator_norm.weight", postNorm: "ffn_norm.weight", finalNorm: "embedding_norm.weight",
        experts: "switch_mlp.")
    /// gpt-oss: Qwen's norms and attention names, `mlp.router` and `mlp.experts.{gate,up,down}_proj`.
    static let gptOss: WeightNames = {
        var n = WeightNames(
            ffn: "mlp.", oProj: "o_proj", qNorm: "q_norm.weight", kNorm: "k_norm.weight",
            inputNorm: "input_layernorm.weight", postNorm: "post_attention_layernorm.weight", finalNorm: "norm.weight",
            experts: "experts.")
        n.router = "router"
        return n
    }()
}

/// Layout of the experts for reading them from the SSD (ExpertStream).
public enum ExpertLayout: Sendable, Equatable {
    /// `switch_mlp` of the MLX safetensors (9 chunks per expert: codes, scales and biases of gate, up and down).
    case mlxSwitchMLP
    /// Single record per expert in experts.bin (our Flash-Next format).
    case ingotRecords
    /// No experts or no ExpertStream support (LFM2).
    case unsupported
}

/// FFN of a layer: dense (SwiGLU), MoE, both in parallel (Gemma 4 26B-A4B: the dense MLP and the MoE without shared
/// expert, each with its pre- and post-norm, summed before the FFN post-norm), or none (Nemotron-H's Mamba2 and attention
/// layers: their only block is the mixer).
public enum FFNKind: Sendable, Equatable { case dense, moe, denseAndMoE, none }
