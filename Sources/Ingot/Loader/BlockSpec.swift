import Foundation

// Norms, activations and scales of a block, read from config.json on top of per-family defaults. Every
// key the engine knows is read here, and whatever it cannot execute yet is reported by `gaps` (ModelSpec turns it
// into a load error), so nothing in the config is ignored silently. With the defaults (every current family) the
// engine runs exactly as before.

/// How an RMSNorm weight is applied: `x·w` (plain), `x·(1 + w)` (Muse: folded into the weight at load time)
/// or no weight (`unit`: a buffer of ones, `x · 1` is exact).
public enum NormWeight: Sendable, Equatable { case plain, onePlus, unit }

/// One RMSNorm: weight convention and epsilon.
public struct NormSpec: Sendable, Equatable {
    public var weight: NormWeight
    public var eps: Float
    public init(_ weight: NormWeight, eps: Float) { self.weight = weight; self.eps = eps }
}

/// Activation of the gated FFN (dense and experts): `act(gate) · up`.
public enum FFNActivation: Sendable, Equatable {
    /// SiLU (SwiGLU): every current family.
    case silu
    /// GELU with the tanh approximation (GeGLU: Gemma 2/3/4; mlx-lm `nn.gelu_approx`).
    case geluTanh
    /// gpt-oss's clamped SwiGLU (mlx-lm `swiglu_oai`): `(clip(up, ±limit) + 1) · g·σ(1.702·g)` with `g = min(gate,
    /// limit)`; `limit` is the config's `swiglu_limit`. Only the MoE kernels have it (`_oai` instances).
    case swigluOAI(limit: Float)
    /// Nemotron-H's `up → relu² → down` without gate (mlx-lm `nn.relu2`): dense, experts and shared expert (`_relu2`
    /// instances; the kernels do not read a gate).
    case relu2
}

/// Norms, activation and scales of the transformer block. Fields that only exist for some families (sandwich
/// post-norms, embedding norm, layer scalar) are nil/false by default.
public struct BlockSpec: Sendable, Equatable {
    /// Norm before the mixer (`n1`), before the FFN (`n2`) and the final one.
    public var mixerNorm: NormSpec, ffnNorm: NormSpec, finalNorm: NormSpec
    /// Sandwich post-norms of the mixer and FFN branches (Gemma 4, Muse, LFM3); nil = none.
    public var mixerPostNorm: NormSpec?, ffnPostNorm: NormSpec?
    /// RMSNorm right after the embedding (Muse: without weight); nil = none.
    public var embedNorm: NormSpec?
    public var activation: FFNActivation
    /// Per-head RMSNorm of Q and K before RoPE (LAB_AT_QKN): its weight convention (`plain` in Qwen3.5/3.6,
    /// Flash-Next, LFM2 and Gemma 4, `unit` without weight in Muse; `onePlus`, x·(1 + w), is supported but no
    /// family uses it today); nil = none (Granite).
    public var qkNorm: NormWeight? = .plain
    /// Per-head RMSNorm of V (Gemma 4: without weight, `unit`); nil = none (every other family).
    public var vNorm: NormWeight?
    /// Embedding multiplier (Granite `embedding_multiplier`, MiniCPM `scale_emb`, Gemma √hidden).
    public var embedScale: Float = 1
    /// Multiplier of every branch output before it is added to the residual stream (Granite `residual_multiplier`,
    /// MiniCPM `scale_depth / √layers`).
    public var residualScale: Float = 1
    /// Logits post-processing, in mlx-lm order: `y = x · logitsMul / logitsDiv`, then `softcap · tanh(y / softcap)`.
    public var logitsMul: Float = 1, logitsDiv: Float = 1
    public var softcap: Float?
    /// Per-layer `layer_scalar` tensor that scales the whole stream after the layer (Gemma 4).
    public var layerScalar = false
    /// Attention score scale (default `1/√head_dim`).
    public var attnScale: Float
    /// Biases of q|k|v and of o_proj.
    public var qkvBias = false, oBias = false

    /// Whether some layer needs the branch epilogue (the branch output cannot be added straight into the stream).
    public var needsBranchEpilogue: Bool {
        mixerPostNorm != nil || ffnPostNorm != nil || residualScale != 1 || oBias || layerScalar
    }

    /// The defaults of the current families: plain norms with the model epsilon, SiLU, no scales or biases.
    public static func standard(eps: Float, headDim: Int) -> BlockSpec {
        let n = NormSpec(.plain, eps: eps)
        return BlockSpec(
            mixerNorm: n, ffnNorm: n, finalNorm: n, mixerPostNorm: nil, ffnPostNorm: nil, embedNorm: nil,
            activation: .silu, softcap: nil, attnScale: 1 / Float(headDim).squareRoot())
    }

    /// Per-family defaults with the config keys on top, and what the engine cannot execute (`gaps`, empty if it can).
    /// Reads from `text` (the `text_config` or the whole config) every block key known to some family in mlx-lm.
    public static func read(
        text t: [String: Any], family: ModelFamily?, eps: Float, headDim: Int, layers: Int, layerTypes: [String] = []
    ) -> (spec: BlockSpec, gaps: [String]) {
        // MLA (GLM-4.7-Flash): the attention head of the kernels is the latent one (576), but the scale is that of the
        // q/k head before absorption, (qk_nope_head_dim + qk_rope_head_dim)^-0.5 (mlx-lm's `Glm4MoeLiteAttention`).
        let headDim = family == .glm4MoeLite ? MLASpec(t)?.qHeadDim ?? headDim : headDim
        var b = standard(eps: eps, headDim: headDim)
        var gaps: [String] = []
        func num(_ k: String) -> Double? { (t[k] as? NSNumber)?.doubleValue }
        func flag(_ k: String) -> Bool { (t[k] as? Bool) ?? ((t[k] as? NSNumber)?.boolValue ?? false) }
        familyDefaults(&b, family: family, eps: eps, hidden: num("hidden_size") ?? 0)
        gaps += sandwich(&b, text: t, family: family, eps: eps)
        // Activation: `hidden_act` (Qwen, Granite, gpt-oss), `hidden_activation` (Gemma, Muse) and `mlp_hidden_act`
        // (Nemotron-H).
        for k in ["hidden_act", "hidden_activation", "mlp_hidden_act"] {
            guard let a = t[k] as? String else { continue }
            switch a {
            case "silu", "swish": b.activation = .silu
            case "gelu_pytorch_tanh", "gelu_tanh", "gelu_new": b.activation = .geluTanh
            case "relu2": b.activation = .relu2
            default: gaps.append("activation \(k) = \(a) (only silu and gelu_pytorch_tanh)")
            }
        }
        // swiglu_limit: gpt-oss's swiglu_oai (mlx-lm gpt_oss.py); other families mean other things by it.
        if let lim = num("swiglu_limit") {
            if family == .gptOss {
                b.activation = .swigluOAI(limit: Float(lim))
            } else {
                gaps.append("swiglu_limit = \(lim) outside gpt-oss (swiglu_oai only runs in the gpt-oss MoE)")
            }
        }
        // Biases: `attention_bias` (q, k, v and o: Granite, Llama, Nemotron-H), `qkv_bias` (q, k, v).
        if flag("attention_bias") { b.qkvBias = true; b.oBias = true }
        if flag("qkv_bias") { b.qkvBias = true }
        for k in ["mlp_bias", "use_bias"] where flag(k) {
            gaps.append("\(k) = true (FFN biases are not supported)")
        }
        // Logits.
        if let c = num("final_logit_softcapping") { b.softcap = Float(c) }
        if let c = num("attn_logit_softcapping") {
            gaps.append("attn_logit_softcapping = \(c) (attention softcap is not supported)")
        }
        if let m = num("output_multiplier") { b.logitsMul = Float(m) }
        if let d = num("logits_scaling") { b.logitsDiv = Float(d) }
        if let base = num("dim_model_base") { b.logitsDiv = Float(Double(Int(num("hidden_size") ?? 0)) / base) }
        // Embedding and residual multipliers.
        if let m = num("embedding_multiplier") { b.embedScale = Float(m) }
        if let m = num("scale_emb") { b.embedScale = Float(m) }
        if let m = num("residual_multiplier") { b.residualScale = Float(m) }
        if let s = num("scale_depth") { b.residualScale = Float(s / Double(layers).squareRoot()) }
        // Attention scale: Granite (`attention_multiplier`, the scale itself), Gemma 2 (`query_pre_attn_scalar`:
        // 1/√s) and Muse (`qk_scale_factor`: s/√head_dim).
        if let s = num("attention_multiplier") { b.attnScale = Float(s) }
        if let s = num("query_pre_attn_scalar") { b.attnScale = Float(pow(s, -0.5)) }
        if let s = num("qk_scale_factor") { b.attnScale = Float(s / Double(headDim).squareRoot()) }
        // Epsilon of the sandwich post-norms (Muse): only meaningful if the family has them.
        if let e = num("post_norm_eps") {
            if b.mixerPostNorm == nil && b.ffnPostNorm == nil {
                gaps.append("post_norm_eps = \(e) without sandwich post-norms in this family")
            } else {
                b.mixerPostNorm?.eps = Float(e)
                b.ffnPostNorm?.eps = Float(e)
            }
        }
        gaps += executionGaps(b, standard: standard(eps: eps, headDim: headDim), family: family, layerTypes: layerTypes)
        return (b, gaps)
    }

    /// LFM3's `use_sandwich_norm` (true by default in Lfm3Config): x·w post-norms after the mixer and the FFN of every
    /// layer, with the model epsilon; false leaves the block without them. No other family reads the key (a gap if on).
    static func sandwich(_ b: inout BlockSpec, text t: [String: Any], family: ModelFamily?, eps: Float) -> [String] {
        guard family == .lfm3 else {
            return (t["use_sandwich_norm"] as? Bool) == true ? ["use_sandwich_norm outside lfm3"] : []
        }
        if (t["use_sandwich_norm"] as? Bool) ?? true {
            b.mixerPostNorm = NormSpec(.plain, eps: eps)
            b.ffnPostNorm = NormSpec(.plain, eps: eps)
        }
        return []
    }

    /// The block of each family before the config keys (mlx-lm's per-family modules).
    static func familyDefaults(_ b: inout BlockSpec, family: ModelFamily?, eps: Float, hidden: Double) {
        switch family {
        // LFM3: LFM2's x·w norms and QK-norm; its sandwich post-norms come from `use_sandwich_norm` (read).
        case .qwen35, .qwen35Moe, .flashNext, .lfm2, .lfm3, nil: break
        case .granite, .gptOss, .nemotronH, .glm4MoeLite:
            b.qkNorm = nil
        case .gemma4:
            // mlx-lm gemma4_text.py: a sandwich block with every RMSNorm (also QK) as x·w, a weightless
            // v_norm, GeGLU, embedding × √hidden rounded to bf16, attention scale 1 and `layer_scalar` per layer.
            let n = NormSpec(.plain, eps: eps)
            b.mixerPostNorm = n; b.ffnPostNorm = n
            b.vNorm = .unit
            b.activation = .geluTanh
            b.embedScale = bf16ToFloat(floatToBF16(Float(hidden).squareRoot()))
            b.attnScale = 1
            b.layerScalar = true
        case .museGlimmer:
            // mlx-lm muse_glimmer.py: the four per-layer norms centered, x·(1 + w) (the post-norms take `post_norm_eps`
            // below), the final one x·w, a weightless RMSNorm on the embeddings with `rms_norm_eps`, a weightless
            // QK-norm (its `qk_scale_factor` is read below) and SwiGLU.
            let n = NormSpec(.onePlus, eps: eps)
            b.mixerNorm = n; b.ffnNorm = n; b.mixerPostNorm = n; b.ffnPostNorm = n
            b.embedNorm = NormSpec(.unit, eps: eps)
            b.qkNorm = .unit
        }
    }

    /// What the engine cannot execute yet of `b` (it first reported everything that differs from the current families;
    /// each change removes what it implements).
    static func executionGaps(_ b: BlockSpec, standard s: BlockSpec, family: ModelFamily?, layerTypes: [String])
        -> [String]
    {
        var out: [String] = []
        // v_norm: only weightless (Gemma 4), which the prework kernels run (LAB_AT_VN).
        if let v = b.vNorm, v != .unit { out.append("v_norm with weight (\(v): only the weightless one runs)") }
        // GELU-tanh: `_gelu` instances of the gated FFN kernels (dense, --int8 and MoE); Flash-Next has its own.
        if b.activation != .silu && family == .flashNext {
            out.append("FFN activation \(b.activation) in Flash-Next (its kernels are SiLU only)")
        }
        // Norms, branch epilogue, embedding and logits pieces run in the skeleton of the Qwen/LFM2 families; Flash-Next
        // has its own hooks (hyper-connections), and the branch epilogue only goes around attention and short-convolution
        // mixers (LFM3; the GDN still accumulates straight into the stream in the prefill).
        var t3 = b
        t3.activation = s.activation
        t3.qkvBias = s.qkvBias
        if family == .flashNext && t3 != s { out.append("block norms, scales or biases in Flash-Next (its own hooks)") }
        let other = Set(layerTypes).subtracting(ModelSpec.attentionLayerTypes).subtracting(["conv"])
        if b.needsBranchEpilogue && !other.isEmpty {
            out.append(
                "branch epilogue (post-norm, residual multiplier, o_proj bias or layer scalar) with "
                    + "\(other.sorted().joined(separator: ", ")) layers (only around attention and short convolution)")
        }
        // q|k|v biases in the prework (LAB_AT_QKVB), not with the gated q_proj of Qwen3.5/3.6 and Flash-Next.
        if b.qkvBias && ModelSpec.attnGated(family) {
            out.append("q/k/v biases with a gated q_proj (\(family?.rawValue ?? "?"))")
        }
        return out
    }

    /// Nothing around the layers beyond the norm weights: no branch epilogue, embedding scale or norm, or logits
    /// post-processing (the per-piece timeline only profiles this skeleton).
    public var plainSkeleton: Bool {
        !needsBranchEpilogue && embedScale == 1 && embedNorm == nil && logitsMul == 1 && logitsDiv == 1
            && softcap == nil
    }
}
