import Foundation

// Family table: what the engine assumes about each architecture comes from here, read from config.json, and a
// family or shape the engine cannot run gives a load error. It adds no new variants and
// changes no kernel: dimensions still reach the kernels as macros (ModelDims), with one library per
// shape.

/// Families the engine can run (or, while a family is being added, recognizes with named load errors).
public enum ModelFamily: String, Sendable, CaseIterable {
    /// Dense Qwen3.5/3.8 (GDN + gated attention; the 27B) and Bonsai 2 (prism_hadamard_qwen35).
    case qwen35
    /// Qwen3.5/3.6 MoE (256 experts top-8 + gated shared expert: Heretic, Ornith).
    case qwen35Moe
    /// Qwen3.8-Flash-Next in our format (hyper-connections, PLE, QSA, Q2 experts from experts.bin).
    case flashNext
    /// LFM2 / LFM2.5 (lfm2_moe): short convolution, attention without gate, MoE with sigmoid router.
    case lfm2
    /// Granite dense (granite: Granite 4.1/4.2): attention only (GQA without gate, full RoPE, no QK-norm, head of
    /// hidden / heads as mlx-lm), dense SwiGLU FFN and the μP multipliers of the block (embedding, residual, attention
    /// and logits; BlockSpec), with the scales and the q/k/v/o biases.
    case granite
    /// Gemma 4 text (gemma4_text; gemma4_unified_text in the 12B): attention only with sliding-window layers, sandwich
    /// post-norms, `x·w` norms (also QK), a weightless v_norm, GeGLU, embedding × √hidden (bf16), attention scale 1,
    /// `layer_scalar` and nested `rope_parameters` (`proportional` in the full layers); tied embedding. What the engine
    /// does not run is a load error with its name (`ModelSpec.gemma4Gaps`).
    case gemma4
    /// gpt-oss (gpt_oss): attention only with alternating sliding-window (128) and full layers, 64 q / 8 kv heads of 64
    /// without gate or QK-norm, q/k/v/o biases, learned sinks in every layer (implied by the family) and flat YaRN
    /// (`truncate: false`); softmax MoE top-4 of 32 experts without shared expert, `intermediate_size` wide, with
    /// `mlp.router` and `mlp.experts.` names, router and expert biases (RouterSpec.logitBias, expertBiases), swiglu_oai
    /// (BlockSpec) and the hidden of 2880, a multiple of 64 but not of 128, and MXFP4 experts (any
    /// other layer in MXFP4 is a load error, `ModelSpec.quantModeGaps` and QuantizedLinear).
    case gptOss
    /// Nemotron-H (nemotron_h: Nano 4B, Lightning 30B-A3B; NemotronH.swift): one block per layer. What the engine does
    /// not run is a load error with its name (`ModelSpec.nemotronHGaps`).
    case nemotronH
    /// GLM-4.7-Flash (glm4_moe_lite; GLM.swift): MLA attention (low-rank q and a shared latent KV, RoPE on the last
    /// `qk_rope_head_dim` dimensions, interleaved), the first `first_k_dense_replace` layers dense and the rest a sigmoid
    /// MoE with selection bias and an ungated shared expert. The MLA runs in its decompressed form
    /// (Layers/MLA.swift); what is not supported is a load error with its name (`ModelSpec.glmGaps`).
    case glm4MoeLite
    /// Muse Glimmer (muse_glimmer, text tower `muse_glimmer_text`): attention only, `[sliding ×3, full]` with a window
    /// of 2048 and NoPE in the global layers (`layer_rope_theta` 0), GQA with heads of 128 and full RoPE, a weightless
    /// QK-norm with `qk_scale_factor`, the output gate in its own `gate_proj` (interleaved with q_proj at load time into
    /// Qwen's `[q | gate]` row; AttentionLayer.gateRowOrder), four centered norms per layer (post-norms with
    /// `post_norm_eps`), a weightless norm after the embedding, dense SwiGLU and logits × `output_multiplier` then
    /// softcapped. The vision tower is not loaded.
    case museGlimmer
    /// LFM3 (lfm3: LFM3-1.7B; LFM3Spec.swift): LFM2's short convolution beside full and sliding attention with a scalar
    /// gate per head, the zero-tail partial RoPE, sandwich post-norms and a `w1/w3/w2` FFN. Recognized by the loader;
    /// what does not run yet is a load error with its name (`ModelSpec.lfm3Gaps`).
    case lfm3

    /// The only place that looks at `model_type`: the text one (`text_config`) and the outer one.
    public static func detect(textType: String, rawType: String?) -> ModelFamily? {
        if textType.hasPrefix("qwen3_5_moe") { return .qwen35Moe }
        if textType.hasPrefix("qwen3_5") || rawType == "prism_hadamard_qwen35" { return .qwen35 }
        if textType.hasPrefix("qwen4_exp") { return .flashNext }
        if textType == "lfm3" { return .lfm3 }
        if textType.hasPrefix("lfm2") { return .lfm2 }
        if textType == "granite" { return .granite }
        if ["gemma4_text", "gemma4_unified_text"].contains(textType) || ["gemma4", "gemma4_unified"].contains(rawType) {
            return .gemma4
        }
        if textType == "gpt_oss" { return .gptOss }
        if textType == "nemotron_h" { return .nemotronH }
        if textType == "glm4_moe_lite" { return .glm4MoeLite }
        if textType == "muse_glimmer_text" || rawType == "muse_glimmer" { return .museGlimmer }
        return nil
    }

    /// Small weights (norms, conv, dt_bias) in fp32 instead of bf16: Bonsai 2 (prism_hadamard_qwen35) and our
    /// Flash-Next format. A_log does not depend on it: it is always loaded in fp32 (GDNLayer).
    public static func smallWeightsF32(_ cfg: ModelConfig) -> Bool {
        (cfg.raw["model_type"] as? String) == "prism_hadamard_qwen35" || cfg.family?.traits.smallWeightsF32 == true
    }

    /// Name of the family in the row calibration files (`Calibration/rows-<name>.json`).
    public var calibrationName: String { traits.calibrationName }

    /// What the family decides by itself: one row per family (the defaults are those of most families).
    var traits: FamilyTraits {
        switch self {
        case .qwen35:
            FamilyTraits(
                calibrationName: "qwen3_5", names: .qwen, calibration: .qwen35Dense,
                defaultDrafter: "incoai/Qwen3.6-35B-A3B-DFlash2", attnGated: true, partialRotary: 0.25)
        case .qwen35Moe:
            FamilyTraits(
                calibrationName: "qwen3_5_moe", names: .qwen, calibration: .qwen36MoE,
                defaultDrafter: "incoai/Qwen3.6-35B-A3B-DFlash2", expertLayout: .mlxSwitchMLP, attnGated: true,
                moeRouter: true, sharedExpert: .gated, partialRotary: 0.25)
        case .flashNext:
            FamilyTraits(
                calibrationName: "qwen4_exp", names: .qwen, calibration: .flashNext,
                defaultDrafter: "incoai/Qwen3.8-Flash-Next-DFlash2", expertLayout: .ingotRecords, int8Experts: true,
                attnGated: true, moeRouter: true, sharedExpert: .gated, smallWeightsF32: true)
        case .lfm2:
            FamilyTraits(
                calibrationName: "lfm2_moe", names: .lfm2, calibration: .lfm2, moeRouter: true, sigmoidRouter: true,
                routerNormEps: 1e-6)
        case .granite: FamilyTraits(calibrationName: "granite", names: .qwen, calibration: .unknown)
        // Gemma 4: z-lab's DFlash v1 drafter of the 26B-A4B (block 8 of its 16, window 2048), by default only with
        // its hidden 2816 (E2B, E4B and 31B run without one and do not look it up); a toy's drafter/ goes first.
        case .gemma4:
            FamilyTraits(
                calibrationName: "gemma4", names: .gemma4, calibration: .gemma4,
                defaultDrafter: "z-lab/gemma-4-26B-A4B-it-DFlash", defaultDrafterHidden: 2816, moeRouter: true,
                ownRouterNorm: true)
        // gpt-oss: z-lab's DFlash v1 of the 20b (heads of 64 with biases and YaRN, block 8, window 8K, rows from
        // rows-gpt_oss.json), by default only with its hidden 2880 and 24 layers (the 120b, also hidden 2880, has 36:
        // it runs without one and does not look it up); a toy's drafter/ goes first.
        case .gptOss:
            FamilyTraits(
                calibrationName: "gpt_oss", names: .gptOss, calibration: .gptOss,
                defaultDrafter: "z-lab/gpt-oss-20b-DFlash", defaultDrafterHidden: 2880, defaultDrafterLayers: 24,
                moeRouter: true, moeBiases: true, attnSinks: true)
        case .nemotronH:
            FamilyTraits(
                calibrationName: "nemotron_h", names: .nemotronH, calibration: .nemotronH, moeRouter: true,
                sharedExpert: .ungated, sigmoidRouter: true, noRope: true)
        case .glm4MoeLite:
            FamilyTraits(
                calibrationName: "glm4_moe_lite", names: .glm, calibration: .glm4MoeLite, moeRouter: true,
                sharedExpert: .ungated, sigmoidRouter: true)
        // Muse Glimmer: incoai's DFlash2, block 16, draft logits × multiplier and softcap; the pack's
        // drafter/ goes first (DrafterFit.resolve).
        case .museGlimmer:
            FamilyTraits(
                calibrationName: "muse_glimmer", names: .muse, calibration: .museGlimmer,
                defaultDrafter: "incoai/Muse-Glimmer-30B-DFlash2", attnGated: true)
        case .lfm3: FamilyTraits(calibrationName: "lfm3", names: .lfm3, calibration: .lfm3, attnGated: true)
        }
    }
}

/// What a family decides by itself, not read from config.json (ModelFamily.traits).
struct FamilyTraits {
    /// Name of the family in the row calibration files (`Calibration/rows-<name>.json`).
    let calibrationName: String
    /// Weight names (WeightNames).
    let names: WeightNames
    /// What the family takes on the GPU besides weights and KV (FamilyCalibration).
    let calibration: FamilyCalibration
    /// DFlash2 (or DFlash v1) drafter used if the pack has none (`drafter/`); nil = the family has none.
    var defaultDrafter: String?
    /// The model hidden size `defaultDrafter` is for (nil = any): other models of the family run without a default
    /// drafter (their pack's `drafter/` still counts).
    var defaultDrafterHidden: Int?
    /// The model layer count `defaultDrafter` is for (nil = any), like `defaultDrafterHidden`.
    var defaultDrafterLayers: Int?
    /// Layout of the experts for reading them from the SSD (ExpertStream).
    var expertLayout = ExpertLayout.unsupported
    /// Lossy prefill (--int8 1): position from which the dense GEMMs run in rotated W8A8 (ModelSpec.int8MinPos).
    var int8MinPos = 1024
    /// Lossy prefill: the MoE experts also take int8 activations (ModelSpec.int8Experts).
    var int8Experts = false
    /// Per-head gate in the q row ([q | gate], LAB_AT_GATE; ModelSpec.attnGated).
    var attnGated = false
    /// A MoE config of the family gets a router (the others are a load error, `routerPresenceGaps`).
    var moeRouter = false
    /// Shared expert: none, gated, or ungated when the config declares `n_shared_experts` (ModelSpec.sharedKind).
    var sharedExpert = RouterSpec.Shared.none
    /// Router defaults when the config has no key (ModelSpec.routerKeys): sigmoid scoring with selection bias, and the
    /// renormalization epsilon.
    var sigmoidRouter = false
    var routerNormEps: Float = 0
    /// Gemma 4's router: its own norm and the per-expert scale (RouterSpec.ownNorm, perExpertScale).
    var ownRouterNorm = false
    /// The router linear and experts carry biases (gpt-oss: every checkpoint has them, no config key says so).
    var moeBiases = false
    /// Learned attention sinks in every layer without declaring them (gpt-oss; ModelSpec.attnSinks).
    var attnSinks = false
    /// No attention layer rotates (Nemotron-H: mlx-lm and transformers rotate nothing there).
    var noRope = false
    /// `partial_rotary_factor` if the config lacks it (ModelSpec.defaultPartialRotary).
    var partialRotary = 1.0
    /// Small weights in fp32 (our Flash-Next format; ModelFamily.smallWeightsF32).
    var smallWeightsF32 = false
}

/// What the engine needs to know about a model, read from its config.json (`ModelSpec(cfg)`).
public struct ModelSpec: Sendable {
    public let family: ModelFamily
    /// Kernel compile-time constants (LAB_* macros).
    public var shape: ModelShape
    /// Shape of the full attention layers when it differs from `shape` (Gemma 4: heads of 512, and with K = V their
    /// own kv heads; ModelDims.fullAttention): those layers get a second MetalContext with this library. nil: one shape.
    public let fullShape: ModelShape?
    /// The full attention layers have no v_proj and take V from the raw k projection (`attention_k_eq_v`, Gemma 4
    /// 12B/26B/31B): the loader concatenates k_proj twice, so the prework normalizes and rotates K and only normalizes V.
    public let kEqV: Bool
    /// The layer whose KV each layer reads (Gemma 4 E2B/E4B, `num_kv_shared_layers`: the last ones have no k/v
    /// projections and read the KV of the last earlier layer of their type, at the same positions, as mlx-lm's
    /// `previous_kvs`); nil = its own KV (every layer of the other families).
    public let kvSource: [Int?]
    /// Width per layer of the per-layer inputs (Gemma 4 E2B/E4B, `hidden_size_per_layer_input`; GemmaPLE.swift); 0 =
    /// none.
    public let pleDim: Int
    /// RoPE of the model (the one of the attention layers and the one of Flash-Next's QSA).
    public let rope: RopeSpec
    /// RoPE of each layer (nil in the non-attention ones): θ and `nope` reach the layer (AttentionLayer). Today all use
    /// the model's except those `no_rope_layers` marks with 0 (SmolLM3 and Llama 4 convention: 1 = with RoPE) and those
    /// with θ 0 in `layer_rope_theta` (Muse Glimmer), which run without RoPE.
    public let ropes: [RopeSpec?]
    /// Sliding window of each layer, counting its own key (row at position q sees keys k with q − k < W, as mlx-lm's
    /// `create_attention_mask(window_size:)`); 0 = full attention. `sliding_window` on `sliding_attention` layers
    /// (Gemma 4, gpt-oss, Muse Glimmer, LFM3).
    public let windows: [Int]
    /// Some attention layer has a scaled RoPE: the library is built with LAB_AT_ROPETAB and every attention layer
    /// passes its table.
    public var hasRopeTable: Bool { ropes.contains { $0?.scaling != nil } }
    public var hasSlidingWindow: Bool { windows.contains { $0 > 0 } }
    /// FFN of each layer (LFM2: the first `num_dense_layers` are dense).
    public let ffnKinds: [FFNKind]
    public let router: RouterSpec?
    public let names: WeightNames
    public let expertLayout: ExpertLayout
    /// Whether the family has a drafter (the pack's `drafter/` is used) and which one is used if the pack has none
    /// (`drafter/`; nil for the models of the family it is not for: Gemma 4 other than the 26B-A4B, gpt-oss other
    /// than the 20b).
    public let hasDrafter: Bool
    public let defaultDrafter: String?
    public let calibration: FamilyCalibration
    /// Lossy prefill (--int8 1): position from which the dense GEMMs run in rotated W8A8 (earlier rows are
    /// exact; 0 = all in int8). Per family, measuring 0, 512 and 1024: the lowest
    /// with mean KL < 1e-3 against the exact one, counting 512, 4K and 16K:
    /// - Qwen3.6 MoE: 1024 (lower KL and more prefill throughput at 4K than 512). With 1024, KL 0 (512, exact) and
    ///   1.3e-4 (4K), 2,674 tok/s at 4K; with 512, 1.7e-4 (4K), 2,494 tok/s (the cut splits the prompt in 3 chunks).
    ///   With 0, the 512 prompt gives 3.2e-3.
    /// - Flash-Next: 1024. None gets below 1e-3 at 4K/16K (1.2–1.4e-3 with all three: the rest of int8 dominates); 1024
    ///   leaves 512 exact (KL 0 vs 1.9e-3 with 0) at the cost of -5% prefill throughput at 4K and 16K.
    /// - Unmeasured (27B, Bonsai, LFM2): 1024.
    public let int8MinPos: Int
    /// Lossy prefill (--int8 1): whether the MoE experts also take int8 activations in the int8 chunks, a fixed
    /// property of the family (no option):
    /// - Flash-Next: on (hidden not a multiple of 128, Q2 experts with x in int8 per group of 64).
    /// - The rest: off. The Qwen3.6 MoE Q4 path (+3.3/+5.0/+4.8 % prefill at 4K/16K/32K on the Heretic,
    ///   but 1.66x the KL of the varied prompts) was retired.
    public let int8Experts: Bool
    /// Norms, activation and scales of the block (BlockSpec.swift).
    public let block: BlockSpec
    /// Multi-head latent attention of every attention layer (GLM-4.7-Flash; GLM.swift); nil in the other families.
    public let mla: MLASpec?

    public init(_ cfg: ModelConfig) throws {
        guard let fam = cfg.family else {
            if let gap = Self.unregisteredFamilyGap(cfg) {
                throw SafetensorsError.format("\(cfg.modelType): not supported yet: \(gap)")
            }
            throw SafetensorsError.format(
                "unsupported family: model_type \(cfg.modelType) (the engine can run "
                    + ModelFamily.allCases.map(\.rawValue).joined(separator: ", ") + ")")
        }
        family = fam
        let base = ModelShape(cfg)
        shape = base
        let full = ModelShape.fullAttention(cfg)
        fullShape = full
        kEqV = (cfg.text["attention_k_eq_v"] as? Bool) ?? false
        let t = cfg.text
        func int(_ k: String) -> Int? { (t[k] as? NSNumber)?.intValue }
        let r0 = RopeSpec(
            theta: RopeSpec.theta(t), rotaryDims: full?.attnRot ?? shape.attnRot,
            scaling: RopeSpec.scaling(t, layerType: "full_attention", family: fam))
        rope = r0
        ropes = Self.layerRopes(cfg, base: base, full: full)
        let window = int("sliding_window") ?? 0
        windows = cfg.layerTypes.map { $0 == "sliding_attention" ? window : 0 }
        kvSource = Self.kvSources(cfg)
        pleDim = fam == .gemma4 ? int("hidden_size_per_layer_input") ?? 0 : 0
        mla = fam == .glm4MoeLite ? MLASpec(t) : nil
        try Self.gdnGuard(cfg, shape: shape)
        let tr = fam.traits
        names = tr.names
        ffnKinds = Self.ffnKinds(cfg)
        if cfg.isMoE, tr.moeRouter, let e = cfg.numExperts, let k = cfg.expertsPerToken {
            let inter = cfg.moeIntermediateSize ?? 0
            // Scoring, selection bias, scale and epsilon from the config, with the family's as the default.
            let sk = Self.sharedKind(cfg)
            let rk = Self.routerKeys(t, family: fam)
            var r = RouterSpec(
                scoring: rk.scoring, shared: sk, experts: e, topK: k, inter: inter,
                sharedInter: sk == .none
                    ? 0 : cfg.sharedExpertIntermediateSize ?? (int("n_shared_experts") ?? 0) * inter,
                selectBias: rk.selectBias, scale: rk.scale, normEps: rk.normEps)
            r.logitBias = tr.moeBiases
            r.expertBiases = tr.moeBiases
            r.ownNorm = tr.ownRouterNorm
            r.perExpertScale = tr.ownRouterNorm
            router = r
        } else {
            router = nil
        }
        expertLayout = tr.expertLayout
        let draftFits =
            (tr.defaultDrafterHidden.map { $0 == cfg.hiddenSize } ?? true)
            && (tr.defaultDrafterLayers.map { $0 == cfg.numLayers } ?? true)
        defaultDrafter = draftFits ? tr.defaultDrafter : nil
        hasDrafter = tr.defaultDrafter != nil
        calibration = tr.calibration
        int8MinPos = tr.int8MinPos
        int8Experts = tr.int8Experts
        let (blk, blockGaps) = BlockSpec.read(
            text: t, family: fam, eps: Float(cfg.rmsNormEps), headDim: shape.attnHD, layers: cfg.numLayers,
            layerTypes: cfg.layerTypes)
        block = blk
        let nameGaps = Self.nameGaps(blk, names: names, ffnKinds: ffnKinds, family: fam)
        let gaps = Self.unsupported(cfg, family: fam, shape: shape, router: router) + blockGaps + nameGaps
        if !gaps.isEmpty {
            throw SafetensorsError.format("\(cfg.modelType): not supported yet: " + gaps.joined(separator: "; "))
        }
    }

    /// RoPE of each layer (`ropes`): nil in the non-attention ones; `nope` where `no_rope_layers` marks 0, where
    /// `layer_rope_theta` is 0 (Muse Glimmer: mlx-lm's `use_rope = bool(layer_rope_theta[i])`, the other layers rotate
    /// with the global θ; another θ is a load error, `layerRopeThetaGaps`) and in every attention layer of Nemotron-H
    /// (mlx-lm and transformers rotate nothing there).
    static func layerRopes(_ cfg: ModelConfig, base: ModelShape, full: ModelShape?) -> [RopeSpec?] {
        let t = cfg.text
        let noRope = (t["no_rope_layers"] as? [NSNumber])?.map(\.intValue)
        let layerTheta = layerRopeThetas(t)
        return cfg.layerTypes.enumerated().map { i, ty in
            guard attentionLayerTypes.contains(ty) else { return nil }
            let nope =
                cfg.family?.traits.noRope == true || (noRope.map { i < $0.count && $0[i] == 0 } ?? false)
                || (layerTheta.map { i < $0.count && $0[i] == 0 } ?? false)
            // Each layer rotates the dimensions of its shape (Gemma 4: the whole 512 head in the full layers).
            let rot = ty == "full_attention" ? full?.attnRot ?? base.attnRot : base.attnRot
            return RopeSpec(
                theta: RopeSpec.theta(t, layerType: ty), rotaryDims: rot, nope: nope,
                scaling: RopeSpec.scaling(t, layerType: ty, family: cfg.family))
        }
    }

    /// Shared expert: LFM2, gpt-oss and Gemma 4, none; Qwen3.5/3.6 and Flash-Next, gated. Nemotron-H and GLM-4.7:
    /// ungated if it has `n_shared_experts` (as mlx-lm; of n × the experts' intermediate size), else none.
    static func sharedKind(_ cfg: ModelConfig) -> RouterSpec.Shared {
        let shared = cfg.family?.traits.sharedExpert ?? .none
        return shared == .ungated && (cfg.text["n_shared_experts"] as? NSNumber) == nil ? .none : shared
    }

    /// FFN of each layer of a MoE: dense are the first `num_dense_layers` (LFM2) and the
    /// first `first_k_dense_replace` (GLM). Without MoE, all dense. `mlp_only_layers` and `decoder_sparse_step` (Qwen2/
    /// Qwen3 MoE) are a load error unless empty or 1 (`unsupported`). Gemma 4's MoE (26B-A4B): dense and MoE in parallel
    /// in every layer.
    static func ffnKinds(_ cfg: ModelConfig) -> [FFNKind] {
        let t = cfg.text
        // Nemotron-H: the MoE blocks (E) are MoE, the MLP ones (-) dense, and Mamba2 (M) and attention (*) have none.
        if cfg.family == .nemotronH {
            return (ModelConfig.nemotronBlocks(t) ?? []).map { $0 == "E" ? .moe : $0 == "-" ? .dense : .none }
        }
        func int(_ k: String) -> Int? { (t[k] as? NSNumber)?.intValue }
        if cfg.family == .gemma4 && cfg.isMoE { return Array(repeating: .denseAndMoE, count: cfg.numLayers) }
        let numDense = max(int("num_dense_layers") ?? 0, int("first_k_dense_replace") ?? 0)
        return (0..<cfg.numLayers).map { i in cfg.isMoE && i >= numDense ? .moe : .dense }
    }

    /// Per-head gate in the q row ([q | gate], LAB_AT_GATE): Qwen3.5/3.6 and Flash-Next carry it in q_proj, Muse Glimmer
    /// in its own gate_proj (interleaved at load time, WeightNames.attnGate), LFM3 a scalar per head in its own gate_proj
    /// (repeated over the head at load time); LFM2 and Qwen3 have none.
    static func attnGated(_ family: ModelFamily?) -> Bool { family?.traits.attnGated ?? true }

    /// The family's router linear and experts carry biases (FamilyTraits.moeBiases).
    static func moeBiases(_ family: ModelFamily?) -> Bool { family?.traits.moeBiases ?? false }

    /// Weight names of a family.
    static func weightNames(_ family: ModelFamily) -> WeightNames { family.traits.names }

    /// Some attention layer of the config has a scaled RoPE (the `hasRopeTable` of its spec; LAB_AT_ROPETAB).
    static func hasScaledRope(_ cfg: ModelConfig) -> Bool {
        Set(cfg.layerTypes).intersection(attentionLayerTypes).contains {
            RopeSpec.scaling(cfg.text, layerType: $0, family: cfg.family) != nil
        }
    }

    /// Router settings of the config, each with the family's value as the default: `scoring_func` (GLM), the selection
    /// bias (`use_expert_bias` in LFM2; `topk_method: noaux_tc` in GLM), `routed_scaling_factor` and the
    /// renormalization epsilon (by family: only LFM2 has one). Nemotron-H has none of those keys: sigmoid with the
    /// `e_score_correction_bias` selection bias by family. GLM-4.7-Flash's config has no `scoring_func` either: sigmoid
    /// by family (mlx-lm glm4_moe_lite's default), and `noaux_tc` (also mlx-lm's default) turns the selection bias on
    /// (its 1e-20 epsilon is 0 here, RouterSpec.normEps). An unknown `scoring_func` keeps the family's scoring here and
    /// fails in `unsupported`.
    static func routerKeys(_ t: [String: Any], family: ModelFamily?) -> (
        scoring: RouterSpec.Scoring, selectBias: Bool, scale: Float, normEps: Float
    ) {
        let scoring: RouterSpec.Scoring
        switch scoringName(t) {
        case "sigmoid": scoring = .sigmoid
        case "softmax": scoring = .softmax
        default: scoring = family?.traits.sigmoidRouter == true ? .sigmoid : .softmax
        }
        let bias =
            (t["use_expert_bias"] as? Bool) ?? (t["topk_method"] as? String).map { $0 == "noaux_tc" }
            ?? (family?.traits.sigmoidRouter == true)
        let scale = (t["routed_scaling_factor"] as? NSNumber)?.floatValue ?? 1
        return (scoring, bias, scale, family?.traits.routerNormEps ?? 0)
    }

    /// The router's `scoring_func` (nil if the config has none).
    static func scoringName(_ t: [String: Any]) -> String? {
        t["scoring_func"] as? String
    }

    /// `partial_rotary_factor` if the config lacks it: 0.25 in Qwen3.5/3.6 (their config.json has it anyway); in the
    /// rest, RoPE rotates the whole head (before, 0.25 in all, and a model with head 128 and full RoPE
    /// would have rotated 32 dimensions silently).
    static func defaultPartialRotary(_ family: ModelFamily?) -> Double { family?.traits.partialRotary ?? 1 }

    /// Learned attention sinks: `softmax_type: learnable` (Flint's key) or the gpt-oss family, which has them in every
    /// layer without declaring them.
    static func attnSinks(_ cfg: ModelConfig) -> Bool {
        cfg.family?.traits.attnSinks == true || (cfg.text["softmax_type"] as? String) == "learnable"
    }

    /// Attention layer types (`layer_types`): full and sliding-window.
    public static let attentionLayerTypes: Set<String> = ["full_attention", "sliding_attention"]
    /// Every layer type the engine runs: attention, GDN (Qwen3.5/3.6, Flash-Next) and short conv (LFM2), and those it
    /// recognizes with their own load error (`nemotronHGaps`: Nemotron-H's Mamba2 and mixer-less layers).
    static let knownLayerTypes: Set<String> = attentionLayerTypes.union(["linear_attention", "conv", "mamba", "ffn"])
}

/// Kernel compile-time constants of a model shape (today, ModelDims).
public typealias ModelShape = ModelDims
