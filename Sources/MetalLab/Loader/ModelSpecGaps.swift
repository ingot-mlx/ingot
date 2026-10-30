import Foundation

// Load checks of a ModelSpec: what a config asks for and the engine cannot run yet is a load error with its name
// (before, several things were silently ignored and the model would have come out wrong).

extension ModelSpec {
    /// Guards for what the GDN kernels assume: 16 k heads of 128 and v of 128.
    static func gdnGuard(_ cfg: ModelConfig, shape: ModelShape) throws {
        guard cfg.layerTypes.contains("linear_attention") else { return }
        let dv = shape.gdnDV
        guard shape.gdnHK == 16, shape.gdnDK == 128, dv == 128, shape.gdnHV % 16 == 0 else {
            throw SafetensorsError.format(
                "GDN not supported: \(shape.gdnHK) k heads of \(shape.gdnDK), v of \(dv) × \(shape.gdnHV) (the "
                    + "kernels assume 16 × 128 and v of 128)")
        }
    }

    /// Block pieces the family's weight names or kernels cannot load.
    static func nameGaps(_ blk: BlockSpec, names: WeightNames, ffnKinds: [FFNKind], family fam: ModelFamily)
        -> [String]
    {
        var nameGaps: [String] = []
        if blk.mixerPostNorm.map({ $0.weight != .unit }) == true && names.mixerPostNorm == nil
            || blk.ffnPostNorm.map({ $0.weight != .unit }) == true && names.ffnPostNorm == nil
        {
            nameGaps.append("sandwich post-norms without weight names in family \(fam.rawValue)")
        }
        if blk.embedNorm.map({ $0.weight != .unit }) == true && names.embedNorm == nil {
            nameGaps.append("embedding norm without a weight name in family \(fam.rawValue)")
        }
        // swiglu_oai only has MoE instances (`_oai` of the gate+up kernels): a dense FFN with it would not find its kernel.
        if case .swigluOAI = blk.activation, ffnKinds.contains(.dense) {
            nameGaps.append("swiglu_oai in a dense FFN (only the MoE kernels have it)")
        }
        // relu² has no gate: only Nemotron-H's weight names (`up_proj`/`down_proj`, experts `fc1`/`fc2`) load it.
        if blk.activation == .relu2 && fam != .nemotronH {
            nameGaps.append("relu² FFN without gate outside nemotron_h")
        }
        return nameGaps
    }

    /// RoPE the kernels cannot run: a scaling type other than default, linear and yarn (flat or per layer
    /// type; `rope_scaling: null` is the default), a rotated width per layer type, or a scaled RoPE in Flash-Next.
    static func ropeGaps(_ t: [String: Any], family: ModelFamily, shape: ModelShape) -> [String] {
        // RoPE: default, linear and yarn, flat or per layer type; `rope_scaling: null` is the default.
        let ropeDicts = RopeSpec.allParameters(t) + [t["rope_scaling"]].compactMap { $0 as? [String: Any] }
        var gaps = Set(ropeDicts.compactMap(RopeScaling.gap)).sorted()
        // One rotated width for the whole model (LAB_AT_ROT): a partial_rotary_factor per layer type would pair other
        // dimensions. Gemma 4's (0.25 in the full layers, 1 in the sliding ones) belongs to its proportional RoPE, which
        // rotates the whole head of its layers with zero frequencies past the factor (RopeTable), so it is not a gap.
        // It is only modeled with Gemma 4's per-layer shapes (the partial factor is applied by the table).
        let proportional = ropeDicts.contains { RopeScaling.type($0) == "proportional" }
        if proportional && family != .gemma4 { gaps.append("proportional RoPE outside Gemma 4") }
        let partials = Set(
            RopeSpec.allParameters(t).compactMap { ($0["partial_rotary_factor"] as? NSNumber)?.doubleValue }.map {
                Int((Double(shape.attnHD) * $0).rounded())
            })
        let nested = (t["rope_parameters"] as? [String: Any]).map(RopeSpec.isNested) ?? false
        // LFM3's partial factor is its zero-tail RoPE over the whole head (lfm3ZeroTail), not a narrower rotation.
        if nested && !proportional && family != .lfm3 && partials.contains(where: { $0 != shape.attnRot }) {
            gaps.append("partial_rotary_factor per layer type (the kernels rotate \(shape.attnRot) dimensions in all)")
        }
        let scaled = ropeDicts.contains { RopeScaling.type($0) != "default" }
        if scaled && family == .flashNext {
            gaps.append("scaled RoPE in Flash-Next (FlashParts and the QSA indexer use θ only)")
        }
        return gaps
    }

    /// What the config asks for and the engine cannot run yet (before, several things were silently ignored and the
    /// model would have come out wrong). Empty if it can be loaded.
    static func unsupported(_ cfg: ModelConfig, family: ModelFamily, shape: ModelShape, router: RouterSpec?)
        -> [String]
    {
        let t = cfg.text
        var out: [String] = []
        out += ropeGaps(t, family: family, shape: shape)
        out += layerRopeThetaGaps(cfg)
        for ty in Set(cfg.layerTypes).subtracting(knownLayerTypes).sorted() {
            out.append("layers of type «\(ty)»")
        }
        if cfg.layerTypes.contains("sliding_attention") {
            // Window mask in pattn_i8 (prefill, verification and decode) over the linear KV.
            let w = (t["sliding_window"] as? NSNumber)?.intValue ?? 0
            if w <= 0 { out.append("sliding_attention without a positive sliding_window") }
            if family == .flashNext { out.append("sliding_attention in Flash-Next (its attention is FlashParts)") }
        }
        out += gemma4Gaps(cfg)
        out += quantModeGaps(cfg)
        out += nemotronHGaps(cfg)
        out += glmGaps(cfg)
        out += lfm3Gaps(cfg)
        if let st = t["softmax_type"] as? String, st != "learnable" && st != "vanilla" {
            out.append("softmax_type «\(st)»")
        }
        if attnSinks(cfg) && family == .flashNext { out.append("attention sinks in Flash-Next (FlashParts)") }
        if let mlp = t["mlp_layer_types"] as? [String], mlp.contains(where: { $0 != "sparse" }) {
            out.append("mlp_layer_types other than sparse")
        }
        if cfg.layerTypes.contains(where: attentionLayerTypes.contains) {
            // verify_attn*.metal: head of 64, 128 or 256, with or without gate (before, 256 with gate and 64 without).
            // pattn/dattn_i8 split P·V across 4 simdgroups of AT_HD / 4 dimensions (fragments of 16).
            // 576: only MLA's latent head ([k_pe 64 | c 512], glmGaps checks its split).
            if ![64, 128, 256, 512].contains(shape.attnHD) && !(shape.attnHD == 576 && family == .glm4MoeLite) {
                out.append(
                    "attention with head of \(shape.attnHD) (the attention kernels accept heads of 64, 128, 256 and "
                        + "512, and 576 for MLA's latent cache)")
            }
        }
        out += vocabGaps(cfg)
        // Dense layers in a MoE: ffnKinds reads them from num_dense_layers and first_k_dense_replace. A MoE with no
        // MoE layer makes no sense (and PrefillMoE would not be created).
        if cfg.isMoE && !ffnKinds(cfg).contains(where: { $0 == .moe || $0 == .denseAndMoE }) {
            out.append("MoE without any MoE layer (num_dense_layers or first_k_dense_replace)")
        }
        // Dense layers between MoE ones (Qwen2/Qwen3 MoE): no tier-1 family uses them (Qwen3.5 declares them empty
        // and mlx-lm's qwen3_5 ignores them); removed.
        if cfg.isMoE, let only = t["mlp_only_layers"] as? [NSNumber], !only.isEmpty {
            out.append("mlp_only_layers (dense layers between MoE ones)")
        }
        if cfg.isMoE, let step = (t["decoder_sparse_step"] as? NSNumber)?.intValue, step != 1 {
            out.append("decoder_sparse_step \(step) (dense layers between MoE ones)")
        }
        // Router keys the kernels do not run yet: grouped top-k (DeepSeek V3, Kimi, Ling; no planned family
        // needs it), router logit softcapping, other top-k methods and scoring functions.
        if let s = scoringName(t), s != "sigmoid" && s != "softmax" { out.append("router scoring_func «\(s)»") }
        let nGroup = (t["n_group"] as? NSNumber)?.intValue ?? 1
        let topkGroup = (t["topk_group"] as? NSNumber)?.intValue ?? nGroup
        if nGroup > 1 && topkGroup < nGroup {
            out.append("grouped top-k routing (n_group \(nGroup), topk_group \(topkGroup))")
        }
        if let v = (t["router_logit_softcapping"] as? NSNumber)?.doubleValue, v > 0 {
            out.append("router logit softcapping (router_logit_softcapping)")
        }
        if let m = t["topk_method"] as? String, m != "noaux_tc" && m != "greedy" { out.append("topk_method «\(m)»") }
        out += routerPresenceGaps(cfg, family: family, router: router)
        guard let r = router else { return out }
        if r.scoring == .softmax && r.selectBias { out.append("selection bias with a softmax router") }
        if family == .flashNext && (r.scoring != .softmax || r.scale != 1) {
            out.append("Flash-Next with a sigmoid or scaled router (its kernels are FlashParts)")
        }
        // norm_topk_prob = false (weights of the full softmax or the sigmoid p, without renormalizing the top-k): no
        // tier-1 family uses it and the kernels always renormalize (removed). Without the key, every tier-1
        // family renormalizes.
        if !((t["norm_topk_prob"] as? Bool) ?? true) {
            out.append("norm_topk_prob = false (the router kernels always renormalize the top-k weights)")
        }
        // Shared expert of its own width: multiple of 64 (column tiles of the round's gate+up, of
        // 16·nf, and of the prefill, of 64, and K steps of 64 of the down). Flash-Next has its own kernels (FN_I): same
        if r.shared != .none && (r.sharedInter <= 0 || r.sharedInter % 64 != 0) {
            out.append(
                "shared expert of \(r.sharedInter) (an intermediate size multiple of 64 is needed: "
                    + "shared_expert_intermediate_size o n_shared_experts)")
        }
        if family == .flashNext && r.sharedInter != r.inter {
            out.append(
                "Flash-Next with shared expert of \(r.sharedInter) vs experts of \(r.inter) (FN_I is a single value)")
        }
        do {
            _ = try MoEKernelSet.forRouter(r, inter: r.inter)
        } catch SafetensorsError.format(let m) {
            out.append(m)
        } catch {
            out.append("\(error)")
        }
        // MoELayer (not Flash-Next, which has its own kernels): hidden multiple of 64 and at least 128. vnax_moe_gateup
        // and the vmoe_route gate walk hidden in groups of 64, vnax_moe_down_split in blocks of 16·nf, and the prefill MoE
        // down (moe_down2) in tiles of 128 columns, the last one shifted back to end at hidden when it is not a multiple
        // of 128 (LAB_MOE_H64, gpt-oss's 2880). KernelCaseTests moe-shared checks hidden 2816 (Gemma 4: 22 × 128) and
        // moe-gptoss hidden 576 and 2880.
        if family != .flashNext && (shape.hidden % 64 != 0 || shape.hidden < 128) {
            out.append(
                "MoE with hidden \(shape.hidden) (a multiple of 64 and at least 128 is needed: the prefill MoE down runs "
                    + "in tiles of 128 hidden columns)")
        }
        return out
    }

    /// The lm_head and the embedding table run through the qmv and prefill GEMM instances, which need N % 64 = 0.
    /// Every tier-1 vocabulary is a multiple of 64, so there is no zero-row padding for the others (removed in
    /// the vocabulary rule); Model.init checks the stored table too.
    static func vocabGap(_ vocab: Int) -> String {
        "vocabulary \(vocab) not a multiple of 64 (the lm_head kernels need N % 64 = 0)"
    }
    static func vocabGaps(_ cfg: ModelConfig) -> [String] {
        cfg.vocabSize > 0 && cfg.vocabSize % 64 != 0 ? [vocabGap(cfg.vocabSize)] : []
    }

    /// A MoE always gets its router or an error (before, a MoE without top-k loaded with router nil).
    static func routerPresenceGaps(_ cfg: ModelConfig, family: ModelFamily, router: RouterSpec?) -> [String] {
        guard cfg.isMoE else { return [] }
        if cfg.expertsPerToken == nil { return ["MoE without a top-k (num_experts_per_tok or top_k_experts)"] }
        return router == nil ? ["MoE in family \(family.rawValue) (no router for it)"] : []
    }

    /// Known families the engine does not run: why (nil if it is not one of them). Dense Qwen3 (qwen3) was never
    /// registered; Qwen3 MoE (qwen3_moe) and Gemma 3 (gemma3_text) were tier 2 without a CI test and were removed.
    static func unregisteredFamilyGap(_ cfg: ModelConfig) -> String? {
        switch cfg.modelType {
        case "qwen3": "dense Qwen3 family: not registered in ModelFamily (untested with real weights)"
        case "qwen3_moe": "Qwen3 MoE family: removed (tier 2 without a CI test)"
        case "gemma3_text": "Gemma 3 family: removed (tier 2 without a CI test; Gemma 4 runs)"
        default: nil
        }
    }
}
