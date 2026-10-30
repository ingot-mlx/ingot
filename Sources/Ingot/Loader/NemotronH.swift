import Foundation

// Nemotron-H (nemotron_h) in the loader: its layer pattern, weight names and the named load errors of what the
// engine does not run. Kept apart from ModelSpec.swift and Config.swift, which only call into it.

extension ModelConfig {
    /// Nemotron-H's block of each layer as mlx-lm's nemotron_h reads it: `hybrid_override_pattern` (a string or a list
    /// of `M`, `*`, `-`, `E`), else `layers_block_type` or `layer_types` with their names (`mamba`/`linear_attention`/
    /// `conv` → M, `attention`/`full_attention` → *, `mlp` → -, `moe` → E). nil without any of them or with an unknown
    /// block.
    static func nemotronBlocks(_ t: [String: Any]) -> [Character]? {
        let chars: [Character]?
        if let s = t["hybrid_override_pattern"] as? String {
            chars = Array(s)
        } else if let l = t["hybrid_override_pattern"] as? [String] {
            chars = l.count == l.filter({ $0.count == 1 }).count ? l.map { Character($0) } : nil
        } else if let l = (t["layers_block_type"] as? [String]) ?? (t["layer_types"] as? [String]) {
            let map: [String: Character] = [
                "mamba": "M", "linear_attention": "M", "conv": "M", "attention": "*", "full_attention": "*",
                "mlp": "-", "moe": "E",
            ]
            let c = l.compactMap { map[$0] }
            chars = c.count == l.count ? c : nil
        } else {
            chars = nil
        }
        guard let chars, !chars.isEmpty, chars.allSatisfy({ "M*-E".contains($0) }) else { return nil }
        return chars
    }

    /// Layer type of a Nemotron-H block (`nemotronBlocks`).
    static func nemotronLayerType(_ c: Character) -> String {
        switch c {
        case "M": "mamba"
        case "*": "full_attention"
        default: "ffn"
        }
    }
}

extension WeightNames {
    /// Nemotron-H: one `norm.weight` per layer (the norm before its only block, which is the mixer norm or the FFN norm),
    /// every block under `mixer.` (attention projections, MLP and MoE), experts `switch_mlp.fc1` (up) / `fc2` (down)
    /// without gate, `shared_experts.` and the root's `embeddings` and `norm_f`.
    static let nemotronH: WeightNames = {
        var n = WeightNames(
            ffn: "mixer.", oProj: "o_proj", qNorm: "q_norm.weight", kNorm: "k_norm.weight", inputNorm: "norm.weight",
            postNorm: "norm.weight", finalNorm: "norm_f.weight", experts: "switch_mlp.")
        n.attn = "mixer."
        n.embedding = "embeddings"
        n.sharedExpert = "shared_experts."
        n.expertGate = nil
        n.expertUp = "fc1"
        n.expertDown = "fc2"
        return n
    }()
}

extension ModelSpec {
    /// Nemotron-H pieces the engine does not run: those left out of the supported scope (LatentMoE, grouped routing, a `dt` limit,
    /// Mamba2 biases or activations, experts at 8 bits) and the shapes its kernels do not take. Mamba2 and single-block
    /// layers run, and so do relu² and the MoE. The Mamba2 and mixer-less layer types outside nemotron_h
    /// are errors by key; the rest only for nemotron_h. The MTP head
    /// (`num_nextn_predict_layers`, `mtp.` tensors) is not an error: the loader skips it as mlx-lm's sanitize does.
    static func nemotronHGaps(_ cfg: ModelConfig) -> [String] {
        let t = cfg.text
        func int(_ k: String) -> Int? { (t[k] as? NSNumber)?.intValue }
        var out: [String] = []
        let kinds = ffnKinds(cfg)
        // Mamba2 and single-block layers (a mixer without FFN, or an FFN without mixer) run, in
        // nemotron_h only (ModelDims.ssm and its weight names).
        guard cfg.family == .nemotronH else {
            if cfg.layerTypes.contains("mamba") { out.append("Mamba2 mixer outside nemotron_h") }
            if cfg.layerTypes.contains("ffn") || kinds.contains(.none) {
                out.append("single-block layers outside nemotron_h")
            }
            return out
        }
        guard let blocks = ModelConfig.nemotronBlocks(t) else {
            return out + [
                "nemotron_h without a readable layer pattern (hybrid_override_pattern, layers_block_type or "
                    + "layer_types with M, *, -, E)"
            ]
        }
        if blocks.count != cfg.numLayers {
            out.append("nemotron_h pattern of \(blocks.count) layers with num_hidden_layers \(cfg.numLayers)")
        }
        // Mamba2 as the GDN kernels run it: state of 128, heads a multiple of 16 wide, whole groups of heads,
        // a conv of 4 with bias, SiLU, no projection biases and mlx-lm's (0, ∞) dt limit.
        if blocks.contains("M") {
            let h = int("mamba_num_heads") ?? 0, p = int("mamba_head_dim") ?? 0, g = int("n_groups") ?? 0
            let n = int("ssm_state_size") ?? 0
            if n != 128 || p <= 0 || p % 16 != 0 || g <= 0 || h <= 0 || h % g != 0 {
                out.append(
                    "Mamba2 of \(h) heads × \(p), \(g) groups, state \(n) (state 128, head a multiple of 16 and whole groups)"
                )
            }
            if (int("conv_kernel") ?? 4) != 4 { out.append("Mamba2 conv of \(int("conv_kernel") ?? 0) (only 4)") }
            if (t["use_conv_bias"] as? Bool) == false { out.append("Mamba2 conv without bias (use_conv_bias false)") }
            if (t["mamba_proj_bias"] as? Bool) == true { out.append("Mamba2 projection biases (mamba_proj_bias)") }
            if let a = t["mamba_hidden_act"] as? String, a != "silu" {
                out.append("Mamba2 activation \(a) (only silu)")
            }
            if let l = t["time_step_limit"] as? [NSNumber],
                l.count != 2 || l[0].doubleValue != 0 || l[1].doubleValue.isFinite
            {
                out.append("Mamba2 time_step_limit \(l.map(\.doubleValue)) (only mlx-lm's (0, ∞))")
            }
        }
        if let l = int("moe_latent_size") {
            out.append("LatentMoE (moe_latent_size \(l): Nemotron 3.5 Super/Bolt, not supported)")
        }
        if let g = int("n_group"), g > 1 { out.append("grouped expert routing (n_group \(g), not supported)") }
        // OptiQ conversions put some experts at 8 bits: the MoE kernels only read 4-bit experts.
        let ex = ".\(WeightNames.nemotronH.ffn)\(WeightNames.nemotronH.experts)"
        let wide = cfg.quantOverrides.filter { $0.key.contains(ex) && $0.value.bits != 4 }.map(\.value.bits)
        if let b = wide.max() {
            out.append("experts at \(b) bits (the MoE kernels read 4-bit experts only: OptiQ, not supported)")
        }
        return out
    }
}
