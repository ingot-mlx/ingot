import Foundation

// LFM3 (lfm3: LiquidAI/LFM3-1.7B) in the loader: weight names, the zero-tail partial RoPE and what does not run
// yet. Semantics per Liquid's modeling_lfm3.py (revision 40ebe0aa) and mlx-lm PR #2016 (lfm3.py, commit 9027ac5).

extension WeightNames {
    /// LFM3: LFM2's layer norms (`operator_norm`, `ffn_norm`, final `embedding_norm`) with Qwen's attention names
    /// (`o_proj`, `q_norm`, `k_norm`), the sandwich post-norms, the scalar per-head gate in `self_attn.gate_proj` and the
    /// dense FFN as `feed_forward.w1/w3/w2` (gate, up, down).
    static let lfm3: WeightNames = {
        var n = WeightNames(
            ffn: "feed_forward.", oProj: "o_proj", qNorm: "q_norm.weight", kNorm: "k_norm.weight",
            inputNorm: "operator_norm.weight", postNorm: "ffn_norm.weight", finalNorm: "embedding_norm.weight",
            experts: "switch_mlp.")
        n.mixerPostNorm = "post_operator_norm.weight"
        n.ffnPostNorm = "post_ffn_norm.weight"
        n.attnGate = "gate_proj"
        n.denseGate = "w1"
        n.denseUp = "w3"
        n.denseDown = "w2"
        return n
    }()
}

extension ModelSpec {
    /// Taps of the short convolutions the kernels run (`conv_L_cache`).
    static let convWidths: Set<Int> = [3, 7]

    /// LFM3's partial RoPE of a layer type: a default entry with `partial_rotary_factor` < 1 rotates the first
    /// `int(partial · head_dim) / 2` pairs of the whole head with θ^(-2i/head_dim) and leaves the rest at angle 0
    /// (`compute_default_rope_parameters`), which is the `proportional` table with factor 1. nil for a full-head RoPE.
    static func lfm3ZeroTail(_ t: [String: Any], layerType: String) -> RopeScaling? {
        guard let p = RopeSpec.parameters(t, layerType: layerType), RopeScaling.type(p) == "default",
            let partial = (p["partial_rotary_factor"] as? NSNumber)?.doubleValue, partial < 1
        else { return nil }
        var s = RopeScaling(kind: .proportional, factor: 1)
        s.partial = partial
        return s
    }

    /// What the engine cannot run of an LFM3 config yet (or ever, with no LFM3 asking for it).
    static func lfm3Gaps(_ cfg: ModelConfig) -> [String] {
        guard cfg.family == .lfm3 else { return [] }
        let t = cfg.text
        var out: [String] = []
        if cfg.layerTypes.contains("conv") {
            let width = (t["conv_L_cache"] as? NSNumber)?.intValue ?? 3
            if !convWidths.contains(width) {
                out.append("short convolution of width \(width) (the kernels run 3 and 7)")
            }
            if (t["conv_bias"] as? Bool) == true { out.append("short convolution with bias (conv_bias)") }
        }
        // A scaled RoPE with a partial factor: Lfm3 evaluates it over the whole head and truncates it to the rotated
        // prefix (`compute_rope_parameters`), which no table here reproduces and no LFM3 checkpoint uses.
        for ty in Set(cfg.layerTypes).intersection(attentionLayerTypes).sorted() {
            guard let p = RopeSpec.parameters(t, layerType: ty), RopeScaling.type(p) != "default",
                let partial = (p["partial_rotary_factor"] as? NSNumber)?.doubleValue, partial < 1
            else { continue }
            out.append("scaled RoPE «\(RopeScaling.type(p))» with partial_rotary_factor \(partial) in \(ty) layers")
        }
        return out
    }
}
