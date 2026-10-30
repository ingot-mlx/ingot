import Foundation

// Muse Glimmer in the loader: its weight names and the per-layer θ list it introduces (`layer_rope_theta`,
// read for any family: 0 = a layer without RoPE).

extension WeightNames {
    /// Muse Glimmer: Gemma's names (sandwich post-norms: `post_attention_layernorm` after the attention branch and
    /// `pre_feedforward_layernorm` before the FFN; no q_norm/k_norm tensors, its QK-norm is weightless) and the
    /// separate attention gate.
    static let muse: WeightNames = {
        var n = WeightNames(
            ffn: "mlp.", oProj: "o_proj", qNorm: "q_norm.weight", kNorm: "k_norm.weight",
            inputNorm: "input_layernorm.weight", postNorm: "pre_feedforward_layernorm.weight", finalNorm: "norm.weight",
            experts: "switch_mlp.")
        n.mixerPostNorm = "post_attention_layernorm.weight"
        n.ffnPostNorm = "post_feedforward_layernorm.weight"
        n.attnGate = "gate_proj"
        return n
    }()
}

extension ModelSpec {
    /// `layer_rope_theta` of a config (Muse Glimmer: one θ per layer, 0 = NoPE); nil without it.
    static func layerRopeThetas(_ t: [String: Any]) -> [Double]? {
        (t["layer_rope_theta"] as? [NSNumber])?.map(\.doubleValue)
    }

    /// `layer_rope_theta` the engine cannot run: a list that does not cover every layer, or an attention layer with a θ
    /// that is neither 0 (NoPE) nor its layer type's (mlx-lm and transformers rotate those with the global θ, so the
    /// list would say something the references ignore).
    static func layerRopeThetaGaps(_ cfg: ModelConfig) -> [String] {
        guard let lt = layerRopeThetas(cfg.text) else { return [] }
        if lt.count != cfg.numLayers {
            return ["layer_rope_theta with \(lt.count) values for \(cfg.numLayers) layers"]
        }
        let odd = cfg.layerTypes.enumerated().filter { i, ty in
            attentionLayerTypes.contains(ty) && lt[i] != 0 && Float(lt[i]) != RopeSpec.theta(cfg.text, layerType: ty)
        }
        guard let (i, ty) = odd.first else { return [] }
        return [
            "layer_rope_theta \(lt[i]) in layer \(i) other than 0 (NoPE) or the rope_theta of its type "
                + "(\(RopeSpec.theta(cfg.text, layerType: ty))), \(odd.count) layer(s) in all"
        ]
    }
}
