import Foundation
import Testing

@testable import Ingot

/// Scaled RoPE from the config: per layer type (Gemma 4 31B with `rope_parameters` nested as in transformers 5: YaRN in
/// the full layers), flat `rope_scaling` (gpt-oss style: every layer), LAB_AT_ROPETAB only when some layer is scaled,
/// and load errors for what the kernels cannot run. No GPU.
extension ModelSpecTests {
    /// YaRN with the rounded correction range (truncate): the yarn-truncate vector of ref/rope_ref.py (factor 16 over
    /// 8192, β 32/1, mscale 0.1 · ln 16 + 1 = 1.277, as mlx-lm computes it).
    static var yarnTruncate: [String: Any] {
        [
            "rope_type": "yarn", "factor": 16.0, "original_max_position_embeddings": 8192, "beta_fast": 32.0,
            "beta_slow": 1.0, "attention_factor": 1.2772588722239782,
        ]
    }

    /// Gemma 4 31B (layer types `[sliding ×5, full]`) with its full-layer `rope_parameters` entry replaced by θ 1e6 plus
    /// `full`'s keys (instead of the proportional RoPE); the sliding layers keep the default RoPE over θ 1e4. The full
    /// layers lose their own shape (`global_head_dim`, K = V), which only runs with the proportional RoPE: every layer
    /// has a head of 256, as the yarn-truncate vector of ref/rope_ref.py.
    static func gemma4Nested(full: [String: Any]) throws -> ModelConfig {
        try gemma4("gemma4-31b") { t in
            for k in ["global_head_dim", "attention_k_eq_v", "num_global_key_value_heads"] { t[k] = nil }
            var f: [String: Any] = ["rope_type": "default", "rope_theta": 1_000_000.0]
            f.merge(full) { _, b in b }
            t["rope_parameters"] = [
                "full_attention": f, "sliding_attention": ["rope_type": "default", "rope_theta": 10_000.0],
            ]
        }
    }

    @Test func scaledRopePerLayer() throws {
        let cfg = try Self.gemma4Nested(full: Self.yarnTruncate)
        #expect(Self.loadError(cfg) == nil)
        let m = try ModelSpec(cfg)
        #expect(m.hasRopeTable && m.shape.attnRopeTab && m.shape.macros["LAB_AT_ROPETAB"] as? NSNumber == 1)
        let yarn = try #require(m.ropes[5]?.scaling)
        #expect(yarn.kind == .yarn && yarn.factor == 16 && yarn.originalMax == 8192 && yarn.truncate)
        #expect(yarn.attentionFactor == 1.2772588722239782)
        #expect(m.ropes.enumerated().allSatisfy { i, r in (r?.scaling != nil) == (i % 6 == 5) })
        #expect(m.ropes.enumerated().allSatisfy { i, r in r?.theta == (i % 6 == 5 ? 1e6 : 1e4) }, "nested θ")
        #expect(m.rope.scaling == yarn, "model RoPE = the full-attention one")
        // The full layer's table: mscale last, low pairs at the original frequency, high ones divided by the factor.
        let full = try #require(m.ropes[5])
        #expect(full.rotaryDims == 256 && m.fullShape == nil)
        let pairs = full.rotaryDims / 2
        let tab = full.table
        #expect(tab.count == pairs + 1 && tab[pairs] == Float(1.2772588722239782) && tab[0] == 1)
        let plain = RopeTable.make(theta: 1e6, rotaryDims: full.rotaryDims, scaling: nil)
        #expect(abs(tab[pairs - 1] * 16 / plain[pairs - 1] - 1) < 1e-6)
        let sliding = try #require(m.ropes[0])
        #expect(sliding.table.count == sliding.rotaryDims / 2 + 1 && sliding.table[sliding.rotaryDims / 2] == 1)
        // Without scaling: no macro (same library as before), tables unused.
        #expect(try ModelSpec(Self.gemma4Nested(full: [:])).shape.macros["LAB_AT_ROPETAB"] == nil)
        #expect(try ModelSpec(Self.heretic()).shape.macros["LAB_AT_ROPETAB"] == nil)
        // MetalContext checks ctx.dims == ModelDims(cfg): the macro must come from the config alone too.
        #expect(ModelDims(cfg).attnRopeTab)
    }

    @Test func flatRopeScalingAppliesToEveryLayer() throws {
        let gptoss: [String: Any] = [
            "rope_type": "yarn", "factor": 32.0, "original_max_position_embeddings": 4096, "beta_fast": 32.0,
            "beta_slow": 1.0, "truncate": false,
        ]
        let s = try ModelSpec(Self.granite(["rope_scaling": gptoss]))
        #expect(s.ropes.allSatisfy { $0?.scaling?.kind == .yarn && $0?.scaling?.truncate == false })
        let lin = try ModelSpec(Self.granite(["rope_scaling": ["type": "linear", "factor": 2.0]]))
        #expect(lin.ropes.allSatisfy { $0?.scaling?.kind == .linear } && lin.shape.attnRopeTab)
    }

    @Test func unsupportedRopeFails() throws {
        let bad: [([String: Any], String)] = [
            (["rope_scaling": ["rope_type": "dynamic", "factor": 2.0]], "dynamic"),
            (["rope_scaling": ["rope_type": "longrope", "short_factor": [1.0], "long_factor": [1.0]]], "longrope"),
            (["rope_scaling": ["rope_type": "proportional"]], "proportional"),
            (["rope_scaling": ["rope_type": "yarn", "factor": 4.0]], "original_max_position_embeddings"),
            (
                [
                    "rope_scaling": [
                        "rope_type": "yarn", "factor": 40.0, "original_max_position_embeddings": 4096,
                        "mscale_all_dim": 1.0,
                    ]
                ], "mscale_all_dim"
            ),
        ]
        for (extra, what) in bad {
            #expect(Self.loadError(try Self.granite(extra))?.contains(what) == true, "\(what)")
        }
        // A rotated width per layer type with the default RoPE (partial_rotary_factor 0.25 in the full layers only).
        let partial = try Self.gemma4Nested(full: ["partial_rotary_factor": 0.25])
        #expect(Self.loadError(partial)?.contains("partial_rotary_factor per layer type") == true)
        // Flash-Next: its attention and QSA indexer rotate with θ only.
        var fn = try Self.flashNext().raw
        var ft = (fn["text_config"] as? [String: Any]) ?? fn
        ft["rope_scaling"] = ["rope_type": "linear", "factor": 2.0]
        if fn["text_config"] != nil { fn["text_config"] = ft } else { fn = ft }
        #expect(Self.loadError(try ModelConfig(json: fn))?.contains("Flash-Next") == true)
    }
}
