import Foundation
import IngotBench
import Testing

@testable import Ingot

/// Nemotron-H in the loader, from the configs of mlx-community's NVIDIA-Nemotron-3-Nano-4B-4bit and
/// NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit in ref/toy/real/ (no weights, no GPU): the family is recognized, its layer
/// pattern, MoE, Mamba2 shape and attention without RoPE read as mlx-lm's nemotron_h, and what stays out of the supported scope
/// (LatentMoE, grouped routing, OptiQ's 8-bit experts, MXFP4) is a named load error. Mamba2 and the single-block layers
/// run, and so do relu² and the Nemotron MoE.
extension ModelSpecTests {

    /// A real Nemotron-H config (ref/toy/real/), edited, through JSON as the loader reads it.
    static func nemotron(_ name: String, _ edit: (inout [String: Any]) -> Void = { _ in }) throws -> ModelConfig {
        var j = try ToyReal.load(name, root: repo).json
        edit(&j)
        let data = try JSONSerialization.data(withJSONObject: j)
        return try ModelConfig(json: try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any]))
    }
    static func lightning(_ edit: (inout [String: Any]) -> Void = { _ in }) throws -> ModelConfig {
        try nemotron("nemotron-3.5-lightning-30b-a3b", edit)
    }
    static func nano(_ edit: (inout [String: Any]) -> Void = { _ in }) throws -> ModelConfig {
        try nemotron("nemotron-3-nano-4b", edit)
    }

    /// The Lightning: 52 layers (23 Mamba2, 6 attention, 23 MoE), hidden 2688, Mamba2 64 × 64 in 8 groups of state 128,
    /// attention 32 q / 2 kv of 128 without RoPE, 128 experts top-6 of 1856 with an ungated shared expert of 3712.
    @Test func nemotronLightning() throws {
        let c = try Self.lightning()
        #expect(c.family == .nemotronH && c.modelType == "nemotron_h" && c.spec != nil)
        #expect(c.hiddenSize == 2688 && c.numLayers == 52 && c.vocabSize == 131_072 && c.rmsNormEps == 1e-5)
        let count = { (ty: String) in c.layerTypes.filter { $0 == ty }.count }
        #expect(c.layerTypes.count == 52 && count("mamba") == 23 && count("full_attention") == 6 && count("ffn") == 23)
        #expect(Array(c.layerTypes.prefix(7)) == ["mamba", "ffn", "mamba", "ffn", "mamba", "full_attention", "ffn"])
        let kinds = ModelSpec.ffnKinds(c)
        #expect(kinds.filter { $0 == .moe }.count == 23 && !kinds.contains(.dense))
        #expect(zip(c.layerTypes, kinds).allSatisfy { ($0 == "ffn") == ($1 != .none) })
        // MoE keys of the family: n_routed_experts, moe_shared_expert_intermediate_size; the router by family
        // (sigmoid, selection bias, no epsilon) with the config's scale, and the ungated shared expert.
        #expect(c.isMoE && c.numExperts == 128 && c.expertsPerToken == 6 && c.moeIntermediateSize == 1856)
        #expect(c.sharedExpertIntermediateSize == 3712 && ModelSpec.sharedKind(c) == .ungated)
        let rk = ModelSpec.routerKeys(c.text, family: .nemotronH)
        #expect(rk.scoring == .sigmoid && rk.selectBias && rk.scale == 2.5 && rk.normEps == 0)
        let d = ModelDims(c)
        #expect(d.moeE == 128 && d.moeK == 6 && d.moeI == 1856 && d.moeShared == 2 && d.moeSigmoid && d.moeBias)
        #expect(d.moeRScale == 2.5 && d.moeEps == 0 && !d.moeH64 && !d.moeLBias)
        // relu² without gate: the `_relu2` instances (LAB_FFN_RELU2) for the experts and the shared expert.
        #expect(d.ffnRelu2 && d.ffnActSuffix == "_relu2" && d.macros["LAB_FFN_RELU2"] == NSNumber(value: 1))
        #expect(try c.requiredSpec.block.activation == .relu2 && c.spec?.router?.shared == .ungated)
        // Attention: GQA 16 with heads of 128, no gate, no QK-norm, no RoPE in any layer.
        #expect(d.attnQH == 32 && d.attnKVH == 2 && d.attnHD == 128 && !d.attnGate && !d.attnQKN && !d.attnSWA)
        let ropes = ModelSpec.layerRopes(c, base: d, full: nil)
        let attn = ropes.compactMap { $0 }
        #expect(attn.count == 6 && attn.allSatisfy { $0.nope })
        #expect(zip(c.layerTypes, ropes).allSatisfy { ($0 == "full_attention") == ($1 != nil) })
        // Mamba2 as a GDN shape: in_proj [10304, 2688] = z 4096 | xBC 6144 | dt 64.
        #expect(d.ssm && d.gdnHV == 64 && d.gdnHK == 8 && d.gdnDK == 128 && d.gdnDV == 64)
        #expect(d.gdnVDim == 4096 && d.gdnConvDim == 6144 && d.gdnProjN == 10304)
        #expect(d.macros["LAB_SSM"] == NSNumber(value: 1) && d.macros["LAB_GDN_DV"] == NSNumber(value: 64))
        // No gaps: Mamba2 and single-block layers run, and so do relu² and the MoE.
        #expect(Self.gapList(c) == nil)
        // Plan: KV of the 6 attention layers only, and the Mamba2 state of 23 layers (H·P·N fp32 + 3 conv rows).
        let fp = ModelFootprint(
            name: "lightning", config: c, weightBytes: 0, largestTensorBytes: 0, drafterConfig: nil,
            drafterWeightBytes: nil)
        #expect(fp.kvBytesPerToken == 6 * 2 * 2 * (128 + 4) + 8 && fp.kvBytesPerTokenPerLayer == 2 * 2 * 132)
        #expect(fp.snapshotBaseBytes == 23 * (64 * 64 * 128 * 4 + 3 * 6144 * 4) && fp.nativeContext == 262_144)
        #expect(fp.calibration == .nemotronH && !fp.hasDrafter && fp.windowLayers == 0)
    }

    /// The Nano 4B: 42 layers (21 Mamba2, 4 attention, 17 relu² MLP, no MoE), hidden 3136, Mamba2 96 × 80 in 8 groups,
    /// attention 40 q / 8 kv of 128 without RoPE.
    @Test func nemotronNano() throws {
        let c = try Self.nano()
        #expect(c.family == .nemotronH && c.spec != nil && !c.isMoE)
        #expect(c.hiddenSize == 3136 && c.numLayers == 42 && c.vocabSize == 131_072 && c.rmsNormEps == 1e-5)
        let count = { (ty: String) in c.layerTypes.filter { $0 == ty }.count }
        #expect(count("mamba") == 21 && count("full_attention") == 4 && count("ffn") == 17)
        #expect(Array(c.layerTypes.prefix(3)) == ["mamba", "ffn", "mamba"])
        let kinds = ModelSpec.ffnKinds(c)
        #expect(kinds.filter { $0 == .dense }.count == 17 && kinds.filter { $0 == .none }.count == 25)
        let d = ModelDims(c)
        #expect(d.attnQH == 40 && d.attnKVH == 8 && d.attnHD == 128 && !d.attnGate && !d.attnQKN)
        let ropes = ModelSpec.layerRopes(c, base: d, full: nil).compactMap { $0 }
        #expect(ropes.count == 4 && ropes.allSatisfy { $0.nope })
        // in_proj [17504, 3136] = z 7680 | xBC 9728 | dt 96.
        #expect(d.ssm && d.gdnHV == 96 && d.gdnHK == 8 && d.gdnDK == 128 && d.gdnDV == 80)
        #expect(d.gdnVDim == 7680 && d.gdnConvDim == 9728 && d.gdnProjN == 17504)
        #expect(Self.gapList(c) == nil && d.ffnActSuffix == "_relu2")
        let fp = ModelFootprint(
            name: "nano", config: c, weightBytes: 0, largestTensorBytes: 0, drafterConfig: nil,
            drafterWeightBytes: nil)
        #expect(fp.kvBytesPerToken == 4 * 2 * 8 * 132 + 8)
        #expect(fp.snapshotBaseBytes == 21 * (96 * 80 * 128 * 4 + 3 * 9728 * 4))
    }

    /// The three spellings of the layer list (mlx-lm nemotron_h's `__post_init__`) give the same layers; an unknown
    /// block or a count that disagrees with num_hidden_layers is a load error.
    @Test func nemotronLayerPattern() throws {
        let pattern = "ME*-"
        let types = ["mamba", "ffn", "full_attention", "ffn"]
        func cfg(_ edit: @escaping (inout [String: Any]) -> Void) throws -> ModelConfig {
            try Self.nano {
                $0["hybrid_override_pattern"] = nil
                $0["num_hidden_layers"] = 4
                edit(&$0)
            }
        }
        let spellings: [(String, Any)] = [
            ("hybrid_override_pattern", pattern), ("hybrid_override_pattern", ["M", "E", "*", "-"]),
            ("layers_block_type", ["mamba", "moe", "attention", "mlp"]),
            ("layers_block_type", ["linear_attention", "moe", "full_attention", "mlp"]),
            ("layer_types", ["conv", "moe", "full_attention", "mlp"]),
        ]
        for (k, v) in spellings {
            let c = try cfg { $0[k] = v }
            #expect(c.layerTypes == types, "\(k)")
            #expect(ModelSpec.ffnKinds(c) == [.none, .moe, .none, .dense], "\(k)")
        }
        // hybrid_override_pattern wins over the lists, as in mlx-lm.
        let both = try cfg {
            $0["hybrid_override_pattern"] = "MM*-"
            $0["layer_types"] = ["moe"]
        }
        #expect(both.layerTypes.first == "mamba")
        let unreadable =
            "nemotron_h without a readable layer pattern (hybrid_override_pattern, layers_block_type or layer_types "
            + "with M, *, -, E)"
        for bad: Any in ["MX*-", ["mamba", "moe", "attention", "gdn"], [String]()] {
            let c = try cfg {
                $0["layers_block_type"] = bad
                $0["hybrid_override_pattern"] = bad as? String
            }
            #expect(c.layerTypes.isEmpty && Self.gapList(c)?.contains(unreadable) == true, "\(bad)")
        }
        #expect(try Self.gapList(cfg { _ in }) == [unreadable])
        let short = try cfg { $0["hybrid_override_pattern"] = "M*-" }
        #expect(Self.gapList(short)?.contains("nemotron_h pattern of 3 layers with num_hidden_layers 4") == true)
    }

    /// What stays out of the supported scope or is not the shape the Mamba2 kernels will run is a named load error.
    @Test func nemotronOutOfScope() throws {
        let cases: [(String, (inout [String: Any]) -> Void)] = [
            (
                "LatentMoE (moe_latent_size 1024: Nemotron 3.5 Super/Bolt, not supported)",
                { $0["moe_latent_size"] = 1024 }
            ),
            (
                "grouped expert routing (n_group 8, not supported)",
                {
                    $0["n_group"] = 8
                    $0["topk_group"] = 4
                }
            ),
            (
                "Mamba2 time_step_limit [0.001, 0.1] (only mlx-lm's (0, ∞))",
                { $0["time_step_limit"] = [0.001, 0.1] }
            ),
            ("Mamba2 projection biases (mamba_proj_bias)", { $0["mamba_proj_bias"] = true }),
            ("Mamba2 activation gelu (only silu)", { $0["mamba_hidden_act"] = "gelu" }),
            ("Mamba2 conv without bias (use_conv_bias false)", { $0["use_conv_bias"] = false }),
            ("Mamba2 conv of 3 (only 4)", { $0["conv_kernel"] = 3 }),
            (
                "Mamba2 of 64 heads × 72, 8 groups, state 128 (state 128, head a multiple of 16 and whole groups)",
                { $0["mamba_head_dim"] = 72 }
            ),
            (
                "Mamba2 of 64 heads × 64, 8 groups, state 64 (state 128, head a multiple of 16 and whole groups)",
                { $0["ssm_state_size"] = 64 }
            ),
            (
                "experts at 8 bits (the MoE kernels read 4-bit experts only: OptiQ, not supported)",
                {
                    $0["quantization"] = [
                        "group_size": 64, "bits": 4, "mode": "affine",
                        "backbone.layers.1.mixer.switch_mlp.fc1": ["group_size": 64, "bits": 8],
                    ]
                }
            ),
            (
                "MXFP4 quantization in nemotron_h (not supported: only the gpt-oss experts run it)",
                { $0["quantization"] = ["group_size": 32, "bits": 4, "mode": "mxfp4"] }
            ),
        ]
        for (gap, edit) in cases {
            #expect(try Self.gapList(Self.lightning(edit))?.contains(gap) == true, "\(gap)")
        }
        // The real values pass: the Lightning's n_group 1, OptiQ's 8-bit Mamba projections (only experts count), and the
        // MTP head is skipped like mlx-lm's sanitize.
        let fine = try Self.lightning {
            $0["quantization"] = [
                "group_size": 64, "bits": 4, "backbone.layers.0.mixer.in_proj": ["group_size": 64, "bits": 8],
            ]
        }
        #expect(Self.gapList(fine) == nil)
        #expect(SafetensorsStore.isSkipped("mtp.layers.0.mixer.q_proj.weight"))
        // Elsewhere gpt-oss keeps its MXFP4 message.
        let mx = try Self.granite(["quantization": ["group_size": 32, "bits": 4, "mode": "mxfp4"]])
        #expect(Self.gapList(mx)?.first?.hasPrefix(Self.goMXFP4) == true)
    }

    /// toy-nemotron v2 against the Lightning it comes from: the same Mamba2, attention and router shapes (only
    /// the hidden and the intermediates cut), the pattern M E M M * E - of single-block layers, relu² without gate, and
    /// no gap (its declared difference: a dense MLP layer, the Nano 4B's kind).
    @Test func toyNemotronMatchesReal() throws {
        let r = try ToyReal.load("nemotron-3.5-lightning-30b-a3b", root: Self.repo)
        let data = try JSONSerialization.data(
            withJSONObject: ToyConfig.nemotronH(ToyDims.nemotronH(r), quant: [:], seed: 1))
        let toy = try ModelConfig(json: try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any]))
        let real = try Self.lightning()
        let spec = try ModelSpec(toy), rspec = try real.requiredSpec
        #expect(toy.family == .nemotronH)
        #expect(toy.layerTypes == ["mamba", "ffn", "mamba", "mamba", "full_attention", "ffn", "ffn"])
        #expect(spec.ffnKinds == [.none, .moe, .none, .none, .none, .moe, .dense] && toy.isMoE)
        #expect(toy.rmsNormEps == real.rmsNormEps && spec.block.activation == .relu2)
        let t = ModelDims(toy), d = ModelDims(real)
        #expect(t.hidden == ToyCut.nemotronHidden && t.hidden % 128 == 0 && d.hidden % 128 == 0)
        #expect(t.ssm && t.gdnHV == d.gdnHV && t.gdnHK == d.gdnHK && t.gdnDK == d.gdnDK && t.gdnDV == d.gdnDV)
        #expect(t.gdnProjN == d.gdnProjN && t.normF32 == d.normF32)
        #expect(t.attnQH == d.attnQH && t.attnKVH == d.attnKVH && t.attnHD == d.attnHD)
        #expect(t.attnGate == d.attnGate && t.attnQKN == d.attnQKN)
        // MoE: the router, top-k, shared expert and activation of the real one; intermediates cut keeping their kind
        // (a multiple of 64 but not of 128, shared twice as wide).
        #expect(t.moeE == d.moeE && t.moeK == d.moeK && t.moeShared == d.moeShared && t.moeSigmoid == d.moeSigmoid)
        #expect(t.moeBias == d.moeBias && t.moeRScale == d.moeRScale && t.moeEps == d.moeEps && t.moeH64 == d.moeH64)
        #expect(t.ffnActSuffix == d.ffnActSuffix && t.macros["LAB_FFN_RELU2"] == d.macros["LAB_FFN_RELU2"])
        let tr = try #require(spec.router), rr = try #require(rspec.router)
        #expect(tr.scoring == rr.scoring && tr.shared == rr.shared && tr.experts == rr.experts && tr.topK == rr.topK)
        #expect(tr.selectBias == rr.selectBias && tr.scale == rr.scale && tr.normEps == rr.normEps)
        #expect(t.moeI % 64 == 0 && t.moeI % 128 != 0 && d.moeI % 64 == 0 && d.moeI % 128 != 0)
        #expect(toy.sharedExpertIntermediateSize == 2 * t.moeI && real.sharedExpertIntermediateSize == 2 * d.moeI)
        let ropes = ModelSpec.layerRopes(toy, base: t, full: nil).compactMap { $0 }
        #expect(ropes.count == 1 && ropes.allSatisfy { $0.nope })
    }

    /// Detection, calibration, names (the `backbone.` root, `lm_head` outside it) and the keys only Nemotron-H reads.
    @Test func nemotronDetectionNamesAndKeys() throws {
        #expect(ModelFamily.detect(textType: "nemotron_h", rawType: "nemotron_h") == .nemotronH)
        #expect(ModelFamily.nemotronH.calibrationName == "nemotron_h")
        #expect(FamilyCalibration.forFamily(.nemotronH) == .nemotronH && FamilyCalibration.nemotronH.measured)
        #expect(!ModelSpec.attnGated(.nemotronH))
        let n = ModelSpec.weightNames(.nemotronH)
        #expect(n.inputNorm == "norm.weight" && n.postNorm == "norm.weight" && n.finalNorm == "norm_f.weight")
        #expect(n.attn == "mixer." && n.ffn == "mixer." && n.oProj == "o_proj" && n.embedding == "embeddings")
        #expect(n.router == "gate" && n.experts == "switch_mlp." && n.sharedExpert == "shared_experts.")
        #expect(n.expertGate == nil && n.expertUp == "fc1" && n.expertDown == "fc2")
        // The other families keep their names.
        for f in ModelFamily.allCases where f != .nemotronH {
            let o = ModelSpec.weightNames(f)
            // GLM-4.7 shares Nemotron-H's `shared_experts.` (GLMSpecTests).
            let shared = f == .glm4MoeLite ? "shared_experts." : "shared_expert."
            #expect(o.attn == "self_attn." && o.embedding == "embed_tokens" && o.sharedExpert == shared)
            #expect(o.expertGate == "gate_proj" && o.expertUp == "up_proj" && o.expertDown == "down_proj")
        }
        // lm_head beside `model.` (and `language_model.model.`), at the top of the file for `backbone.`.
        #expect(HybridModel.lmHeadName(root: "model.") == "lm_head")
        #expect(HybridModel.lmHeadName(root: "language_model.model.") == "language_model.lm_head")
        #expect(HybridModel.lmHeadName(root: "backbone.") == "lm_head")
        // layer_norm_epsilon only in Nemotron-H; n_routed_experts and moe_shared_expert_intermediate_size by key.
        #expect(try Self.lightning { $0["layer_norm_epsilon"] = 1e-6 }.rmsNormEps == 1e-6)
        #expect(try Self.granite(["layer_norm_epsilon": 1e-3]).rmsNormEps == 1e-5)
        // Without n_shared_experts, no shared expert.
        #expect(ModelSpec.sharedKind(try Self.lightning { $0["n_shared_experts"] = nil }) == .none)
        // The Heretic's GDN shape is unchanged (no ssm, 16 × 128 k heads, v of 128).
        let h = ModelDims(try Self.heretic())
        #expect(!h.ssm && h.gdnDV == 128 && h.gdnConvDim == 8192 && h.gdnProjN == 8192 + 4096 + 64)
        #expect(h.macros["LAB_GDN_DV"] == nil && h.macros["LAB_SSM"] == nil)
        // relu² stays out of every other library (LAB_FFN_RELU2), and the gated families keep their instances.
        #expect(h.macros["LAB_FFN_RELU2"] == nil && !h.ffnRelu2 && h.ffnActSuffix == "")
    }
}
