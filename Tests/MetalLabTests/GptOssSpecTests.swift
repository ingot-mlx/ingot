import Foundation
import LabBench
import Testing

@testable import MetalLab

/// gpt-oss in the loader, from the config of mlx-community/gpt-oss-20b-MXFP4-Q8 in ref/toy/real/ (no
/// weights, no GPU): the family is recognized, its shape reads as mlx-lm's gpt_oss, and it loads without quantization
/// and with MXFP4-Q8's (router and expert biases, swiglu_oai and the hidden of 2880 run, and so do MXFP4 experts). Also the two quantization
/// fixes that apply to every family: an override without `mode` is affine, and a non-affine mode is a load error (MXFP4 only in the gpt-oss experts).
extension ModelSpecTests {
    static let goMXFP4 = "MXFP4 quantization"

    /// The quantization of MXFP4-Q8 for layer 0 (the rest repeat it per layer): mxfp4 g32 by default (the experts),
    /// affine 8 bits g64 with `mode` for attention, embedding and lm_head, and the router without `mode`.
    static var mxfp4Q8: [String: Any] {
        [
            "group_size": 32, "bits": 4, "mode": "mxfp4",
            "model.embed_tokens": ["group_size": 64, "bits": 8, "mode": "affine"],
            "model.layers.0.self_attn.q_proj": ["group_size": 64, "bits": 8, "mode": "affine"],
            "model.layers.0.mlp.router": ["group_size": 64, "bits": 8],
            "lm_head": ["group_size": 64, "bits": 8, "mode": "affine"],
        ]
    }

    /// The 20b's config (ref/toy/real/gpt-oss-20b.json), with `quantization` if given, through JSON as the loader reads it.
    static func gptOss(quantization: [String: Any]? = nil, _ edit: (inout [String: Any]) -> Void = { _ in }) throws
        -> ModelConfig
    {
        var j = try ToyReal.load("gpt-oss-20b", root: repo).json
        if let quantization { j["quantization"] = quantization }
        edit(&j)
        let data = try JSONSerialization.data(withJSONObject: j)
        return try ModelConfig(json: try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any]))
    }

    @Test func gptOssFamily() throws {
        let c = try Self.gptOss()
        #expect(c.family == .gptOss && c.modelType == "gpt_oss" && c.spec != nil)
        #expect(c.hiddenSize == 2880 && c.numLayers == 24 && c.vocabSize == 201_088 && c.rmsNormEps == 1e-5)
        #expect(c.layerTypes.count == 24)
        #expect(c.layerTypes.enumerated().allSatisfy { $1 == ($0 % 2 == 0 ? "sliding_attention" : "full_attention") })
        // MoE: 32 experts top-4, `intermediate_size` wide (no moe_intermediate_size), softmax without shared expert.
        #expect(c.isMoE && c.numExperts == 32 && c.expertsPerToken == 4 && c.moeIntermediateSize == 2880)
        // The block: plain norms with ε 1e-5, swiglu_oai with limit 7, no QK-norm, q/k/v/o biases, 1/√64.
        let (b, gaps) = BlockSpec.read(
            text: c.text, family: c.family, eps: Float(c.rmsNormEps), headDim: 64, layers: c.numLayers,
            layerTypes: c.layerTypes)
        let n = NormSpec(.plain, eps: 1e-5)
        #expect(b.mixerNorm == n && b.ffnNorm == n && b.finalNorm == n && b.mixerPostNorm == nil)
        #expect(b.ffnPostNorm == nil && b.embedNorm == nil && b.qkNorm == nil && b.vNorm == nil)
        #expect(
            b.activation == .swigluOAI(limit: 7) && b.qkvBias && b.oBias && b.attnScale == 0.125 && b.embedScale == 1)
        #expect(b.softcap == nil && b.residualScale == 1 && !b.layerScalar && b.needsBranchEpilogue)
        #expect(gaps.isEmpty, "\(gaps)")
        // Shape: 64 q / 8 kv heads of 64 without gate or QK-norm, whole-head RoPE with a table (YaRN), window and
        // sinks in every layer (implied by the family: the config has no softmax_type).
        let d = ModelDims(c)
        #expect(d.hidden == 2880 && d.attnQH == 64 && d.attnKVH == 8 && d.attnHD == 64 && d.attnRot == 64)
        #expect(!d.attnGate && !d.attnQKN && d.attnSWA && d.attnSinks && d.attnRopeTab && d.attnQKVB && !d.ffnGelu)
        #expect(d.moeE == 32 && d.moeK == 4 && d.moeI == 2880 && d.moeShared == 0 && !d.moeSigmoid && !d.moeBias)
        #expect(d.moeRScale == 1 && d.moeEps == 0 && d.macros["LAB_AT_SINKS"] as? NSNumber == 1)
        // Router linear bias, expert biases, swiglu_oai (the `_oai` instances) and the shifted last down tile.
        #expect(d.ffnLimit == 7 && d.ffnActSuffix == "_oai" && d.moeLBias && d.moeEBias && d.moeH64)
        #expect(d.macros["LAB_FFN_LIMIT"] as? String == "7.0f" && d.macros["LAB_MOE_H64"] as? NSNumber == 1)
        for k in ["LAB_FFN_OAI", "LAB_MOE_LBIAS", "LAB_MOE_EBIAS"] { #expect(d.macros[k] as? NSNumber == 1, "\(k)") }
        // Names of the family: Qwen's norms, `mlp.router` and `mlp.experts.`.
        let names = ModelSpec.weightNames(.gptOss)
        #expect(names.ffn == "mlp." && names.router == "router" && names.experts == "experts.")
        #expect(names.inputNorm == "input_layernorm.weight" && names.postNorm == "post_attention_layernorm.weight")
        #expect(names.sinks == "sinks" && names.mixerPostNorm == nil && names.ffnPostNorm == nil)
        // Without quantization it loads, with the router's biases on, and so does MXFP4-Q8: its MXFP4 experts
        // turn on the `_mxfp4` instances (LAB_MXFP4) and, being another format, not the affine g32 ones (LAB_G32).
        let spec = try ModelSpec(c)
        let r = try #require(spec.router)
        #expect(r.logitBias && r.expertBiases && !r.selectBias && r.scoring == .softmax && r.shared == .none)
        #expect(!d.moeMXFP4 && d.macros["LAB_MXFP4"] == nil)
        let q8 = try Self.gptOss(quantization: Self.mxfp4Q8)
        #expect(Self.loadError(q8) == nil)
        let d8 = ModelDims(q8)
        #expect(d8.moeMXFP4 && d8.macros["LAB_MXFP4"] as? NSNumber == 1)
        #expect(!d8.quantG32 && d8.macros["LAB_G32"] == nil)
    }

    /// What is still refused: swiglu_limit outside gpt-oss, swiglu_oai with a dense FFN, a MoE hidden that is not a
    /// multiple of 64 (or under 128), and the router biases are off in every other family.
    @Test func gptOssGuards() throws {
        let g = try Self.granite(["swiglu_limit": 7.0])
        #expect(Self.gapList(g)?.contains { $0.hasPrefix("swiglu_limit = 7.0 outside gpt-oss") } == true)
        let dense = try Self.gptOss {
            $0["num_local_experts"] = nil; $0["num_experts_per_tok"] = nil
        }
        #expect(Self.gapList(dense)?.contains("swiglu_oai in a dense FFN (only the MoE kernels have it)") == true)
        let odd = try Self.gptOss { $0["hidden_size"] = 2848 }
        #expect(Self.gapList(odd)?.contains { $0.hasPrefix("MoE with hidden 2848") } == true)
        #expect(Self.loadError(try Self.gptOss { $0["hidden_size"] = 576 }) == nil)
        for c in [try Self.heretic(), try Self.lfm2()] {
            let r = try #require(try ModelSpec(c).router)
            #expect(!r.logitBias && !r.expertBiases)
            let d = ModelDims(c)
            #expect(!d.moeLBias && !d.moeEBias && !d.moeH64 && d.ffnLimit == nil && d.ffnActSuffix == "")
        }
    }

    /// Flat YaRN with `truncate: false` in the 24 layers, sliding and full alike (no `rope_local_base_freq`), θ 150000.
    @Test func gptOssRope() throws {
        let c = try Self.gptOss()
        #expect(ModelSpec.hasScaledRope(c))
        for ty in Set(c.layerTypes) {
            let s = try #require(RopeSpec.scaling(c.text, layerType: ty), "\(ty)")
            #expect(s.kind == .yarn && s.truncate == false && s.factor == 32, "\(ty)")
            #expect(RopeSpec.theta(c.text, layerType: ty) == 150_000, "\(ty)")
        }
        #expect(ModelSpec.ropeGaps(c.text, family: .gptOss, shape: ModelDims(c)).isEmpty)
    }

    /// Detection, calibration, sinks by family and the experts' width read from `intermediate_size` only in gpt-oss.
    @Test func gptOssDetectionAndKeys() throws {
        #expect(ModelFamily.detect(textType: "gpt_oss", rawType: "gpt_oss") == .gptOss)
        #expect(ModelFamily.detect(textType: "gpt_oss_text", rawType: nil) == nil)
        #expect(ModelFamily.gptOss.calibrationName == "gpt_oss")
        #expect(FamilyCalibration.forFamily(.gptOss) == .gptOss && FamilyCalibration.gptOss.measured)
        #expect(ModelSpec.weightNames(.qwen35Moe).router == "gate" && ModelSpec.weightNames(.lfm2).router == "gate")
        let c = try Self.gptOss()
        #expect(!ModelSpec.attnGated(.gptOss) && ModelSpec.sharedKind(c) == .none)
        // Sinks: by family in gpt-oss, by softmax_type elsewhere.
        #expect(!ModelSpec.attnSinks(try Self.granite()))
        #expect(ModelSpec.attnSinks(try Self.granite(["softmax_type": "learnable"])))
        // Another family with num_local_experts and intermediate_size keeps no experts' width (its error, not 2880).
        let g = try Self.granite(["num_local_experts": 8, "num_experts_per_tok": 2])
        #expect(g.isMoE && g.moeIntermediateSize == nil)
        // moe_intermediate_size still wins in gpt-oss.
        #expect(try Self.gptOss { $0["moe_intermediate_size"] = 1440 }.moeIntermediateSize == 1440)
    }

    /// An override without `mode` is affine, as in mlx (`to_quantized(**params)`), and does not inherit the default's
    /// mode: MXFP4-Q8's routers (`{bits 8, group_size 64}`) were marked mxfp4. Any non-affine mode is a load error.
    @Test func quantOverrideModeIsAffine() throws {
        let c = try Self.gptOss(quantization: Self.mxfp4Q8)
        #expect(c.quantDefault == QuantParams(bits: 4, groupSize: 32, mode: "mxfp4"))
        #expect(c.quant(for: "model.layers.0.mlp.router") == QuantParams(bits: 8, groupSize: 64, mode: "affine"))
        #expect(c.quant(for: "model.layers.0.self_attn.q_proj") == QuantParams(bits: 8, groupSize: 64))
        #expect(c.quant(for: "model.layers.0.mlp.experts.gate_proj")?.mode == "mxfp4")
        // An explicit mode in an override is kept.
        var q = Self.mxfp4Q8
        q["model.layers.0.mlp.router"] = ["group_size": 32, "bits": 4, "mode": "mxfp4"]
        #expect(try Self.gptOss(quantization: q).quant(for: "model.layers.0.mlp.router")?.mode == "mxfp4")
        // Affine default (every current model): overrides stay affine, the spec loads (Granite 4.1 with its g32).
        let ga = try Self.granite([
            "quantization": ["group_size": 32, "bits": 4, "model.lm_head": ["group_size": 64, "bits": 8]]
        ])
        #expect(ga.quant(for: "model.lm_head")?.mode == "affine" && Self.loadError(ga) == nil)
        // A non-affine mode is a gap whatever the family (here Granite), even only in an override.
        let nv = try Self.granite(["quantization": ["group_size": 16, "bits": 4, "mode": "nvfp4"]])
        #expect(Self.gapList(nv) == ["quantization mode «nvfp4» (affine only)"])
        // MXFP4 runs in gpt-oss only (its experts), and only as 4 bits g32.
        let mx = try Self.granite([
            "quantization": ["group_size": 64, "bits": 4, "lm_head": ["group_size": 32, "bits": 4, "mode": "mxfp4"]]
        ])
        #expect(Self.gapList(mx) == ["MXFP4 quantization outside gpt-oss (only its experts have MXFP4 kernels)"])
        #expect(!ModelDims(mx).moeMXFP4 && !ModelDims(mx).quantG32)
        var q16 = Self.mxfp4Q8
        q16["group_size"] = 16
        #expect(
            Self.gapList(try Self.gptOss(quantization: q16)) == [
                "MXFP4 quantization with 4 bits g16 (it is 4 bits g32)"
            ])
    }

    /// QuantizedLinear reads affine, and mxfp4 (U8 e8m0 scales, no biases) only where the caller has its kernels (the
    /// MoE experts): elsewhere it is a load error instead of reading its scales as bf16. MXFP4-Q8's router (an override
    /// without `mode`) loads as affine.
    @Test func quantizedLinearRejectsNonAffine() throws {
        let fmt = FormatTests()
        let dir = try fmt.tmpDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        // Experts: 2 × 4 rows × 64 inputs in mxfp4 g32 (weight [2, 4, 8] u32, scales [2, 4, 2] u8); router: 64 rows
        // × 64 inputs affine 8 bits g64 (weight [64, 16] u32, scales and biases [64, 1] bf16).
        let e = "model.layers.0.mlp.experts.gate_proj", r = "model.layers.0.mlp.router"
        try fmt.writeSafetensors(
            dir,
            tensors: [
                (e + ".weight", "U32", [2, 4, 8], Data(count: 2 * 4 * 8 * 4)),
                (e + ".scales", "U8", [2, 4, 2], Data(repeating: 127, count: 2 * 4 * 2)),
                (r + ".weight", "U32", [64, 16], Data(count: 64 * 16 * 4)),
                (r + ".scales", "BF16", [64, 1], Data(count: 64 * 2)),
                (r + ".biases", "BF16", [64, 1], Data(count: 64 * 2)),
            ])
        let st = try SafetensorsStore(directory: dir)
        let cfg = try Self.gptOss(quantization: Self.mxfp4Q8)
        do {
            _ = try QuantizedLinear(store: st, path: e, config: cfg)
            Issue.record("an mxfp4 layer loaded as affine")
        } catch SafetensorsError.format(let m) {
            #expect(
                m == "\(e): quantization mode «mxfp4» not supported (affine only; MXFP4 runs only in the MoE experts)")
        }
        let fp4 = try QuantizedLinear(store: st, path: e, config: cfg, mxfp4: true)
        #expect(fp4.isMXFP4 && fp4.params == QuantParams(bits: 4, groupSize: 32, mode: "mxfp4"))
        #expect(fp4.experts == 2 && fp4.outFeatures == 4 && fp4.inFeatures == 64 && fp4.biases == nil)
        // Scale 127 (2⁰) and zero codes: the CPU dequantization gives zeros of the right size.
        #expect(st.dequantizeRows(fp4, expert: 1, rows: 0..<4) == [Float](repeating: 0, count: 4 * 64))
        let router = try QuantizedLinear(store: st, path: r, config: cfg)
        #expect(router.params == QuantParams(bits: 8, groupSize: 64) && router.outFeatures == 64)
        #expect(
            QuantizedLinear.modeMessage(path: "x", mode: "nvfp4")
                == "x: quantization mode «nvfp4» not supported (affine only)")
    }
}
