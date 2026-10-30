import Foundation
import LabBench
import Testing

@testable import MetalLab

/// Family table (ModelSpec.swift) with the config.json fields of each family (the Heretic's, the 27B's,
/// Flash-Next's and LFM2.5's, trimmed to what the engine reads). No GPU.
@Suite struct ModelSpecTests {
    static var rope: [String: Any] {
        [
            "mrope_interleaved": true, "mrope_section": [11, 11, 10], "partial_rotary_factor": 0.25,
            "rope_theta": 10_000_000, "rope_type": "default",
        ]
    }
    static func qwen(_ type: String, hidden: Int, extra: [String: Any]) throws -> ModelConfig {
        var t: [String: Any] = [
            "model_type": type, "hidden_size": hidden, "num_hidden_layers": 40, "head_dim": 256,
            "num_attention_heads": 16, "num_key_value_heads": 2, "linear_num_value_heads": 32,
            "partial_rotary_factor": 0.25, "rope_parameters": rope,
        ]
        t.merge(extra) { _, b in b }
        return try ModelConfig(json: ["model_type": String(type.dropLast(5)), "text_config": t])
    }
    static func heretic() throws -> ModelConfig {
        try qwen(
            "qwen3_5_moe_text", hidden: 2048,
            extra: [
                "num_experts": 256, "num_experts_per_tok": 8, "moe_intermediate_size": 512,
                "shared_expert_intermediate_size": 512,
            ])
    }
    static func q27() throws -> ModelConfig {
        try qwen(
            "qwen3_5_text", hidden: 5120,
            extra: [
                "intermediate_size": 17408, "num_attention_heads": 24, "num_key_value_heads": 4,
                "linear_num_value_heads": 48,
            ])
    }
    static func flashNext() throws -> ModelConfig {
        try qwen(
            "qwen4_exp_text", hidden: 2560,
            extra: [
                "num_experts": 512, "num_experts_per_tok": 10, "moe_intermediate_size": 640,
                "shared_expert_intermediate_size": 640, "num_attention_heads": 24, "linear_num_value_heads": 48,
                "output_gate_type": "sigmoid", "hc_count": 4, "hc_lowrank": 320, "heads_per_ngram": 8, "ngram_size": 3,
                "ple_embed_dim": 2560,
            ])
    }
    static func lfm2() throws -> ModelConfig {
        try ModelConfig(json: [
            "model_type": "lfm2_moe", "hidden_size": 2048, "num_hidden_layers": 40, "num_dense_layers": 2,
            "num_attention_heads": 32, "num_key_value_heads": 8, "num_experts": 64, "num_experts_per_tok": 4,
            "moe_intermediate_size": 1536, "rope_theta": 1_000_000, "norm_eps": 1e-5,
        ])
    }

    static func bonsai27() throws -> ModelConfig {
        var t = try q27().text
        t["model_type"] = "qwen3_5_text"
        return try ModelConfig(json: ["model_type": "prism_hadamard_qwen35", "text_config": t])
    }

    @Test func families() throws {
        let h = try ModelSpec(Self.heretic())
        #expect(h.family == .qwen35Moe && h.expertLayout == .mlxSwitchMLP && h.hasDrafter)
        #expect(
            h.router
                == RouterSpec(
                    scoring: .softmax, shared: .gated, experts: 256, topK: 8, inter: 512, sharedInter: 512))
        #expect(h.rope == RopeSpec(theta: 1e7, rotaryDims: 64))
        #expect(h.names == .qwen && h.ffnKinds == Array(repeating: .moe, count: 40) && h.block.qkNorm == .plain)
        #expect(h.ropes.count == 40 && h.ropes.allSatisfy { $0 == h.rope })
        #expect(h.shape == ModelDims())  // the kernels' defaults are Qwen3.6's values
        #expect(h.defaultDrafter == "incoai/Qwen3.6-35B-A3B-DFlash2")

        let d = try ModelSpec(Self.q27())
        #expect(d.family == .qwen35 && d.router == nil && d.expertLayout == .unsupported)
        #expect(d.shape.hidden == 5120 && d.shape.attnQH == 24 && d.shape.gdnHV == 48 && d.rope.rotaryDims == 64)

        let f = try ModelSpec(Self.flashNext())
        #expect(f.family == .flashNext && f.expertLayout == .metalLabRecords)
        #expect(f.router?.experts == 512 && f.router?.topK == 10 && f.rope == RopeSpec(theta: 1e7, rotaryDims: 64))
        #expect(f.shape.normF32 && f.shape.gdnSigmoid && f.shape.moeE == 512 && f.shape.moeK == 10)
        #expect(f.shape.hcS == 4 && f.shape.pleH == 16 && f.shape.pleHPN == 8 && f.shape.pleDim == 2560)
        #expect(f.defaultDrafter == "incoai/Qwen3.8-Flash-Next-DFlash2")

        let l = try ModelSpec(Self.lfm2())
        #expect(l.family == .lfm2 && !l.hasDrafter && l.defaultDrafter == nil && l.expertLayout == .unsupported)
        #expect(l.router?.scoring == .sigmoid && l.router?.shared == RouterSpec.Shared.none)
        #expect(l.router?.selectBias == true && l.router?.scale == 1 && l.router?.normEps == 1e-6)
        #expect(l.names == .lfm2 && l.ffnKinds == [.dense, .dense] + Array(repeating: .moe, count: 38))
        #expect(d.ffnKinds == Array(repeating: .dense, count: 40) && l.router?.sharedInter == 0)
        #expect(l.shape.attnHD == 64 && !l.shape.attnGate && l.rope == RopeSpec(theta: 1e6, rotaryDims: 64))
        #expect(l.calibration.source == FamilyCalibration.lfm2.source)
        // W8A8 per-position threshold: 1024 in all of them (measured on the Heretic and on Flash-Next).
        #expect(h.int8MinPos == 1024 && f.int8MinPos == 1024 && d.int8MinPos == 1024 && l.int8MinPos == 1024)
    }

    @Test func bonsaiIsQwen35() throws {
        let c = try ModelConfig(json: [
            "model_type": "prism_hadamard_qwen35",
            "text_config": ["model_type": "qwen3_5_text", "hidden_size": 4096, "num_hidden_layers": 32],
        ])
        #expect(c.family == .qwen35)
        #expect(try ModelSpec(c).shape.normF32)
    }

    @Test func unsupportedShapeFails() throws {
        // GDN with 32 k heads (the kernels assume 16 × 128): a load error instead of wrong kernels.
        let c = try Self.qwen(
            "qwen3_5_text", hidden: 2048,
            extra: ["intermediate_size": 6144, "linear_num_key_heads": 32, "layer_types": ["linear_attention"]])
        #expect(throws: SafetensorsError.self) { try ModelSpec(c) }
    }

    @Test func moeKernelsByRouter() throws {
        let h = try #require(try ModelSpec(Self.heretic()).router)
        #expect(try MoEKernelSet.forRouter(h, inter: 512).route == "vmoe_route")
        let l = try #require(try ModelSpec(Self.lfm2()).router)
        let lk = try MoEKernelSet.forRouter(l, inter: 1536)
        #expect(lk.prefillRoute == "moe_route_topk_cfg")
        // Qwen3.6's router takes E and top-k from the config: Flash-Next's shape (512 top-10 of 640) fits, even though
        // Flash-Next runs on its own kernels (FlashParts).
        let f = try #require(try ModelSpec(Self.flashNext()).router)
        #expect(try MoEKernelSet.forRouter(f, inter: 640).prefillRoute == "moe_route_topk_cfg")
        // Outside what vmoe_route and moe_route_topk accept: E > 512, top-k > 16, intermediate not a multiple of 64.
        for (e, k, i) in [(1024, 8, 512), (256, 20, 512), (256, 8, 520)] {
            let r = RouterSpec(
                scoring: .softmax, shared: .gated, experts: e, topK: k, inter: i, sharedInter: 512)
            #expect(throws: SafetensorsError.self) { try MoEKernelSet.forRouter(r, inter: i) }
        }
    }

    /// The router settings come from the config (scoring_func, selection bias, routed_scaling_factor), with
    /// the family's as the default; what the kernels do not run yet fails when loading.
    @Test func routerKeysFromConfig() throws {
        func heretic(_ extra: [String: Any]) throws -> ModelConfig {
            var t = try Self.heretic().text
            t.merge(extra) { _, b in b }
            return try ModelConfig(json: ["model_type": "qwen3_5_moe", "text_config": t])
        }
        let h = try #require(try ModelSpec(Self.heretic()).router)
        #expect(h.scoring == .softmax && !h.selectBias && h.scale == 1 && h.normEps == 0)
        func keys(_ extra: [String: Any], _ base: [String: Any]? = nil, family: ModelFamily = .qwen35Moe) throws
            -> (scoring: RouterSpec.Scoring, selectBias: Bool, scale: Float, normEps: Float)
        {
            var t = try base ?? Self.heretic().text
            t.merge(extra) { _, b in b }
            return ModelSpec.routerKeys(t, family: family)
        }
        // GLM-4.7-Flash's keys: sigmoid, noaux_tc (selection bias) and the scale.
        let g = try keys(["scoring_func": "sigmoid", "topk_method": "noaux_tc", "routed_scaling_factor": 1.8])
        #expect(g.scoring == .sigmoid && g.selectBias && g.scale == 1.8 && g.normEps == 0)
        // LFM2 without use_expert_bias: sigmoid without bias, LFM2's epsilon.
        let lr = try keys(["use_expert_bias": false], try Self.lfm2().text, family: .lfm2)
        #expect(lr.scoring == .sigmoid && !lr.selectBias && lr.normEps == 1e-6)
        // Explicit errors: grouped top-k, router softcapping, unknown scoring or top-k method, softmax with bias.
        let cases: [([String: Any], String)] = [
            (["n_group": 8, "topk_group": 4], "grouped top-k routing"),
            (["router_logit_softcapping": 30.0], "router logit softcapping"),
            (["scoring_func": "sqrtsoftplus"], "scoring_func «sqrtsoftplus»"),
            (["topk_method": "group_limited_greedy"], "topk_method «group_limited_greedy»"),
            (["use_expert_bias": true], "selection bias with a softmax router"),
        ]
        for (extra, want) in cases {
            let m = Self.loadError(try heretic(extra))
            #expect(m?.contains(want) == true, "\(extra): \(m ?? "loads")")
        }
        // One group (or every group kept) is plain top-k.
        #expect(Self.loadError(try heretic(["n_group": 1, "topk_group": 1])) == nil)
        #expect(Self.loadError(try heretic(["n_group": 4, "topk_group": 4])) == nil)
        // Softmax with 32 experts (gpt-oss): the logits go with the padded stride (routerLD), E multiple of 32.
        let e32 = RouterSpec(
            scoring: .softmax, shared: .none, experts: 32, topK: 4, inter: 512, sharedInter: 0)
        #expect(try MoEKernelSet.forRouter(e32, inter: 512).route == "vmoe_route")
    }

    /// The error message when loading (nil if it loads).
    static func loadError(_ c: ModelConfig) -> String? {
        do {
            _ = try ModelSpec(c)
            return nil
        } catch SafetensorsError.format(let m) {
            return m
        } catch {
            return "\(error)"
        }
    }

    @Test func partialRotaryDefaultByFamily() throws {
        // Without partial_rotary_factor: 0.25 only in Qwen3.5/3.6; Flash-Next and the rest, the whole head.
        func strip(_ c: ModelConfig) throws -> ModelConfig {
            var t = c.text
            t["partial_rotary_factor"] = nil
            t["rope_parameters"] = ["rope_theta": 10_000_000, "rope_type": "default"]
            return try ModelConfig(json: ["model_type": c.raw["model_type"]!, "text_config": t])
        }
        #expect(try ModelSpec(strip(Self.heretic())).rope.rotaryDims == 64)
        #expect(try ModelSpec(strip(Self.q27())).rope.rotaryDims == 64)
        #expect(try ModelSpec(strip(Self.flashNext())).rope.rotaryDims == 256)
        #expect(try ModelSpec(Self.lfm2()).rope.rotaryDims == 64)
    }

    /// Assumptions that used to be ignored silently now give a clear error when loading.
    @Test func silentAssumptionsFail() throws {
        func heretic(_ extra: [String: Any]) throws -> ModelConfig {
            var t = try Self.heretic().text
            t.merge(extra) { _, b in b }
            return try ModelConfig(json: ["model_type": "qwen3_5_moe", "text_config": t])
        }
        // The usual config loads (with rope_scaling: null, as in many config.json files).
        #expect(Self.loadError(try heretic(["rope_scaling": NSNull()])) == nil)
        let yarn: [String: Any] = ["rope_type": "yarn", "factor": 4.0, "rope_theta": 10_000_000]
        let cases: [([String: Any], String)] = [
            (["rope_parameters": yarn], "yarn» without original_max_position_embeddings"),
            (["rope_scaling": ["type": "dynamic", "factor": 2.0]], "rope_type «dynamic»"),
            (["shared_expert_intermediate_size": 1000], "shared expert of 1000"),
            (["shared_expert_intermediate_size": NSNull()], "shared expert of 0"),
            (["hidden_size": 2848], "hidden 2848"),
            (["head_dim": 96], "head of 96"),
            (["first_k_dense_replace": 40], "MoE without any MoE layer"),
            (["mlp_only_layers": [0, 5]], "mlp_only_layers"),
            (["decoder_sparse_step": 2], "decoder_sparse_step"),
            (["norm_topk_prob": false], "norm_topk_prob = false"),
            (["num_experts": 1024], "unsupported MoE"),
        ]
        for (extra, want) in cases {
            let m = Self.loadError(try heretic(extra))
            #expect(m?.contains(want) == true, "\(extra): \(m ?? "loads")")
        }
        // Several at once: a single error with all of them.
        let m = Self.loadError(try heretic(["shared_expert_intermediate_size": 1000, "head_dim": 96]))
        #expect(m?.contains("shared expert of 1000") == true && m?.contains("head of 96") == true)
        // norm_topk_prob = false with the sigmoid router too (no tier-1 family uses it: the kernels always
        // renormalize), and the empty mlp_only_layers / decoder_sparse_step 1 that Qwen3.5 configs declare
        // load.
        var l = try Self.lfm2().text
        l["norm_topk_prob"] = false
        #expect(Self.loadError(try ModelConfig(json: l))?.contains("norm_topk_prob = false") == true)
        #expect(Self.loadError(try heretic(["mlp_only_layers": [Int](), "decoder_sparse_step": 1])) == nil)
    }

    /// Dense layers of a MoE from num_dense_layers and first_k_dense_replace.
    @Test func denseLayersFromConfig() throws {
        func kinds(_ extra: [String: Any]) throws -> [FFNKind] {
            var t = try Self.heretic().text
            t.merge(["num_hidden_layers": 8, "layer_types": Array(repeating: "linear_attention", count: 8)]) { _, b in b
            }
            t.merge(extra) { _, b in b }
            return try ModelSpec(ModelConfig(json: ["model_type": "qwen3_5_moe", "text_config": t])).ffnKinds
        }
        let d = FFNKind.dense, m = FFNKind.moe
        #expect(try kinds([:]) == Array(repeating: m, count: 8))
        #expect(try kinds(["first_k_dense_replace": 1]) == [d] + Array(repeating: m, count: 7))
        #expect(try kinds(["num_dense_layers": 2]) == [d, d] + Array(repeating: m, count: 6))
        #expect(try ModelSpec(Self.q27()).ffnKinds == Array(repeating: d, count: 40))
    }

    /// Sigmoid routers load through vmoe_route and moe_route_topk_cfg with their macros; the library of the current
    /// models does not change (Qwen emits none of them).
    @Test func sigmoidRoutersLoad() throws {
        // GLM-4.7 style: sigmoid + noaux_tc bias + 1.8 with the gated shared expert of Qwen's naming. (Nemotron-H's
        // sigmoid router, with its ungated shared expert, is in NemotronHSpecTests.)
        var t = try Self.heretic().text
        t.merge(["scoring_func": "sigmoid", "topk_method": "noaux_tc", "routed_scaling_factor": 1.8]) { _, b in b }
        let g = try ModelSpec(ModelConfig(json: ["model_type": "qwen3_5_moe", "text_config": t]))
        let r = try #require(g.router)
        #expect(r.scoring == .sigmoid && r.selectBias && r.scale == 1.8 && r.shared == .gated)
        let k = try MoEKernelSet.forRouter(r, inter: 512)
        #expect(k.route == "vmoe_route" && k.prefillRoute == "moe_route_topk_cfg")
        #expect(g.shape.macros["LAB_MOE_SCORE"] as? NSNumber == 1 && g.shape.macros["LAB_MOE_BIAS"] as? NSNumber == 1)
        #expect(g.shape.macros["LAB_MOE_RSCALE"] as? String == "1.8f" && g.shape.macros["LAB_MOE_EPS"] == nil)
        // LFM2 runs on the common kernels with the sigmoid macros.
        let l = try ModelSpec(Self.lfm2())
        #expect(try MoEKernelSet.forRouter(l.router!, inter: 1536).route == "vmoe_route")
        #expect(l.shape.macros["LAB_MOE_EPS"] as? String == "1e-06f")
        for m in ["LAB_MOE_SCORE", "LAB_MOE_BIAS", "LAB_MOE_RSCALE", "LAB_MOE_EPS"] {
            #expect(
                try ModelSpec(Self.heretic()).shape.macros[m] == nil
                    && ModelSpec(Self.granite()).shape.macros[m] == nil)
        }
    }

    /// Shared expert gated, ungated (n_shared_experts) and with its own width; LAB_MOE_SHARED.
    @Test func sharedExpertKinds() throws {
        let h = try ModelSpec(Self.heretic())
        #expect(h.router?.shared == .gated && h.router?.sharedInter == 512 && h.shape.moeShared == 1)
        // Gated and wider than the experts (1024 vs 512; the kernels carry its width apart) also loads.
        var t = try Self.heretic().text
        t["shared_expert_intermediate_size"] = 1024
        let w = try ModelSpec(ModelConfig(json: ["model_type": "qwen3_5_moe", "text_config": t]))
        #expect(w.router?.shared == .gated && w.router?.sharedInter == 1024)
        // Ungated (n_shared_experts, Nemotron-H; the rest of its router in NemotronHSpecTests): no gate, its own width.
        let n = try ModelSpec(Self.lightning())
        #expect(n.router?.shared == .ungated && n.shape.macros["LAB_MOE_SHARED"] as? NSNumber == 2)
        #expect(try MoEKernelSet.forRouter(n.router!, inter: 1856).combineNoShared == nil)
        // LFM2: no shared expert (LAB_MOE_SHARED = 0).
        #expect(try ModelSpec(Self.lfm2()).shape.moeShared == 0)
        // No shared expert with softmax: the prefill combine is LFM2's.
        let none = RouterSpec(
            scoring: .softmax, shared: .none, experts: 128, topK: 8, inter: 768, sharedInter: 0)
        #expect(try MoEKernelSet.forRouter(none, inter: 768).combineNoShared == "moe_combine_acc_ns")
    }

    /// Head 128 (gated: Muse Glimmer; ungated), layers without RoPE through `no_rope_layers`, and QK-norm per family.
    @Test func attentionShapesB0T2() throws {
        let h = try ModelSpec(Self.muse())
        #expect(h.shape.attnHD == 128 && h.shape.attnRot == 128 && h.shape.attnGate && h.shape.attnQKN)
        #expect(h.shape.macros["LAB_AT_QKN"] as? NSNumber == 1 && h.shape.macros["LAB_AT_HD"] as? NSNumber == 128)
        #expect(h.ropes.allSatisfy { $0.map { $0.theta == 5e5 && $0.rotaryDims == 128 } ?? false })
        // Ungated: LFM2 with 128 heads (hidden 2048 / 16) loads.
        var l = try Self.lfm2().text
        l["num_attention_heads"] = 16
        #expect(try ModelSpec(ModelConfig(json: l)).shape.attnHD == 128)
        // no_rope_layers (1 = with RoPE): attention layer 3 runs without RoPE; the others, with the model's.
        let types = ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
        var nr = try Self.heretic().text
        nr["num_hidden_layers"] = 8
        nr["layer_types"] = types + types
        nr["no_rope_layers"] = [1, 1, 1, 0, 1, 1, 1, 1]
        let n = try ModelSpec(ModelConfig(json: ["model_type": "qwen3_5_moe", "text_config": nr]))
        #expect(n.ropes[3]?.nope == true && n.ropes[3]?.theta == 1e7 && n.ropes[7]?.nope == false)
        #expect(n.ropes.compactMap { $0 }.count == 2 && n.ropes[0] == nil)
        // QK-norm by family (BlockSpec): x·w except Muse Glimmer (weightless), Granite, gpt-oss,
        // Nemotron-H and GLM-4.7 (none);
        // V-norm only in Gemma 4.
        for f in ModelFamily.allCases {
            let b = BlockSpec.read(text: [:], family: f, eps: 1e-6, headDim: 128, layers: 1).spec
            let qkn: NormWeight? =
                [.granite, .gptOss, .nemotronH, .glm4MoeLite].contains(f)
                ? nil : f == .museGlimmer ? .unit : .plain
            #expect(b.qkNorm == qkn, "\(f)")
            #expect(b.vNorm == (f == .gemma4 ? .unit : nil), "\(f)")
        }
    }

    /// Fields of the real config.json of LFM2.5-8B-A1B (LiquidAI/LFM2.5-8B-A1B-MLX-4bit), trimmed to what the engine
    /// reads.
    static func lfm25_8b() throws -> ModelConfig {
        try ModelConfig(json: [
            "model_type": "lfm2_moe", "hidden_size": 2048, "num_hidden_layers": 24, "num_dense_layers": 2,
            "num_attention_heads": 32, "num_key_value_heads": 8, "num_experts": 32, "num_experts_per_tok": 4,
            "moe_intermediate_size": 1792, "intermediate_size": 7168, "norm_topk_prob": true,
            "rope_parameters": ["rope_theta": 5_000_000, "rope_type": "default"], "rope_theta": 5_000_000,
            "norm_eps": 1e-5,
            "layer_types": ["conv", "conv", "full_attention"] + Array(repeating: "conv", count: 21),
        ])
    }

    /// LFM2.5-8B-A1B (the lfm2 real model of the CI): 32 experts top-4 and two dense layers; its config.json from the
    /// Hugging Face cache, if there, loads too.
    @Test func lfm25_8bLoads() throws {
        let l = try ModelSpec(Self.lfm25_8b())
        #expect(l.router?.experts == 32 && l.shape.moeE == 32 && l.shape.moeK == 4)
        #expect(l.ffnKinds == [.dense, .dense] + Array(repeating: .moe, count: 22))
        let hub = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".cache/huggingface/hub")
        let snaps = hub.appendingPathComponent("models--LiquidAI--LFM2.5-8B-A1B-MLX-4bit/snapshots")
        if let snap = try? FileManager.default.contentsOfDirectory(atPath: snaps.path).first {
            let m = Self.loadError(try ModelConfig(directory: snaps.appendingPathComponent(snap)))
            #expect(m == nil, "LFM2.5-8B-A1B: \(m ?? "loads")")
        }
    }

    /// Every key of the configs of Gemma 2/4, Muse Glimmer, Granite 4.1/4.2 and gpt-oss, and an unsupported activation
    /// (trimmed to those keys) is read into BlockSpec; what the engine cannot execute yet is a load error.
    @Test func blockKeysFromConfig() throws {
        func read(_ t: [String: Any], hd: Int = 128) -> (spec: BlockSpec, gaps: [String]) {
            BlockSpec.read(text: t, family: nil, eps: 1e-6, headDim: hd, layers: 40)
        }
        // The current families: the standard block, nothing to report (real keys of the Heretic config).
        let std = BlockSpec.standard(eps: 1e-6, headDim: 256)
        let q = read(["hidden_act": "silu", "attention_bias": false], hd: 256)
        #expect(q.spec == std && q.gaps.isEmpty && std.attnScale == 1 / Float(256).squareRoot())
        #expect(!std.needsBranchEpilogue)
        for c in [try Self.heretic(), try Self.q27(), try Self.flashNext(), try Self.lfm2()] {
            let s = try ModelSpec(c)
            #expect(s.block == .standard(eps: Float(c.rmsNormEps), headDim: s.shape.attnHD))
        }
        let gemma2 = read(
            [
                "hidden_activation": "gelu_pytorch_tanh", "query_pre_attn_scalar": 256, "attention_bias": false,
                "final_logit_softcapping": NSNull(), "attn_logit_softcapping": NSNull(),
            ], hd: 256)
        #expect(gemma2.spec.activation == .geluTanh && gemma2.spec.attnScale == 1.0 / 16 && gemma2.spec.softcap == nil)
        let gemma4 = read(["hidden_activation": "gelu_pytorch_tanh", "final_logit_softcapping": 30.0], hd: 256)
        #expect(gemma4.spec.activation == .geluTanh && gemma4.spec.softcap == 30)
        let muse = read([
            "hidden_activation": "silu", "final_logit_softcapping": 20.0, "output_multiplier": 0.19611613513818404,
            "qk_scale_factor": 3.87, "post_norm_eps": 1e-8,
        ])
        #expect(muse.spec.softcap == 20 && muse.spec.logitsMul == Float(0.19611613513818404))
        #expect(muse.spec.attnScale == Float(3.87 / Double(128).squareRoot()))
        #expect(muse.gaps.contains { $0.contains("post_norm_eps") })
        let granite41 = read([
            "hidden_act": "silu", "attention_bias": false, "mlp_bias": false, "embedding_multiplier": 12.0,
            "residual_multiplier": 0.22, "logits_scaling": 10.0, "attention_multiplier": 0.015625,
        ])
        #expect(granite41.spec.embedScale == 12 && granite41.spec.residualScale == 0.22)
        #expect(granite41.spec.logitsDiv == 10 && granite41.spec.attnScale == 0.015625 && !granite41.spec.qkvBias)
        #expect(granite41.spec.needsBranchEpilogue)
        let granite42 = read([
            "hidden_act": "silu", "attention_bias": false, "embedding_multiplier": 1.0, "residual_multiplier": 1.0,
            "logits_scaling": 1.0, "attention_multiplier": 0.0078125,
        ])
        #expect(granite42.spec.attnScale == 0.0078125 && granite42.gaps.isEmpty)
        let oss = read(["hidden_act": "silu", "swiglu_limit": 7.0, "attention_bias": true], hd: 64)
        #expect(oss.spec.qkvBias && oss.spec.oBias && oss.gaps.contains { $0.contains("swiglu_limit") })
        let relu2 = read(["mlp_hidden_act": "relu2", "attention_bias": false, "mlp_bias": false, "use_bias": false])
        #expect(relu2.gaps.isEmpty && relu2.spec.activation == .relu2)
        // Through ModelSpec: on a family the engine knows, each key gives its error (before, they loaded silently).
        func heretic(_ extra: [String: Any]) throws -> ModelConfig {
            var t = try Self.heretic().text
            t.merge(extra) { _, b in b }
            return try ModelConfig(json: ["model_type": "qwen3_5_moe", "text_config": t])
        }
        #expect(Self.loadError(try heretic(["hidden_act": "silu", "attention_bias": false, "mlp_bias": false])) == nil)
        let hybridTypes = (0..<40).map { $0 % 4 == 3 ? "full_attention" : "linear_attention" }
        let cases: [([String: Any], String)] = [
            (["hidden_activation": "relu2"], "relu² FFN without gate outside nemotron_h"),
            (["swiglu_limit": 7.0], "swiglu_limit"),
            (["attention_bias": true], "q/k/v biases with a gated q_proj"),
            (["qkv_bias": true], "q/k/v biases with a gated q_proj"),
            (["mlp_bias": true], "mlp_bias = true"),
            (["attn_logit_softcapping": 50.0], "attn_logit_softcapping"),
            (["post_norm_eps": 1e-8], "post_norm_eps"),
            // The branch epilogue only goes around attention (the real Heretic has GDN layers; the fixture
            // above has no layer_types, which means all full attention).
            (["residual_multiplier": 0.22, "layer_types": hybridTypes], "branch epilogue"),
        ]
        for (extra, want) in cases {
            let m = Self.loadError(try heretic(extra))
            #expect(m?.contains(want) == true, "\(extra): \(m ?? "loads")")
        }
        // GELU-tanh runs (LAB_FFN_GELU and the `_gelu` instances), except in Flash-Next.
        let g = try heretic(["hidden_act": "gelu_pytorch_tanh"])
        #expect(Self.loadError(g) == nil && g.spec?.block.activation == .geluTanh && g.spec!.shape.ffnGelu)
        #expect(g.spec!.shape.macros["LAB_FFN_GELU"] as? NSNumber == 1 && g.spec!.shape.ffnActSuffix == "_gelu")
        #expect(try Self.heretic().spec!.shape.macros["LAB_FFN_GELU"] == nil)
        func flash(_ extra: [String: Any]) throws -> String? {
            var t = try Self.flashNext().text
            t.merge(extra) { _, b in b }
            return Self.loadError(try ModelConfig(json: ["model_type": "qwen4_exp", "text_config": t]))
        }
        #expect(try flash(["hidden_act": "gelu_pytorch_tanh"])?.contains("in Flash-Next") == true)
        #expect(try flash(["final_logit_softcapping": 30.0])?.contains("in Flash-Next") == true)
        // Logits, embedding and attention scales run in the Qwen/LFM2 skeleton (any layer types).
        let scales: [String: Any] = [
            "final_logit_softcapping": 30.0, "output_multiplier": 0.5, "logits_scaling": 8.0,
            "embedding_multiplier": 12.0, "attention_multiplier": 0.015625,
        ]
        let sc = try heretic(scales)
        #expect(Self.loadError(sc) == nil, "\(Self.loadError(sc) ?? "")")
        #expect(sc.spec?.block.softcap == 30 && sc.spec?.block.embedScale == 12 && sc.spec?.block.logitsDiv == 8)
        #expect(sc.spec?.block.plainSkeleton == false && sc.spec?.block.needsBranchEpilogue == false)
        #expect(Self.loadError(try heretic(["query_pre_attn_scalar": 144])) == nil)
        // The branch epilogue around attention-only layers (Granite: residual multiplier 0.22, graniteFamily); q|k|v
        // and o_proj biases (attention_bias, gpt-oss's) load on an ungated attention, with LAB_AT_QKVB.
        let ab = try Self.granite(["attention_bias": true])
        #expect(Self.loadError(ab) == nil && ab.spec?.block.qkvBias == true && ab.spec?.block.oBias == true)
        #expect(ab.spec?.shape.macros["LAB_AT_QKVB"] as? NSNumber == 1)
        #expect(try Self.granite().spec?.shape.macros["LAB_AT_QKVB"] == nil)
        // The same value as the default is not a gap (Gemma 2 style, head_dim = query_pre_attn_scalar = 256).
        #expect(
            Self.loadError(try heretic(["query_pre_attn_scalar": 256, "final_logit_softcapping": NSNull()])) == nil)
    }

    /// The real config of Granite 4.1 3B (ref/toy/real/), with `extra` merged.
    static func granite(_ extra: [String: Any] = [:]) throws -> ModelConfig {
        var j = try ToyReal.load("granite-4.1-3b", root: repo).json
        j.merge(extra) { _, b in b }
        return try ModelConfig(json: j)
    }
    /// The real config of Qwen3.5 0.8B (ref/toy/real/qwen3.5-0.8b.json, mlx-community/Qwen3.5-0.8B-4bit).
    static func q35small() throws -> ModelConfig {
        try ModelConfig(json: try ToyReal.load("qwen3.5-0.8b", root: repo).json)
    }

    /// Qwen3.5 0.8B: dense qwen35 with the three shapes no other model runs (GDN with as many v heads as k heads,
    /// FFN of 3584, 8 q / 2 KV gated heads of 256), bf16 small weights (A_log is read in fp32 whatever its dtype).
    @Test func qwen35Small() throws {
        let s = try ModelSpec(Self.q35small())
        #expect(s.family == .qwen35 && s.router == nil && s.ffnKinds == Array(repeating: .dense, count: 24))
        #expect(s.shape.hidden == 1024 && s.shape.gdnHV == 16 && s.shape.gdnHK == 16 && s.shape.gdnDK == 128)
        #expect(s.shape.attnQH == 8 && s.shape.attnKVH == 2 && s.shape.attnHD == 256 && s.shape.attnGate)
        #expect(s.rope == RopeSpec(theta: 1e7, rotaryDims: 64) && !s.shape.normF32 && s.block.qkNorm == .plain)
        #expect(try Self.q35small().text["intermediate_size"] as? Int == 3584)
    }

    /// Granite 4.1 3B (ref/toy/real/granite-4.1-3b.json) loads as the granite family: attention only without gate
    /// or QK-norm, 40 q heads over 8 KV heads (G = 5), head of hidden / heads (mlx-lm), dense FFN, its μP multipliers
    /// in the block and the g32 kernels (mlx-community's conversion).
    @Test func graniteFamily() throws {
        let g = try Self.spec(try ToyReal.load("granite-4.1-3b", root: Self.repo).json)
        #expect(g.family == .granite && g.names == .qwen && !g.hasDrafter && g.router == nil)
        #expect(g.block.qkNorm == nil && !g.shape.attnQKN && !g.shape.attnGate)
        #expect(g.shape.attnHD == 64 && g.shape.attnRot == 64 && g.rope == RopeSpec(theta: 1e7, rotaryDims: 64))
        #expect(g.shape.hidden == 2560 && g.shape.attnQH == 40 && g.shape.attnKVH == 8 && g.shape.quantG32)
        #expect(g.ffnKinds.count == 40 && g.ffnKinds.allSatisfy { $0 == .dense } && !g.hasSlidingWindow)
        let b = g.block
        #expect(b.embedScale == 12 && b.residualScale == 0.22 && b.attnScale == 0.015625 && b.logitsDiv == 10)
        #expect(b.needsBranchEpilogue && !b.qkvBias && !b.oBias && b.activation == .silu && b.mixerNorm.eps == 1e-5)
        #expect(g.shape.macros["LAB_AT_QKN"] as? NSNumber == 0 && g.shape.macros["LAB_AT_QKVB"] == nil)
    }

    /// Layer types from the config: Gemma 4 31B (`[sliding ×5, full]`, window 1024) gets a window per sliding layer and
    /// LAB_AT_SWA; a sliding layer without a window and an unknown layer type are refused; models without windows
    /// compile no LAB_AT_SWA (same library as before).
    @Test func slidingWindowLayers() throws {
        let g = try ModelSpec(Self.gemma4("gemma4-31b"))
        #expect(g.windows.count == 60 && g.windows.filter { $0 == 0 }.count == 10 && g.windows[5] == 0)
        #expect(g.windows[0] == 1024 && g.shape.attnSWA)
        let noWindow = try Self.gemma4("gemma4-31b") { $0["sliding_window"] = NSNull() }
        #expect(throws: (any Error).self) { try ModelSpec(noWindow) }
        let chunked = try Self.gemma4("gemma4-31b") { t in
            t["layer_types"] = ["chunked_attention"] + ((t["layer_types"] as? [String]) ?? []).dropFirst()
        }
        #expect(throws: (any Error).self) { try ModelSpec(chunked) }
        #expect(g.shape.macros["LAB_AT_SWA"] as? NSNumber == 1)
        #expect(try ModelSpec(Self.heretic()).shape.macros["LAB_AT_SWA"] == nil)
    }

    /// The removed tier-2 families and the never registered dense Qwen3 are load errors with their name.
    @Test func removedFamiliesFail() throws {
        for (type, what) in [("qwen3_moe", "Qwen3 MoE"), ("gemma3_text", "Gemma 3"), ("qwen3", "dense Qwen3")] {
            let c = try ModelConfig(json: ["model_type": type, "hidden_size": 1024, "num_hidden_layers": 4])
            #expect(c.family == nil && ModelFamily.detect(textType: type, rawType: nil) == nil, "\(type)")
            #expect(Self.loadError(c)?.contains("\(type): not supported yet: \(what)") == true, "\(type)")
        }
    }

    @Test func unknownFamilyFails() throws {
        let c = try ModelConfig(json: ["model_type": "llama", "hidden_size": 4096, "num_hidden_layers": 32])
        #expect(c.family == nil)
        #expect(throws: SafetensorsError.self) { try ModelSpec(c) }
    }

    /// A vocabulary that is not a multiple of 64 is a load error with its reason: the lm_head has no zero-row padding
    /// (every tier-1 vocabulary is a multiple of 64, Granite 4.1's 100352 = 64 · 1568 among them).
    @Test func vocabMultipleOf64() throws {
        func withVocab(_ v: Int) throws -> ModelConfig {
            var t = try Self.heretic().text
            t["vocab_size"] = v
            return try ModelConfig(json: ["model_type": "qwen3_5_moe", "text_config": t])
        }
        #expect(try withVocab(49_159).vocabSize == 49_159)
        #expect(Self.loadError(try withVocab(248_320)) == nil)
        #expect(Self.loadError(try withVocab(100_352)) == nil)
        let m = Self.loadError(try withVocab(49_159))
        #expect(m?.contains("vocabulary 49159 not a multiple of 64") == true, "\(m ?? "loads")")
    }
}

/// Toys against their real model: each toy's config comes from the real config.json
/// (ref/toy/real/) with ToyCut's trims, and the ModelSpec of both can only differ in what is trimmed (hidden,
/// intermediates, layers: the toy is the first period of the real one). No GPU.
extension ModelSpecTests {
    static let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        .deletingLastPathComponent()

    static func spec(_ json: [String: Any]) throws -> ModelSpec {
        // The config goes through JSON as when the loader reads it.
        let data = try JSONSerialization.data(withJSONObject: json)
        let back = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        return try ModelSpec(ModelConfig(json: back))
    }

    /// Compares the toy with the real one except for the trims; returns the expected router so the caller can check
    /// its declared differences.
    static func expectMatches(
        _ real: ModelSpec, _ toy: ModelSpec, _ name: String, moeI: Int = ToyCut.moeI,
        shapeFix: (inout ModelDims) -> Void = { _ in }
    ) {
        let n = toy.ffnKinds.count
        #expect(n == ToyCut.layers, "\(name): layers")
        #expect(toy.family == real.family, "\(name): family")
        #expect(toy.ffnKinds == Array(real.ffnKinds.prefix(n)), "\(name): FFN kinds")
        #expect(toy.ropes == Array(real.ropes.prefix(n)), "\(name): RoPE per layer")
        #expect(toy.rope == real.rope && toy.block.qkNorm == real.block.qkNorm, "\(name): RoPE and QK-norm")
        #expect(toy.hasDrafter == real.hasDrafter && toy.names == real.names, "\(name): drafter and names")
        #expect(toy.expertLayout == real.expertLayout, "\(name): experts")
        if let r = real.router, let t = toy.router {
            var want = RouterSpec(
                scoring: r.scoring, shared: r.shared, experts: r.experts, topK: r.topK, inter: t.inter,
                sharedInter: t.sharedInter, selectBias: r.selectBias, scale: r.scale, normEps: r.normEps)
            want.logitBias = r.logitBias; want.expertBiases = r.expertBiases
            want.ownNorm = r.ownNorm; want.perExpertScale = r.perExpertScale
            #expect(t == want, "\(name): router")
            #expect(t.inter == moeI && t.sharedInter * r.inter == r.sharedInter * t.inter, "\(name): intermedios")
        } else {
            #expect(real.router == nil && toy.router == nil, "\(name): router")
        }
        var s = toy.shape
        s.hidden = real.shape.hidden; s.moeI = real.shape.moeI
        shapeFix(&s)
        #expect(s == real.shape, "\(name): shape except hidden and intermediates")
    }

    @Test func toyMatchesReal() throws {
        func real(_ n: String) throws -> ToyReal { try ToyReal.load(n, root: Self.repo) }
        let q36 = try real("qwen3.6-35b-a3b"), q27 = try real("qwen3.8-27b")
        let fn = try real("qwen3.8-flash-next"), lfm = try real("lfm2.5-24b-a2b")
        func qwen(_ r: ToyReal) throws -> ModelSpec {
            try Self.spec(ToyConfig.qwen(ToyDims.qwen(r), quant: [:], seed: 1))
        }
        let heretic = try Self.spec(q36.json)
        Self.expectMatches(heretic, try qwen(q36), "toy-qwen36-moe")
        Self.expectMatches(try Self.spec(q27.json), try qwen(q27), "toy-qwen38-dense")
        let fnToy = try Self.spec(
            ToyConfig.flash(ToyDims.flashNext(fn), pleLayer: 1, quant: [:], metalLab: [:], seed: 1))
        Self.expectMatches(try Self.spec(fn.json), fnToy, "toy-flashnext")
        #expect(fnToy.shape.hidden == 2560, "toy-flashnext: real hidden")

        // toy-lfm2: only the cuts, no declared deviations (2 dense layers, θ 1e7, 131072 positions).
        let lfmReal = try Self.spec(lfm.json)
        let toyLFM = try Self.spec(ToyConfig.lfm(ToyDims.lfm(lfm), quant: [:], seed: 1))
        Self.expectMatches(lfmReal, toyLFM, "toy-lfm2")
        #expect(toyLFM.ffnKinds == [.dense, .dense, .moe, .moe] && toyLFM.rope.theta == 1e7)

        // toy-granite: besides the cuts (4 layers), attention_bias true (ToyConfig.graniteAttentionBias).
        let gr = try real("granite-4.1-3b")
        let grReal = try Self.spec(gr.json)
        let grDims = ToyDims.granite(gr)
        #expect(grDims.realGroup4 == 32, "toy-granite: g32 like the real one")
        let grQuant: [String: Any] = ["group_size": grDims.realGroup4, "bits": 4]
        let grToy = try Self.spec(ToyConfig.granite(grDims, quant: grQuant, seed: 1))
        Self.expectMatches(grReal, grToy, "toy-granite") { s in s.attnQKVB = false }
        var grBlock = grToy.block
        grBlock.qkvBias = false; grBlock.oBias = false
        #expect(grBlock == grReal.block && grToy.block.qkvBias && grToy.block.oBias, "toy-granite: block")
        #expect(grToy.shape.hidden == grReal.shape.hidden, "toy-granite: real hidden (head = hidden / heads)")

        // toy-gptoss: only the cuts (hidden 576 and expert intermediate 192, both multiples of 64 and not of 128 like the
        // real 2880); since v2 its experts are MXFP4 like the real MXFP4-Q8 (ToyConfig.gptOssExpertQuant, the
        // quantization default, which the real config in ref/toy/real/ does not carry).
        let go = try real("gpt-oss-20b")
        let goToy = try Self.spec(ToyConfig.gptOss(ToyDims.gptOss(go), quant: [:], seed: 1))
        let goReal = try Self.spec(go.json)
        Self.expectMatches(goReal, goToy, "toy-gptoss", moeI: ToyCut.gptOssMoeI)
        #expect(goToy.block == goReal.block && goToy.windows == [128, 0, 128, 0], "toy-gptoss: block and window")
        #expect(goToy.shape.hidden == 576 && goToy.shape.moeH64 && goToy.shape.attnQH == 64, "toy-gptoss: shape")
        #expect(
            ToyConfig.gptOssExpertQuant["bits"] as? Int == 4 && ToyConfig.gptOssExpertQuant["group_size"] as? Int == 32
                && ToyConfig.gptOssExpertQuant["mode"] as? String == "mxfp4")
    }
}

// MARK: - Attention sinks

extension ModelSpecTests {
    /// Learned attention sinks: `softmax_type: learnable` turns on LAB_AT_SINKS (Granite 4.1 3B's config); other softmax
    /// types are refused. gpt-oss has them by family (its config carries no softmax_type).
    @Test func attentionSinks() throws {
        let s = try ModelSpec(Self.granite(["softmax_type": "learnable"]))
        #expect(s.shape.attnSinks && s.shape.macros["LAB_AT_SINKS"] as? NSNumber == 1 && s.names.sinks == "sinks")
        #expect(try ModelSpec(Self.granite()).shape.macros["LAB_AT_SINKS"] == nil)
        #expect(try ModelSpec(Self.granite(["softmax_type": "vanilla"])).shape.attnSinks == false)
        #expect(throws: (any Error).self) { try ModelSpec(Self.granite(["softmax_type": "sparsemax"])) }
        let go = try ToyReal.load("gpt-oss-20b", root: Self.repo)
        let toy = try Self.spec(ToyConfig.gptOss(ToyDims.gptOss(go), quant: [:], seed: 1))
        #expect(toy.shape.attnSinks && toy.windows == [128, 0, 128, 0], "toy-gptoss")
    }
}
