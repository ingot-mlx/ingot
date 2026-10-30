import Foundation
import IngotBench
import Metal
import Testing

@testable import Ingot

/// GLM-4.7-Flash in the loader, from the config of mlx-community/GLM-4.7-Flash-4bit (revision 1454cffb) in
/// ref/toy/real/ (no weights, no GPU): the family is recognized, the router, the dense first layer, the ungated shared
/// expert and the MLA shape read as mlx-lm's glm4_moe_lite, the interleaved RoPE moves to the engine's convention by a
/// row permutation and the per-head MultiLinear tensors read as one matrix; the MLA runs, over the latent cache.
extension ModelSpecTests {

    /// The real GLM-4.7-Flash config (ref/toy/real/glm-4.7-flash.json), edited, through JSON as the loader reads it.
    static func glm(_ edit: (inout [String: Any]) -> Void = { _ in }) throws -> ModelConfig {
        var j = try ToyReal.load("glm-4.7-flash", root: repo).json
        edit(&j)
        let data = try JSONSerialization.data(withJSONObject: j)
        return try ModelConfig(json: try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any]))
    }

    /// The Flash: 47 layers, all MLA attention; layer 0 dense SwiGLU of 10240 and 46 MoE of 64 experts top-4 of 1536
    /// with a sigmoid router (selection bias, scale 1.8, no epsilon) and an ungated shared expert of 1536; MLA of 20
    /// heads, ranks 768/512, heads 192 + 64 and v of 256.
    @Test func glmFlash() throws {
        let c = try Self.glm()
        #expect(c.family == .glm4MoeLite && c.modelType == "glm4_moe_lite")
        #expect(c.hiddenSize == 2048 && c.numLayers == 47 && c.vocabSize == 154_880 && c.rmsNormEps == 1e-5)
        #expect(c.layerTypes == Array(repeating: "full_attention", count: 47))
        #expect(ModelSpec.ffnKinds(c) == [.dense] + Array(repeating: .moe, count: 46))
        // MoE: n_routed_experts and n_shared_experts by key; the router by family (no scoring_func in the config).
        #expect(c.isMoE && c.numExperts == 64 && c.expertsPerToken == 4 && c.moeIntermediateSize == 1536)
        #expect(c.intermediateSize == 10240 && c.sharedExpertIntermediateSize == nil)
        #expect(ModelSpec.sharedKind(c) == .ungated && c.text["scoring_func"] == nil)
        let rk = ModelSpec.routerKeys(c.text, family: .glm4MoeLite)
        #expect(rk.scoring == .sigmoid && rk.selectBias && rk.scale == 1.8 && rk.normEps == 0)
        let d = ModelDims(c)
        #expect(d.moeE == 64 && d.moeK == 4 && d.moeI == 1536 && d.moeShared == 2 && d.moeSigmoid && d.moeBias)
        #expect(d.moeRScale == 1.8 && d.moeEps == 0 && !d.moeH64 && !d.moeLBias && !d.moePES && !d.moeMXFP4)
        #expect(d.ffnActSuffix == "" && !d.quantG32)
        // The MoE kernels take it: the common sigmoid router with the ungated shared expert (LAB_MOE_SHARED 2).
        let r = RouterSpec(
            scoring: rk.scoring, shared: .ungated, experts: 64, topK: 4, inter: 1536, sharedInter: 1536,
            selectBias: rk.selectBias, scale: rk.scale, normEps: rk.normEps)
        let ks = try MoEKernelSet.forRouter(r, inter: 1536)
        #expect(ks.route == "vmoe_route" && ks.combineNoShared == nil)
        // MLA over the latent cache: 20 query heads on one KV head of 64 + 512 = 576 ([k_pe | c], G 20), no gate,
        // no QK-norm, RoPE on 64 (not 576 · partial_rotary_factor 1.0), θ 1e6 in every layer; the attention output is
        // 20 × 576 (o_proj reads the 20 × 256 of unembed_out).
        #expect(d.attnQH == 20 && d.attnKVH == 1 && d.attnHD == 576 && d.attnRot == 64 && d.attnG == 20)
        #expect(!d.attnGate && !d.attnQKN && !d.attnSWA && !d.attnSinks && !d.attnRopeTab && !d.attnVN && !d.attnQKVB)
        #expect(d.attnProjN == 22 * 576 && d.attnOutDim == 20 * 576)
        #expect(d.macros["LAB_AT_ROT"] == NSNumber(value: 64) && d.macros["LAB_AT_HD"] == NSNumber(value: 576))
        let ropes = ModelSpec.layerRopes(c, base: d, full: nil)
        #expect(ropes.count == 47 && ropes.allSatisfy { $0 == RopeSpec(theta: 1e6, rotaryDims: 64) })
        let block = BlockSpec.read(text: c.text, family: c.family, eps: 1e-5, headDim: d.attnHD, layers: 47)
        #expect(block.gaps.isEmpty && block.spec.qkNorm == nil && block.spec.activation == .silu)
        // The scale is that of the q/k head before absorption (192 + 64), not 576^-0.5.
        #expect(block.spec.attnScale == Float(1) / 16)
        let m = try #require(MLASpec(c.text))
        #expect(m.heads == 20 && m.qLoraRank == 768 && m.kvLoraRank == 512 && m.kvAWidth == 576)
        #expect(m.qkNopeHeadDim == 192 && m.qkRopeHeadDim == 64 && m.vHeadDim == 256 && m.qHeadDim == 256)
        #expect(m.latentHeadDim == 576)
        // No gap: the MLA runs.
        #expect(c.spec != nil && Self.gapList(c) == nil)
        // Plan: the latent KV (1 head × (576 + 2 · 4) bytes per layer: K with two scales, V read from K; 47 layers:
        // 27.5 KB per token) and the family's measured calibration.
        let fp = ModelFootprint(
            name: "glm", config: c, weightBytes: 0, largestTensorBytes: 0, drafterConfig: nil, drafterWeightBytes: nil)
        #expect(fp.kvBytesPerTokenPerLayer == 584 && fp.kvBytesPerToken == 47 * 584 + 8)
        #expect(fp.calibration == .glm4MoeLite && !fp.hasDrafter && fp.windowLayers == 0 && fp.snapshotBaseBytes == 0)
        #expect(fp.nativeContext == 202_752)
    }

    /// What the engine does not support is a load error with its name.
    @Test func glmGaps() throws {
        func gaps(_ edit: @escaping (inout [String: Any]) -> Void) throws -> [String]? {
            Self.gapList(try Self.glm(edit))
        }
        #expect(
            try gaps { $0["q_lora_rank"] = NSNull() } == [
                "MLA without q_lora_rank (a direct q_proj, "
                    + "DeepSeek-V2-Lite: not supported)"
            ])
        #expect(
            try gaps { $0["rope_scaling"] = ["type": "yarn", "factor": 40, "mscale_all_dim": 1.0] }?.contains(
                "MLA RoPE scaling with mscale_all_dim (DeepSeek's YaRN attention scale: not supported)") == true)
        #expect(try gaps { $0["moe_layer_freq"] = 2 } == ["moe_layer_freq 2 (dense layers between MoE ones)"])
        // v_head_dim is free (unembed_out per head); the latent head is the kernels' 64 + 512.
        #expect(try gaps { $0["v_head_dim"] = 128 } == nil)
        #expect(try gaps { $0["v_head_dim"] = 100 } == ["MLA heads of nope 192 and v 100 (multiples of 64)"])
        #expect(
            try gaps { $0["qk_rope_head_dim"] = 32 } == [
                "MLA latent head of 32 + 512 (the attention kernels have 64 + 512 only)",
                "attention with head of 544 (the attention kernels accept heads of 64, 128, 256 and 512, and 576 for "
                    + "MLA's latent cache)",
            ])
        #expect(try gaps { $0["attention_bias"] = true } == ["MLA projection biases (attention_bias)"])
        #expect(
            try gaps { $0["kv_lora_rank"] = 448 } == [
                "MLA latent head of 64 + 448 (the attention kernels have 64 + 512 only)"
            ])
        #expect(
            try gaps { $0["kv_lora_rank"] = nil } == [
                "glm4_moe_lite without its MLA keys (kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim, v_head_dim)"
            ])
        // Grouped routing and a softmax router with the selection bias: the errors every family has.
        let grouped = try gaps { $0["n_group"] = 2 }
        #expect(grouped == ["grouped top-k routing (n_group 2, topk_group 1)"])
        let softmax = try gaps { $0["scoring_func"] = "softmax" }
        #expect(softmax?.contains("selection bias with a softmax router") == true)
        // Without topk_method, the family's selection bias (mlx-lm's default noaux_tc); with greedy, none.
        #expect(ModelSpec.routerKeys(try Self.glm { $0["topk_method"] = nil }.text, family: .glm4MoeLite).selectBias)
        #expect(
            !ModelSpec.routerKeys(try Self.glm { $0["topk_method"] = "greedy" }.text, family: .glm4MoeLite).selectBias)
        // The MTP key is not an error (the MLX conversions drop its layer).
        #expect(try gaps { $0["num_nextn_predict_layers"] = 0 } == nil)
    }

    /// toy-glm is the real config with only the cuts (4 layers: the dense one and 3 MoE; hidden 1024; expert
    /// and shared intermediates 128): the real MLA, router, ungated shared expert and RoPE.
    @Test func glmToyMatchesReal() throws {
        let r = try ToyReal.load("glm-4.7-flash", root: Self.repo)
        let real = try Self.spec(r.json), toy = try Self.spec(ToyConfig.glm(ToyDims.glm(r), quant: [:], seed: 1))
        Self.expectMatches(real, toy, "toy-glm")
        #expect(toy.ffnKinds == [.dense, .moe, .moe, .moe] && toy.mla == real.mla && toy.shape.hidden == 1024)
        #expect(toy.block == real.block, "toy-glm: block")
    }

    /// The row orders move the interleaved RoPE (pairs (2i, 2i + 1) on the last 64 dimensions of each q head, and on
    /// k_pe after the latent) to the engine's (pairs (i, i + 32) on the first 64): rotating the permuted rows by halves
    /// gives the permuted rows of the interleaved rotation, so q·k does not change.
    @Test func glmRopeRowOrder() throws {
        let m = try #require(MLASpec(try Self.glm().text))
        #expect(MLASpec.deinterleave(8) == [0, 2, 4, 6, 1, 3, 5, 7])
        let qo = m.qBRowOrder, ko = m.kvARowOrder
        #expect(qo.count == 5120 && Set(qo).count == 5120 && ko.count == 576 && Set(ko).count == 576)
        #expect(Array(qo.prefix(4)) == [192, 194, 196, 198] && qo[32] == 193 && qo[64] == 0 && qo[255] == 191)
        #expect(qo[256] == 256 + 192 && qo[256 + 64] == 256)
        #expect(Array(ko.prefix(512)) == Array(0..<512) && ko[512] == 512 && ko[513] == 514 && ko[544] == 513)
        // The rotations as mlx (traditional) and as the engine's prework (halves), in Double.
        let rot = 64, theta = 1e6
        func freq(_ i: Int) -> Double { pow(theta, -Double(2 * i) / Double(rot)) }
        func interleaved(_ x: [Double], at base: Int, pos: Int) -> [Double] {
            var y = x
            for i in 0..<rot / 2 {
                let a = Double(pos) * freq(i), x0 = x[base + 2 * i], x1 = x[base + 2 * i + 1]
                y[base + 2 * i] = x0 * cos(a) - x1 * sin(a)
                y[base + 2 * i + 1] = x0 * sin(a) + x1 * cos(a)
            }
            return y
        }
        func halves(_ x: [Double], pos: Int) -> [Double] {
            var y = x
            for i in 0..<rot / 2 {
                let a = Double(pos) * freq(i), x0 = x[i], x1 = x[i + rot / 2]
                y[i] = x0 * cos(a) - x1 * sin(a)
                y[i + rot / 2] = x0 * sin(a) + x1 * cos(a)
            }
            return y
        }
        var rng = SplitMix64(seed: 7)
        let d = m.qHeadDim
        // One q head (rows 0..<256 of q_b: [q_nope | q_pe]) at position 37, and k_pe (rows 512..<576 of kv_a) at 11.
        let q = (0..<d).map { _ in Double(rng.uniform()) * 2 - 1 }
        let kva = (0..<m.kvAWidth).map { _ in Double(rng.uniform()) * 2 - 1 }
        let qRef = interleaved(q, at: m.qkNopeHeadDim, pos: 37), kRef = interleaved(kva, at: m.kvLoraRank, pos: 11)
        let qEng = halves(qo.prefix(d).map { q[$0] }, pos: 37)
        let kPe = halves(ko.suffix(rot).map { kva[$0] }, pos: 11)
        for j in 0..<d { #expect(abs(qEng[j] - qRef[qo[j]]) < 1e-12, "q row \(j)") }
        for j in 0..<rot { #expect(abs(kPe[j] - kRef[ko[m.kvLoraRank + j]]) < 1e-12, "k_pe row \(j)") }
        // The RoPE part of q·k: q_pe·k_pe is the same in both layouts.
        let ref = (0..<rot).reduce(0.0) { $0 + qRef[m.qkNopeHeadDim + $1] * kRef[m.kvLoraRank + $1] }
        let eng = (0..<rot).reduce(0.0) { $0 + qEng[$1] * kPe[$1] }
        #expect(abs(ref - eng) < 1e-12)
    }

    /// A per-head MultiLinear (embed_q [heads, out, in], groups along `in`) reads as one matrix of heads · out rows:
    /// row h·out + r is head h's row r. Without the flag the leading dimension stays stacked experts, and a 2D tensor
    /// is not a MultiLinear.
    @Test func glmMultiLinear() throws {
        let fmt = FormatTests()
        let dir = try fmt.tmpDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let H = 3, out = 4, K = 128
        let q = (0..<(H * out * K)).map { UInt32(($0 * 5 + 1) % 16) }
        let s = (0..<(H * out * 2)).map { floatToBF16(0.25 + Float($0) / 8) }
        let b = (0..<(H * out * 2)).map { floatToBF16(Float($0 % 5) - 2) }
        func data<T>(_ a: [T]) -> Data { a.withUnsafeBytes { Data($0) } }
        let p = "model.layers.0.self_attn.embed_q"
        try fmt.writeSafetensors(
            dir,
            tensors: [
                (p + ".weight", "U32", [H, out, K / 8], data(packQ(q, bits: 4))),
                (p + ".scales", "BF16", [H, out, K / 64], data(s)),
                (p + ".biases", "BF16", [H, out, K / 64], data(b)),
                (
                    "model.layers.0.self_attn.o_proj.weight", "U32", [out, K / 8],
                    data(packQ(Array(q.prefix(out * K)), bits: 4))
                ),
                ("model.layers.0.self_attn.o_proj.scales", "BF16", [out, K / 64], data(Array(s.prefix(out * 2)))),
                ("model.layers.0.self_attn.o_proj.biases", "BF16", [out, K / 64], data(Array(b.prefix(out * 2)))),
            ])
        let cfg = try ModelConfig(json: [
            "hidden_size": K, "num_hidden_layers": 1, "quantization": ["group_size": 64, "bits": 4],
        ])
        let st = try SafetensorsStore(directory: dir)
        let ml = try QuantizedLinear(store: st, path: p, config: cfg, multiLinear: true)
        #expect(ml.outFeatures == H * out && ml.experts == 1 && ml.inFeatures == K && ml.params.bits == 4)
        let stacked = try QuantizedLinear(store: st, path: p, config: cfg)
        #expect(stacked.outFeatures == out && stacked.experts == H && stacked.inFeatures == K)
        let flat = st.dequantizeRows(ml, rows: 0..<(H * out))
        for h in 0..<H {
            let head = st.dequantizeRows(stacked, expert: h, rows: 0..<out)
            #expect(Array(flat[(h * out * K)..<((h + 1) * out * K)]) == head, "head \(h)")
        }
        #expect(throws: SafetensorsError.self) {
            try QuantizedLinear(store: st, path: "model.layers.0.self_attn.o_proj", config: cfg, multiLinear: true)
        }
        // DenseQ: the same bytes, N = heads · out (the stacked reading is not a dense matrix).
        let dev = try #require(MTLCreateSystemDefaultDevice())
        let dq = try DenseQ(store: st, config: cfg, paths: [p], device: dev, multiLinear: true)
        #expect(dq.N == H * out && dq.K == K && dq.bits == 4 && dq.gs == 64)
        #expect(memcmp(dq.w.contents(), st.bytes(ml.weight).baseAddress!, ml.weight.byteCount) == 0)
        #expect(throws: SafetensorsError.self) { try DenseQ(store: st, config: cfg, paths: [p], device: dev) }
    }

    /// Detection, calibration name, names (Qwen's with `shared_experts.` and the MLA projections) and the keys only GLM
    /// reads by family.
    @Test func glmDetectionAndNames() throws {
        #expect(ModelFamily.detect(textType: "glm4_moe_lite", rawType: "glm4_moe_lite") == .glm4MoeLite)
        #expect(ModelFamily.detect(textType: "glm4_moe", rawType: "glm4_moe") == nil)
        #expect(ModelFamily.glm4MoeLite.calibrationName == "glm4_moe_lite")
        #expect(FamilyCalibration.forFamily(.glm4MoeLite) == .glm4MoeLite && FamilyCalibration.glm4MoeLite.measured)
        #expect(!ModelSpec.attnGated(.glm4MoeLite) && !ModelSpec.moeBiases(.glm4MoeLite))
        let n = ModelSpec.weightNames(.glm4MoeLite)
        #expect(n.attn == "self_attn." && n.ffn == "mlp." && n.oProj == "o_proj" && n.embedding == "embed_tokens")
        #expect(n.inputNorm == "input_layernorm.weight" && n.postNorm == "post_attention_layernorm.weight")
        #expect(n.finalNorm == "norm.weight" && n.router == "gate" && n.experts == "switch_mlp.")
        #expect(n.sharedExpert == "shared_experts." && n.expertGate == "gate_proj" && n.expertDown == "down_proj")
        let mla = try #require(n.mla)
        #expect(mla.qA == "q_a_proj" && mla.qANorm == "q_a_layernorm.weight" && mla.qB == "q_b_proj")
        #expect(mla.kvA == "kv_a_proj_with_mqa" && mla.kvANorm == "kv_a_layernorm.weight")
        #expect(mla.embedQ == "embed_q" && mla.unembedOut == "unembed_out")
        // No other family has MLA names, and Qwen's keep `shared_expert.`.
        for f in ModelFamily.allCases where f != .glm4MoeLite { #expect(ModelSpec.weightNames(f).mla == nil, "\(f)") }
        #expect(ModelSpec.weightNames(.qwen35Moe).sharedExpert == "shared_expert.")
        // The sigmoid router by family only for GLM among the families without scoring_func: Qwen stays softmax.
        let h = try Self.heretic()
        #expect(ModelSpec.routerKeys(h.text, family: h.family).scoring == .softmax)
        #expect(ModelDims(h).attnRot == 64 && ModelDims(h).attnKVH == 2 && ModelDims(h).attnHD == 256)
    }
}
