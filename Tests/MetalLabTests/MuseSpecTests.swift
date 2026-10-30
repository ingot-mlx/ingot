import Foundation
import LabBench
import Metal
import Testing

@testable import MetalLab

/// Muse Glimmer in the loader, from the config of mlx-community/Muse-Glimmer-30B-4bit (revision 3e7677d7) in
/// ref/toy/real/ (no weights, no GPU): the family is recognized from the text and the outer model_type, the block reads
/// as mlx-lm's muse_glimmer (centered norms, post-norms with their own epsilon, weightless embedding norm and QK-norm,
/// qk_scale_factor, output multiplier and softcap), NoPE comes from `layer_rope_theta`, the separate gate_proj is
/// interleaved with q_proj into the gated q row, and what does not run yet (its DFlash2, odd θ lists, a gate of
/// another shape) is a load error with its name.
extension ModelSpecTests {

    /// The real Muse Glimmer config (ref/toy/real/muse-glimmer-30b.json), edited, through JSON as the loader reads it.
    static func muse(_ edit: (inout [String: Any]) -> Void = { _ in }) throws -> ModelConfig {
        var j = try ToyReal.load("muse-glimmer-30b", root: repo).json
        edit(&j)
        let data = try JSONSerialization.data(withJSONObject: j)
        return try ModelConfig(json: try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any]))
    }

    /// Edits the `text_config` of the real config.
    static func museText(_ edit: @escaping (inout [String: Any]) -> Void) throws -> ModelConfig {
        try muse { j in
            var t = j["text_config"] as? [String: Any] ?? [:]
            edit(&t)
            j["text_config"] = t
        }
    }

    /// The 30B: 52 attention layers `[sliding ×3, full]` (window 2048 in the sliding ones, NoPE in the full ones), GQA
    /// 32/2 of 128 with full RoPE θ 5e5 and the gate, dense SwiGLU of 19,968, vocabulary 202,048 untied.
    @Test func museGlimmer() throws {
        let c = try Self.muse()
        #expect(c.family == .museGlimmer && c.modelType == "muse_glimmer_text")
        #expect(c.hiddenSize == 6656 && c.numLayers == 52 && c.vocabSize == 202_048 && c.rmsNormEps == 1e-5)
        #expect(c.intermediateSize == 19968 && !c.isMoE && c.quantDefault == nil)
        let pattern = ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"]
        #expect(c.layerTypes == Array((0..<13).map { _ in pattern }.joined()))
        let s = try #require(c.spec, "\(Self.loadError(c) ?? "")")
        #expect(Self.gapList(c) == nil)
        #expect(s.router == nil && s.ffnKinds == Array(repeating: .dense, count: 52) && s.expertLayout == .unsupported)
        #expect(s.windows == c.layerTypes.map { $0 == "sliding_attention" ? 2048 : 0 })
        // RoPE: θ 5e5 on the whole head in the sliding layers; NoPE (layer_rope_theta 0) in the 13 full ones.
        for (i, ty) in c.layerTypes.enumerated() {
            let want = RopeSpec(theta: 5e5, rotaryDims: 128, nope: ty == "full_attention")
            #expect(s.ropes[i] == want, "layer \(i)")
        }
        #expect(s.ropes.filter { $0?.nope == true }.count == 13 && !s.hasRopeTable && s.hasSlidingWindow)
        // Attention: 32 q / 2 KV heads of 128 (G 16), gate, weightless QK-norm (kernels with QKN and a buffer of ones),
        // window, no sinks or biases; the q|gate|k|v row of 8704 and an output of 4096.
        let d = s.shape
        #expect(d.attnQH == 32 && d.attnKVH == 2 && d.attnHD == 128 && d.attnRot == 128 && d.attnG == 16)
        #expect(d.attnGate && d.attnQKN && d.attnSWA && !d.attnSinks && !d.attnRopeTab && !d.attnVN && !d.attnQKVB)
        #expect(d.attnProjN == 8704 && d.attnOutDim == 4096 && s.fullShape == nil && !s.kEqV)
        #expect(d.macros["LAB_AT_GATE"] as? NSNumber == 1 && d.macros["LAB_AT_HD"] as? NSNumber == 128)
        #expect(d.ffnActSuffix == "" && !d.quantG32)
        // Block (mlx-lm muse_glimmer.py): input and pre-FFN norms 1 + w with 1e-5, post-norms 1 + w with 1e-8, final
        // x·w, weightless embedding norm with 1e-5, SwiGLU, scale 3.87/√128, logits × 0.196 then softcap 20.
        let b = s.block
        #expect(b.mixerNorm == NormSpec(.onePlus, eps: 1e-5) && b.ffnNorm == NormSpec(.onePlus, eps: 1e-5))
        #expect(b.mixerPostNorm == NormSpec(.onePlus, eps: 1e-8) && b.ffnPostNorm == NormSpec(.onePlus, eps: 1e-8))
        #expect(b.finalNorm == NormSpec(.plain, eps: 1e-5) && b.embedNorm == NormSpec(.unit, eps: 1e-5))
        #expect(b.qkNorm == .unit && b.vNorm == nil && b.activation == .silu)
        #expect(b.attnScale == Float(3.87 / Double(128).squareRoot()) && abs(b.attnScale - 0.342) < 1e-3)
        #expect(b.logitsMul == Float(0.19611613513818404) && b.logitsDiv == 1 && b.softcap == 20)
        #expect(b.embedScale == 1 && b.residualScale == 1 && !b.layerScalar && !b.qkvBias && !b.oBias)
        #expect(b.needsBranchEpilogue && !b.plainSkeleton)
        // incoai's DFlash2 by default (the pack's drafter/ goes first); prefill int8.
        #expect(s.hasDrafter && s.defaultDrafter == "incoai/Muse-Glimmer-30B-DFlash2" && s.int8MinPos == 1024)
        #expect(!s.int8Experts)
        #expect(s.calibration == .museGlimmer && s.calibration.measured)
    }

    /// Detection (text model_type, or the outer one alone), calibration name, names and the family's switches.
    @Test func museDetectionAndNames() throws {
        #expect(ModelFamily.detect(textType: "muse_glimmer_text", rawType: "muse_glimmer") == .museGlimmer)
        #expect(ModelFamily.detect(textType: "muse_glimmer", rawType: "muse_glimmer") == .museGlimmer)
        #expect(ModelFamily.detect(textType: "muse_glimmer_text", rawType: nil) == .museGlimmer)
        #expect(ModelFamily.detect(textType: "muse_glimmer_assistant", rawType: "muse_glimmer_assistant") == nil)
        #expect(ModelFamily.museGlimmer.calibrationName == "muse_glimmer")
        #expect(FamilyCalibration.forFamily(.museGlimmer) == .museGlimmer)
        #expect(ModelSpec.attnGated(.museGlimmer) && !ModelSpec.moeBiases(.museGlimmer))
        // The flattened config (mlx-lm accepts both): the same family and spec.
        var flat = try ToyReal.load("muse-glimmer-30b", root: Self.repo).text
        flat["model_type"] = "muse_glimmer"
        let f = try ModelConfig(json: flat)
        let nested = try Self.muse()
        #expect(f.family == .museGlimmer && f.spec?.ropes == nested.spec?.ropes && f.spec?.block == nested.spec?.block)
        let n = ModelSpec.weightNames(.museGlimmer)
        #expect(n.attn == "self_attn." && n.ffn == "mlp." && n.oProj == "o_proj" && n.embedding == "embed_tokens")
        #expect(n.inputNorm == "input_layernorm.weight" && n.postNorm == "pre_feedforward_layernorm.weight")
        #expect(n.mixerPostNorm == "post_attention_layernorm.weight")
        #expect(n.ffnPostNorm == "post_feedforward_layernorm.weight" && n.finalNorm == "norm.weight")
        #expect(n.embedNorm == nil && n.attnGate == "gate_proj" && n.mla == nil)
        for fam in ModelFamily.allCases where fam != .museGlimmer && fam != .lfm3 {
            #expect(ModelSpec.weightNames(fam).attnGate == nil, "\(fam)")
        }
        // The vision tower, its adapter and its projection are not loaded nor planned.
        for t in ["vision_tower.layers.0.mlp.fc1.weight", "vision_adapter.fc1.weight", "vision_projection.weight"] {
            #expect(SafetensorsStore.isSkipped(t), "\(t)")
        }
        #expect(!SafetensorsStore.isSkipped("language_model.model.layers.0.self_attn.gate_proj.weight"))
    }

    /// The block keys read with the family: nothing to report (with no family, `post_norm_eps` is still a gap).
    @Test func museBlockKeys() throws {
        let t = try ToyReal.load("muse-glimmer-30b", root: Self.repo).text
        let b = BlockSpec.read(text: t, family: .museGlimmer, eps: 1e-5, headDim: 128, layers: 52)
        #expect(b.gaps.isEmpty, "\(b.gaps)")
        #expect(b.spec.mixerPostNorm?.eps == 1e-8 && b.spec.ffnPostNorm?.eps == 1e-8 && b.spec.mixerNorm.eps == 1e-5)
        let none = BlockSpec.read(text: t, family: nil, eps: 1e-5, headDim: 128, layers: 52)
        #expect(none.gaps.contains { $0.contains("post_norm_eps") })
    }

    /// NoPE from `layer_rope_theta`: 0 runs without RoPE in any layer; without the list, every layer rotates (mlx-lm then
    /// builds it from the layer types, which this config never needs). A θ other than 0 and the global one, or a list
    /// that does not cover the layers, is a load error.
    @Test func museLayerRopeTheta() throws {
        let c = try Self.museText { t in
            var lt = t["layer_rope_theta"] as? [Double] ?? []
            lt[1] = 0
            t["layer_rope_theta"] = lt
        }
        let s = try #require(c.spec)
        #expect(s.ropes[1]?.nope == true && s.ropes[0]?.nope == false && s.ropes[3]?.nope == true)
        // The same list on another family (θ per layer is generic).
        var g = try ToyReal.load("granite-4.1-3b", root: Self.repo).json
        let theta = Double(RopeSpec.theta(g))
        g["layer_rope_theta"] = (0..<((g["num_hidden_layers"] as? Int) ?? 0)).map { $0 == 2 ? 0.0 : theta }
        let gr = try ModelConfig(json: g)
        if let r = gr.spec?.ropes {
            #expect(r[2]?.nope == true && r[0]?.nope == false)
        } else {
            Issue.record("granite with layer_rope_theta: \(Self.loadError(gr) ?? "")")
        }
        let odd = try Self.museText { t in
            var lt = t["layer_rope_theta"] as? [Double] ?? []
            lt[4] = 1e6
            lt[8] = 1e6
            t["layer_rope_theta"] = lt
        }
        let og = try #require(Self.gapList(odd))
        #expect(og.count == 1 && og[0].contains("layer_rope_theta 1000000.0 in layer 4"), "\(og)")
        #expect(og.first?.contains("2 layer(s) in all") == true)
        let short = try Self.museText { t in t["layer_rope_theta"] = [500000.0, 0] }
        #expect(Self.gapList(short) == ["layer_rope_theta with 2 values for 52 layers"])
        // A NoPE global layer whose θ is also the global one is not a gap (θ 0 is read only as NoPE).
        let rot = try Self.museText { t in t["layer_rope_theta"] = Array(repeating: 500000.0, count: 52) }
        #expect(Self.gapList(rot) == nil && rot.spec?.ropes.allSatisfy { $0?.nope == false } == true)
    }

    /// toy-muse is the real config with only the cuts (4 layers, one period: sliding ×3 and full; hidden 1024;
    /// intermediate 9216) and the softcap of 20.5: the real attention shape (32/2 of 128, gate, weightless QK-norm,
    /// window 2048), the RoPE per layer (θ 5e5 on the whole head, NoPE in the full one from `layer_rope_theta`), the
    /// names (separate gate_proj) and the block.
    @Test func museToyMatchesReal() throws {
        let r = try ToyReal.load("muse-glimmer-30b", root: Self.repo)
        let real = try Self.spec(r.json), toy = try Self.spec(ToyConfig.muse(ToyDims.muse(r), quant: [:], seed: 1))
        Self.expectMatches(real, toy, "toy-muse")
        #expect(toy.ffnKinds == [.dense, .dense, .dense, .dense] && toy.shape.hidden == 1024)
        #expect(toy.ropes[3]?.nope == true && toy.ropes[0]?.nope == false && toy.ropes[0]?.rotaryDims == 128)
        var block = toy.block
        #expect(block.softcap == Float(ToyConfig.museSoftcap))
        block.softcap = real.block.softcap
        #expect(block == real.block, "toy-muse: block")
        #expect(toy.windows == Array(real.windows.prefix(4)) && toy.windows == [2048, 2048, 2048, 0])
    }

    /// The memory plan from the config: KV int8 of the 13 full layers (2 · 2 heads × (128 + 4) bytes each, 6.9 KB per
    /// token) and the 39 sliding ones in rings of 4224 slots with 2048-row chunks; no drafter here; the family's measured calibration.
    @Test func museFootprint() throws {
        let c = try Self.muse()
        let fp = ModelFootprint(
            name: "muse", config: c, weightBytes: 15_670_313_984, largestTensorBytes: 0, drafterConfig: nil,
            drafterWeightBytes: nil)
        #expect(fp.kvBytesPerTokenPerLayer == 528 && fp.kvBytesPerToken == 13 * 528 + 8)
        #expect(fp.windowLayers == 39 && fp.window == 2048 && fp.kvFixedBytes(chunkRows: 2048) == 39 * 528 * 4224)
        #expect(fp.calibration == .museGlimmer && !fp.hasDrafter && fp.snapshotBaseBytes == 0)
        #expect(fp.nativeContext == 131_072)
    }

    /// incoai's DFlash2 for Muse (config.json of revision 8336acb8, trimmed to what the check reads) fits the model:
    /// block 16 and the draft logits post-processed (× output_multiplier, softcap) are supported; the
    /// pack's `drafter/` is the one it runs with.
    @Test func museDrafter() throws {
        let dc: [String: Any] = [
            "hidden_size": 6656, "num_target_layers": 52, "head_dim": 128, "num_attention_heads": 32,
            "num_key_value_heads": 8, "sliding_window": 2048,
            "dflash_config": [
                "block_size": 16, "conv_group_size": 16, "conv_kernel_size": 2, "selector_rank": 256,
                "selector_top_k": 16, "target_layer_ids": [1, 13, 25, 37, 49], "mask_token_id": 201_818,
                "output_multiplier": 0.19611613513818404, "final_logit_softcapping": 20.0,
            ] as [String: Any],
        ]
        #expect(DFlash2Drafter.mismatch(config: dc, hidden: 6656, layers: 52, v1: false) == nil)
        let t = DrafterFitTests.Tree()
        let model = try t.dir("Muse-Glimmer-30B-4bit", try ToyReal.load("muse-glimmer-30b", root: Self.repo).json)
        _ = try t.dir("Muse-Glimmer-30B-4bit/drafter", dc)
        var logs: [String] = []
        let r = try DFlash2Drafter.resolve(
            modelDir: model, config: try Self.muse(), option: nil, toyPack: true, locate: { _ in model },
            log: { logs.append($0) })
        #expect(r?.lastPathComponent == "drafter" && logs.isEmpty)
    }

    /// The separate gate: q_proj | gate_proj | k_proj | v_proj concatenated and reordered by `gateRowOrder` gives
    /// the gated q row of the prework, `[q_h | gate_h]` per head and k, v behind. Checked on the CPU against the separate
    /// form: every row is the same bytes as its source row (4-bit; the gate at 8 bits widens q|k|v to 8, exact), and the
    /// projection read where the prework reads q (h·AT_QS + d) and the gate (h·AT_QS + AT_HD + d) equals q_proj(x) and
    /// gate_proj(x) of mlx-lm's form. A gate_proj of another width is a named error.
    @Test func museGateInterleave() throws {
        #expect(
            AttentionLayer.gateRowOrder(qHeads: 2, headDim: 2, kvRows: 2) == [0, 1, 4, 5, 2, 3, 6, 7, 8, 9])
        let real = AttentionLayer.gateRowOrder(qHeads: 32, headDim: 128, kvRows: 512)
        #expect(real.count == 8704 && Set(real).count == 8704 && real[128] == 4096 && real[256] == 128)
        #expect(real[8191] == 8191 && Array(real[8192...]) == Array(8192..<8704))
        let fmt = FormatTests()
        let dev = try #require(MTLCreateSystemDefaultDevice())
        let H = 3, hd = 128, kvH = 1, K = 128
        func data<T>(_ a: [T]) -> Data { a.withUnsafeBytes { Data($0) } }
        // One quantized matrix per projection, each with its own values (seed), at `bits`.
        func lin(_ name: String, rows: Int, bits: Int, seed: Int) -> [(String, String, [Int], Data)] {
            let q = (0..<(rows * K)).map { UInt32(($0 * (5 + 2 * seed) + seed) % (1 << bits)) }
            let s = (0..<(rows * 2)).map { floatToBF16(0.01 * Float(1 + ($0 + seed) % 13)) }
            let b = (0..<(rows * 2)).map { floatToBF16(Float(($0 * 3 + seed) % 7) - 3) }
            let p = "model.layers.0.self_attn." + name
            return [
                (p + ".weight", "U32", [rows, K * bits / 32], data(packQ(q, bits: bits))),
                (p + ".scales", "BF16", [rows, K / 64], data(s)), (p + ".biases", "BF16", [rows, K / 64], data(b)),
            ]
        }
        for gateBits in [4, 8] {
            let dir = try fmt.tmpDir()
            defer { try? FileManager.default.removeItem(at: dir) }
            try fmt.writeSafetensors(
                dir,
                tensors: lin("q_proj", rows: H * hd, bits: 4, seed: 1)
                    + lin("gate_proj", rows: H * hd, bits: gateBits, seed: 2)
                    + lin("k_proj", rows: kvH * hd, bits: 4, seed: 3) + lin("v_proj", rows: kvH * hd, bits: 4, seed: 4))
            var quant: [String: Any] = ["group_size": 64, "bits": 4]
            if gateBits == 8 { quant["model.layers.0.self_attn.gate_proj"] = ["group_size": 64, "bits": 8] }
            let cfg = try ModelConfig(json: ["hidden_size": K, "num_hidden_layers": 1, "quantization": quant])
            let st = try SafetensorsStore(directory: dir)
            let pre = "model.layers.0.self_attn."
            let proj = try AttentionLayer.qkvProjection(
                store: st, config: cfg, prefix: pre, gate: "gate_proj", keqv: false, qHeads: H, headDim: hd,
                device: dev)
            let bits = gateBits == 8 ? 8 : 4
            #expect(proj.N == 2 * H * hd + 2 * kvH * hd && proj.K == K && proj.bits == bits && proj.gs == 64)
            // Dequantized rows of the interleaved matrix (on the CPU, from its buffers) and of each source matrix.
            func rowOf(_ r: Int) -> [Float] {
                let wp = proj.w.contents().bindMemory(to: UInt32.self, capacity: proj.N * K * bits / 32)
                let sp = proj.s.contents().bindMemory(to: UInt16.self, capacity: proj.N * 2)
                let bp = proj.b.contents().bindMemory(to: UInt16.self, capacity: proj.N * 2)
                return (0..<K).map { j in
                    let q = unpackQ(wp + r * K * bits / 32, j, bits: bits)
                    return bf16ToFloat(sp[r * 2 + j / 64]) * Float(q) + bf16ToFloat(bp[r * 2 + j / 64])
                }
            }
            func src(_ name: String) throws -> [[Float]] {
                let l = try QuantizedLinear(store: st, path: pre + name, config: cfg)
                let f = st.dequantizeRows(l, rows: 0..<l.outFeatures)
                return (0..<l.outFeatures).map { Array(f[($0 * K)..<(($0 + 1) * K)]) }
            }
            let (q, g, k, v) = (try src("q_proj"), try src("gate_proj"), try src("k_proj"), try src("v_proj"))
            let qs = 2 * hd  // AT_QS with the gate
            for h in 0..<H {
                for d in 0..<hd {
                    #expect(rowOf(h * qs + d) == q[h * hd + d], "gate \(gateBits): q head \(h) row \(d)")
                    #expect(rowOf(h * qs + hd + d) == g[h * hd + d], "gate \(gateBits): gate head \(h) row \(d)")
                }
            }
            for r in 0..<(kvH * hd) {
                #expect(rowOf(H * qs + r) == k[r] && rowOf(H * qs + kvH * hd + r) == v[r], "k|v row \(r)")
            }
            // The projection of x in both forms, in Double: the engine's row read at the prework's offsets vs mlx-lm's
            // q_proj(x) and gate_proj(x), and the gated output o · σ(gate) of each head.
            var rng = SplitMix64(seed: 3)
            let x = (0..<K).map { _ in Double(rng.uniform()) * 2 - 1 }
            func dot(_ w: [Float]) -> Double { zip(w, x).reduce(0) { $0 + Double($1.0) * $1.1 } }
            let row = (0..<proj.N).map { dot(rowOf($0)) }
            let o = (0..<(H * hd)).map { _ in Double(rng.uniform()) - 0.5 }
            func sigmoid(_ z: Double) -> Double { 1 / (1 + exp(-z)) }
            for h in 0..<H {
                for d in 0..<hd {
                    let i = h * hd + d
                    #expect(row[h * qs + d] == dot(q[i]))
                    // at_gate: proj[t * AT_QP + h * AT_QS + AT_HD + d]; mlx-lm: out * sigmoid(gate_proj(x)) on h·hd + d.
                    #expect(o[i] * sigmoid(row[h * qs + hd + d]) == o[i] * sigmoid(dot(g[i])))
                }
            }
            // A gate_proj of another width (here, a q_proj-wide k head) is a load error with its name.
            do {
                _ = try AttentionLayer.qkvProjection(
                    store: st, config: cfg, prefix: pre, gate: "k_proj", keqv: false, qHeads: H, headDim: hd,
                    device: dev)
                Issue.record("a k_proj-wide gate loaded")
            } catch {
                #expect("\(error)".contains("the separate gate interleaves with q_proj"), "\(error)")
            }
        }
    }
}
