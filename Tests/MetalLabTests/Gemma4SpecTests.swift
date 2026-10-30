import Foundation
import LabBench
import Testing

@testable import MetalLab

/// Gemma 4 in the loader, from the five mlx-community configs in ref/toy/real/ (no weights, no GPU):
/// the family is recognized, the block reads as mlx-lm's gemma4_text, the full layers get their own shape (head 512,
/// K = V kv heads, proportional RoPE) and what the engine does not run yet is a load error that names every missing
/// piece, in a fixed order (the 12B and the 31B load; so do the E2B and the E4B, with PLE and shared KV, and the 26B-A4B,
/// with its MoE in parallel with the dense MLP).
extension ModelSpecTests {
    struct Gemma4Case {
        let name: String, textType: String
        let hidden: Int, layers: Int, full: Int, window: Int
        let qh: Int, kvh: Int
        /// bf16(√hidden), as mlx-lm's `h * embed_scale` on the bf16 embedding.
        let embedScale: Float
        /// Head and kv heads of the full layers (their own shape).
        let fullKVH: Int
        /// Fragments of the expected load errors, one per gap and in order (empty: it loads).
        let gaps: [String]
    }

    static let g4KeqV = "K = V in the full layers"
    static let g4PLE = "per-layer inputs (PLE"
    static let g4Shared = "KV shared across layers"
    static let g4MoE = "MoE in parallel with the dense MLP outside Gemma 4"

    static let gemma4Cases = [
        Gemma4Case(
            name: "gemma4-e2b", textType: "gemma4_text", hidden: 1536, layers: 35, full: 7, window: 512, qh: 8, kvh: 1,
            embedScale: 39.25, fullKVH: 1, gaps: []),
        Gemma4Case(
            name: "gemma4-e4b", textType: "gemma4_text", hidden: 2560, layers: 42, full: 7, window: 512, qh: 8, kvh: 2,
            embedScale: 50.5, fullKVH: 2, gaps: []),
        Gemma4Case(
            name: "gemma4-12b", textType: "gemma4_unified_text", hidden: 3840, layers: 48, full: 8, window: 1024,
            qh: 16, kvh: 8, embedScale: 62, fullKVH: 1, gaps: []),
        Gemma4Case(
            name: "gemma4-26b-a4b", textType: "gemma4_text", hidden: 2816, layers: 30, full: 5, window: 1024, qh: 16,
            kvh: 8, embedScale: 53, fullKVH: 2, gaps: []),
        Gemma4Case(
            name: "gemma4-31b", textType: "gemma4_text", hidden: 5376, layers: 60, full: 10, window: 1024, qh: 32,
            kvh: 16, embedScale: 73.5, fullKVH: 4, gaps: []),
    ]

    static func gemma4(_ name: String, _ edit: (inout [String: Any]) -> Void = { _ in }) throws -> ModelConfig {
        var j = try ToyReal.load(name, root: repo).json
        var t = try #require(j["text_config"] as? [String: Any])
        edit(&t)
        j["text_config"] = t
        // Through JSON, as the loader reads it.
        let data = try JSONSerialization.data(withJSONObject: j)
        return try ModelConfig(json: try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any]))
    }

    /// The gaps of a load error, split (nil if the config loads).
    static func gapList(_ c: ModelConfig) -> [String]? {
        guard let m = loadError(c), let r = m.range(of: ": not supported yet: ") else { return nil }
        return m[r.upperBound...].components(separatedBy: "; ")
    }

    @Test func gemma4Family() throws {
        for k in Self.gemma4Cases {
            let c = try Self.gemma4(k.name)
            #expect(c.family == .gemma4 && c.modelType == k.textType && (c.spec == nil) == !k.gaps.isEmpty, "\(k.name)")
            #expect(c.hiddenSize == k.hidden && c.numLayers == k.layers && c.vocabSize == 262_144, "\(k.name)")
            #expect(c.layerTypes.filter { $0 == "full_attention" }.count == k.full, "\(k.name): full layers")
            #expect(c.layerTypes.last == "full_attention", "\(k.name): the last layer is a full one")
            // The block: a sandwich with x·w norms, weightless v_norm, GeGLU, bf16(√hidden), scale 1,
            // layer_scalar and the final softcap; no gap (the weightless v_norm runs).
            let (b, gaps) = BlockSpec.read(
                text: c.text, family: c.family, eps: Float(c.rmsNormEps), headDim: 256, layers: c.numLayers,
                layerTypes: c.layerTypes)
            let n = NormSpec(.plain, eps: 1e-6)
            #expect(b.mixerNorm == n && b.ffnNorm == n && b.finalNorm == n, "\(k.name): norms")
            #expect(b.mixerPostNorm == n && b.ffnPostNorm == n && b.embedNorm == nil, "\(k.name): post-norms")
            #expect(b.qkNorm == .plain && b.vNorm == .unit && b.activation == .geluTanh, "\(k.name)")
            #expect(b.embedScale == k.embedScale && b.attnScale == 1 && b.layerScalar, "\(k.name): scales")
            #expect(b.softcap == 30 && b.residualScale == 1 && !b.qkvBias && !b.oBias, "\(k.name)")
            #expect(gaps.isEmpty, "\(k.name): \(gaps)")
            // Shape of the sliding layers: no gate, QK-norm, v_norm, whole-head RoPE from the table, window.
            let d = ModelDims(c)
            #expect(d.hidden == k.hidden && d.attnQH == k.qh && d.attnKVH == k.kvh && d.attnHD == 256, "\(k.name)")
            #expect(!d.attnGate && d.attnQKN && d.attnVN && d.attnRot == 256 && d.attnSWA && d.ffnGelu, "\(k.name)")
            #expect(d.attnRopeTab && !d.attnSinks && !d.attnQKVB, "\(k.name)")
            // The full layers' own shape: head 512 rotated whole, kv heads of K = V (or the model's without it).
            let f = try #require(ModelDims.fullAttention(c), "\(k.name): full shape")
            var same = f
            same.attnHD = d.attnHD; same.attnKVH = d.attnKVH; same.attnRot = d.attnRot
            #expect(f.attnHD == 512 && f.attnRot == 512 && f.attnKVH == k.fullKVH && same == d, "\(k.name)")
            if k.gaps.isEmpty {
                let sp = try #require(c.spec, "\(k.name)")
                // K = V (12B, 26B, 31B) is where the full layers have their own kv heads.
                #expect(sp.fullShape == f && sp.kEqV == (k.fullKVH != k.kvh), "\(k.name)")
                // Full layers: proportional RoPE over 512 (θ 1e6, 64 rotated pairs); sliding: default θ 1e4 over 256.
                let fi = try #require(c.layerTypes.firstIndex(of: "full_attention"))
                let si = try #require(c.layerTypes.firstIndex(of: "sliding_attention"))
                let fr = try #require(sp.ropes[fi]), sr = try #require(sp.ropes[si])
                #expect(fr.rotaryDims == 512 && fr.theta == 1e6 && fr.scaling?.kind == .proportional, "\(k.name)")
                #expect(sr.rotaryDims == 256 && sr.theta == 1e4 && sr.scaling == nil, "\(k.name)")
                let tab = fr.table
                #expect(tab.count == 257 && tab[63] > 0 && tab[64] == 0 && tab[255] == 0 && tab[256] == 1)
                continue
            }
            // Load error with every missing piece, by name and in order.
            let list = try #require(Self.gapList(c), "\(k.name) loads")
            #expect(list.count == k.gaps.count, "\(k.name): \(list)")
            for (got, want) in zip(list, k.gaps) { #expect(got.contains(want), "\(k.name): «\(got)» vs «\(want)»") }
            #expect(Self.loadError(c)?.hasPrefix("\(k.textType): not supported yet: ") == true, "\(k.name)")
        }
    }

    /// E2B and E4B: the last `num_kv_shared_layers` layers read the KV of the last earlier layer of their type
    /// (mlx-lm's `previous_kvs`), PLE of 256 per layer; the others have neither. The same keys outside Gemma 4, or a
    /// shared layer without a source of its type, are load errors.
    @Test func gemma4SharedKVAndPLE() throws {
        for (name, shared, sliding, full) in [("gemma4-e2b", 20, 13, 14), ("gemma4-e4b", 18, 22, 23)] {
            let c = try Self.gemma4(name)
            let sp = try #require(c.spec, "\(name)")
            let n = c.numLayers, first = n - shared
            #expect(sp.pleDim == 256, "\(name)")
            #expect(sp.kvSource.prefix(first).allSatisfy { $0 == nil }, "\(name)")
            for i in first..<n {
                #expect(sp.kvSource[i] == (c.layerTypes[i] == "full_attention" ? full : sliding), "\(name) layer \(i)")
            }
            #expect(c.layerTypes[sliding] == "sliding_attention" && c.layerTypes[full] == "full_attention")
        }
        for name in ["gemma4-12b", "gemma4-26b-a4b", "gemma4-31b"] {
            let c = try Self.gemma4(name) { $0["enable_moe_block"] = false }
            let sp = try #require(c.spec, "\(name)")
            #expect(sp.pleDim == 0 && sp.kvSource.allSatisfy { $0 == nil }, "\(name)")
        }
        // Every layer shared: no source for any of them.
        let all = try Self.gemma4("gemma4-e2b") { $0["num_kv_shared_layers"] = 35 }
        #expect(Self.gapList(all)?.contains { $0.contains("have no earlier layer of their type") } == true)
        let odd = try Self.gemma4("gemma4-e2b") { $0["hidden_size_per_layer_input"] = 96 }
        #expect(Self.gapList(odd)?.contains { $0.contains("PLE) of width 96") } == true)
    }

    /// The MoE keys of Gemma 4 (`top_k_experts`, `enable_moe_block`) and a MoE without a top-k as an error. The
    /// 26B-A4B loads: every layer runs the dense MLP and the MoE in parallel, with Gemma's router (own input norm and
    /// per_expert_scale, LAB_MOE_PES) and its weight names; elsewhere `enable_moe_block` is a load error.
    @Test func gemma4MoEKeys() throws {
        let moe = try Self.gemma4("gemma4-26b-a4b")
        #expect(moe.isMoE && moe.numExperts == 128 && moe.expertsPerToken == 8 && moe.moeIntermediateSize == 704)
        #expect(moe.intermediateSize == 2112 && moe.quant(for: "language_model.model.layers.3.router.proj")?.bits == 8)
        let d = ModelDims(moe)
        #expect(d.moeE == 128 && d.moeK == 8 && d.moeI == 704 && d.moeShared == 0 && !d.moeSigmoid && !d.moeBias)
        let heretic = try Self.heretic()
        #expect(d.moePES && d.macros["LAB_MOE_PES"] != nil && ModelDims(heretic).macros["LAB_MOE_PES"] == nil)
        let sp = try #require(moe.spec)
        #expect(sp.ffnKinds == Array(repeating: .denseAndMoE, count: 30))
        let r = try #require(sp.router)
        #expect(r.scoring == .softmax && r.shared == .none && r.ownNorm && r.perExpertScale && r.inter == 704)
        #expect(sp.names.moe == "" && sp.names.router == "router.proj" && sp.names.experts == "experts.switch_glu.")
        #expect(sp.names.ffn == "mlp." && sp.names.postNorm == "pre_feedforward_layernorm.weight")
        #expect(heretic.spec?.router.map { !$0.ownNorm && !$0.perExpertScale } == true)
        var other = heretic.raw
        var ot = try #require(other["text_config"] as? [String: Any])
        ot["enable_moe_block"] = true
        other["text_config"] = ot
        let otherCfg = try ModelConfig(json: other)
        #expect(Self.gapList(otherCfg)?.contains { $0.contains(Self.g4MoE) } == true)
        for name in ["gemma4-e2b", "gemma4-e4b", "gemma4-12b", "gemma4-31b"] {
            let c = try Self.gemma4(name)
            #expect(!c.isMoE && c.numExperts == nil && c.expertsPerToken == nil, "\(name)")
        }
        // enable_moe_block false is dense whatever num_experts says.
        let off = try Self.gemma4("gemma4-26b-a4b") { $0["enable_moe_block"] = false }
        #expect(!off.isMoE && Self.gapList(off) == nil, "26B-A4B without its MoE loads")
        // A MoE without a top-k is a load error (before, it loaded with router nil), in any family.
        let noTopK = try Self.gemma4("gemma4-26b-a4b") { $0["top_k_experts"] = NSNull() }
        #expect(Self.loadError(noTopK)?.contains("MoE without a top-k") == true)
        var h = try Self.heretic().raw
        var ht = try #require(h["text_config"] as? [String: Any])
        ht["num_experts_per_tok"] = nil
        h["text_config"] = ht
        #expect(Self.loadError(try ModelConfig(json: h))?.contains("MoE without a top-k") == true)
    }

    /// toy-gemma4 v3: the 26B-A4B config with ToyConfig.gemma4Deviations (full layer every 2, window 300,
    /// softcap 30.5, and the E2B pieces: PLE of 256, the last 2 layers sharing KV with a double-wide MLP) and the cuts;
    /// both attention shapes, K = V, the RoPE per layer and the MoE in parallel (router, 128 experts top-8) are the real
    /// ones, with the expert intermediate cut.
    @Test func toyGemma4MatchesReal() throws {
        let r = try ToyReal.load("gemma4-26b-a4b", root: Self.repo)
        var j = r.json
        var t = try #require(j["text_config"] as? [String: Any])
        ToyConfig.gemma4Deviate(&t, layers: 30)
        j["text_config"] = t
        let real = try Self.spec(j)
        let toy = try Self.spec(ToyConfig.gemma4(ToyDims.gemma4(r), quant: [:], seed: 1))
        Self.expectMatches(real, toy, "toy-gemma4")
        var full = try #require(toy.fullShape)
        full.hidden = real.shape.hidden; full.moeI = real.shape.moeI
        #expect(full == real.fullShape && full.attnHD == 512 && full.attnKVH == 2, "toy-gemma4: full shape")
        #expect(toy.kEqV && real.kEqV && toy.windows == [300, 0, 300, 0] && toy.block.softcap == 30.5)
        #expect(toy.kvSource == [nil, nil, 0, 1] && toy.pleDim == 256 && real.pleDim == 256, "toy-gemma4: E2B pieces")
        #expect(real.kvSource.suffix(2) == [26, 27], "26B-A4B deviated: its last two layers share")
        #expect(toy.ffnKinds.allSatisfy { $0 == .denseAndMoE } && toy.router?.inter == ToyCut.moeI, "toy-gemma4: MoE")
        var b = toy.block
        b.embedScale = real.block.embedScale
        #expect(b == real.block && toy.block.embedScale == 32, "toy-gemma4: block")
    }

    /// Detection by the text and the outer model_type, and the keys read by name in any family.
    @Test func gemma4DetectionAndKeys() throws {
        #expect(ModelFamily.detect(textType: "gemma4_text", rawType: nil) == .gemma4)
        #expect(ModelFamily.detect(textType: "gemma4_unified_text", rawType: nil) == .gemma4)
        #expect(ModelFamily.detect(textType: "?", rawType: "gemma4") == .gemma4)
        #expect(ModelFamily.detect(textType: "?", rawType: "gemma4_unified") == .gemma4)
        // Gemma 3 is not a family any more: a load error with its name.
        #expect(ModelFamily.detect(textType: "gemma3_text", rawType: nil) == nil)
        #expect(ModelFamily.gemma4.calibrationName == "gemma4")
        #expect(FamilyCalibration.forFamily(.gemma4) == .gemma4 && FamilyCalibration.gemma4.measured)
        // use_bidirectional_attention: "vision" (12B, 26B, 31B) loads; true or another string does not.
        let bidi = try Self.gemma4("gemma4-31b") { $0["use_bidirectional_attention"] = true }
        #expect(Self.gapList(bidi)?.contains("use_bidirectional_attention") == true)
        let all = try Self.gemma4("gemma4-31b") { $0["use_bidirectional_attention"] = "all" }
        #expect(Self.gapList(all)?.contains("use_bidirectional_attention «all»") == true)
        // The Gemma 4 keys are gaps in any family (here, on Granite 4.1 3B), not only in gemma4.
        let other = try Self.granite([
            "hidden_size_per_layer_input": 256, "num_kv_shared_layers": 4, "attention_k_eq_v": true,
        ])
        let e = try #require(Self.loadError(other))
        #expect(e.contains(Self.g4PLE) && e.contains(Self.g4Shared) && e.contains(Self.g4KeqV))
        #expect(Self.loadError(try Self.granite()) == nil)
    }
}
