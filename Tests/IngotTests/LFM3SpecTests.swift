import Foundation
import IngotBench
import Metal
import Testing

@testable import Ingot

/// LFM3 in the loader, from the config of LiquidAI/LFM3-1.7B-MLX-8bit (revision ee7667f3) in
/// ref/toy/real/ (no weights): the family is recognized, the attention reads as Liquid's modeling_lfm3.py (GQA 16/4 of
/// 128, QK-norm, the zero-tail partial RoPE of the full layers as the proportional table, θ 1e4 on the whole head in the
/// sliding ones, window 256, the scalar gate per head), the short convolution of 7 taps, the sandwich post-norms (also
/// around the convolution) and the `w1/w3/w2` FFN; what no LFM3 brings is a load error with its name.
extension ModelSpecTests {

    /// The real LFM3-1.7B config (ref/toy/real/lfm3-1.7b.json), edited, through JSON as the loader reads it.
    static func lfm3(_ edit: (inout [String: Any]) -> Void = { _ in }) throws -> ModelConfig {
        var j = try ToyReal.load("lfm3-1.7b", root: repo).json
        edit(&j)
        let data = try JSONSerialization.data(withJSONObject: j)
        return try ModelConfig(json: try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any]))
    }

    /// The 1.7B: 24 layers (13 conv of 7 taps, 6 full, 5 sliding), hidden 2048, dense SwiGLU of 7680, vocabulary
    /// 128,000 tied; it loads with no gap.
    @Test func lfm3Family() throws {
        let c = try Self.lfm3()
        #expect(c.family == .lfm3 && c.modelType == "lfm3")
        #expect(c.hiddenSize == 2048 && c.numLayers == 24 && c.vocabSize == 128_000 && c.rmsNormEps == 1e-5)
        #expect(c.intermediateSize == 7680 && !c.isMoE && c.quantDefault == nil)
        let count = { (ty: String) in c.layerTypes.filter { $0 == ty }.count }
        #expect(count("conv") == 13 && count("full_attention") == 6 && count("sliding_attention") == 5)
        let s = try #require(c.spec, "\(Self.loadError(c) ?? "")")
        #expect(Self.gapList(c) == nil)
        // The conv width reaches the kernels as LAB_SC_L (7); the snapshot holds its 6 window rows per conv layer.
        #expect(s.shape.convL == 7 && s.shape.macros["LAB_SC_L"] as? NSNumber == 7)
        let fp = ModelFootprint(
            name: "lfm3", config: c, weightBytes: 0, largestTensorBytes: 0, drafterConfig: nil, drafterWeightBytes: nil)
        #expect(fp.snapshotBaseBytes == 13 * 6 * 2048 * 4 && fp.kvBytesPerToken == 6 * 1056 + 8 && fp.windowLayers == 5)
    }

    /// The real config reads as modeling_lfm3.py and mlx-lm's lfm3.py (PR #2016).
    @Test func lfm3Spec() throws {
        let c = try Self.lfm3()
        let s = try #require(c.spec, "\(Self.loadError(c) ?? "")")
        #expect(s.family == .lfm3 && s.router == nil && s.ffnKinds == Array(repeating: .dense, count: 24))
        #expect(s.expertLayout == .unsupported && !s.hasDrafter && s.defaultDrafter == nil && s.int8MinPos == 1024)
        #expect(!s.int8Experts && s.calibration == .lfm3 && s.names == .lfm3 && s.mla == nil && s.pleDim == 0)
        #expect(s.windows == c.layerTypes.map { $0 == "sliding_attention" ? 256 : 0 } && s.hasSlidingWindow)
        #expect(s.windows.filter { $0 == 256 }.count == 5 && s.fullShape == nil && !s.kEqV)
        // RoPE: the full layers θ 1e6 with the zero tail (proportional, factor 1, partial 0.5) over the whole head; the
        // sliding ones θ 1e4 on the whole head without scaling; with a scaled layer, every attention layer reads a table.
        var zeroTail = RopeScaling(kind: .proportional, factor: 1)
        zeroTail.partial = 0.5
        for (i, ty) in c.layerTypes.enumerated() {
            switch ty {
            case "full_attention": #expect(s.ropes[i] == RopeSpec(theta: 1e6, rotaryDims: 128, scaling: zeroTail))
            case "sliding_attention": #expect(s.ropes[i] == RopeSpec(theta: 1e4, rotaryDims: 128))
            default: #expect(s.ropes[i] == nil, "layer \(i)")
            }
        }
        #expect(s.rope == RopeSpec(theta: 1e6, rotaryDims: 128, scaling: zeroTail) && s.hasRopeTable)
        // Attention: 16 q / 4 KV heads of 128 (G 4), the gate (repeated per head into the gated q row, AT_QS 256),
        // x·w QK-norm, window, no sinks or biases; the q|gate|k|v row of 5120 and an output of 2048.
        let d = s.shape
        #expect(d.attnQH == 16 && d.attnKVH == 4 && d.attnHD == 128 && d.attnRot == 128 && d.attnG == 4)
        #expect(d.attnGate && d.attnQKN && d.attnSWA && !d.attnSinks && d.attnRopeTab && !d.attnVN && !d.attnQKVB)
        #expect(d.attnProjN == 5120 && d.attnOutDim == 2048 && d.convL == 7)
        #expect(d.macros["LAB_AT_GATE"] as? NSNumber == 1 && d.macros["LAB_AT_ROPETAB"] as? NSNumber == 1)
        #expect(d.ffnActSuffix == "" && !d.quantG32)
        // Block: x·w norms with 1e-5 (the post-norms too), x·w QK-norm, SwiGLU, scale 128^-0.5, nothing else.
        let b = s.block, n = NormSpec(.plain, eps: 1e-5)
        #expect(b.mixerNorm == n && b.ffnNorm == n && b.finalNorm == n && b.embedNorm == nil)
        #expect(b.mixerPostNorm == n && b.ffnPostNorm == n && b.needsBranchEpilogue)
        #expect(
            b.qkNorm == .plain && b.vNorm == nil && b.activation == .silu && b.attnScale == 1 / Float(128).squareRoot())
        #expect(b.embedScale == 1 && b.residualScale == 1 && b.softcap == nil && !b.layerScalar && !b.oBias)
    }

    /// The sandwich post-norms (`use_sandwich_norm`, true in the 1.7B and by default): x·w with the model epsilon after
    /// the mixer and the FFN, around attention and short-convolution layers; without them, the plain block.
    @Test func lfm3Sandwich() throws {
        let t = try ToyReal.load("lfm3-1.7b", root: Self.repo).text
        let n = NormSpec(.plain, eps: 1e-5)
        for (edit, want) in [(true, true), (false, false), (nil, true)] as [(Bool?, Bool)] {
            var e = t
            e.removeValue(forKey: "use_sandwich_norm")
            if let edit { e["use_sandwich_norm"] = edit }
            let b = BlockSpec.read(text: e, family: .lfm3, eps: 1e-5, headDim: 128, layers: 24).spec
            #expect((b.mixerPostNorm, b.ffnPostNorm) == (want ? n : nil, want ? n : nil), "\(String(describing: edit))")
        }
        let off = try Self.lfm3 { $0["use_sandwich_norm"] = false }
        let s = try #require(off.spec, "\(Self.loadError(off) ?? "")")
        #expect(s.block.mixerPostNorm == nil && s.block.ffnPostNorm == nil && !s.block.needsBranchEpilogue)
        // The epilogue still does not go around the GDN (the prefill accumulates it straight into the stream).
        let gdn = try Self.lfm3 { $0["layer_types"] = (0..<24).map { $0 % 2 == 0 ? "conv" : "linear_attention" } }
        #expect(Self.loadError(gdn)?.contains("with linear_attention layers (only around attention and short") == true)
        // Another family with the key on is a gap; off, nothing.
        var g = try ToyReal.load("granite-4.1-3b", root: Self.repo).json
        g["use_sandwich_norm"] = true
        #expect(Self.gapList(try ModelConfig(json: g)) == ["use_sandwich_norm outside lfm3"])
        g["use_sandwich_norm"] = false
        #expect(try ModelConfig(json: g).spec != nil)
    }

    /// The tables the kernels read (LAB_AT_ROPETAB) against `compute_default_rope_parameters` of modeling_lfm3.py
    /// evaluated from the real config: `inv_freq_i = 1 / base^(2i / head_dim)` for i < int(partial · head_dim) / 2 and 0
    /// for the rest, with mscale 1. In float64 the table is the formula rounded once (≤ 1 ulp); against the float32
    /// evaluation of torch, a few ulp. The kernel's default formula (`pow(θ, -i / RH)`) is the same RoPE in the sliding
    /// layers.
    @Test func lfm3RopeTables() throws {
        let c = try Self.lfm3()
        let s = try #require(c.spec, "\(Self.loadError(c) ?? "")")
        let rp = try #require(c.text["rope_parameters"] as? [String: [String: Any]])
        let dim = try #require(c.text["head_dim"] as? Int)
        func ulps(_ a: Float, _ b: Float) -> Int { a == b ? 0 : Int(abs(Int64(a.bitPattern) - Int64(b.bitPattern))) }
        for (ty, layer) in [("full_attention", 2), ("sliding_attention", 4)] {
            let p = try #require(rp[ty])
            let base = try #require((p["rope_theta"] as? NSNumber)?.doubleValue)
            let partial = (p["partial_rotary_factor"] as? NSNumber)?.doubleValue ?? 1
            let angles = Int(partial * Double(dim)) / 2
            let ref64 = (0..<(dim / 2)).map { i in i < angles ? 1 / pow(base, Double(2 * i) / Double(dim)) : 0 }
            // torch: base ** (arange(0, 2·angles, 2, dtype=float) / dim) in float32, then 1 / that.
            let ref32 = (0..<(dim / 2)).map { i in
                i < angles ? 1 / pow(Float(base), Float(2 * i) / Float(dim)) : Float(0)
            }
            let table = try #require(s.ropes[layer]).table
            #expect(c.layerTypes[layer] == ty && table.count == dim / 2 + 1 && table.last == 1)
            let u64 = zip(table, ref64).map { ulps($0, Float($1)) }.max() ?? 0
            let u32 = zip(table, ref32).map { ulps($0, $1) }.max() ?? 0
            #expect(u64 <= 1 && u32 <= 8, "\(ty): \(u64) ulp vs float64, \(u32) vs float32")
            #expect(table[angles..<(dim / 2)].allSatisfy { $0 == 0 } && table[..<angles].allSatisfy { $0 > 0 })
            print("lfm3 rope \(ty): θ \(base), \(angles) pairs rotated, max \(u64) ulp vs float64, \(u32) vs float32")
        }
    }

    /// Detection, calibration name, the family's switches, the names and the gaps of configs no LFM3 brings.
    @Test func lfm3DetectionNamesAndGaps() throws {
        #expect(ModelFamily.detect(textType: "lfm3", rawType: "lfm3") == .lfm3)
        #expect(ModelFamily.detect(textType: "lfm2_moe", rawType: "lfm2_moe") == .lfm2)
        #expect(ModelFamily.detect(textType: "lfm2", rawType: nil) == .lfm2)
        #expect(ModelFamily.lfm3.calibrationName == "lfm3" && FamilyCalibration.forFamily(.lfm3) == .lfm3)
        #expect(FamilyCalibration.lfm3.measured)
        #expect(ModelSpec.attnGated(.lfm3) && !ModelSpec.moeBiases(.lfm3) && ModelSpec.defaultPartialRotary(.lfm3) == 1)
        let n = ModelSpec.weightNames(.lfm3)
        #expect(
            n.attn == "self_attn." && n.ffn == "feed_forward." && n.oProj == "o_proj" && n.embedding == "embed_tokens")
        #expect(n.qNorm == "q_norm.weight" && n.kNorm == "k_norm.weight" && n.attnGate == "gate_proj")
        #expect(n.inputNorm == "operator_norm.weight" && n.postNorm == "ffn_norm.weight")
        #expect(n.mixerPostNorm == "post_operator_norm.weight" && n.ffnPostNorm == "post_ffn_norm.weight")
        #expect(n.finalNorm == "embedding_norm.weight" && n.embedNorm == nil && n.mla == nil)
        #expect(n.denseGate == "w1" && n.denseUp == "w3" && n.denseDown == "w2")
        for f in ModelFamily.allCases where f != .lfm3 {
            let o = ModelSpec.weightNames(f)
            #expect(o.denseGate == "gate_proj" && o.denseUp == "up_proj" && o.denseDown == "down_proj", "\(f)")
        }
        // Gaps: a conv with bias, a width other than 3 and 7, a scaled RoPE with a partial factor.
        func gaps(_ edit: @escaping (inout [String: Any]) -> Void) throws -> [String] {
            Self.gapList(try Self.lfm3(edit)) ?? []
        }
        #expect(try gaps { $0["conv_bias"] = true }.contains("short convolution with bias (conv_bias)"))
        #expect(try gaps { $0["conv_L_cache"] = 5 }.first == "short convolution of width 5 (the kernels run 3 and 7)")
        let yarn = try gaps { j in
            var rp = j["rope_parameters"] as? [String: Any] ?? [:]
            rp["full_attention"] = [
                "rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 65536, "rope_theta": 1e6,
                "partial_rotary_factor": 0.5,
            ]
            j["rope_parameters"] = rp
        }
        #expect(yarn.contains("scaled RoPE «yarn» with partial_rotary_factor 0.5 in full_attention layers"), "\(yarn)")
        // LFM2.5 keeps its spec (the same rotated head of 64, no gate, no post-norms).
        let l = try ModelSpec(Self.lfm2())
        #expect(l.family == .lfm2 && l.names == .lfm2 && l.rope.rotaryDims == 64 && !l.shape.attnGate)
        #expect(l.shape.convL == 3 && l.shape.macros["LAB_SC_L"] == nil && l.block.mixerPostNorm == nil)
    }

    /// The scalar gate per head: q_proj | gate_proj (one row per head) | k_proj | v_proj concatenated and reordered by
    /// `headGateRowOrder` gives the gated q row of the prework, `[q_h | gate_h × head_dim]` per head and k, v behind.
    /// Checked on the CPU at 4 and 8 bits (the real one is 8 bits g64): every row is its source row, and σ of the row
    /// read where `at_gate` reads (h·AT_QS + AT_HD + d) is mlx-lm's σ(gate_proj(x))_h for every d.
    @Test func lfm3HeadGate() throws {
        #expect(AttentionLayer.headGateRowOrder(qHeads: 2, headDim: 2, kvRows: 2) == [0, 1, 4, 4, 2, 3, 5, 5, 6, 7])
        let real = AttentionLayer.headGateRowOrder(qHeads: 16, headDim: 128, kvRows: 1024)
        #expect(real.count == 5120 && Set(real).count == 2048 + 16 + 1024)
        #expect(real[127] == 127 && Array(real[128..<256]) == Array(repeating: 2048, count: 128) && real[256] == 128)
        #expect(Array(real[4096...]) == Array(2064..<3088))
        let fmt = FormatTests()
        let dev = try #require(MTLCreateSystemDefaultDevice())
        let H = 3, hd = 128, kvH = 1, K = 128
        func data<T>(_ a: [T]) -> Data { a.withUnsafeBytes { Data($0) } }
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
        for bits in [4, 8] {
            let dir = try fmt.tmpDir()
            defer { try? FileManager.default.removeItem(at: dir) }
            try fmt.writeSafetensors(
                dir,
                tensors: lin("q_proj", rows: H * hd, bits: bits, seed: 1)
                    + lin("gate_proj", rows: H, bits: bits, seed: 2)
                    + lin("k_proj", rows: kvH * hd, bits: bits, seed: 3)
                    + lin("v_proj", rows: kvH * hd, bits: bits, seed: 4))
            let cfg = try ModelConfig(json: [
                "hidden_size": K, "num_hidden_layers": 1, "quantization": ["group_size": 64, "bits": bits],
            ])
            let st = try SafetensorsStore(directory: dir)
            let pre = "model.layers.0.self_attn."
            let proj = try AttentionLayer.qkvProjection(
                store: st, config: cfg, prefix: pre, gate: "gate_proj", keqv: false, qHeads: H, headDim: hd,
                device: dev)
            #expect(proj.N == 2 * H * hd + 2 * kvH * hd && proj.K == K && proj.bits == bits && proj.gs == 64)
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
            let qs = 2 * hd
            for h in 0..<H {
                for d in 0..<hd {
                    #expect(rowOf(h * qs + d) == q[h * hd + d], "\(bits) bits: q head \(h) row \(d)")
                    #expect(rowOf(h * qs + hd + d) == g[h], "\(bits) bits: gate head \(h) row \(d)")
                }
            }
            for r in 0..<(kvH * hd) {
                #expect(rowOf(H * qs + r) == k[r] && rowOf(H * qs + kvH * hd + r) == v[r], "k|v row \(r)")
            }
            var rng = SplitMix64(seed: 5)
            let x = (0..<K).map { _ in Double(rng.uniform()) * 2 - 1 }
            func dot(_ w: [Float]) -> Double { zip(w, x).reduce(0) { $0 + Double($1.0) * $1.1 } }
            func sigmoid(_ z: Double) -> Double { 1 / (1 + exp(-z)) }
            for h in 0..<H {
                let want = sigmoid(dot(g[h]))
                #expect(
                    (0..<hd).allSatisfy { sigmoid(dot(rowOf(h * qs + hd + $0))) == want }, "\(bits) bits: head \(h)")
            }
            // A gate of neither width (here, k_proj's 128 rows for 3 heads of 128) is a named error.
            do {
                _ = try AttentionLayer.qkvProjection(
                    store: st, config: cfg, prefix: pre, gate: "k_proj", keqv: false, qHeads: H, headDim: hd,
                    device: dev)
                Issue.record("a k_proj-wide gate loaded")
            } catch {
                #expect("\(error)".contains("expected 3 heads × 128 or 3"), "\(error)")
            }
        }
    }
}

extension ModelSpecTests {
    /// toy-lfm3 is the real config with only the cuts: the real layers 1-4 (conv, full, conv, sliding:
    /// ToyCut.lfm3FirstLayer, the first period with all three kinds), vocabulary 32768 and the dense intermediate.
    /// The rest reads as the real one's: shape (hidden, heads, gate, conv of 7, the RoPE table), names, block with the
    /// sandwich, the RoPE and window of each kept layer.
    @Test func lfm3ToyMatchesReal() throws {
        let r = try ToyReal.load("lfm3-1.7b", root: Self.repo)
        let real = try Self.spec(r.json)
        let toy = try Self.spec(ToyConfig.lfm3(ToyDims.lfm3(r), quant: [:], seed: 1))
        let kept = ToyCut.lfm3FirstLayer..<(ToyCut.lfm3FirstLayer + ToyCut.layers)
        #expect(toy.family == .lfm3 && toy.names == real.names && toy.block == real.block && toy.router == nil)
        #expect(toy.ffnKinds == Array(real.ffnKinds[kept]) && toy.ropes == Array(real.ropes[kept]))
        #expect(toy.windows == Array(real.windows[kept]) && toy.windows == [0, 0, 0, 256])
        #expect(toy.rope == real.rope && toy.hasDrafter == real.hasDrafter && toy.expertLayout == real.expertLayout)
        #expect(toy.shape == real.shape && toy.shape.convL == 7 && toy.shape.hidden == 2048, "toy-lfm3: shape")
        #expect(toy.block.mixerPostNorm != nil && toy.block.ffnPostNorm != nil, "toy-lfm3: sandwich")
    }
}
