import Foundation
import IngotBench
import Metal

@testable import Ingot

// MLX checkpoints with per-path bits in the attention (q|k|v at 4 and 8 bits in one concatenation, as mixed
// conversions carry): concatenating 4- and 8-bit pieces widens the 4-bit ones losslessly (DenseQ.widen4to8Words),
// 8-bit scales near the bottom of half's range (vnax_qmv8 reads scale and bias as half), the shared-expert guard
// (only 4-bit: it is fused with the experts' kernels) and the GDN guard (only 4-bit or ternary).
extension KernelCase {
    static let mixedBits: [KernelCase] = [
        mixedConcat, q8SmallScales, q8NF1, sharedBitsGuard, gdnBitsGuard, routerBF16,
    ]

    /// Synthetic MLX-quantized piece: random levels, bf16 scale and bias per group of 64, and its float value.
    struct SynPiece {
        let name: String, n: Int, bits: Int
        var words: [UInt32] = [], scales: [UInt16] = [], biases: [UInt16] = [], w: [Float] = [], q: [UInt32] = []

        init(_ name: String, n: Int, k: Int, bits: Int, scale: ClosedRange<Float>, _ g: inout SplitMix64) {
            self.name = name; self.n = n; self.bits = bits
            let levels = 1 << bits, groups = k / 64
            q = [UInt32](repeating: 0, count: n * k)
            for r in 0..<n {
                for gi in 0..<groups {
                    let sc = scale.lowerBound + (scale.upperBound - scale.lowerBound) * g.uniform()
                    let bi = -Float(levels / 2) * sc * (0.5 + g.uniform())
                    let (s16, b16) = (floatToBF16(sc), floatToBF16(bi))
                    scales.append(s16); biases.append(b16)
                    let se = bf16ToFloat(s16), be = bf16ToFloat(b16)
                    for j in 0..<64 {
                        let v = UInt32(g.next() % UInt64(levels))
                        q[r * k + gi * 64 + j] = v
                        w.append(se * Float(v) + be)
                    }
                }
            }
            words = packQ(q, bits: bits)
        }

        func tensors(k: Int) -> [(String, String, [Int], Data)] {
            func data<T>(_ a: [T]) -> Data { a.withUnsafeBytes { Data($0) } }
            return [
                (name + ".weight", "U32", [n, k * bits / 32], data(words)),
                (name + ".scales", "BF16", [n, k / 64], data(scales)),
                (name + ".biases", "BF16", [n, k / 64], data(biases)),
            ]
        }
    }

    /// A DenseQ dequantized on the CPU (4 or 8 bits, groups of 64 or 32).
    static func dequant(_ m: DenseQ) -> [Float] {
        let words = Syn.uints(m.w, m.N * m.K * m.bits / 32)
        let s = m.s.contents().bindMemory(to: UInt16.self, capacity: m.N * m.K / m.gs)
        let b = m.b.contents().bindMemory(to: UInt16.self, capacity: m.N * m.K / m.gs)
        let per = 32 / m.bits, mask = UInt32((1 << m.bits) - 1)
        return (0..<(m.N * m.K)).map { i in
            let q = (words[i / per] >> UInt32(m.bits * (i % per))) & mask, gi = i / m.gs
            return bf16ToFloat(s[gi]) * Float(q) + bf16ToFloat(b[gi])
        }
    }

    static func tmpDir() throws -> URL {
        let d = FileManager.default.temporaryDirectory.appendingPathComponent("ingot-mb6-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
        return d
    }

    /// q|k|v-like concatenation with 8-bit and 4-bit pieces and padding to 64 rows, loaded
    /// through DenseQ(store:) from a safetensors file: the result is 8-bit, dequantizes bit for bit to the pieces and
    /// the round qmv (vnax_qmv8) and the prefill GEMM (qmm_t_nax3_8) match the CPU.
    static let mixedConcat = KernelCase(
        id: "mixed-concat", model: "qwen3.6-35b-a3b",
        catches: "4-bit pieces concatenated with 8-bit ones widened losslessly (q·16, scale / 16) "
            + "in DenseQ(store:), with row offsets and padding, read by vnax_qmv8 and qmm_t_nax3_8"
    ) { ctx, g in
        let K = 2048
        let pieces = [
            SynPiece("l.p0", n: 96, k: K, bits: 8, scale: 0.0005...0.002, &g),
            SynPiece("l.p1", n: 64, k: K, bits: 4, scale: 0.005...0.02, &g),
            SynPiece("l.p2", n: 32, k: K, bits: 4, scale: 0.005...0.02, &g),
            SynPiece("l.p3", n: 16, k: K, bits: 8, scale: 0.0005...0.002, &g),
        ]
        let dir = try tmpDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        try FormatTests().writeSafetensors(dir, tensors: pieces.flatMap { $0.tensors(k: K) })
        var quant: [String: Any] = ["group_size": 64, "bits": 4]
        for p in pieces where p.bits == 8 { quant[p.name] = ["group_size": 64, "bits": 8] }
        let cfg = try ModelConfig(json: ["hidden_size": K, "num_hidden_layers": 1, "quantization": quant])
        let st = try SafetensorsStore(directory: dir)
        let m = try DenseQ(store: st, config: cfg, paths: pieces.map(\.name), device: ctx.device, padTo: 64)
        let n0 = pieces.reduce(0) { $0 + $1.n }
        let ref = pieces.flatMap(\.w) + [Float](repeating: 0, count: (m.N - n0) * K)
        var checks = [
            Check(
                label: "bits, N, group", gpu: [m.bits, m.N, m.gs].map(Float.init), ref: [8, 256, 64], tol: .bitExact),
            Check(
                label: "dequantized weights (8-bit, 4-bit widened, padding)", gpu: dequant(m), ref: ref,
                tol: .bitExact),
        ]
        for rows in [16, 5, 1] {
            let x = Syn.roundHalf(Syn.normal(&g, rows * K))
            checks.append(
                Check(
                    label: "vnax_qmv8 mixed \(m.N)×\(K), \(rows) rows", gpu: try qmv(ctx, m, x, rows: rows),
                    ref: CPU.matmulT(x, ref, M: rows, N: m.N, K: K), tol: .maxRel(1e-3)))
        }
        let M = 64
        let x = Syn.roundHalf(Syn.normal(&g, M * K))
        let xb = Syn.halfBuffer(ctx, x), yf = ctx.buffer(length: M * m.N * 4), yh = ctx.buffer(length: M * m.N * 2)
        try ctx.run { e in
            try m.encodeQmmF32(ctx, e, x: xb, y: yf, rows: M)
            try m.encodeQmmF32(ctx, e, x: xb, y: yh, rows: M, halfOut: true)
        }
        let refG = CPU.matmulT(x, ref, M: M, N: m.N, K: K)
        checks.append(Check(label: "qmm_t_nax3_8 mixed", gpu: Syn.floats(yf, M * m.N), ref: refG, tol: .maxRel(1e-3)))
        checks.append(
            Check(label: "qmm_t_nax3_8 mixed half", gpu: readHalf(yh, count: M * m.N), ref: refG, tol: .maxRel(2e-3)))
        return checks
    }

    /// The bf16 MoE router (Nemotron-H, GLM) loaded by MoELayer.loadRouter as bf16, rows padded with zeros up to
    /// 64, and applied by the round qmv (vnax_qmvbf16, 16, 5 and 1 rows) and the prefill GEMM (qmm_t_nax_f32_bf16)
    /// against the CPU in fp32 with the same bf16 weights. Weights of the real routers' scale (~0.02): the bf16 → half
    /// conversion inside the kernels is exact there.
    static let routerBF16 = KernelCase(
        id: "router-bf16", model: "nemotron-3.5-lightning",
        catches: "bf16 router quantized to 8 bits at load again, padding rows not zero, or a wrong permutation or "
            + "reduction in vnax_qmvbf16 / the bf16 branch of nax_load_w"
    ) { ctx, g in
        let E = 96, K = 2048
        let w16 = Syn.normal(&g, E * K, sd: 0.02).map(floatToBF16)
        let dir = try tmpDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        try FormatTests().writeSafetensors(
            dir, tensors: [("l.gate.weight", "BF16", [E, K], w16.withUnsafeBytes { Data($0) })])
        let cfg = try ModelConfig(json: ["hidden_size": K, "num_hidden_layers": 1])
        let m = try MoELayer.loadRouter(
            store: try SafetensorsStore(directory: dir), config: cfg, path: "l.gate", device: ctx.device)
        let ref = w16.map(bf16ToFloat) + [Float](repeating: 0, count: (128 - E) * K)
        var checks = [Check(label: "bits, N", gpu: [m.bits, m.N].map(Float.init), ref: [16, 128], tol: .bitExact)]
        for rows in [16, 5, 1] {
            let x = Syn.roundHalf(Syn.normal(&g, rows * K))
            checks.append(
                Check(
                    label: "vnax_qmvbf16 \(m.N)×\(K), \(rows) rows", gpu: try qmv(ctx, m, x, rows: rows),
                    ref: CPU.matmulT(x, ref, M: rows, N: m.N, K: K), tol: .maxRel(1e-4)))
        }
        checks.append(try rowInvariance("vnax_qmvbf16", K: K, &g) { x, rows in try qmv(ctx, m, x, rows: rows) })
        let M = 128
        let x = Syn.roundHalf(Syn.normal(&g, M * K))
        let xb = Syn.halfBuffer(ctx, x), y = ctx.buffer(length: M * m.N * 4)
        var qp = QmmParams(M: M, N: m.N, K: K)
        try ctx.run { e in
            e.setComputePipelineState(try ctx.pipeline("qmm_t_nax_f32_bf16"))
            for (i, b) in [xb, m.w, m.s, m.b, y].enumerated() { e.setBuffer(b, offset: 0, index: i) }
            e.setBytes(&qp, length: MemoryLayout<QmmParams>.size, index: 5)
            e.dispatchThreadgroups(
                MTLSize(width: m.N / 64, height: M / 64, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        }
        checks.append(
            Check(
                label: "qmm_t_nax_f32_bf16 \(M)×\(m.N)×\(K)", gpu: Syn.floats(y, M * m.N),
                ref: CPU.matmulT(x, ref, M: M, N: m.N, K: K), tol: .maxRel(1e-4)))
        return checks
    }

    /// 8-bit scales are ~16 times smaller than 4-bit ones (range / 255; widened: scale₄ / 16) and vnax_qmv8 converts
    /// scale and bias to half: rows whose scales fall below half's smallest normal (6.1e-5) lose precision there.
    /// Per scale class against the exact CPU (bf16 scales, as mlx-lm reads them) and against the CPU with the scale
    /// rounded to half (what the kernel computes).
    static let q8SmallScales = KernelCase(
        id: "q8-small-scales", model: "qwen3.6-35b-a3b",
        catches: "8-bit scales near or below half's smallest normal in vnax_qmv8 (scale and bias read as half)"
    ) { ctx, g in
        let K = 2048, rows = 8, perClass = 32
        // Normal half (≥ 6.1e-5) must be exact to 1e-3; below it, the half subnormal step (6e-8) shows.
        let classes: [(Float, Float)] = [(1e-3, 1e-3), (1e-4, 1e-3), (6.2e-5, 1e-3), (2e-5, 5e-3), (2e-6, 0.05)]
        var checks: [Check] = []
        for (sc, tol) in classes {
            let p = SynPiece("p", n: perClass, k: K, bits: 8, scale: sc...(sc * 1.5), &g)
            func buf<T>(_ a: [T]) -> MTLBuffer { ctx.buffer(a) }
            let m = DenseQ(w: buf(p.words), s: buf(p.scales), b: buf(p.biases), N: perClass, K: K, bits: 8)
            let x = Syn.roundHalf(Syn.normal(&g, rows * K))
            let gpu = try qmv(ctx, m, x, rows: rows)
            // The same weights with the scale and bias rounded to half.
            let wh = (0..<(perClass * K)).map { i -> Float in
                let gi = i / 64
                let s = Float(Float16(bf16ToFloat(p.scales[gi]))), b = Float(Float16(bf16ToFloat(p.biases[gi])))
                return s * Float(p.q[i]) + b
            }
            checks.append(
                Check(
                    label: "vnax_qmv8 scales \(sc) vs exact", gpu: gpu,
                    ref: CPU.matmulT(x, p.w, M: rows, N: perClass, K: K),
                    tol: .maxRel(tol)))
            checks.append(
                Check(
                    label: "vnax_qmv8 scales \(sc) vs half scales", gpu: gpu,
                    ref: CPU.matmulT(x, wh, M: rows, N: perClass, K: K), tol: .maxRel(1e-3)))
        }
        return checks
    }

    /// The round's 8-bit qmv (VQMV.encode: fn_qmv8_nf1_sg16, 16 columns per threadgroup) gives bit for bit
    /// what vnax_qmv8_nf2_sg16 gave (32 columns, the same split-K and summation order), with and without
    /// accumulation, in LFM3's down shape and the MoE router's.
    static let q8NF1 = KernelCase(
        id: "qmv8-nf1", model: "lfm3-1.7b",
        catches: "fn_qmv8_nf1_sg16 in VQMV's 8-bit branch not bit-identical to vnax_qmv8_nf2_sg16 (another split-K or "
            + "summation order, a wrong column offset or threadgroup count)"
    ) { ctx, g in
        var checks: [Check] = []
        for (N, K) in [(2048, 7680), (256, 2048)] {
            let p = SynPiece("p", n: N, k: K, bits: 8, scale: 0.0005...0.002, &g)
            let m = DenseQ(
                w: ctx.buffer(p.words), s: ctx.buffer(p.scales), b: ctx.buffer(p.biases), N: N, K: K, bits: 8)
            for rows in [1, 4, 16] {
                let xb = Syn.halfBuffer(ctx, Syn.roundHalf(Syn.normal(&g, rows * K)))
                let y0 = Syn.normal(&g, rows * N)
                let (yNew, yOld) = (ctx.buffer(y0), ctx.buffer(y0))
                try ctx.run { e in try VQMV(ctx).encode(e, x: xb, m, y: yNew, rows: rows, accumulate: rows > 1) }
                var vp = VParams(
                    M: UInt32(rows), N: UInt32(N), K: UInt32(K), flags: rows > 1 ? 1 : 0, ldx: UInt32(K), ldy: UInt32(N)
                )
                try ctx.run { e in
                    e.setComputePipelineState(try ctx.pipeline("vnax_qmv8_nf2_sg16"))
                    for (i, b) in [xb, m.w, m.s, m.b, yOld].enumerated() { e.setBuffer(b, offset: 0, index: i) }
                    e.setBytes(&vp, length: MemoryLayout<VParams>.size, index: 5)
                    e.dispatchThreadgroups(
                        MTLSize(width: N / 32, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 32 * 16, height: 1, depth: 1))
                }
                checks.append(
                    Check(
                        label: "qmv8 nf1 vs nf2 \(N)×\(K), \(rows) rows\(rows > 1 ? ", accumulate" : "")",
                        gpu: Syn.floats(yNew, rows * N), ref: Syn.floats(yOld, rows * N), tol: .bitExact))
            }
        }
        return checks
    }

    /// A shared expert that is not 4-bit fails at load (MoELayer and PrefillMoE) instead of being read as 4-bit by the
    /// fused kernels; the same store with a 4-bit shared expert loads (so the error is the guard's). Likewise experts
    /// in groups of 32 and a 4-bit router (only g64 expert kernels and 8-bit or bf16 router kernels exist).
    static let sharedBitsGuard = KernelCase(
        id: "shared-bits-guard", model: "qwen3.6-35b-a3b",
        catches: "shared expert at 8 bits g64 accepted by the group-only guard and read as 4-bit (MoE.swift); "
            + "experts g32 or a 4-bit router loading instead of failing with a named error"
    ) { ctx, _ in
        let real = try ToyReal.load("qwen3.6-35b-a3b", root: ModelSpecTests.repo)
        let d = ToyDims.qwen(real)
        let E = 64, H = d.hidden, I = d.moeI, sI = d.sharedI
        func loadMoE(sharedBits: Int = 4, routerBits: Int = 8, expertGroup: Int = 64) throws -> MoELayer {
            let dir = try tmpDir()
            defer { try? FileManager.default.removeItem(at: dir) }
            let p = "language_model.model.layers.0.mlp."
            var t: [(String, String, [Int], Data)] = []
            var quant: [String: Any] = ["group_size": 64, "bits": 4]
            func q(_ n: String, _ rows: Int, _ cols: Int, bits: Int, experts: Int? = nil, gs: Int = 64) {
                let lead = experts.map { [$0] } ?? [], count = (experts ?? 1) * rows
                let wshape = lead + [rows, cols * bits / 32]
                t.append((p + n + ".weight", "U32", wshape, Data(count: count * cols * bits / 8)))
                for s in ["scales", "biases"] {
                    t.append((p + n + "." + s, "BF16", lead + [rows, cols / gs], Data(count: count * cols / gs * 2)))
                }
                if bits != 4 || gs != 64 { quant[p + n] = ["group_size": gs, "bits": bits] }
            }
            q("gate", E, H, bits: routerBits)
            q("shared_expert_gate", 1, H, bits: 8)
            q("shared_expert.gate_proj", sI, H, bits: sharedBits)
            q("shared_expert.up_proj", sI, H, bits: sharedBits)
            q("shared_expert.down_proj", H, sI, bits: sharedBits)
            for n in ["gate_proj", "up_proj"] { q("switch_mlp." + n, I, H, bits: 4, experts: E, gs: expertGroup) }
            q("switch_mlp.down_proj", H, I, bits: 4, experts: E, gs: expertGroup)
            try FormatTests().writeSafetensors(dir, tensors: t)
            var json = ToyConfig.qwen(d, quant: quant, seed: 1)
            var text = json["text_config"] as? [String: Any] ?? [:]
            text["num_experts"] = E
            json["text_config"] = text
            let cfg = try ModelConfig(json: json)
            return try MoELayer(ctx: ctx, store: SafetensorsStore(directory: dir), config: cfg, layer: 0)
        }
        let ok = try loadMoE()
        func loadError(_ r: Result<MoELayer, Error>) -> String {
            if case .failure(let e) = r { return "\(e)" }
            return ""
        }
        let message = loadError(Result { try loadMoE(sharedBits: 8) })
        let g32Message = loadError(Result { try loadMoE(expertGroup: 32) })
        let router4Message = loadError(Result { try loadMoE(routerBits: 4) })
        // PrefillMoE with the 4-bit layer's weights but an 8-bit (widened) shared expert.
        let cfg4 = try ModelConfig(
            json: {
                var j = ToyConfig.qwen(d, quant: ["group_size": 64, "bits": 4], seed: 1)
                var text = j["text_config"] as? [String: Any] ?? [:]
                text["num_experts"] = E
                j["text_config"] = text
                return j
            }())
        let wide = [ok.sGate, ok.sUp, ok.sDown].map { DenseQ.widen4to8($0, device: ctx.device).asDQL() }
        let w = [ok.router.asDQL(), ok.sharedGate.asDQL()] + wide + [ok.eGate, ok.eUp, ok.eDown]
        var prefillMessage = ""
        do { _ = try PrefillMoE(ctx: ctx, weights: w, config: cfg4, rows: 64) } catch { prefillMessage = "\(error)" }
        let w4 =
            [ok.router, ok.sharedGate, ok.sGate, ok.sUp, ok.sDown].map { $0.asDQL() } + [ok.eGate, ok.eUp, ok.eDown]
        var prefill4 = true
        do { _ = try PrefillMoE(ctx: ctx, weights: w4, config: cfg4, rows: 64) } catch { prefill4 = false }
        print(
            "kcase shared-bits-guard | MoELayer: \(message) | PrefillMoE: \(prefillMessage) | g32: \(g32Message) "
                + "| router 4-bit: \(router4Message)")
        func has(_ s: String) -> Float { s.contains("shared expert at 8/8/8 bits") ? 1 : 0 }
        return [
            Check(label: "MoELayer rejects an 8-bit shared expert", gpu: [has(message)], ref: [1], tol: .bitExact),
            Check(
                label: "PrefillMoE rejects an 8-bit shared expert", gpu: [has(prefillMessage)], ref: [1],
                tol: .bitExact),
            Check(label: "PrefillMoE loads the 4-bit one", gpu: [prefill4 ? 1 : 0], ref: [1], tol: .bitExact),
            Check(
                label: "MoELayer rejects experts in groups of 32",
                gpu: [g32Message.contains("MoE in groups of 32/32/32/64/64/64") ? 1 : 0], ref: [1], tol: .bitExact),
            Check(
                label: "MoELayer rejects a 4-bit router", gpu: [router4Message.contains("router at 4 bits") ? 1 : 0],
                ref: [1], tol: .bitExact),
        ]
    }
}

extension KernelCase {
    /// GDN matrices at 8 bits (the OptiQ mixed conversions) fail at load with a named error (no tier-1
    /// model carries them and no toy runs them since toy-qwen36-mix8 was removed); 4-bit ones load.
    static let gdnBitsGuard = KernelCase(
        id: "gdn-bits-guard", model: "qwen3.5-0.8b",
        catches: "an 8-bit GDN projection (qkv|z|b|a or out_proj) loading instead of failing with a named error"
    ) { ctx, _ in
        let d = ctx.dims, H = d.hidden
        func loadGDN(projBits: Int = 4, outBits: Int = 4) throws -> GDNLayer {
            let dir = try tmpDir()
            defer { try? FileManager.default.removeItem(at: dir) }
            let p = "language_model.model.layers.0.linear_attn."
            var json = try ToyReal.load("qwen3.5-0.8b", root: ModelSpecTests.repo).json
            var quant = json["quantization"] as? [String: Any] ?? [:]
            let gs = quant["group_size"] as? Int ?? 64
            var t: [(String, String, [Int], Data)] = []
            func q(_ n: String, _ rows: Int, _ cols: Int, bits: Int) {
                t.append((p + n + ".weight", "U32", [rows, cols * bits / 32], Data(count: rows * cols * bits / 8)))
                for s in ["scales", "biases"] {
                    t.append((p + n + "." + s, "BF16", [rows, cols / gs], Data(count: rows * cols / gs * 2)))
                }
                if bits != 4 { quant[p + n] = ["group_size": gs, "bits": bits] }
            }
            q("in_proj_qkv", d.gdnConvDim, H, bits: projBits)
            q("in_proj_z", d.gdnVDim, H, bits: projBits)
            q("in_proj_b", d.gdnHV, H, bits: projBits)
            q("in_proj_a", d.gdnHV, H, bits: projBits)
            q("out_proj", H, d.gdnVDim, bits: outBits)
            for (n, shape) in [
                ("conv1d.weight", [d.gdnConvDim, 4, 1]), ("dt_bias", [d.gdnHV]), ("norm.weight", [d.gdnDV]),
                ("A_log", [d.gdnHV]),
            ] {
                t.append((p + n, "BF16", shape, Data(count: shape.reduce(1, *) * 2)))
            }
            try FormatTests().writeSafetensors(dir, tensors: t)
            json["quantization"] = quant; json["quantization_config"] = quant
            let cfg = try ModelConfig(json: json)
            return try GDNLayer(ctx: ctx, store: SafetensorsStore(directory: dir), config: cfg, layer: 0)
        }
        func loadError(_ r: Result<GDNLayer, Error>) -> String {
            if case .failure(let e) = r { return "\(e)" }
            return ""
        }
        let ok4 = loadError(Result { try loadGDN() })
        let proj8 = loadError(Result { try loadGDN(projBits: 8) })
        let out8 = loadError(Result { try loadGDN(outBits: 8) })
        print(
            "kcase gdn-bits-guard | 4-bit: \(ok4.isEmpty ? "loads" : ok4) | qkv|z|b|a 8-bit: \(proj8) | out 8-bit: \(out8)"
        )
        func has(_ s: String) -> Float { s.contains("4 bits per matrix, or ternary") ? 1 : 0 }
        return [
            Check(label: "GDNLayer loads 4-bit matrices", gpu: [ok4.isEmpty ? 1 : 0], ref: [1], tol: .bitExact),
            Check(label: "GDNLayer rejects an 8-bit qkv|z|b|a", gpu: [has(proj8)], ref: [1], tol: .bitExact),
            Check(label: "GDNLayer rejects an 8-bit out_proj", gpu: [has(out8)], ref: [1], tol: .bitExact),
        ]
    }
}
