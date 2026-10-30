import Foundation
import Metal
import Testing

@testable import Ingot

// Kernel cases of the gated FFN activation instances. The CPU references are in fp32; the GELU-tanh
// formula is checked against the mlx-lm vectors of ref/act_ref.py.

extension KernelCase {
    static let activations: [KernelCase] = [ffnGelu, moeSharedCase(gelu: true), denseRelu2]

    /// gate | up rows with a spread that covers the saturated tails of both activations.
    static func gateUp(_ g: inout SplitMix64, rows: Int, n: Int) -> [Float] {
        Syn.normal(&g, rows * 2 * n, sd: 2.5).enumerated().map { i, v in i % 97 == 0 ? v * 6 : v }
    }

    static let ffnGelu = KernelCase(
        id: "ffn-gelu", model: "qwen3.6-gelu",
        catches:
            "GELU-tanh in the `_gelu` instances of the dense FFN (dswiglu of the round, dswiglu_hh of the prefill, "
            + "swiglu_quant_rot_i8 of --int8) and SiLU still in the plain ones (the drafter's dswiglu)"
    ) { ctx, g in
        let L = 8, N = 1024
        func act(_ gelu: Bool, _ gu: [Float]) -> [Float] {
            (0..<(L * N)).map { i in
                let t = i / N, c = i % N
                let a = gu[t * 2 * N + c], u = gu[t * 2 * N + N + c]
                return Float(Float16((gelu ? CPU.geluTanh(a) : a / (1 + exp(-a))) * u))
            }
        }
        func run(_ name: String, _ src: MTLBuffer, threads: Int) throws -> [Float] {
            let h = ctx.buffer(length: L * N * 2)
            var n = UInt32(N)
            try ctx.run { e in
                e.setComputePipelineState(try ctx.pipeline(name))
                e.setBuffer(src, offset: 0, index: 0)
                e.setBuffer(h, offset: 0, index: 1)
                e.setBytes(&n, length: 4, index: 2)
                e.dispatchThreads(
                    MTLSize(width: threads, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            }
            return readHalf(h, count: L * N)
        }
        let guF = gateUp(&g, rows: L, n: N), guH = Syn.roundHalf(gateUp(&g, rows: L, n: N))
        let fb = ctx.buffer(guF), hb = Syn.halfBuffer(ctx, guH)
        let roundGelu = try run("dswiglu_gelu", fb, threads: L * N)
        let prefillGelu = try run("dswiglu_hh_gelu", hb, threads: L * N / 4)
        var checks = [
            Check(label: "dswiglu_gelu (round)", gpu: roundGelu, ref: act(true, guF), tol: .halfUlp(1)),
            Check(label: "dswiglu_hh_gelu (prefill)", gpu: prefillGelu, ref: act(true, guH), tol: .halfUlp(1)),
            Check(
                label: "dswiglu (SiLU)", gpu: try run("dswiglu", fb, threads: L * N), ref: act(false, guF),
                tol: .halfUlp(1)),
            Check(
                label: "dswiglu_hh (SiLU)", gpu: try run("dswiglu_hh", hb, threads: L * N / 4),
                ref: act(false, guH), tol: .halfUlp(1)),
        ]
        // --int8: the fused activation + rotated int8 quantization gives the same bits as quantizing the output of
        // dswiglu_hh_gelu (same lab_glu, same rounding to half).
        let i8 = DenseI8Scratch(ctx: ctx, maxNK: 16, maxN: 4, maxRows: L, maxK: N)
        try ctx.run { e in try i8.quantize(ctx, e, x: hb, rows: L, K: N, swiglu: true) }
        let fusedQ = Syn.floats(i8.xq, L * N / 4), fusedS = Syn.floats(i8.sx, L)
        try ctx.run { e in try i8.quantize(ctx, e, x: Syn.halfBuffer(ctx, prefillGelu), rows: L, K: N) }
        checks += [
            Check(
                label: "swiglu_quant_rot_i8_gelu: int8", gpu: fusedQ, ref: Syn.floats(i8.xq, L * N / 4),
                tol: .bitExact),
            Check(label: "swiglu_quant_rot_i8_gelu: scales", gpu: fusedS, ref: Syn.floats(i8.sx, L), tol: .bitExact),
        ]
        return checks
    }
}

extension KernelCase {
    /// Nemotron-H's dense MLP without gate as DenseMLP runs it, with the Nano 4B's hidden 3136 = 49·64 (not a
    /// multiple of 128): up qmv → dswiglu_relu2 → down qmv added to the residual in the round, and the up GEMM (half)
    /// → dswiglu_hh_relu2 → accumulated down GEMM in the prefill, against the CPU.
    static let denseRelu2 = KernelCase(
        id: "dense-relu2", model: "nemotron-3-nano-4b",
        catches:
            "relu² without gate in the dense FFN (dswiglu_relu2 and dswiglu_hh_relu2 read up alone, [rows, I]) and "
            + "the qmv and prefill GEMM with K = N = 3136 (the Nano 4B's hidden)"
    ) { ctx, g in
        let D = 3136, I = 1024, B = 5, M = 64
        let up = Syn.q4(ctx, N: I, K: D, &g), down = Syn.q4(ctx, N: D, K: I, &g)
        func reference(_ x: [Float], rows: Int, halfU: Bool) -> [Float] {
            let u = CPU.matmulT(x, up.w, M: rows, N: I, K: D)
            let h = u.map { v -> Float in
                let r = max(halfU ? Float(Float16(v)) : v, 0)
                return Float(Float16(r * r))
            }
            return CPU.matmulT(h, down.w, M: rows, N: D, K: I)
        }
        var n = UInt32(I)
        func act(_ e: MTLComputeCommandEncoder, _ name: String, _ src: MTLBuffer, _ dst: MTLBuffer, threads: Int) throws
        {
            e.setComputePipelineState(try ctx.pipeline(name))
            e.setBuffer(src, offset: 0, index: 0)
            e.setBuffer(dst, offset: 0, index: 1)
            e.setBytes(&n, length: 4, index: 2)
            e.dispatchThreads(
                MTLSize(width: threads, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
        // Round: B rows.
        let xr = Syn.roundHalf(Syn.normal(&g, B * D, sd: 0.5))
        let xb = Syn.halfBuffer(ctx, xr), gu = ctx.buffer(length: B * I * 4), hb = ctx.buffer(length: B * I * 2)
        let out = ctx.buffer([Float](repeating: 0, count: B * D))
        let vq = VQMV(ctx), dvq = VQMV(ctx)
        try ctx.run { e in
            try vq.encode(e, x: xb, up.m, y: gu, rows: B)
            try act(e, "dswiglu_relu2", gu, hb, threads: B * I)
            try dvq.encode(e, x: hb, down.m, y: out, rows: B, accumulate: true)
        }
        // Prefill: M rows.
        let xp = Syn.roundHalf(Syn.normal(&g, M * D, sd: 0.5))
        let xpb = Syn.halfBuffer(ctx, xp), gp = ctx.buffer(length: M * I * 2), hp = ctx.buffer(length: M * I * 2)
        let outP = ctx.buffer([Float](repeating: 0, count: M * D))
        try ctx.run { e in
            try up.m.encodePrefillGemm(ctx, e, x: xpb, y: gp, rows: M, i8: nil, halfOut: true)
            try act(e, "dswiglu_hh_relu2", gp, hp, threads: M * I / 4)
            try down.m.encodePrefillGemm(ctx, e, x: hp, y: outP, rows: M, i8: nil, accumulate: true)
        }
        return [
            Check(
                label: "round (B = 5, K = N = 3136)", gpu: Syn.floats(out, B * D),
                ref: reference(xr, rows: B, halfU: false), tol: .maxRel(5e-3)),
            Check(
                label: "prefill (M = 64, K = N = 3136)", gpu: Syn.floats(outP, M * D),
                ref: reference(xp, rows: M, halfU: true), tol: .maxRel(5e-3)),
        ]
    }
}

/// The CPU relu² of the kernel cases against mlx-lm `nn.relu2` in fp32 (ref/act_ref.py). No GPU.
@Suite struct Relu2ReferenceTests {
    static let x: [Float] = [-12, -1, -0.0625, 0, 0.03125, 0.400000006, 1, 2.70000005, 15, 255]
    static let y: [Float] = [0, 0, 0, 0, 0.0009765625, 0.160000011, 1, 7.29000044, 225, 65025]

    @Test func cpuRelu2MatchesMLX() {
        for (x, want) in zip(Self.x, Self.y) {
            #expect(max(x, 0) * max(x, 0) == want, "relu2(\(x)), mlx \(want)")
        }
    }
}

/// The CPU GELU-tanh of the kernel cases against mlx-lm `nn.gelu_approx` in fp32 (ref/act_ref.py). No GPU.
@Suite struct GeluReferenceTests {
    static let x: [Float] = [
        -12, -6.5, -4, -3, -2.25, -1.5, -1, -0.75, -0.5, -0.25, -0.0625, 0, 0.03125, 0.125, 0.400000006, 0.699999988,
        1, 1.29999995, 2, 2.70000005, 3.5, 5, 8, 15,
    ]
    static let y: [Float] = [
        -0, -0, -7.00950623e-05, -0.00363743305, -0.0272013545, -0.100428388, -0.158807993, -0.170039445,
        -0.154285997, -0.100324646, -0.0296926498, 0, 0.0160145275, 0.0687171966, 0.262161195, 0.530570149,
        0.841192007, 1.17392898, 1.95459771, 2.69111228, 3.49938369, 5, 8, 15,
    ]

    @Test func cpuGeluMatchesMLX() {
        for (x, want) in zip(Self.x, Self.y) {
            let got = CPU.geluTanh(x)
            // mlx computes 0.5·x·(1 + tanh z) in fp32: up to ~1e-7 of cancellation error in the negative tail.
            #expect(abs(got - want) <= 2e-7 + 1e-6 * abs(want), "gelu(\(x)) = \(got), mlx \(want)")
        }
    }
}

// MARK: - Branch epilogue, embedding, logits and attention scale

extension KernelCase {
    static let blockPieceCases: [KernelCase] = [blockPieces, attnScale, attnQKVBias]

    static let attnQKVBias = KernelCase(
        id: "attn-qkv-bias", model: "granite-4.1-3b-qkvb",
        catches:
            "q|k|v biases (attention_bias / qkv_bias) added in the prework before the QK-norm and the RoPE "
            + "(LAB_AT_QKVB, buffer 12), in the round (vattn_prework_i8) and the prefill"
    ) { ctx, g in try attnChecks(ctx, &g, prefill: true, qkvBias: true) }

    static let blockPieces = KernelCase(
        id: "block-pieces", model: "qwen3.6-35b-a3b",
        catches:
            "branch_add_rows (post-norm with its eps, o_proj bias, residual multiplier, layer scalar; branch left at "
            + "zero), scale_rows and rmsnorm_rows_f of the embedding, logits_post (multiplier, divisor, softcap)"
    ) { ctx, g in
        let D = ctx.dims.hidden, R = 5, eps: Float = 1e-8
        let h0 = Syn.normal(&g, R * D), t0 = Syn.normal(&g, R * D, sd: 3)
        let w = Syn.roundBF16(Syn.normal(&g, D, sd: 0.2).map { 1 + $0 }), b = Syn.normal(&g, D, sd: 0.1)
        var checks: [Check] = []
        // branch_add_rows: with norm and bias (mul 0.22, scale 0.7), and without either (mul 1.5, scale 1).
        for (norm, bias, mul, scale) in [(true, true, Float(0.22), Float(0.7)), (false, false, 1.5, 1)] {
            let hb = ctx.buffer(h0), tb = ctx.buffer(t0), wb = Syn.bf16Buffer(ctx, w), bb = ctx.buffer(b)
            var p = (UInt32(D), eps, mul, scale, UInt32((norm ? 1 : 0) | (bias ? 2 : 0)))
            try ctx.run { e in
                e.setComputePipelineState(try ctx.pipeline("branch_add_rows"))
                for (i, x) in [hb, tb, wb, bb].enumerated() { e.setBuffer(x, offset: 0, index: i) }
                e.setBytes(&p, length: MemoryLayout.size(ofValue: p), index: 4)
                e.dispatchThreadgroups(
                    MTLSize(width: R, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            }
            var ref = [Float](repeating: 0, count: R * D)
            for r in 0..<R {
                let v = (0..<D).map { t0[r * D + $0] + (bias ? b[$0] : 0) }
                let rs = norm ? 1 / (v.reduce(0) { $0 + $1 * $1 } / Float(D) + eps).squareRoot() : 1
                for i in 0..<D { ref[r * D + i] = (h0[r * D + i] + mul * (norm ? v[i] * rs * w[i] : v[i])) * scale }
            }
            let tag = norm ? "norm + bias" : "plain"
            checks += [
                Check(label: "branch_add_rows (\(tag))", gpu: Syn.floats(hb, R * D), ref: ref, tol: .maxRel(2e-6)),
                Check(
                    label: "branch_add_rows (\(tag)): branch left at zero", gpu: Syn.floats(tb, R * D),
                    ref: [Float](repeating: 0, count: R * D), tol: .bitExact),
            ]
        }
        // Embedding: ×s, then RMSNorm in place (fp32 → fp32).
        let xb = ctx.buffer(h0), wb = Syn.bf16Buffer(ctx, w)
        var sp = (Float(45.25), UInt32(R * D))
        var np = (UInt32(D), Float(1e-6), UInt32(D), UInt32(D))
        try ctx.run { e in
            e.setComputePipelineState(try ctx.pipeline("scale_rows"))
            e.setBuffer(xb, offset: 0, index: 0)
            e.setBytes(&sp, length: 8, index: 1)
            e.dispatchThreads(
                MTLSize(width: R * D, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
        let scaled = Syn.floats(xb, R * D)
        checks.append(Check(label: "scale_rows", gpu: scaled, ref: h0.map { $0 * 45.25 }, tol: .bitExact))
        try ctx.run { e in
            e.setComputePipelineState(try ctx.pipeline("rmsnorm_rows_f"))
            e.setBuffer(xb, offset: 0, index: 0)
            e.setBuffer(wb, offset: 0, index: 1)
            e.setBytes(&np, length: 16, index: 2)
            e.dispatchThreadgroups(
                MTLSize(width: R, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
        var nref = [Float](repeating: 0, count: R * D)
        for r in 0..<R {
            let row = scaled[(r * D)..<((r + 1) * D)]
            let rs = 1 / (row.reduce(0) { $0 + $1 * $1 } / Float(D) + 1e-6).squareRoot()
            for i in 0..<D { nref[r * D + i] = scaled[r * D + i] * rs * w[i] }
        }
        checks.append(Check(label: "rmsnorm_rows_f", gpu: Syn.floats(xb, R * D), ref: nref, tol: .maxRel(2e-6)))
        // Logits: Muse order (×mul, then softcap 20) and Granite (÷8, no softcap), with logits up to ~±600 (saturated
        // tails of tanh).
        let n = 3 * 4096
        let lg = Syn.normal(&g, n, sd: 40).enumerated().map { i, v in i % 101 == 0 ? v * 15 : v }
        for (mul, div, cap) in [(Float(0.196_116_14), Float(1), Float(20)), (1, 8, 0)] {
            let lb = ctx.buffer(lg)
            var lp = (UInt32(n), mul, div, cap)
            try ctx.run { e in
                e.setComputePipelineState(try ctx.pipeline("logits_post"))
                e.setBuffer(lb, offset: 0, index: 0)
                e.setBytes(&lp, length: 16, index: 1)
                e.dispatchThreads(
                    MTLSize(width: n, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            }
            let ref = lg.map { x -> Float in
                let y = x * mul / div
                return cap > 0 ? Float(Double(cap) * tanh(Double(y) / Double(cap))) : y
            }
            checks.append(
                Check(
                    label: "logits_post (×\(mul) ÷\(div), softcap \(cap))", gpu: Syn.floats(lb, n), ref: ref,
                    tol: cap > 0 ? .maxRel(3e-7) : .bitExact))
        }
        return checks
    }

    static let attnScale = KernelCase(
        id: "attn-scale", model: "qwen3.6-35b-a3b",
        catches:
            "attention score scale from the block spec (Granite attention_multiplier, Gemma query_pre_attn_scalar) in "
            + "the round and the prefill instead of 1/√head_dim"
    ) { ctx, g in
        var checks: [Check] = []
        let f = try AttnFixture(ctx, positions: 700, &g, scale: 0.03)
        let x = Syn.roundHalf(Syn.normal(&g, 16 * f.hidden))
        checks.append(
            Check(
                label: "round, scale 0.03", gpu: try f.round(x, rows: 16, P: 700),
                ref: f.reference(x, rows: 16, P: 700), tol: .maxRel(5e-3)))
        let xp = Syn.roundHalf(Syn.normal(&g, 100 * f.hidden))
        checks.append(
            Check(
                label: "prefill, scale 0.03", gpu: try f.prefill(xp, rows: 100, P: 716),
                ref: f.reference(xp, rows: 100, P: 716), tol: .maxRel(5e-3)))
        return checks
    }
}

/// `1 + w` folded into a bf16 norm weight at load time rounds to nearest even, as mlx-lm's `1.0 + weight` in bf16
/// (halfway cases between two bf16 values next to 1). No GPU.
@Suite struct OnePlusNormTests {
    @Test func onePlusRoundsToNearestEven() {
        // Next to 1 the bf16 step is 2⁻⁷: 1 + 2⁻⁸ is halfway between 1 and 1 + 2⁻⁷ (even: 1), 1 + 3·2⁻⁸ between
        // 1 + 2⁻⁷ and 1 + 2⁻⁶ (even: 1 + 2⁻⁶), and −2⁻⁸ is exact (1 − 2⁻⁸ is a bf16 value).
        let cases: [(Float, Float)] = [
            (0.003_906_25, 1), (0.011_718_75, 1.015_625), (-0.003_906_25, 0.996_093_75), (0.5, 1.5), (-1, 0),
        ]
        for (w, want) in cases {
            #expect(bf16ToFloat(floatToBF16(1 + w)) == want, "1 + \(w)")
        }
    }
}
