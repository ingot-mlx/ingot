import Foundation
import Metal

@testable import MetalLab

// KernelCaseTests cases for Flash-Next (Shaders/zf_flashnext.metal and the GDN sigmoid gate), against the
// CPU with synthetic tensors: what the end-to-end harness (lab-dev fncheck) does not isolate.

extension KernelCase {
    static let flashNext: [KernelCase] = [fnHC, fnPLE, fnMoE, fnGDNGate, fnQSA, fnQSAM, fnBlockList]

    /// Synthetic Flash-Next format (real PLE multipliers; sample sizes and offsets).
    static func fnFormat() throws -> FlashNextFormat {
        let ple: [String: Any] = [
            "rows": 320_001_536, "row_bytes": 100, "rows_per_page": 163, "page_bytes": 16384, "layer": 1,
            "multipliers": [23_703_573_157_769, 20_109_073_645_365, 8_052_911_324_071],
            "head_offsets": [
                0, 20_000_003, 40_000_026, 60_000_059, 80_000_106, 100_000_165, 120_000_228, 140_000_297, 160_000_374,
                180_000_455, 200_000_548, 220_000_655, 240_000_774, 260_000_903, 280_001_052, 300_001_203,
            ],
            "head_vocab_sizes": [
                20_000_003, 20_000_023, 20_000_033, 20_000_047, 20_000_059, 20_000_063, 20_000_069, 20_000_077,
                20_000_081, 20_000_093, 20_000_107, 20_000_119, 20_000_129, 20_000_149, 20_000_151, 20_000_161,
            ],
            "eos_token_id": 248044,
        ]
        let ex: [String: Any] = [
            "record_bytes": 1_392_640, "experts": 512, "layers": 48,
            "offsets": [
                "gate": ["codes": 0, "scales": 409_600], "up": ["codes": 460_800, "scales": 870_400],
                "down": ["codes": 921_600, "scales": 1_331_200],
            ],
        ]
        let cfg = try ModelConfig(json: [
            "text_config": ["model_type": "qwen4_exp_text", "hidden_size": 2560, "num_hidden_layers": 48],
            "metal_lab": ["experts": ex, "ple": ple],
        ])
        return try FlashNextFormat(cfg)
    }

    /// Arena of `slots` Flash-Next Q2 expert records with expert i in slot slotOf[i]: 2-bit codes q and an fp16 scale d
    /// of random sign per group of 64, w = d·(q − 1). Also returns each expert's gate, up ([I, D]) and down ([D, I]) in
    /// float.
    static func fnExperts(
        _ f: FlashNextFormat, slots: Int, slotOf: [Int], D: Int, I: Int, _ g: inout SplitMix64
    ) -> (arena: [UInt8], gate: [[Float]], up: [[Float]], down: [[Float]]) {
        var arena = [UInt8](repeating: 0, count: slots * f.recordBytes)
        var wG: [[Float]] = [], wU: [[Float]] = [], wD: [[Float]] = []
        func fill(_ e: Int, codes: Int, scales: Int, N: Int, Kd: Int) -> [Float] {
            var w = [Float](repeating: 0, count: N * Kd)
            arena.withUnsafeMutableBytes { p in
                let base = slotOf[e] * f.recordBytes
                for n in 0..<N {
                    for gi in 0..<(Kd / 64) {
                        let d = Float16(0.01 + 0.03 * g.uniform()) * (g.next() & 1 == 0 ? 1 : -1)
                        p.storeBytes(of: d, toByteOffset: base + scales + (n * (Kd / 64) + gi) * 2, as: Float16.self)
                        for j in 0..<64 {
                            let k = gi * 64 + j, q = UInt32(g.next() & 3)
                            let o = base + codes + (n * Kd + k) / 16 * 4
                            let old = p.load(fromByteOffset: o, as: UInt32.self)
                            p.storeBytes(of: old | q << UInt32(2 * (k % 16)), toByteOffset: o, as: UInt32.self)
                            w[n * Kd + k] = Float(d) * (Float(q) - 1)
                        }
                    }
                }
            }
            return w
        }
        for e in slotOf.indices {
            wG.append(fill(e, codes: f.gateCodes, scales: f.gateScales, N: I, Kd: D))
            wU.append(fill(e, codes: f.upCodes, scales: f.upScales, N: I, Kd: D))
            wD.append(fill(e, codes: f.downCodes, scales: f.downScales, N: D, Kd: I))
        }
        return (arena, wG, wU, wD)
    }

    static func sigmoid(_ x: Float) -> Float { 1 / (1 + exp(-x)) }

    // MARK: hyper-connections

    static let fnHC = KernelCase(
        id: "fn-hc", model: "qwen3.8-flash-next",
        catches: "Flash-Next hyper-connections (fn_gnorm per stream, fn_bf16_mm down|inject and up, fn_hc_act, "
            + "fn_hc_mix, fn_hc_inject, fn_stream_mean): mlx-vlm formula, layout of the 4 streams and that a row "
            + "does not depend on the others"
    ) { ctx, g in
        let D = 2560, W = 4 * D, R = 320
        let wn = Syn.normal(&g, W, sd: 0.2).map { 1 + $0 }
        let down = Syn.roundBF16(Syn.normal(&g, (R + 4) * W, sd: 0.02))
        let up = Syn.roundBF16(Syn.normal(&g, W * R, sd: 0.05))
        let nb = ctx.buffer(wn), db = Syn.bf16Buffer(ctx, down), ub = Syn.bf16Buffer(ctx, up)
        func gpu(_ x: [Float], _ br: [Float], rows: Int) throws -> (mixed: [Float], xout: [Float], mean: [Float]) {
            let S = ctx.buffer(x), nrm = ctx.buffer(length: rows * W * 4), dp = ctx.buffer(length: rows * 324 * 4)
            let low = ctx.buffer(length: rows * R * 4), inj = ctx.buffer(length: rows * 16),
                gg = ctx.buffer(length: rows * W * 4)
            let xh = ctx.buffer(length: rows * D * 2), xf = ctx.buffer(length: rows * D * 4), brb = ctx.buffer(br)
            let mean = ctx.buffer(length: rows * D * 2)
            try ctx.run { e in
                var np = (UInt32(D), Float(1e-6), UInt32(W), UInt32(W))
                e.setComputePipelineState(try ctx.pipeline("fn_gnorm"))
                e.setBuffer(S, offset: 0, index: 0); e.setBuffer(nb, offset: 0, index: 1)
                e.setBuffer(nrm, offset: 0, index: 2)
                e.setBytes(&np, length: 16, index: 3)
                e.setBuffer(ctx.buffer(length: rows * W * 2), offset: 0, index: 4)
                e.dispatchThreadgroups(
                    MTLSize(width: 4, height: rows, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                func mm(_ x: MTLBuffer, _ w: MTLBuffer, _ y: MTLBuffer, N: Int, K: Int, ldy: Int) throws {
                    var p = (UInt32(rows), UInt32(N), UInt32(K), UInt32(K), UInt32(ldy), UInt32(0))
                    e.setComputePipelineState(try ctx.pipeline("fn_bf16_mm"))
                    e.setBuffer(x, offset: 0, index: 0); e.setBuffer(w, offset: 0, index: 1)
                    e.setBuffer(y, offset: 0, index: 2)
                    e.setBytes(&p, length: 24, index: 3)
                    e.dispatchThreadgroups(
                        MTLSize(width: (N + 7) / 8, height: (rows + 15) / 16, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                }
                func th(_ name: String, _ n: Int, _ bufs: [MTLBuffer], _ prm: [UInt32]) throws {
                    e.setComputePipelineState(try ctx.pipeline(name))
                    for (i, b) in bufs.enumerated() { e.setBuffer(b, offset: 0, index: i) }
                    prm.withUnsafeBytes { e.setBytes($0.baseAddress!, length: $0.count, index: bufs.count) }
                    e.dispatchThreads(
                        MTLSize(width: n, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                }
                try mm(nrm, db, dp, N: R + 4, K: W, ldy: 324)
                try th(
                    "fn_hc_act", rows * (R + 4), [dp, low, inj, ctx.buffer(length: rows * R * 2)],
                    [UInt32(rows), UInt32(R), 324, 1])
                try mm(low, ub, gg, N: W, K: R, ldy: W)
                try th("fn_hc_mix", rows * D, [nrm, gg, xh, xf], [UInt32(rows), 1])
                try th("fn_hc_inject", rows * W, [S, brb, inj], [UInt32(rows)])
                try th("fn_stream_mean", rows * D, [S, mean], [UInt32(rows), UInt32(D)])
            }
            return (Syn.floats(xf, rows * D), Syn.floats(S, rows * W), readHalf(mean, count: rows * D))
        }
        func cpu(_ x: [Float], _ br: [Float], rows: Int) -> (mixed: [Float], xout: [Float], mean: [Float]) {
            var mixed = [Float](), xout = x, mean = [Float]()
            for r in 0..<rows {
                var n = [Float](repeating: 0, count: W)
                for s in 0..<4 {
                    let seg = x[(r * W + s * D)..<(r * W + s * D + D)]
                    let rr = 1 / (seg.reduce(0) { $0 + $1 * $1 } / Float(D) + 1e-6).squareRoot()
                    for i in 0..<D { n[s * D + i] = x[r * W + s * D + i] * rr * wn[s * D + i] }
                }
                let dpv = CPU.matmulT(n, down, M: 1, N: R + 4, K: W)
                let lowv = (0..<R).map { j -> Float in
                    let v = dpv[j] / 4; return v * sigmoid(v)
                }
                let injv = (0..<4).map { 2 * sigmoid(dpv[R + $0] / 4) }
                let gv = CPU.matmulT(lowv, up, M: 1, N: W, K: R)
                for d in 0..<D {
                    mixed.append((0..<4).reduce(0) { $0 + n[$1 * D + d] * sigmoid(gv[$1 * D + d]) } / 4)
                }
                for s in 0..<4 { for d in 0..<D { xout[r * W + s * D + d] += br[r * D + d] * injv[s] } }
                for d in 0..<D { mean.append(Float(Float16((0..<4).reduce(0) { $0 + xout[r * W + $1 * D + d] } / 4))) }
            }
            return (mixed, xout, mean)
        }
        var checks: [Check] = []
        for rows in [3, 16] {
            let x = Syn.normal(&g, rows * W), br = Syn.normal(&g, rows * D, sd: 0.5)
            let a = try gpu(x, br, rows: rows), b = cpu(x, br, rows: rows)
            checks += [
                Check(label: "mix, \(rows) rows", gpu: a.mixed, ref: b.mixed, tol: .maxRel(2e-4)),
                Check(label: "injection, \(rows) rows", gpu: a.xout, ref: b.xout, tol: .maxRel(2e-4)),
                Check(label: "stream mean, \(rows) rows", gpu: a.mean, ref: b.mean, tol: .maxRel(2e-3)),
            ]
        }
        // per-row invariance: row 0 alone and with 15 neighbors.
        let x = Syn.normal(&g, 16 * W), br = Syn.normal(&g, 16 * D)
        let one = try gpu(Array(x[0..<W]), Array(br[0..<D]), rows: 1), all = try gpu(x, br, rows: 16)
        checks.append(
            Check(
                label: "per-row invariance", gpu: one.mixed + one.xout,
                ref: Array(all.mixed[0..<D]) + Array(all.xout[0..<W]),
                tol: .bitExact))
        return checks
    }

    // MARK: PLE

    static let fnPLE = KernelCase(
        id: "fn-ple", model: "qwen3.8-flash-next",
        catches: "Flash-Next PLE: n-gram hash on the CPU (pleIndices) by hand with EOS that cuts segments; gather of "
            + "4-bit g32 rows (fn_ple_gather); gate and norms (fn_ple_gate); conv dilated with 9 history rows in "
            + "prefill and in rounds with tape and nacc (fn_ple_conv)"
    ) { ctx, g in
        let f = try fnFormat()
        var checks: [Check] = []
        // 1) hash: CPU (FlashNextFormat, what the GPU's fn_ple_round mirrors) by hand, with EOS in the context.
        let eos = f.pleEos
        let ctxT: [UInt32] = [17, eos], toks: [UInt32] = [5, 248_319, eos, 7, 7, 0, 123_456, eos, eos, 99]
        let cpuIdx = f.pleIndices(context: ctxT, tokens: toks)
        // By hand: first token after the context [17, EOS]: its n-grams are (5, EOS) and (5, EOS, EOS).
        let m = f.pleMult
        let h2 = (UInt64(5) &* m[0]) ^ (UInt64(eos) &* m[1]), h3 = h2 ^ (UInt64(eos) &* m[2])
        let hand = (0..<16).map { k in Float(UInt32((k < 8 ? h2 : h3) % f.pleSize[k] + f.pleOff[k])) }
        checks.append(
            Check(label: "hash by hand (row 0)", gpu: cpuIdx[0..<16].map { Float($0) }, ref: hand, tol: .bitExact))
        // 2) gather: random rows of 100 B.
        let T = 5
        var raw = [UInt8](repeating: 0, count: T * 16 * 100)
        var want = [Float](repeating: 0, count: T * 2560)
        for j in 0..<(T * 16) {
            var codes = [UInt32](repeating: 0, count: 20)
            let sc = (0..<5).map { _ in floatToBF16(0.01 + 0.02 * g.uniform()) },
                bi = (0..<5).map { _ in floatToBF16(-0.1 * g.uniform()) }
            for k in 0..<160 {
                let q = UInt32(g.next() & 15)
                codes[k / 8] |= q << UInt32(4 * (k % 8))
                want[(j / 16) * 2560 + (j % 16) * 160 + k] = Float(
                    Float16(Float(q) * bf16ToFloat(sc[k / 32]) + bf16ToFloat(bi[k / 32])))
            }
            raw.withUnsafeMutableBytes { p in
                for k in 0..<20 { p.storeBytes(of: codes[k], toByteOffset: j * 100 + 4 * k, as: UInt32.self) }
                for k in 0..<5 {
                    p.storeBytes(of: sc[k], toByteOffset: j * 100 + 80 + 2 * k, as: UInt16.self)
                    p.storeBytes(of: bi[k], toByteOffset: j * 100 + 90 + 2 * k, as: UInt16.self)
                }
            }
        }
        let rb = ctx.buffer(raw), sb = ctx.buffer((0..<(T * 16)).map { UInt32($0) }),
            eb = ctx.buffer(length: T * 2560 * 2)
        try ctx.run { e in
            e.setComputePipelineState(try ctx.pipeline("fn_ple_gather"))
            e.setBuffer(rb, offset: 0, index: 0); e.setBuffer(sb, offset: 0, index: 1)
            e.setBuffer(eb, offset: 0, index: 2)
            var n = UInt32(T)
            e.setBytes(&n, length: 4, index: 3)
            e.dispatchThreads(
                MTLSize(width: T * 2560, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
        checks.append(Check(label: "gather 4-bit g32", gpu: readHalf(eb, count: T * 2560), ref: want, tol: .bitExact))
        // 3) gate + conv: 13-row prefill over a history and then a 3-row round with nacc = 2.
        let D = 2560, W = 4 * D
        let wk = Syn.normal(&g, W, sd: 0.1).map { 1 + $0 }, wq = Syn.normal(&g, W, sd: 0.1).map { 1 + $0 }
        let wc = Syn.normal(&g, W, sd: 0.1).map { 1 + $0 }, cw = Syn.normal(&g, W * 4, sd: 0.3)
        let hist0 = Syn.normal(&g, (9 + 16) * W)
        let hp0 = ctx.buffer(hist0), hp1 = ctx.buffer(length: (9 + 16) * W * 4),
            hp2 = ctx.buffer(length: (9 + 16) * W * 4)
        func cpuStep(_ key: [Float], _ val: [Float], _ x: [Float], window: [Float], rows: Int) -> (
            x: [Float], normd: [Float]
        ) {
            var xo = x, nd = [Float](repeating: 0, count: rows * W)
            var gated = [Float](repeating: 0, count: rows * W)
            for r in 0..<rows {
                for s in 0..<4 {
                    var sk: Float = 0, sq: Float = 0, dk: Float = 0
                    for i in 0..<D {
                        let a = key[r * W + s * D + i], b = x[r * W + s * D + i]
                        sk += a * a; sq += b * b; dk += a * wk[s * D + i] * b * wq[s * D + i]
                    }
                    var gg =
                        dk / (sk / Float(D) + 1e-6).squareRoot() / (sq / Float(D) + 1e-6).squareRoot()
                        / Float(D).squareRoot()
                    gg = (gg < 0 ? -1 : gg > 0 ? 1 : 0) * max(abs(gg), 1e-6).squareRoot()
                    let sg = sigmoid(gg)
                    var sv: Float = 0
                    for i in 0..<D { let a = sg * val[r * D + i]; sv += a * a; gated[r * W + s * D + i] = a }
                    let rc = 1 / (sv / Float(D) + 1e-6).squareRoot()
                    for i in 0..<D { nd[r * W + s * D + i] = gated[r * W + s * D + i] * rc * wc[s * D + i] }
                }
            }
            for t in 0..<rows {
                for c in 0..<W {
                    var acc: Float = 0
                    for j in 0..<4 {
                        let i = t - 3 * (3 - j)
                        acc += cw[c * 4 + j] * (i < 0 ? window[(9 + i) * W + c] : nd[i * W + c])
                    }
                    xo[t * W + c] += gated[t * W + c] + acc * sigmoid(acc)
                }
            }
            return (xo, nd)
        }
        func gpuStep(
            _ key: [Float], _ val: [Float], _ x: [Float], rows: Int, nacc: Int, tape: Bool, hin: MTLBuffer,
            hout: MTLBuffer
        ) throws -> [Float] {
            let kb = ctx.buffer(key), vb = ctx.buffer(val), xb = ctx.buffer(x)
            let gb = ctx.buffer(length: rows * W * 4), nbf = ctx.buffer(length: rows * W * 4)
            let wkb = ctx.buffer(wk), wqb = ctx.buffer(wq), wcb = ctx.buffer(wc), cwb = ctx.buffer(cw)
            try ctx.run { e in
                var ep = Float(1e-6)
                e.setComputePipelineState(try ctx.pipeline("fn_ple_gate"))
                for (i, b) in [kb, vb, xb, wkb, wqb, wcb, gb, nbf].enumerated() { e.setBuffer(b, offset: 0, index: i) }
                e.setBytes(&ep, length: 4, index: 8)
                e.dispatchThreadgroups(
                    MTLSize(width: 4, height: rows, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                var cp = (UInt32(rows), UInt32(nacc), UInt32(tape ? 1 : 0), UInt32(0))
                e.setComputePipelineState(try ctx.pipeline("fn_ple_conv"))
                for (i, b) in [nbf, gb, cwb, hin, hout, xb].enumerated() { e.setBuffer(b, offset: 0, index: i) }
                e.setBytes(&cp, length: 16, index: 6)
                e.setBuffer(kb, offset: 0, index: 7)
                e.dispatchThreads(
                    MTLSize(width: rows * W, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            }
            return Syn.floats(xb, rows * W)
        }
        // 13-row prefill: window = rows 0..8 of hist0.
        let r1 = 13
        let k1 = Syn.normal(&g, r1 * W), v1 = Syn.normal(&g, r1 * D), x1 = Syn.normal(&g, r1 * W)
        let c1 = cpuStep(k1, v1, x1, window: Array(hist0[0..<(9 * W)]), rows: r1)
        checks.append(
            Check(
                label: "PLE 13-row prefill",
                gpu: try gpuStep(k1, v1, x1, rows: r1, nacc: 0, tape: false, hin: hp0, hout: hp1),
                ref: c1.x, tol: .maxRel(2e-4)))
        let win1 = Array(c1.normd[((r1 - 9) * W)..<(r1 * W)])
        checks.append(
            Check(label: "window after the prefill", gpu: Syn.floats(hp1, 9 * W), ref: win1, tol: .maxRel(2e-5)))
        // 3-row round (tape) from that window, and another 2-row one that folds nacc = 2 of the previous one.
        let k2 = Syn.normal(&g, 3 * W), v2 = Syn.normal(&g, 3 * D), x2 = Syn.normal(&g, 3 * W)
        let c2 = cpuStep(k2, v2, x2, window: win1, rows: 3)
        checks.append(
            Check(
                label: "PLE 3-row round",
                gpu: try gpuStep(k2, v2, x2, rows: 3, nacc: 0, tape: true, hin: hp1, hout: hp2),
                ref: c2.x, tol: .maxRel(2e-4)))
        let win2 = Array((win1 + c2.normd)[(2 * W)..<(11 * W)])
        let k3 = Syn.normal(&g, 2 * W), v3 = Syn.normal(&g, 2 * D), x3 = Syn.normal(&g, 2 * W)
        let c3 = cpuStep(k3, v3, x3, window: win2, rows: 2)
        checks.append(
            Check(
                label: "PLE round after accepting 2",
                gpu: try gpuStep(k3, v3, x3, rows: 2, nacc: 2, tape: true, hin: hp2, hout: hp0),
                ref: c3.x, tol: .maxRel(2e-4)))
        return checks
    }

    // MARK: MoE Q2

    static let fnMoE = KernelCase(
        id: "fn-moe-q2", model: "qwen3.8-flash-next",
        catches: "Flash-Next MoE: top-10 of 512 with the renormalized softmax and the shared-expert gate "
            + "(moe_route_topk_fn), grouped by expert in tiles (moe_group), gate|up and down Q2 with fp16 scale and "
            + "w = d·(q − 1) read from the registry via the expert → slot table (round vnax and prefill NAX); "
            + "combining with the shared expert is done on the CPU; round of 1, 7 and 16 rows and a chunk of 100"
    ) { ctx, g in
        let f = try fnFormat()
        let D = 2560, I = 640, E = 512, K = 10, nExp = 24, slots = 32
        // 24 distinct experts in permuted slots; the router only picks among them.
        let slotOf = (0..<nExp).map { (($0 * 7) + 3) % slots }
        let (arena, wG, wU, wD) = fnExperts(f, slots: slots, slotOf: slotOf, D: D, I: I, &g)
        let expertId = (0..<nExp).map { ($0 * 37 + 11) % E }  // expert id in [0, 512)
        var smap = [UInt32](repeating: 0, count: E)
        for (i, e) in expertId.enumerated() { smap[e] = UInt32(slotOf[i]) }
        let ab = ctx.buffer(arena), mb = ctx.buffer(smap)
        var checks: [Check] = []
        for T in [1, 7, 16, 100] {
            let x = Syn.roundHalf(Syn.normal(&g, T * D, sd: 0.5))
            var logits = [Float](repeating: -30, count: T * 576)
            for r in 0..<T {
                for (i, e) in expertId.enumerated() { logits[r * 576 + e] = Syn.normal(&g, 1)[0] + Float(i % 3) * 0.1 }
                logits[r * 576 + E] = Syn.normal(&g, 1)[0]
            }
            let sh = Syn.normal(&g, T * D)
            // CPU
            var refOut = [Float](repeating: 0, count: T * D), refInds = [Float](), refW = [Float]()
            for r in 0..<T {
                let l = Array(logits[(r * 576)..<(r * 576 + E)])
                let top = l.indices.sorted { l[$0] > l[$1] || (l[$0] == l[$1] && $0 < $1) }.prefix(K)
                let mx = l[top.first!]
                let ex = top.map { exp(l[$0] - mx) }, s = ex.reduce(0, +)
                let xr = Array(x[(r * D)..<(r * D + D)])
                for (k, e) in top.enumerated() {
                    let i = expertId.firstIndex(of: e)!
                    let gg = CPU.matmulT(xr, wG[i], M: 1, N: I, K: D), uu = CPU.matmulT(xr, wU[i], M: 1, N: I, K: D)
                    let h = (0..<I).map { gg[$0] * sigmoid(gg[$0]) * uu[$0] }
                    let y = CPU.matmulT(h, wD[i], M: 1, N: D, K: I)
                    for n in 0..<D { refOut[r * D + n] += ex[k] / s * y[n] }
                    refInds.append(Float(e)); refW.append(ex[k] / s)
                }
                let sg = sigmoid(logits[r * 576 + E])
                for n in 0..<D { refOut[r * D + n] += sg * sh[r * D + n] }
            }
            // GPU
            let P = T * K
            let lb = ctx.buffer(logits), ib = ctx.buffer(length: P * 4), wb = ctx.buffer(length: P * 4),
                sgb = ctx.buffer(length: T * 4)
            let srt = ctx.buffer(length: P * 4), pos = ctx.buffer(length: P * 4), tl = ctx.buffer(length: (P + E) * 16)
            let offs = ctx.buffer(length: (E + 1) * 4)
            let args = ctx.buffer(length: 32), xb = Syn.halfBuffer(ctx, x), hb = ctx.buffer(length: P * I * 4)
            let yb = ctx.buffer(length: P * D * 4)
            // Two paths: round vnax (tiles of ≤ 16) and prefill NAX (tiles of 32, h in half).
            for (variant, guN, dnN, bm) in [("vnax", I / 16, D / 32, 0), ("nax", I / 64, D / 128, 32)] {
                try ctx.run { e in
                    var rp = (UInt32(T), UInt32(E), UInt32(576), UInt32(K))
                    e.setComputePipelineState(try ctx.pipeline("moe_route_topk_fn"))
                    for (i, b) in [lb, lb, lb, lb, lb, ib, wb, sgb].enumerated() { e.setBuffer(b, offset: 0, index: i) }
                    e.setBytes(&rp, length: 16, index: 8)
                    e.dispatchThreadgroups(
                        MTLSize(width: (T + 3) / 4, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
                    let tr = UInt32(bm > 0 ? bm : 16)
                    var gp = (UInt32(P), UInt32(E), tr, tr, UInt32(guN), UInt32(dnN), UInt32(0), UInt32(0))
                    e.setComputePipelineState(try ctx.pipeline("moe_group"))
                    for (i, b) in [ib, offs, srt, pos, tl, tl].enumerated() { e.setBuffer(b, offset: 0, index: i) }
                    e.setBuffer(args, offset: 0, index: 6)
                    e.setBuffer(args, offset: 16, index: 7)
                    e.setBytes(&gp, length: 32, index: 8)
                    e.setBuffer(ib, offset: 0, index: 9)
                    e.dispatchThreadgroups(
                        MTLSize(width: 1, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
                    var qp = (
                        UInt32(f.recordBytes), UInt32(f.gateCodes), UInt32(f.gateScales), UInt32(f.upCodes),
                        UInt32(f.upScales), UInt32(f.downCodes), UInt32(f.downScales), UInt32(T)
                    )
                    let names = [
                        "nax": ("fn_moe_gateup_nax_bm32_bk64", "fn_moe_down_nax_bm32_bk64"),
                        "vnax": ("fn_moe_gateup_vnax_sg4", "fn_moe_down_vnax_sg2"),
                    ][variant]!
                    let tpg = MTLSize(width: 128, height: 1, depth: 1)
                    let tpgB = variant == "vnax" ? MTLSize(width: 64, height: 1, depth: 1) : tpg
                    e.setComputePipelineState(try ctx.pipeline(names.0))
                    for (i, b) in [xb, ab, mb, tl, srt, hb].enumerated() { e.setBuffer(b, offset: 0, index: i) }
                    e.setBytes(&qp, length: 32, index: 6)
                    e.dispatchThreadgroups(indirectBuffer: args, indirectBufferOffset: 0, threadsPerThreadgroup: tpg)
                    e.setComputePipelineState(try ctx.pipeline(names.1))
                    for (i, b) in [hb, ab, mb, tl, yb].enumerated() { e.setBuffer(b, offset: 0, index: i) }
                    e.setBytes(&qp, length: 32, index: 5)
                    e.dispatchThreadgroups(indirectBuffer: args, indirectBufferOffset: 16, threadsPerThreadgroup: tpgB)
                }
                // Combination on the CPU with the routing, the positions and the GPU outputs.
                let y = Syn.floats(yb, P * D), pp = Syn.uints(pos, P), ww = Syn.floats(wb, P), sgv = Syn.floats(sgb, T)
                var out = [Float](repeating: 0, count: T * D)
                for r in 0..<T {
                    for n in 0..<D {
                        var acc: Float = 0
                        for k in 0..<K { acc += ww[r * K + k] * y[Int(pp[r * K + k]) * D + n] }
                        out[r * D + n] = acc + sgv[r] * sh[r * D + n]
                    }
                }
                if variant == "vnax" {
                    checks.append(
                        Check(
                            label: "top-10, \(T) rows", gpu: Syn.uints(ib, P).map { Float($0) }, ref: refInds,
                            tol: .bitExact))
                    checks.append(
                        Check(label: "weights, \(T) rows", gpu: Syn.floats(wb, P), ref: refW, tol: .maxRel(1e-5)))
                }
                // NAX: h in half between gate|up and down.
                checks.append(
                    Check(
                        label: "MoE output (\(variant)), \(T) rows", gpu: out, ref: refOut, tol: .maxRel(6e-3)))
            }
        }
        return checks
    }

    // MARK: GDN with sigmoid gate

    static let fnGDNGate = KernelCase(
        id: "fn-gdn-gate", model: "qwen3.8-flash-next",
        catches: "Flash-Next GDN gated norm (gdn_normgate with LAB_GDN_SIG: σ(z) instead of SiLU, "
            + "fp32 weights and 48 v heads)"
    ) { ctx, g in
        let HV = 48, B = 3, VD = HV * 128, C = 2 * 2048 + VD, ldp = C + VD + 2 * HV
        let y = Syn.normal(&g, B * VD), proj = Syn.normal(&g, B * ldp), nw = Syn.normal(&g, 128, sd: 0.2).map { 1 + $0 }
        let ob = ctx.buffer(length: B * VD * 2)
        try ctx.run { e in
            var gp = (UInt32(B), UInt32(0), UInt32(ldp), Float(1e-6), UInt32(0), UInt32(0))
            e.setComputePipelineState(try ctx.pipeline("gdn_normgate"))
            for (i, b) in [ctx.buffer(y), ctx.buffer(proj), ctx.buffer(nw), ob].enumerated() {
                e.setBuffer(b, offset: 0, index: i)
            }
            e.setBytes(&gp, length: 24, index: 4)
            e.dispatchThreadgroups(
                MTLSize(width: (B * HV + 3) / 4, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        }
        var ref = [Float]()
        for t in 0..<B {
            for h in 0..<HV {
                let v = Array(y[(t * VD + h * 128)..<(t * VD + h * 128 + 128)])
                let r = 1 / (v.reduce(0) { $0 + $1 * $1 } / 128 + 1e-6).squareRoot()
                for i in 0..<128 {
                    ref.append(Float(Float16(v[i] * r * nw[i] * sigmoid(proj[t * ldp + C + h * 128 + i]))))
                }
            }
        }
        return [Check(label: "sigmoid normgate", gpu: readHalf(ob, count: B * VD), ref: ref, tol: .maxRel(2e-3))]
    }
}

// MARK: QSA in isolation

extension KernelCase {
    static let fnQSA = KernelCase(
        id: "fn-qsa", model: "qwen3.8-flash-next",
        catches:
            "Flash-Next QSA in isolation (zg_qsa.metal): indexer queries and block keys (norm 1 + w, "
            + "64-dim RoPE), Σ relu score, top-512 with ties going to the lower-index block, the own block "
            + "that competes and can lose, and attention over the bitmap; dense with ≤ 512 visible blocks"
    ) { ctx, g in
        let D = 128, H = 4, theta: Float = 1e7, eps: Float = 1e-6
        let hist = 2196, rows = 4, pos0 = hist, nb = (pos0 + rows) / 4  // 550 blocks; rows with 549–550 visible
        let wq = Syn.normal(&g, D, sd: 0.1).map { 1 + $0 }, wk = Syn.normal(&g, D, sd: 0.1).map { 1 + $0 }
        var rawk = Syn.normal(&g, (pos0 + rows) * D)
        // Ties: blocks 10, 20 and 30 equal; the last block (row 3's own) with keys that give 0.
        for b in [20, 30] { for j in 0..<(4 * D) { rawk[b * 4 * D + j] = rawk[10 * 4 * D + j] } }
        var qk = Syn.normal(&g, rows * 640)
        // Null keys in row 3's own block: score 0 (tie goes to the lower index).
        for r in 0..<rows { for d in 0..<D { qk[r * 640 + 512 + d] = 0 } }
        // CPU
        func norm(_ v: [Float], _ w: [Float]) -> [Float] {
            let rr = 1 / (v.reduce(0) { $0 + $1 * $1 } / Float(D) + eps).squareRoot()
            return v.enumerated().map { $1 * rr * w[$0] }
        }
        func rope(_ x: [Float], _ pos: Int) -> [Float] {
            var y = x
            for d in 0..<64 {
                let i = d % 32, ang = Float(pos) * pow(theta, -Float(i) / 32), c = cos(ang), s = sin(ang)
                y[d] = d < 32 ? x[d] * c - x[d + 32] * s : x[d] * c + x[d - 32] * s
            }
            return y
        }
        var raw = rawk
        for r in 0..<rows { for d in 0..<D { raw[(pos0 + r) * D + d] = qk[r * 640 + 512 + d] } }
        let bkeys = (0..<nb).map { b -> [Float] in
            let m = (0..<D).map { d in (0..<4).reduce(Float(0)) { $0 + raw[(4 * b + $1) * D + d] } / 4 }
            return rope(norm(m, wk), 4 * b)
        }
        var refBits = [Float](), refScores = [[Float]]()
        let nw = (nb + 31) / 32
        var selected: [Set<Int>] = []
        for r in 0..<rows {
            let qs = (0..<H).map { h in rope(norm(Array(qk[(r * 640 + h * D)..<(r * 640 + h * D + D)]), wq), pos0 + r) }
            let vis = (pos0 + r + 1) / 4
            let sc = (0..<nb).map { b -> Float in
                guard b < vis else { return -.infinity }
                return qs.reduce(Float(0)) { acc, qv in acc + max(0, zip(qv, bkeys[b]).reduce(0) { $0 + $1.0 * $1.1 }) }
                    / Float(D).squareRoot()
            }
            refScores.append(sc)
            let order = (0..<vis).sorted { sc[$0] > sc[$1] || (sc[$0] == sc[$1] && $0 < $1) }
            let sel = Set(vis > 512 ? Array(order.prefix(512)) : Array(0..<vis))
            selected.append(sel)
            for w in 0..<nw {
                var bits: UInt32 = 0
                for j in 0..<32 where sel.contains(w * 32 + j) { bits |= 1 << UInt32(j) }
                refBits.append(Float(bits))
            }
        }
        // GPU
        let qkb = ctx.buffer(qk), rkb = ctx.buffer(rawk), qb = ctx.buffer(length: rows * H * D * 4)
        let bkb = ctx.buffer(length: nb * D * 4), sb = ctx.buffer(length: rows * nb * 4),
            bmb = ctx.buffer(length: rows * nw * 4)
        var p = (UInt32(rows), UInt32(pos0), UInt32(nb), UInt32(D), eps, theta, UInt32(0), UInt32(0))
        try ctx.run { e in
            e.setComputePipelineState(try ctx.pipeline("qsa_prep"))
            for (i, b) in [qkb, ctx.buffer(wq), qb, rkb].enumerated() { e.setBuffer(b, offset: 0, index: i) }
            e.setBytes(&p, length: 32, index: 4)
            e.dispatchThreadgroups(
                MTLSize(width: 5, height: rows, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            var b0 = UInt32(0)
            e.setComputePipelineState(try ctx.pipeline("qsa_blockkeys"))
            for (i, b) in [rkb, ctx.buffer(wk), bkb].enumerated() { e.setBuffer(b, offset: 0, index: i) }
            e.setBytes(&p, length: 32, index: 3)
            e.setBytes(&b0, length: 4, index: 4)
            e.dispatchThreadgroups(
                MTLSize(width: nb, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)
            )
            e.setComputePipelineState(try ctx.pipeline("qsa_score"))
            for (i, b) in [qb, bkb, sb].enumerated() { e.setBuffer(b, offset: 0, index: i) }
            e.setBytes(&p, length: 32, index: 3)
            e.dispatchThreadgroups(
                MTLSize(width: (nb + 7) / 8, height: rows, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            e.setComputePipelineState(try ctx.pipeline("qsa_select"))
            e.setBuffer(sb, offset: 0, index: 0); e.setBuffer(bmb, offset: 0, index: 1)
            e.setBytes(&p, length: 32, index: 2)
            e.dispatchThreadgroups(
                MTLSize(width: rows, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
        }
        let gs = Syn.floats(sb, rows * nb), rs = refScores.flatMap { $0 }
        var checks = [
            Check(
                label: "scores", gpu: gs.map { $0.isFinite ? $0 : -1 }, ref: rs.map { $0.isFinite ? $0 : -1 },
                tol: .maxRel(1e-4)),
            Check(
                label: "top-512 bitmap (ties to the lower index)", gpu: Syn.uints(bmb, rows * nw).map { Float($0) },
                ref: refBits, tol: .bitExact),
        ]
        // Row 3's own block (549, complete) does not get in: the token does not attend to itself.
        checks.append(
            Check(
                label: "own block outside the top-512", gpu: [selected[3].contains(549) ? 1 : 0], ref: [0],
                tol: .bitExact))
        // Attention over the bitmap.
        let qh = 24, kvh = 2, T = pos0 + rows
        let qa = Syn.normal(&g, rows * qh * 256, sd: 0.1), ka = Syn.normal(&g, T * kvh * 256),
            va = Syn.normal(&g, T * kvh * 256)
        let ob = ctx.buffer(length: rows * qh * 256 * 4)
        var ap = (
            UInt32(rows), UInt32(pos0), UInt32(nb), UInt32(qh), UInt32(kvh), Float(1 / 16.0), UInt32(0), UInt32(0)
        )
        try ctx.run { e in
            e.setComputePipelineState(try ctx.pipeline("qsa_attn_ref"))
            for (i, b) in [ctx.buffer(qa), ctx.buffer(ka), ctx.buffer(va), bmb, ob].enumerated() {
                e.setBuffer(b, offset: 0, index: i)
            }
            e.setBytes(&ap, length: 32, index: 5)
            e.dispatchThreadgroups(
                MTLSize(width: qh, height: rows, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
        var refO = [Float]()
        for r in 0..<rows {
            let pos = pos0 + r, vis = (pos + 1) / 4
            let toks = (0...pos).filter { t in !(vis > 512 && t / 4 < vis && !selected[r].contains(t / 4)) }
            for h in 0..<qh {
                let kh = h / (qh / kvh)
                let s = toks.map { t in
                    (0..<256).reduce(Float(0)) { $0 + qa[(r * qh + h) * 256 + $1] * ka[(t * kvh + kh) * 256 + $1] } / 16
                }
                let mx = s.max()!, e = s.map { exp($0 - mx) }, l = e.reduce(0, +)
                for d in 0..<256 {
                    refO.append(zip(toks, e).reduce(Float(0)) { $0 + $1.1 * va[($1.0 * kvh + kh) * 256 + d] } / l)
                }
            }
        }
        checks.append(
            Check(
                label: "bitmap attention", gpu: Syn.floats(ob, rows * qh * 256), ref: refO, tol: .maxRel(1e-4)))
        return checks
    }
}

// MARK: engine QSA against the isolated one

extension KernelCase {
    static let fnQSAM = KernelCase(
        id: "fn-qsa-m", model: "qwen3.8-flash-next",
        catches:
            "qsa_m_score (one simdgroup per block for up to 16 rows) and qsa_m_select (radix digit and final bitmap "
            + "without single-thread loops) give the same bits as qsa_score and qsa_select (the old algorithm), with "
            + "1–40 rows, with ≤ 512 visible, with 32K and with many ties at the threshold"
    ) { ctx, g in
        let D = 128, H = 4
        // (blocks, rows, P, block keys: 0 normal, 1 all null, 2 with few distinct values)
        let cases: [(nb: Int, rows: Int, pos0: Int, kind: Int)] = [
            (8200, 16, 4 * 8200 - 40, 0), (8200, 40, 4 * 8200 - 60, 0), (8200, 1, 4 * 8200 - 4, 0),
            (2060, 5, 2051, 0), (300, 4, 1000, 0), (8200, 5, 4 * 8200 - 30, 1), (8200, 16, 4 * 8200 - 40, 2),
            (1500, 13, 4 * 1500 - 60, 2),
        ]
        func bits(_ v: [UInt32]) -> [Float] { v.flatMap { [Float($0 & 0xFFFF), Float($0 >> 16)] } }
        var checks: [Check] = []
        for c in cases {
            let q = Syn.normal(&g, c.rows * H * D, sd: 0.2)
            var bk: [Float]
            switch c.kind {
            case 1: bk = [Float](repeating: 0, count: c.nb * D)
            case 2:
                // Only 2 dimensions with values in {−1, 0, 1}: scores with very few distinct values.
                bk = [Float](repeating: 0, count: c.nb * D)
                for b in 0..<c.nb { for d in 0..<2 { bk[b * D + d] = Float(Int(g.uniform() * 3) - 1) } }
            default: bk = Syn.normal(&g, c.nb * D)
            }
            let qb = ctx.buffer(q), bkb = ctx.buffer(bk), nw = (c.nb + 31) / 32
            let s0 = ctx.buffer(length: c.rows * c.nb * 4), s1 = ctx.buffer(length: c.rows * c.nb * 4)
            let m0 = ctx.buffer(length: c.rows * nw * 4), m1 = ctx.buffer(length: c.rows * nw * 4)
            let st = ctx.buffer(length: 16)
            var p = (
                UInt32(c.rows), UInt32(c.pos0), UInt32(c.nb), UInt32(0), Float(1e-6), Float(1e7), UInt32(0), UInt32(0)
            )
            try ctx.run { e in
                for (score, sel, sb, mb, m) in [
                    ("qsa_score", "qsa_select", s0, m0, false), ("qsa_m_score", "qsa_m_select", s1, m1, true),
                ] {
                    e.setComputePipelineState(try ctx.pipeline(score))
                    for (i, b) in [qb, bkb, sb].enumerated() { e.setBuffer(b, offset: 0, index: i) }
                    e.setBytes(&p, length: 32, index: 3)
                    if m { e.setBuffer(st, offset: 0, index: 4) }
                    e.dispatchThreadgroups(
                        MTLSize(width: (c.nb + 7) / 8, height: m ? (c.rows + 15) / 16 : c.rows, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                    e.setComputePipelineState(try ctx.pipeline(sel))
                    e.setBuffer(sb, offset: 0, index: 0); e.setBuffer(mb, offset: 0, index: 1)
                    e.setBytes(&p, length: 32, index: 2)
                    if m { e.setBuffer(st, offset: 0, index: 3) }
                    e.dispatchThreadgroups(
                        MTLSize(width: c.rows, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
                }
            }
            let tag = "nb \(c.nb), \(c.rows) rows, P \(c.pos0), keys \(c.kind)"
            checks.append(
                Check(
                    label: "scores (\(tag))", gpu: bits(Syn.uints(s1, c.rows * c.nb)),
                    ref: bits(Syn.uints(s0, c.rows * c.nb)), tol: .bitExact))
            let b1 = Syn.uints(m1, c.rows * nw), b0 = Syn.uints(m0, c.rows * nw)
            checks.append(Check(label: "bitmap (\(tag))", gpu: bits(b1), ref: bits(b0), tol: .bitExact))
            // Each row with more than 512 visible picks exactly 512.
            let want = (0..<c.rows).map { r in Float(min(512, min((c.pos0 + r + 1) / 4, c.nb))) }
            let got = (0..<c.rows).map { r in Float(b1[(r * nw)..<((r + 1) * nw)].reduce(0) { $0 + $1.nonzeroBitCount })
            }
            checks.append(Check(label: "chosen blocks per row (\(tag))", gpu: got, ref: want, tol: .bitExact))
        }
        return checks
    }
}
