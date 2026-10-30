import Foundation
import Testing

@testable import MetalLab

// RoPE with a per-layer frequency table (LAB_AT_ROPETAB) in vattn_prework_i8 (rounds and prefill) against
// the CPU with the same table, and the Hadamard of q and k that follows it.

extension KernelCase {
    /// gpt-oss-20b's tables: its YaRN from the config (θ 150000, rotated 64, factor 32 over 4096 with `truncate:
    /// false`) and the default one over the same θ, at position 0, around the original context (4096) and past 32K.
    static let attnRopeTab = KernelCase(
        id: "attn-ropetab", model: "gpt-oss-20b",
        catches: "RoPE from a per-layer table (inv_freq + mscale, buffer 11, LAB_AT_ROPETAB) in vattn_prework_i8: "
            + "YaRN and default tables, rounds and prefill at 0, ~4K and >32K positions"
    ) { ctx, g in
        let spec = try ModelSpec(ModelSpecTests.gptOss(quantization: ModelSpecTests.mxfp4Q8))
        let yarn = try #require(spec.rope.scaling)
        let rot = ctx.dims.attnRot, theta = Double(spec.rope.theta)
        let tables: [(String, [Float])] = [
            ("yarn", RopeTable.make(theta: theta, rotaryDims: rot, scaling: yarn)),
            ("default", RopeTable.make(theta: theta, rotaryDims: rot, scaling: nil)),
        ]
        var checks: [Check] = []
        for (name, tab) in tables {
            for P in [0, 4090, 33_000] {
                let f = try AttnFixture(
                    ctx, positions: P, &g, maxCtx: 34_000, qkvBias: ctx.dims.attnQKVB, ropeTable: tab)
                // The rotated queries (vattn_prework_i8 alone): the RoPE without the int8 KV and softmax noise.
                let xq = Syn.roundHalf(Syn.normal(&g, 16 * f.hidden))
                checks.append(
                    Check(
                        label: "\(name): rotated q of 16 rows at \(P)", gpu: try f.preworkQ(xq, rows: 16, P: P),
                        ref: f.referenceQ(xq, rows: 16, P: P), tol: .maxRel(2e-3)))
                let x = Syn.roundHalf(Syn.normal(&g, 16 * f.hidden))
                checks.append(
                    Check(
                        label: "\(name): round of 16 rows at \(P)", gpu: try f.round(x, rows: 16, P: P),
                        ref: f.reference(x, rows: 16, P: P), tol: .maxRel(5e-3)))
                let xp = Syn.roundHalf(Syn.normal(&g, 100 * f.hidden))
                // Norm-wise: the 100-row prefill has isolated elements up to 7e-3 off in max-relative terms with the
                // default table too (int8 K steps), so the RoPE itself is checked on q above.
                checks.append(
                    Check(
                        label: "\(name): prefill of 100 rows at \(P + 16)",
                        gpu: try f.prefill(xp, rows: 100, P: P + 16),
                        ref: f.reference(xp, rows: 100, P: P + 16), tol: .rms(2e-3)))
            }
        }
        return checks
    }
}

extension KernelCase {
    /// Hadamard of q and k in vattn_prework_i8, for heads of 256, 128 and 64 (8, 7 and 6 butterfly steps, 3, 2 and 1
    /// of them through threadgroup memory).
    static let attnHadamard: [KernelCase] = [
        ("qwen3.6-35b-a3b", 256), ("muse-glimmer-30b", 128), ("lfm2.5-24b-a2b", 64),
    ].map { model, hd in
        KernelCase(
            id: "attn-hadk-\(hd)", model: model,
            catches: "Hadamard of q and k after the RoPE in vattn_prework_i8 (head of \(hd)): the cache holds int8 of "
                + "H k and the query leaves as H q, so the score (H q) . (H k_hat) equals the unrotated q . k_hat"
        ) { ctx, g in
            let P = 40, rows = 16
            let f = try AttnFixture(ctx, positions: P, &g)
            let x = Syn.roundHalf(Syn.normal(&g, rows * f.hidden))
            let qg = try f.preworkQ(x, rows: rows, P: P)
            let hd = f.hd, ld = f.a.ldkv
            let kc = f.a.kc.contents().bindMemory(to: Int8.self, capacity: f.kvH * ld * hd)
            let ks = f.a.ks.contents().bindMemory(to: Float.self, capacity: f.kvH * ld)
            var kGPU = [Float](), kRef = [Float](), sGPU = [Float](), sRef = [Float]()
            for t in 0..<rows {
                let pr = f.projRow(x, t)
                // CPU, unrotated basis: k_hat = H^T kv8(H k) (H is symmetric), the key the rotated cache stands for.
                var kHat: [[Float]] = []
                for h in 0..<f.kvH {
                    let k0 = f.qH * f.qs + h * hd
                    let (kq, sk) = CPU.quantI8(f.hadamard(f.normRope(pr[k0..<(k0 + hd)], f.knw, pos: P + t))[...])
                    let kd = kq.map { $0 * sk }
                    kRef += kd
                    kHat.append(f.hadamard(kd))
                    let base = (h * ld + P + t) * hd
                    kGPU += (0..<hd).map { Float(kc[base + $0]) * ks[h * ld + P + t] }
                }
                for h in 0..<f.qH {
                    let q = f.normRope(pr[(h * f.qs)..<(h * f.qs + hd)], f.qnw, pos: P + t)
                    sRef.append(zip(q, kHat[h / f.G]).reduce(0) { $0 + $1.0 * $1.1 })
                    let qr = qg[((t * f.qH) + h) * hd..<((t * f.qH) + h + 1) * hd]
                    let kr = kGPU[((t * f.kvH) + h / f.G) * hd..<((t * f.kvH) + h / f.G + 1) * hd]
                    sGPU.append(zip(qr, kr).reduce(0) { $0 + $1.0 * $1.1 })
                }
            }
            return [
                // One int8 step is 1/127 of the head's max; a rounding flip from the fast-math norm stays far below
                // the RMS bound.
                Check(label: "rotated int8 K in the cache", gpu: kGPU, ref: kRef, tol: .rms(2e-3)),
                Check(label: "score (H q)·(H k) against unrotated q·k", gpu: sGPU, ref: sRef, tol: .maxRel(5e-3)),
            ]
        }
    }
}

extension AttnFixture {
    /// Queries after vattn_prework_i8 (half [rows · qH · hd], row-major by row and head) for `rows` rows at P.
    func preworkQ(_ x: [Float], rows: Int, P: Int) throws -> [Float] {
        let xb = Syn.halfBuffer(ctx, x), out = ctx.buffer(length: rows * hidden * 4)
        a.pos = P
        try ctx.run { e in
            try a.encode(e, x: xb, out: out, rows: rows, only: .proj)
            try a.encode(e, x: xb, out: out, rows: rows, only: .prework)
        }
        // a.q: [kvH, G · rows, hd] with m = row · G + h.
        let q = a.q.contents().bindMemory(to: Float16.self, capacity: kvH * G * rows * hd)
        var r = [Float]()
        for t in 0..<rows {
            for h in 0..<qH {
                let base = ((h / G) * G * rows + t * G + h % G) * hd
                r += (0..<hd).map { Float(q[base + $0]) }
            }
        }
        return r
    }

    /// CPU queries (norm + RoPE with the fixture's table + Hadamard, rounded to half) in the layout of `preworkQ`.
    func referenceQ(_ x: [Float], rows: Int, P: Int) -> [Float] {
        var r = [Float]()
        for t in 0..<rows {
            let pr = projRow(x, t)
            for h in 0..<qH { r += Syn.roundHalf(hadamard(normRope(pr[(h * qs)..<(h * qs + hd)], qnw, pos: P + t))) }
        }
        return r
    }
}
