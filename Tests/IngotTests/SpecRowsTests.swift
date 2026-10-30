import Foundation
import Metal
import Testing

@testable import Ingot

/// spec_rows_choose (rows chosen on the GPU) against its CPU reference, and the [kmin, kmax] limits.
@Suite struct SpecRowsTests {
    @Test func gpuMatchesCPU() throws {
        let ctx = try MetalContext()
        var g = SplitMix64(seed: 11)
        var cfg = try #require(try SpecRowsConfig.bundled(family: "qwen3_5_moe"))
        let n = 2000
        var same = 0, near = 0, hist = [Int](repeating: 0, count: 9)
        let st = ctx.buffer(length: 32), conf = ctx.buffer(length: 64)
        for i in 0..<n {
            cfg.kmin = i % 5 == 0 ? 3 : 2
            // Typical confidences: mostly high, some low (like those of dselect with τ = 1.5: ≥ 1/16).
            let c = (0..<7).map { _ -> Float in
                let u = g.uniform(); return u < 0.3 ? 0.0625 + 0.5 * g.uniform() : 0.6 + 0.4 * g.uniform()
            }
            memcpy(conf.contents(), c, 28)
            let ctxTok = [100, 700, 3000, 9000, 20000, 40000][i % 6]
            var w = cfg.words(ctx: ctxTok)
            try ctx.run { e in
                e.setComputePipelineState(try ctx.pipeline("spec_rows_choose"))
                e.setBuffer(st, offset: 0, index: 0)
                e.setBuffer(conf, offset: 0, index: 1)
                e.setBytes(&w, length: w.count * 4, index: 2)
                e.dispatchThreads(
                    MTLSize(width: 1, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
            }
            let k = Int(st.contents().load(fromByteOffset: 24, as: UInt32.self))
            let ref = cfg.chooseCPU(conf: c, ctx: ctxTok)
            #expect(k >= cfg.kmin && k <= cfg.kmax)
            if k == ref { same += 1 } else if abs(k - ref) == 1 { near += 1 }
            hist[k] += 1
        }
        // GPU and CPU exp/log may differ in the last bit: an occasional tie broken the other way is allowed.
        #expect(same >= n * 995 / 1000, "equal \(same) of \(n), one row apart \(near)")
        #expect(hist[2...8].filter { $0 > 0 }.count >= 4, "\(hist)")
    }

    /// dpick_top1 (DFlash v1): the draft is candidate 0 of the sorted top-16 and its confidence is softmax(v/τ) of it.
    @Test func v1PickConfidence() throws {
        let ctx = try MetalContext()
        var g = SplitMix64(seed: 5)
        let m = 7, tk = 16
        var cand = [UInt32](repeating: 0, count: m * tk), val = [Float](repeating: 0, count: m * tk)
        for t in 0..<m {
            // Descending values with gaps from ~0 (flat) to several units (peaked), as after the softcap.
            var v = 10 + 20 * g.uniform()
            for j in 0..<tk {
                cand[t * tk + j] = UInt32(1000 * t + j)
                val[t * tk + j] = v
                v -= t == 0 ? 0 : g.uniform() * Float(t)
            }
        }
        let cb = ctx.buffer(length: cand.count * 4), vb = ctx.buffer(length: val.count * 4)
        let tok = ctx.buffer(length: (m + 1) * 4), conf = ctx.buffer(length: m * 4)
        memcpy(cb.contents(), cand, cand.count * 4)
        memcpy(vb.contents(), val, val.count * 4)
        var mm = UInt32(m), tau = DFlash2Drafter.confTau
        try ctx.run { e in
            e.setComputePipelineState(try ctx.pipeline("dpick_top1"))
            e.setBuffer(cb, offset: 0, index: 0)
            e.setBuffer(tok, offset: 0, index: 1)
            e.setBytes(&mm, length: 4, index: 2)
            e.setBuffer(vb, offset: 0, index: 3)
            e.setBuffer(conf, offset: 0, index: 4)
            e.setBytes(&tau, length: 4, index: 5)
            e.dispatchThreads(
                MTLSize(width: m, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: m, height: 1, depth: 1))
        }
        let tp = tok.contents().bindMemory(to: UInt32.self, capacity: m + 1)
        let cp = conf.contents().bindMemory(to: Float.self, capacity: m)
        for t in 0..<m {
            #expect(tp[1 + t] == cand[t * tk])
            let z = (0..<tk).reduce(Float(0)) { $0 + exp((val[t * tk + $1] - val[t * tk]) / tau) }
            #expect(abs(cp[t] - 1 / z) < 1e-5, "row \(t): \(cp[t]) vs \(1 / z)")
        }
        #expect(abs(cp[0] - 1 / Float(tk)) < 1e-6)  // flat row: 1/16
    }

}
