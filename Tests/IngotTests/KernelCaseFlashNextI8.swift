import Foundation
import Metal
import Testing

@testable import Ingot

// Per-kernel tests that were missing, against the CPU: the LFM2.5 experts (MoE without shared expert with a
// sigmoid router: vmoe_route → vnax_moe_gateup/down_split with LFM's split + vmoe_combine9 in the round, and
// moe_route_topk_cfg → moe_group → moe_gateup_swiglu2 → moe_down2 → moe_combine_acc_ns in prefill), the ternary W8A8
// of Bonsai (--int8 1, encodeQmmI8T: quant_rows_i8 + tern_i8row + qmm_i8) and the Flash-Next Q2 experts via the
// prefill int8 path (zi_fn_i8.metal: fn_q8g_perm + fn_moe_gateup_i8 + fn_moe_down_i8).
/// Per-kernel cases in their own suite (filtered with `--filter KernelCaseFlashNextI8Tests`), with the same harness
/// and seed as KernelCaseTests.
@Suite(.serialized) struct KernelCaseFlashNextI8Tests {
    nonisolated(unsafe) static var contexts: [String: MetalContext] = [:]

    @Test(arguments: KernelCase.cq1)
    func matchesReference(_ c: KernelCase) throws {
        try KernelCase.evaluate(c, contexts: &Self.contexts)
    }
}

extension KernelCase {
    static let cq1: [KernelCase] = [
        lfmExperts(id: "lfm-experts-e64", model: "lfm2.5-24b-a2b", E: 64, I: 1536),
        lfmExperts(id: "lfm-experts-e32", model: "lfm2.5-8b-a1b", E: 32, I: 1792),
        ternaryI8, fnExpertsI8,
    ]

    // MARK: LFM2.5: experts

    static func lfmExperts(id: String, model: String, E: Int, I: Int) -> KernelCase {
        KernelCase(
            id: id, model: model,
            catches: "LFM2.5 experts (no shared expert, \(E) top-4 experts of \(I)) with the real sigmoid router "
                + "upstream: round (vmoe_route, vnax_moe_gateup/down_split nf1 sg16, vmoe_combine9 4 slots, with and "
                + "without accumulating) with each row independent, and prefill (moe_route_topk_cfg, moe_group, "
                + "moe_gateup_swiglu2, moe_down2, moe_combine_acc_ns) in a chunk of several tiles"
        ) { ctx, g in try lfmExpertsRun(ctx, &g, E: E, I: I) }
    }

    static func lfmExpertsRun(_ ctx: MetalContext, _ g: inout SplitMix64, E: Int, I: Int) throws -> [Check] {
        let D = 2048, K = 4, nPool = 6
        let poolIds = [2, 11, 19, 25, 30, E - 1]
        // Six distinct experts in slots 0…5 (the expert → slot table), the rest without weights: the router only picks
        // among them (their logits are −30 and their bias, −1).
        let eg = Syn.q4(ctx, N: nPool * I, K: D, &g), eu = Syn.q4(ctx, N: nPool * I, K: D, &g)
        let ed = Syn.q4(ctx, N: nPool * D, K: I, &g)
        var smapArr = [UInt32](repeating: 0, count: 1024)
        for (s, e) in poolIds.enumerated() { smapArr[e] = UInt32(s) }
        var moe = MoEPipeline(
            ctx, D: D, I: I, E: E, K: K,
            experts: (MoEPipeline.bufs(eg.m), MoEPipeline.bufs(eu.m), MoEPipeline.bufs(ed.m)),
            smap: ctx.buffer(smapArr))
        moe.nf = 1
        moe.sgs = 16
        let zero = ctx.buffer(length: D * 2)  // unused input rows and shared gate of the route kernels
        var bias = [Float](repeating: -1, count: E)
        for e in poolIds { bias[e] = 0.6 * g.uniform() - 0.3 }
        let ld = MoELayer.routerLD(E)

        func makeLogits(_ rows: Int) -> [Float] {
            var l = [Float](repeating: -30, count: rows * E)
            for r in 0..<rows { for e in poolIds { l[r * E + e] = 4 * g.uniform() - 2 } }
            return l
        }
        func padded(_ l: [Float], _ rows: Int) -> [Float] {
            var p = [Float](repeating: 0, count: rows * ld)
            for r in 0..<rows { p.replaceSubrange((r * ld)..<(r * ld + E), with: l[(r * E)..<(r * E + E)]) }
            return p
        }
        /// Reference: top-4 by σ(l) + bias (ties to the lower index), weights σ(l) / (Σ + 1e-6) without bias.
        func reference(_ x: [Float], _ logits: [Float], rows: Int, halfGU: Bool) -> [Float] {
            var y = [Float](repeating: 0, count: rows * D)
            for r in 0..<rows {
                let p = (0..<E).map { 1 / (1 + exp(-logits[r * E + $0])) }
                let top = (0..<E).sorted { (p[$0] + bias[$0], -$0) > (p[$1] + bias[$1], -$1) }.prefix(K)
                let sum = top.reduce(Float(0)) { $0 + p[$1] }
                let xr = Array(x[(r * D)..<((r + 1) * D)])
                for e in top {
                    let s = poolIds.firstIndex(of: e)!
                    let gg = CPU.matmulT(xr, Array(eg.w[(s * I * D)..<((s + 1) * I * D)]), M: 1, N: I, K: D)
                    let uu = CPU.matmulT(xr, Array(eu.w[(s * I * D)..<((s + 1) * I * D)]), M: 1, N: I, K: D)
                    let h = zip(gg, uu).map { a, b -> Float in
                        let (ga, ub) = halfGU ? (Float(Float16(a)), Float(Float16(b))) : (a, b)
                        return Float(Float16(ga / (1 + exp(-ga)) * ub))
                    }
                    let o = CPU.matmulT(h, Array(ed.w[(s * D * I)..<((s + 1) * D * I)]), M: 1, N: D, K: I)
                    let w = p[e] / (sum + 1e-6)
                    for c in 0..<D { y[r * D + c] += w * o[c] }
                }
            }
            return y
        }

        // Round: vmoe_route → gate+up → down per expert → combination of 4 slots.
        func decode(_ x: [Float], _ logits: [Float], rows B: Int, base: [Float]?) throws -> [Float] {
            let P = B * K, mp = moe.roundParams(rows: B, accumulate: base != nil)
            let lb = ctx.buffer(padded(logits, B)), bb = ctx.buffer(bias)
            let rb = moe.roundBuffers(
                x: Syn.halfBuffer(ctx, x), ent: ctx.buffer(length: (P + 1) * 16), pairs: ctx.buffer(length: P * 8),
                args: ctx.buffer(length: 64), inds: ctx.buffer(length: P * 4),
                out: ctx.buffer(base ?? [Float](repeating: 0, count: B * D)), rows: B)
            try ctx.run { e in
                try encodeRouteCommon(
                    ctx, e, logits: lb, bias: bb, zero: zero, rowLimit: rb.args,
                    outs: [rb.ent, rb.pairs, rb.args, rb.inds, ctx.buffer(length: P * 4)], params: mp)
                try moe.encodeRound(e, rb, rows: B, params: mp)
            }
            return Syn.floats(rb.out, B * D)
        }

        var checks: [Check] = []
        let B = 16
        let xr = Syn.roundHalf(Syn.normal(&g, B * D, sd: 0.5)), lr = makeLogits(B)
        let refR = reference(xr, lr, rows: B, halfGU: false)
        let full = try decode(xr, lr, rows: B, base: nil)
        checks.append(
            Check(label: "round (B = 16, no shared expert)", gpu: full, ref: refR, tol: .maxRel(5e-3)))
        let base = Syn.normal(&g, B * D)
        checks.append(
            Check(
                label: "round accumulating (out += MoE)", gpu: try decode(xr, lr, rows: B, base: base),
                ref: zip(base, refR).map { $0 + $1 }, tol: .maxRel(5e-3)))
        var alone: [Float] = []
        for r in 0..<B {
            alone += try decode(
                Array(xr[(r * D)..<((r + 1) * D)]), Array(lr[(r * E)..<((r + 1) * E)]), rows: 1, base: nil)
        }
        checks.append(Check(label: "round: row alone against B = 16", gpu: alone, ref: full, tol: .bitExact))

        // Prefill: T rows (several tiles of 32 per expert), the whole block with the combiner without shared expert.
        let T = 70, PT = T * K
        let xp = Syn.roundHalf(Syn.normal(&g, T * D, sd: 0.5)), lp = makeLogits(T)
        let xpb = Syn.halfBuffer(ctx, xp), lb = ctx.buffer(padded(lp, T)), bb = ctx.buffer(bias)
        let ib = ctx.buffer(length: PT * 4), wb = ctx.buffer(length: PT * 4), sgT = ctx.buffer(length: T * 4)
        let baseP = Syn.normal(&g, T * D)
        let outP = ctx.buffer(baseP)
        try ctx.run { e in
            try encodePrefillRouteCommon(
                ctx, e, logits: lb, bias: bb, zero: zero, outs: [ib, wb, sgT],
                params: [UInt32(T), UInt32(E), UInt32(D), UInt32(K)])
            try moe.encodePrefill(e, x: xpb, inds: ib, wts: wb, sgate: nil, out: outP, rows: T, tiles: .init())
        }
        let refP = reference(xp, lp, rows: T, halfGU: true)
        checks.append(
            Check(
                label: "prefill (T = 70, accumulating into the stream)", gpu: Syn.floats(outP, T * D),
                ref: zip(baseP, refP).map { $0 + $1 }, tol: .maxRel(5e-3)))
        return checks
    }

    // MARK: Bonsai: ternary W8A8

    static let ternaryI8 = KernelCase(
        id: "bonsai-int8", model: "bonsai-2-27b",
        catches: "prefill ternary W8A8 (--int8 1 with Bonsai; DenseQ.encodeQmmI8T: rotation, quant_rows_i8, "
            + "tern_i8row with one scale per row and qmm_i8) against its CPU emulation, bounded error against the "
            + "exact one, half and accumulated, and a 64-row chunk giving the same bits alone or in a 128-row one"
    ) { ctx, g in
        var checks: [Check] = []
        // (N, K): 2×2 tile (K = 2048) and 2×4 with K > 8192 (9216 = 9 Hadamard blocks of 1024).
        for (N, K) in [(128, 2048), (128, 9216)] {
            let M = 128
            let (m, w, sg) = Syn.ternary(ctx, N: N, K: K, &g)
            let x = Syn.roundHalf(Syn.normal(&g, M * K))
            let xr = CPU.rotate(x, rows: M, K: K, signs: sg)
            // Emulation: x rotated (half) to int8 per row; W = t·s_g with the group scale in units of the row's.
            var xq = xr, sx = [Float](repeating: 0, count: M)
            for r in 0..<M {
                let (q, s) = CPU.quantI8(xr[(r * K)..<(r * K + K)])
                xq.replaceSubrange((r * K)..<(r * K + K), with: q)
                sx[r] = s
            }
            var wq = w, sw = [Float](repeating: 0, count: N)
            for n in 0..<N {
                let gs = (0..<(K / 128)).map { gi in (0..<128).map { abs(w[n * K + gi * 128 + $0]) }.max()! }
                let srow = gs.max()! / 127
                sw[n] = srow
                for k in 0..<K {
                    wq[n * K + k] =
                        (w[n * K + k] == 0 ? 0 : (w[n * K + k] < 0 ? -1 : 1))
                        * (gs[k / 128] / srow).rounded(.toNearestOrEven)
                }
            }
            var emu = CPU.matmulT(xq, wq, M: M, N: N, K: K)
            for r in 0..<M { for j in 0..<N { emu[r * N + j] *= sx[r] * sw[j] } }
            let exact = CPU.matmulT(xr, w, M: M, N: N, K: K)
            let sc = DenseI8Scratch(ctx: ctx, maxNK: N * K, maxN: N, maxRows: M, maxK: K)
            func run(_ rows: Int, from r0: Int, acc: [Float]? = nil, half: Bool = false) throws -> [Float] {
                let xs = Syn.halfBuffer(ctx, Array(x[(r0 * K)..<((r0 + rows) * K)]))
                let y = ctx.buffer(acc ?? [Float](repeating: 0, count: rows * N))
                let yh = ctx.buffer(length: rows * N * 2)
                try ctx.run { e in
                    try m.encodeQmmI8T(
                        ctx, e, x: xs, y: half ? yh : y, rows: rows, scratch: sc, accumulate: acc != nil, halfOut: half)
                }
                return half ? readHalf(yh, count: rows * N) : Syn.floats(y, rows * N)
            }
            let tag = "\(M)×\(N)×\(K)"
            let y = try run(M, from: 0)
            checks.append(Check(label: "\(tag) against the emulated W8A8", gpu: y, ref: emu, tol: .rms(2e-3)))
            checks.append(Check(label: "\(tag) against the exact one", gpu: y, ref: exact, tol: .rms(0.03)))
            checks.append(
                Check(
                    label: "\(tag) half output", gpu: try run(M, from: 0, half: true),
                    ref: y.map { Float(Float16($0)) },
                    tol: .maxRel(2e-3)))
            let base = Syn.normal(&g, M * N)
            checks.append(
                Check(
                    label: "\(tag) accum.", gpu: try run(M, from: 0, acc: base), ref: zip(base, y).map { $0 + $1 },
                    tol: .maxRel(1e-4)))
            checks.append(
                Check(
                    label: "\(tag) 64-row chunk, first and second, against 128 rows",
                    gpu: try run(64, from: 0) + run(64, from: 64), ref: y, tol: .bitExact))
        }
        return checks
    }

    // MARK: Flash-Next: Q2 experts in int8

    static let fnExpertsI8 = KernelCase(
        id: "fn-moe-i8", model: "qwen3.8-flash-next",
        catches: "Flash-Next Q2 experts via the prefill int8 path (fn_q8g_perm, fn_moe_gateup_i8 with h already "
            + "quantized, fn_moe_down_i8; tiles of 16, 32, 48 and 64 rows with 1 to 4 live fragments and partial tile) "
            + "against its CPU emulation (x and h to int8 per group of 64) and the exact one"
    ) { ctx, g in
        let f = try fnFormat()
        let D = 2560, I = 640, K = 10, nExp = 12, slots = 16
        let slotOf = (0..<nExp).map { (($0 * 5) + 3) % slots }
        let (arena, wG, wU, wD) = fnExperts(f, slots: slots, slotOf: slotOf, D: D, I: I, &g)
        let expertId = (0..<nExp).map { ($0 * 37 + 11) % 512 }
        var smapArr = [UInt32](repeating: 0, count: 512)
        for (i, e) in expertId.enumerated() { smapArr[e] = UInt32(slotOf[i]) }
        let ab = ctx.buffer(arena), mb = ctx.buffer(smapArr)
        func quantGroups(_ v: [Float]) -> [Float] {
            var out = v
            for b0 in stride(from: 0, to: v.count, by: 64) {
                let (q, s) = CPU.quantI8(v[b0..<(b0 + 64)])
                for i in 0..<64 { out[b0 + i] = q[i] * s }
            }
            return out
        }
        var checks: [Check] = []
        for (T, bm) in [(7, 16), (40, 32), (100, 48), (100, 64)] {
            let P = T * K
            let x = Syn.roundHalf(Syn.normal(&g, T * D, sd: 0.5))
            // Each row picks 10 of the 12 experts with slots (without repeating) and row 0, always the same ones.
            var inds = [UInt32](), pairExpert = [Int]()
            for r in 0..<T {
                var order = Array(0..<nExp)
                for i in stride(from: nExp - 1, to: 0, by: -1) { order.swapAt(i, Int(g.next() % UInt64(i + 1))) }
                let pick = r == 0 ? Array(0..<K) : Array(order.prefix(K))
                for i in pick {
                    inds.append(UInt32(expertId[i]))
                    pairExpert.append(i)
                }
            }
            // Per-pair reference: x quantized in groups of 64, h quantized in groups of 64 of I.
            var refQ = [Float](), refX = [Float]()
            for p in 0..<P {
                let i = pairExpert[p], xr = Array(x[((p / K) * D)..<((p / K) * D + D)])
                let xd = quantGroups(xr)
                for (xin, into) in [(xd, 0), (xr, 1)] {
                    let gg = CPU.matmulT(xin, wG[i], M: 1, N: I, K: D), uu = CPU.matmulT(xin, wU[i], M: 1, N: I, K: D)
                    var h = (0..<I).map { gg[$0] / (1 + exp(-gg[$0])) * uu[$0] }
                    if into == 0 { h = quantGroups(h) }
                    let y = CPU.matmulT(h, wD[i], M: 1, N: D, K: I)
                    if into == 0 { refQ += y } else { refX += y }
                }
            }
            let xb = Syn.halfBuffer(ctx, x), ib = ctx.buffer(inds)
            let offs = ctx.buffer(length: (512 + 1) * 4), srt = ctx.buffer(length: P * 4),
                pos = ctx.buffer(length: P * 4)
            let tl = ctx.buffer(length: (P + 512) * 16), args = ctx.buffer(length: 32)
            let xq = ctx.buffer(length: T * D), xs = ctx.buffer(length: T * D / 16)
            let hq = ctx.buffer(length: P * I), hs = ctx.buffer(length: P * I / 16), yb = ctx.buffer(length: P * D * 4)
            var gp = (
                UInt32(P), UInt32(512), UInt32(bm), UInt32(bm), UInt32(I / 64), UInt32(D / 128), UInt32(0), UInt32(0)
            )
            var qp = (
                UInt32(f.recordBytes), UInt32(f.gateCodes), UInt32(f.gateScales), UInt32(f.upCodes), UInt32(f.upScales),
                UInt32(f.downCodes), UInt32(f.downScales), UInt32(T)
            )
            var nx = UInt32(T * D / 16)
            try ctx.run { e in
                e.setComputePipelineState(try ctx.pipeline("moe_group"))
                for (i, b) in [ib, offs, srt, pos, tl, tl].enumerated() { e.setBuffer(b, offset: 0, index: i) }
                e.setBuffer(args, offset: 0, index: 6)
                e.setBuffer(args, offset: 16, index: 7)
                e.setBytes(&gp, length: 32, index: 8)
                e.setBuffer(ib, offset: 0, index: 9)
                e.dispatchThreadgroups(
                    MTLSize(width: 1, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
                e.setComputePipelineState(try ctx.pipeline("fn_q8g_perm"))
                for (i, b) in [xb, xq, xs].enumerated() { e.setBuffer(b, offset: 0, index: i) }
                e.setBytes(&nx, length: 4, index: 3)
                e.dispatchThreads(
                    MTLSize(width: T * D / 16, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                let tpg = MTLSize(width: 128, height: 1, depth: 1)
                e.setComputePipelineState(try ctx.pipeline("fn_moe_gateup_i8_bm\(bm)_ng1"))
                for (k, b) in [xq, xs, ab, mb, tl, srt, hq, hs].enumerated() { e.setBuffer(b, offset: 0, index: k) }
                e.setBytes(&qp, length: 32, index: 8)
                e.dispatchThreadgroups(indirectBuffer: args, indirectBufferOffset: 0, threadsPerThreadgroup: tpg)
                e.setComputePipelineState(try ctx.pipeline("fn_moe_down_i8_bm\(bm)_ng1"))
                for (k, b) in [hq, hs, ab, mb, tl, yb].enumerated() { e.setBuffer(b, offset: 0, index: k) }
                e.setBytes(&qp, length: 32, index: 6)
                e.dispatchThreadgroups(indirectBuffer: args, indirectBufferOffset: 16, threadsPerThreadgroup: tpg)
            }
            let y = Syn.floats(yb, P * D), pp = Syn.uints(pos, P)
            var got = [Float](repeating: 0, count: P * D)
            for p in 0..<P {
                got.replaceSubrange((p * D)..<((p + 1) * D), with: y[(Int(pp[p]) * D)..<(Int(pp[p]) * D + D)])
            }
            checks.append(
                Check(label: "T = \(T), tile \(bm): against the emulated int8", gpu: got, ref: refQ, tol: .rms(3e-3)))
            checks.append(
                Check(label: "T = \(T), tile \(bm): against the exact one", gpu: got, ref: refX, tol: .rms(0.06)))
        }
        return checks
    }
}
