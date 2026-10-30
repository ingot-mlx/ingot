import Foundation
import Metal
import Testing

@testable import MetalLab

// gpt-oss's MoE pieces against the CPU: expert biases (gate and up before the activation, down before the
// weight), swiglu_oai with limit 7 and a hidden that is a multiple of 64 but not of 128 (the shifted last tile of
// moe_down2, LAB_MOE_H64), in the round (vnax_moe_gateup_oai, vnax_moe_down_split, vmoe_combine9) and in the prefill
// (moe_group, moe_gateup_swiglu2_oai, moe_down2, moe_combine_acc_ns), with hidden 576 (toy-gptoss) and 2880 (the 20b).
// No shared expert, like gpt-oss. The union and top-k are set by hand (the router bias has its router-mlx case).
extension KernelCase {
    static let moeGptOss = KernelCase(
        id: "moe-gptoss", model: "gpt-oss-20b",
        catches:
            "expert biases (gate and up before swiglu_oai, down before the weight), swiglu_oai with its clip at ±7 and "
            + "hidden 576 and 2880 (multiples of 64, not of 128: last moe_down2 tile shifted back) in the round and "
            + "prefill MoE, without shared expert"
    ) { ctx, g in
        try [576, 2880].flatMap { try moeGptOssRun(ctx, &g, hidden: $0) }
    }

    static func moeGptOssRun(_ ctx: MetalContext, _ g: inout SplitMix64, hidden D: Int) throws -> [Check] {
        let I = 192, E = 8, K = 4
        let eg = Syn.q4(ctx, N: E * I, K: D, &g), eu = Syn.q4(ctx, N: E * I, K: D, &g)
        let ed = Syn.q4(ctx, N: E * D, K: I, &g)
        // Biases: gate and up of sd 3 so swiglu_oai's clip at 7 bites (gate from above, up at ±7); down of sd 0.5.
        let bg = Syn.normal(&g, E * I, sd: 3), bu = Syn.normal(&g, E * I, sd: 3), bd = Syn.normal(&g, E * D, sd: 0.5)
        var moe = MoEPipeline(
            ctx, D: D, I: I, E: E, K: K,
            experts: (MoEPipeline.bufs(eg.m), MoEPipeline.bufs(eu.m), MoEPipeline.bufs(ed.m)),
            smap: MoELayer.identityMap(ctx))
        moe.act = "_oai"
        moe.bias = [ctx.buffer(bg), ctx.buffer(bu), ctx.buffer(bd)]
        var reference = MoEReference(D: D, I: I, E: E, K: K) { e in
            (Syn.slab(eg.w, e, I * D), Syn.slab(eu.w, e, I * D), Syn.slab(ed.w, e, D * I))
        }
        reference.act = { CPU.swigluOAI($0, $1) }
        reference.bias = (bg, bu, bd)
        func routing(_ rows: Int) -> (inds: [Int], wts: [Float]) {
            var inds = [Int](), wts = [Float]()
            for r in 0..<rows {
                // K distinct experts per row, in increasing order, and weights that sum to 1.
                let start = (r * 3) % E
                inds += (0..<K).map { (start + 2 * $0 + r % 2) % E }.sorted()
                let w = (0..<K).map { _ in 0.2 + g.uniform() }, s = w.reduce(0, +)
                wts += w.map { $0 / s }
            }
            return (inds, wts)
        }

        // Round: B rows, no shared expert. Prefill: T rows (one full 32-row tile per busy expert and partial ones).
        let B = 5, T = 40
        let xr = Syn.roundHalf(Syn.normal(&g, B * D, sd: 0.5)), rr = routing(B)
        let xp = Syn.roundHalf(Syn.normal(&g, T * D, sd: 0.5)), rp = routing(T)
        return [
            Check(
                label: "round (B = 5, hidden \(D))", gpu: try moe.round(xr, rows: B, inds: rr.inds, wts: rr.wts),
                ref: reference(xr, rows: B, inds: rr.inds, wts: rr.wts, halfGU: false), tol: .maxRel(5e-3)),
            Check(
                label: "prefill (T = 40, hidden \(D))", gpu: try moe.prefill(xp, rows: T, inds: rp.inds, wts: rp.wts),
                ref: reference(xp, rows: T, inds: rp.inds, wts: rp.wts, halfGU: true), tol: .maxRel(5e-3)),
        ]
    }
}

/// The CPU swiglu_oai of the kernel cases against mlx-lm's `swiglu_oai` in fp32 (ref/act_ref.py). No GPU.
@Suite struct SwigluOAIReferenceTests {
    static let g: [Float] = [-9, -7.5, -3, -1, -0.2, 0, 0.3, 1.5, 4, 6.9, 7, 7.4, 12]
    static let u: [Float] = [-11, -7, -2.5, -1, 0, 0.6, 3, 6.99, 8]
    static let y: [Float] = [
        1.20190707e-05, 1.20190707e-05, 3.00476768e-06, -0, -2.00317845e-06, -3.20508548e-06, -8.01271381e-06,
        -1.60053951e-05, -1.60254276e-05, 0.000128659565, 0.000128659565, 3.21648913e-05, -0, -2.14432603e-05,
        -3.43092179e-05, -8.57730411e-05, -0.000171331645, -0.000171546082, 0.108427852, 0.108427852, 0.0271069631, -0,
        -0.0180713087, -0.028914094, -0.0722852349, -0.144389749, -0.14457047, 0.925225317, 0.925225317, 0.231306329,
        -0, -0.15420422, -0.246726751, -0.616816878, -1.23209167, -1.23363376, 0.498854756, 0.498854756, 0.124713689,
        -0, -0.0831424594, -0.133027941, -0.332569838, -0.66430825, -0.665139675, -0, -0, -0, 0, 0, 0, 0, 0, 0,
        -1.12490487, -1.12490487, -0.281226218, 0, 0.187484145, 0.29997465, 0.749936581, 1.49799824, 1.49987316,
        -8.34997272, -8.34997272, -2.08749318, 0, 1.39166212, 2.22665954, 5.56664848, 11.11938, 11.133297, -23.9735107,
        -23.9735107, -5.99337769, 0, 3.9955852, 6.39293623, 15.9823408, 31.9247246, 31.9646816, -41.3996735,
        -41.3996735, -10.3499184, 0, 6.89994526, 11.0399122, 27.599781, 55.1305618, 55.1995621, -41.9997177,
        -41.9997177, -10.4999294, 0, 6.99995327, 11.1999254, 27.9998131, 55.9296265, 55.9996262, -41.9997177,
        -41.9997177, -10.4999294, 0, 6.99995327, 11.1999254, 27.9998131, 55.9296265, 55.9996262, -41.9997177,
        -41.9997177, -10.4999294, 0, 6.99995327, 11.1999254, 27.9998131, 55.9296265, 55.9996262,
    ]

    @Test func cpuSwigluOAIMatchesMLX() {
        #expect(Self.y.count == Self.g.count * Self.u.count)
        for (i, want) in Self.y.enumerated() {
            let a = Self.g[i / Self.u.count], b = Self.u[i % Self.u.count]
            let got = CPU.swigluOAI(a, b)
            #expect(abs(got - want) <= 1e-6 * max(1, abs(want)), "swiglu_oai(\(a), \(b)) = \(got), mlx \(want)")
        }
    }
}
