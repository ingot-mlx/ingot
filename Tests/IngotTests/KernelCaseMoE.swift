import Foundation
import Metal
import Testing

@testable import Ingot

// MoE with a shared expert of its own width and hidden a multiple of 128 but not of 512, in the
// round (vnax_moe_gateup, vnax_moe_down_split, vmoe_combine9) and in the prefill (moe_group, moe_gateup_swiglu2,
// moe_down2, moe_combine_acc), against the CPU. The union and top-k are set by hand (the routers have their cases).

/// Shape and activation of a MoE case (the defaults: moe-shared's).
struct MoECaseShape {
    enum Act: String { case silu = "", gelu = "_gelu", relu2 = "_relu2" }
    var D = 2816, I = 128, S = 256, E = 8, K = 2, rounds = [5], T = 40
    var act = Act.silu
}
extension KernelCase {
    static let moeShared = moeSharedCase()

    static func moeSharedCase(gelu: Bool = false) -> KernelCase {
        KernelCase(
            id: "moe-shared" + (gelu ? "-gelu" : ""), model: "qwen3.6-35b-a3b" + (gelu ? "-gelu" : ""),
            catches:
                "shared expert with its own intermediate (sinter = 2·inter: extra tiles, h rows with stride sinter, "
                + "down K) and hidden 2816 (multiple of 128, not of 512) in the round and prefill MoE"
                + (gelu ? ", with GELU-tanh (instance choice and activation of the `_gelu` gate+up kernels)" : "")
        ) { ctx, g in try moeSharedRun(ctx, &g, MoECaseShape(act: gelu ? .gelu : .silu)) }
    }

    /// Nemotron 3.5 Lightning's MoE: relu² without gate in the experts and the ungated shared expert (the
    /// gate buffers are the up ones, as MoELayer binds them), its real widths (hidden 2688, intermediate 1856 = 29·64,
    /// shared 3712) and top-6, in rounds of 1 and 16 rows and a prefill. The router (128 experts, sigmoid, selection
    /// bias, ×2.5) has its own case against mlx-lm (router-mlx-nemotron-e128-k6).
    static let moeNemotron = KernelCase(
        id: "moe-nemotron", model: "nemotron-3.5-lightning",
        catches:
            "relu² without gate in vnax_moe_gateup_relu2 and moe_gateup_swiglu2_relu2 (up only, max(u, 0)²) for the "
            + "experts and the shared expert, with Nemotron's widths (inter 1856 not a multiple of 128, sinter 2·inter, "
            + "hidden 2688) and top-6 in the round (1 and 16 rows) and the prefill"
    ) { ctx, g in
        try moeSharedRun(
            ctx, &g, MoECaseShape(D: 2688, I: 1856, S: 3712, E: 8, K: 6, rounds: [1, 16], T: 40, act: .relu2))
    }

    static func moeSharedRun(_ ctx: MetalContext, _ g: inout SplitMix64, _ shape: MoECaseShape) throws -> [Check] {
        let gated = shape.act != .relu2
        let D = shape.D, I = shape.I, S = shape.S, E = shape.E, K = shape.K
        // Without gate (relu²) the gate matrices are the up ones, as MoELayer binds them.
        let eg = Syn.q4(ctx, N: E * I, K: D, &g), eu = gated ? Syn.q4(ctx, N: E * I, K: D, &g) : eg
        let ed = Syn.q4(ctx, N: E * D, K: I, &g)
        let sg = Syn.q4(ctx, N: S, K: D, &g), su = gated ? Syn.q4(ctx, N: S, K: D, &g) : sg
        let sd = Syn.q4(ctx, N: D, K: S, &g)
        let q = MoEPipeline.bufs
        var moe = MoEPipeline(
            ctx, D: D, I: I, E: E, K: K, experts: (q(eg.m), q(eu.m), q(ed.m)), shared: (q(sg.m), q(su.m), q(sd.m), S),
            smap: MoELayer.identityMap(ctx))
        moe.act = shape.act.rawValue
        // Each matrix is applied once to all the rows that need it; Qwen3.6's prefill rounds gate and up to half.
        var reference = MoEReference(D: D, I: I, E: E, K: K) { e in
            (Syn.slab(eg.w, e, I * D), Syn.slab(eu.w, e, I * D), Syn.slab(ed.w, e, D * I))
        }
        reference.gated = gated
        reference.act = { ga, ub in
            switch shape.act {
            case .silu: ga / (1 + exp(-ga)) * ub
            case .gelu: CPU.geluTanh(ga) * ub
            case .relu2: max(ub, 0) * max(ub, 0)
            }
        }
        reference.shared = ((sg.w, su.w, sd.w), S)
        func routing(_ rows: Int) -> (inds: [Int], wts: [Float], sw: [Float]) {
            var inds = [Int](), wts = [Float]()
            for r in 0..<rows {
                if K == 2 {
                    let a = (r * 3) % E, b = (a + 1 + r % (E - 1)) % E
                    let w0 = 0.3 + 0.4 * g.uniform()
                    inds += [min(a, b), max(a, b)]
                    wts += [w0, 1 - w0]
                } else {
                    // K distinct experts (partial Fisher-Yates) with weights that sum to 2.5, like Nemotron's scale.
                    var perm = Array(0..<E)
                    for k in 0..<K { perm.swapAt(k, k + Int(g.uniform() * Float(E - k)) % (E - k)) }
                    let w = (0..<K).map { _ in 0.2 + g.uniform() }, s = w.reduce(0, +)
                    inds += perm.prefix(K)
                    wts += w.map { 2.5 * $0 / s }
                }
            }
            return (inds, wts, (0..<rows).map { _ in gated ? 0.2 + 0.6 * g.uniform() : 1 })
        }

        // Rounds of B rows (union with the shared expert last); a prefill of T rows (one 32 tile and a partial one of
        // the shared expert).
        let tag = "top-\(K), inter \(I), sinter \(S), hidden \(D)"
        var checks: [Check] = []
        for B in shape.rounds {
            let xr = Syn.roundHalf(Syn.normal(&g, B * D, sd: 0.5))
            let rr = routing(B)
            checks.append(
                Check(
                    label: "round (B = \(B), \(tag))",
                    gpu: try moe.round(xr, rows: B, inds: rr.inds, wts: rr.wts, sw: rr.sw),
                    ref: reference(xr, rows: B, inds: rr.inds, wts: rr.wts, sw: rr.sw, halfGU: false),
                    tol: .maxRel(5e-3)))
        }
        let T = shape.T
        let xp = Syn.roundHalf(Syn.normal(&g, T * D, sd: 0.5))
        let rp = routing(T)
        checks.append(
            Check(
                label: "prefill (T = \(T), \(tag))",
                gpu: try moe.prefill(xp, rows: T, inds: rp.inds, wts: rp.wts, sw: rp.sw),
                ref: reference(xp, rows: T, inds: rp.inds, wts: rp.wts, sw: rp.sw, halfGU: true), tol: .maxRel(5e-3)))
        return checks
    }
}
