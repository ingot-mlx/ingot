import Foundation
import Testing

@testable import MetalLab

// Gemma 4's attention against the CPU: the full layers' own shape (head of 512, 4 kv heads, GQA 8,
// proportional RoPE table) and the sliding layers' (head of 256, 16 kv heads, GQA 2, window and ring), both with the
// weightless v_norm (LAB_AT_VN) and the Hadamard of q and k, with the 31B's dimensions.

extension KernelCase {
    /// Proportional RoPE of the full layers (θ 1e6, partial 0.25 over the whole 512 head).
    static func g4ProportionalTable() throws -> [Float] {
        let s = try #require(try RopeScaling.parse(["rope_type": "proportional", "partial_rotary_factor": 0.25]))
        return RopeTable.make(theta: 1_000_000, rotaryDims: 512, scaling: s)
    }

    static let attnG4Full = KernelCase(
        id: "attn-g4-full", model: "gemma4-31b-full",
        catches: "attention with heads of 512 (Gemma 4's full layers: prework with 512 threads, Hadamard of 512, "
            + "pattn/dattn_i8 with 8 P·V fragments per simdgroup, vattn_reduce), the proportional RoPE table and "
            + "the weightless v_norm, in rounds (one and several splits), prefill (also with int8 Q·Kᵀ) and per row"
    ) { ctx, g in
        let tab = try g4ProportionalTable()
        var checks: [Check] = []
        for P in [40, 700, 1500] {
            let f = try AttnFixture(ctx, positions: P, &g, maxCtx: 4096, ropeTable: tab)
            let xq = Syn.roundHalf(Syn.normal(&g, 16 * f.hidden))
            checks.append(
                Check(
                    label: "rotated q of 16 rows at \(P)", gpu: try f.preworkQ(xq, rows: 16, P: P),
                    ref: f.referenceQ(xq, rows: 16, P: P), tol: .maxRel(2e-3)))
            let x = Syn.roundHalf(Syn.normal(&g, 16 * f.hidden))
            checks.append(
                Check(
                    label: "round of 16 rows at \(P)", gpu: try f.round(x, rows: 16, P: P),
                    ref: f.reference(x, rows: 16, P: P), tol: .maxRel(5e-3)))
            let xp = Syn.roundHalf(Syn.normal(&g, 100 * f.hidden))
            checks.append(
                Check(
                    label: "prefill of 100 rows at \(P + 16)", gpu: try f.prefill(xp, rows: 100, P: P + 16),
                    ref: f.reference(xp, rows: 100, P: P + 16), tol: .rms(2e-3)))
        }
        let f = try AttnFixture(ctx, positions: 700, &g, maxCtx: 4096, ropeTable: tab)
        checks.append(try rowInvariance("round", K: f.hidden, &g) { x, r in try f.round(x, rows: r, P: 700) })
        let f8 = try AttnFixture(ctx, positions: 300, &g, ropeTable: tab)
        let x8 = Syn.roundHalf(Syn.normal(&g, 64 * f8.hidden))
        checks.append(
            Check(
                label: "prefill with int8 Q·Kᵀ", gpu: try f8.prefill(x8, rows: 64, P: 300, qk8: true),
                ref: f8.reference(x8, rows: 64, P: 300, qk8: true), tol: .rms(2e-3)))
        return checks
    }

    static let attnG4Sliding = KernelCase(
        id: "attn-g4-swa", model: "gemma4-31b",
        catches: "Gemma 4's sliding layers (head of 256, GQA 2, v_norm, default RoPE from the table) with a window of "
            + "300 over the linear KV and over a ring that wraps, in rounds and prefill"
    ) { ctx, g in
        let tab = RopeTable.make(theta: 10_000, rotaryDims: 256, scaling: nil)
        var checks: [Check] = []
        for P in [40, 700, 1500] {
            let f = try AttnFixture(ctx, positions: P, &g, maxCtx: 4096, window: 300, ropeTable: tab)
            let x = Syn.roundHalf(Syn.normal(&g, 16 * f.hidden))
            // At 40 positions the 16 new keys weigh a lot, and the v_norm (rsqrt on the GPU, fast math) can move a new
            // V across an int8 rounding boundary against the CPU: one level (1/127 of the largest value) shows in the
            // max-relative error (7.4e-3 measured; the prefill of the same case, 3.5e-4 RMS), as with W = 1 in attn-swa.
            checks.append(
                Check(
                    label: "W 300: round of 16 rows at \(P)", gpu: try f.round(x, rows: 16, P: P),
                    ref: f.reference(x, rows: 16, P: P), tol: .maxRel(P < 300 ? 1e-2 : 5e-3)))
            let xp = Syn.roundHalf(Syn.normal(&g, 100 * f.hidden))
            checks.append(
                Check(
                    label: "W 300: prefill of 100 rows at \(P)", gpu: try f.prefill(xp, rows: 100, P: P),
                    ref: f.reference(xp, rows: 100, P: P), tol: .rms(2e-3)))
        }
        // ringRows 128: ring = 300 + 128 + 16 → 512 slots; P = 1500 wraps it.
        let f = try AttnFixture(ctx, positions: 1500, &g, maxCtx: 8192, window: 300, ringRows: 128, ropeTable: tab)
        let x = Syn.roundHalf(Syn.normal(&g, 16 * f.hidden))
        checks.append(
            Check(
                label: "ring \(f.a.ring), W 300: round of 16 rows at 1500", gpu: try f.round(x, rows: 16, P: 1500),
                ref: f.reference(x, rows: 16, P: 1500), tol: .maxRel(5e-3)))
        let xp = Syn.roundHalf(Syn.normal(&g, 128 * f.hidden))
        checks.append(
            Check(
                label: "ring \(f.a.ring), W 300: prefill of 128 rows at 1516",
                gpu: try f.prefill(xp, rows: 128, P: 1516), ref: f.reference(xp, rows: 128, P: 1516),
                tol: .rms(2e-3)))
        return checks
    }
}
