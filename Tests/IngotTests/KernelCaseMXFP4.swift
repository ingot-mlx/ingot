import Foundation
import Metal
import Testing

@testable import Ingot

// MXFP4 experts against the CPU: the `_mxfp4` instances of the round (vnax_moe_gateup_oai_mxfp4,
// vnax_moe_down_split_mxfp4, vmoe_combine9) and of the prefill (moe_group, moe_gateup_swiglu2_oai_mxfp4 and
// moe_down2_mxfp4 with the two tile shapes of ModelPrefill.configurePrefillMoE, its fp32 Yd and
// moe_combine_acc_ns_f32), with gpt-oss's
// expert biases and swiglu_oai, E = 32 top-4, in the toy's shape (hidden 576, intermediate 192) and the real one (2880 ×
// 2880). The CPU dequantization they are compared against is anchored to mlx's by MXFP4ReferenceTests.

/// Synthetic MXFP4 matrices: `slots` stacked [rows, K] of uniform e2m1 codes, with one e8m0 exponent per group of 32:
/// row n's is `bases[n % bases.count]` ± 1 at random, clamped to gpt-oss-20b's range (2⁻¹¹ … 2⁸).
struct SynMXFP4 {
    let rows: Int, K: Int, slots: Int
    let words: [UInt32], scales: [UInt8]
    let w: MTLBuffer, s: MTLBuffer
    /// Exponent around which row n's scales are drawn.
    let rowBase: [Int]

    init(_ ctx: MetalContext, slots: Int, rows: Int, K: Int, bases: [Int], _ g: inout SplitMix64) {
        self.rows = rows; self.K = K; self.slots = slots
        rowBase = (0..<rows).map { bases[$0 % bases.count] }
        let words = (0..<(slots * rows * K / 8)).map { _ in UInt32(truncatingIfNeeded: g.next()) }
        let base = rowBase
        let scales = (0..<(slots * rows * K / 32)).map { i -> UInt8 in
            let b = base[(i / (K / 32)) % rows] + Int(g.next() % 3) - 1
            return UInt8(127 + min(max(b, -11), 8))
        }
        self.words = words; self.scales = scales
        w = ctx.buffer(words); s = ctx.buffer(scales)
    }

    /// Slot `slot` dequantized [rows, K] on the CPU (fp4E2M1ToFloat and e8m0ToFloat, those of dequantizeRows).
    func dense(slot: Int) -> [Float] {
        let lut = (0..<16).map { fp4E2M1ToFloat(UInt32($0)) }
        var out = [Float](repeating: 0, count: rows * K)
        let w0 = slot * rows * K / 8, s0 = slot * rows * K / 32
        for i in 0..<(rows * K) {
            out[i] = e8m0ToFloat(scales[s0 + i / 32]) * lut[Int((words[w0 + i / 8] >> UInt32(4 * (i % 8))) & 15)]
        }
        return out
    }
}

extension KernelCase {
    static let moeMXFP4 = KernelCase(
        id: "moe-mxfp4", model: "gpt-oss-20b",
        catches:
            "MXFP4 experts (e2m1 codes, e8m0 scales per group of 32 over 2⁻¹¹ … 2⁸) in the round and prefill MoE with "
            + "gpt-oss's expert biases and swiglu_oai, E = 32 top-4, hidden and intermediate 576 × 192 and 2880 × 2880"
    ) { ctx, g in
        try moeMXFP4Run(ctx, &g, hidden: 576, inter: 192) + moeMXFP4Run(ctx, &g, hidden: 2880, inter: 2880)
    }

    static func moeMXFP4Run(_ ctx: MetalContext, _ g: inout SplitMix64, hidden D: Int, inter I: Int) throws -> [Check] {
        // E = 32 experts top-4; the routing only uses `used`, whose weights live in as many slots (smap: expert → slot),
        // so the 2880 × 2880 case does not hold 32 experts in memory twice (GPU and the CPU reference).
        let E = 32, K = 4, used = [0, 7, 13, 21, 26, 31]
        var slotOf = [UInt32](repeating: 0, count: E)
        for (i, e) in used.enumerated() { slotOf[e] = UInt32(i) }
        // Exponents: every row runs through the whole range (with x of sd 0.05 the gate and up outputs stay well inside
        // half, which the prefill rounds them to). With h near 56 where gate and up saturate swiglu_oai's clip, the
        // down rows at 2⁸ give outputs past half's 65504: the prefill's Yd is fp32 with MXFP4.
        let eg = SynMXFP4(ctx, slots: used.count, rows: I, K: D, bases: Array(-11...8), &g)
        let eu = SynMXFP4(
            ctx, slots: used.count, rows: I, K: D, bases: Array(stride(from: 8, through: -11, by: -1)), &g)
        let ed = SynMXFP4(ctx, slots: used.count, rows: D, K: I, bases: Array(-11...8), &g)
        let wg = (0..<used.count).map { eg.dense(slot: $0) }, wu = (0..<used.count).map { eu.dense(slot: $0) }
        let wd = (0..<used.count).map { ed.dense(slot: $0) }
        // Biases at the scale of their row, so they do not drown the small rows (the clip itself has moe-gptoss).
        func bias(_ m: SynMXFP4, _ gain: Float) -> [Float] {
            let n = Syn.normal(&g, E * m.rows, sd: gain)
            return n.indices.map { n[$0] * Float(sign: .plus, exponent: m.rowBase[$0 % m.rows], significand: 1) }
        }
        let bg = bias(eg, 4), bu = bias(eu, 4), bd = bias(ed, 2)
        var moe = MoEPipeline(
            ctx, D: D, I: I, E: E, K: K, experts: ([eg.w, eg.s, eg.s], [eu.w, eu.s, eu.s], [ed.w, ed.s, ed.s]),
            smap: ctx.buffer(slotOf))
        moe.act = "_oai"
        moe.mxfp4 = true
        moe.bias = [ctx.buffer(bg), ctx.buffer(bu), ctx.buffer(bd)]
        var reference = MoEReference(D: D, I: I, E: E, K: K) { e in
            let si = used.firstIndex(of: e)!
            return (wg[si], wu[si], wd[si])
        }
        reference.act = { CPU.swigluOAI($0, $1) }
        reference.bias = (bg, bu, bd)
        func routing(_ rows: Int) -> (inds: [Int], wts: [Float]) {
            var inds = [Int](), wts = [Float]()
            for r in 0..<rows {
                // K distinct experts of `used` per row, in increasing order, and weights that sum to 1.
                inds += (0..<K).map { used[(r + $0 + r / used.count) % used.count] }.sorted()
                let w = (0..<K).map { _ in 0.2 + g.uniform() }, s = w.reduce(0, +)
                wts += w.map { $0 / s }
            }
            return (inds, wts)
        }
        /// The overall check and one per magnitude class of the down rows (each output column has its row's scale),
        /// so the small scales are judged on their own and not against the largest.
        func checks(_ label: String, _ gpu: [Float], _ ref: [Float], rows: Int) -> [Check] {
            var out = [Check(label: "\(label), hidden \(D)", gpu: gpu, ref: ref, tol: .maxRel(5e-3))]
            for (lo, hi) in [(-11, -7), (-6, -2), (-1, 3), (4, 8)] {
                let cols = (0..<D).filter { (lo...hi).contains(ed.rowBase[$0]) }
                let idx = (0..<rows).flatMap { r in cols.map { r * D + $0 } }
                out.append(
                    Check(
                        label: "\(label), hidden \(D), down rows 2^\(lo)…2^\(hi)", gpu: idx.map { gpu[$0] },
                        ref: idx.map { ref[$0] }, tol: .maxRel(5e-3)))
            }
            return out
        }

        // Round: B rows, no shared expert.
        let B = 5
        let xr = Syn.roundHalf(Syn.normal(&g, B * D, sd: 0.05))
        let rr = routing(B)
        let refR = reference(xr, rows: B, inds: rr.inds, wts: rr.wts, halfGU: false)
        var result = checks(
            "round (B = 5)", try moe.round(xr, rows: B, inds: rr.inds, wts: rr.wts), refR, rows: B)

        // Prefill: T rows, top-k set by hand, with each tile shape the model uses (ModelPrefill.configurePrefillMoE).
        let T = 40
        let xp = Syn.roundHalf(Syn.normal(&g, T * D, sd: 0.05))
        let rp = routing(T)
        let refP = reference(xp, rows: T, inds: rp.inds, wts: rp.wts, halfGU: true)
        for tiles in [PrefillMoE.Tiles(), PrefillMoE.Tiles(bmA: 48, bmB: 48, bkA: 32, bkB: 32)] {
            let got = try moe.prefill(xp, rows: T, inds: rp.inds, wts: rp.wts, tiles: tiles)
            result += checks("prefill (T = 40, BM \(tiles.bmA))", got, refP, rows: T)
        }
        return result
    }
}

/// The CPU MXFP4 dequantization (SafetensorsStore.dequantizeRows with fp4E2M1ToFloat and e8m0ToFloat) against
/// mx.dequantize(..., mode="mxfp4") in fp32 (ref/mxfp4_ref.py): every e2m1 pattern, both signs and -0, scales from 2⁻¹¹
/// to 2⁸. The packing is that of the affine 4 bits (mlx's mx.quantize packs those values back to the same words).
/// No GPU.
@Suite struct MXFP4ReferenceTests {
    static let words: [UInt32] = [
        0x1A3C_5E70, 0x92B4_D6F8, 0x4D6F_81A3, 0xC5E7_092B, 0x7092_B4D6, 0xF81A_3C5E, 0xA3C5_E709, 0x2B4D_6F81,
        0xD6F8_1A3C, 0x5E70_92B4, 0x092B_4D6F, 0x81A3_C5E7, 0x3C5E_7092, 0xB4D6_F81A, 0x6F81_A3C5, 0xE709_2B4D,
    ]
    static let scales: [UInt8] = [116, 135, 124, 129]
    static let values: [Float] = [
        0, 0.0029296875, -0.001953125, 0.00146484375, -0.0009765625, 0.000732421875, -0.00048828125, 0.000244140625,
        -0.0, -0.0029296875, 0.001953125, -0.00146484375, 0.0009765625, -0.000732421875, 0.00048828125,
        -0.000244140625, 0.000732421875, -0.00048828125, 0.000244140625, -0.0, -0.0029296875, 0.001953125,
        -0.00146484375, 0.0009765625, -0.000732421875, 0.00048828125, -0.000244140625, 0, 0.0029296875,
        -0.001953125, 0.00146484375, -0.0009765625, 1024, -768, 512, -384, 256, -128, 0, 1536, -1024, 768, -512,
        384, -256, 128, -0.0, -1536, -128, 0, 1536, -1024, 768, -512, 384, -256, 128, -0.0, -1536, 1024, -768, 512,
        -384, 256, -0.25, 0.1875, -0.125, 0.0625, -0.0, -0.75, 0.5, -0.375, 0.25, -0.1875, 0.125, -0.0625, 0, 0.75,
        -0.5, 0.375, -0.75, 0.5, -0.375, 0.25, -0.1875, 0.125, -0.0625, 0, 0.75, -0.5, 0.375, -0.25, 0.1875, -0.125,
        0.0625, -0.0, 4, -2, 0, 24, -16, 12, -8, 6, -4, 2, -0.0, -24, 16, -12, 8, -6, 12, -8, 6, -4, 2, -0.0, -24, 16,
        -12, 8, -6, 4, -2, 0, 24, -16,
    ]

    @Test func cpuDequantizeMatchesMLX() throws {
        let fmt = FormatTests()
        let dir = try fmt.tmpDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let e = "model.layers.0.mlp.experts.down_proj"
        let wd = Self.words.withUnsafeBufferPointer { Data(buffer: $0) }
        try fmt.writeSafetensors(
            dir,
            tensors: [(e + ".weight", "U32", [1, 2, 8], wd), (e + ".scales", "U8", [1, 2, 2], Data(Self.scales))])
        let st = try SafetensorsStore(directory: dir)
        let cfg = try ModelSpecTests.gptOss(quantization: ModelSpecTests.mxfp4Q8)
        let l = try QuantizedLinear(store: st, path: e, config: cfg, mxfp4: true)
        let got = st.dequantizeRows(l, rows: 0..<2)
        #expect(got.count == Self.values.count)
        // Bit for bit, the sign of zero included.
        #expect(zip(got, Self.values).allSatisfy { $0.bitPattern == $1.bitPattern }, "\(got)")
        #expect(
            (0..<16).map { fp4E2M1ToFloat(UInt32($0)) } == [
                0, 0.5, 1, 1.5, 2, 3, 4, 6, -0, -0.5, -1, -1.5, -2, -3, -4, -6,
            ])
        #expect(e8m0ToFloat(116) == 0x1p-11 && e8m0ToFloat(135) == 256 && e8m0ToFloat(255).isNaN)
    }
}
