import Foundation
import Metal
import Testing

@testable import MetalLab

// Nemotron-H's Mamba2 in the GDN kernels (LAB_SSM) against mlx-lm's own NemotronHMamba2Mixer in fp32
// (ref/ssm_vectors.json, written by ref/ssm_ref.py): the sequential prefill (gdn_prework with the conv bias,
// gdn_recur_dv4_g2 without the delta term and with D·x, mamba_normgate), a round of 1 row, a round of 16 with nacc 1,
// the snapshot fold of 4 and 16 of its rows, a round of 2 with nacc 4 (a rejection) and the in-place fold of those 2,
// through GDNLayer's own launchers. Inputs and weights come from the same integer hash on both sides. The outputs are
// checked against mlx-lm (y within its own fp32 error, ~3e-4 of float64; o in half) and the states against the float64
// sequential recurrence of the same formulas (mlx-lm's fp32 state is ~1e-3 from it; ssm_ref.py checks both agree).

/// One case of ref/ssm_vectors.json: shape and outputs sampled every `yStride` channels / `sStride` state elements.
struct SSMVector: Sendable, Decodable {
    let name: String
    let H: Int, P: Int, G: Int, N: Int, rows: Int, yStride: Int, sStride: Int
    let prefillRows: [Int], prefillY: [[Float]], prefillO: [[Float]], prefillState: [Float]
    let stepY: [Float], stepO: [Float]
    let stateKeep: [Float], stateRound: [Float]
    let roundRows: [Int], roundY: [[Float]], roundO: [[Float]]
    let newY: [[Float]], newO: [[Float]], stateNew: [Float]

    enum CodingKeys: String, CodingKey {
        case name, H, P, G, N, rows
        case yStride = "y_stride", sStride = "s_stride", prefillRows = "prefill_rows", prefillY = "prefill_y"
        case prefillO = "prefill_o", prefillState = "prefill_state", stepY = "step_y", stepO = "step_o"
        case stateKeep = "state_keep", stateRound = "state_round", roundRows = "round_rows", roundY = "round_y"
        case roundO = "round_o", newY = "new_y", newO = "new_o", stateNew = "state_new"
    }

    private struct File: Decodable { let cases: [SSMVector] }
    static let all: [SSMVector] = {
        let url = ModelSpecTests.repo.appendingPathComponent("ref/ssm_vectors.json")
        guard let d = try? Data(contentsOf: url), let f = try? JSONDecoder().decode(File.self, from: d) else {
            preconditionFailure("ref/ssm_vectors.json: missing or unreadable (ref/ssm_ref.py writes it)")
        }
        return f.cases
    }()

    /// The integer hash of ssm_ref.py (`hval`): murmur3's finalizer of i·0x9E3779B1 + seed.
    static func hval(_ i: Int, _ seed: UInt64) -> UInt32 {
        var h = UInt32(truncatingIfNeeded: UInt64(i) &* 0x9E37_79B1 &+ seed)
        h ^= h >> 15
        h = h &* 0x85EB_CA77
        h ^= h >> 13
        h = h &* 0xC2B2_AE3D
        h ^= h >> 16
        return h
    }
    static func unit(_ n: Int, _ seed: UInt64) -> [Float] {
        (0..<n).map { Float((Double(hval($0, seed) >> 8) - 8_388_608) / 8_388_608) }
    }
    static func byte(_ n: Int, _ seed: UInt64) -> [Int] { (0..<n).map { Int(hval($0, seed) >> 24) } }
}

extension KernelCase {
    static let mamba2: [KernelCase] = [
        mamba2Case(id: "mamba2-64x64", model: "nemotron-3.5-lightning", vector: "mamba2-64x64"),
        mamba2Case(id: "mamba2-96x80", model: "nemotron-3-nano-4b", vector: "mamba2-96x80"),
    ]

    static func mamba2Case(id: String, model: String, vector: String) -> KernelCase {
        KernelCase(
            id: id, model: model,
            catches:
                "Mamba2 in the GDN kernels against mlx-lm fp32: conv with bias and channels x|B|C after z, beta = dt, "
                + "no delta term, D·x, v heads of 64 or 80 (GDN_DV, GDN_R from the groups), the gated group norm "
                + "(mamba_normgate), sequential prefill, rounds with nacc 0, 1 and 4 and the tape folds"
        ) { ctx, _ in
            try mamba2Run(ctx, SSMVector.all.first { $0.name == vector }!)
        }
    }

    static func mamba2Run(_ ctx: MetalContext, _ v: SSMVector) throws -> [Check] {
        let dm = ctx.dims
        precondition(dm.ssm && dm.gdnHV == v.H && dm.gdnDV == v.P && dm.gdnHK == v.G && dm.gdnDK == v.N)
        let H = v.H, VD = dm.gdnVDim, conv = dm.gdnConvDim, W = dm.gdnProjN, Wp = (W + 63) / 64 * 64
        let proj = SSMVector.unit(v.rows * W, 11)
        let convW = SSMVector.byte(conv * 4, 12).map { Float($0 - 128) / 256 }
        let convB = SSMVector.byte(conv, 13).map { Float($0 - 128) / 1024 }
        let dtBias = SSMVector.byte(H, 14).map { -Float(128 + $0 / 2) / 64 }
        let aLog = SSMVector.byte(H, 15).map { Float($0 % 181) / 64 }
        let dSkip = SSMVector.byte(H, 16).map { 1 + Float($0 - 128) / 512 }
        let normW = SSMVector.byte(VD, 17).map { Float($0 / 2 + 64) / 128 }
        let layer = GDNLayer(
            ctx: ctx, proj: Syn.oneHot4(ctx, N: Wp, K: 64, cols: [Int](repeating: 0, count: Wp)),
            outProj: Syn.oneHot4(ctx, N: 64, K: VD, cols: [Int](repeating: 0, count: 64)),
            convW: Syn.bf16Buffer(ctx, convW), convB: Syn.bf16Buffer(ctx, convB), aLog: ctx.buffer(aLog),
            dtBias: Syn.bf16Buffer(ctx, dtBias), dSkip: ctx.buffer(dSkip), normW: Syn.bf16Buffer(ctx, normW), eps: 1e-5)
        /// Rows `r0 ..< r0 + n` of the projected input into `buf` with the projection's padded stride.
        func put(_ buf: MTLBuffer, from r0: Int, rows n: Int) {
            let p = buf.contents().bindMemory(to: Float.self, capacity: n * Wp)
            for r in 0..<n { for c in 0..<W { p[r * Wp + c] = proj[(r0 + r) * W + c] } }
        }
        func rowY(_ b: MTLBuffer, _ r: Int) -> [Float] { Syn.floats(b, VD, offset: r * VD * 4) }
        func rowO(_ b: MTLBuffer, _ r: Int) -> [Float] { readHalf(b, count: VD, offset: r * VD * 2) }
        func every(_ a: [Float], _ s: Int) -> [Float] { stride(from: 0, to: a.count, by: s).map { a[$0] } }
        func state(_ b: MTLBuffer) -> [Float] { every(Syn.floats(b, H * v.P * v.N), v.sStride) }
        var checks: [Check] = []
        let yTol = Tol.maxRel(4e-4), oTol = Tol.maxRel(4e-3), sTol = Tol.maxRel(1e-5)
        func rows(_ label: String, _ b: MTLBuffer, _ rs: [Int], _ want: [[Float]], half: Bool) {
            let got = rs.flatMap { every(half ? rowO(b, $0) : rowY(b, $0), v.yStride) }
            checks.append(
                Check(label: "\(v.name) \(label)", gpu: got, ref: want.flatMap { $0 }, tol: half ? oTol : yTol))
        }

        // 1. Prefill of 300 rows over the zero state (sequential recurrence, everything in fp32).
        let T = 300, M = (T + 63) / 64 * 64, QK = dm.gdnHK * dm.gdnDK
        let s = (
            proj: ctx.buffer(length: M * Wp * 4), hcur: ctx.buffer(length: (3 + M) * conv * 4),
            q: ctx.buffer(length: M * QK * 4), k: ctx.buffer(length: M * QK * 4), v: ctx.buffer(length: M * VD * 4),
            g: ctx.buffer(length: M * H * 4), b: ctx.buffer(length: M * H * 4), y: ctx.buffer(length: M * VD * 4),
            o: ctx.buffer(length: M * VD * 2)
        )
        put(s.proj, from: 0, rows: T)
        try ctx.run { e in try layer.encodePrefill(e, x: nil, out: nil, rows: T, padded: M, s: s) }
        rows("prefill y", s.y, v.prefillRows, v.prefillY, half: false)
        rows("prefill o", s.o, v.prefillRows, v.prefillO, half: true)
        checks.append(
            Check(label: "\(v.name) prefill state", gpu: state(layer.state[layer.cur]), ref: v.prefillState, tol: sTol))

        // 2-4. Rounds over the layer's own buffers: pieces after the projection (written here into projOut).
        let dummy = ctx.buffer(length: 16)
        func round(from r0: Int, rows n: Int, nacc: Int) throws {
            put(layer.projOut, from: r0, rows: n)
            try ctx.run { e in
                for p in [GDNLayer.Piece.prework, .recur, .normgate] {
                    try layer.encode(e, x: dummy, out: dummy, rows: n, nacc: nacc, only: p)
                }
            }
            layer.advance()
        }
        try round(from: T, rows: 1, nacc: 0)
        rows("round of 1 y", layer.y, [0], [v.stepY], half: false)
        rows("round of 1 o", layer.o, [0], [v.stepO], half: true)
        try round(from: T + 1, rows: 16, nacc: 1)
        rows("round of 16 y", layer.y, v.roundRows, v.roundY, half: false)
        rows("round of 16 o", layer.o, v.roundRows, v.roundO, half: true)
        let sKeep = ctx.buffer(length: layer.state[0].length), sAll = ctx.buffer(length: layer.state[0].length)
        let hKeep = ctx.buffer(length: 3 * conv * 4), hAll = ctx.buffer(length: 3 * conv * 4)
        try ctx.run { e in
            try layer.encodeFold(e, nacc: 4, stateOut: sKeep, histOut: hKeep)
            try layer.encodeFold(e, nacc: 16, stateOut: sAll, histOut: hAll)
        }
        checks.append(Check(label: "\(v.name) fold 4 of 16", gpu: state(sKeep), ref: v.stateKeep, tol: sTol))
        checks.append(Check(label: "\(v.name) fold 16 of 16", gpu: state(sAll), ref: v.stateRound, tol: sTol))
        // A rejection: 4 of the 16 accepted, then 2 new rows; then those 2 folded in place.
        try round(from: T + 17, rows: 2, nacc: 4)
        rows("round of 2 after 4 y", layer.y, [0, 1], v.newY, half: false)
        rows("round of 2 after 4 o", layer.o, [0, 1], v.newO, half: true)
        try ctx.run { e in try layer.encodeFold(e, nacc: 2) }
        checks.append(
            Check(label: "\(v.name) in-place fold of 2", gpu: state(layer.state[layer.cur]), ref: v.stateNew, tol: sTol)
        )
        return checks
    }
}
