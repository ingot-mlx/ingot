import Foundation
import Metal
import Testing

@testable import Ingot

/// Per-kernel tests: a single parametrized harness that compares kernels (through the engine's launchers) with a CPU
/// reference on small synthetic tensors. Only for what the end-to-end exactness of `ci.sh --real`
/// (gen --check, reference tokens, reuse, prefixcache and rowscheck with the Heretic and the 27B) does not see:
/// families that are not on the Mac when ci runs (ternary Bonsai, LFM2.5), lossy kernels whose failure does not change
/// the tokens of `ingot-dev gen` (the W8A8 of `ingot serve`) and per-row invariance outside the MoE.
/// Each case states in one line which failure it catches (`catches`). It runs with the GPU tests (scripts/ci.sh --toy, when the branch
/// touches kernels).
struct KernelCase: CustomTestStringConvertible, Sendable {
    let id: String
    /// Model whose dimensions are compiled (KernelLibraryTests.dims).
    let model: String
    let catches: String
    let run: @Sendable (MetalContext, inout SplitMix64) throws -> [Check]
    var testDescription: String { id }

    /// Runs the case on the context of its model (created on first use and kept in `contexts`) and checks every result.
    /// Prints `kcase-time <id> <seconds>` with the wall time of context creation (library compilation, including the
    /// wait for its lock) plus the run and the checks; scripts/ci.sh copies it to .build/ci/timings.txt.
    static func evaluate(_ c: KernelCase, contexts: inout [String: MetalContext]) throws {
        let t0 = ContinuousClock.now
        defer {
            let d = ContinuousClock.now - t0
            let s = Double(d.components.seconds) + Double(d.components.attoseconds) * 1e-18
            print("kcase-time \(c.id) \(String(format: "%.3f", s))")
        }
        let dims = try KernelLibraryTests.dims(c.model)
        if contexts[c.model] == nil { contexts[c.model] = try MetalContext(dims: dims) }
        var g = SplitMix64(seed: 2026_10_21)
        let checks = try c.run(contexts[c.model]!, &g)
        #expect(!checks.isEmpty)
        for ch in checks {
            let r = ch.evaluate()
            print("kcase \(c.id) | \(r.text)")
            #expect(r.ok, "\(c.id) [\(c.catches)] \(r.text)")
        }
    }
}

@Suite(.serialized) struct KernelCaseTests {
    /// One context per model (compiling the library takes ~1 s); synthetic layers read their shape from the context.
    nonisolated(unsafe) static var contexts: [String: MetalContext] = [:]

    @Test(arguments: KernelCase.all)
    func matchesReference(_ c: KernelCase) throws {
        try KernelCase.evaluate(c, contexts: &Self.contexts)
    }
}

// MARK: - Cases

extension KernelCase {
    static let all: [KernelCase] =
        [
            bonsaiQmv, bonsaiPrefill, bonsaiEmbed, bonsaiAttn, lfmQmv6, lfmRouter(64), lfmRouter(32),
            lfmAttn, qmv4Rows, int8Serve, moeRouteE512, attnHD128, attnNoQKNorm, moeShared, moeNemotron, q4G32,
            attnSWA, attnSinks, attnRing, attnRopeTab, attnG4Full, attnG4Sliding, moeGptOss,
            moeMXFP4,
        ] + lfmConvs + mamba2 + flashNext + activations + blockPieceCases + routerMLX + mixedBits + attnHadamard + mla

    /// Per-row invariance: row i of a block gives the same bits with i + 1 rows as with 16, and with other
    /// neighbouring rows (the greedy speculative round must match single-token decoding bit for bit).
    static func rowInvariance(
        _ label: String, K: Int, _ g: inout SplitMix64, run: ([Float], Int) throws -> [Float]
    ) rethrows -> Check {
        let R = 16
        let x = Syn.roundHalf(Syn.normal(&g, R * K))
        let full = try run(x, R)
        let N = full.count / R
        var got: [Float] = [], want: [Float] = []
        for b in 1...R {
            let y = try run(Array(x[0..<(b * K)]), b)
            got += y[((b - 1) * N)..<(b * N)]
            want += full[((b - 1) * N)..<(b * N)]
        }
        var x2 = Syn.roundHalf(Syn.normal(&g, R * K))
        x2.replaceSubrange(0..<K, with: x[0..<K])
        got += try run(x2, R)[0..<N]
        want += full[0..<N]
        return Check(label: "\(label): per-row invariance", gpu: got, ref: want, tol: .bitExact)
    }

    /// Round qmv (VQMV.encode) of `rows` rows with half x.
    static func qmv(_ ctx: MetalContext, _ m: DenseQ, _ x: [Float], rows: Int) throws -> [Float] {
        let xb = Syn.halfBuffer(ctx, x), y = ctx.buffer(length: rows * m.N * 4)
        try ctx.run { e in try VQMV(ctx).encode(e, x: xb, m, y: y, rows: rows) }
        return Syn.floats(y, rows * m.N)
    }

    // MARK: Bonsai 2 (rotated ternary)

    static let bonsaiQmv = KernelCase(
        id: "bonsai-qmv", model: "bonsai-2-27b",
        catches:
            "ternary round qmv (had_rows_h + vnax_qmvt, split-K of 8 and 16 simdgroups): 2-bit decoding, "
            + "per-group scale of 128, Hadamard rotation or reduction; and that a row does not depend on the others"
    ) { ctx, g in
        var checks: [Check] = []
        for (N, K, rows) in [(64, 2048, 16), (64, 2048, 5), (32, 9216, 16)] {
            let (m, w, sg) = Syn.ternary(ctx, N: N, K: K, &g)
            let x = Syn.roundHalf(Syn.normal(&g, rows * K))
            let ref = CPU.matmulT(CPU.rotate(x, rows: rows, K: K, signs: sg), w, M: rows, N: N, K: K)
            checks.append(
                Check(
                    label: "\(N)×\(K), \(rows) rows", gpu: try qmv(ctx, m, x, rows: rows), ref: ref, tol: .maxRel(1e-4)
                ))
        }
        let (m, _, _) = Syn.ternary(ctx, N: 64, K: 2048, &g)
        checks.append(try rowInvariance("64×2048", K: 2048, &g) { x, r in try qmv(ctx, m, x, rows: r) })
        return checks
    }

    static let bonsaiPrefill = KernelCase(
        id: "bonsai-prefill", model: "bonsai-2-27b",
        catches: "ternary prefill GEMM (qmm_t_nax3t, 64×64 and 64×128 tiles) after the input rotation"
    ) { ctx, g in
        try [(64, 2048), (128, 4096)].map { N, K in
            let M = 64
            let (m, w, sg) = Syn.ternary(ctx, N: N, K: K, &g)
            let x = Syn.roundHalf(Syn.normal(&g, M * K))
            let xb = Syn.halfBuffer(ctx, x), y = ctx.buffer(length: M * N * 4)
            try ctx.run { e in try m.encodeQmmF32(ctx, e, x: xb, y: y, rows: M) }
            let ref = CPU.matmulT(CPU.rotate(x, rows: M, K: K, signs: sg), w, M: M, N: N, K: K)
            return Check(label: "\(M)×\(N)×\(K)", gpu: Syn.floats(y, M * N), ref: ref, tol: .maxRel(1e-4))
        }
    }

    static let bonsaiEmbed = KernelCase(
        id: "bonsai-embed", model: "bonsai-2-27b",
        catches: "ternary embedding (embed_rows_t: dequantization and inverse Hadamard) and the fp32 b|a projection of "
            + "the GDN (rowdot_f32w)"
    ) { ctx, g in
        let V = 40, K = 2048, rows = 6
        let (m, w, sg) = Syn.ternary(ctx, N: V, K: K, &g)
        let tok: [UInt32] = [3, 0, 39, 17, 17, 8]
        let out = ctx.buffer(length: rows * K * 4)
        try ctx.run { e in try m.encodeEmbed(ctx, e, tok: ctx.buffer(tok), outF: out, rows: rows) }
        var ref = [Float]()
        for t in tok {
            var v = Array(w[(Int(t) * K)..<(Int(t) * K + K)])
            for b0 in stride(from: 0, to: K, by: 1024) {
                var blk = Array(v[b0..<(b0 + 1024)])
                CPU.fwht(&blk)
                for i in 0..<1024 { v[b0 + i] = blk[i] / 32 * sg[b0 + i] }
            }
            ref += v
        }
        // rowdot_f32w: y[m, col + n] += x[m]·W[n] with fp32 W (96 rows in the 27B: in_proj_b | in_proj_a).
        let N = 96, M = 5, ld = 160, col = 40
        let x = Syn.roundHalf(Syn.normal(&g, M * K)), W = Syn.normal(&g, N * K, sd: 0.02)
        let y0 = Syn.normal(&g, M * ld)
        let y = ctx.buffer(y0)
        var p = (UInt32(M), UInt32(N), UInt32(K), UInt32(K), UInt32(ld), UInt32(1))
        try ctx.run { e in
            e.setComputePipelineState(try ctx.pipeline("rowdot_f32w_f"))
            e.setBuffer(Syn.halfBuffer(ctx, x), offset: 0, index: 0)
            e.setBuffer(ctx.buffer(W), offset: 0, index: 1)
            e.setBuffer(y, offset: col * 4, index: 2)
            e.setBytes(&p, length: 24, index: 3)
            e.dispatchThreadgroups(
                MTLSize(width: (N + 7) / 8, height: M, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
        let xw = CPU.matmulT(x, W, M: M, N: N, K: K)
        var yref = y0
        for r in 0..<M { for n in 0..<N { yref[r * ld + col + n] += xw[r * N + n] } }
        return [
            Check(label: "embedding \(rows) rows", gpu: Syn.floats(out, rows * K), ref: ref, tol: .maxRel(1e-5)),
            Check(label: "rowdot_f32w accumulated", gpu: Syn.floats(y, M * ld), ref: yref, tol: .maxRel(1e-5)),
        ]
    }

    /// Attention with synthetic weights against the CPU: with little context (the new keys weigh, a single chunk) and
    /// with 700 positions (two chunks of 512 and their reduction), in the round and, if `prefill`, in a chunk of 100
    /// rows; and the per-row invariance of the round.
    static func attnChecks(
        _ ctx: MetalContext, _ g: inout SplitMix64, prefill: Bool, qkvBias: Bool = false, nope: Bool = false
    ) throws -> [Check] {
        var checks: [Check] = []
        for P in [40, 700] {
            let f = try AttnFixture(ctx, positions: P, &g, nope: nope, qkvBias: qkvBias)
            let x = Syn.roundHalf(Syn.normal(&g, 16 * f.hidden))
            let gr = try f.round(x, rows: 16, P: P)
            checks.append(
                Check(
                    label: "round of 16 rows at \(P)", gpu: gr, ref: f.reference(x, rows: 16, P: P),
                    tol: .maxRel(5e-3)))
            if P == 700 {
                checks.append(try rowInvariance("round", K: f.hidden, &g) { x, r in try f.round(x, rows: r, P: P) })
            }
            if prefill {
                let xp = Syn.roundHalf(Syn.normal(&g, 100 * f.hidden))
                let gp = try f.prefill(xp, rows: 100, P: P)
                checks.append(
                    Check(
                        label: "prefill of 100 rows at \(P)", gpu: gp, ref: f.reference(xp, rows: 100, P: P),
                        tol: .maxRel(5e-3)))
            }
        }
        return checks
    }

    static let bonsaiAttn = KernelCase(
        id: "bonsai-attn", model: "bonsai-2-27b",
        catches: "round attention with the 27B/Bonsai dimensions (GQA 6, gate, Bonsai fp32 norms): against the CPU "
            + "and that a row does not depend on the others (rowscheck only checks it in the MoE)"
    ) { ctx, g in try attnChecks(ctx, &g, prefill: false) }

    // MARK: LFM2.5

    static let lfmQmv6 = KernelCase(
        id: "lfm-qmv6", model: "lfm2.5-24b-a2b",
        catches: "6-bit unpacking of the tied lm_head and embedding of LFM2.5 (qmv6_nw4, embed_rows_b6), and that a "
            + "row of the head does not depend on the others or on the number of rows (qmv6m did)"
    ) { ctx, g in
        let N = 128, K = 2048
        let (m, w) = Syn.q6(ctx, N: N, K: K, &g)
        var checks: [Check] = []
        for rows in [1, 16] {
            let x = Syn.roundHalf(Syn.normal(&g, rows * K))
            checks.append(
                Check(
                    label: "qmv6 \(rows) rows", gpu: try qmv(ctx, m, x, rows: rows),
                    ref: CPU.matmulT(x, w, M: rows, N: N, K: K), tol: .maxRel(1e-4)))
        }
        checks.append(try rowInvariance("qmv6", K: K, &g) { x, r in try qmv(ctx, m, x, rows: r) })
        let tok: [UInt32] = [5, 127, 0, 64]
        let out = ctx.buffer(length: tok.count * K * 4)
        try ctx.run { e in try m.encodeEmbed(ctx, e, tok: ctx.buffer(tok), outF: out, rows: tok.count) }
        let ref = tok.flatMap { w[(Int($0) * K)..<(Int($0) * K + K)] }
        checks.append(Check(label: "embedding", gpu: Syn.floats(out, tok.count * K), ref: ref, tol: .maxRel(1e-5)))
        return checks
    }

    static let lfmConv = lfmConvCase(id: "lfm-conv", model: "lfm2.5-24b-a2b", bits: 4)

    /// The short convolution of every width and bits the engine runs: LFM2.5's 3 taps and LFM3-1.7B's 7 (LAB_SC_L), with
    /// in_proj/out_proj at 4 bits (LFM2.5) and 8 (the LFM3-1.7B reference, MLX-8bit).
    static let lfmConvs = [
        lfmConv, lfmConvCase(id: "lfm-conv-8bit", model: "lfm2.5-24b-a2b", bits: 8),
        lfmConvCase(id: "lfm-conv7", model: "lfm3-1.7b", bits: 4),
        lfmConvCase(id: "lfm-conv7-8bit", model: "lfm3-1.7b", bits: 8),
    ]

    static func lfmConvCase(id: String, model: String, bits: Int) -> KernelCase {
        KernelCase(
            id: id, model: model,
            catches:
                "short convolution of the model's width (LAB_SC_L) with \(bits)-bit projections (sconv_prefill with "
                + "padding, sconv_hist, sconv_step) and its window across chunks and rounds, also when folding part of the "
                + "previous round (nacc, from the state on the GPU), the snapshot fold and per-row invariance"
        ) { ctx, g in try lfmConvChecks(ctx, &g, bits: bits) }
    }

    /// One-hot matrix at `bits` (4 or 8; row n: column cols[n]): y = x·Wᵀ is an exact permutation of x.
    static func oneHot(_ ctx: MetalContext, N: Int, K: Int, cols: [Int], bits: Int) -> DenseQ {
        guard bits == 8 else { return Syn.oneHot4(ctx, N: N, K: K, cols: cols) }
        var words = [UInt32](repeating: 0, count: N * K / 4)
        for n in 0..<N where cols[n] >= 0 { words[(n * K + cols[n]) / 4] |= 1 << UInt32(8 * (cols[n] % 4)) }
        let s = [UInt16](repeating: floatToBF16(1), count: N * K / 64), b = [UInt16](repeating: 0, count: N * K / 64)
        return DenseQ(w: ctx.buffer(words), s: ctx.buffer(s), b: ctx.buffer(b), N: N, K: K, bits: 8)
    }

    static func lfmConvChecks(_ ctx: MetalContext, _ g: inout SplitMix64, bits: Int) throws -> [Check] {
        let D = ctx.dims.hidden, L = ctx.dims.convL, W = L - 1
        let colIn = (0..<(3 * D)).map { _ in Int(g.next() % UInt64(D)) }
        let colOut = (0..<D).map { _ in Int(g.next() % UInt64(D)) }
        let cw = Syn.roundBF16(Syn.normal(&g, L * D, sd: 0.5))
        let c = ShortConvLayer(
            ctx: ctx, layer: 0, inProj: oneHot(ctx, N: 3 * D, K: D, cols: colIn, bits: bits),
            outProj: oneHot(ctx, N: D, K: D, cols: colOut, bits: bits), convW: Syn.bf16Buffer(ctx, cw))
        // Reference: Bx = B ⊙ x, causal L-tap conv over the Bx (tap 0 the oldest), y = C ⊙ conv (half), out = permutation
        // of y.
        var bxHist = [[Float]](repeating: [Float](repeating: 0, count: D), count: W)
        func ref(_ x: [Float], rows: Int, keep: Int? = nil) -> [Float] {
            var out = [Float](repeating: 0, count: rows * D)
            var h = bxHist
            for t in 0..<rows {
                let pr = colIn.map { x[t * D + $0] }
                let bx = (0..<D).map { pr[$0] * pr[2 * D + $0] }
                h.append(bx)
                let y = (0..<D).map { ch in
                    var cv: Float = 0
                    for j in 0..<L { cv += cw[ch * L + j] * h[h.count - L + j][ch] }
                    return Float(Float16(pr[D + ch] * cv))
                }
                for n in 0..<D { out[t * D + n] = y[colOut[n]] }
            }
            // Window after accepting `keep` rows (all if nil).
            let k = keep ?? rows
            bxHist = Array(h[k..<(k + W)])
            return out
        }
        var checks: [Check] = []
        // Prefill: 40 real rows in a chunk of 64 (the padding rows add nothing).
        let r = 40, M = 64
        let x0 = Syn.roundHalf(Syn.normal(&g, M * D))
        let out0 = ctx.buffer(length: M * D * 4)
        memset(out0.contents(), 0, out0.length)
        let s = (proj: ctx.buffer(length: M * 3 * D * 4), y: ctx.buffer(length: M * D * 2))
        try ctx.run { e in
            try c.encodePrefill(e, x: Syn.halfBuffer(ctx, x0), out: out0, rows: r, padded: M, s: s)
        }
        let g0 = Syn.floats(out0, M * D)
        checks.append(
            Check(label: "prefill 40 of 64", gpu: Array(g0[0..<(r * D)]), ref: ref(x0, rows: r), tol: .maxRel(1e-4)))
        checks.append(
            Check(
                label: "prefill: padding rows", gpu: Array(g0[(r * D)...]),
                ref: [Float](repeating: 0, count: (M - r) * D), tol: .bitExact))
        // A second chunk of 3 real rows (fewer than the window: part of it comes from the previous chunk's).
        let x0b = Syn.roundHalf(Syn.normal(&g, M * D))
        memset(out0.contents(), 0, out0.length)
        try ctx.run { e in
            try c.encodePrefill(e, x: Syn.halfBuffer(ctx, x0b), out: out0, rows: 3, padded: M, s: s)
        }
        checks.append(
            Check(
                label: "second chunk of 3 rows", gpu: Array(Syn.floats(out0, 3 * D)), ref: ref(x0b, rows: 3),
                tol: .maxRel(1e-4)))
        // Round 1: 5 rows (nacc 0, fixed); 3 are accepted. Round 2: 7 rows, nacc = 3 read from the round state.
        let x1 = Syn.roundHalf(Syn.normal(&g, 5 * D)), out1 = ctx.buffer(length: 5 * D * 4)
        try ctx.run { e in try c.encode(e, x: Syn.halfBuffer(ctx, x1), out: out1, rows: 5, nacc: 0, accumulate: false) }
        c.advance()
        checks.append(
            Check(label: "round of 5", gpu: Syn.floats(out1, 5 * D), ref: ref(x1, rows: 5, keep: 3), tol: .maxRel(1e-4))
        )
        let x2 = Syn.roundHalf(Syn.normal(&g, 7 * D)), out2 = ctx.buffer(length: 7 * D * 4)
        let st = ctx.buffer([UInt32(0), 3, 0, 0, 0, 0, 0, 0])
        try ctx.run { e in
            try c.encode(
                e, x: Syn.halfBuffer(ctx, x2), out: out2, rows: 7, nacc: 0, accumulate: false, roundState: st)
        }
        c.advance()
        checks.append(
            Check(
                label: "round of 7 after accepting 3 (nacc from the state)", gpu: Syn.floats(out2, 7 * D),
                ref: ref(x2, rows: 7, keep: 2), tol: .maxRel(1e-4)))
        // Snapshot fold: after accepting 2 of the 7 rows, the window written to a snapshot buffer is the reference's
        // (W rows; StateSnapshot.conv) and equals the one the fold leaves in the next history.
        let snap = ctx.buffer(length: W * D * 4)
        try ctx.run { e in try c.encodeFold(e, nacc: 2, histOut: snap) }
        checks.append(
            Check(
                label: "snapshot fold (\(W) rows)", gpu: Syn.floats(snap, W * D), ref: bxHist.flatMap { $0 },
                tol: .bitExact))
        try ctx.run { e in try c.encodeFold(e, nacc: 2) }
        checks.append(
            Check(
                label: "fold into the history", gpu: Syn.floats(c.hist[c.cur], W * D), ref: Syn.floats(snap, W * D),
                tol: .bitExact))
        // Per-row invariance of a round from the same state (the speculative round vs single-token decoding).
        checks.append(
            try rowInvariance("conv round (\(L) taps, \(bits) bits)", K: D, &g) { x, rows in
                c.reset()
                let o = ctx.buffer(length: rows * D * 4)
                try ctx.run { e in
                    try c.encode(e, x: Syn.halfBuffer(ctx, x), out: o, rows: rows, nacc: 0, accumulate: false)
                }
                c.advance()
                return Syn.floats(o, rows * D)
            })
        return checks
    }

    /// E = 64 (LFM2.5-24B) and E = 32 (LFM2.5-8B; E and top-k come from the config).
    static func lfmRouter(_ E: Int) -> KernelCase {
        KernelCase(
            id: "lfm-router-e\(E)", model: E == 64 ? "lfm2.5-24b-a2b" : "lfm2.5-8b-a1b",
            catches: "LFM2.5 sigmoid router on the common kernels (vmoe_route, moe_route_topk_cfg) with \(E) experts: "
                + "top-4 by p + expert_bias, ties to lower index, unbiased renormalized weights (1e-6), expert union"
        ) { ctx, g in try lfmRouterChecks(ctx, &g, E: E) }
    }
    static func lfmRouterChecks(_ ctx: MetalContext, _ g: inout SplitMix64, E: Int) throws -> [Check] {
        let K = 4, B = 12
        var logits = (0..<(B * E)).map { _ in 8 * g.uniform() - 4 }
        var bias = (0..<E).map { _ in 0.6 * g.uniform() - 0.3 }
        // Row 0: exact tie for 4th place between 10 and 20 (10 wins). Row 1: the bias changes the order (7 gets in
        // with a lower p than 30) and the weight is that of p, without bias.
        for e in 0..<E { logits[e] = -3; logits[E + e] = -3 }
        for e in [1, 2, 3] { logits[e] = 5 }
        logits[10] = 2; logits[20] = 2; bias[10] = 0; bias[20] = 0
        bias[7] = 0.25; bias[30] = -0.25; logits[E + 7] = 1.0; logits[E + 30] = 1.2
        for e in [E - 8, E - 7, E - 6] { logits[E + e] = 6; bias[e] = 0 }
        // Reference.
        var inds = [Float](), wts = [Float]()
        for r in 0..<B {
            let p = (0..<E).map { 1 / (1 + exp(-logits[r * E + $0])) }
            let order = (0..<E).sorted { (p[$0] + bias[$0], -$0) > (p[$1] + bias[$1], -$1) }.prefix(K)
            let sum = order.reduce(Float(0)) { $0 + p[$1] }
            inds += order.map { Float($0) }
            wts += order.map { p[$0] / (sum + 1e-6) }
        }
        // Union (vmoe_route format): experts in increasing order with (offset, count) and (row, weight) pairs.
        var entries = [Float](), pairRows = [Float](), pairW = [Float]()
        var off = 0
        for e in 0..<E {
            let hits = (0..<(B * K)).filter { Int(inds[$0]) == e }
            if hits.isEmpty { continue }
            entries += [Float(e), Float(off), Float(hits.count)]
            off += hits.count
            pairRows += hits.map { Float($0 / K) }
            pairW += hits.map { wts[$0] }
        }
        // Logits with the router row stride padded to 64 (LFM_LD, MoELayer.routerLD).
        let ld = MoELayer.routerLD(E)
        var padded = [Float](repeating: 0, count: B * ld)
        for r in 0..<B { padded.replaceSubrange((r * ld)..<(r * ld + E), with: logits[(r * E)..<(r * E + E)]) }
        let lb = ctx.buffer(padded), bb = ctx.buffer(bias)
        let ent = ctx.buffer(length: E * 16), pairs = ctx.buffer(length: B * K * 8), args = ctx.buffer(length: 64)
        let ib = ctx.buffer(length: B * K * 4), wb = ctx.buffer(length: B * K * 4)
        let ib2 = ctx.buffer(length: B * K * 4), wb2 = ctx.buffer(length: B * K * 4)
        let z = ctx.buffer(length: 2048 * 2), sg2 = ctx.buffer(length: B * 4)
        try ctx.run { e in
            try encodeRouteCommon(
                ctx, e, logits: lb, bias: bb, zero: z, rowLimit: args, outs: [ent, pairs, args, ib, wb],
                params: [2048, 1536, UInt32(E), UInt32(B), 48, 0, 1536])
            try encodePrefillRouteCommon(
                ctx, e, logits: lb, bias: bb, zero: z, outs: [ib2, wb2, sg2],
                params: [UInt32(B), UInt32(E), 2048, UInt32(K)])
        }
        let U = Int(Syn.uints(args, 8)[1])
        let ge = Syn.uints(ent, 4 * U)
        let gp = Syn.uints(pairs, 2 * B * K)
        return [
            Check(label: "round: experts", gpu: Syn.uints(ib, B * K).map { Float($0) }, ref: inds, tol: .bitExact),
            Check(label: "round: weights", gpu: Syn.floats(wb, B * K), ref: wts, tol: .maxRel(1e-5)),
            Check(
                label: "round: union (expert, offset, count)",
                gpu: (0..<U).flatMap { [Float(ge[4 * $0]), Float(ge[4 * $0 + 1]), Float(ge[4 * $0 + 2])] },
                ref: entries, tol: .bitExact),
            Check(
                label: "round: pair rows", gpu: (0..<(B * K)).map { Float(gp[2 * $0]) }, ref: pairRows,
                tol: .bitExact),
            Check(
                label: "round: pair weights", gpu: (0..<(B * K)).map { Float(bitPattern: gp[2 * $0 + 1]) },
                ref: pairW, tol: .maxRel(1e-5)),
            Check(label: "prefill: experts", gpu: Syn.uints(ib2, B * K).map { Float($0) }, ref: inds, tol: .bitExact),
            Check(label: "prefill: weights", gpu: Syn.floats(wb2, B * K), ref: wts, tol: .maxRel(1e-5)),
        ]
    }

    static let moeRouteE512 = KernelCase(
        id: "moe-route-e512", model: "qwen3.8-flash-next",
        catches: "Qwen3.6 router with E and top-k from the config (vmoe_route and moe_route_topk_cfg with 512 top-10): "
            + "top-k with ties to the lower index, renormalized weights, 8-bit shared gate with hidden 2560 (not a "
            + "multiple of 2048) and the round's expert union"
    ) { ctx, g in
        let E = 512, K = 10, B = 12, D = 2560
        var logits = (0..<(B * E)).map { _ in 8 * g.uniform() - 4 }
        // Row 0: nine clear ones and an exact tie for 10th place between 301 and 450 (301 wins).
        for e in 0..<E { logits[e] = -3 }
        for e in [5, 17, 33, 64, 100, 200, 255, 256, 511] { logits[e] = 5 }
        logits[301] = 2; logits[450] = 2
        let x = Syn.roundHalf(Syn.normal(&g, B * D, sd: 0.5))
        var codes = [UInt32](repeating: 0, count: D / 4)
        for i in codes.indices { codes[i] = UInt32(truncatingIfNeeded: g.next()) }
        let sc = (0..<(D / 64)).map { _ in floatToBF16(0.001 + 0.002 * g.uniform()) }
        let bi = (0..<(D / 64)).map { _ in floatToBF16(-0.25 * g.uniform()) }
        // Reference.
        var inds = [Float](), wts = [Float](), gate = [Float]()
        for r in 0..<B {
            let l = Array(logits[(r * E)..<(r * E + E)])
            let top = l.indices.sorted { l[$0] > l[$1] || (l[$0] == l[$1] && $0 < $1) }.prefix(K)
            let ex = top.map { exp(l[$0] - l[top.first!]) }, sum = ex.reduce(0, +)
            inds += top.map { Float($0) }
            wts += ex.map { $0 / sum }
            var acc: Float = 0
            for k in 0..<D {
                let q = Float((codes[k / 4] >> UInt32(8 * (k % 4))) & 0xff)
                acc += x[r * D + k] * (bf16ToFloat(sc[k / 64]) * q + bf16ToFloat(bi[k / 64]))
            }
            gate.append(1 / (1 + exp(-acc)))
        }
        var entries = [Float](), pairRows = [Float](), pairW = [Float]()
        var off = 0
        for e in 0..<E {
            let hits = (0..<(B * K)).filter { Int(inds[$0]) == e }
            if hits.isEmpty { continue }
            entries += [Float(e), Float(off), Float(hits.count)]
            off += hits.count
            pairRows += hits.map { Float($0 / K) }
            pairW += hits.map { wts[$0] }
        }
        entries += [Float(E), Float(B * K), Float(B)]
        let lb = ctx.buffer(logits), xb = Syn.halfBuffer(ctx, x), wb8 = ctx.buffer(codes), sb = ctx.buffer(sc),
            bb = ctx.buffer(bi)
        let ent = ctx.buffer(length: (B * K + 1) * 16), pairs = ctx.buffer(length: (B * K + B) * 8)
        let args = ctx.buffer(length: 32), ib = ctx.buffer(length: B * K * 4), wb = ctx.buffer(length: B * K * 4)
        let ib2 = ctx.buffer(length: B * K * 4), wb2 = ctx.buffer(length: B * K * 4), sg2 = ctx.buffer(length: B * 4)
        var mp = (
            UInt32(D), UInt32(640), UInt32(E), UInt32(B), UInt32(20), UInt32(K) << 16 | UInt32(K + 1) << 24, UInt32(640)
        )
        var rp = (UInt32(B), UInt32(E), UInt32(D), UInt32(K))
        try ctx.run { e in
            e.setComputePipelineState(try ctx.pipeline("vmoe_route"))
            for (i, b) in [lb, xb, wb8, sb, bb, ent, pairs, args, ib, wb].enumerated() {
                e.setBuffer(b, offset: 0, index: i)
            }
            e.setBytes(&mp, length: 28, index: 10)
            e.setBuffer(args, offset: 0, index: 11)
            e.dispatchThreadgroups(
                MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 512, height: 1, depth: 1))
            e.setComputePipelineState(try ctx.pipeline("moe_route_topk_cfg"))
            for (i, b) in [lb, xb, wb8, sb, bb, ib2, wb2, sg2].enumerated() { e.setBuffer(b, offset: 0, index: i) }
            e.setBytes(&rp, length: 16, index: 8)
            e.dispatchThreadgroups(
                MTLSize(width: (B + 3) / 4, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        }
        let U = Int(Syn.uints(args, 8)[1])
        let ge = Syn.uints(ent, 4 * U)
        let gp = Syn.uints(pairs, 2 * (B * K + B))
        return [
            Check(label: "round: experts", gpu: Syn.uints(ib, B * K).map { Float($0) }, ref: inds, tol: .bitExact),
            Check(label: "round: weights", gpu: Syn.floats(wb, B * K), ref: wts, tol: .maxRel(1e-5)),
            Check(
                label: "round: union (expert, offset, count) and shared",
                gpu: (0..<U).flatMap { [Float(ge[4 * $0]), Float(ge[4 * $0 + 1]), Float(ge[4 * $0 + 2])] },
                ref: entries, tol: .bitExact),
            Check(
                label: "round: pair rows", gpu: (0..<(B * K)).map { Float(gp[2 * $0]) }, ref: pairRows,
                tol: .bitExact),
            Check(
                label: "round: pair weights", gpu: (0..<(B * K)).map { Float(bitPattern: gp[2 * $0 + 1]) },
                ref: pairW, tol: .maxRel(1e-5)),
            Check(
                label: "round: shared gate",
                gpu: (0..<B).map { Float(bitPattern: gp[2 * (B * K + $0) + 1]) }, ref: gate, tol: .maxRel(1e-4)),
            Check(label: "prefill: experts", gpu: Syn.uints(ib2, B * K).map { Float($0) }, ref: inds, tol: .bitExact),
            Check(label: "prefill: weights", gpu: Syn.floats(wb2, B * K), ref: wts, tol: .maxRel(1e-5)),
            Check(label: "prefill: shared gate", gpu: Syn.floats(sg2, B), ref: gate, tol: .maxRel(1e-4)),
        ]
    }

    static let lfmAttn = KernelCase(
        id: "lfm-attn", model: "lfm2.5-24b-a2b",
        catches: "LFM2.5 attention (64-wide heads without gate, GQA 4) in the round (dattn_i8) and in the prefill "
            + "(pattn_i8) against the CPU, and that a row of the round does not depend on the others"
    ) { ctx, g in try attnChecks(ctx, &g, prefill: true) }

    // MARK: Generic attention

    static let attnHD128 = KernelCase(
        id: "attn-hd128", model: "muse-glimmer-30b",
        catches: "attention with a 128-wide head and full RoPE in Muse Glimmer's shape (GQA 32/2, gate, LAB_AT_SWA; "
            + "end to end in toy-muse): vattn_prework_i8 (128 threads, 4 simdgroups in the norm), dattn_i8 and "
            + "pattn_i8 (P·V of 32 dimensions per simdgroup) and vattn_reduce against the CPU (vattn_nax_i8, only "
            + "for GQA 8 and only in the ingot-dev pattn bench, is no longer checked here: Muse's G is 16); prefill "
            + "int8 Q·Kᵀ (q_quant_i8 with 4 values per lane); layer without RoPE (nope bit of AttnParams.dyn) and "
            + "weightless QK-norm (buffer of ones); and Muse's two layers with its attention scale "
            + "(qk_scale_factor 3.87 / √128): the sliding one (RoPE θ 5e5, window 2048, ring of 4224 slots) with the "
            + "window start inside, on and past 2048 and wrapping the ring, and the full one without RoPE"
    ) { ctx, g in
        var checks = try attnChecks(ctx, &g, prefill: true)
        let f8 = try AttnFixture(ctx, positions: 300, &g)
        let x8 = Syn.roundHalf(Syn.normal(&g, 64 * f8.hidden))
        checks.append(
            Check(
                label: "prefill with Q·Kᵀ int8", gpu: try f8.prefill(x8, rows: 64, P: 300, qk8: true),
                ref: f8.reference(x8, rows: 64, P: 300, qk8: true), tol: .maxRel(5e-3)))
        for (nope, unit, tag) in [(true, false, "no RoPE"), (false, true, "weightless QK-norm")] {
            let f = try AttnFixture(ctx, positions: 700, &g, nope: nope, unitNorm: unit)
            let x = Syn.roundHalf(Syn.normal(&g, 16 * f.hidden))
            checks.append(
                Check(
                    label: "\(tag): round of 16 rows at 700", gpu: try f.round(x, rows: 16, P: 700),
                    ref: f.reference(x, rows: 16, P: 700), tol: .maxRel(5e-3)))
            let xp = Syn.roundHalf(Syn.normal(&g, 100 * f.hidden))
            checks.append(
                Check(
                    label: "\(tag): prefill of 100 rows at 700", gpu: try f.prefill(xp, rows: 100, P: 700),
                    ref: f.reference(xp, rows: 100, P: 700), tol: .maxRel(5e-3)))
        }
        // Muse's layers: weightless QK-norm, q × qk_scale_factor folded into the score scale (BlockSpec.attnScale),
        // sliding (RoPE, W 2048, default ring: 2048 + 2048 + 16 rounded up, 4224 slots) and full (NoPE, no window).
        let museScale = Float(3.87 / Double(128).squareRoot())
        for (P, W) in [(2000, 2048), (2048, 2048), (2100, 2048), (4300, 2048), (9000, 2048), (3000, 0)] {
            let f = try AttnFixture(
                ctx, positions: P, &g, nope: W == 0, unitNorm: true, maxCtx: 16384, scale: museScale, window: W)
            precondition(W == 0 || f.a.ring == 4224)
            let tag = W == 0 ? "Muse full (NoPE)" : "Muse sliding W \(W), ring \(f.a.ring)"
            let x = Syn.roundHalf(Syn.normal(&g, 16 * f.hidden))
            checks.append(
                Check(
                    label: "\(tag): round of 16 rows at \(P)", gpu: try f.round(x, rows: 16, P: P),
                    ref: f.reference(x, rows: 16, P: P), tol: .maxRel(5e-3)))
            // Norm-wise in the sliding layer, as attn-ropetab's prefill: with the window start inside the chunk, isolated
            // elements land up to 7e-3 off in max-relative terms (int8 K steps; 0.0015-0.007 over seeds at the same
            // positions), while the rounds above stay within 3e-3.
            let xp = Syn.roundHalf(Syn.normal(&g, 100 * f.hidden))
            checks.append(
                Check(
                    label: "\(tag): prefill of 100 rows at \(P + 16)", gpu: try f.prefill(xp, rows: 100, P: P + 16),
                    ref: f.reference(xp, rows: 100, P: P + 16), tol: W == 0 ? .maxRel(5e-3) : .rms(2e-3)))
        }
        return checks
    }

    static let attnNoQKNorm = KernelCase(
        id: "attn-noqkn", model: "nemotron-3.5-lightning",
        catches:
            "attention without QK-norm (LAB_AT_QKN = 0: the prework normalizes neither q nor k) in Nemotron 3.5 "
            + "Lightning's shape (128-wide heads, GQA 32/2, no gate, no RoPE), in the round and in the prefill"
    ) { ctx, g in try attnChecks(ctx, &g, prefill: true, nope: true) }

    // MARK: Common

    static let qmv4Rows = KernelCase(
        id: "qmv4-rows", model: "qwen3.8-27b",
        catches: "per-row invariance of the 4-bit round qmv (vnax_qmv, with ≤ 8 and > 8 rows) in the dense 27B "
            + "shapes, which rowscheck does not check"
    ) { ctx, g in
        let (m, _) = Syn.q4(ctx, N: 256, K: 5120, &g)
        return [try rowInvariance("256×5120", K: 5120, &g) { x, r in try qmv(ctx, m, x, rows: r) }]
    }

    static let int8Serve = KernelCase(
        id: "int8-serve", model: "qwen3.6-35b-a3b",
        catches: "rotated W8A8 of the ingot serve prefill (w4_i8rot with and without precomputed scales, "
            + "quant_rows_rot_i8g, qmm_i8; int8 Q·Kᵀ of the attention) against its CPU emulation, and bounded error "
            + "against the exact one (gen --check runs exact, so no token sees it)"
    ) { ctx, g in
        var checks: [Check] = []
        let M = 64
        // (K, N, precomputed scales): Hadamard of 256 (K/32 a multiple of 32) or only of 32 (1536), 2×2 and 2×4 tiles.
        for (K, N, pre) in [(2048, 256, false), (2048, 256, true), (1536, 256, false), (9216, 256, false)] {
            let (m, w) = Syn.q4(ctx, N: N, K: K, &g)
            // Activations with a few outlier channels (what the rotation is for).
            var x = Syn.normal(&g, M * K)
            for ch in [7, 300, K - 5] { for r in 0..<M { x[r * K + ch] *= 20 } }
            x = Syn.roundHalf(x)
            let sc = DenseI8Scratch(ctx: ctx, maxNK: N * K, maxN: N, maxRows: M, maxK: K)
            let y = ctx.buffer(length: M * N * 4)
            if pre { try ctx.run { e in try m.precomputeRowScales(ctx, e, scratch: sc) } }
            try ctx.run { e in try m.encodeQmmI8(ctx, e, x: Syn.halfBuffer(ctx, x), y: y, rows: M, scratch: sc) }
            let lb = K / 32 % 32 == 0 ? 3 : 0
            let tag = "qmm_i8 \(M)×\(N)×\(K)\(pre ? " (precomputed scales)" : "")"
            checks.append(
                Check(
                    label: "\(tag) against the emulated W8A8", gpu: Syn.floats(y, M * N),
                    ref: CPU.w8a8(x, w, M: M, N: N, K: K, lb: lb), tol: .rms(2e-3)))
            checks.append(
                Check(
                    label: "\(tag) against the exact one", gpu: Syn.floats(y, M * N),
                    ref: CPU.matmulT(x, w, M: M, N: N, K: K), tol: .rms(0.05)))
        }
        // int8 Q·Kᵀ (pattn_q8, --qk8 1 in serve with the MoE): prefill attention of the Heretic.
        let P = 300
        let f = try AttnFixture(ctx, positions: P, &g)
        let xp = Syn.roundHalf(Syn.normal(&g, 64 * f.hidden))
        let gp = try f.prefill(xp, rows: 64, P: P, qk8: true)
        let rp = f.reference(xp, rows: 64, P: P, qk8: true)
        checks.append(Check(label: "prefill attention with int8 Q·Kᵀ", gpu: gp, ref: rp, tol: .maxRel(5e-3)))
        return checks
    }
}

// MARK: - Sliding window

extension KernelCase {
    static let attnSWA = KernelCase(
        id: "attn-swa", model: "gemma4-31b",
        catches:
            "sliding-window attention (dattn_i8w, pattn_i8w, pattn_q8w; Gemma 4's sliding layers, head of 256 without "
            + "gate, v_norm) against "
            + "the CPU with the window start inside and on a 128-key step, W = 1 (only its own key), W around the "
            + "context (P + 1, P + 16: the first row sees everything), decode splits entirely before the window, a "
            + "prefill chunk longer than W, and that a row does not depend on the others"
    ) { ctx, g in
        var checks: [Check] = []
        for (P, W) in [(700, 300), (700, 1), (700, 128), (700, 701), (700, 716), (700, 100), (1500, 300), (40, 300)] {
            let f = try AttnFixture(ctx, positions: P, &g, maxCtx: 4096, window: W)
            // W = 1: the output is the row's own int8 V, with no average to hide a one-level difference of the int8
            // quantization of the new V between GPU and CPU (1/127 of the largest value): 1e-2 there.
            let tol = Tol.maxRel(W == 1 ? 1e-2 : 5e-3)
            let x = Syn.roundHalf(Syn.normal(&g, 16 * f.hidden))
            checks.append(
                Check(
                    label: "W \(W): round of 16 rows at \(P)", gpu: try f.round(x, rows: 16, P: P),
                    ref: f.reference(x, rows: 16, P: P), tol: tol))
            let xp = Syn.roundHalf(Syn.normal(&g, 100 * f.hidden))
            checks.append(
                Check(
                    label: "W \(W): prefill of 100 rows at \(P)", gpu: try f.prefill(xp, rows: 100, P: P),
                    ref: f.reference(xp, rows: 100, P: P), tol: tol))
        }
        let f = try AttnFixture(ctx, positions: 700, &g, maxCtx: 4096, window: 300)
        checks.append(try rowInvariance("W 300 round", K: f.hidden, &g) { x, r in try f.round(x, rows: r, P: 700) })
        // Prefill chunk longer than the window, from position 0 (rows mask inside the chunk), and Q·Kᵀ int8.
        for W in [300, 128] {
            let f0 = try AttnFixture(ctx, positions: 0, &g, maxCtx: 4096, window: W)
            let x0 = Syn.roundHalf(Syn.normal(&g, 448 * f0.hidden))
            checks.append(
                Check(
                    label: "W \(W): prefill of 448 rows at 0", gpu: try f0.prefill(x0, rows: 448, P: 0),
                    ref: f0.reference(x0, rows: 448, P: 0), tol: .maxRel(5e-3)))
        }
        let f8 = try AttnFixture(ctx, positions: 500, &g, maxCtx: 4096, window: 300)
        let x8 = Syn.roundHalf(Syn.normal(&g, 64 * f8.hidden))
        checks.append(
            Check(
                label: "W 300: prefill with Q·Kᵀ int8", gpu: try f8.prefill(x8, rows: 64, P: 500, qk8: true),
                ref: f8.reference(x8, rows: 64, P: 500, qk8: true), tol: .maxRel(5e-3)))
        return checks
    }
}

extension KernelCase {
    static let attnSinks = KernelCase(
        id: "attn-sinks", model: "gpt-oss-20b",
        catches: "learned attention sinks (LAB_AT_SINKS: the pattn_i8 prefill epilogue and vattn_reduce) against the "
            + "CPU, with and without gpt-oss's sliding window of 128, in the round (one and several splits) and the "
            + "prefill"
    ) { ctx, g in
        var checks: [Check] = []
        // gpt-oss's shape: 64 q heads of 64 over 8 KV, no QK-norm, q|k|v biases and the YaRN table (the default one
        // here; attn-ropetab checks its own).
        for (P, W) in [(40, 0), (700, 0), (1500, 0), (700, 128)] {
            let f = try AttnFixture(ctx, positions: P, &g, maxCtx: 4096, qkvBias: ctx.dims.attnQKVB, window: W)
            let x = Syn.roundHalf(Syn.normal(&g, 16 * f.hidden))
            checks.append(
                Check(
                    label: "W \(W): round of 16 rows at \(P)", gpu: try f.round(x, rows: 16, P: P),
                    ref: f.reference(x, rows: 16, P: P), tol: .maxRel(5e-3)))
            let xp = Syn.roundHalf(Syn.normal(&g, 100 * f.hidden))
            checks.append(
                Check(
                    label: "W \(W): prefill of 100 rows at \(P)", gpu: try f.prefill(xp, rows: 100, P: P),
                    ref: f.reference(xp, rows: 100, P: P), tol: .maxRel(5e-3)))
        }
        return checks
    }
}

extension KernelCase {
    static let attnRing = KernelCase(
        id: "attn-ring", model: "gemma4-31b",
        catches:
            "ring KV of a sliding-window layer (slot = position % ring in vattn_prework_i8, dattn_i8w and pattn_i8w): "
            + "rounds and prefill chunks that wrap around the ring, with the window start inside a 128-key step"
    ) { ctx, g in
        var checks: [Check] = []
        // ringRows 128: ring = 300 + 128 + 16 → 512 slots; P up to 3000 wraps it several times.
        for P in [400, 500, 1500, 3000] {
            let f = try AttnFixture(ctx, positions: P, &g, maxCtx: 8192, window: 300, ringRows: 128)
            precondition(f.a.ring == 512)
            let x = Syn.roundHalf(Syn.normal(&g, 16 * f.hidden))
            checks.append(
                Check(
                    label: "ring 512, W 300: round of 16 rows at \(P)", gpu: try f.round(x, rows: 16, P: P),
                    ref: f.reference(x, rows: 16, P: P), tol: .maxRel(5e-3)))
            let xp = Syn.roundHalf(Syn.normal(&g, 128 * f.hidden))
            checks.append(
                Check(
                    label: "ring 512, W 300: prefill of 128 rows at \(P + 16)",
                    gpu: try f.prefill(xp, rows: 128, P: P + 16), ref: f.reference(xp, rows: 128, P: P + 16),
                    tol: .maxRel(5e-3)))
        }
        return checks
    }
}
