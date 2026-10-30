import Foundation
import Metal
import Testing

@testable import Ingot

// Affine 4-bit weights with groups of 32 (families with q4_0 QAT) on the dense paths, against the
// CPU: round qmv (vnax_qmv<…, VLoad4<32>> per NAX, with nPrefix, and the scalar vqmv), prefill GEMM
// (qmm_t_nax3<…, 4, 32>, 64×64 and 64×128 tiles, fp32, accumulated, half), embedding (embed_rows_g32), rotated W8A8
// of --int8 (w4_i8rot_g32, with and without precomputed scales). On the Granite 4.1 library (g32 from its config).
extension KernelCase {
    static let q4G32 = KernelCase(
        id: "q4-g32", model: "granite-4.1-3b-qkvb",
        catches:
            "groups of 32 in the dense 4-bit matrices: group index (kk / 32) and scale stride (K / 32) in "
            + "NAX and scalar qmv, prefill GEMM, embedding and W8A8 conversion; and that the row "
            + "does not depend on the others"
    ) { ctx, g in
        var checks: [Check] = []
        // round qmv (vnax_qmv_g32_nf1_sg32): 16, 5 and 1 rows; K not a multiple of 64·32 (uneven split-K).
        for (N, K, rows) in [(64, 2048, 16), (48, 2048, 5), (32, 1536, 1), (256, 5120, 9)] {
            let (m, w) = Syn.q4(ctx, N: N, K: K, &g, group: 32)
            let x = Syn.roundHalf(Syn.normal(&g, rows * K))
            checks.append(
                Check(
                    label: "vnax_qmv g32 \(N)×\(K), \(rows) rows", gpu: try qmv(ctx, m, x, rows: rows),
                    ref: CPU.matmulT(x, w, M: rows, N: N, K: K), tol: .maxRel(1e-3)))
        }
        // nPrefix (first 32 outputs only, compact) and the scalar qmv (vqmv_b4_g32: naxFromRows = 0), half and fp32.
        do {
            let N = 64, K = 2048, rows = 6
            let (m, w) = Syn.q4(ctx, N: N, K: K, &g, group: 32)
            let x = Syn.roundHalf(Syn.normal(&g, rows * K))
            let ref = CPU.matmulT(x, w, M: rows, N: N, K: K)
            let xb = Syn.halfBuffer(ctx, x), yp = ctx.buffer(length: rows * 32 * 4),
                ys = ctx.buffer(length: rows * N * 4)
            let yf = ctx.buffer(length: rows * N * 4)
            var scalar = VQMV(ctx)
            scalar.naxFromRows = 0
            try ctx.run { e in
                try VQMV(ctx).encode(e, x: xb, m, y: yp, rows: rows, nPrefix: 32)
                try scalar.encode(e, x: xb, m, y: ys, rows: rows)
                try scalar.encode(e, x: ctx.buffer(x), xFloat: true, m, y: yf, rows: rows)
            }
            let refP = (0..<rows).flatMap { r in ref[(r * N)..<(r * N + 32)] }
            checks.append(
                Check(label: "vnax_qmv g32 nPrefix 32", gpu: Syn.floats(yp, rows * 32), ref: refP, tol: .maxRel(1e-3)))
            checks.append(
                Check(label: "vqmv_b4_g32 (half)", gpu: Syn.floats(ys, rows * N), ref: ref, tol: .maxRel(1e-3)))
            checks.append(
                Check(label: "vqmv_b4_g32 (fp32)", gpu: Syn.floats(yf, rows * N), ref: ref, tol: .maxRel(1e-3)))
        }
        let (mi, _) = Syn.q4(ctx, N: 256, K: 2048, &g, group: 32)
        checks.append(try rowInvariance("vnax_qmv g32 256×2048", K: 2048, &g) { x, r in try qmv(ctx, mi, x, rows: r) })

        // Prefill GEMM: 64×64 bk32 (K < 4096) and 64×128 bk64 (K ≥ 4096); fp32, accumulated fp32 and half.
        for (N, K) in [(128, 2048), (256, 4096)] {
            let M = 64
            let (m, w) = Syn.q4(ctx, N: N, K: K, &g, group: 32)
            let x = Syn.roundHalf(Syn.normal(&g, M * K))
            let ref = CPU.matmulT(x, w, M: M, N: N, K: K)
            let y0 = Syn.normal(&g, M * N)
            let xb = Syn.halfBuffer(ctx, x), yf = ctx.buffer(length: M * N * 4), ya = ctx.buffer(y0)
            let yh = ctx.buffer(length: M * N * 2)
            try ctx.run { e in
                try m.encodeQmmF32(ctx, e, x: xb, y: yf, rows: M)
                try m.encodeQmmF32(ctx, e, x: xb, y: ya, rows: M, accumulate: true)
                try m.encodeQmmF32(ctx, e, x: xb, y: yh, rows: M, halfOut: true)
            }
            let tag = "qmm_t_nax3 g32 \(M)×\(N)×\(K)"
            checks.append(Check(label: tag, gpu: Syn.floats(yf, M * N), ref: ref, tol: .maxRel(1e-3)))
            checks.append(
                Check(
                    label: "\(tag) accumulated", gpu: Syn.floats(ya, M * N), ref: zip(y0, ref).map { $0 + $1 },
                    tol: .maxRel(1e-3)))
            checks.append(
                Check(label: "\(tag) half", gpu: readHalf(yh, count: M * N), ref: ref, tol: .maxRel(2e-3)))
        }

        // Embedding (embed_rows_g32): exact rows (fp32 and half).
        do {
            let V = 40, K = 2048, tok: [UInt32] = [3, 0, 39, 17, 17, 8]
            let (m, w) = Syn.q4(ctx, N: V, K: K, &g, group: 32)
            let of = ctx.buffer(length: tok.count * K * 4), oh = ctx.buffer(length: tok.count * K * 2)
            try ctx.run { e in try m.encodeEmbed(ctx, e, tok: ctx.buffer(tok), outF: of, outH: oh, rows: tok.count) }
            let ref = tok.flatMap { t in w[(Int(t) * K)..<(Int(t) * K + K)] }
            checks.append(
                Check(label: "embed_rows_g32 fp32", gpu: Syn.floats(of, tok.count * K), ref: ref, tol: .bitExact))
            checks.append(
                Check(
                    label: "embed_rows_g32 half", gpu: readHalf(oh, count: tok.count * K), ref: Syn.roundHalf(ref),
                    tol: .bitExact))
        }

        // Rotated W8A8 of --int8 with g32 weights (w4_i8rot_g32): against its emulation and, with slack, the exact one.
        for (K, N, pre) in [(2048, 256, false), (2048, 256, true), (1536, 256, false)] {
            let M = 64
            let (m, w) = Syn.q4(ctx, N: N, K: K, &g, group: 32)
            var x = Syn.normal(&g, M * K)
            for ch in [7, 300, K - 5] { for r in 0..<M { x[r * K + ch] *= 20 } }
            x = Syn.roundHalf(x)
            let sc = DenseI8Scratch(ctx: ctx, maxNK: N * K, maxN: N, maxRows: M, maxK: K)
            let y = ctx.buffer(length: M * N * 4)
            if pre { try ctx.run { e in try m.precomputeRowScales(ctx, e, scratch: sc) } }
            try ctx.run { e in try m.encodeQmmI8(ctx, e, x: Syn.halfBuffer(ctx, x), y: y, rows: M, scratch: sc) }
            let lb = K / 32 % 32 == 0 ? 3 : 0
            let tag = "qmm_i8 g32 \(M)×\(N)×\(K)\(pre ? " (escalas precalculadas)" : "")"
            checks.append(
                Check(
                    label: "\(tag) against the emulated W8A8", gpu: Syn.floats(y, M * N),
                    ref: CPU.w8a8(x, w, M: M, N: N, K: K, lb: lb), tol: .rms(2e-3)))
            checks.append(
                Check(
                    label: "\(tag) against the exact one", gpu: Syn.floats(y, M * N),
                    ref: CPU.matmulT(x, w, M: M, N: N, K: K),
                    tol: .rms(0.05)))
        }
        // Size in memory: scale and bias per group of 32 (double that of 64).
        let (mb, _) = Syn.q4(ctx, N: 64, K: 2048, &g, group: 32)
        checks.append(
            Check(
                label: "DenseQ.bytes g32", gpu: [Float(mb.bytes)], ref: [Float(64 * 2048 / 2 + 2 * 64 * 64 * 2)],
                tol: .bitExact))
        return checks
    }
}
