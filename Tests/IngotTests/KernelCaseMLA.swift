import Foundation
import Metal
import Testing

@testable import Ingot

// GLM-4.7-Flash's MLA over the latent cache: the attention kernels with the 576 head (one KV head [k_pe | c]
// for 20 query heads, the block Hadamard 64 ⊕ 512 of the int8 K with two scales, V read from K's latent part,
// mattn_i8 in decode) and the per-head matmuls of embed_q / unembed_out (zm_mla.metal) against the CPU. The layer end
// to end is covered by toy-glm against mlx-lm fp32 (fidelity, scripts/ci/fidelity.sh) and its golden.

extension KernelCase {
    static let mla: [KernelCase] = [mlaMmaKN, mlaAttn576, mlaHeads, mlaDecompressed]

    /// A probe kernel (one simdgroup) over the library's own nax_load_i8 and nax_mma2_kn: C = A · B with B read
    /// key-major from an int8 buffer of stride 576, as the P·V of the latent head reads V from kc.
    static let mmaKNProbe = """
        [[kernel]] void mla_mma_kn_probe(const device half* a [[buffer(0)]], const device char* b [[buffer(1)]],
                                         device float* c [[buffer(2)]], ushort lane [[thread_index_in_simdgroup]]) {
            const short2 co = nax_coord(lane);
            nax_hfrag fa, b0, b1;
            nax_load(fa, a, 16, co);
            nax_load_i8(b0, b, 576, co);
            nax_load_i8(b1, b + 16, 576, co);
            nax_ffrag c0 = 0, c1 = 0;
            nax_mma2_kn(c0, c1, fa, b0, b1);
            for (int i = 0; i < 2; i++) for (int j = 0; j < 4; j++) {
                c[(co.x + 8 * i) * 32 + co.y + j] = c0[4 * i + j];
                c[(co.x + 8 * i) * 32 + 16 + co.y + j] = c1[4 * i + j];
            }
        }
        """

    static let mlaMmaKN = KernelCase(
        id: "mla-mma-kn", model: "glm-4.7-flash",
        catches:
            "the fragment layout of nax_mma2_kn's right operand (16×32×16 matmul2d, not transposed) loaded key-major "
            + "with nax_load_i8: the P·V of MLA's latent head reads V from kc this way"
    ) { ctx, g in
        let probe = try MetalContext(dims: ctx.dims, extraSources: [mmaKNProbe])
        let a = Syn.roundHalf(Syn.normal(&g, 16 * 16))
        var b = [Int8](repeating: 0, count: 16 * 576)
        for k in 0..<16 { for n in 0..<32 { b[k * 576 + n] = Int8(Int(g.next() % 255) - 127) } }
        let c = probe.buffer(length: 16 * 32 * 4)
        let ab = Syn.halfBuffer(probe, a), bb = probe.buffer(b)
        try probe.run { e in
            e.setComputePipelineState(try probe.pipeline("mla_mma_kn_probe"))
            for (i, x) in [ab, bb, c].enumerated() { e.setBuffer(x, offset: 0, index: i) }
            e.dispatchThreadgroups(
                MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        }
        var ref = [Float](repeating: 0, count: 16 * 32)
        for m in 0..<16 {
            for n in 0..<32 { for k in 0..<16 { ref[m * 32 + n] += a[m * 16 + k] * Float(b[k * 576 + n]) } }
        }
        return [Check(label: "A·B, B key-major", gpu: Syn.floats(c, 16 * 32), ref: ref, tol: .maxRel(1e-3))]
    }

    static let mlaAttn576 = KernelCase(
        id: "mla-attn576", model: "glm-4.7-flash",
        catches:
            "MLA's latent head of 576 in the attention kernels: vattn_prework_i8 (576 threads, 18 simdgroups, the "
            + "block Hadamard H₆₄ ⊕ H₅₁₂ of at_fwht, RoPE on the first 64, K only with one scale for k_pe in ks and "
            + "one for c in vs), dattn_i8 and pattn_i8 with G = 20 (two groups of 16 queries per row; S of k_pe and "
            + "of c apart, each with its scale), P·V over the 512 latent dims read key-major from kc (nax_mma2_kn) "
            + "in prefill and decode, the decode kernel mattn_i8 (8 simdgroups, steps of 256 keys) with the latent "
            + "decode chunk of 256 (3 chunks at 700) and of 1024 (3 chunks of 4 steps at 2100), the output rotated "
            + "back by H₅₁₂ (vattn_reduce, mla_out_unrotate; dims 0…63 zero) and q_quant_i8 (18 values per lane) "
            + "against the CPU, and the per-row invariance of the decode round"
    ) { ctx, g in
        var checks = try attnChecks(ctx, &g, prefill: true)
        // Long chunk (C 1024, the latent chunk from 8K of prompt): several 256-key steps per chunk.
        let fl = try AttnFixture(ctx, positions: 2100, &g, maxCtx: 4096)
        fl.a.fixedChunk = 1024
        let xl = Syn.roundHalf(Syn.normal(&g, 16 * fl.hidden))
        checks.append(
            Check(
                label: "round of 16 rows at 2100 (C 1024)", gpu: try fl.round(xl, rows: 16, P: 2100),
                ref: fl.reference(xl, rows: 16, P: 2100), tol: .maxRel(5e-3)))
        checks.append(
            try rowInvariance("round at 2100 (C 1024)", K: fl.hidden, &g) { x, r in
                try fl.round(x, rows: r, P: 2100)
            })
        let f8 = try AttnFixture(ctx, positions: 300, &g)
        let x8 = Syn.roundHalf(Syn.normal(&g, 64 * f8.hidden))
        checks.append(
            Check(
                label: "prefill with Q·Kᵀ int8", gpu: try f8.prefill(x8, rows: 64, P: 300, qk8: true),
                ref: f8.reference(x8, rows: 64, P: 300, qk8: true), tol: .maxRel(5e-3)))
        return checks
    }

    static let mlaHeads = KernelCase(
        id: "mla-heads", model: "glm-4.7-flash",
        catches:
            "the per-head matmuls of the latent MLA (vnax_qmv_heads, qmm_t_nax3_heads): per-head offsets of the weights, "
            + "scales and biases, of the input (q_nope packed; the latent dims 64…575 of each 576 head) and of the "
            + "output (into the 576 q heads of the projection row; packed half for o_proj), fp32 and half outputs, "
            + "decode (1, 5 and 16 rows) and prefill (128 rows)"
    ) { ctx, g in
        let H = 20, hd = 576, rope = 64
        var checks: [Check] = []
        // embed_q: q_nope (H × 192, packed) → q_lat in columns h·576 + 64… of a row of (H + 2)·576 (fp32).
        // unembed_out: dims 64…575 of each 576 head of the attention output → H × 256 packed (half).
        for (tag, N, K, halfOut) in [("embed_q", 512, 192, false), ("unembed_out", 256, 512, true)] {
            let (m, w) = Syn.q4(ctx, N: H * N, K: K, &g)
            let ldx = halfOut ? H * hd : H * K, xh = halfOut ? hd : K, xOff = halfOut ? rope : 0
            let ldy = halfOut ? H * N : (H + 2) * hd, yh = halfOut ? N : hd, yOff = halfOut ? 0 : rope
            for (rows, prefill) in [(1, false), (5, false), (16, false), (128, true)] {
                let x = Syn.roundHalf(Syn.normal(&g, rows * ldx))
                let xb = Syn.halfBuffer(ctx, x), y = ctx.buffer(length: rows * ldy * (halfOut ? 2 : 4))
                memset(y.contents(), 0, y.length)
                try ctx.run { e in
                    try MLAProjection.encodeHeads(
                        ctx, e, m, heads: H,
                        x: .init(buf: xb, offset: xOff * 2, ld: ldx, perHead: xh, half: true),
                        y: .init(buf: y, offset: yOff * (halfOut ? 2 : 4), ld: ldy, perHead: yh, half: halfOut),
                        rows: rows, prefill: prefill)
                }
                let got = halfOut ? readHalf(y, count: rows * ldy) : Syn.floats(y, rows * ldy)
                var gpu: [Float] = [], ref: [Float] = []
                for h in 0..<H {
                    var xh0 = [Float]()
                    let wh = Array(w[(h * N * K)..<((h + 1) * N * K)])
                    for t in 0..<rows { xh0 += x[(t * ldx + xOff + h * xh)..<(t * ldx + xOff + h * xh + K)] }
                    var r = CPU.matmulT(xh0, wh, M: rows, N: N, K: K)
                    if halfOut { r = Syn.roundHalf(r) }
                    ref += r
                    for t in 0..<rows { gpu += got[(t * ldy + yOff + h * yh)..<(t * ldy + yOff + h * yh + N)] }
                }
                checks.append(
                    Check(
                        label: "\(tag) \(rows) rows\(prefill ? " (prefill)" : "")", gpu: gpu, ref: ref,
                        tol: .maxRel(halfOut ? 5e-3 : 1e-3)))
            }
        }
        return checks
    }

    /// Synthetic inputs of the decompressed prefill attention (MLAProjection.encodeDecompressed) over P + rows keys of
    /// the latent cache: GPU output [rows, H · 256] (blocks of `block` keys) and its CPU reference.
    static func mlaDecRun(
        _ ctx: MetalContext, _ g: inout SplitMix64, P: Int, rows: Int, block: Int
    ) throws -> (gpu: [Float], ref: [Float]) {
        let H = 20, hd = 576, nope = 192, vd = 256, R = 512, keys = P + rows, ld = (keys + 127) / 128 * 128
        let scale = 1 / Float(256).squareRoot()
        let (eq, ew) = Syn.q4(ctx, N: H * R, K: nope, &g)  // embed_q [H · 512, 192]
        let (uq, uw) = Syn.q4(ctx, N: H * vd, K: R, &g)  // unembed_out [H · 256, 512]
        var k8 = [Int8](repeating: 0, count: ld * hd)
        for t in 0..<keys { for i in 0..<hd { k8[t * hd + i] = Int8(Int(g.next() % 255) - 127) } }
        let ks = (0..<ld).map { _ in 0.011 + 0.005 * g.uniform() },
            vs = (0..<ld).map { _ in 0.0005 + 0.0005 * g.uniform() }
        let qn = Syn.roundHalf(Syn.normal(&g, rows * H * nope, sd: 0.3))
        let qp = Syn.roundHalf(Syn.normal(&g, rows * H * hd))  // only dims 0…63 of each head are read
        let s = MLAProjection.Scratch()
        s.qNope = Syn.halfBuffer(ctx, qn)
        s.out = ctx.buffer(length: rows * H * vd * 2)
        MLAProjection.decBuffers(ctx, s, heads: H, rows: rows, block: block)
        let qb = Syn.halfBuffer(ctx, qp), kcb = ctx.buffer(k8), ksb = ctx.buffer(ks), vsb = ctx.buffer(vs)
        try ctx.run { e in
            try MLAProjection.encodeDecompressed(
                ctx, e, embedQ: eq, unembedOut: uq, scratch: s, heads: H,
                .init(q: qb, kc: kcb, ks: ksb, vs: vsb, ldkv: ld, pos: P, rows: rows, scale: scale), block: block)
        }
        let gpu = readHalf(s.out!, count: rows * H * vd)
        // CPU: c = H₅₁₂ (vs · c8), k_nope = embed_qᵀ c, v = unembed_out c, k_pe = ks · k8_pe (rotated as q_pe).
        var c = [Float](repeating: 0, count: keys * R)
        for t in 0..<keys {
            var v = (0..<R).map { Float(k8[t * hd + 64 + $0]) * vs[t] }
            CPU.fwht(&v)
            for n in 0..<R { c[t * R + n] = v[n] / Float(R).squareRoot() }
        }
        var ref = [Float](repeating: 0, count: rows * H * vd)
        for h in 0..<H {
            let e = Array(ew[(h * R * nope)..<((h + 1) * R * nope)])
            var et = [Float](repeating: 0, count: nope * R)  // embed_q_hᵀ [192, 512]
            for n in 0..<R { for k in 0..<nope { et[k * R + n] = e[n * nope + k] } }
            let kn = CPU.matmulT(c, et, M: keys, N: nope, K: R)
            let v = CPU.matmulT(c, Array(uw[(h * vd * R)..<((h + 1) * vd * R)]), M: keys, N: vd, K: R)
            for t in 0..<rows {
                var sc = [Float](repeating: -.infinity, count: keys)
                for j in 0...(P + t) {
                    var a: Float = 0
                    for k in 0..<nope { a += qn[(t * H + h) * nope + k] * kn[j * nope + k] }
                    var b: Float = 0
                    for k in 0..<64 { b += qp[(t * H + h) * hd + k] * Float(k8[j * hd + k]) }
                    sc[j] = (a + b * ks[j]) * scale
                }
                let m = sc.max()!
                let p = sc.map { exp($0 - m) }, l = p.reduce(0, +)
                for d in 0..<vd {
                    var o: Float = 0
                    for j in 0...(P + t) { o += p[j] * v[j * vd + d] }
                    ref[(t * H + h) * vd + d] = o / l
                }
            }
        }
        return (gpu, ref)
    }

    static let mlaDecompressed = KernelCase(
        id: "mla-dec", model: "glm-4.7-flash",
        catches:
            "the decompressed prefill attention of GLM's long chunks (zn_mla_dec.metal): W' = [embed_qᵀ; unembed_out] · "
            + "H₅₁₂ dequantized per head, the decompression GEMM over the int8 latent with its scale vs (keys past "
            + "the chunk clamped), the 256-head attention with q_nope · k_nope + ks · (q_pe · k8_pe), the causal "
            + "mask, the running state across key blocks (first, middle and last) and the output straight in o_proj's "
            + "layout"
    ) { ctx, g in
        var checks: [Check] = []
        // One block; two (first and last); four (middle blocks resume and store the state); rows not a multiple of 32.
        for (P, rows, block) in [(64, 64, 256), (200, 96, 256), (300, 128, 128), (130, 100, 128)] {
            let (gpu, ref) = try mlaDecRun(ctx, &g, P: P, rows: rows, block: block)
            checks.append(
                Check(label: "\(rows) rows at \(P), blocks of \(block)", gpu: gpu, ref: ref, tol: .maxRel(5e-3)))
        }
        return checks
    }
}
