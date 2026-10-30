import Foundation
import Metal

// MARK: - LFM2 short convolution

/// LFM2/LFM3 short convolution block (`conv` in mlx-lm lfm2_moe.ShortConv): in_proj (D → 3D: B | C | x), Bx = B ⊙ x,
/// causal per-channel conv with L taps (`conv_L_cache`: 3 in LFM2, 7 in LFM3; LAB_SC_L), y = C ⊙ conv, out_proj. Same
/// ping-pong and "tape" scheme as GDNLayer: each round takes the window (W = L − 1 Bx rows) after the `nacc` rows of the
/// previous block and saves the Bx of its block, so any row of the last two rounds can be rolled back to. Kernels in
/// Shaders/yw_lfm.metal.
public final class ShortConvLayer {
    public static let maxRows = 16
    public let ctx: MetalContext
    public let layer: Int
    public let inProj, outProj: DenseQ
    public let convW: MTLBuffer
    /// Ping-pong: [W window rows | ≤ 16 Bx rows of the block] × D, fp32.
    public let hist: [MTLBuffer]
    public let projOut, y: MTLBuffer
    public var cur = 0
    public let D: Int
    /// Window rows (taps − 1).
    public let W: Int
    let vq: VQMV

    public convenience init(ctx: MetalContext, store: SafetensorsStore, config cfg: ModelConfig, layer: Int) throws {
        let pre = try store.layerPrefix(layer) + "conv."
        let d = ctx.device
        let inProj = try DenseQ(store: store, config: cfg, paths: [pre + "in_proj"], device: d)
        let outProj = try DenseQ(store: store, config: cfg, paths: [pre + "out_proj"], device: d)
        let D = cfg.hiddenSize, L = ctx.dims.convL
        let cw = try store.info(pre + "conv.weight")
        // 4 or 8 bits each (LFM2.5: 4; the LFM3-1.7B reference: 8), through the dense qmv and prefill GEMM instances.
        guard inProj.N == 3 * D, inProj.K == D, outProj.N == D, outProj.K == D, [4, 8].contains(inProj.bits),
            [4, 8].contains(outProj.bits), cw.dtype == .bf16, cw.shape.reduce(1, *) == L * D, cw.shape[0] == D
        else {
            throw SafetensorsError.format(
                "unsupported ShortConv shape (in_proj \(inProj.N)×\(inProj.K) \(inProj.bits) bits, out_proj "
                    + "\(outProj.N)×\(outProj.K) \(outProj.bits) bits, conv \(cw.shape) \(cw.dtype); expected 4 or 8 bits "
                    + "and a bf16 conv of \(D) × \(L))")
        }
        self.init(
            ctx: ctx, layer: layer, inProj: inProj, outProj: outProj,
            convW: try alignedTensor(store, pre + "conv.weight", d))
    }

    /// With the matrices and the conv (bf16 [D, L]) already on the GPU (the loader above and the per-kernel tests).
    init(ctx: MetalContext, layer: Int, inProj: DenseQ, outProj: DenseQ, convW: MTLBuffer) {
        self.ctx = ctx
        self.layer = layer
        vq = VQMV(ctx)
        self.inProj = inProj
        self.outProj = outProj
        self.convW = convW
        D = outProj.N
        W = ctx.dims.convL - 1
        let R = Self.maxRows
        hist = [ctx.buffer(length: (W + R) * D * 4), ctx.buffer(length: (W + R) * D * 4)]
        for h in hist { memset(h.contents(), 0, h.length) }
        projOut = ctx.buffer(length: R * 3 * D * 4)
        y = ctx.buffer(length: R * D * 2)
    }

    public var weightBytes: Int { inProj.bytes + outProj.bytes + (W + 1) * D * 2 }

    public func advance() { cur = 1 - cur }

    public func reset() {
        cur = 0
        memset(hist[0].contents(), 0, hist[0].length)
    }

    /// One round: x half [rows, D] → out float [rows, D] (with `accumulate`, out += conv). First folds `nacc` rows
    /// of the previous block. Calls `advance()` afterwards.
    public func encode(
        _ e: MTLComputeCommandEncoder, x: MTLBuffer, out: MTLBuffer, rows: Int, nacc: Int,
        accumulate: Bool = true, roundState: MTLBuffer? = nil
    ) throws {
        precondition(rows >= 1 && rows <= Self.maxRows && nacc >= 0 && nacc <= Self.maxRows)
        let a = cur, b = 1 - cur
        try vq.encode(e, x: x, inProj, y: projOut, rows: rows)
        var p = (UInt32(rows), UInt32(nacc), UInt32(0), UInt32(roundState == nil ? 0 : 1))
        e.setComputePipelineState(try ctx.pipeline("sconv_step"))
        for (i, buf) in [projOut, convW, hist[a], hist[b], y].enumerated() { e.setBuffer(buf, offset: 0, index: i) }
        e.setBytes(&p, length: 16, index: 5)
        e.setBuffer(roundState ?? projOut, offset: 0, index: 6)
        e.dispatchThreadgroups(
            MTLSize(width: D / 128, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        try vq.encode(e, x: y, outProj, y: out, rows: rows, accumulate: accumulate)
    }

    /// Window after folding the first `nacc` rows of the tape, written to `histOut` (snapshot) or, without it, to the
    /// next ping-pong (and advances): the state ends up as after a prefill.
    func encodeFold(_ e: MTLComputeCommandEncoder, nacc: Int, histOut: MTLBuffer? = nil) throws {
        precondition(nacc >= 0 && nacc <= Self.maxRows)
        var n4 = UInt32(W * D / 4)
        e.setComputePipelineState(try ctx.pipeline("copy_f32"))
        e.setBuffer(hist[cur], offset: nacc * D * 4, index: 0)
        e.setBuffer(histOut ?? hist[1 - cur], offset: 0, index: 1)
        e.setBytes(&n4, length: 4, index: 2)
        e.dispatchThreads(
            MTLSize(width: Int(n4), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        if histOut == nil { advance() }
    }

    /// Prefill chunk: `rows` real rows (the GEMMs use `padded`, a multiple of 64). Leaves the window in rows
    /// 0..<W of the next history (no tape) and advances.
    func encodePrefill(
        _ e: MTLComputeCommandEncoder, x: MTLBuffer, out: MTLBuffer, rows r: Int, padded M: Int,
        s: (proj: MTLBuffer, y: MTLBuffer), i8: DenseI8Scratch? = nil, i8Out: DenseI8Scratch? = nil
    ) throws {
        let a = cur, b = 1 - cur
        try inProj.encodePrefillGemm(ctx, e, x: x, y: s.proj, rows: M, i8: i8)
        var p = (UInt32(r), UInt32(0), UInt32(M), UInt32(0))
        e.setComputePipelineState(try ctx.pipeline("sconv_prefill"))
        for (i, buf) in [s.proj, convW, hist[a], s.y].enumerated() { e.setBuffer(buf, offset: 0, index: i) }
        e.setBytes(&p, length: 16, index: 4)
        e.dispatchThreadgroups(
            MTLSize(width: D / 128, height: (M + 15) / 16, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        e.setComputePipelineState(try ctx.pipeline("sconv_hist"))
        for (i, buf) in [s.proj, hist[a], hist[b]].enumerated() { e.setBuffer(buf, offset: 0, index: i) }
        e.setBytes(&p, length: 16, index: 3)
        e.dispatchThreads(
            MTLSize(width: D, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        try outProj.encodePrefillGemm(ctx, e, x: s.y, y: out, rows: M, i8: i8Out, accumulate: true)
        advance()
    }
}
