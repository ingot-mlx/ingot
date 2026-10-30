import Foundation
import Metal

// QSA (sparse attention) for Flash-Next: per-layer state (FlashQSA) and its encoding inside FlashParts (indexer,
// block selection and window captures). Split out of FlashParts.swift, which had grown past 800 lines.

/// QSA (sparse attention) of one Flash-Next attention layer: indexer (BF16 [640, 2560], norms 1 + w), window
/// of raw keys of the open block (3 rows + the round's tape, ping-pong like the GDN) and per-block keys
/// ([blocks, 128] fp32, like the KV). Semantics in zg_qsa.metal.
public final class FlashQSA {
    public let layer: Int
    public let wqk: MTLBuffer, wq: MTLBuffer, wk: MTLBuffer
    public let hist: [MTLBuffer]
    public var cur = 0
    public private(set) var bk: MTLBuffer
    public static let window = 3, topk = 512, D = 128
    /// A row uses sparse attention with more than 512 full visible blocks: position ≥ 4·513 − 1.
    public static let firstSparsePos = 4 * (topk + 1) - 1

    init(ctx: MetalContext, store: SafetensorsStore, prefix pre: String, layer: Int, maxCtx: Int, pool: KVPool? = nil)
        throws
    {
        self.layer = layer
        let ix = pre + "self_attn.indexer."
        let t = try store.info(ix + "index_qk_proj.weight")
        guard t.dtype == .bf16, t.shape == [640, ctx.dims.hidden] else {
            throw SafetensorsError.format("QSA indexer: expected bf16 [640, hidden]")
        }
        let w = ctx.device.makeBuffer(length: t.byteCount, options: .storageModeShared)!
        w.contents().copyMemory(from: store.bytes(t).baseAddress!, byteCount: t.byteCount)
        wqk = w
        func onePlus(_ n: String) throws -> MTLBuffer { try onePlusF32Tensor(ctx, store, n) }
        wq = try onePlus(ix + "q_layernorm.weight")
        wk = try onePlus(ix + "k_layernorm.weight")
        hist = (0..<2).map { _ in ctx.buffer(length: (Self.window + HybridModel.decodeRows) * Self.D * 4) }
        for h in hist { memset(h.contents(), 0, h.length) }
        // KV that grows on demand (KVPool): sized for the current capacity; the pool reallocates it as it grows.
        bk = ctx.buffer(length: ((pool?.capacity ?? maxCtx) / 4 + 8) * Self.D * 4)
        if let pool {
            pool.bytesPerPosition += Self.D
            pool.resizers.append { [unowned self, unowned pool] cap in
                let nb = ctx.buffer(length: (cap / 4 + 8) * Self.D * 4)
                memset(nb.contents(), 0, nb.length)
                let old = self.bk
                pool.blit {
                    $0.copy(from: old, sourceOffset: 0, to: nb, destinationOffset: 0, size: min(old.length, nb.length))
                }
                self.bk = nb
            }
        }
    }
    public func advance() { cur = 1 - cur }
    public func reset() {
        cur = 0
        memset(hist[0].contents(), 0, hist[0].length)
    }
    public var windowBytes: Int { Self.window * Self.D * 4 }
    public var weightBytes: Int { wqk.length + wq.length + wk.length }

    /// Window after folding the first `nacc` rows of the tape (like FlashPLE.encodeFold).
    func encodeFold(_ ctx: MetalContext, _ e: MTLComputeCommandEncoder, nacc: Int, histOut: MTLBuffer? = nil) throws {
        var n4 = UInt32(Self.window * Self.D / 4)
        e.setComputePipelineState(try ctx.pipeline("copy_f32"))
        e.setBuffer(hist[cur], offset: nacc * Self.D * 4, index: 0)
        e.setBuffer(histOut ?? hist[1 - cur], offset: 0, index: 1)
        e.setBytes(&n4, length: 4, index: 2)
        e.dispatchThreads(
            MTLSize(width: Int(n4), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        if histOut == nil { advance() }
    }
}

extension FlashParts {
    /// Window of `W` rows × `width` after row `r` of the chunk (from `src`, or from `hprev` for earlier rows) into `out`.
    func encodeWinCapture(
        _ e: MTLComputeCommandEncoder, src: MTLBuffer, hprev: MTLBuffer, out: MTLBuffer, row r: Int, window W: Int,
        width: Int
    ) throws {
        try threads(
            e, "fn_win_capture", W * width, [(src, 0), (hprev, 0), (out, 0)], [UInt32(r), UInt32(W), UInt32(width), 0])
    }
}

/// QsaMParams from zg_qsa.metal.
struct QsaMParams {
    var rows: UInt32, pos0: UInt32, nb: UInt32, nacc: UInt32
    var eps: Float, theta: Float, flags: UInt32, pad: UInt32 = 0
}

extension FlashParts {
    /// QSA indexer projection (br fp32 [rows, D] → qsaQK [rows, 640]).
    func encodeIndexer(_ e: MTLComputeCommandEncoder, _ q: FlashQSA, rows: Int) throws {
        // In prefill, on NAX with x split into 3 bf16 chunks (159.5 → 19.7 ms per chunk; meets the fidelity bar).
        if rows > 16 {
            var p = (UInt32(rows), UInt32(640), UInt32(hidden), UInt32(hidden), UInt32(640), UInt32(0))
            e.setComputePipelineState(try ctx.pipeline("fn_f32bf16_mm_nax3"))
            e.setBuffer(br, offset: 0, index: 0); e.setBuffer(q.wqk, offset: 0, index: 1)
            e.setBuffer(qsaQK, offset: 0, index: 2)
            e.setBytes(&p, length: 24, index: 3)
            e.dispatchThreadgroups(
                MTLSize(width: 640 / 64, height: (rows + 63) / 64, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            return
        }
        try bf16mm(e, x: br, w: q.wqk, y: qsaQK, M: rows, N: 640, K: hidden, ldx: hidden, ldy: 640)
    }

    /// QSA indexer over the block input (fp32 in `br`, written by hcMix with `f32`) for `rows` rows at
    /// positions P… (P from st[0] with `state`): queries, raw keys into the window and keys of the blocks that get
    /// completed. With `sparse`, also scores and selection (per-row bitmap in `qbm`), leaving the mask in `attn`.
    /// `prefill`: no tape (the window keeps the chunk's last 3 positions). `cut` (`--round-trace 3`): cuts
    /// the encoder before each piece and returns the next one (without it, one encoder from start to finish).
    @discardableResult
    func encodeQSA(  // swiftlint:disable:this function_parameter_count - hot path: every attention layer and round
        _ e0: MTLComputeCommandEncoder, _ q: FlashQSA, attn: AttentionLayer, rows: Int, pos: Int, nacc: Int,
        state: MTLBuffer?, prefill: Bool, sparse: Bool, theta: Float, skipIndexer: Bool = false,
        cut: ((MTLComputeCommandEncoder, String) -> MTLComputeCommandEncoder)? = nil
    ) throws -> MTLComputeCommandEncoder {
        let D = FlashQSA.D
        var e = e0
        if !skipIndexer {
            if let cut { e = cut(e, "qsa.idx") }
            try encodeIndexer(e, q, rows: rows)
        }
        if let cut { e = cut(e, "qsa.prep") }
        let nb = (pos + rows) / 4 + 1
        var p = QsaMParams(
            rows: UInt32(rows), pos0: UInt32(pos), nb: UInt32(nb), nacc: UInt32(prefill ? 0 : nacc), eps: eps,
            theta: theta, flags: (state != nil ? 2 : 0) | (prefill ? 0 : 4))
        let st = state ?? qsaQK
        e.setComputePipelineState(try ctx.pipeline("qsa_m_prep"))
        for (i, b) in [qsaQK, q.wq, qsaQ, qsaRK].enumerated() { e.setBuffer(b, offset: 0, index: i) }
        e.setBytes(&p, length: MemoryLayout<QsaMParams>.size, index: 4)
        e.setBuffer(st, offset: 0, index: 5)
        e.dispatchThreadgroups(
            MTLSize(width: 5, height: rows, depth: 1), threadsPerThreadgroup: MTLSize(width: D, height: 1, depth: 1))
        let a = q.cur, b = 1 - q.cur
        e.setComputePipelineState(try ctx.pipeline("qsa_m_blocks"))
        for (i, bf) in [qsaRK, q.hist[a], q.wk, q.bk].enumerated() { e.setBuffer(bf, offset: 0, index: i) }
        e.setBytes(&p, length: MemoryLayout<QsaMParams>.size, index: 4)
        e.setBuffer(st, offset: 0, index: 5)
        e.dispatchThreadgroups(
            MTLSize(width: rows / 4 + 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: D, height: 1, depth: 1))
        e.setComputePipelineState(try ctx.pipeline("qsa_m_hist"))
        for (i, bf) in [qsaRK, q.hist[a], q.hist[b]].enumerated() { e.setBuffer(bf, offset: 0, index: i) }
        e.setBytes(&p, length: MemoryLayout<QsaMParams>.size, index: 3)
        e.setBuffer(st, offset: 0, index: 4)
        e.dispatchThreads(
            MTLSize(width: (prefill ? 3 : 3 + rows) * D, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        q.advance()
        attn.blockList = nil
        guard sparse else { attn.qsaMask = nil; return e }
        let nw = (nb + 31) / 32
        precondition(nw * 4 * rows <= qsaBM.length && nb <= qsaNBMax)
        // Scores and selection in batches of rows (one batch's scores: rows × blocks, fp32).
        let RB = max(1, min(rows, qsaS.length / (nb * 4)))
        var r0 = 0
        while r0 < rows {
            let n = min(RB, rows - r0)
            var pb = p
            pb.rows = UInt32(n); pb.pos0 = UInt32(pos + r0)
            if state != nil { precondition(r0 == 0) }
            if let cut { e = cut(e, "qsa.score") }
            e.setComputePipelineState(try ctx.pipeline("qsa_m_score"))
            e.setBuffer(qsaQ, offset: r0 * 4 * D * 4, index: 0); e.setBuffer(q.bk, offset: 0, index: 1)
            e.setBuffer(qsaS, offset: 0, index: 2)
            e.setBytes(&pb, length: MemoryLayout<QsaMParams>.size, index: 3)
            e.setBuffer(st, offset: 0, index: 4)
            // One simdgroup per block for up to 16 rows (QSA_RT in zg_qsa.metal).
            e.dispatchThreadgroups(
                MTLSize(width: (nb + 7) / 8, height: (n + 15) / 16, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            if let cut { e = cut(e, "qsa.select") }
            e.setComputePipelineState(try ctx.pipeline("qsa_m_select"))
            e.setBuffer(qsaS, offset: 0, index: 0); e.setBuffer(qsaBM, offset: r0 * nw * 4, index: 1)
            e.setBytes(&pb, length: MemoryLayout<QsaMParams>.size, index: 2)
            e.setBuffer(st, offset: 0, index: 3)
            e.dispatchThreadgroups(
                MTLSize(width: n, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1)
            )
            r0 += n
        }
        attn.qsaMask = (qsaBM, nw)
        // In a round, the bitmap becomes lists (one per row) and attention walks the lists.
        if !prefill {
            if let cut { e = cut(e, "qsa.list") }
            let sel = BitmapSelector(bitmap: qsaBM, words: nw, lists: qsaLists)
            attn.blockList = try sel.encodeBlockList(e, rows: rows, pos: pos, state: state)
        }
        return e
    }
}
