import Foundation
import Metal

/// GPU timeline by piece: each piece goes in its own compute encoder with timestamps at the start and at the
/// end (MTLCounterSampleBuffer at stage boundaries, the only thing the M6 supports). Separate encoders add a
/// little overhead of their own, so it is for splitting up the time, not for measuring the total.
public final class GPUTimeline {
    let ctx: MetalContext
    let sb: MTLCounterSampleBuffer
    let cb: MTLCommandBuffer
    var names: [String] = []
    /// Prepended to the names of the segments that follow (the prefill sets the layer kind: `conv.`, `full.`…).
    var prefix = ""
    public init(ctx: MetalContext, maxSegments: Int = 2048) throws {
        self.ctx = ctx
        guard let set = ctx.device.counterSets?.first(where: { $0.name == MTLCommonCounterSet.timestamp.rawValue })
        else {
            throw SafetensorsError.io("no timestamp counter")
        }
        let d = MTLCounterSampleBufferDescriptor()
        d.counterSet = set
        d.sampleCount = 2 * maxSegments
        d.storageMode = .shared
        sb = try ctx.device.makeCounterSampleBuffer(descriptor: d)
        cb = ctx.queue.makeCommandBuffer()!
    }

    public func seg(_ name: String, _ f: (MTLComputeCommandEncoder) throws -> Void) throws {
        let pd = MTLComputePassDescriptor()
        let i = names.count
        pd.sampleBufferAttachments[0].sampleBuffer = sb
        pd.sampleBufferAttachments[0].startOfEncoderSampleIndex = 2 * i
        pd.sampleBufferAttachments[0].endOfEncoderSampleIndex = 2 * i + 1
        let e = cb.makeComputeCommandEncoder(descriptor: pd)!
        try f(e)
        e.endEncoding()
        names.append(prefix + name)
    }

    /// Runs and returns (ms per name in order of appearance, GPU ms of the command buffer, ms between segments).
    public func run() throws -> (pieces: [(String, Double)], total: Double, gaps: Double) {
        cb.commit()
        cb.waitUntilCompleted()
        if let e = cb.error { throw e }
        let data = try sb.resolveCounterRange(0..<(2 * names.count))!
        let ts = data.withUnsafeBytes { Array($0.bindMemory(to: UInt64.self)) }
        let total = (cb.gpuEndTime - cb.gpuStartTime) * 1e3
        let span = Double(ts[2 * names.count - 1] - ts[0])
        let scale = total / span  // ms per tick (the counter is not in ns on every machine)
        var acc: [String: Double] = [:], order: [String] = []
        var gaps = 0.0
        for (i, n) in names.enumerated() {
            if acc[n] == nil { order.append(n) }
            acc[n, default: 0] += Double(ts[2 * i + 1] - ts[2 * i]) * scale
            if i > 0 { gaps += Double(Int64(ts[2 * i]) - Int64(ts[2 * i - 1])) * scale }
        }
        return (order.map { ($0, acc[$0]!) }, total, gaps)
    }
}

extension HybridModel {
    /// A verification of B rows (like encodeVerify) split into pieces on the timeline.
    public func timelineVerify(rows B: Int, drafter: DFlash2Drafter?) throws -> (
        pieces: [(String, Double)], total: Double, gaps: Double
    ) {
        guard spec.block.plainSkeleton else {
            throw SafetensorsError.format(
                "timeline: block pieces (branch epilogue, embedding or logits) not profiled")
        }
        let t = try GPUTimeline(ctx: ctx)
        try t.seg("embed") { try embedRows($0, tok: tok, out: h, rows: B) }
        for (i, l) in layers.enumerated() {
            try t.seg("norm") { try norm($0, h, l.n1, xn, rows: B) }
            switch l.mixer {
            case .gdn(let g):
                for p in GDNLayer.Piece.allCases {
                    try t.seg("gdn." + p.rawValue) {
                        try g.encode($0, x: xn, out: h, rows: B, nacc: nacc, accumulate: true, only: p)
                    }
                }
                g.advance()
            case .conv(let c):
                try t.seg("conv") { try c.encode($0, x: xn, out: h, rows: B, nacc: nacc, accumulate: true) }
                c.advance()
            case .attn(let a):
                a.pos = P
                for p in AttentionLayer.Piece.allCases {
                    try t.seg("attn." + p.rawValue) {
                        try a.encode($0, x: xn, out: h, rows: B, accumulate: true, only: p)
                    }
                }
            case .none: preconditionFailure("timeline: layers without mixer not profiled")
            }
            try t.seg("norm") { try norm($0, h, l.n2, xn, rows: B) }
            if let f = l.dense {
                for p in DenseMLP.Piece.allCases {
                    try t.seg("ffn." + p.rawValue) { try f.encode($0, x: xn, out: h, rows: B, only: p) }
                }
            } else {
                for p in MoELayer.Piece.allCases {
                    try t.seg("moe." + p.rawValue) {
                        try l.moe.encode($0, x: xn, out: h, rows: B, accumulate: true, only: p)
                    }
                }
            }
            if let dr = drafter, let j = dr.targetLayers.firstIndex(of: i) {
                try t.seg("capture") { try capture($0, h, into: dr.caps, width: dr.ctxWidth, slot: j, rows: B) }
            }
        }
        try t.seg("norm") { try norm($0, h, finalNorm, xn, rows: B) }
        try t.seg("lm_head") {
            try vqHead.encode($0, x: xn, lmHead, y: logits, rows: B)
            try encodeLogitsPost($0, rows: B)
        }
        try t.seg("argmax") { try argmax($0, rows: B) }
        return try t.run()
    }
}

/// With a timeline, each piece goes in its own segment; without it, in the usual encoder.
@inline(__always) func tseg(
    _ tl: GPUTimeline?, _ e: MTLComputeCommandEncoder?, _ name: String,
    _ f: (MTLComputeCommandEncoder) throws -> Void
) throws {
    if let tl { try tl.seg(name, f) } else { try f(e!) }
}

extension HybridModel.Layer {
    /// Layer kind in the prefill timeline's names: `gdn`, `conv`, `full` or `window` attention, or `mlp` (no mixer).
    var timelineKind: String {
        switch mixer {
        case .gdn: return "gdn"
        case .conv: return "conv"
        case .attn(let a): return a.window > 0 ? "window" : "full"
        case .none: return "mlp"
        }
    }
}

extension HybridModel {
    /// One prefill chunk of `rows` rows from position `p0` by the real prefill path (encodePrefillGroup), split into
    /// pieces on the timeline, with each name prefixed by its layer kind. Positions p0..<p0 + rows are rewritten with
    /// filler tokens (for timing only; P is restored, the cache contents there are not). `plain` is the GPU ms of the
    /// same chunk in a single encoder, to bound what the timeline's separate encoders add.
    public func timelinePrefill(rows: Int, at p0: Int) throws -> (
        pieces: [(String, Double)], total: Double, gaps: Double, plain: Double
    ) {
        guard fn == nil, expertStream == nil else {
            throw SafetensorsError.format("timeline: prefill with experts on the SSD not profiled")
        }
        precondition(rows > 0 && rows <= chunkRows && p0 + rows <= maxCtx)
        let P0 = P
        defer {
            P = P0
            for l in layers { if case .attn(let a) = l.mixer { a.pos = P0 } }
        }
        let ids = [UInt32](repeating: 1000, count: rows)
        let parts = [PrefillPart(ids: ids[...], drafter: nil, head: false)]
        P = p0
        guard let cb = ctx.queue.makeCommandBuffer(), let e = cb.makeComputeCommandEncoder() else {
            throw SafetensorsError.io("cb")
        }
        try encodePrefillGroup(e, parts: parts, cb: cb)?.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if let err = cb.error { throw err }
        P = p0
        let t = try GPUTimeline(ctx: ctx)
        try encodePrefillGroup(nil, parts: parts, tl: t)
        let r = try t.run()
        return (r.pieces, r.total, r.gaps, (cb.gpuEndTime - cb.gpuStartTime) * 1e3)
    }
}
