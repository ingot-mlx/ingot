import Foundation
import Metal

// The decode round of HybridModel (Model.swift): B rows through the verification path (1 = normal, up to 16 with
// DFlash2). A single skeleton (`encodeVerify`) for all families, with the prefill hooks (embedding,
// mixer, FFN, capture and head); the Flash-Next hooks (HC, PLE, QSA and ExpertStream experts) are in
// FlashHybrid.swift and the generation loops in Generate.swift.

extension HybridModel {
    /// A normal decode step (B = 1) with token `t`; leaves the logits in `logits` and returns the argmax.
    public func stepLogits(_ t: UInt32) throws -> UInt32 {
        tok.contents().storeBytes(of: t, as: UInt32.self)
        guard let cb = ctx.queue.makeCommandBuffer(), let e = cb.makeComputeCommandEncoder() else {
            throw SafetensorsError.io("cb")
        }
        try encodeVerify(e, rows: 1, drafter: nil, cb: cb).endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if let err = cb.error { throw err }
        commit(accepted: 1)
        return top1.contents().load(as: UInt32.self)
    }

    /// What the round skeleton hands to the hooks of each family.
    struct VerifyCall {
        let rows: Int
        let drafter: DFlash2Drafter?
        let state: MTLBuffer?
        /// Position of the first row (`posBound` or P).
        let pos: Int
        let rowLimit: Bool
        let cb: MTLCommandBuffer?
        /// Flash-Next: PLE rows that the CPU fetches while the GPU computes the first layers.
        var vPle: UInt64 = 0
    }

    /// B rows (tok[0..<B]) at positions P..P+B−1: the layers (the GDN first folds the `nacc` accepted rows of the
    /// previous round), the head and the argmax into top1. With `drafter`, stores the residual stream after the
    /// drafter layers. Does not advance P (`commit` does).
    /// With `rowLimit` (requires `state`), the pieces whose cost depends on the rows (MoE routing and expert union,
    /// GDN recurrence) only work on the first st[6]; the other rows come out as garbage and are
    /// never accepted (spec_accept reads the same st[6]).
    /// With `expertStream`, each MoE layer cuts the encoder after the router (needs `cb`); the function returns the
    /// current encoder, which the caller must use for whatever it encodes next.
    /// A single skeleton for all families; what is specific to each goes in its embedding, mixer, FFN, capture and
    /// head hooks (Flash-Next's, in FlashHybrid.swift), with the same dispatch order as
    /// the two earlier functions.
    @discardableResult
    public func encodeVerify(
        _ e0: MTLComputeCommandEncoder, rows B: Int, drafter: DFlash2Drafter?, state: MTLBuffer? = nil,
        posBound: Int? = nil, rowLimit: Bool = false, cb: MTLCommandBuffer? = nil
    ) throws -> MTLComputeCommandEncoder {
        precondition(!rowLimit || state != nil)
        precondition(
            expertStream == nil || cb != nil, "this path needs the command buffer with --expert-cache-gb")
        precondition(fn == nil || expertStream != nil, "Flash-Next without ExpertStream")
        var v = VerifyCall(rows: B, drafter: drafter, state: state, pos: posBound ?? P, rowLimit: rowLimit, cb: cb)
        var e = e0
        if let fn {
            e = try encodeFlashVerifyEmbed(e, fn: fn, &v)
        } else {
            try encodeEmbed(e, tok: tok, tokOffset: 0, resid: h, rows: B)
        }
        for i in layers.indices {
            if let fn {
                e = try encodeFlashVerifyMixer(e, fn: fn, layer: i, v)
                e = try encodeFlashVerifyFFN(e, fn: fn, layer: i, v)
            } else {
                try encodeVerifyMixer(e, layer: i, v)
                e = try encodeVerifyFFN(e, layer: i, v)
            }
            if let dr = drafter, let j = dr.targetLayers.firstIndex(of: i) {
                e = flashCut(v, e, "capture", i)
                try encodeCapture(e, resid: h, caps: dr.caps, width: dr.ctxWidth, slot: j, rows: B)
            }
            if debugStopAfter == i { break }
        }
        e = flashCut(v, e, "head", -1)
        try encodeHead(e, resid: h, rows: B)
        try argmax(e, rows: B)
        return e
    }

    /// Batched round: row r of tok is the next token of conversation `convs[r]` (its slot,
    /// BatchDecode.swift), at that conversation's position. What only reads weights (embedding, norms, the attention
    /// and GDN projections, the GDN norm-gate, FFN, head and argmax) goes once over the N rows; the state pieces
    /// (attention prework, attn and reduce over its KV; GDN prework and recurrence over its state, conv history and
    /// tape) go once per conversation, with its slot's buffers, position or `nacc`, and its row offset. Dense models
    /// of attention and GDN layers without a drafter (`batchUnsupported`). Does not advance the positions
    /// (`generateBatch` does); flips the GDN ping-pong of every conversation in the round.
    func encodeVerifyBatch(_ e: MTLComputeCommandEncoder, convs: [Int], slots s: ConversationSlots) throws {
        let B = convs.count
        precondition(B >= 1 && B <= Self.decodeRows && fn == nil && expertStream == nil)
        try encodeEmbed(e, tok: tok, tokOffset: 0, resid: h, rows: B)
        var ka = 0, kg = 0  // attention and GDN layer indices (those of the slots' state)
        for i in layers.indices {
            let l = layers[i]
            let out = branchOut(l, resid: h, prefill: false)
            try norm(e, h, l.n1, xn, rows: B, eps: spec.block.mixerNorm.eps)
            switch l.mixer {
            case .attn(let a):
                a.pos = P
                try a.encode(e, x: xn, out: out, rows: B, only: .proj)
                for (r, j) in convs.enumerated() {
                    let parked = j != s.bound
                    if parked { a.exchangeSlot(&s.parked[j]!.kv[ka]) }
                    a.pos = parked ? s.parked[j]!.P : P
                    for p in [AttentionLayer.Piece.prework, .attn, .reduce] {
                        try a.encode(e, x: xn, out: out, rows: 1, only: p, rowOffset: r)
                    }
                    if parked { a.exchangeSlot(&s.parked[j]!.kv[ka]) }
                }
                a.pos = P
                try a.encode(e, x: xn, out: out, rows: B, accumulate: true, only: .oproj)
                ka += 1
            case .gdn(let g):
                try g.encode(e, x: xn, out: out, rows: B, nacc: 0, only: .proj)
                for (r, j) in convs.enumerated() {
                    let parked = j != s.bound
                    if parked { g.exchangeSlot(&s.parked[j]!.gdn[kg]) }
                    let na = parked ? s.parked[j]!.nacc : nacc
                    for p in [GDNLayer.Piece.prework, .recur] {
                        try g.encode(e, x: xn, out: out, rows: 1, nacc: na, only: p, rowOffset: r)
                    }
                    g.advance()
                    if parked { g.exchangeSlot(&s.parked[j]!.gdn[kg]) }
                }
                for p in [GDNLayer.Piece.normgate, .outproj] {
                    try g.encode(e, x: xn, out: out, rows: B, nacc: 0, accumulate: true, only: p)
                }
                kg += 1
            case .conv, .none: preconditionFailure("batched round: no short convolution or mixer-less layers")
            }
            try encodeBranchEpilogue(e, l, resid: h, rows: B, ffn: false, prefill: false)
            _ = try encodeVerifyFFN(
                e, layer: i, VerifyCall(rows: B, drafter: nil, state: nil, pos: P, rowLimit: false, cb: nil))
        }
        try encodeHead(e, resid: h, rows: B)
        try argmax(e, rows: B)
    }

    /// After a round: `accepted` rows of the block stay (anchor included).
    public func commit(accepted: Int) {
        P += accepted
        nacc = accepted
    }

    // MARK: - Round hooks

    /// Mixer of layer `l` over `x`, adding into `out` (shared by the families; what surrounds it goes in their hooks).
    func encodeVerifyMixerCore(
        _ e: MTLComputeCommandEncoder, _ l: Layer, x: MTLBuffer, out: MTLBuffer, _ v: VerifyCall
    ) throws {
        switch l.mixer {
        case .gdn(let g):
            try g.encode(
                e, x: x, out: out, rows: v.rows, nacc: nacc, accumulate: true, roundState: v.state,
                rowLimit: v.rowLimit)
            g.advance()
        case .attn(let a):
            a.pos = v.pos
            try a.encode(e, x: x, out: out, rows: v.rows, accumulate: true, state: v.state)
        case .conv(let c):
            try c.encode(e, x: x, out: out, rows: v.rows, nacc: nacc, accumulate: true, roundState: v.state)
            c.advance()
        case .none: break
        }
    }

    /// Qwen/LFM2 mixer hook: norm and mixer (nothing in a layer without mixer, Nemotron-H's MLP and MoE blocks).
    private func encodeVerifyMixer(_ e: MTLComputeCommandEncoder, layer i: Int, _ v: VerifyCall) throws {
        let l = layers[i], B = v.rows
        guard l.hasMixer else { return }
        try norm(e, h, l.n1, xn, rows: B, eps: spec.block.mixerNorm.eps)
        try encodeVerifyMixerCore(e, l, x: xn, out: branchOut(l, resid: h, prefill: false), v)
        try encodeBranchEpilogue(e, l, resid: h, rows: B, ffn: false, prefill: false)
    }

    /// Qwen/LFM2 FFN hook: norm and dense FFN or MoE (nothing in a layer without FFN, Nemotron-H's Mamba2 and attention
    /// blocks). With `expertStream`, the MoE cuts the encoder.
    private func encodeVerifyFFN(_ e0: MTLComputeCommandEncoder, layer i: Int, _ v: VerifyCall) throws
        -> MTLComputeCommandEncoder
    {
        var e = e0
        let l = layers[i], B = v.rows, rl = v.rowLimit ? v.state : nil
        guard l.hasFFN else { return e }
        if l.parallel != nil {
            // Gemma 4 26B-A4B: dense MLP and MoE in parallel, each with its own norms (ParallelFFN.swift).
            try encodeParallelFFN(e, layer: i, rows: B, prefill: false, rowLimit: rl)
            try encodeBranchEpilogue(e, l, resid: self.h, rows: B, ffn: true, prefill: false)
            try encodePLEBranch(e, layer: i, resid: self.h, rows: B, prefill: false)
            return e
        }
        let h = branchOut(l, resid: self.h, prefill: false)
        try norm(e, self.h, l.n2, xn, rows: B, eps: spec.block.ffnNorm.eps)
        if let rt = routeTrace { try rt.encodePrediction(e, model: self, layer: i, x: xn, rows: B) }
        if let f = l.dense {
            try f.encode(e, x: xn, out: h, rows: B)
        } else if let es = expertStream, let cb = v.cb {
            // Router and expert union; the CPU fetches the missing ones (and prefetches those of the next layer)
            // while the GPU waits; then gate+up and down with the experts in their slots.
            let moe = l.moe
            try moe.encode(e, x: xn, out: h, rows: B, accumulate: true, only: .router, rowLimit: rl)
            try moe.encode(e, x: xn, out: h, rows: B, accumulate: true, only: .route, rowLimit: rl)
            var nx: Int?
            if es.prefetchK > 0, i + 1 < layers.count, let nm = layers[i + 1].moeOpt, let pb = es.pred[i + 1] {
                try nm.vq.encode(e, x: xn, nm.router, y: pb, rows: B)
                nx = i + 1
            }
            // The GPU computes the resident experts while the CPU fetches the missing ones.
            try moe.encodeSplit(e)
            let (e1, sig) = es.signalOnly(cb, e, layer: i, kind: .verify(moe: moe, rowState: rl, rows: B, next: nx))
            for p in [MoELayer.Piece.gateup, .down] {
                try moe.encode(e1, x: xn, out: h, rows: B, accumulate: true, only: p, rowLimit: rl, pass: .resident)
            }
            e = es.waitAnswer(cb, e1, value: sig)
            for p in [MoELayer.Piece.gateup, .down] {
                try moe.encode(e, x: xn, out: h, rows: B, accumulate: true, only: p, rowLimit: rl, pass: .missing)
            }
        } else {
            try l.moe.encode(e, x: xn, out: h, rows: B, accumulate: true, rowLimit: rl)
        }
        try encodeBranchEpilogue(e, l, resid: self.h, rows: B, ffn: true, prefill: false)
        try encodePLEBranch(e, layer: i, resid: self.h, rows: B, prefill: false)
        return e
    }

    /// Diagnostic cut of `--round-trace 2` between pieces (Flash-Next only, as before).
    @inline(__always) func flashCut(_ v: VerifyCall, _ e: MTLComputeCommandEncoder, _ label: String, _ layer: Int)
        -> MTLComputeCommandEncoder
    {
        guard fn != nil, let cb = v.cb else { return e }
        return tcut(cb, e, label, layer)
    }

    // MARK: - Common prefill and round hooks

    /// Embedding hook: `rows` tokens of `tok` (from byte `tokOffset`) into the residual stream `resid`
    /// (Flash-Next: into its current streams, `fn.S`).
    func encodeEmbed(
        _ e: MTLComputeCommandEncoder, tok t: MTLBuffer, tokOffset: Int, resid: MTLBuffer, rows: Int
    ) throws {
        if let fn {
            try fn.embedStreams(e, embed: embed, tb: t, tokOffset: tokOffset, rows: rows)
        } else {
            try embedRows(e, tok: t, tokOffset: tokOffset, out: resid, rows: rows)
            try encodeEmbedPost(e, resid: resid, rows: rows)
            // Gemma 4 E2B/E4B: the per-layer inputs of these rows (round or prefill chunk, by the residual stream).
            let round = resid === h
            try ple?.encodeInputs(
                e, tok: t, tokOffset: tokOffset, resid: resid, xh: round ? xn : pxn, rows: rows, prefill: !round)
        }
    }

    /// Drafter capture hook: `rows` rows of the residual stream `resid` (Flash-Next: the mean of its
    /// current streams) into slot `slot` of `caps` (rows of `width`: DFlash2Drafter.ctxWidth).
    func encodeCapture(
        _ e: MTLComputeCommandEncoder, resid: MTLBuffer, caps: MTLBuffer, width: Int, slot j: Int, rows: Int
    ) throws {
        if let fn {
            try fn.encodeCapture(e, caps: caps, width: width, slot: j, rows: rows)
        } else {
            try capture(e, resid, into: caps, width: width, slot: j, rows: rows)
        }
    }

    /// Head hook: logits of `rows` rows of `resid` from row `rowOffset` into `logits` (final norm and lm_head;
    /// Flash-Next: the final HC of its current streams and its lm_head). The skeleton adds the argmax.
    func encodeHead(_ e: MTLComputeCommandEncoder, resid: MTLBuffer, rows: Int, rowOffset: Int = 0) throws {
        if let fn {
            try fn.hcMix(e, fn.finalHC, rows: rows, rowOffset: rowOffset)
            try fn.vq.encode(e, x: fn.xh, lmHead, y: logits, rows: rows)
        } else {
            try norm(e, resid, finalNorm, xn, rows: rows, rowOffset: rowOffset, eps: spec.block.finalNorm.eps)
            try vqHead.encode(e, x: xn, lmHead, y: logits, rows: rows)
            try encodeLogitsPost(e, rows: rows)
        }
    }
}
