import Foundation
import Metal

// Round and prefill hooks of Flash-Next inside HybridModel (pieces in FlashParts.swift; the skeletons are in
// ModelRound.swift and ModelPrefill.swift). Experts come from ExpertStream
// (single experts.bin record) and PLE rows from FlashPLERows on the ExpertStream thread. Fidelity is
// checked against the mlx-vlm harness (`ingot-dev fncheck`, ref/flashnext/harness.py and tfcompare.py).

extension HybridModel {
    /// fncheck: fp32 streams [rows, 4·hidden] of the last prefill's first chunk (with `debugStopAfter`, those after
    /// that layer).
    public func flashStreams(rows: Int) -> [Float] {
        guard let fn else { return [] }
        let n = rows * fn.W
        precondition(fn.sPool[0].length >= n * 4)
        return Array(
            UnsafeBufferPointer(start: fn.sPool[0].contents().bindMemory(to: Float.self, capacity: n), count: n))
    }

    /// Embedding hook of the Flash-Next round: streams of tok[0..<B] and, for the PLE, indices on the GPU (the round
    /// may come chained: the CPU does not know its tokens) and rows on the CPU while the GPU computes the first
    /// layers (awaited at the PLE layer, `v.vPle`).
    func encodeFlashVerifyEmbed(
        _ e0: MTLComputeCommandEncoder, fn: FlashParts, _ v: inout VerifyCall
    ) throws -> MTLComputeCommandEncoder {
        let es = expertStream!, cb = v.cb!, B = v.rows
        var e = tcut(cb, e0, "embed+ple.idx", -1)
        fn.S = fn.sPool[0]
        try encodeEmbed(e, tok: tok, tokOffset: 0, resid: h, rows: B)
        try fn.encodePleRound(e, tok: tok, hist: ptok, rows: B, pos: v.pos, state: v.state)
        let prow = fn.prowPool[0], pidx = fn.pidx, rowsCache = fn.pleRows
        (e, v.vPle) = es.signalOnly(
            cb, e, layer: 0,
            kind: .custom(
                work: {
                    do {
                        try rowsCache.fetch(
                            pidx.contents().bindMemory(to: UInt32.self, capacity: B * fn.pleH), count: B * fn.pleH,
                            into: prow.contents())
                    } catch { preconditionFailure("Flash-Next: \(error)") }
                }, ahead: false))
        return e
    }

    /// Mixer hook of the Flash-Next round: PLE (at its layer), attention HC, QSA and mixer over the
    /// streams.
    func encodeFlashVerifyMixer(
        _ e0: MTLComputeCommandEncoder, fn: FlashParts, layer i: Int, _ v: VerifyCall
    ) throws -> MTLComputeCommandEncoder {
        let cb = v.cb!, B = v.rows, l = layers[i], lp = fn.parts[i]
        var e = e0
        if i == fn.fmt.pleLayer {
            e = expertStream!.waitAnswer(cb, e, value: v.vPle, label: "ple", layer: i)
            try fn.encodePLE(e, prow: fn.prowPool[0], rows: B, padded: B, prefill: false, nacc: nacc, state: v.state)
        }
        let q = fn.qsa[i]
        e = tcut(cb, e, "hc.a", i)
        let brZero = try fn.hcMix(e, lp.hcA, rows: B, f32: q != nil, zeroBr: true)
        // Round trace (`--round-trace 3`): cuts inside the QSA and attention, and a copy of each layer's bitmap.
        let fine = roundTrace?.fine == true
        if case .attn(let a) = l.mixer, let q {
            e = tcut(cb, e, "qsa", i)
            let sparse = v.pos + B - 1 >= FlashQSA.firstSparsePos
            e = try fn.encodeQSA(
                e, q, attn: a, rows: B, pos: v.pos, nacc: nacc, state: v.state, prefill: false, sparse: sparse,
                theta: fn.theta, cut: fine ? { self.tcut(cb, $0, $1, i) } : nil)
            if fine, sparse, let qm = a.qsaMask {
                e = roundTrace!.captureBitmap(cb, e, qm.bm, rows: B, nw: qm.nw, layer: i, G: a.G)
            }
        }
        if !brZero { try fn.zeroBr(e, rows: B) }
        e = tcut(cb, e, l.mixer.traceName, i)
        if case .conv = l.mixer { preconditionFailure("Flash-Next has no short convolution") }
        if fine, case .attn(let a) = l.mixer {
            a.pos = v.pos
            for p in AttentionLayer.Piece.allCases {
                e = tcut(cb, e, "attn." + p.rawValue, i)
                try a.encode(e, x: fn.xh, out: fn.br, rows: B, accumulate: true, only: p, state: v.state)
            }
        } else {
            try encodeVerifyMixerCore(e, l, x: fn.xh, out: fn.br, v)
        }
        if case .attn(let a) = l.mixer { a.qsaMask = nil; a.blockList = nil }
        return e
    }

    /// FFN hook of the Flash-Next round: MoE HC, router and shared expert, early prediction of the next layer,
    /// ExpertStream experts (resident ones while the CPU fetches the missing, resident-first compute) and combine.
    func encodeFlashVerifyFFN(
        _ e0: MTLComputeCommandEncoder, fn: FlashParts, layer i: Int, _ v: VerifyCall
    ) throws -> MTLComputeCommandEncoder {
        let es = expertStream!, cb = v.cb!, B = v.rows, lp = fn.parts[i], rs = v.rowLimit ? v.state : nil
        var e = tcut(cb, e0, "hc.f", i)
        try fn.hcMix(e, lp.hcF, rows: B, injectFirst: true)
        e = tcut(cb, e, "router", i)
        try fn.encodeMoEFront(e, lp, rows: B, padded: B, prefill: false, rowState: rs)
        var nx: Int?
        if es.prefetchK > 0, i + 1 < layers.count, let pb = es.pred[i + 1] {
            e = tcut(cb, e, "pred", i)
            try fn.mm(e, fn.parts[i + 1].router, x: fn.xh, y: pb, rows: B, prefill: false)
            nx = i + 1
        }
        let smapOff = i * fn.E * 4, arena = es.arenas[0]
        let kind = ExpertStream.JobKind.verifyInds(inds: fn.inds, wts: fn.wts, rowState: rs, rows: B, next: nx)
        e = tcut(cb, e, "split", i)
        try fn.encodeSplit(e, smapR: es.smapR, smapOff: smapOff)
        let (e2, sig) = es.signalOnly(cb, e, layer: i, kind: kind)
        try fn.encodeExperts(
            e2, arena: arena, smap: es.smapR, smapOff: smapOff, tiles: fn.tilesR, args: fn.argsR, rows: B)
        e = es.waitAnswer(cb, e2, value: sig, layer: i)
        try fn.encodeExperts(
            e, arena: arena, smap: es.smap, smapOff: smapOff, tiles: fn.tilesM, args: fn.argsM, rows: B)
        e = tcut(cb, e, "combine", i)
        try fn.encodeCombineInject(e, rows: B)
        return e
    }

    /// Layer hook of the Flash-Next prefill (skeleton in `encodePrefillGroup`, ModelPrefill.swift): PLE, HC,
    /// mixer (GDN or attention, with the QSA), MoE with the ExpertStream experts of chunk `lc.part`.
    /// State captures inside the pass (GDN, PLE window and QSA windows) go in the
    /// group's last chunk if it is the `head` one (`capHere`); with `lc.endSlot`, the state at the end of the chunk is
    /// copied as is into that capture (the group's next chunk writes to the other ping-pong buffer), like
    /// `InferenceEngine.copy` after a call to `prefill` that ended there.
    /// Lossy prefill: GDN projection and out_proj and attention q|k|v and o_proj in rotated W8A8
    /// (`lc.i8`) and Q·Kᵀ in int8 (`lc.qk8`, with the QSA mask if applicable); with `int8Experts`, the Q2
    /// experts with int8 activations per group of 64 (exact weights). HC and PLE stay exact.
    func encodeFlashPrefillLayer(
        _ e0: MTLComputeCommandEncoder, fn: FlashParts, _ lc: PrefillLayerCall, group g: PrefillGroupState,
        capHere: Bool, cb: MTLCommandBuffer
    ) throws -> MTLComputeCommandEncoder {
        guard let es = expertStream else { preconditionFailure("Flash-Next without ExpertStream") }
        var e = e0
        let i = lc.layer, c = lc.part, l = layers[i], lp = fn.parts[i], G = g.rs.count
        let r = g.rs[c], M = g.Ms[c], i8 = lc.i8, i8o = options.int8
        let lastCap = capHere && c == G - 1
        let capRows = lastCap ? pendingCapture!.rows : [], convCaps = lastCap ? (pendingConvCapture ?? []) : []
        precondition(capRows.isEmpty || convCaps.count == capRows.count)
        fn.S = fn.sPool[c]
        if i == fn.fmt.pleLayer {
            if c == 0 { e = es.waitReady(cb, e, value: g.vPle) }
            try fn.encodePLE(e, prow: g.prows[c], rows: r, padded: M, prefill: true, nacc: 0, state: nil)
            if let k = lc.endSlot {
                try copyBytes(e, fn.ple.hist[fn.ple.cur], captureConvTargets![k][0])
            }
            for (k, row) in capRows.enumerated() {
                try fn.encodeWinCapture(
                    e, src: fn.normd, hprev: fn.ple.hist[1 - fn.ple.cur], out: convCaps[k][0], row: row,
                    window: FlashPLE.window, width: fn.W)
            }
        }
        let q = fn.qsa[i]
        let brZero = try fn.hcMix(e, lp.hcA, rows: r, padded: M, f32: q != nil, zeroBr: true)
        if case .attn(let a) = l.mixer, let q {
            try fn.encodeQSA(
                e, q, attn: a, rows: r, pos: g.p0s[c], nacc: 0, state: nil, prefill: true,
                sparse: g.p0s[c] + r - 1 >= FlashQSA.firstSparsePos, theta: fn.theta)
            let qi = 1 + fn.qsaIndex[i]!
            if let k = lc.endSlot {
                try copyBytes(e, q.hist[q.cur], captureConvTargets![k][qi])
            }
            for (k, row) in capRows.enumerated() {
                try fn.encodeWinCapture(
                    e, src: fn.qsaRK, hprev: q.hist[1 - q.cur], out: convCaps[k][qi], row: row,
                    window: FlashQSA.window, width: FlashQSA.D)
            }
        }
        if !brZero { try fn.zeroBr(e, rows: r) }
        switch l.mixer {
        case .gdn(let gdn):
            try gdn.encodePrefill(
                e, x: fn.xh, out: fn.br, rows: r, padded: M, s: gs, chunk: gdnChunk,
                i8: i8o.parts.contains("gp") ? i8 : nil, i8Out: i8o.parts.contains("go") ? i8 : nil,
                capture: lc.capture)
            if let k = lc.endSlot {
                let t = captureTargets![k][lc.gdnIndex]
                try copyBytes(e, gdn.state[gdn.cur], t.state)
                try copyBytes(e, gdn.hist[gdn.cur], t.hist)
            }
        case .attn(let a):
            a.pos = g.p0s[c]
            try a.encodePrefill(
                e, x: fn.xh, out: fn.br, rows: r, padded: M, s: ats,
                i8: i8o.parts.contains("ap") ? i8 : nil, i8Out: i8o.parts.contains("ao") ? i8 : nil,
                qk8: lc.qk8)
            a.qsaMask = nil; a.blockList = nil
        case .conv, .none:
            preconditionFailure()
        }
        try fn.hcMix(e, lp.hcF, rows: r, padded: M, injectFirst: true)
        try fn.encodeMoEFront(e, lp, rows: r, padded: M, prefill: true)
        let smapOff = i * fn.E * 4, arena = es.arenas[0]
        if g.bulk {
            // The CPU is already fetching (ahead) what the layer is missing; wait once per group.
            if c == 0 { e = es.prefillBeginBulk(cb, e, layer: i) }
        } else {
            e = es.prefillBeginInds(cb, e, layer: i, inds: fn.inds, rows: r)
        }
        try fn.encodeExperts(
            e, arena: arena, smap: es.smap, smapOff: smapOff, tiles: fn.tiles, args: fn.args, rows: r,
            nax: fn.naxBM(rows: r), i8: i8 != nil && int8Experts)
        try fn.encodeCombineInject(e, rows: r)
        if !g.bulk || c == G - 1 { e = es.prefillEnd(cb, e) }
        return e
    }

    /// Copies the first `dst.length` bytes of `src` into `dst` (whole words: bit-exact, whatever the type).
    func copyBytes(_ e: MTLComputeCommandEncoder, _ src: MTLBuffer, _ dst: MTLBuffer) throws {
        precondition(dst.length % 16 == 0 && src.length >= dst.length)
        var n4 = UInt32(dst.length / 16)
        e.setComputePipelineState(try ctx.pipeline("copy_u32"))
        e.setBuffer(src, offset: 0, index: 0)
        e.setBuffer(dst, offset: 0, index: 1)
        e.setBytes(&n4, length: 4, index: 2)
        e.dispatchThreads(
            MTLSize(width: Int(n4), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    /// Flash-Next: tokens `ids[from − 2 ..< from]` into the per-position history (PLE context) before reading
    /// a prompt from a prefix cache or a snapshot (another conversation may have rewritten their positions).
    func flashSetHistory(_ ids: [UInt32], before c: Int) {
        guard fn != nil, c > 0 else { return }
        let hp = ptok.contents().bindMemory(to: UInt32.self, capacity: maxCtx + chunkRows)
        for q in max(0, c - 2)..<c { hp[q] = ids[q] }
    }
}

extension HybridModel {
    /// Per-piece profile (ingot-dev fnprof): a round of B rows (prefill = nil) or a prefill chunk of `prefill` rows
    /// over a history of P = `pos` tokens, with the expert slot table pointing at arbitrary slots (no
    /// SSD reads or waits: only kernel time). Numeric results are meaningless; the model
    /// state is left unusable (call reset()).
    public func flashTimeline(rows B: Int, pos: Int, prefill: Int? = nil) throws -> (
        pieces: [(String, Double)], total: Double, gaps: Double
    ) {
        guard let fn, let es = expertStream else { throw SafetensorsError.format("fnprof: Flash-Next only") }
        let E = fn.E, L = layers.count
        let sm = es.smap.contents().bindMemory(to: UInt32.self, capacity: L * E)
        let sr = es.smapR.contents().bindMemory(to: UInt32.self, capacity: L * E)
        for k in 0..<(L * E) {
            sm[k] = UInt32(k % es.slots); sr[k] = sm[k]
        }
        let R = prefill ?? B, M = (R + 63) / 64 * 64, pf = prefill != nil
        precondition(R <= chunkRows && pos + R <= maxCtx)
        let tp = ptok.contents().bindMemory(to: UInt32.self, capacity: maxCtx + chunkRows)
        for j in 0..<(pos + M) { tp[j] = UInt32((j * 7919) % 200_000 + 100) }
        let tk = tok.contents().bindMemory(to: UInt32.self, capacity: 16)
        for j in 0..<min(B, 16) { tk[j] = UInt32(1000 + j) }
        let t = try GPUTimeline(ctx: ctx)
        func hc(_ h: FlashHyperConn, f32: Bool = false, inj: Bool = false, zero: Bool = false) throws -> Bool {
            var z = false
            try t.seg("hc") {
                z = try fn.hcMix($0, h, rows: R, padded: pf ? M : nil, f32: f32, injectFirst: inj, zeroBr: zero)
            }
            return z
        }
        fn.S = fn.sPool[0]
        try t.seg("embed") {
            try fn.embedStreams($0, embed: embed, tb: pf ? ptok : tok, tokOffset: pf ? pos * 4 : 0, rows: pf ? M : B)
        }
        if !pf {
            try t.seg("ple.hash") { try fn.encodePleRound($0, tok: tok, hist: ptok, rows: B, pos: pos, state: nil) }
        }
        for (i, l) in layers.enumerated() {
            let lp = fn.parts[i]
            if i == fn.fmt.pleLayer {
                try t.seg("ple") {
                    try fn.encodePLE(
                        $0, prow: fn.prowPool[0], rows: R, padded: pf ? M : R, prefill: pf, nacc: 0, state: nil)
                }
            }
            let q = fn.qsa[i]
            let brZero = try hc(lp.hcA, f32: q != nil, zero: true)
            if case .attn(let a) = l.mixer, let q {
                try t.seg("qsa.idx") { try fn.encodeIndexer($0, q, rows: R) }
                try t.seg("qsa") {
                    try fn.encodeQSA(
                        $0, q, attn: a, rows: R, pos: pos, nacc: 0, state: nil, prefill: pf,
                        sparse: pos + R - 1 >= FlashQSA.firstSparsePos, theta: fn.theta, skipIndexer: true)
                }
            }
            if !brZero { try t.seg("hc") { try fn.zeroBr($0, rows: R) } }
            switch l.mixer {
            case .gdn(let g):
                if pf {
                    try t.seg("gdn") {
                        try g.encodePrefill(
                            $0, x: fn.xh, out: fn.br, rows: R, padded: M, s: gs, chunk: gdnChunk)
                    }
                } else {
                    for p in GDNLayer.Piece.allCases {
                        try t.seg("gdn." + p.rawValue) {
                            try g.encode($0, x: fn.xh, out: fn.br, rows: B, nacc: 0, accumulate: true, only: p)
                        }
                    }
                    g.advance()
                }
            case .attn(let a):
                a.pos = pos
                if pf {
                    try t.seg("attn") {
                        try a.encodePrefill(
                            $0, x: fn.xh, out: fn.br, rows: R, padded: M, s: ats)
                    }
                } else {
                    for p in AttentionLayer.Piece.allCases {
                        try t.seg("attn." + p.rawValue) {
                            try a.encode($0, x: fn.xh, out: fn.br, rows: B, accumulate: true, only: p)
                        }
                    }
                }
                a.qsaMask = nil; a.blockList = nil
            case .conv, .none:
                preconditionFailure()
            }
            _ = try hc(lp.hcF, inj: true)
            try t.seg("moe.router") {
                try fn.mm($0, lp.router, x: fn.xh, y: fn.logitsR, rows: pf ? M : R, prefill: pf)
            }
            try t.seg("moe.route+group") { try fn.encodeRouteGroup($0, rows: R, prefill: pf) }
            try t.seg("moe.shared") {
                try fn.mm($0, lp.sGU, x: fn.xh, y: fn.gu, rows: pf ? M : R, prefill: pf)
                try fn.threads($0, "fn_swiglu", R * fn.I, [(fn.gu, 0), (fn.hs, 0)], [UInt32(R), UInt32(fn.I)])
                try fn.mm($0, lp.sDown, x: fn.hs, y: fn.sh, rows: pf ? M : R, prefill: pf)
            }
            try t.seg("moe.experts") {
                try fn.encodeExperts(
                    $0, arena: es.arenas[0], smap: es.smap, smapOff: i * E * 4, tiles: fn.tiles, args: fn.args, rows: R,
                    nax: pf ? fn.naxBM(rows: R) : 0)
            }
            try t.seg("moe.combine") {
                try fn.encodeCombineInject($0, rows: R)
            }
        }
        try t.seg("head") {
            try fn.hcMix($0, fn.finalHC, rows: pf ? 1 : B)
            try fn.vq.encode($0, x: fn.xh, lmHead, y: logits, rows: pf ? 1 : B)
            try argmax($0, rows: pf ? 1 : B)
        }
        return try t.run()
    }
}
