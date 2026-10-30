import Foundation
import Metal

// Prefill of HybridModel (Model.swift): chunked and in layer-by-layer groups, with the per-position W8A8 prefill
// and the prefill MoE. A single skeleton (`encodePrefillGroup`) for all families, with embedding, layer
// and head hooks; the Flash-Next layer hook is in FlashHybrid.swift and prompt preparation in Prompt.swift. What
// changes between families outside the hooks goes in `PrefillCaps`.

extension HybridModel {
    /// Named prefill capabilities of each family (previously, `fn != nil` branches scattered across Prompt.swift,
    /// Engine.swift, ModelCapture.swift, PrefixCache.swift and this file). Flash-Next has them all; the rest, per
    /// the options.
    public struct PrefillCaps {
        /// prefix cache cuts go in the same `prefill` call (`cuts`) and are captured inside the pass in their entries,
        /// instead of splitting the prefill into separate calls (and SSD passes).
        public let cutsInPass: Bool
        /// the conversation cut is captured as one more row of the last chunk (state, windows and drafter ring,
        /// `pendingRingCapture`); it does not split the pass. Every family that captures in the pass, not only
        /// Flash-Next (on the Qwen3.6 MoE the split added a 51-row chunk to a 3K prompt, ~75 ms of TTFT in
        /// lab serve); outside Flash-Next every cut whose entry the engine reserved (the system cut added a
        /// 310-row chunk to pi's first turn, −133 ms), up to two per pass.
        public let rowCuts: Bool
        /// Captures inside the pass also carry the convolution windows (Flash-Next's PLE and QSA).
        public let convCaptures: Bool
        /// Chunks per prefill group with experts on the SSD (`expertPrefillGroup` / `flashPrefillGroup`) and
        /// whether a group can carry the captures inside the pass (outside Flash-Next, the captures
        /// chunk goes alone).
        public let groupParts: Int, groupWithCaptures: Bool
        /// Chunks with drafter per group (`drafterCaptureBuffers`: drafter.caps, capsAlt, and in Flash-Next capsExtra).
        public let maxDraftParts: Int
        /// Rows from which the prefill fetches ahead all the missing experts (`expertBulkMinRows`,
        /// Flash-Next `flashBulkMinRows`).
        public let bulkMinRows: Int
    }

    public var prefillCaps: PrefillCaps {
        let o = options, flash = fn != nil
        return PrefillCaps(
            cutsInPass: flash, rowCuts: flash || canCaptureInPass, convCaptures: flash,
            groupParts: flash ? o.flashPrefillGroup : o.expertPrefillGroup, groupWithCaptures: flash,
            maxDraftParts: flash ? Self.flashDraftParts : 2,
            bulkMinRows: flash ? o.flashBulkMinRows : o.expertBulkMinRows)
    }

    /// A chunk of a prefill group.
    public struct PrefillPart {
        public let ids: ArraySlice<UInt32>
        public let drafter: DFlash2Drafter?
        public let head: Bool
        /// capture index (in `captureTargets`) of the state at the end of this chunk (a prefix cache cut
        /// inside the pass).
        public var endSlot: Int?
        public init(ids: ArraySlice<UInt32>, drafter: DFlash2Drafter?, head: Bool, endSlot: Int? = nil) {
            self.ids = ids; self.drafter = drafter; self.head = head; self.endSlot = endSlot
        }
    }

    /// Several consecutive prefill chunks computed layer by layer (with `expertStream`; Flash-Next): in each
    /// layer, the chunks go in order (each on the GDN state and the KV cache the previous one leaves in that layer),
    /// so each chunk does exactly the same math as chunk by chunk; only the order across layers of different
    /// chunks changes. The experts of each layer are fetched once per group. Each chunk carries its residual stream
    /// (`phPool`; in Flash-Next, its streams in `fn.sPool`). At most `maxDraftParts` chunks with drafter
    /// (`drafterCaptureBuffers`).
    /// A single skeleton for both families; what is specific to each goes in its hooks: embedding
    /// (`embedPrefillParts`, with `encodeEmbed`), layer (`encodePrefillLayer` / `encodeFlashPrefillLayer`), capture
    /// (`encodeCapture`) and head (`encodeHead`); the last three, shared with the round (ModelRound.swift).
    @discardableResult
    public func encodePrefillGroup(
        _ e0: MTLComputeCommandEncoder?, parts: [PrefillPart], tl: GPUTimeline? = nil, cb: MTLCommandBuffer? = nil
    ) throws -> MTLComputeCommandEncoder? {
        precondition(expertStream == nil || (cb != nil && e0 != nil), "the prefill needs the command buffer")
        let G = parts.count, caps = prefillCaps
        precondition(
            G >= 1 && parts.filter { $0.drafter != nil }.count <= caps.maxDraftParts
                && parts.dropLast().allSatisfy { !$0.head },
            "unsupported prefill group")
        // Outside Flash-Next, groups of more than one chunk carry no timeline, captures or debug stop.
        precondition(
            fn != nil || G == 1 || (expertStream != nil && tl == nil && pendingCapture == nil && debugStopAfter == nil),
            "unsupported prefill group")
        // The PLE inputs of a chunk live in one buffer until its layers run (GemmaPLE; Gemma 4 has no grouped prefill).
        precondition(ple == nil || G == 1, "PLE: one chunk per prefill group")
        // captures of the state at the end of a chunk (prefix cache cuts inside the pass).
        if parts.contains(where: { $0.endSlot != nil }) {
            precondition(captureTargets != nil && captureConvTargets != nil, "cut capture without targets")
        }
        let tl = fn == nil ? tl : nil  // Flash-Next has no per-piece timeline
        var e = e0
        var g = try embedPrefillParts(e, parts: parts, tl: tl)
        lastChunkRows = g.rs[G - 1]
        let capsOf = drafterCaptureBuffers(parts)
        // the chunk's padding rows do not enter the MoE (they dragged in experts nobody uses; bit-exact).
        if fn == nil, let pmoe { configurePrefillMoE(pmoe, rows: g.Ms[0]); pmoe.active = g.rs[0] }
        // Lossy prefill: dense GEMMs in rotated W8A8 from position `int8MinPos` and Q·Kᵀ in int8 from
        // qk8MinCtx (prefillLossyChoices); which pieces, in each layer hook.
        let i8o = options.int8
        (g.i8, g.qk8) = try prefillLossyChoices(starts: g.p0s)
        // the prefill fetches ahead all the missing experts in each layer if the group has enough
        // rows (Flash-Next: also with more than one chunk; with a single short chunk, only what it routes).
        let rows = g.rs.reduce(0, +)
        g.bulk = fn != nil ? G > 1 || rows >= caps.bulkMinRows : expertStream != nil && rows >= caps.bulkMinRows
        // State captures inside the pass: only in the prompt's last chunk (the `head` one), which may go
        // in a group with others; its rows are indices within that chunk (fitCaptures relocates them if a threshold
        // splits the span).
        let capHere = pendingCapture != nil && parts.last!.head
        var gi = 0  // index of the GDN layer (captures)
        for (i, l) in layers.enumerated() {
            let layerOn = i8o.layers?.contains(i) ?? true
            tl?.prefix = l.timelineKind + "."
            for c in 0..<G {
                var cap: (rows: [Int], targets: [(state: MTLBuffer, hist: MTLBuffer)])?
                if capHere && c == G - 1, case .gdn = l.mixer {
                    cap = pendingCapture.map { (rows: $0.rows, targets: $0.targets.map { $0[gi] }) }
                }
                let lc = PrefillLayerCall(
                    layer: i, part: c, gdnIndex: gi, i8: layerOn ? g.i8[c] : nil, qk8: g.qk8[c], capture: cap,
                    endSlot: parts[c].endSlot)
                if let fn {
                    e = try encodeFlashPrefillLayer(e!, fn: fn, lc, group: g, capHere: capHere, cb: cb!)
                } else {
                    e = try encodePrefillLayer(e, lc, group: g, tl: tl, cb: cb)
                }
                if let dr = parts[c].drafter, let j = dr.targetLayers.firstIndex(of: i), let cbuf = capsOf[c] {
                    try tseg(tl, e, "capture") {
                        try encodeCapture($0, resid: ph, caps: cbuf, width: dr.ctxWidth, slot: j, rows: g.Ms[c])
                    }
                }
            }
            if case .gdn = l.mixer { gi += 1 }
            if debugStopAfter == i { break }
        }
        tl?.prefix = ""
        // After the layers: the drafter context of each chunk with drafter and the head (argmax of the last row) of
        // those that ask for it.
        for c in 0..<G {
            selectPrefillPart(c)
            if let dr = parts[c].drafter, let cbuf = capsOf[c] {
                try tseg(tl, e, "draft.ctx") { e in
                    try encodeDrafterContext(
                        e, dr, caps: cbuf, part: parts[c], rows: g.Ms[c], nvalid: g.rs[c], pos0: g.p0s[c])
                }
            }
            if parts[c].head {
                try tseg(tl, e, "head") { e in
                    try encodeHead(e, resid: ph, rows: 1, rowOffset: g.rs[c] - 1)
                    try argmax(e, rows: 1)
                }
            }
        }
        P = g.end
        nacc = 0
        return e
    }

    /// What the prefill skeleton hands to the hooks of each family, per group.
    struct PrefillGroupState {
        /// Position, real and padded rows (multiple of 64) of each chunk, and the position after the last.
        var p0s: [Int] = [], rs: [Int] = [], Ms: [Int] = [], end = 0
        /// Lossy prefill of each chunk (prefillLossyChoices).
        var i8: [DenseI8Scratch?] = [], qk8: [(q: MTLBuffer, s: MTLBuffer)?] = []
        /// Fetch ahead all the experts of each layer.
        var bulk = false
        /// Flash-Next: PLE rows of each chunk (ExpertStream fetches them ahead; awaited with `vPle`).
        var prows: [MTLBuffer] = [], vPle: UInt64 = 0
    }

    /// A layer of a chunk of the group: what the skeleton decides.
    struct PrefillLayerCall {
        let layer: Int, part: Int
        /// Index of the layer among the GDNs (capture targets).
        let gdnIndex: Int
        let i8: DenseI8Scratch?
        let qk8: (q: MTLBuffer, s: MTLBuffer)?
        /// capture of the GDN state after some rows of the chunk (only in the last one, if it is a GDN).
        let capture: (rows: [Int], targets: [(state: MTLBuffer, hist: MTLBuffer)])?
        /// capture of the state at the end of the chunk (Flash-Next).
        let endSlot: Int?
    }

    /// Chunk `c` of the group becomes the current one: its residual stream (`ph`) or its streams (Flash-Next, `fn.S`).
    func selectPrefillPart(_ c: Int) {
        if let fn { fn.S = fn.sPool[c] } else { ph = phPool[c] }
    }

    /// Embedding hook: tokens of each chunk in its zone of `ptok` (the per-position history) and embeddings in its
    /// residual stream (`phPool`, which grows if needed; Flash-Next: its streams and the PLE rows of all the
    /// chunks ahead of the GPU, which awaits the PLE layer).
    private func embedPrefillParts(
        _ e: MTLComputeCommandEncoder?, parts: [PrefillPart], tl: GPUTimeline?
    ) throws -> PrefillGroupState {
        var g = PrefillGroupState()
        if let fn { fn.ensureParts(parts.count) }
        while fn == nil && phPool.count < parts.count { phPool.append(ctx.buffer(length: chunkRows * hidden * 4)) }
        let hp = ptok.contents().bindMemory(to: UInt32.self, capacity: maxCtx + chunkRows)
        var idxs: [[UInt32]] = []
        var pc = P
        for (c, part) in parts.enumerated() {
            let r = part.ids.count, M = (r + 63) / 64 * 64
            precondition(r > 0 && M <= chunkRows && pc + r <= maxCtx)
            if let fn {
                // PLE context: the previous two tokens of the per-position history (EOS before the start).
                let eos = fn.fmt.pleEos
                let ctx2 = [pc >= 2 ? hp[pc - 2] : eos, pc >= 1 ? hp[pc - 1] : eos]
                idxs.append(fn.fmt.pleIndices(context: ctx2, tokens: Array(part.ids)))
            }
            let tp = hp + pc
            for (i, t) in part.ids.enumerated() { tp[i] = t }
            for i in r..<M { tp[i] = 0 }
            selectPrefillPart(c)
            try tseg(tl, e, "embed") { try encodeEmbed($0, tok: ptok, tokOffset: pc * 4, resid: ph, rows: M) }
            g.p0s.append(pc); g.rs.append(r); g.Ms.append(M)
            pc += r
        }
        g.end = pc
        if let fn, let es = expertStream {
            let prows = Array(fn.prowPool.prefix(parts.count)), rowsCache = fn.pleRows, lists = idxs
            g.prows = prows
            g.vPle = es.enqueueAhead {
                do {
                    for (c, l) in lists.enumerated() {
                        try l.withUnsafeBufferPointer {
                            try rowsCache.fetch($0.baseAddress!, count: l.count, into: prows[c].contents())
                        }
                    }
                } catch { preconditionFailure("Flash-Next: \(error)") }
            }
        }
        return g
    }

    /// Drafter captures of each chunk: the first with drafter in drafter.caps; the second, in capsAlt; the rest
    /// (Flash-Next), in capsExtra.
    private func drafterCaptureBuffers(_ parts: [PrefillPart]) -> [MTLBuffer?] {
        var capsOf = [MTLBuffer?](repeating: nil, count: parts.count)
        var nd = 0
        for (c, part) in parts.enumerated() {
            guard let dr = part.drafter else { continue }
            if nd == 0 {
                capsOf[c] = dr.caps
            } else if nd == 1 {
                if capsAlt == nil || capsAlt!.length < chunkRows * dr.ctxWidth * 2 {
                    capsAlt = ctx.buffer(length: chunkRows * dr.ctxWidth * 2)
                }
                capsOf[c] = capsAlt
            } else {
                while capsExtra.count < nd - 1 {
                    capsExtra.append(ctx.buffer(length: chunkRows * dr.ctxWidth * 2))
                }
                capsOf[c] = capsExtra[nd - 2]
            }
            nd += 1
        }
        return capsOf
    }

    /// Layer hook of the Qwen/LFM2 models: norm, mixer (GDN, attention or short convolution), norm and dense FFN
    /// or MoE of chunk `lc.part`. With `expertStream`, the MoE cuts the encoder and returns the current encoder.
    private func encodePrefillLayer(
        _ e0: MTLComputeCommandEncoder?, _ lc: PrefillLayerCall, group g: PrefillGroupState, tl: GPUTimeline?,
        cb: MTLCommandBuffer?
    ) throws -> MTLComputeCommandEncoder? {
        var e = e0
        let i = lc.layer, c = lc.part, l = layers[i], G = g.rs.count
        let r = g.rs[c], M = g.Ms[c], i8 = lc.i8
        ph = phPool[c]
        if G > 1, let pmoe { configurePrefillMoE(pmoe, rows: M); pmoe.active = r }
        pmoe?.accumulateInto = ph
        // With a branch epilogue, the mixer and the FFN accumulate into pbranch (attention layers only).
        let out = branchOut(l, resid: ph, prefill: true)
        // Nemotron-H's single-block layers skip the half they do not have, its norm included.
        if l.hasMixer {
            try encodePrefillMixer(e, lc, l, out: out, group: g, tl: tl)
        }
        guard l.hasFFN else { return e }
        if l.parallel != nil {
            // Gemma 4 26B-A4B: dense MLP and MoE in parallel, each with its own norms (ParallelFFN.swift).
            try tseg(tl, e, "ffn") { try encodeParallelFFN($0, layer: i, rows: M, prefill: true, i8: i8) }
        } else {
            try tseg(tl, e, "norm") { try norm($0, ph, l.n2, pxn, rows: M, eps: spec.block.ffnNorm.eps) }
            if let f = l.dense {
                try tseg(tl, e, "ffn") { try f.encodePrefill($0, x: pxn, out: out, rows: M, s: pff!, i8: i8) }
            } else if pmoe != nil {
                pmoe?.accumulateInto = out
                e = try encodePrefillMoE(e, layer: i, part: c, of: G, rows: r, bulk: g.bulk, tl: tl, cb: cb)
            }
        }
        if l.post != nil {
            try tseg(tl, e, "post") { try encodeBranchEpilogue($0, l, resid: ph, rows: M, ffn: true, prefill: true) }
        }
        if ple != nil {
            try tseg(tl, e, "ple") { try encodePLEBranch($0, layer: i, resid: ph, rows: M, prefill: true) }
        }
        return e
    }

    /// Mixer half of a prefill layer of chunk `lc.part`: norm, mixer (GDN, attention or short convolution) and the
    /// mixer's branch epilogue.
    private func encodePrefillMixer(
        _ e: MTLComputeCommandEncoder?, _ lc: PrefillLayerCall, _ l: Layer, out: MTLBuffer, group g: PrefillGroupState,
        tl: GPUTimeline?
    ) throws {
        let c = lc.part, r = g.rs[c], M = g.Ms[c], i8 = lc.i8, i8o = options.int8
        try tseg(tl, e, "norm") { try norm($0, ph, l.n1, pxn, rows: M, eps: spec.block.mixerNorm.eps) }
        switch l.mixer {
        case .gdn(let gdn):
            try gdn.encodePrefill(
                e, x: pxn, out: ph, rows: r, padded: M, s: gs, chunk: options.bench.gdnChunk ? gdnChunk : nil,
                i8: i8o.parts.contains("gp") ? i8 : nil, i8Out: i8o.parts.contains("go") ? i8 : nil,
                tl: tl, capture: lc.capture)
        case .attn(let a):
            // Attention carries its own position: after a decode (which sets it round by round) it must be aligned with
            // the chunk's.
            a.pos = g.p0s[c]
            try a.encodePrefill(
                e, x: pxn, out: out, rows: r, padded: M, s: ats,
                i8: i8o.parts.contains("ap") ? i8 : nil, i8Out: i8o.parts.contains("ao") ? i8 : nil,
                qk8: lc.qk8, tl: tl)
        case .conv(let cv):
            // Into the branch buffer with a branch epilogue (LFM3's post-norm), else straight into the stream (LFM2).
            try tseg(tl, e, "conv") {
                try cv.encodePrefill(
                    $0, x: pxn, out: out, rows: r, padded: M, s: scs,
                    i8: i8o.parts.contains("gp") ? i8 : nil, i8Out: i8o.parts.contains("go") ? i8 : nil)
            }
        case .none: break
        }
        if l.post != nil {
            try tseg(tl, e, "post") { try encodeBranchEpilogue($0, l, resid: ph, rows: M, ffn: false, prefill: true) }
        }
    }

    /// The prefill MoE (`pmoe`) of layer `i` for chunk `c` of `G` (`r` real rows). With `expertStream`, it cuts
    /// the encoder to fetch the experts and returns the current encoder.
    private func encodePrefillMoE(
        _ e: MTLComputeCommandEncoder?, layer i: Int, part c: Int, of G: Int, rows r: Int, bulk: Bool,
        tl: GPUTimeline?, cb: MTLCommandBuffer?
    ) throws -> MTLComputeCommandEncoder? {
        let pmoe = pmoe!
        bindPrefillMoE(pmoe, layer: i)
        if let es = expertStream, let cb, let ee = e {
            if bulk {
                // the CPU is already fetching ahead what the layer is missing; wait once per group.
                var ne = c == 0 ? es.prefillBegin(cb, ee, layer: i, bulk: true, pmoe: pmoe, rows: r) : ee
                try pmoe.encodeAll(ne)
                if c == G - 1 { ne = es.prefillEnd(cb, ne) }
                return ne
            }
            // after the router, the CPU fetches the experts the chunk uses and the GPU goes on.
            precondition(G == 1)
            try pmoe.encode(ee, .router)
            var ne = es.prefillBegin(cb, ee, layer: i, bulk: false, pmoe: pmoe, rows: pmoe.liveRows)
            for p in PrefillMoE.Piece.allCases where p != .router { try pmoe.encode(ne, p) }
            ne = es.prefillEnd(cb, ne)
            return ne
        }
        if let tl {
            for p in PrefillMoE.Piece.allCases {
                try tl.seg("moe." + p.rawValue) { try pmoe.encode($0, p) }
            }
        } else {
            try pmoe.encodeAll(e!)
        }
        return e
    }

    /// Points the prefill MoE at layer `i`'s weights, biases and slot table (its work buffers are shared).
    func bindPrefillMoE(_ pmoe: PrefillMoE, layer i: Int) {
        let l = layers[i]
        pmoe.router = l.moeW[0]; pmoe.sharedGate = l.moeW[1]; pmoe.sGate = l.moeW[2]
        pmoe.sUp = l.moeW[3]; pmoe.sDown = l.moeW[4]; pmoe.eGate = l.moeW[5]; pmoe.eUp = l.moeW[6]
        pmoe.eDown = l.moeW[7]
        pmoe.setBiases(l.moe)
        pmoe.smap = l.moe.smap
    }

    /// Drafter context of a prefill chunk with its captures in `caps`; either with the one-row ring of the
    /// last chunk (`pendingRingCapture`), or the drafter ring as it stands at the end of the cut (the next chunks
    /// overwrite it).
    private func encodeDrafterContext(
        _ e: MTLComputeCommandEncoder, _ dr: DFlash2Drafter, caps: MTLBuffer, part: PrefillPart, rows: Int,
        nvalid: Int, pos0: Int
    ) throws {
        let own = dr.caps
        dr.caps = caps
        defer { dr.caps = own }
        try dr.encodeContext(e, rows: rows, nvalid: nvalid, pos0: pos0, ringAt: part.head ? pendingRingCapture : [])
        if let k = part.endSlot, let dt = captureDraftTargets?[k] {
            for (l, t) in zip(dr.layers, dt) {
                try copyBytes(e, l.kc, t.kc)
                try copyBytes(e, l.vt, t.vt)
            }
        }
    }

    /// Redoes the int8 state at the next chunk (per-row scales, buffers), e.g. after changing its options
    /// (lab-dev kl --cfgs).
    public func reconfigureI8() { denseI8 = nil }

    func makeDenseI8() throws {
        var mats: [DenseQ] = []
        for l in layers {
            switch l.mixer {
            case .gdn(let g): mats += [g.proj, g.outProj]
            case .attn(let a): mats += [a.proj, a.oProj]
            case .conv(let c): mats += [c.inProj, c.outProj]
            case .none: break
            }
            if let f = l.dense { mats += [f.gu, f.down] }
        }
        // 8-bit ones do not go through int8 (DenseQ.int8OK).
        mats = mats.filter { $0.tern || $0.int8OK }
        // A MoE whose GDN and attention matrices are all 8-bit (Ornith's mixed variant) has nothing to rotate:
        // no scratch, and the prefill stays exact (denseI8 nil).
        guard !mats.isEmpty else { return }
        // Ternary ones (Bonsai) need no precomputed per-row scales (their int8 is scaled per row in each chunk,
        // tern_i8row), but do need room in the int8 weight buffer.
        let conv = mats.filter { !$0.tern }
        let sc = DenseI8Scratch(
            ctx: ctx, maxNK: mats.map { $0.N * $0.K }.max() ?? 16, maxN: mats.map { $0.N }.max()!,
            maxRows: chunkRows, maxK: mats.map { $0.K }.max()!)
        // Per-row scales of all rotated matrices, once (each chunk's conversion makes a single pass).
        try ctx.run { e in for m in conv { try m.precomputeRowScales(ctx, e, scratch: sc) } }
        denseI8 = sc
    }

    func configurePrefillMoE(_ pmoe: PrefillMoE, rows M: Int) {
        pmoe.rows = M
        // Large chunks: 48-row tiles with K steps of 32 (2048 rows: gate+up 5.27 → 5.05 ms, down 2.87 →
        // 2.79); with ≤ 1024 rows they do not win.
        pmoe.tiles = M >= 1536 ? PrefillMoE.Tiles(bmA: 48, bmB: 48, bkA: 32, bkB: 32) : PrefillMoE.Tiles()
    }

    /// Prefill in balanced chunks of ≤ chunkRows rows (multiples of 64 except the remainder). Returns the argmax of
    /// the last row (its logits stay in `logits`) and the time (wall s). With `continuing`, the ids are appended
    /// from P onto the current state (which cannot have tape rows left to fold: `foldPending`); otherwise it
    /// starts from zero.
    /// With `split`, `ids[split...]` (≤ chunkRows) is the last chunk, and `ids[..<split]` goes in balanced chunks
    /// as if it were another call; all can go in the same group (a single pass over the SSD experts).
    /// `cuts` (Flash-Next, sorted, before `split`) splits the call as if they were consecutive calls
    /// (`prefill(ids[..<c1], head: false)`, `prefill(ids[c1..<c2], …)`…: same chunks and drafter on the same ones),
    /// but in the same groups; with `cutSlots`, the state at the end of each cut is captured in that capture.
    /// `drafterEnds`: points that count as the end of a call for the drafter window (a ring capture
    /// in the middle of the last chunk has to see the same positions as a pass that ended there).
    public func prefill(
        _ ids: [UInt32], drafter: DFlash2Drafter?, continuing: Bool = false, head: Bool = true, split: Int? = nil,
        cuts: [Int] = [], cutSlots: [Int] = [], drafterEnds: [Int] = []
    ) throws
        -> (token: UInt32, seconds: Double)
    {
        precondition(!ids.isEmpty)
        precondition(split.map { $0 > 0 && $0 < ids.count && ids.count - $0 <= chunkRows } ?? true)
        let caps = prefillCaps
        precondition(cuts.isEmpty || (caps.cutsInPass && cuts.first! > 0 && cuts.last! < (split ?? ids.count)))
        precondition(cutSlots.isEmpty || cutSlots.count == cuts.count)
        precondition(zip(cuts, cuts.dropFirst()).allSatisfy { $0 < $1 })
        if continuing { precondition(nacc == 0, "prefill over unfolded rows") } else { reset() }
        guard try ensureContext(P + ids.count + DFlash2Drafter.margin(drafter)) else {
            throw SafetensorsError.io("KVPool: the prompt does not fit in the context")
        }
        setDecodeChunk(promptTokens: P + ids.count)
        let t0 = DispatchTime.now().uptimeNanoseconds
        var ranges: [Range<Int>] = []
        // Balanced chunks of each span (between cuts; the second to last, up to `split`, and the last, up to the end).
        // With the lossy prefill, the thresholds (absolute positions) also split the spans, so that
        // no chunk crosses them (lossyBounds).
        let bounds = lossyBounds(from: P, count: ids.count)
        var lo = 0
        for hi in cuts + (split.map { [$0] } ?? []) + [ids.count] {
            let inner = bounds.filter { $0 > lo && $0 < hi }
            for (a, b) in zip([lo] + inner, inner + [hi]) {
                let n = (b - a + chunkRows - 1) / chunkRows
                let per = min(chunkRows, ((b - a + n - 1) / n + 63) / 64 * 64)
                ranges += stride(from: a, to: b, by: per).map { $0..<min(b, $0 + per) }
            }
            lo = hi
        }
        fitCaptures(&ranges, headStart: split ?? cuts.last ?? 0, count: ids.count)
        var last: MTLCommandBuffer?
        // With experts on the SSD, up to `expertPrefillGroup` chunks per group, layer by layer (at most two with
        // drafter). Flash-Next: `flashPrefillGroup`, and the in-pass captures also in a group.
        let G = expertStream == nil || (pendingCapture != nil && !caps.groupWithCaptures) ? 1 : caps.groupParts
        var ri = 0
        while ri < ranges.count {
            var parts: [PrefillPart] = []
            let maxDr = caps.maxDraftParts
            while ri < ranges.count && parts.count < G {
                let r = ranges[ri], end = r.upperBound
                // The drafter only needs the last positions of its widest window (2048; 8192 with the full layer of
                // a DFlash v1) of each call (for cuts: of each span up to a cut, as if they were separate calls).
                let callEnd = (cuts + drafterEnds).filter { $0 >= end }.min() ?? ids.count
                let dr = end > callEnd - (drafter?.maxWindow ?? DFlash2Drafter.window) ? drafter : nil
                if dr != nil && parts.filter({ $0.drafter != nil }).count == maxDr { break }
                let ci = cutSlots.isEmpty ? nil : cuts.firstIndex(of: end)
                parts.append(
                    PrefillPart(
                        ids: ids[r], drafter: dr, head: head && end == ids.count, endSlot: ci.map { cutSlots[$0] }))
                ri += 1
            }
            guard let cb = ctx.queue.makeCommandBuffer(), let e = cb.makeComputeCommandEncoder() else {
                throw SafetensorsError.io("cb")
            }
            let e2 = try encodePrefillGroup(e, parts: parts, cb: cb)
            e2?.endEncoding()
            cb.commit()
            last = cb
        }
        last!.waitUntilCompleted()
        if let err = last!.error { throw err }
        let t1 = DispatchTime.now().uptimeNanoseconds
        return (top1.contents().load(as: UInt32.self), Double(t1 - t0) / 1e9)
    }
}

// Per-position lossy prefill: each row goes in int8 according to its absolute position,
// not according to how the prompt is split, so the state at a cut does not depend on the chunks.
extension HybridModel {
    /// Position from which the prefill's dense GEMMs go in W8A8 with --int8: the family's (ModelSpec) unless
    /// the options set it (lab-dev kl, `R=` card).
    public var int8MinPos: Int { options.int8.minPos ?? spec.int8MinPos }

    /// Whether the MoE experts take int8 activations in the int8 chunks: a fixed property of the family
    /// (`ModelSpec.int8Experts`, Flash-Next only).
    public var int8Experts: Bool { spec.int8Experts }

    /// Lossy prefill (options.prefillInt8/prefillQK8; in `lab serve` on by default for MoEs): which chunks go with
    /// dense GEMMs in rotated W8A8 (from `int8MinPos`) and which with Q·Kᵀ in int8 (from qk8MinCtx), by their
    /// start position. Common to all families (previously, outside Flash-Next, by chunk size and end).
    func prefillLossyChoices(starts p0s: [Int]) throws -> (
        i8: [DenseI8Scratch?], qk8: [(q: MTLBuffer, s: MTLBuffer)?]
    ) {
        let o = options, i8o = o.int8
        if o.prefillInt8 && denseI8 == nil { try makeDenseI8() }
        // Decided by the chunk's start position, and no chunk crosses the thresholds (lossyBounds): each
        // row goes in int8 according to its position, not the chunk size or what comes after.
        let i8Alls = p0s.map { o.prefillInt8 && $0 >= int8MinPos ? denseI8 : nil }
        if o.prefillQK8 && atq8 == nil {
            // q rows (query heads × chunk rows) and their width, for the larger attention shape.
            let shapes = [ctx.dims] + (fullCtx.map { [$0.dims] } ?? [])
            let rows = shapes.map(\.attnQH).max()! * chunkRows, hd = shapes.map(\.attnHD).max()!
            atq8 = (ctx.buffer(length: rows * hd), ctx.buffer(length: rows * 4))
        }
        // GLM's latent MLA: no Q·Kᵀ int8. Its long chunks take the decompressed attention (exact, MLAProjection
        // .decompressed) whatever their position, so a row would go lossy or exact depending on its chunk's rows.
        let qk8s = p0s.map { o.prefillQK8 && spec.mla == nil && $0 >= i8o.qk8MinCtx ? atq8 : nil }
        prefillI8Chunks += i8Alls.filter { $0 != nil }.count
        prefillQK8Chunks += qk8s.filter { $0 != nil }.count
        return (i8Alls, qk8s)
    }

    /// Forced chunk boundaries of the prefill of `n` rows from position `p0` (relative to p0): with the lossy prefill,
    /// the int8 thresholds (`int8MinPos`) and Q·Kᵀ int8 (qk8MinCtx), in absolute positions. Previously
    /// int8 was decided by chunk size (≥ minRows rows) and Q·Kᵀ by its end, so a row went in int8 or
    /// not depending on how the prompt was split: the state at a cut captured inside the pass differed from the prefill
    /// split at that cut.
    func lossyBounds(from p0: Int, count n: Int) -> [Int] {
        let o = options
        var t: [Int] = []
        if o.prefillInt8 && int8MinPos > 0 { t.append((int8MinPos + 63) / 64 * 64) }
        if o.prefillQK8 && spec.mla == nil { t.append((o.int8.qk8MinCtx + 63) / 64 * 64) }
        return Set(t.map { $0 - p0 }.filter { $0 > 0 && $0 < n }).sorted()
    }

    /// In-pass captures (`pendingCapture` and `pendingRingCapture`) are rows of the last span, which
    /// started at `headStart` and was a single chunk; if a forced boundary splits it, they count from the new last
    /// chunk. If any capture falls before the boundary (the prompt ends right after the threshold), the last span is
    /// a single chunk again (and runs exact, because it starts before the threshold).
    func fitCaptures(_ ranges: inout [Range<Int>], headStart h0: Int, count n: Int) {
        guard let pc = pendingCapture, let h1 = ranges.last?.lowerBound, h1 > h0 else { return }
        if pc.rows.allSatisfy({ $0 >= h1 - h0 }) {
            pendingCapture = (rows: pc.rows.map { $0 - (h1 - h0) }, targets: pc.targets)
            // The ring of the "conversation" cut is one of those rows and is also counted from the new
            // last chunk (previously not: with the threshold far from the cut, `ringAt` went past the chunk's rows and
            // lab serve crashed; otherwise, it captured another row's ring).
            pendingRingCapture = pendingRingCapture.map { (rows: $0.rows - (h1 - h0), targets: $0.targets) }
        } else {
            ranges.removeAll { $0.lowerBound >= h0 }
            ranges.append(h0..<n)
        }
    }
}
