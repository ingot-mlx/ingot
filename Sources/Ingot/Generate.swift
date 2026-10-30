import Foundation
import Metal

// Greedy generation loops of HybridModel: normal (one row per step), speculative with DFlash2 (one wait per
// round) and chained speculative (the CPU enqueues round N+1 before waiting for N; the one ingot-dev gen uses).

extension HybridModel {
    public struct GenStats {
        public var tokens: [UInt32] = []
        public var prefillSeconds = 0.0
        public var decodeSeconds = 0.0  // first token to last
        public var rounds = 0
        public var acceptedHist = [Int](repeating: 0, count: 17)
        public var gpuMs: [Double] = []  // per round
        public var encodeMs: [Double] = []  // CPU encoding each round
        /// Chained rounds: GPU start to end of each round (for the first, its GPU) and gap between consecutive
        /// command buffers (gpuStartTime of one minus gpuEndTime of the previous; negative if they overlap).
        public var wallMs: [Double] = [], gapMs: [Double] = []
        public var rowsHist = [Int](repeating: 0, count: 17)  // rows verified per round
        public var tokensPerRound: Double { rounds > 0 ? Double(tokens.count - 1) / Double(rounds) : 0 }
        /// Mean rows verified per round.
        public var rowsPerRound: Double {
            let n = rowsHist.reduce(0, +)
            return n > 0 ? Double(rowsHist.enumerated().reduce(0) { $0 + $1.0 * $1.1 }) / Double(n) : 0
        }
        /// Mean and median of wall and GPU time per round (the same statistic for both) and gaps between command
        /// buffers. Wall time goes from GPU end to end, so it does not include the encoding of the first round.
        public var roundSummary: String {
            func mean(_ a: [Double]) -> Double { a.isEmpty ? 0 : a.reduce(0, +) / Double(a.count) }
            func med(_ a: [Double]) -> Double { a.isEmpty ? 0 : a.sorted()[a.count / 2] }
            return String(
                format:
                    "round: wall mean %.2f ms (median %.2f), GPU mean %.2f (median %.2f), "
                    + "wall - GPU mean %.2f; gap between command buffers mean %.3f ms (median %.3f, max %.2f, "
                    + "%d > 0.1 ms); %d CPU waits",
                mean(wallMs), med(wallMs), mean(gpuMs), med(gpuMs), mean(wallMs) - mean(gpuMs), mean(gapMs), med(gapMs),
                gapMs.max() ?? 0, gapMs.filter { $0 > 0.1 }.count, deferred)
        }
        public var rowsSummary: String {
            String(format: "%.2f rows per round", rowsPerRound)
        }
        public var finish = FinishReason.length
        /// Context copy: copy rounds, their verified rows, emitted tokens and GPU ms; times the CPU waited for the
        /// previous round to decide (and of those, the ones that ended in a DFlash2 round).
        public var copyRounds = 0, copyRows = 0, copyTokens = 0, copyGpuMs = 0.0, deferred = 0, deferredNoCopy = 0
        public var copySummary: String {
            String(
                format:
                    "copy %d/%d rounds (%.0f%%), %.2f tok/copy round, %d mean rows, %.0f%% tokens; waits %d (%d w/o)",
                copyRounds, rounds, rounds > 0 ? 100 * Double(copyRounds) / Double(rounds) : 0,
                copyRounds > 0 ? Double(copyTokens) / Double(copyRounds) : 0,
                copyRounds > 0 ? copyRows / copyRounds : 0,
                tokens.count > 1 ? 100 * Double(copyTokens) / Double(tokens.count - 1) : 0, deferred, deferredNoCopy)
        }
    }

    /// Record of each chained round (REC_STRIDE, REC_P and RND_PER_ROUND of yy_model.metal).
    static let recStride = 24, recP = 20, recRows = 21, rndPerRound = 32

    static let eos: Set<UInt32> = [248044, 248046]

    /// Normal decoding (one row per step), greedy.
    public func generate(prompt: [UInt32], maxTokens: Int, stopAtEOS: Bool = true, genStart: Int? = nil) throws
        -> GenStats
    {
        var st = GenStats()
        let (first, _, ps) = try preparePrompt(prompt, from: 0, genStart: genStart, drafter: nil)
        st.prefillSeconds = ps
        st.tokens = [first]
        let t0 = DispatchTime.now().uptimeNanoseconds
        var cur = first
        while st.tokens.count < maxTokens && !(stopAtEOS && eosSet.contains(cur)) {
            if P + 1 > ctxLimit { try ensureContext(P + 1 + KVPool.extent) }  // KVPool (GPU stopped)
            precondition(P + 1 <= ctxLimit)
            tok.contents().storeBytes(of: cur, as: UInt32.self)
            let te = DispatchTime.now().uptimeNanoseconds
            guard let cb = ctx.queue.makeCommandBuffer(), let e = cb.makeComputeCommandEncoder() else {
                throw SafetensorsError.io("cb")
            }
            try encodeVerify(e, rows: 1, drafter: nil, cb: cb).endEncoding()
            st.encodeMs.append(Double(DispatchTime.now().uptimeNanoseconds - te) / 1e6)
            cb.commit()
            cb.waitUntilCompleted()
            if let err = cb.error { throw err }
            st.gpuMs.append((cb.gpuEndTime - cb.gpuStartTime) * 1e3)
            commit(accepted: 1)
            cur = top1.contents().load(as: UInt32.self)
            st.tokens.append(cur)
            st.rounds += 1
        }
        st.decodeSeconds = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9
        return st
    }

    /// Speculative decoding with DFlash2, greedy: each round is a command buffer (drafter context with
    /// the accepted rows of the previous round -> drafter block -> verification of 8 rows) and a wait.
    /// `onRound(m)`: after each round, with the m accepted drafts (that round's confidences remain in `dr.conf`;
    /// used by `ingot-dev rowscal`).
    ///
    /// Forced text (`forced`, ingot-dev accept): the answer is `forced` (its first token acts as first token) and a
    /// draft is valid if it matches the forced token of its position, not the big model's argmax; with `rows`,
    /// the round only accepts within the k rows the row selector would choose with the fixed table (`chooseCPU`).
    /// `onRound` receives those accepted; `acceptedAll` (if not nil), those that would be accepted with all 8 rows.
    public func generateSpec(
        prompt: [UInt32], drafter dr: DFlash2Drafter, maxTokens: Int, stopAtEOS: Bool = true,
        genStart: Int? = nil, forced: [UInt32]? = nil, rows: SpecRowsConfig? = nil,
        acceptedAll: ((Int, Int) -> Void)? = nil, onRound: ((Int) -> Void)? = nil
    ) throws -> GenStats {
        var st = GenStats()
        let (first0, _, ps) = try preparePrompt(prompt, from: 0, genStart: genStart, drafter: dr)
        let first = forced?.first ?? first0
        let maxTokens = forced.map { min(maxTokens, $0.count) } ?? maxTokens
        st.prefillSeconds = ps
        st.tokens = [first]
        let t0 = DispatchTime.now().uptimeNanoseconds
        var anchor = first
        var pending = 0, pendingPos = 0  // accepted rows whose context is missing in the drafter
        let B = dr.block
        let tp = tok.contents().bindMemory(to: UInt32.self, capacity: B)
        let t1p = top1.contents().bindMemory(to: UInt32.self, capacity: B)
        while st.tokens.count < maxTokens && !(stopAtEOS && eosSet.contains(anchor)) {
            if P + B > ctxLimit { try ensureContext(P + B + KVPool.extent) }  // KVPool (GPU stopped)
            precondition(P + B <= ctxLimit)
            tp[0] = anchor
            dr.setBlockTokens(anchor: anchor)
            let te = DispatchTime.now().uptimeNanoseconds
            guard let cb = ctx.queue.makeCommandBuffer(), let e = cb.makeComputeCommandEncoder() else {
                throw SafetensorsError.io("cb")
            }
            if pending > 0 { try dr.encodeContext(e, rows: B, nvalid: pending, pos0: pendingPos) }
            try dr.encodeBlock(e, P: P, embed: embed, lmHead: lmHead, vtok: tok)
            try encodeVerify(e, rows: B, drafter: dr, cb: cb).endEncoding()
            st.encodeMs.append(Double(DispatchTime.now().uptimeNanoseconds - te) / 1e6)
            cb.commit()
            cb.waitUntilCompleted()
            if let err = cb.error { throw err }
            st.gpuMs.append((cb.gpuEndTime - cb.gpuStartTime) * 1e3)
            // Greedy acceptance: draft i (tok[i]) is valid if it matches the argmax of row i - 1 (or the forced token
            // of its position).
            var m = 0
            if let f = forced {
                let j = st.tokens.count
                while m < B - 1 && j + m < f.count && tp[m + 1] == f[j + m] { m += 1 }
                let k =
                    rows.map { cfg in
                        cfg.chooseCPU(
                            conf: Array(
                                UnsafeBufferPointer(
                                    start: dr.conf.contents().bindMemory(to: Float.self, capacity: 8), count: 8)),
                            ctx: P)
                    } ?? B
                acceptedAll?(m, k)
                m = min(m, k - 1)
                st.rowsHist[k] += 1
                for i in 0..<B { t1p[i] = i <= m && j + i < f.count ? f[j + i] : t1p[i] }
            } else {
                while m < B - 1 && tp[m + 1] == t1p[m] { m += 1 }
            }
            onRound?(m)
            var stop = false
            for i in stride(from: 1, through: m, by: 1) {
                st.tokens.append(tp[i])
                if (stopAtEOS && eosSet.contains(tp[i])) || st.tokens.count >= maxTokens { stop = true; break }
            }
            st.acceptedHist[m] += 1
            st.rounds += 1
            pendingPos = P
            pending = m + 1
            commit(accepted: m + 1)
            anchor = t1p[m]
            if stop { break }
            st.tokens.append(anchor)
        }
        st.decodeSeconds = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9
        if st.tokens.count > maxTokens { st.tokens.removeLast(st.tokens.count - maxTokens) }
        return st
    }

    /// Why a generation stopped.
    public enum FinishReason: String, Sendable { case stop, length, contextFull, cancelled }

    /// Prefill + chained greedy decoding (`decode`; the one ingot-dev gen uses): with a drafter, `decodeSpec`; without one
    /// (LFM2, the ingot serve decode), `decodePlain`, or `decodeSpec` without a drafter if context copy is on.
    public func generatePipelined(
        prompt: [UInt32], drafter dr: DFlash2Drafter?, maxTokens: Int, stopAtEOS: Bool = true, genStart: Int? = nil
    ) throws -> GenStats {
        let (first, _, ps) = try preparePrompt(prompt, from: 0, genStart: genStart, drafter: dr)
        let ci =
            options.copyEnabled(moe: isMoE, drafter: dr != nil)
            ? CopyIndex(capacity: prompt.count + maxTokens + 64) : nil
        ci?.sync(to: prompt)
        var rng = SplitMix64(seed: 0)
        var st = try decode(
            first: first, drafter: dr, maxTokens: maxTokens, rng: &rng, stopTokens: stopAtEOS ? eosSet : [],
            copyIndex: ci)
        st.prefillSeconds = ps
        return st
    }

    /// Chained decoding from the state of `preparePrompt`: `decodeSpec` with a drafter or with `copyIndex` (context
    /// copy without a drafter: one-row rounds and copy rounds); otherwise `decodePlain`. Shared by ingot-dev gen
    /// (`generatePipelined`) and the ingot serve engine (`Engine.generate`).
    public func decode(
        first: UInt32, drafter dr: DFlash2Drafter?, maxTokens: Int, sampling: SamplingParams = .init(),
        rng: inout SplitMix64, stopTokens: Set<UInt32>, copyIndex: CopyIndex? = nil,
        copyState: CopyPolicyState? = nil, onTokens: (([UInt32]) -> Bool)? = nil
    ) throws -> GenStats {
        if dr != nil || copyIndex != nil {
            return try decodeSpec(
                first: first, drafter: dr, maxTokens: maxTokens, sampling: sampling, rng: &rng, stopTokens: stopTokens,
                copyIndex: copyIndex, copyState: copyState, onTokens: onTokens)
        }
        return try decodePlain(
            first: first, maxTokens: maxTokens, sampling: sampling, rng: &rng, stopTokens: stopTokens,
            onTokens: onTokens)
    }

    /// Chained speculative decoding from the current state (that of `preparePrompt`: the last rows of the
    /// prompt may still be on the GDN tape), with `first` as first emitted token (the one that comes out of the prompt,
    /// still outside the cache). The round state (P, rows to
    /// fold, anchor) lives on the GPU and a kernel accepts the drafts at the end of each round, so the CPU encodes
    /// and enqueues round N+1 while the GPU runs N (the GPU never waits for the CPU).
    ///
    /// With `copyIndex` (history = the prompt; CopySpec.swift) and copy on (options.copy), a round may be **a copy**:
    /// anchor + up to 15 tokens that followed the previous occurrence of the history suffix, verified without going
    /// through the drafter, when the policy estimates it gives more tokens per ms than DFlash2. As the CPU needs exact
    /// history to propose, before a copy round the previous one is awaited (only when the history it already
    /// knows has an acceptable copy). Verification is the same, so the output does not change.
    ///
    /// Stops when a token from `stopTokens` comes out (it is emitted), on reaching `maxTokens` emitted, when `onTokens`
    /// returns false (after the last token of that batch) or when no other round fits in the cache. On finishing, the
    /// state is left **reusable**: the extra round that was already enqueued is rolled back (the GDN keeps the state
    /// from before each block and the block's tape in the ping-pong, so any row of the last two rounds can be
    /// returned to), the pending rows are folded and P ends right after the last emitted token (or the last one
    /// that entered the cache if the round that would contain it was never enqueued). Tokens in the cache: the prompt
    /// plus the first `P - P_initial` emitted.
    ///
    /// SwiftLint: `encodeRound`, `copyProposal` and the round loop share the generation's local state;
    /// pulling them out into methods would pass that state through a struct on every round.
    public func decodeSpec(  // swiftlint:disable:this function_body_length - hot path: each round
        first: UInt32, drafter dr: DFlash2Drafter?, maxTokens: Int, sampling: SamplingParams = .init(),
        rng: inout SplitMix64, stopTokens: Set<UInt32>, copyIndex: CopyIndex? = nil,
        copyState: CopyPolicyState? = nil, onTokens: (([UInt32]) -> Bool)? = nil
    ) throws -> GenStats {
        var st = GenStats()
        st.tokens = [first]
        if let rt = routeTrace {
            rt.begin(label: (rt.nextLabel ?? "decode") + " P0=\(P)")
            rt.nextLabel = nil
        }
        let t0 = DispatchTime.now().uptimeNanoseconds
        // Without a drafter (LFM2): normal rounds are one row and only copy speculates.
        let B = dr?.block ?? 1, ring = 4, RS = Self.recStride
        let dtok = dr?.dtok ?? ctx.buffer(length: 16 * 4)
        let sbuf = ctx.buffer(length: 8 * 4), rec = ctx.buffer(length: ring * RS * 4)
        let sp = sbuf.contents().bindMemory(to: UInt32.self, capacity: 8)
        for i in 0..<8 { sp[i] = 0 }
        sp[0] = UInt32(P); sp[1] = UInt32(nacc); sp[4] = first  // prompt rows still on the tape (preparePrompt)
        tok.contents().storeBytes(of: first, as: UInt32.self)
        dr?.setBlockTokens(anchor: first)
        var finish = FinishReason.length
        if stopTokens.contains(first) { finish = .stop }
        let keepGoing = onTokens?([first]) ?? true
        if !keepGoing { finish = .cancelled }
        // Context the KV admits now (KVPool: what is mapped; grows below, with the tail empty, up to maxCtx).
        var lim = ctxLimit
        if P + 2 * B > lim && lim < maxCtx { _ = try ensureContext(P + 2 * B + KVPool.extent); lim = ctxLimit }
        if stopTokens.contains(first) || !keepGoing || maxTokens <= 1 || P + 2 * B > lim {
            st.finish = P + 2 * B > lim && finish == .length ? .contextFull : finish
            try foldPending(drafter: nil, context: nil)
            return st
        }
        // Uniforms of each round (2 per row), indexed on the GPU by the round counter.
        // Penalties: counters on the GPU (penalize_rows before the top-64, count_emitted after the accept); greedy
        // with penalties takes the sampling path with top-k 1 (chainSampling).
        let pen = sampling.hasPenalties
        // With a reproducible output (greedy or a seed) the copy policy does not learn from measured ms,
        // with or without a drafter, so the round structure depends only on the tokens. Otherwise the timings change
        // which rounds copy, and with them the uniforms of each round (sampling) or where the last round stops and
        // which token the next request prefills (greedy). With DFlash2 (Heretic agent replay): same ms per round
        // (+0.2%), -3% ms per token. Without a drafter the priors use `CopyPolicy.defaultShape` (recorded pi
        // conversation, ABBA: gpt-oss 20b -0.87% ms per token, Gemma 4 26B-A4B +0.72%). Unseeded sampling keeps the
        // timings: its text is not reproducible anyway.
        let untimed = options.bench.copyPolicyUntimed || sampling.greedy || sampling.seed != nil
        startPenalties(first: first, params: sampling)
        let sampling = sampling.chainSampling
        let rnd = sampling == nil ? nil : roundUniforms(maxTokens: maxTokens, rng: &rng)
        // Context copy: the index history is the prompt; each emitted token is appended to it.
        let copyOn = copyIndex != nil && options.copyEnabled(moe: isMoE, drafter: dr != nil)
        var pol =
            copyState?.policy
            ?? CopyPolicy(maxT: CopyPolicy.maxRows, margin: CopyPolicy.defaultMargin, plainBase: dr == nil)
        defer { copyState?.policy = pol }
        copyIndex?.append(first)
        var pKnown = P  // exact P of the last enqueued round whose predecessor is already known
        var inflight: [MTLCommandBuffer] = []
        var r = 0
        // Rows verified per DFlash2 round. Row selection: with family calibration (rowsConfig), each round is enqueued
        // with B rows and the GPU chooses how many count; without it, all B. The v1 drafter has no selector: its
        // confidence is that of the argmax over the top-16 (`dpick_top1`). options.bench.specRowsFixed (benches)
        // fixes them.
        let fixedRows = options.bench.specRowsFixed
        let rowsCfg = dr != nil && fixedRows == nil ? rowsConfig : nil
        let gpuRows = rowsCfg != nil
        // Tree verification: DFlash2 rounds with an options.treeNodes-row tree (greedy, dense). On accepting, the path
        // is compacted into contiguous rows, so copy rounds, rollback and foldPending do not change. With key chunks
        // larger than treeMaxChunk (prompts of >= 16K) row-wise tree attention costs more than it gains
        // (25K: -3 to -6%): chain.
        let decodeChunk =
            layers.lazy.compactMap { if case .attn(let a) = $0.mixer { return a.fixedChunk } else { return nil } }.first
            ?? 0
        let treeNodes = options.treeNodes, treeOK = dr.map { !$0.v1 } ?? false
        let treeN =
            treeOK && sampling == nil && !isMoE && !spec.hasSlidingWindow && treeNodes >= 2
                && decodeChunk <= Self.treeMaxChunk ? treeNodes : 0
        struct RoundInfo {
            let rows: Int; let copyMatch: Int?; let encodeMs: Double; let gpuRows: Bool; var tree = false
        }
        var info: [RoundInfo] = []
        let tp = tok.contents().bindMemory(to: UInt32.self, capacity: AttentionLayer.maxRows)
        /// DFlash2 round (`copy` nil) or copy round (anchor + `copy.drafts`, no drafter; only with an empty tail, since
        /// the CPU writes the copied tokens into `tok`). Both start by writing into the drafter the context of the
        /// accepted rows of the previous round (which may have had up to 16).
        func encodeRound(posBound: Int, copy: (drafts: [UInt32], match: Int)? = nil) throws -> MTLCommandBuffer {
            let rows: Int
            if let c = copy {
                precondition(inflight.isEmpty && c.drafts.count + 1 <= AttentionLayer.maxRows)
                rows = c.drafts.count + 1
                for (i, t) in c.drafts.enumerated() { tp[1 + i] = t }
            } else {
                rows = dr == nil ? 1 : treeN > 0 ? treeN : gpuRows ? B : fixedRows ?? B
            }
            let onGPU = gpuRows && copy == nil && treeN == 0
            let isTree = treeN > 0 && copy == nil
            precondition(posBound + rows <= lim)
            let prevRows = info.last?.rows ?? B
            let te = DispatchTime.now().uptimeNanoseconds
            let cb = ctx.queue.makeCommandBuffer()!
            var e = roundTrace?.begin(cb, label: "dr.ctx") ?? cb.makeComputeCommandEncoder()!
            if let dr {
                let ctxRows = max(B, prevRows)
                try dr.encodeContext(e, rows: ctxRows, nvalid: ctxRows, pos0: 0, state: sbuf)
                if copy == nil {
                    let rc: ((MTLComputeCommandEncoder, String) -> MTLComputeCommandEncoder)? =
                        roundTrace?.cuts == true ? { self.tcut(cb, $0, $1, -1) } : nil
                    e =
                        try dr.encodeBlock(
                            e, P: 0, embed: embed, lmHead: lmHead, vtok: tok, state: sbuf,
                            tree: isTree ? (treeState, treeN) : nil, roundCut: rc) ?? e
                }
                if onGPU, let rowsCfg {
                    e = tcut(cb, e, "rows.choose", -1)
                    try encodeRowsChoose(e, drafter: dr, state: sbuf, ctx: posBound, config: rowsCfg)
                }
            }
            if isTree { setTree(treeState) }
            e = try encodeVerify(e, rows: rows, drafter: dr, state: sbuf, posBound: posBound, rowLimit: onGPU, cb: cb)
            if isTree { setTree(nil) }
            var rg = UInt32(ring), nr = UInt32(onGPU ? 0 : rows)
            e = tcut(cb, e, "accept", -1)
            if isTree, let dr {
                try encodeTreeFinish(e, tree: treeState, state: sbuf, rec: rec, ring: ring, drafter: dr)
            } else if let sampling, let rnd {
                if pen { try encodePenalties(e, rows: rows, params: sampling) }
                try encodeTopK(e, rows: rows, params: sampling)
                var kp = SampleKernelParams(sampling, vocab: vocab)
                e.setComputePipelineState(try ctx.pipeline("spec_accept_sample"))
                for (i, b) in [sbuf, tok, sampleBufs.idx, sampleBufs.val, sampleBufs.z, dtok, rec, rnd].enumerated() {
                    e.setBuffer(b, offset: 0, index: i)
                }
                e.setBytes(&kp, length: MemoryLayout<SampleKernelParams>.size, index: 8)
                e.setBytes(&rg, length: 4, index: 9)
                e.setBytes(&nr, length: 4, index: 10)
            } else {
                e.setComputePipelineState(try ctx.pipeline("spec_accept"))
                for (i, b) in [sbuf, tok, top1, dtok, rec].enumerated() { e.setBuffer(b, offset: 0, index: i) }
                e.setBytes(&rg, length: 4, index: 5)
                e.setBytes(&nr, length: 4, index: 6)
            }
            if !isTree {
                e.dispatchThreads(
                    MTLSize(width: 1, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
            }
            if pen { try encodeCountEmitted(e, state: sbuf, rec: rec, ring: ring) }
            e.endEncoding()
            roundTrace?.endEncode()
            let ems = Double(DispatchTime.now().uptimeNanoseconds - te) / 1e6
            st.encodeMs.append(ems)
            info.append(RoundInfo(rows: rows, copyMatch: copy?.match, encodeMs: ems, gpuRows: onGPU, tree: isTree))
            cb.commit()
            return cb
        }
        func record(_ i: Int) -> (m: Int, toks: [UInt32], p: Int, rows: Int) {
            let rp = (rec.contents() + (i % ring) * RS * 4).bindMemory(to: UInt32.self, capacity: RS)
            let m = Int(rp[0])
            return (m, (1...(m + 1)).map { rp[$0] }, Int(rp[Self.recP]), Int(rp[Self.recRows]))
        }
        /// Copy proposal the policy accepts with the history the CPU knows (copied tokens, match).
        func copyProposal(room: Int) -> (drafts: [UInt32], match: Int)? {
            guard copyOn, let ci = copyIndex else { return nil }
            let maxK = min(pol.maxT - 1, room - 1, maxTokens - st.tokens.count)
            guard maxK >= 1 else { return nil }
            let p = ci.propose(maxK: maxK)
            guard !p.isEmpty, let T = pol.choose(k: p.count, match: ci.lastMatch) else { return nil }
            return (Array(p.prefix(T - 1)), ci.lastMatch)
        }
        // Round 0 with known P; then, each new round is enqueued before waiting for the previous one, unless the
        // known history (up to round r - 1) already has a copy the policy would accept: then round r is awaited
        // and decided with the exact history (the GPU waits as long as the CPU takes to encode the round).
        inflight.append(try encodeRound(posBound: pKnown, copy: copyProposal(room: lim - pKnown)))
        var lastRound = (m: 0, p: 0), lastPos = P - 1  // round where it stopped and position of the last emitted token
        var extra = false
        var lastGpuEnd = 0.0
        while true {
            // The next round has P <= pKnown + rows of round r.
            let bound = pKnown + info[r].rows
            // With the routing trace each round waits for the previous one: the MoE buffers belong to a single round.
            // KVPool: if the extra round would fall outside the mapped KV, the tail is left empty and mapped below.
            let grow = bound + 2 * B > lim && lim < maxCtx
            let defer_ = grow || routeTrace != nil || (copyOn && copyProposal(room: lim - bound) != nil)
            // ingot serve batching: no further round, so the last emitted token stays outside the cache.
            let yieldNow = decodeYield?() ?? false
            extra = !yieldNow && !defer_ && bound + B <= lim
            if extra { inflight.append(try encodeRound(posBound: bound)) }
            let cb = inflight.removeFirst()
            cb.waitUntilCompleted()
            if let err = cb.error { throw err }
            let gms = (cb.gpuEndTime - cb.gpuStartTime) * 1e3
            st.gpuMs.append(gms)
            if lastGpuEnd > 0 { st.gapMs.append((cb.gpuStartTime - lastGpuEnd) * 1e3) }
            st.wallMs.append(lastGpuEnd > 0 ? (cb.gpuEndTime - lastGpuEnd) * 1e3 : gms)
            // Round duration for the policy: from the previous one finishing to this one finishing (includes GPU
            // wait for the CPU before a round decided with the tail empty); clamped in case the CPU dawdled.
            let wall = lastGpuEnd > 0 ? min((cb.gpuEndTime - lastGpuEnd) * 1e3, gms + 3) : gms
            lastGpuEnd = cb.gpuEndTime
            let (m, toks, p0, vrows) = record(r)
            precondition(p0 == pKnown)
            let ri = info[r]
            precondition(
                vrows == ri.rows || (ri.gpuRows && vrows >= 2 && vrows <= ri.rows) || (ri.tree && vrows <= ri.rows)
                    || dr == nil)
            st.rowsHist[ri.gpuRows ? vrows : ri.rows] += 1
            roundTrace?.finish(cb, rows: ri.gpuRows ? vrows : ri.rows, accepted: m, P: p0, stream: expertStream)
            routeTrace?.record(
                model: self, P: p0, rows: ri.gpuRows ? vrows : ri.rows, accepted: m, copy: ri.copyMatch != nil,
                sampled: sampling != nil)
            if let match = ri.copyMatch {
                pol.observeCopy(
                    rows: ri.rows, accepted: m, match: match, ms: untimed ? 0 : wall)
                st.copyRounds += 1; st.copyRows += ri.rows; st.copyTokens += m + 1; st.copyGpuMs += gms
            } else {
                pol.observeBase(tokens: m + 1, ms: untimed ? 0 : wall)
            }
            st.acceptedHist[m] += 1
            st.rounds += 1
            r += 1
            var done = false, emitted: [UInt32] = []
            for (i, t) in toks.enumerated() {
                emitted.append(t)
                st.tokens.append(t)
                copyIndex?.append(t)
                lastPos = p0 + 1 + i
                if stopTokens.contains(t) { done = true; finish = .stop; break }
                if st.tokens.count >= maxTokens { done = true; finish = .length; break }
            }
            if let onTokens, !onTokens(emitted), !done { done = true; finish = .cancelled }
            lastRound = (m, p0)
            pKnown += m + 1
            if !extra && !done && (yieldNow || !defer_) {
                done = true
                finish = yieldNow ? .cancelled : .contextFull
                decodeYielded = yieldNow
            }
            if done { break }
            if defer_ {
                // Empty tail: decides with the exact history.
                st.deferred += 1
                if grow {
                    // The GPU is stopped (empty tail): the KV gains a stretch (or as many as needed).
                    _ = try ensureContext(min(maxCtx, pKnown + 2 * B + KVPool.extent))
                    lim = ctxLimit
                }
                if let c = copyProposal(room: lim - pKnown), pKnown + c.drafts.count + 1 <= lim {
                    inflight.append(try encodeRound(posBound: pKnown, copy: c))
                } else if pKnown + B <= lim {
                    st.deferredNoCopy += 1
                    inflight.append(try encodeRound(posBound: pKnown))
                } else {
                    finish = .contextFull; break
                }
            }
        }
        for cb in inflight {
            cb.waitUntilCompleted()
            if let err = cb.error { throw err }
            roundTrace?.drop(cb)
        }
        st.decodeSeconds = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9
        try rewindExtraRound(lastPos: lastPos, lastRound: lastRound, extra: extra, drafter: dr)
        st.finish = finish
        if st.tokens.count > maxTokens { st.tokens.removeLast(st.tokens.count - maxTokens) }
        return st
    }

    /// Uniforms of each round (2 per row) for sampling, indexed on the GPU by the round counter.
    func roundUniforms(maxTokens: Int, rng: inout SplitMix64) -> MTLBuffer {
        let n = (maxTokens + 2) * Self.rndPerRound
        let b = ctx.buffer(length: n * 4)
        let p = b.contents().bindMemory(to: Float.self, capacity: n)
        for i in 0..<n { p[i] = rng.uniform() }
        return b
    }

    /// At the end of `decodeSpec`, rolls back the extra round: returns to the block (of the round where it stopped,
    /// or the extra one's, N+1) that contains the last emitted token and folds its rows up to it.
    private func rewindExtraRound(
        lastPos: Int, lastRound: (m: Int, p: Int), extra: Bool, drafter dr: DFlash2Drafter?
    ) throws {
        var pending: (rows: Int, pos0: Int)?
        let pN = lastRound.p, mN = lastRound.m
        let k: Int, blockP: Int
        if lastPos <= pN + mN || !extra {
            // Block N: in the GDN, the ping-pong from before the extra round (if enqueued). Its drafter context was
            // written by the extra round; without it, it remains to be written.
            k = min(lastPos, pN + mN) - pN + 1
            blockP = pN
            if extra { flipGDN() } else { pending = (k, pN) }
        } else {
            // Block N+1 (its anchor is the last emitted token): current ping-pong; drafter context pending.
            k = lastPos - (pN + mN + 1) + 1
            blockP = pN + mN + 1
            pending = (k, blockP)
        }
        P = blockP + k
        nacc = k
        lastRewind = blockP == pN ? (extra ? "N" : "N without extra round") : "N+1"
        // dr.caps holds the captures of the last verified round (N+1 if enqueued, otherwise N).
        try foldPending(drafter: dr, context: pending)
    }

    /// Chained decoding without a drafter (B = 1; LFM2) from the state of `preparePrompt`, with `first` as the first
    /// emitted token. The GPU chooses each token (plain_accept[_sample]) and leaves it as input of the next round, so
    /// the CPU enqueues round N+1 before waiting for N (as decodeSpec). On finishing, the state is left
    /// reusable: the extra round that was already enqueued read the last emitted token, which is correct, so
    /// it is enough to fold its row (P ends right after it); if it was never enqueued, P ends after the second to last.
    public func decodePlain(
        first: UInt32, maxTokens: Int, sampling: SamplingParams = .init(), rng: inout SplitMix64,
        stopTokens: Set<UInt32>, onTokens: (([UInt32]) -> Bool)? = nil
    ) throws -> GenStats {
        var st = GenStats()
        st.tokens = [first]
        let t0 = DispatchTime.now().uptimeNanoseconds
        let ring = 4, RS = Self.recStride
        let sbuf = ctx.buffer(length: 8 * 4), rec = ctx.buffer(length: ring * RS * 4)
        let sp = sbuf.contents().bindMemory(to: UInt32.self, capacity: 8)
        for i in 0..<8 { sp[i] = 0 }
        sp[0] = UInt32(P); sp[1] = UInt32(nacc); sp[4] = first
        tok.contents().storeBytes(of: first, as: UInt32.self)
        var finish = FinishReason.length
        if stopTokens.contains(first) { finish = .stop }
        let keepGoing = onTokens?([first]) ?? true
        if !keepGoing { finish = .cancelled }
        // KVPool: without a drafter the KV does not grow here (Engine maps the prompt and 4096 tokens): stops at mapped
        let lim = ctxLimit
        if stopTokens.contains(first) || !keepGoing || maxTokens <= 1 || P + 2 > lim {
            st.finish = P + 2 > lim && finish == .length ? .contextFull : finish
            try foldPending(drafter: nil, context: nil)
            return st
        }
        let pen = sampling.hasPenalties
        startPenalties(first: first, params: sampling)
        let sampling = sampling.chainSampling
        let rnd = sampling == nil ? nil : roundUniforms(maxTokens: maxTokens, rng: &rng)
        let P0 = P
        func encodeRound(posBound: Int) throws -> MTLCommandBuffer {
            let te = DispatchTime.now().uptimeNanoseconds
            let cb = ctx.queue.makeCommandBuffer()!
            let e = try encodeVerify(
                cb.makeComputeCommandEncoder()!, rows: 1, drafter: nil, state: sbuf, posBound: posBound, cb: cb)
            var rg = UInt32(ring)
            if let sampling, let rnd {
                if pen { try encodePenalties(e, rows: 1, params: sampling) }
                try encodeTopK(e, rows: 1, params: sampling)
                var kp = SampleKernelParams(sampling, vocab: vocab)
                e.setComputePipelineState(try ctx.pipeline("plain_accept_sample"))
                for (i, b) in [sbuf, tok, sampleBufs.idx, sampleBufs.val, sampleBufs.z].enumerated() {
                    e.setBuffer(b, offset: 0, index: i)
                }
                e.setBuffer(rec, offset: 0, index: 6)
                e.setBuffer(rnd, offset: 0, index: 7)
                e.setBytes(&kp, length: MemoryLayout<SampleKernelParams>.size, index: 8)
                e.setBytes(&rg, length: 4, index: 9)
            } else {
                e.setComputePipelineState(try ctx.pipeline("plain_accept"))
                for (i, b) in [sbuf, tok, top1].enumerated() { e.setBuffer(b, offset: 0, index: i) }
                e.setBuffer(rec, offset: 0, index: 4)
                e.setBytes(&rg, length: 4, index: 5)
            }
            e.dispatchThreads(
                MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
            if pen { try encodeCountEmitted(e, state: sbuf, rec: rec, ring: ring) }
            e.endEncoding()
            st.encodeMs.append(Double(DispatchTime.now().uptimeNanoseconds - te) / 1e6)
            cb.commit()
            return cb
        }
        var inflight: [MTLCommandBuffer] = [try encodeRound(posBound: P0)]
        var enqueued = 1, r = 0
        var lastGpuEnd: CFTimeInterval = 0
        while true {
            // Round `enqueued` reads the token at position P0 + enqueued.
            // ingot serve batching (decodeYield): no further round, so the last emitted token stays outside the cache.
            let yieldNow = decodeYield?() ?? false
            let extra = !yieldNow && P0 + enqueued + 1 <= lim && st.tokens.count + inflight.count < maxTokens
            if extra { inflight.append(try encodeRound(posBound: P0 + enqueued)); enqueued += 1 }
            let cb = inflight.removeFirst()
            cb.waitUntilCompleted()
            if let err = cb.error { throw err }
            let gms = (cb.gpuEndTime - cb.gpuStartTime) * 1e3
            st.gpuMs.append(gms)
            // Wall and gaps as in decodeSpec (ingot serve writes its `round:` line from them).
            if lastGpuEnd > 0 { st.gapMs.append((cb.gpuStartTime - lastGpuEnd) * 1e3) }
            st.wallMs.append(lastGpuEnd > 0 ? (cb.gpuEndTime - lastGpuEnd) * 1e3 : gms)
            lastGpuEnd = cb.gpuEndTime
            let rp = (rec.contents() + (r % ring) * RS * 4).bindMemory(to: UInt32.self, capacity: RS)
            precondition(Int(rp[Self.recP]) == P0 + r)
            let t = rp[1]
            r += 1
            st.rounds += 1
            st.acceptedHist[0] += 1
            st.tokens.append(t)
            var done = false
            if stopTokens.contains(t) {
                done = true; finish = .stop
            } else if st.tokens.count >= maxTokens {
                done = true; finish = .length
            }
            if let onTokens, !onTokens([t]), !done { done = true; finish = .cancelled }
            if yieldNow && !done && inflight.isEmpty { done = true; finish = .cancelled; decodeYielded = true }
            if !done && inflight.isEmpty { done = true; finish = .contextFull }
            if done { break }
        }
        for cb in inflight { cb.waitUntilCompleted(); if let err = cb.error { throw err } }
        st.decodeSeconds = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9
        // The executed rounds read tokens[0..<enqueued] (the extra ones, correct tokens already emitted); at most
        // up to the last emitted.
        P = P0 + enqueued
        nacc = 1
        precondition(enqueued <= st.tokens.count)
        try foldPending(drafter: nil, context: nil)
        st.finish = finish
        return st
    }

    /// Switches the ping-pong of all GDN layers (back to the block of the previous round).
    func flipGDN() {
        for l in layers {
            if case .gdn(let g) = l.mixer { g.advance() }
            if case .conv(let c) = l.mixer { c.advance() }
        }
        fn?.ple.advance()
        for q in fn?.qsaList ?? [] { q.advance() }
    }

    /// Folds the `nacc` rows of the GDN tape and writes the pending drafter context (rows of dr.caps):
    /// leaves the state as after a prefill of the P tokens.
    public func foldPending(drafter dr: DFlash2Drafter?, context: (rows: Int, pos0: Int)?) throws {
        guard nacc > 0 || context != nil else { return }
        let k = nacc
        try ctx.run { e in
            if k > 0 {
                for l in layers {
                    if case .gdn(let g) = l.mixer { try g.encodeFold(e, nacc: k) }
                    if case .conv(let c) = l.mixer { try c.encodeFold(e, nacc: k) }
                }
                if let p = fn?.ple { try p.encodeFold(ctx, e, nacc: k) }
                for q in fn?.qsaList ?? [] { try q.encodeFold(ctx, e, nacc: k) }
            }
            if let dr, let c = context {
                try dr.encodeContext(e, rows: max(dr.block, c.rows), nvalid: c.rows, pos0: c.pos0)
            }
        }
        nacc = 0
    }
}
