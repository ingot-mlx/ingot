import Foundation
import Metal

// Batched decoding (the simple path): N fixed conversations, each in its own slot (its own KV, GDN state,
// counters and sampling buffers), each prompt prefilled alone in its slot as today, then rounds of one row per
// conversation (`generateBatch`), greedy or sampled with each conversation's own parameters, seed and penalties.
// What only reads weights goes once over the N rows; the attention over each KV and the GDN recurrence over each
// state go once per conversation (`encodeVerifyBatch`, ModelRound.swift), and so do the penalties and the sampling
// pick (`encodeSampleRows`). Each conversation gives bit for bit the logits and tokens it gives alone (`lab-dev
// batchcheck`). One conversation never comes here: N = 1 is today's path, untouched.

/// The conversations of a batch that are not bound to the model: their position, their KV per attention
/// layer and their recurrent state per GDN layer. Slot 0 is the model's own buffers; the bound slot is the one the
/// layers hold now (its entry is nil).
final class ConversationSlots {
    struct Parked {
        var P: Int, nacc: Int
        /// One per attention layer, in layer order.
        var kv: [AttentionLayer.KVSlot]
        /// One per GDN layer, in layer order.
        var gdn: [GDNLayer.GDNSlot]
        /// Its sampling candidates and penalty counters.
        var sample: HybridModel.SampleBuffers
    }
    /// Positions each extra slot holds (slot 0 holds the model's `maxCtx`).
    let positions: Int
    var parked: [Parked?]
    var bound = 0

    init(positions: Int, parked: [Parked?]) {
        self.positions = positions
        self.parked = parked
    }
}

extension HybridModel {
    /// Why this model cannot keep conversations in slots (nil = it can; lab serve then gives each conversation its
    /// own live state, served one at a time or in joint rounds): the slots swap the int8 KV of the attention layers
    /// and the state of the GDN layers, without Flash-Next, SSD experts or a KV that grows.
    var slotsUnsupported: String? {
        if fn != nil || expertStream != nil || kvPool != nil { return "Flash-Next, SSD experts or KVPool" }
        // Gemma 4 E2B/E4B: not yet (a clear error until batching reaches Gemma).
        if ple != nil { return "per-layer inputs (PLE)" }
        for l in layers {
            // Nemotron-H's single-block layers and Mamba2 wait for batching to test them.
            if !l.hasFFN { return "layers without FFN (Nemotron-H)" }
            switch l.mixer {
            case .attn(let a):
                // MLA (GLM-4.7-Flash) keeps neither slots nor joint rounds until a batchcheck with toy-glm
                // covers it.
                if a.mla != nil { return "MLA attention (GLM-4.7-Flash)" }
                if a.kvSource != nil { return "KV shared between layers" }
                if a.treeBuf != nil || a.qsaMask != nil || a.blockList != nil {
                    return "draft tree or sparse attention"
                }
            case .gdn(let g):
                if g.kind == .mamba2 { return "Mamba2 layers" }
                if g.treeBuf != nil { return "draft tree" }
            case .conv: return "short convolution layers"
            case .none: return "layers without mixer (Nemotron-H)"
            }
        }
        return nil
    }

    /// Why this model cannot decode several conversations in one round yet (nil = it can): the simple path covers
    /// the models that can keep slots (`slotsUnsupported`) whose layers are all dense.
    public var batchUnsupported: String? {
        if let why = slotsUnsupported { return why }
        if layers.contains(where: { $0.dense == nil || $0.moeOpt != nil }) { return "MoE layers" }
        return nil
    }

    private var attentionLayers: [AttentionLayer] {
        layers.compactMap { if case .attn(let a) = $0.mixer { a } else { nil } }
    }

    private var gdnLayers: [GDNLayer] {
        layers.compactMap { if case .gdn(let g) = $0.mixer { g } else { nil } }
    }

    /// Slots 1..<n of `positions` positions each (slot 0 is the model's own buffers). Reuses the current ones if
    /// they are enough; otherwise binds slot 0 and allocates new ones.
    func ensureSlots(_ n: Int, positions: Int) -> ConversationSlots {
        if let s = convSlots, s.parked.count >= n, s.positions >= positions { return s }
        bindSlot(0)
        convSlots = nil  // the old slots go before the new ones are allocated
        let attn = attentionLayers, gdn = gdnLayers
        let s = ConversationSlots(
            positions: positions,
            parked: [nil]
                + (1..<max(n, 1)).map { _ in
                    ConversationSlots.Parked(
                        P: 0, nacc: 0, kv: attn.map { $0.makeSlot(positions: positions) },
                        gdn: gdn.map { $0.makeSlot() }, sample: SampleBuffers(ctx))
                })
        convSlots = s
        return s
    }

    /// Points the layers at slot `j`'s KV and GDN state and the model at its position and sampling buffers (only
    /// references are swapped: O(layers)). Prefill, snapshots (`encodeFold`), the normal round and the sampling and
    /// penalty helpers then work on that conversation as they do today.
    func bindSlot(_ j: Int) {
        guard let s = convSlots, j != s.bound, var inc = s.parked[j] else { return }
        s.parked[j] = nil
        swap(&P, &inc.P)
        swap(&nacc, &inc.nacc)
        swap(&sampleBufs, &inc.sample)
        for (k, a) in attentionLayers.enumerated() { a.exchangeSlot(&inc.kv[k]) }
        for (k, g) in gdnLayers.enumerated() { g.exchangeSlot(&inc.gdn[k]) }
        s.parked[s.bound] = inc
        s.bound = j
    }

    /// Decoding of N conversations (2…16) in shared rounds: each prompt prefilled alone in its slot, then one
    /// command buffer and one wait per round, with one row per conversation still running (a conversation leaves the
    /// batch when it reaches `maxTokens` or, with `stopAtEOS`, an end token). Greedy by default; `sampling[j]` gives
    /// conversation j its own temperature, top-k, top-p, min-p, seed and penalties. Each conversation gives the same
    /// tokens as alone: greedy, the tokens and logits of `generate`, bit for bit; with `sampling[j]`, those of the
    /// `lab serve` path (`preparePenalties` + `sampleFromLogits` for the first token, then `decodePlain` with the same
    /// generator), because each row is sampled with the same kernels, its own counters and its own uniforms indexed by
    /// its own round. `onRound(convs)`: after each round, with row r of `logits` and `top1` being conversation
    /// `convs[r]` (with penalties, row r is already penalized). At the end the model is back on its own buffers
    /// (slot 0) with the state of conversation 0; the extra slots stay allocated for the next call.
    public func generateBatch(
        prompts: [[UInt32]], maxTokens: Int, stopAtEOS: Bool = true, genStarts: [Int?]? = nil,
        sampling: [SamplingParams]? = nil, onRound: (([Int]) -> Void)? = nil
    ) throws -> [GenStats] {
        let n = prompts.count
        guard n >= 2 && n <= Self.decodeRows else {
            throw SafetensorsError.format("generateBatch: 2…\(Self.decodeRows) conversations (got \(n))")
        }
        guard sampling == nil || sampling!.count == n else {
            throw SafetensorsError.format("generateBatch: one SamplingParams per conversation")
        }
        if let why = batchUnsupported { throw SafetensorsError.format("generateBatch: not with \(why)") }
        let need = (prompts.map(\.count).max() ?? 0) + maxTokens + AttentionLayer.maxRows
        guard prompts[0].count + maxTokens + AttentionLayer.maxRows <= maxCtx else {
            throw SafetensorsError.format("generateBatch: conversation 0 does not fit in the context (\(maxCtx))")
        }
        let s = ensureSlots(n, positions: need)
        defer { bindSlot(0) }
        var st = [GenStats](repeating: GenStats(), count: n)
        var cur = [UInt32](repeating: 0, count: n)
        var samplers = [RowSampler?](repeating: nil, count: n)
        for j in 0..<n {
            bindSlot(j)
            let (argmax, row, ps) = try preparePrompt(prompts[j], from: 0, genStart: genStarts?[j], drafter: nil)
            var first = argmax
            if let sp = sampling?[j] {
                // As Engine.generate + decodePlain: the first token, then the uniforms of the rounds, from one generator.
                var rng = SplitMix64(seed: sp.seed ?? UInt64.random(in: 0...UInt64.max))
                preparePenalties(prompt: prompts[j], params: sp)
                first = try sampleFromLogits(argmax: argmax, row: row, params: sp, rng: &rng)
                if let cs = sp.chainSampling {
                    startPenalties(first: first, params: sp)
                    samplers[j] = RowSampler(
                        params: cs, penalties: sp.hasPenalties, rnd: roundUniforms(maxTokens: maxTokens, rng: &rng),
                        ctx: ctx)
                }
            }
            st[j].prefillSeconds = ps
            st[j].tokens = [first]
            cur[j] = first
        }
        let t0 = DispatchTime.now().uptimeNanoseconds
        let tp = tok.contents().bindMemory(to: UInt32.self, capacity: Self.decodeRows)
        let t1p = top1.contents().bindMemory(to: UInt32.self, capacity: Self.decodeRows)
        while true {
            let convs = (0..<n).filter { st[$0].tokens.count < maxTokens && !(stopAtEOS && eosSet.contains(cur[$0])) }
            if convs.isEmpty { break }
            for (r, j) in convs.enumerated() { tp[r] = cur[j] }
            let te = DispatchTime.now().uptimeNanoseconds
            guard let cb = ctx.queue.makeCommandBuffer(), let e = cb.makeComputeCommandEncoder() else {
                throw SafetensorsError.io("cb")
            }
            try encodeVerifyBatch(e, convs: convs, slots: s)
            try encodeSampleRows(e, convs: convs, samplers: samplers, slots: s)
            e.endEncoding()
            let encMs = Double(DispatchTime.now().uptimeNanoseconds - te) / 1e6
            cb.commit()
            cb.waitUntilCompleted()
            if let err = cb.error { throw err }
            let gpu = (cb.gpuEndTime - cb.gpuStartTime) * 1e3
            onRound?(convs)
            for (r, j) in convs.enumerated() {
                if j == s.bound {
                    commit(accepted: 1)
                } else {
                    s.parked[j]!.P += 1
                    s.parked[j]!.nacc = 1
                }
                cur[j] = samplers[j]?.token ?? t1p[r]
                st[j].tokens.append(cur[j])
                st[j].rounds += 1
                st[j].gpuMs.append(gpu)
                st[j].encodeMs.append(encMs)
            }
        }
        let secs = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9
        for j in 0..<n { st[j].decodeSeconds = secs }
        return st
    }
}

/// The sampling state of one conversation in `generateBatch`: what `decodePlain` keeps for its single row (round
/// state, record and uniforms), so the row is sampled by the same kernels with the same uniforms as alone.
struct RowSampler {
    /// `chainSampling` of the conversation's parameters.
    let params: SamplingParams
    let penalties: Bool
    /// Uniforms of each round (`roundUniforms`), indexed by the round counter st[5].
    let rnd: MTLBuffer
    /// plain_accept_sample state: [P, nacc, -, -, last token, round, scratch, -] (P is only recorded).
    let st: MTLBuffer
    /// Its one-entry record (ring 1), which count_emitted reads.
    let rec: MTLBuffer
    init(params: SamplingParams, penalties: Bool, rnd: MTLBuffer, ctx: MetalContext) {
        self.params = params
        self.penalties = penalties
        self.rnd = rnd
        st = ctx.buffer(length: 8 * 4)
        memset(st.contents(), 0, 8 * 4)
        rec = ctx.buffer(length: HybridModel.recStride * 4)
    }
    /// The token the last round sampled.
    var token: UInt32 { st.contents().bindMemory(to: UInt32.self, capacity: 8)[4] }
}

extension HybridModel {
    /// Samples the rows of the conversations with a sampler after the joint round (row r = conversation `convs[r]`),
    /// as one `decodePlain` round does for each: its penalties with its counters, its 64 candidates and
    /// plain_accept_sample with its uniforms. The candidates of all the rows go in one pass (topk64_*_each,
    /// row r with its own parameters; the same code as alone), into the bound
    /// `sampleBufs`; the counters of a parked conversation are bound to the model only while its kernels are encoded
    /// (the helpers use `sampleBufs`).
    func encodeSampleRows(
        _ e: MTLComputeCommandEncoder, convs: [Int], samplers: [RowSampler?], slots s: ConversationSlots
    ) throws {
        let rows = convs.enumerated().compactMap { r, j in samplers[j].map { (r: r, j: j, rs: $0) } }
        guard let last = rows.last else { return }
        func withCounters(_ j: Int, _ body: () throws -> Void) rethrows {
            let parked = j != s.bound
            if parked { swap(&sampleBufs, &s.parked[j]!.sample) }
            defer { if parked { swap(&sampleBufs, &s.parked[j]!.sample) } }
            try body()
        }
        for x in rows where x.rs.penalties {
            try withCounters(x.j) { try encodePenalties(e, rows: 1, params: x.rs.params, firstRow: x.r) }
        }
        // Rows 0...last.r; a row without a sampler (greedy) gets candidates nobody reads.
        let kp = (0...last.r).map { r in
            SampleKernelParams(samplers[convs[r]]?.params ?? SamplingParams(temperature: 1), vocab: vocab)
        }
        try TopK64.encode(
            e, ctx: ctx, x: logits, xOffset: 0, rows: kp.count, params: kp,
            out: (sampleBufs.idx, sampleBufs.val, sampleBufs.z),
            scratch: sampleBufs.part)
        let K = SamplingParams.candidates
        var ring = UInt32(1)
        for x in rows {
            var kp = SampleKernelParams(x.rs.params, vocab: vocab)
            e.setComputePipelineState(try ctx.pipeline("plain_accept_sample"))
            e.setBuffer(x.rs.st, offset: 0, index: 0)
            e.setBuffer(x.rs.st, offset: 6 * 4, index: 1)  // the token also goes to st[6], not to `tok`
            e.setBuffer(sampleBufs.idx, offset: x.r * K * 4, index: 2)
            e.setBuffer(sampleBufs.val, offset: x.r * K * 4, index: 3)
            e.setBuffer(sampleBufs.z, offset: x.r * 2 * 4, index: 4)
            e.setBuffer(x.rs.rec, offset: 0, index: 6)
            e.setBuffer(x.rs.rnd, offset: 0, index: 7)
            e.setBytes(&kp, length: MemoryLayout<SampleKernelParams>.size, index: 8)
            e.setBytes(&ring, length: 4, index: 9)
            e.dispatchThreads(
                MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
            if x.rs.penalties {
                try withCounters(x.j) { try encodeCountEmitted(e, state: x.rs.st, rec: x.rs.rec, ring: 1) }
            }
        }
    }
}
