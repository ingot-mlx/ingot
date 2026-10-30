import Foundation
import Metal

// ingot serve batching (the simple path). Requests go through one engine loop (`InferenceEngine.serve`):
// - one conversation alone takes today's path (`generate`: prefix cache, snapshots, context copy, chained rounds);
// - when another request waits and a free slot fits it, the one running stops before its next chained round
//   (`decodeYield`) and goes on in joint rounds (`encodeVerifyBatch`, one row per conversation, not chained, without
//   copy), and the new one has its prompt read alone in its slot (the others wait meanwhile) and joins them;
// - a conversation that joined the joint rounds stays there until it ends, even if it is left alone.
// Each slot keeps its own live state and snapshots (its next turn reuses them); the prefix cache entries are shared.
// Only models without a drafter whose layers the joint round covers (`joint`) join rounds. The models with a
// drafter or MoE layers whose state the slots cover (`batchUnsupported`) keep the slots but are served one at a time:
// each request binds its slot (with its own drafter context ring) and takes today's path without yielding, so the next
// turn of each conversation resumes from its live state instead of reading the previous answer again. The others
// have a single slot, as before.

/// A request for the engine loop of ingot serve. The callbacks run on the engine thread.
public final class EngineRequest: @unchecked Sendable {
    public let prompt: [UInt32]
    public let maxTokens: Int
    public let sampling: SamplingParams
    public let stopTokens: Set<UInt32>
    /// When the request leaves the queue; false = the client left (it is dropped).
    let onStart: () -> Bool
    /// Each round's tokens; false stops the generation.
    let onTokens: ([UInt32]) -> Bool
    let onDone: (Swift.Result<InferenceEngine.Result, Error>) -> Void

    public init(
        prompt: [UInt32], maxTokens: Int, sampling: SamplingParams, stopTokens: Set<UInt32>,
        onStart: @escaping () -> Bool, onTokens: @escaping ([UInt32]) -> Bool,
        onDone: @escaping (Swift.Result<InferenceEngine.Result, Error>) -> Void
    ) {
        self.prompt = prompt
        self.maxTokens = maxTokens
        self.sampling = sampling
        self.stopTokens = stopTokens
        self.onStart = onStart
        self.onTokens = onTokens
        self.onDone = onDone
    }
}

/// Requests waiting for the engine loop, in arrival order (thread-safe).
public final class EngineQueue: @unchecked Sendable {
    private let lock = NSLock()
    private var items: [EngineRequest] = []
    public init() {}
    public func push(_ r: EngineRequest) { lock.withLock { items.append(r) } }
    public var isEmpty: Bool { lock.withLock { items.isEmpty } }
    func peek() -> EngineRequest? { lock.withLock { items.first } }
    func pop() { lock.withLock { _ = items.removeFirst() } }
}

/// Cache state of a slot not bound to the model.
struct ParkedCache {
    var live: [UInt32] = []
    var snaps: [StateSnapshot] = []
    var lastUse: UInt64 = 0
    /// Its drafter context ring (kc, vt of each drafter layer), nil until the slot is first left.
    var ring: [MTLBuffer]?
}

/// A conversation in the joint rounds.
final class JointConversation {
    let req: EngineRequest
    let slot: Int
    var res: InferenceEngine.Result
    /// Last emitted token, outside the cache (the input of the next round).
    var cur: UInt32
    var sampler: RowSampler?
    let joined = DispatchTime.now().uptimeNanoseconds
    /// Decode seconds before joining (alone) and joint rounds.
    let aloneSeconds: Double
    var joint = 0
    init(req: EngineRequest, slot: Int, res: InferenceEngine.Result, sampler: RowSampler?) {
        self.req = req
        self.slot = slot
        self.res = res
        self.cur = res.stats.tokens.last!
        self.sampler = sampler
        aloneSeconds = res.stats.decodeSeconds
    }
}

extension InferenceEngine {
    /// Why this engine keeps a single conversation in `serve` (nil = it can keep several in slots): what the slots
    /// do not cover (`HybridModel.slotsUnsupported`). Whether they also join rounds is `joint`.
    public var batchUnsupported: String? { model.slotsUnsupported }

    /// The slotted conversations also share rounds: models without a drafter whose layers the joint round
    /// covers. False: each request takes today's path alone in its slot, one after another.
    var joint: Bool { drafter == nil && model.batchUnsupported == nil }

    /// Keeps up to `slots` conversations in `serve` (1 = a single one), in joint rounds if `joint`, else one at a
    /// time; the extra slots hold `positions` positions each and are allocated at the first concurrent request (one
    /// at a time: at the first request bound to another slot), each keeping up to `snapshots` snapshots (nil: as
    /// many as slot 0).
    public func enableBatching(slots: Int, positions: Int, snapshots: Int? = nil) {
        precondition(slots == 1 || batchUnsupported == nil)
        precondition(slots >= 1 && slots <= HybridModel.decodeRows)
        batchSlots = slots
        batchPositions = min(positions, model.maxCtx)
        batchSnapshots = min(snapshots ?? maxSnapshots, maxSnapshots)
        parkedCache = Array(repeating: ParkedCache(), count: slots)
        flushPendingKV { _ in true }
        // A model whose slots another engine left bound elsewhere starts again from its own buffers (ingot-dev
        // batchcheck --serve: one engine per pass on the same model).
        model.bindSlot(0)
    }

    /// Serves the requests of `q` until it is empty and no conversation is left (call it on the engine thread; a
    /// request pushed while it runs is served by it). One request at a time takes today's path; see the top of
    /// BatchServe.swift for how they are joined.
    public func serve(_ q: EngineQueue) {
        var active: [JointConversation] = []
        while true {
            if active.isEmpty {
                guard let r = q.peek() else { return }
                q.pop()
                guard r.onStart() else { continue }
                let j = pickSlot(r, busy: []) ?? 0
                bind(j)
                do {
                    let yield: (() -> Bool)? =
                        batchSlots > 1 && joint ? { [unowned self] in self.admissible(q, busy: [j]) != nil } : nil
                    let res = try generate(
                        prompt: r.prompt, maxTokens: r.maxTokens, sampling: r.sampling, stopTokens: r.stopTokens,
                        onTokens: r.onTokens, yield: yield)
                    if res.yielded {
                        active.append(join(r, res, slot: j))
                        log?("batch: slot \(j) goes on in the joint rounds after \(res.stats.tokens.count) tokens")
                    } else {
                        r.onDone(.success(res))
                    }
                } catch {
                    r.onDone(.failure(error))
                }
                continue
            }
            // Admits the head of the queue while a free slot fits it (its prompt is read alone: the others wait).
            while active.count < batchSlots, let (r, j) = admissible(q, busy: Set(active.map(\.slot))) {
                q.pop()
                guard r.onStart() else { continue }
                do {
                    if let c = try startJoint(r, slot: j) { active.append(c) }
                } catch {
                    r.onDone(.failure(error))
                }
            }
            if active.isEmpty { continue }
            do {
                try jointRound(&active)
            } catch {
                for c in active { c.req.onDone(.failure(error)) }
                active = []
                resetSlots()
            }
        }
    }

    /// Positions slot `j` holds.
    func slotPositions(_ j: Int) -> Int { j == 0 ? model.maxCtx : batchPositions }

    /// The head of the queue and the free slot it would take, if any.
    func admissible(_ q: EngineQueue, busy: Set<Int>) -> (EngineRequest, Int)? {
        guard busy.count < batchSlots, let r = q.peek(), let j = pickSlot(r, busy: busy) else { return nil }
        return (r, j)
    }

    /// The free slot for `r`: among those it fits in (prompt, answer and one round, or two drafter blocks as `start`
    /// asks), the one whose live state or snapshot covers more of its prompt (its next turn), else the least recently
    /// used.
    func pickSlot(_ r: EngineRequest, busy: Set<Int>) -> Int? {
        let n = r.prompt.count
        let margin =
            drafter == nil ? AttentionLayer.maxRows : max(AttentionLayer.maxRows, DFlash2Drafter.margin(drafter))
        func reuse(_ j: Int) -> Int {
            let (l, sn) = j == boundSlot ? (live, snaps) : (parkedCache[j].live, parkedCache[j].snaps)
            var best = !l.isEmpty && l.count < n && Self.isPrefix(l, of: r.prompt) ? l.count : 0
            for s in sn where s.lastRow == nil && s.tokens.count < n && Self.isPrefix(s.tokens, of: r.prompt) {
                best = max(best, s.tokens.count)
            }
            return best
        }
        let free = (0..<batchSlots).filter {
            !busy.contains($0) && n + r.maxTokens + margin <= slotPositions($0)
        }
        return free.max { a, b in
            let ra = reuse(a), rb = reuse(b)
            return ra != rb ? ra < rb : parkedCache[a].lastUse > parkedCache[b].lastUse
        }
    }

    /// Binds slot `j` (its KV, GDN state, counters and sampling buffers in the model; its live state, snapshots and
    /// drafter context ring in the engine). The extra slots are allocated the first time.
    func bind(_ j: Int) {
        useClock += 1
        parkedCache[j].lastUse = useClock
        guard j != boundSlot else { return }
        if !slotsAllocated { allocateSlots() }
        // Entries captured in the pass keep their KV in the live cache of the bound slot, which binding parks: copy it
        // into them first (only the bound slot ever has pending entries).
        flushPendingKV { _ in true }
        parkedCache[boundSlot].live = live
        parkedCache[boundSlot].snaps = snaps
        live = parkedCache[j].live
        snaps = parkedCache[j].snaps
        parkedCache[j].live = []
        parkedCache[j].snaps = []
        if drafter != nil { exchangeRing(from: boundSlot, to: j) }
        model.bindSlot(j)
        boundSlot = j
        snapshotLimit = j == 0 ? maxSnapshots : batchSnapshots
    }

    /// The drafter's context ring is one per drafter (its layers' kc/vt, which the captures, snapshots and prefix
    /// cache entries copy): binding another slot copies it out to the slot left (`out`) and copies in the one of slot
    /// `j` (zeros if `j` was never left, as a fresh drafter). Two blits of ~2 · 6 · 2080 rows of K/V per bind, once per
    /// request and not per round. Goes on the queue, after what is queued.
    private func exchangeRing(from out: Int, to j: Int) {
        guard let dr = drafter else { return }
        let cur = dr.layers.flatMap { [$0.kc, $0.vt] }
        let saved = freeRing ?? cur.map { model.ctx.buffer(length: $0.length) }
        let cb = model.ctx.queue.makeCommandBuffer()!
        // Two encoders: the second overwrites what the first reads.
        let b0 = cb.makeBlitCommandEncoder()!
        for (x, y) in zip(cur, saved) {
            b0.copy(from: x, sourceOffset: 0, to: y, destinationOffset: 0, size: x.length)
        }
        b0.endEncoding()
        let b1 = cb.makeBlitCommandEncoder()!
        if let inc = parkedCache[j].ring {
            for (x, y) in zip(inc, cur) {
                b1.copy(from: x, sourceOffset: 0, to: y, destinationOffset: 0, size: x.length)
            }
        } else {
            for x in cur { b1.fill(buffer: x, range: 0..<x.length, value: 0) }
        }
        b1.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        freeRing = parkedCache[j].ring
        parkedCache[j].ring = nil
        parkedCache[out].ring = saved
    }

    /// The model has the extra slots this engine asks for (another engine may have left fewer or shorter ones).
    private var slotsAllocated: Bool {
        guard let s = model.convSlots else { return false }
        return s.parked.count >= batchSlots && s.positions >= batchPositions
    }

    private func allocateSlots() {
        precondition(boundSlot == 0)
        let t = DispatchTime.now().uptimeNanoseconds
        _ = model.ensureSlots(batchSlots, positions: batchPositions)
        log?(
            String(
                format: "batch: %d extra slots of %d positions allocated in %.2f s (Metal memory %.2f GB)",
                batchSlots - 1, batchPositions, Double(DispatchTime.now().uptimeNanoseconds - t) / 1e9,
                Double(model.ctx.device.currentAllocatedSize) / 1e9))
    }

    /// A conversation that stopped alone (`yielded`) goes on in the joint rounds from its last emitted token. Its
    /// penalty counters are already in its slot; with sampling, the rest of the answer takes new uniforms.
    func join(_ r: EngineRequest, _ res: InferenceEngine.Result, slot j: Int) -> JointConversation {
        if !slotsAllocated { allocateSlots() }
        var sampler: RowSampler?
        if let cs = r.sampling.chainSampling {
            let seed = r.sampling.seed ?? UInt64.random(in: 0...UInt64.max)
            var rng = SplitMix64(seed: seed &+ UInt64(res.stats.tokens.count))
            sampler = RowSampler(
                params: cs, penalties: r.sampling.hasPenalties,
                rnd: model.roundUniforms(maxTokens: r.maxTokens, rng: &rng), ctx: model.ctx)
        }
        return JointConversation(req: r, slot: j, res: res, sampler: sampler)
    }

    /// Reads the prompt of `r` in slot `j` (as `generate`: live state, snapshots, prefix cache) and samples its first
    /// token; nil if that already ends it. The rest as `decodePlain` would: the same uniforms and counters.
    func startJoint(_ r: EngineRequest, slot j: Int) throws -> JointConversation? {
        bind(j)
        var (res, first, rng) = try start(prompt: r.prompt, maxTokens: r.maxTokens, sampling: r.sampling)
        res.stats.tokens = [first]
        log?(
            "batch: slot \(j) reads the prompt (cache: \(res.cacheNote), \(res.cachedTokens) of \(r.prompt.count) "
                + String(format: "tokens reused, %.2f s)", res.prefillSeconds))
        let keep = r.onTokens([first])
        if r.stopTokens.contains(first) || !keep || r.maxTokens <= 1 {
            res.stats.finish = r.stopTokens.contains(first) ? .stop : !keep ? .cancelled : .length
            try finishSlot(j, r, &res)
            r.onDone(.success(res))
            return nil
        }
        var sampler: RowSampler?
        if let cs = r.sampling.chainSampling {
            model.startPenalties(first: first, params: r.sampling)
            sampler = RowSampler(
                params: cs, penalties: r.sampling.hasPenalties,
                rnd: model.roundUniforms(maxTokens: r.maxTokens, rng: &rng), ctx: model.ctx)
        }
        return JointConversation(req: r, slot: j, res: res, sampler: sampler)
    }

    /// One joint round of the active conversations (row r = `active[r]`); the finished ones leave.
    func jointRound(_ active: inout [JointConversation]) throws {
        let m = model
        guard let s = m.convSlots else { preconditionFailure("joint round without slots") }
        let convs = active.map(\.slot)
        let tp = m.tok.contents().bindMemory(to: UInt32.self, capacity: HybridModel.decodeRows)
        let t1p = m.top1.contents().bindMemory(to: UInt32.self, capacity: HybridModel.decodeRows)
        for (r, c) in active.enumerated() { tp[r] = c.cur }
        var samplers = [RowSampler?](repeating: nil, count: batchSlots)
        for c in active { samplers[c.slot] = c.sampler }
        guard let cb = m.ctx.queue.makeCommandBuffer(), let e = cb.makeComputeCommandEncoder() else {
            throw SafetensorsError.io("cb")
        }
        try m.encodeVerifyBatch(e, convs: convs, slots: s)
        try m.encodeSampleRows(e, convs: convs, samplers: samplers, slots: s)
        e.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if let err = cb.error { throw err }
        var left: [JointConversation] = []
        for (r, c) in active.enumerated() {
            let p: Int
            if c.slot == s.bound {
                m.commit(accepted: 1)
                p = m.P
            } else {
                s.parked[c.slot]!.P += 1
                s.parked[c.slot]!.nacc = 1
                p = s.parked[c.slot]!.P
            }
            let t = c.sampler?.token ?? t1p[r]
            c.cur = t
            c.res.stats.tokens.append(t)
            c.res.stats.rounds += 1
            c.joint += 1
            var finish: HybridModel.FinishReason?
            if c.req.stopTokens.contains(t) {
                finish = .stop
            } else if c.res.stats.tokens.count >= c.req.maxTokens {
                finish = .length
            }
            if !c.req.onTokens([t]) && finish == nil { finish = .cancelled }
            if finish == nil && p + 2 > slotPositions(c.slot) { finish = .contextFull }
            if let finish {
                c.res.stats.finish = finish
                c.res.stats.decodeSeconds =
                    c.aloneSeconds + Double(DispatchTime.now().uptimeNanoseconds - c.joined) / 1e9
                try finishSlot(c.slot, c.req, &c.res)
                log?("batch: slot \(c.slot) done after \(c.joint) joint rounds (\(active.count) conversations now)")
                c.req.onDone(.success(c.res))
            } else {
                left.append(c)
            }
        }
        active = left
    }

    /// At the end of a conversation of the joint rounds: its rows still on the tape are folded and its live state is
    /// the prompt plus the emitted tokens that entered the cache (as `generate` leaves it).
    private func finishSlot(_ j: Int, _ r: EngineRequest, _ res: inout InferenceEngine.Result) throws {
        bind(j)
        try model.foldPending(drafter: nil, context: nil)
        live = r.prompt + res.stats.tokens.prefix(model.P - r.prompt.count)
        precondition(live.count == model.P)
        res.stats.prefillSeconds = res.prefillSeconds
    }

    /// After an error in a joint round: every slot starts from scratch.
    private func resetSlots() {
        for j in 0..<batchSlots {
            bind(j)
            live = []; snaps = []; model.nacc = 0; model.reset()
        }
        bind(0)
        copyIndex.reset()
    }
}
