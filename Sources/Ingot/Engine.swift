import Foundation
import Metal

/// Recoverable engine errors (the server returns them as HTTP 400).
public enum EngineError: Error, CustomStringConvertible {
    case contextOverflow(prompt: Int, maxCtx: Int)
    case emptyPrompt
    public var description: String {
        switch self {
        case .contextOverflow(let p, let m):
            return
                "prompt has \(p) tokens and the server's max context is \(m) (16 or more must stay free to generate)"
        case .emptyPrompt: return "empty prompt"
        }
    }
}

/// Snapshot of the recurrent state at a point of the prompt (idea from oMLX; points in Prompt.swift): state and conv
/// window of each GDN layer and the drafter's context ring. The attention KV cache is not copied: positions < P are
/// only rewritten if a prompt diverging earlier arrives, and then the snapshot is dropped.
final class StateSnapshot {
    var tokens: [UInt32] = []
    /// Point (c), the whole prompt: the logits of its last row and their argmax, as the prefill left them. An exact
    /// repetition of the prompt starts decoding from here without reading anything, with the same logits and state as
    /// the first time (reading the last token alone from (b) goes through the verification path, which sums in
    /// another order: different logits and, with sampling, a different text).
    var lastRow: (argmax: UInt32, logits: [Float])?
    let gdn: [(state: MTLBuffer, hist: MTLBuffer)]
    let draft: [(kc: MTLBuffer, vt: MTLBuffer)]
    /// Window (L − 1 Bx rows) of each short-convolution layer (LFM2, LFM3).
    let conv: [MTLBuffer]
    init(model m: HybridModel, drafter dr: DFlash2Drafter?) {
        var g: [(MTLBuffer, MTLBuffer)] = []
        var c: [MTLBuffer] = []
        for l in m.layers {
            if case .gdn(let x) = l.mixer {
                g.append((m.ctx.buffer(length: x.state[0].length), m.ctx.buffer(length: 3 * x.convDim * 4)))
            }
            if case .conv(let x) = l.mixer { c.append(m.ctx.buffer(length: x.W * x.D * 4)) }
        }
        // Flash-Next: the PLE conv window goes as one more conv (the last).
        if let p = m.fn?.ple { c.append(m.ctx.buffer(length: p.windowBytes)) }
        // and the QSA raw-key windows, in layer order.
        for q in m.fn?.qsaList ?? [] { c.append(m.ctx.buffer(length: q.windowBytes)) }
        gdn = g
        conv = c
        draft = dr?.layers.map { (m.ctx.buffer(length: $0.kc.length), m.ctx.buffer(length: $0.vt.length)) } ?? []
    }
    var bytes: Int {
        gdn.reduce(0) { $0 + $1.state.length + $1.hist.length } + draft.reduce(0) { $0 + $1.kc.length + $1.vt.length }
            + conv.reduce(0) { $0 + $1.length }
    }
}

/// A model with its drafter, one client at a time, with a prefix cache:
/// 1. **Live state:** after each answer the state stays reusable (`decodeSpec` undoes the extra round), so if the
///    next prompt starts with everything in the cache (prompt + answer verbatim), only the new part is read.
/// 2. **Snapshots at two points of each prompt** (after `<|im_start|>assistant\n` and before the last token):
///    the template may rewrite the answer its own way (without the reasoning block, or with a `\n\n` that
///    merges two tokens), and the GDN cannot be trimmed midway, so the next turn starts from the longest point
///    that is still a prefix. Several are kept nested (each a prefix of the
///    next). A third one, (c), after the whole prompt and with the logits of its last row, serves only an exact
///    repetition of the prompt (same seed, same text).
public final class InferenceEngine {
    /// KVPool: answer tokens for which KV is mapped at the start of each request (the rest, during decoding; as in
    /// splash, min(max_tokens, 4096)).
    static let kvAhead = 4096
    public let model: HybridModel
    /// nil: decoding without a drafter (LFM2).
    public let drafter: DFlash2Drafter?
    public let maxSnapshots: Int
    /// Snapshots kept now (KVPool: drops to 2 when the KV exceeds what the plan accounts for).
    public internal(set) var snapshotLimit: Int
    /// Tokens in the cache (positions 0..<model.P).
    public internal(set) var live: [UInt32] = []
    /// Where the answer starts in `live` (the length of the prompt that left it): only for `liveMiss`.
    var liveAnswerStart = 0
    var snaps: [StateSnapshot] = []
    var spare: [StateSnapshot] = []
    public var log: (@Sendable (String) -> Void)?
    /// Tests: called after reading the prompt with the `model.logits` row holding the logits of its last
    /// position.
    public var onPromptLogits: ((Int) -> Void)?

    // Prefix cache across sessions (PrefixCache.swift).
    /// Byte limit of the entries (0 = off). Default 1 GB (ingot serve: the memory plan's or --prefix-cache-gb).
    public var prefixCacheBytes: Int
    /// What KVPool has currently taken from `prefixCacheBytes` (while the KV exceeds what the plan accounts for).
    var kvPrefixCut = 0
    var entries: [PrefixEntry] = []
    var useClock: UInt64 = 0
    /// Last prompts (for the boundary of the prefix shared with the next one).
    var recentPrompts: [[UInt32]] = []
    /// Ids of `<|im_start|>user` (end of the system block). Empty = no such point.
    public var userMarker: [UInt32] = []
    /// Alignment of the cut points (prefill chunks go in multiples of 64).
    public var cutAlign = 64
    /// An entry must save at least this much.
    public var minEntryTokens = 1024

    // ingot serve batching (BatchServe.swift).
    /// Conversations per round (1 = one at a time, today's path) and positions of each extra slot.
    public internal(set) var batchSlots = 1
    var batchPositions = 0
    /// Snapshots each extra slot keeps (slot 0: `maxSnapshots`).
    var batchSnapshots = 0
    /// Slot whose state the model and `live`/`snaps` hold now, and the cache state of the others.
    var boundSlot = 0
    var parkedCache: [ParkedCache] = []
    /// Drafter ring buffers no slot holds (the next bind copies the bound one out into them).
    var freeRing: [MTLBuffer]?

    public init(model: HybridModel, drafter: DFlash2Drafter?, maxSnapshots: Int = 4, prefixCacheBytes: Int? = nil) {
        self.model = model
        self.drafter = drafter
        self.maxSnapshots = maxSnapshots
        snapshotLimit = maxSnapshots
        self.prefixCacheBytes = prefixCacheBytes ?? 1_000_000_000
    }

    /// KVPool: the KV beyond what the plan accounts for is paid first by the prefix cache (entries are evicted), then
    /// by the snapshots (from `maxSnapshots` to 2); the rest by the budget slack. When the KV shrinks again, they
    /// are recovered. With the GPU stopped.
    func balanceKV() {
        guard let p = model.kvPool else { return }
        var over = p.overAllowanceBytes
        let base = prefixCacheBytes + kvPrefixCut
        let cut = min(base, over)
        over -= cut
        if cut != kvPrefixCut {
            prefixCacheBytes = base - cut
            kvPrefixCut = cut
            while entriesBytes > prefixCacheBytes, let lru = entries.min(by: { $0.lastUse < $1.lastUse }) {
                entries.removeAll { $0 === lru }
                log?("prefix cache: evicted entry of \(lru.tokens.count) tokens (KV is growing)")
            }
        }
        let limit = over > 0 ? min(maxSnapshots, 2) : maxSnapshots
        if limit < snapshotLimit {
            while snaps.count > limit { snaps.removeFirst() }
            // Spares take space too: keep the two used for in-pass captures.
            if spare.count > 2 { spare.removeFirst(spare.count - 2) }
        }
        snapshotLimit = limit
    }

    public struct Result {
        public var stats: HybridModel.GenStats
        public var promptTokens = 0
        /// Prompt tokens that did not need reading (prefix cache) and where they came from.
        public var cachedTokens = 0
        public var cacheSource = "none"
        public var prefillSeconds = 0.0
        /// ms spent updating the context-copy index before decoding.
        public var copyIndexMs = 0.0
        /// ingot serve batching: decoding stopped to go on in the joint rounds (not finished).
        var yielded = false
        /// Length of the live state that went back to the common prefix with the prompt (`cacheSource` "live").
        public internal(set) var rolledBackFrom: Int?
        /// Debugging (log and `ingot serve --dump-requests`): why the live state was not followed whole, nil if it was
        /// or there was none.
        public internal(set) var liveMiss: String?
        /// `cacheSource` for the logs, with the rollback if there was one.
        var cacheNote: String { cacheSource + (rolledBackFrom.map { " (rolled back from \($0))" } ?? "") }
    }

    /// Trigram index of the context copy (CopySpec.swift), kept across requests.
    public let copyIndex = CopyIndex()
    /// Copy policy of the last request (reset at the start of each one).
    public let copyState = CopyPolicyState()

    /// Bytes of each snapshot.
    public var snapshotBytes: Int {
        snaps.first?.bytes ?? spare.first?.bytes ?? StateSnapshot(model: model, drafter: drafter).bytes
    }

    /// The whole state is the KV (no GDN, conv, drafter, PLE or QSA windows: snapshots of 0 bytes; Gemma 4, GLM,
    /// gpt-oss): the live state can then go back to any position whose ring window is intact just by setting P.
    lazy var stateIsKVOnly = snapshotBytes == 0

    static func isPrefix(_ a: [UInt32], of b: [UInt32]) -> Bool {
        a.count <= b.count
            && a.withUnsafeBufferPointer { pa in
                b.withUnsafeBufferPointer { pb in memcmp(pa.baseAddress, pb.baseAddress, a.count * 4) == 0 }
            }
    }

    /// Ids of `<|im_start|>assistant\n` (or the start of the model's generation prompt): marks snapshot point (a)
    /// (see Prompt.swift). Empty = only point (b).
    public var generationMarker: [UInt32] = []

    /// Position after the last occurrence of `generationMarker` in `ids`.
    public static func generationStart(_ ids: [UInt32], marker: [UInt32]) -> Int? {
        guard !marker.isEmpty, ids.count >= marker.count else { return nil }
        var i = ids.count - marker.count
        while i >= 0 {
            if ids[i] == marker[0] && Array(ids[i..<(i + marker.count)]) == marker { return i + marker.count }
            i -= 1
        }
        return nil
    }

    /// Generates the answer to `prompt` (ids already templated). `onTokens` receives each round's tokens and can stop
    /// generation by returning false.
    public func generate(
        prompt: [UInt32], maxTokens: Int, sampling: SamplingParams, stopTokens: Set<UInt32>,
        onTokens: (([UInt32]) -> Bool)? = nil
    ) throws -> Result {
        try generate(
            prompt: prompt, maxTokens: maxTokens, sampling: sampling, stopTokens: stopTokens, onTokens: onTokens,
            yield: nil)
    }

    /// `generate`, and with `yield` (ingot serve batching) decoding stops early when it returns true before a chained
    /// round: the result then has `yielded` and the last emitted token outside the cache, so the conversation goes on
    /// in the joint rounds (BatchServe.swift).
    func generate(
        prompt: [UInt32], maxTokens: Int, sampling: SamplingParams, stopTokens: Set<UInt32>,
        onTokens: (([UInt32]) -> Bool)?, yield: (() -> Bool)?
    ) throws -> Result {
        let m = model
        var (res, first, rng) = try start(prompt: prompt, maxTokens: maxTokens, sampling: sampling)
        let n = prompt.count
        m.decodeYield = yield
        m.decodeYielded = false
        defer { m.decodeYield = nil }
        do {
            // Context-copy index (with a drafter or, in LFM2, without: one-row rounds and copy rounds): only the new
            // part is added if the prompt continues the indexed history.
            var ci: CopyIndex?
            if m.options.copyEnabled(moe: m.isMoE, drafter: drafter != nil) {
                let tc = DispatchTime.now().uptimeNanoseconds
                copyIndex.sync(to: prompt)
                res.copyIndexMs = Double(DispatchTime.now().uptimeNanoseconds - tc) / 1e6
                ci = copyIndex
            }
            // Every request starts from the policy priors, so its round structure (and, with sampling, the
            // uniforms indexed by round) does not depend on earlier requests.
            copyState.policy = nil
            res.stats = try m.decode(
                first: first, drafter: drafter, maxTokens: maxTokens, sampling: sampling, rng: &rng,
                stopTokens: stopTokens, copyIndex: ci, copyState: copyState, onTokens: onTokens)
        } catch {
            // Unknown state: the next request starts from scratch.
            live = []; snaps = []; m.nacc = 0; m.reset(); copyIndex.reset()
            throw error
        }
        res.yielded = m.decodeYielded
        live += res.stats.tokens.prefix(m.P - n)
        precondition(live.count == m.P)
        precondition(!res.yielded || live.count == n + res.stats.tokens.count - 1)
        res.stats.prefillSeconds = res.prefillSeconds
        log?(
            "cache: \(res.cacheNote), \(res.cachedTokens) of \(n) tokens reused; live state \(live.count) tokens; "
                + "snaps \(snaps.map { $0.tokens.count })\(entries.isEmpty ? "" : "; entries " + prefixCacheSummary)"
                + (res.liveMiss.map { "; " + $0 } ?? "")
        )
        balanceKV()  // the KV may have grown during decoding
        return res
    }

    /// The part of `generate` before decoding: the starting point (live state, snapshot or prefix cache entry), the
    /// prompt read on top of it and the first token sampled. On return `live` is the prompt and the first token is
    /// outside the cache.
    func start(prompt: [UInt32], maxTokens: Int, sampling: SamplingParams) throws -> (Result, UInt32, SplitMix64) {
        let m = model
        guard !prompt.isEmpty else { throw EngineError.emptyPrompt }
        guard prompt.count + DFlash2Drafter.margin(drafter) <= m.maxCtx else {
            throw EngineError.contextOverflow(prompt: prompt.count, maxCtx: m.maxCtx)
        }
        var res = Result(stats: .init())
        res.promptTokens = prompt.count
        let n = prompt.count
        // Starting point (at least the last token must be left to read, except for an exact repetition from
        // snapshot (c)): the live state if it is a prefix, or the longest snapshot that is.
        // Ring KV: the live state, a snapshot or an entry restart at c only if the window of c is still in the ring
        // (`kvIntact`; for the live state it always is: what was written past it are at most the rejected rows of the
        // last round, which the ring size allows for).
        var liveOK = !live.isEmpty && live.count < n && Self.isPrefix(live, of: prompt) && kvIntact(live.count)
        let snap = snaps.last {
            // Snapshot (c) only for an exact repetition: the paths of every other prompt do not change.
            ($0.lastRow == nil ? $0.tokens.count < n : $0.tokens.count == n) && Self.isPrefix($0.tokens, of: prompt)
                && kvIntact($0.tokens.count)
        }
        // With a KV-only state, a live state that is not a prefix (the template rewrote the answer its own
        // way: harmony's tojson, a token the tokenizer merges) goes back to the common prefix c, as if it had ended
        // there, when that beats the snapshot and the entry it would take otherwise. The conversation is not
        // abandoned (no promote); the turn is read from c by the same path as after a live state ending at c.
        // Only for a divergence inside the answer: c reaches the end of the prompt that left the live state (its
        // longest snapshot, (b) or (c)). A prompt that diverges earlier is another conversation sharing a prefix
        // (pi sessions in another cwd): the prefix cache keeps handling it (promote and the "common" entry at the
        // shared prefix, which later prompts may return to).
        if !liveOK && !live.isEmpty { res.liveMiss = liveMissReason(prompt) }
        if let s = snaps.last(where: { Self.isPrefix($0.tokens, of: prompt) && !kvIntact($0.tokens.count) }),
            s.tokens.count > (snap?.tokens.count ?? 0)
        {
            res.liveMiss = (res.liveMiss.map { $0 + "; " } ?? "") + "snapshot \(s.tokens.count) dropped: ring"
        }
        if !liveOK && !live.isEmpty && stateIsKVOnly {
            let c = min(Self.lcp(live, prompt), n - 1)
            let answerStart = snaps.filter { Self.isPrefix($0.tokens, of: live) }.map { $0.tokens.count }.max() ?? 0
            let entLen =
                entries.filter { $0.tokens.count < n && Self.isPrefix($0.tokens, of: prompt) }.map {
                    $0.tokens.count
                }.max() ?? 0
            if c >= answerStart && c > max(snap?.tokens.count ?? 0, entLen) && kvIntact(c) {
                res.rolledBackFrom = live.count
                live = Array(live.prefix(c))
                liveOK = true
            } else {
                let why =
                    c < answerStart
                    ? "before the answer" : !kvIntact(c) ? "ring" : "snapshot or entry reaches further"
                res.liveMiss = (res.liveMiss.map { $0 + "; " } ?? "") + "no rollback to \(c): \(why)"
            }
        }
        let t0 = DispatchTime.now().uptimeNanoseconds
        // Entries whose KV is still in the live cache and are not a prefix of the prompt copy it now (what comes next
        // rewrites or trims it).
        flushPendingKV { !Self.isPrefix($0.tokens, of: prompt) }
        // The previous conversation is abandoned (its KV will be rewritten): its longest snapshot becomes an entry.
        if !liveOK && prefixCacheUsable,
            let old = snaps.filter({ $0.lastRow == nil && !Self.isPrefix($0.tokens, of: prompt) }).max(by: {
                $0.tokens.count < $1.tokens.count
            }),
            old.tokens.count >= minEntryTokens
        {
            promote(old)
        }
        // KVPool (Flash-Next in serve): KV for the prompt and up to 4096 answer tokens (the rest is mapped during
        // decoding); spare ranges of an earlier, longer conversation go back to the experts.
        // After promote (copies the KV of the abandoned conversation) and before restoring anything.
        if m.kvPool != nil {
            let need = n + min(maxTokens, Self.kvAhead) + DFlash2Drafter.margin(drafter)
            try m.trimContext(need)
            let ok = try m.ensureContext(need) || m.ensureContext(n + DFlash2Drafter.margin(drafter))
            if !ok {
                throw EngineError.contextOverflow(prompt: n, maxCtx: m.ctxLimit)
            }
            balanceKV()
        }
        // After promote: saving the abandoned conversation may evict entries (with one chosen earlier, an already
        // evicted entry was restored and preparePrompt failed its precondition).
        let ent = entries.filter { $0.tokens.count < n && Self.isPrefix($0.tokens, of: prompt) }.max {
            $0.tokens.count < $1.tokens.count
        }
        let liveLen = liveOK ? live.count : 0, snapLen = snap?.tokens.count ?? 0
        // Entries still pending are a prefix of the prompt; they stay so if what will be rewritten starts after them
        // (restoring another entry rewrites the KV from 0).
        let fromEntry = ent.map { $0.tokens.count > max(liveLen, snapLen) } ?? false
        let intact = fromEntry ? (ent!.kvPending ? ent!.tokens.count : 0) : max(liveLen, snapLen)
        flushPendingKV { $0.tokens.count > intact }
        if let ent, ent.tokens.count > max(liveLen, snapLen) {
            restoreEntry(ent)
            res.cachedTokens = ent.tokens.count; res.cacheSource = "entry"
        } else if liveOK && live.count >= snapLen {
            if res.rolledBackFrom != nil { model.restoreState(live) }
            res.cachedTokens = live.count; res.cacheSource = "live"
        } else if let snap {
            restore(snap)
            res.cachedTokens = snap.tokens.count; res.cacheSource = "snapshot"
        } else {
            live = []
        }
        // Snapshots that are not a prefix of the prompt become invalid (their KV region will be rewritten).
        let kept = snaps.filter { Self.isPrefix($0.tokens, of: prompt) }
        spare += snaps.filter { s in !kept.contains { $0 === s } }
        snaps = kept
        let gs = Self.generationStart(prompt, marker: generationMarker)
        let cuts = prefixCuts(prompt, from: res.cachedTokens)
        recentPrompts.append(prompt)
        if recentPrompts.count > 8 { recentPrompts.removeFirst() }
        var first: UInt32
        let row: Int
        if res.cachedTokens == n, let lr = snap?.lastRow {
            // Exact repetition: snapshot (c) holds the state after the whole prompt and its last row.
            m.setDecodeChunk(promptTokens: n)
            lr.logits.withUnsafeBytes { m.logits.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
            (first, row) = (lr.argmax, 0)
        } else {
            (first, row) = try readPrompt(prompt, res: res, gs: gs, cuts: cuts)
        }
        res.prefillSeconds = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9
        onPromptLogits?(row)
        var rng = SplitMix64(seed: sampling.seed ?? UInt64.random(in: 0...UInt64.max))
        m.preparePenalties(prompt: prompt, params: sampling)
        first = try m.sampleFromLogits(argmax: first, row: row, params: sampling, rng: &rng)
        live = prompt
        liveAnswerStart = n
        return (res, first, rng)
    }

    /// Why the live state is not a usable prefix of `prompt` (only for the log): where it diverges (and the answer
    /// token, counted from the prompt that left it), or that it is a prefix the ring no longer allows.
    func liveMissReason(_ prompt: [UInt32]) -> String {
        let c = Self.lcp(live, prompt)
        if c == live.count {
            return live.count >= prompt.count ? "live state not shorter than the prompt" : "live: ring"
        }
        let a = liveAnswerStart
        let at = c >= a ? "answer token \(c - a) of \(live.count - a)" : "before the answer (prompt of \(a))"
        return "live diverged at \(c) (\(at))"
    }

    /// Reads `prompt[res.cachedTokens...]` (preparePrompt with the snapshots and prefix cache entries of its points),
    /// then takes snapshot (c). Returns the argmax of the last row and its row in `logits`.
    private func readPrompt(_ prompt: [UInt32], res: Result, gs: Int?, cuts: [(Int, String)]) throws -> (UInt32, Int) {
        let m = model
        var first: UInt32
        let row: Int
        // Free snapshots as targets of in-pass captures (unused ones are returned).
        var capSnaps: [StateSnapshot] = [], usedCap = Set<Int>()
        var cutEntries: [PrefixEntry] = []
        if maxSnapshots > 0 && m.canCaptureInPass {
            capSnaps = (0..<HybridModel.cutSlotBase).map { _ in
                spare.popLast() ?? StateSnapshot(model: m, drafter: drafter)
            }
            // Flash-Next: prefix cache cuts are captured inside the pass directly into their entries, reserved
            // here (as in `saveEntry`, evicting if needed); without room for all, the prefill is split at the
            // cuts.
            // Outside Flash-Next, the same reservation, but they are captured as rows of the last chunk
            // (`rowCuts`); those that do not fall in it split the prefill, filling their reserved entry.
            let caps = m.prefillCaps
            cutEntries = caps.rowCuts ? reserveCutEntries(prompt, cuts) : []
            let targets = capSnaps + cutEntries.map(\.snap)
            m.captureTargets = targets.map { $0.gdn }
            m.captureConvTargets = targets.map { $0.conv }
            m.captureDraftTargets = drafter == nil ? nil : targets.map { $0.draft }
        }
        defer {
            m.captureTargets = nil
            m.captureConvTargets = nil
            m.captureDraftTargets = nil
            spare += capSnaps.enumerated().filter { !usedCap.contains($0.offset) }.map(\.element)
        }
        do {
            // The conversation cut does not split the pass: it is captured as one more row. Outside Flash-Next, every
            // cut whose entry was reserved (system and common too; else they split the prefill, as before).
            let caps = m.prefillCaps
            let rowCuts = Set(
                !caps.rowCuts
                    ? []
                    : caps.cutsInPass ? cuts.filter { $0.1 == "conversation" }.map(\.0) : cutEntries.map(\.tokens.count)
            )
            (first, row, _) = try m.preparePrompt(
                prompt, from: res.cachedTokens, genStart: gs, drafter: drafter, cuts: cuts.map(\.0), rowCuts: rowCuts
            ) { point, len in
                if case .current = point, let k = cuts.first(where: { $0.0 == len }) {
                    self.saveEntry(tokens: Array(prompt[..<len]), kind: k.1)
                } else if case .captured(let k) = point, k >= HybridModel.cutSlotBase {
                    self.finishCutEntry(cutEntries[k - HybridModel.cutSlotBase], tokens: Array(prompt[..<len]))
                } else if case .captured(let k) = point {
                    usedCap.insert(k)
                    self.addCaptured(capSnaps[k], tokens: Array(prompt[..<len]))
                } else {
                    self.takeSnapshot(point, tokens: Array(prompt[..<len]))
                }
            }
        } catch {
            live = []; snaps = []; m.nacc = 0; m.reset()
            entries.removeAll { e in cutEntries.contains { $0 === e } && !e.done }
            throw error
        }
        // Reserved entries that were not filled (the prefill did not go through the capturing pass) are removed.
        entries.removeAll { e in cutEntries.contains { $0 === e } && !e.done }
        takeWholePromptSnapshot(prompt, argmax: first, row: row)
        return (first, row)
    }

    /// Snapshot (c): the state after the whole prompt (folding the rows still on the tape) with the logits of its last
    /// row, before sampling penalizes them. Only with room for three snapshots, so that (a) and (b) of the same prompt
    /// stay.
    private func takeWholePromptSnapshot(_ prompt: [UInt32], argmax: UInt32, row: Int) {
        guard snapshotLimit >= 3 else { return }
        let m = model, V = m.vocab
        let p = m.logits.contents().bindMemory(to: Float.self, capacity: (row + 1) * V)
        takeSnapshot(m.nacc > 0 ? .tape(rows: m.nacc) : .current, tokens: prompt)
        snaps.last!.lastRow = (argmax, Array(UnsafeBufferPointer(start: p + row * V, count: V)))
    }

    func takeSnapshot(_ point: HybridModel.SnapshotPoint, tokens: [UInt32]) {
        guard snapshotLimit > 0 else { return }
        if snaps.count >= snapshotLimit { spare.append(snaps.removeFirst()) }
        let s = spare.popLast() ?? StateSnapshot(model: model, drafter: drafter)
        s.tokens = tokens
        s.lastRow = nil
        copy(
            s, restore: false,
            foldTape: {
                if case .tape(let r) = point { return r }; return nil
            }())
        snaps.append(s)
    }

    /// Snapshot whose GDN was already captured inside the pass; the drafter ring is still missing.
    func addCaptured(_ s: StateSnapshot, tokens: [UInt32]) {
        if snaps.count >= max(1, snapshotLimit) { spare.append(snaps.removeFirst()) }
        s.tokens = tokens
        s.lastRow = nil
        copy(s, restore: false, foldTape: nil, gdn: false)
        snaps.append(s)
    }

    func restore(_ s: StateSnapshot) {
        copy(s, restore: true, foldTape: nil)
        model.restoreState(s.tokens)
        live = s.tokens
    }

    /// Copies the current state into the snapshot (or the reverse, `restore`). With `foldTape`, the GDN state saved is
    /// the current one plus those tape rows (point (b) of Prompt.swift). Goes on the queue, after what is queued.
    func copy(_ s: StateSnapshot, restore: Bool, foldTape: Int?, gdn: Bool = true) {
        let m = model
        let cb = m.ctx.queue.makeCommandBuffer()!
        if let k = foldTape {
            // Folding only fails if a pipeline of the package itself is missing: a programming error, not an input one.
            func fold(_ body: () throws -> Void) {
                do { try body() } catch { preconditionFailure("snapshot tape fold: \(error)") }
            }
            let e = cb.makeComputeCommandEncoder()!
            var i = 0, j = 0
            for l in m.layers {
                if case .gdn(let g) = l.mixer {
                    fold { try g.encodeFold(e, nacc: k, stateOut: s.gdn[i].state, histOut: s.gdn[i].hist) }
                    i += 1
                }
                if case .conv(let c) = l.mixer {
                    fold { try c.encodeFold(e, nacc: k, histOut: s.conv[j]) }
                    j += 1
                }
            }
            if let p = m.fn?.ple {
                fold { try p.encodeFold(m.ctx, e, nacc: k, histOut: s.conv[j]) }
                j += 1
            }
            for q in m.fn?.qsaList ?? [] {
                fold { try q.encodeFold(m.ctx, e, nacc: k, histOut: s.conv[j]) }
                j += 1
            }
            e.endEncoding()
        }
        let b = cb.makeBlitCommandEncoder()!
        func cp(_ live: MTLBuffer, _ snap: MTLBuffer, _ n: Int) {
            if restore {
                b.copy(from: snap, sourceOffset: 0, to: live, destinationOffset: 0, size: n)
            } else {
                b.copy(from: live, sourceOffset: 0, to: snap, destinationOffset: 0, size: n)
            }
        }
        if foldTape == nil && gdn {
            var i = 0, j = 0
            for l in m.layers {
                if case .gdn(let g) = l.mixer {
                    cp(g.state[g.cur], s.gdn[i].state, s.gdn[i].state.length)
                    cp(g.hist[g.cur], s.gdn[i].hist, s.gdn[i].hist.length)
                    i += 1
                }
                if case .conv(let c) = l.mixer {
                    cp(c.hist[c.cur], s.conv[j], s.conv[j].length)
                    j += 1
                }
            }
            if let p = m.fn?.ple {
                cp(p.hist[p.cur], s.conv[j], s.conv[j].length)
                j += 1
            }
            for q in m.fn?.qsaList ?? [] {
                cp(q.hist[q.cur], s.conv[j], s.conv[j].length)
                j += 1
            }
        }
        for (j, l) in (drafter?.layers ?? []).enumerated() {
            cp(l.kc, s.draft[j].kc, s.draft[j].kc.length)
            cp(l.vt, s.draft[j].vt, s.draft[j].vt.length)
        }
        b.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
    }

    /// For ingot-dev reuse: restores the longest snapshot that is a prefix of `prompt` (no generation).
    public func restoreForTest(prefixOf prompt: [UInt32]) {
        if let s = snaps.last(where: { Self.isPrefix($0.tokens, of: prompt) }) { restore(s) }
    }

    /// Forgets the cache (the next request starts from scratch).
    public func clear() {
        spare += snaps; snaps = []; live = []; model.reset()
        clearPrefixCache()
    }
}

extension HybridModel {
    /// After restoring a snapshot or a prefix cache entry with these tokens: P = tokens.count in the cache, no rows
    /// left to fold. Flash-Next: the PLE reads the two tokens before each history position by position (`ptok`),
    /// which is not in the snapshot and may have been rewritten by another conversation; they are rewritten here so
    /// the restored state stands on its own (not only behind preparePrompt).
    func restoreState(_ tokens: [UInt32]) {
        P = tokens.count
        nacc = 0
        flashSetHistory(tokens, before: tokens.count)
    }
}
