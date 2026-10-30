import CryptoKit
import Foundation
import Metal

// Prefix cache across sessions.
//
// The light snapshots (Engine.swift) keep only the recurrent state and depend on the live KV: as soon as a prompt
// diverges earlier, they are dropped. That is enough within one conversation, but every new pi session shares with
// the previous one ~4,360 of its ~4,600 tokens (system + tools; only the `<cwd>` changes) and they are re-read in
// full. An entry of this cache also keeps a compact copy of the attention KV for positions 0..<P, so it survives any
// later prompt. Several are kept (LRU with a byte limit) and a request resumes from the longest entry that is a
// prefix of its prompt.
//
// Where entries are taken (always splitting the prefill, so the entry is the exact state at that point):
// - shortly before the end of the system block (the first `<|im_start|>user`, aligned to `cutAlign`, 64 tokens
//   earlier): new pi sessions in any folder, if what changes (the `<cwd>`) fits in those 64-127 tokens;
// - the boundary of the prefix shared with recent prompts and entries (LCP - 8, aligned; idea from Rapid-MLX):
//   from the third session with another `<cwd>`, only ~250 tokens are read;
// - and the abandoned conversation: if a prompt does not continue the previous one, its longest light snapshot
//   becomes an entry before the KV is overwritten (going back to an earlier conversation does not re-read it).

final class PrefixEntry {
    let snap: StateSnapshot
    /// Per attention layer: kc [h][P][hd] int8, vt [h·hd][P] int8, ks [h][P] f32, vs [h][P] f32. A ring layer
    /// keeps only the last W positions instead of P (`AttentionLayer.entryWindow`), in position order.
    let kv: MTLBuffer
    let kind: String
    var lastUse: UInt64 = 0
    var hits = 0
    /// False while the entry is reserved for a cut captured inside the pass (no data yet).
    var done = true
    /// The KV 0..<P is still only in the live cache (entry captured in the pass): it is copied to `kv` when the
    /// conversation is abandoned (`flushPendingKV`). Until then, restoring it only restores the state.
    var kvPending = false
    var tokens: [UInt32] { snap.tokens }
    var bytes: Int { snap.bytes + kv.length }
    init(snap: StateSnapshot, kv: MTLBuffer, kind: String) {
        self.snap = snap
        self.kv = kv
        self.kind = kind
    }
}

extension InferenceEngine {
    /// The attention layers with their own KV (a KV-sharing layer, Gemma 4 E2B/E4B, reads its source's: the entries,
    /// the ring checks and the copies go through the source only).
    var attnLayers: [AttentionLayer] {
        model.layers.compactMap {
            if case .attn(let a) = $0.mixer, a.kvSource == nil { return a }; return nil
        }
    }

    /// KV bytes of an entry of P tokens: per attention layer, the positions it keeps (all, or the last W with a
    /// ring).
    /// Flash-Next: plus the QSA block keys (128 fp32 every 4 tokens).
    func entryKVBytes(tokens P: Int) -> Int {
        // Per layer, its own shape (Gemma 4's full layers have other heads).
        return attnLayers.reduce(0) {
            $0 + AttentionLayer.kvBytesPerPosition(kvHeads: $1.kvHeads, hd: $1.hd) * $1.entryWindow(P).n
        }
            + (model.fn?.qsa.count ?? 0) * FlashQSA.D * P
    }

    /// Whether some attention layer keeps its KV in a ring.
    var hasRingKV: Bool { attnLayers.contains { $0.ring > 0 } }

    /// Whether the entry cache is on (it has a byte budget).
    var prefixCacheUsable: Bool { prefixCacheBytes > 0 }

    /// Whether the live KV can restart at position c (every ring layer keeps the window of c).
    func kvIntact(_ c: Int) -> Bool { attnLayers.allSatisfy { $0.ringIntact(c) } }

    static func lcp(_ a: [UInt32], _ b: [UInt32]) -> Int {
        let n = min(a.count, b.count)
        var i = 0
        while i < n && a[i] == b[i] { i += 1 }
        return i
    }

    /// Copies the KV 0..<P between the live cache and an entry (into `b`). With a ring, the last W positions of the
    /// layer: slot = position % ring, so the window can wrap around the end of the ring (two pieces per row).
    func blitKV(_ b: MTLBlitCommandEncoder, entry kv: MTLBuffer, P: Int, restore: Bool) {
        var off = 0
        func cp(_ live: MTLBuffer, _ lo: Int, _ eo: Int, _ n: Int) {
            guard n > 0 else { return }
            if restore {
                b.copy(from: kv, sourceOffset: eo, to: live, destinationOffset: lo, size: n)
            } else {
                b.copy(from: live, sourceOffset: lo, to: kv, destinationOffset: eo, size: n)
            }
        }
        for a in attnLayers {
            let ld = a.ldkv, h = a.kvHeads, hd = a.hd
            let (lo, n) = a.entryWindow(P)
            // (first slot, first position in the entry, positions): one piece, or two if the window wraps the ring.
            var pieces = [(a.ring > 0 ? lo % a.ring : lo, 0, n)]
            if a.ring > 0 && pieces[0].0 + n > a.ring {
                let first = a.ring - pieces[0].0
                pieces = [(pieces[0].0, 0, first), (0, first, n - first)]
            }
            for (s, e, len) in pieces {
                for i in 0..<h { cp(a.kc, (i * ld + s) * hd, off + (i * n + e) * hd, len * hd) }
            }
            off += h * n * hd
            if !a.latentKV {  // a latent KV has no vt (V is read from kc)
                for (s, e, len) in pieces {
                    for r in 0..<(h * hd) { cp(a.vt, r * ld + s, off + r * n + e, len) }
                }
                off += h * hd * n
            }
            for (s, e, len) in pieces {
                for i in 0..<h { cp(a.ks, (i * ld + s) * 4, off + (i * n + e) * 4, len * 4) }
            }
            off += h * n * 4
            for (s, e, len) in pieces {
                for i in 0..<h { cp(a.vs, (i * ld + s) * 4, off + (i * n + e) * 4, len * 4) }
            }
            off += h * n * 4
            if restore { a.ringRestored(at: P) }
        }
        // Flash-Next: keys of the complete QSA blocks (those of the open block are recomputed).
        for q in model.fn?.qsaList ?? [] {
            if P / 4 > 0 { cp(q.bk, 0, off, (P / 4) * FlashQSA.D * 4) }
            off += P * FlashQSA.D
        }
    }

    var entriesBytes: Int { entries.reduce(0) { $0 + $1.bytes } }

    /// Bytes taken by an entry of P tokens (snapshot + KV in multiples of 1024 tokens, as in `makeEntry`).
    public func prefixEntryBytes(tokens P: Int) -> Int {
        snapshotBytes + max(16, entryKVBytes(tokens: (P + 1023) / 1024 * 1024))
    }

    /// Reserves room for an entry of P tokens (evicts the least recently used); nil if it does not fit.
    func makeEntry(P: Int, kind: String) -> PrefixEntry? {
        // KV in multiples of 1024 tokens: buffers of an evicted entry can usually be reused for the new one (creating
        // new buffers is costly: Metal zeroes them on first touch).
        let kvLen = max(16, entryKVBytes(tokens: (P + 1023) / 1024 * 1024))
        let need = prefixEntryBytes(tokens: P)
        guard need <= prefixCacheBytes else { return nil }
        var freed: [PrefixEntry] = []
        // Entries reserved for a cut of the pass in progress are not evicted.
        while entriesBytes + need > prefixCacheBytes,
            let lru = entries.filter({ $0.done }).min(by: { $0.lastUse < $1.lastUse })
        {
            entries.removeAll { $0 === lru }
            freed.append(lru)
            log?(
                "prefix cache: evicted entry of \(lru.tokens.count) tokens (\(lru.kind), \(lru.hits) uses)"
            )
        }
        if entriesBytes + need > prefixCacheBytes { return nil }
        let reuse = freed.filter { $0.kv.length >= kvLen }.min { $0.kv.length < $1.kv.length }
        let snap = reuse?.snap ?? freed.first?.snap ?? StateSnapshot(model: model, drafter: drafter)
        let e = PrefixEntry(snap: snap, kv: reuse?.kv ?? model.ctx.buffer(length: kvLen), kind: kind)
        useClock += 1
        e.lastUse = useClock
        return e
    }

    /// Entry with the current state (P tokens in the cache, no rows left to fold).
    func saveEntry(tokens: [UInt32], kind: String) {
        precondition(model.P == tokens.count && model.nacc == 0)
        if entries.contains(where: { $0.tokens == tokens && $0.done }) { return }
        // Ring KV: the window of P must still be in the ring (always, right after reading up to P from a state
        // the ring keeps; not after a rewind past what the ring holds).
        guard kvIntact(tokens.count) else { return }
        let t0 = DispatchTime.now().uptimeNanoseconds
        // The one reserved for capture in the pass, if the prefill ended up split at the cut.
        let reserved = entries.first { $0.tokens == tokens && !$0.done }
        guard let e = reserved ?? makeEntry(P: tokens.count, kind: kind) else { return }
        e.snap.tokens = tokens
        e.done = true
        copy(e.snap, restore: false, foldTape: nil)
        let cb = model.ctx.queue.makeCommandBuffer()!
        let b = cb.makeBlitCommandEncoder()!
        blitKV(b, entry: e.kv, P: tokens.count, restore: false)
        b.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        if reserved == nil { entries.append(e) }
        dropSupersededConversations(by: e)
        log?(
            String(
                format: "prefix cache: entry of %d tokens (%@) in %.1f ms", tokens.count, kind,
                Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6))
    }

    /// Reserves the entries for the cuts `cuts` (those of `prefixCuts`) to capture them inside the pass
    /// (Prompt.swift): they go into `entries` right away so they count when reserving the next one, but hold no data
    /// (`done` = false) until `finishCutEntry`. If one does not fit, none is reserved (prefill is split at the cuts).
    func reserveCutEntries(_ prompt: [UInt32], _ cuts: [(Int, String)]) -> [PrefixEntry] {
        // Ring KV: the pass goes on writing the ring after the cut, so the entry cannot wait for the end of the
        // pass nor keep its KV pending in the live cache; the prefill is split at the cuts (`saveEntry`).
        if hasRingKV { return [] }
        var r: [PrefixEntry] = []
        for (cut, kind) in cuts {
            guard let e = makeEntry(P: cut, kind: kind) else {
                entries.removeAll { e in r.contains { $0 === e } }
                return []
            }
            e.snap.tokens = Array(prompt[..<cut])
            e.done = false
            entries.append(e)
            r.append(e)
            // With room for only one, reserving the second evicts (and reuses) the first.
            if !r.allSatisfy({ x in entries.contains { $0 === x } }) {
                entries.removeAll { e in r.contains { $0 === e } }
                return []
            }
        }
        return r
    }

    /// Completes a reserved entry: the state is already captured in `e.snap` (GDN, windows and drafter ring); the KV
    /// 0..<P is missing, which the pass does not rewrite after the cut.
    /// The KV stays in the live cache until needed (`flushPendingKV`): within a conversation it is never copied (the
    /// next entry of the conversation replaces this one).
    func finishCutEntry(_ e: PrefixEntry, tokens: [UInt32]) {
        precondition(e.snap.tokens == tokens && !e.done && !hasRingKV)
        e.done = true
        e.kvPending = true
        dropSupersededConversations(by: e)
        log?("prefix cache: entry of \(tokens.count) tokens (\(e.kind), captured in the pass; KV pending)")
    }

    /// Copies into their entry the KV of the pending entries matching `pred`, before the live cache overwrites it.
    /// Invariant: the KV 0..<P of a pending entry is still intact in the live cache.
    func flushPendingKV(_ pred: (PrefixEntry) -> Bool) {
        let list = entries.filter { $0.kvPending && pred($0) }
        guard !list.isEmpty else { return }
        let t0 = DispatchTime.now().uptimeNanoseconds
        let cb = model.ctx.queue.makeCommandBuffer()!
        let b = cb.makeBlitCommandEncoder()!
        for e in list { blitKV(b, entry: e.kv, P: e.tokens.count, restore: false) }
        b.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        for e in list { e.kvPending = false }
        log?(
            String(
                format: "prefix cache: KV of %@ copied in %.1f ms",
                list.map { "\($0.tokens.count) (\($0.kind))" }.joined(separator: ", "),
                Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6))
    }

    /// A new "conversation" entry replaces the earlier ones of the same conversation (its prefixes).
    func dropSupersededConversations(by e: PrefixEntry) {
        guard e.kind == "conversation" else { return }
        entries.removeAll {
            $0 !== e && $0.done && $0.kind == "conversation" && Self.isPrefix($0.tokens, of: e.tokens)
        }
    }

    /// A light snapshot (valid with the live KV) becomes an entry.
    func promote(_ s: StateSnapshot) {
        // Only those aligned to 64: continuing from an unaligned entry does not match the cacheless result.
        // Aligned "conversation" entries are taken by `prefixCuts` when reading the prompt.
        if s.tokens.count % cutAlign != 0 { return }
        if entries.contains(where: { InferenceEngine.isPrefix(s.tokens, of: $0.tokens) }) { return }
        // Ring KV: only if the window of the snapshot is still in the ring.
        guard kvIntact(s.tokens.count) else { return }
        let t0 = DispatchTime.now().uptimeNanoseconds
        guard let e = makeEntry(P: s.tokens.count, kind: "conversation") else { return }
        e.snap.tokens = s.tokens
        let cb = model.ctx.queue.makeCommandBuffer()!
        let b = cb.makeBlitCommandEncoder()!
        for (x, y) in zip(s.gdn, e.snap.gdn) {
            b.copy(from: x.state, sourceOffset: 0, to: y.state, destinationOffset: 0, size: x.state.length)
            b.copy(from: x.hist, sourceOffset: 0, to: y.hist, destinationOffset: 0, size: x.hist.length)
        }
        for (x, y) in zip(s.draft, e.snap.draft) {
            b.copy(from: x.kc, sourceOffset: 0, to: y.kc, destinationOffset: 0, size: x.kc.length)
            b.copy(from: x.vt, sourceOffset: 0, to: y.vt, destinationOffset: 0, size: x.vt.length)
        }
        for (x, y) in zip(s.conv, e.snap.conv) {
            b.copy(from: x, sourceOffset: 0, to: y, destinationOffset: 0, size: x.length)
        }
        blitKV(b, entry: e.kv, P: s.tokens.count, restore: false)
        b.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        entries.append(e)
        log?(
            String(
                format: "prefix cache: conversation of %d tokens saved in %.1f ms", s.tokens.count,
                Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6))
    }

    func restoreEntry(_ e: PrefixEntry) {
        // With the KV pending, the live cache already holds its KV.
        if !e.kvPending {
            let cb = model.ctx.queue.makeCommandBuffer()!
            let b = cb.makeBlitCommandEncoder()!
            blitKV(b, entry: e.kv, P: e.tokens.count, restore: true)
            b.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        }
        copy(e.snap, restore: true, foldTape: nil)
        model.restoreState(e.tokens)
        live = e.tokens
        useClock += 1
        e.lastUse = useClock
        e.hits += 1
    }

    /// Kind of an entry read from disk: the stored kind with the `,disk` suffix, once.
    static func diskKind(_ stored: String) -> String {
        stored.replacingOccurrences(of: ",disk", with: "") + ",disk"
    }

    /// Prefill cut points for a prompt read from `c` (see above).
    func prefixCuts(_ prompt: [UInt32], from c: Int) -> [(Int, String)] {
        guard prefixCacheUsable else { return [] }
        let n = prompt.count
        let al = { (x: Int) in x / self.cutAlign * self.cutAlign }
        var cuts: [(Int, String)] = []
        if !userMarker.isEmpty, let u = Self.firstIndex(of: userMarker, in: prompt) {
            // 64-127 tokens before the end of the system block, which tends to change between sessions (pi's
            // `<cwd>`). A single point: a cut before the last chunk adds a prefill chunk, and a short chunk reads
            // almost all experts (in the last chunk it is captured as a row, Prompt.swift `rowCuts`).
            cuts.append((al(u) - cutAlign, "system"))
        }
        var d = 0
        for r in recentPrompts { d = max(d, Self.lcp(r, prompt)) }
        for e in entries { d = max(d, Self.lcp(e.tokens, prompt)) }
        if d > 8 { cuts.append((al(d - 8), "common")) }
        // The conversation: an entry at the multiple of 64 before point (a) (or n - 1), so it can be resumed with the
        // prefill split at an aligned point (chunked GDN groups by 64: bit-identical to reading the whole prompt)
        // at the cost of recomputing at most 63 tokens. Snapshots (a)/(b) do not work for this:
        // they are not aligned.
        let gs = Self.generationStart(prompt, marker: generationMarker)
        let plan = model.promptPlan(count: n, from: c, genStart: gs)
        let conv = al(plan.s)
        var seen = Set<Int>()
        // Flash-Next (`cutsInPass`), only when n - c >= expertBulkMinRows (before, if not captured inside the pass,
        // it split the prefill into two SSD passes: +1.2 s with 242 rows from the entry). There it never splits the
        // pass: it is captured as a row of the last chunk or not taken (Prompt.swift, `rowCuts`), and its KV is copied
        // when the conversation is abandoned. The other families capture it as a row too when its entry can be
        // reserved (Engine.readPrompt), and otherwise split the prefill at it.
        if !model.prefillCaps.cutsInPass || n - c >= model.options.expertBulkMinRows {
            cuts.append((conv, "conversation"))
        }
        return cuts.filter { cut, kind in
            cut >= c + 64 && (kind == "conversation" ? cut < plan.s : cut <= n - 32) && cut >= minEntryTokens
                && seen.insert(cut).inserted
                && !entries.contains { $0.tokens.count == cut && Self.isPrefix($0.tokens, of: prompt) }
        }.sorted { $0.0 < $1.0 }
    }

    static func firstIndex(of m: [UInt32], in ids: [UInt32]) -> Int? {
        guard !m.isEmpty, ids.count >= m.count else { return nil }
        for i in 0...(ids.count - m.count) where ids[i] == m[0] {
            if Array(ids[i..<(i + m.count)]) == m { return i }
        }
        return nil
    }

    public var prefixCacheSummary: String {
        entries.sorted { $0.tokens.count < $1.tokens.count }.map { "\($0.tokens.count)(\($0.kind),\($0.hits))" }.joined(
            separator: " ")
    }

    /// Forgets the prefix cache entries.
    public func clearPrefixCache() {
        entries = []
        recentPrompts = []
    }
}

extension InferenceEngine {
    /// For lab-dev prefixcache: restores the entry with exactly these tokens (no generation).
    public func restoreEntryForTest(tokens: [UInt32]) -> Bool {
        guard let e = entries.first(where: { $0.tokens == tokens }) else { return false }
        spare += snaps; snaps = []
        restoreEntry(e)
        return true
    }

    /// Forgets only the live state: the next request resumes from the light snapshot or the longest entry.
    public func forgetLiveOnly() { live = [] }

    /// Forgets the live state and the light snapshots, but not the entries.
    public func forgetLive() {
        flushPendingKV { _ in true }
        spare += snaps; snaps = []; live = []; model.reset()
    }
}

// MARK: - On disk

extension InferenceEngine {
    func entryBuffers(_ e: PrefixEntry) -> [MTLBuffer] {
        e.snap.gdn.flatMap { [$0.state, $0.hist] } + e.snap.draft.flatMap { [$0.kc, $0.vt] } + e.snap.conv + [e.kv]
    }

    static func sha(_ t: [UInt32]) -> String {
        t.withUnsafeBytes { SHA256.hash(data: $0) }.map { String(format: "%02x", $0) }.joined()
    }

    /// Saves the prefix entries (not the conversation ones), most used first, up to `maxEntries`.
    /// `fingerprint` identifies the model and binary: on load, if it does not match, the file is ignored.
    @discardableResult
    public func savePrefixCache(to url: URL, fingerprint: String, maxEntries: Int = 4) throws -> Int {
        flushPendingKV { $0.kind != "conversation" }
        let keep = entries.filter { $0.kind != "conversation" }.sorted { ($0.hits, $0.lastUse) > ($1.hits, $1.lastUse) }
            .prefix(maxEntries)
        guard !keep.isEmpty else { return 0 }
        let meta: [[String: Any]] = keep.map { e in
            [
                "kind": e.kind, "n": e.tokens.count, "sha": Self.sha(e.tokens), "hits": e.hits,
                "bufs": entryBuffers(e).map { $0.length },
            ]
        }
        let head = try JSONSerialization.data(withJSONObject: [
            "version": 1, "fingerprint": fingerprint, "entries": meta,
        ])
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        let tmp = url.appendingPathExtension("tmp")
        FileManager.default.createFile(atPath: tmp.path, contents: nil)
        let fh = try FileHandle(forWritingTo: tmp)
        try fh.write(contentsOf: head + Data([10]))
        for e in keep {
            try fh.write(contentsOf: e.tokens.withUnsafeBytes { Data($0) })
            for b in entryBuffers(e) {
                try fh.write(contentsOf: Data(bytesNoCopy: b.contents(), count: b.length, deallocator: .none))
            }
        }
        try fh.close()
        _ = try? FileManager.default.removeItem(at: url)
        try FileManager.default.moveItem(at: tmp, to: url)
        return keep.count
    }

    /// Loads the entries saved by `savePrefixCache` if the file is for this model and binary and the tokens match
    /// their hash. Returns how many were loaded.
    @discardableResult
    public func loadPrefixCache(from url: URL, fingerprint: String) throws -> Int {
        guard prefixCacheUsable, FileManager.default.fileExists(atPath: url.path) else { return 0 }
        let data = try Data(contentsOf: url, options: .alwaysMapped)
        guard let nl = data.firstIndex(of: 10),
            let head = try JSONSerialization.jsonObject(with: data[..<nl]) as? [String: Any],
            head["version"] as? Int == 1
        else { log?("prefix cache on disk: unknown format, ignored"); return 0 }
        guard head["fingerprint"] as? String == fingerprint else {
            log?("prefix cache on disk: other model or binary, ignored"); return 0
        }
        var off = nl + 1
        var loaded = 0
        for meta in head["entries"] as? [[String: Any]] ?? [] {
            guard let n = meta["n"] as? Int, let sha = meta["sha"] as? String, let lens = meta["bufs"] as? [Int] else {
                break
            }
            let total = n * 4 + lens.reduce(0, +)
            guard off + total <= data.count else { log?("prefix cache on disk: file too short"); break }
            let toks: [UInt32] = data[off..<(off + n * 4)].withUnsafeBytes { Array($0.bindMemory(to: UInt32.self)) }
            guard Self.sha(toks) == sha,
                let e = makeEntry(
                    P: n, kind: Self.diskKind(meta["kind"] as? String ?? "?")
                )
            else { off += total; continue }
            let bufs = entryBuffers(e)
            guard bufs.map({ $0.length }) == lens else {
                log?("prefix cache on disk: sizes differ, ignored"); return loaded
            }
            var o = off + n * 4
            for b in bufs {
                data[o..<(o + b.length)].withUnsafeBytes {
                    b.contents().copyMemory(from: $0.baseAddress!, byteCount: b.length)
                }
                o += b.length
            }
            e.snap.tokens = toks
            entries.append(e)
            loaded += 1
            off += total
        }
        return loaded
    }
}
