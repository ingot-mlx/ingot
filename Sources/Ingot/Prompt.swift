import Foundation
import Metal

// Prompt preparation before decoding, shared by ingot-dev gen and the server.
//
// The GDN cannot be trimmed midway, so the prefix cache can only reuse state at points where a
// snapshot was taken. The usable ones are those that do not merge with what follows when the conversation is
// retokenized: (a) right after `<|im_start|>assistant\n` (the start of the generation prompt: without reasoning, the
// template strips the `<think>` block from the history) and (b) before the last prompt token (with reasoning, the
// `\n` of `<think>\n` merges with the `\n` of an empty reasoning into a single `\n\n` token). So the prompt is read
// as follows: normal prefill up to (a); snapshot (a); the k ≤ 16 remaining tokens, in one go through the
// verification path (exact, and cheap with few rows: it only touches the experts of those rows), with the GDN tape
// unfolded; snapshot (b), folding k − 1 tape rows directly into their buffers. The first decode round
// folds the k rows.
//
// When continuing a conversation, what is new up to (a) is usually short (a message or a tool result): if it does not
// exceed `shortPassesMax` batches it is also read through the verification path, in batches of ≤ 16 rows (each
// distinct expert is read once; a ~20-row prefill touches almost all the MoE experts). The last batch is merged with
// the final stretch if both fit in 16 rows, and then snapshot (a) is also taken by folding tape rows. Not from a start
// aligned to 64 (a prefix-cache entry): there it goes through prefill, as without cache (`promptPlan`).

extension HybridModel {
    /// Snapshot point during preparation: `.current` = the current state (P tokens, no rows left to fold);
    /// `.tape(rows)` = the current state plus `rows` rows of the GDN tape (position P − k + rows).
    /// `.captured(slot)` = the state captured inside the pass in `captureTargets[slot]` (the engine already has the
    /// GDN at that snapshot; it only copies the drafter ring).
    public enum SnapshotPoint { case current, tape(rows: Int), captured(slot: Int) }
    /// Captures for the prefix-cache cuts inside the pass go in `captureTargets[2...]` (the first
    /// two are for points (a) and (b)).
    public static let cutSlotBase = 2
    /// Flash-Next: rows from which a span with prefix-cache cuts (more than one chunk per group, so it fetches all the
    /// experts) is read in a single pass with the cuts captured inside it. It was `flashBulkMinRows` (512) until it
    /// raised that one; kept at 512 so the cuts do not change (this threshold is tuned separately).
    static let cutGroupMinRows = 512

    /// Reads `ids[from...]` on top of the current state (which already has `from` tokens in the cache, no rows to fold;
    /// with from = 0 it starts from scratch). `genStart`: position after `<|im_start|>assistant\n` (else only (b)).
    /// Returns the argmax of the last row (its logits stay in row `lastRow` of `logits`) and leaves the last k
    /// rows on the tape (nacc = k) and their drafter context written. `snapshot` is called at points (a) and (b) with
    /// the number of tokens it covers.
    /// `cuts`: intermediate points (sorted) where to split the prefill and call `snapshot(.current, point)` (prefix
    /// cache across sessions, PrefixCache.swift); only used if the span goes through prefill.
    /// `rowCuts`: cuts from `cuts` that do not split the prefill (Flash-Next: the conversation cut; the other
    /// families: every cut whose entry the engine reserved). They are captured as one more row of the last chunk of
    /// the pass (the state, windows and drafter ring, as if the pass ended there); if they do not fall in it,
    /// Flash-Next leaves them uncaptured (`snapshot` is not called for them) and the other families split the prefill
    /// there (also those beyond the captures a pass can take).
    public func preparePrompt(
        _ ids: [UInt32], from c: Int, genStart: Int?, drafter dr: DFlash2Drafter?, cuts: [Int] = [],
        rowCuts: Set<Int> = [], snapshot: ((SnapshotPoint, Int) throws -> Void)? = nil
    ) throws -> (token: UInt32, lastRow: Int, seconds: Double) {
        let caps = prefillCaps
        precondition(rowCuts.isEmpty || caps.rowCuts)
        let n = ids.count
        let allCuts = cuts
        var cuts = cuts.filter { !rowCuts.contains($0) }
        // KVPool: KV for the whole prompt (Engine already requested it with margin; this covers the benches).
        guard try ensureContext(n + DFlash2Drafter.margin(dr)) else {
            throw SafetensorsError.io("KVPool: the \(n)-token prompt does not fit in the context")
        }
        let t0 = DispatchTime.now().uptimeNanoseconds
        let trace0 = options.bench.prepTrace  // --prep-trace: ms per step
        var trace: [String] = []
        var tl = t0
        func mark(_ s: String) {
            guard trace0 else { return }; let t = DispatchTime.now().uptimeNanoseconds
            trace.append(String(format: "%@ %.1f", s, Double(t - tl) / 1e6)); tl = t
        }
        let snapshot: ((SnapshotPoint, Int) throws -> Void)? = snapshot.map { f in
            { p, l in
                mark("·"); try f(p, l); mark("snap\(l)")
            }
        }
        defer { if trace0 { print("prep n=\(n) c=\(c): " + trace.joined(separator: ", ")) } }
        if c == 0 { reset() }
        precondition(c < n && nacc == 0 && P == c)
        flashSetHistory(ids, before: c)
        setDecodeChunk(promptTokens: n)
        let plan = promptPlan(count: n, from: c, genStart: genStart)
        let s = plan.s, head = plan.head
        // If the span goes through prefill, a single pass to the end of the prompt, with points (a) and (b)
        // captured inside it (GDN state after those rows, Layers/GDN.swift); before, the final stretch went in
        // another pass through the verification path (~16 ms) and (a) split the prefill.
        if plan.batches == nil && s > c && canCaptureInPass {
            var pts: [Int] = []
            if let g = genStart, s == g { pts.append(s) }
            if !pts.contains(n - 1) { pts.append(n - 1) }
            let targets = snapshot == nil ? [] : (captureTargets ?? [])
            // With more than one chunk after the last cut, normal prefill up to a 64-aligned cut
            // (bit-exact with the whole) and the last chunk with the captures; the whole prompt always goes through
            // prefill, with or without prefix cache or int8 (the verification path sums in another order and
            // the last rows gave different logits depending on where the read started: prefixcache step 4 with int8,
            // and without int8 on the dense toy). Before, only Flash-Next and lossy prefill.
            // Balanced chunks (k chunks of `per` rows aligned to 64 and the last with the remainder), as
            // `prefill` splits them; before, the last was ~chunkRows rows and the rest went first (with 3K, 1024 + 2032
            // instead of 1536 + 1520: +0.17 s).
            func layout(_ cuts: [Int]) -> (a0: Int, x: Int, multi: Bool) {
                let a0 = cuts.filter { $0 > c && $0 < s }.max() ?? c
                let L = n - a0, k = (L + chunkRows - 1) / chunkRows
                let per = ((L + k - 1) / k + 63) / 64 * 64
                let x =
                    k > 1 && (k - 1) * per < L - 64 ? a0 + (k - 1) * per : a0 + max(0, (L - chunkRows + 63) / 64 * 64)
                return (a0, x, n - a0 > chunkRows && pts.allSatisfy { $0 - 1 - x >= 0 })
            }
            var (a0, x, multi) = layout(cuts)
            // Outside Flash-Next, a row cut that falls before the last chunk splits the prefill, as the other cuts
            // (`saveEntry` fills the entry the engine reserved for it), and so do the lowest ones of the last chunk
            // beyond the captures left after (a) and (b); Flash-Next leaves them uncaptured. Each split moves the
            // last chunk (`splitRowCuts`).
            if !caps.cutsInPass {
                cuts = Self.splitRowCuts(rowCuts, cuts: cuts, span: c..<s, free: GDNChunk.maxCaptures - pts.count) {
                    let l = layout($0)
                    return l.multi ? l.x : l.a0
                }
                (a0, x, multi) = layout(cuts)
            }
            let inPass = n - a0 <= chunkRows || multi
            // The prefix-cache cuts (Flash-Next) in the same call, captured at the end of their chunk
            // in the captures that follow those of `pts` (before, another call and another SSD pass for each).
            // Cut captures go in `captureTargets[cutSlotBase...]`, in the order of `allCuts` (that of the
            // entries reserved by the engine).
            let slotOf = Dictionary(
                uniqueKeysWithValues: allCuts.filter { $0 > c && $0 < s }.enumerated().map {
                    ($1, Self.cutSlotBase + $0)
                })
            let inCuts = cuts.filter { $0 > c && $0 < s }
            // The row cuts of the last chunk (starts at `x` or at the last cut); Flash-Next, only the conversation
            // cut.
            let last0r = multi ? x : a0
            let rowAll = rowCuts.filter { $0 > c && $0 < s && $0 - 1 - last0r >= 0 }.sorted()
            let rowIn = Array(caps.cutsInPass ? rowAll.suffix(1) : rowAll[...])
            let cutInPass =
                caps.cutsInPass && !(inCuts.isEmpty && rowIn.isEmpty) && snapshot != nil
                && pts.count <= Self.cutSlotBase
                && pts.count + rowIn.count <= GDNChunk.maxCaptures
                && targets.count >= Self.cutSlotBase + slotOf.count
                // Several chunks in a group only pay off with enough rows to fetch everything (`cutGroupMinRows`);
                // a captured row adds no chunks.
                && (inCuts.isEmpty || n - c >= Self.cutGroupMinRows)
                && (captureDraftTargets == nil) == (dr == nil)
            if inPass && (snapshot == nil || targets.count >= pts.count) && cutInPass {
                let split = multi ? x - c : nil
                let last0 = split.map { c + $0 } ?? a0
                let rows = (pts + rowIn).map { $0 - 1 - last0 }
                let slots = Array(0..<pts.count) + rowIn.map { slotOf[$0]! }
                let (tok, _) = try prefillCapturing(
                    Array(ids[c..<n]), drafter: dr, continuing: c > 0, rows: rows, targets: slots.map { targets[$0] },
                    split: split, cuts: inCuts.map { $0 - c }, cutSlots: inCuts.map { slotOf[$0]! }, slots: slots,
                    rings: Array(pts.count..<(pts.count + rowIn.count)))
                setDecodeChunk(promptTokens: n)
                mark("pass\(n - c)")
                for cut in inCuts + rowIn { try snapshot?(.captured(slot: slotOf[cut]!), cut) }
                for (k, pt) in pts.enumerated() { try snapshot?(.captured(slot: k), pt) }
                return (tok, 0, Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9)
            }
            if inPass && (snapshot == nil || targets.count >= pts.count) {
                var a = c
                for cut in cuts where cut > a && cut < s {
                    _ = try prefill(Array(ids[a..<cut]), drafter: dr, continuing: a > 0, head: false)
                    try snapshot?(.current, cut)
                    a = cut
                }
                // The chunks up to x and the last one, in the same call (with `flashPrefillGroup` chunks per group,
                // a single pass over the experts).
                let split = multi ? x - a : nil
                let last0 = a + (split ?? 0)
                // Outside Flash-Next, the cuts of the last chunk as more rows of it, into the entries the engine
                // reserved for them (`captureTargets[cutSlotBase...]`, in the order of the sorted row cuts):
                // splitting the prefill there added a short chunk, which reads almost all the MoE experts (for
                // the conversation cut, and for the system one: −133 ms on pi's first turn).
                let rowSlots = Dictionary(
                    uniqueKeysWithValues: rowCuts.sorted().enumerated().map { ($1, Self.cutSlotBase + $0) })
                let rowHere =
                    caps.cutsInPass || snapshot == nil || pts.count > Self.cutSlotBase
                        || pts.count + rowIn.count > GDNChunk.maxCaptures || (captureDraftTargets == nil) != (dr == nil)
                    ? [] : rowIn.filter { rowSlots[$0]! < targets.count }
                let slots = snapshot == nil ? [] : Array(0..<pts.count) + rowHere.map { rowSlots[$0]! }
                let rows = snapshot == nil ? [] : (pts + rowHere).map { $0 - 1 - last0 }
                let (tok, _) = try prefillCapturing(
                    Array(ids[a..<n]), drafter: dr, continuing: a > 0, rows: rows,
                    targets: slots.map { targets[$0] }, split: split, slots: slots,
                    rings: Array(pts.count..<(pts.count + rowHere.count)))
                setDecodeChunk(promptTokens: n)
                mark("pass\(n - a)")
                for cut in rowHere { try snapshot?(.captured(slot: rowSlots[cut]!), cut) }
                if snapshot != nil { for (k, pt) in pts.enumerated() { try snapshot?(.captured(slot: k), pt) } }
                return (tok, 0, Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9)
            }
        }
        // Not read in the capturing pass: outside Flash-Next the row cuts split the prefill again (as before).
        if !caps.cutsInPass { cuts = allCuts }
        if s > c {
            if let batches = plan.batches {
                // Short chunk (e.g. the new message when continuing a conversation): through the verification path in
                // batches of ≤ 16 rows, which read each distinct expert once; prefill touches almost all the experts.
                for r in batches {
                    try stepRows(Array(ids[r]), drafter: dr)
                    try foldPending(drafter: nil, context: nil)
                    mark("batch\(r.count)")
                }
            } else {
                var a = c
                for cut in cuts where cut > a && cut < s {
                    _ = try prefill(Array(ids[a..<cut]), drafter: dr, continuing: a > 0, head: false)
                    try snapshot?(.current, cut)
                    a = cut
                }
                _ = try prefill(Array(ids[a..<s]), drafter: dr, continuing: a > 0, head: false)
                setDecodeChunk(promptTokens: n)
                mark("prefill\(s - a)")
            }
            if head == 0 && s == genStart { try snapshot?(.current, s) }
        }
        // Final stretch through the verification path.
        let tail = Array(ids[(s - head)..<n])
        try stepRows(tail, drafter: dr)
        mark("tail\(tail.count)")
        // (a) if its span went in the same batch: the state plus the first `head` rows of the tape.
        if head > 0 && s == genStart { try snapshot?(.tape(rows: head), s) }
        // (b): with a one-token stretch it is the state from before the stretch (if not already taken at (a)).
        if tail.count > 1 && !(head == tail.count - 1 && s == genStart) {
            try snapshot?(.tape(rows: tail.count - 1), n - 1)
        } else if s > c && s != genStart {
            try snapshot?(.tape(rows: 0), n - 1)
        }
        let row = tail.count - 1
        let tok = top1.contents().bindMemory(to: UInt32.self, capacity: AttentionLayer.maxRows)[row]
        return (tok, row, Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9)
    }

    /// Outside Flash-Next, the cuts the prefill splits at: `cuts` plus the row cuts (strictly inside `span`) that fall
    /// before the last chunk, which starts at `lastStart(cuts)`, and the lowest of those in it beyond `free` captures.
    /// Each split moves the last chunk, so it is repeated until none is left (every round adds cuts).
    static func splitRowCuts(
        _ rowCuts: Set<Int>, cuts: [Int], span: Range<Int>, free: Int, lastStart: ([Int]) -> Int
    ) -> [Int] {
        var cuts = cuts
        while true {
            let last0 = lastStart(cuts)
            let inSpan = rowCuts.filter { $0 > span.lowerBound && $0 < span.upperBound }
            let missed = inSpan.filter { $0 - 1 - last0 < 0 && !cuts.contains($0) }
            let over = inSpan.filter { $0 - 1 - last0 >= 0 }.sorted().dropLast(max(0, free))
            if missed.isEmpty && over.isEmpty { return cuts }
            cuts = (cuts + missed + over).sorted()
        }
    }

    /// How `preparePrompt` reads a prompt of `n` tokens with `from` = c cached: `s` = start of the final stretch (point
    /// (a) if it falls past what is cached and leaves ≤ 16 tokens; else n − 1); `batches` = the ≤ 16-row batches
    /// (folded one by one) of short chunk c..<s − head, or nil if c..<s uses prefill; `head` = rows at the end of
    /// c..<s that go in the same batch as the final stretch. Also used by ingot-dev reuse to build the snapshot reference
    /// through the same path.
    /// Whether `preparePrompt` will read c..<n in a single pass with captures (without prefix-cache cuts). Used by
    /// ingot-dev reuse to build the snapshot reference through the same path: one pass that ends at the cut.
    public func promptUsesCapture(count n: Int, from c: Int, genStart: Int?) -> Bool {
        let plan = promptPlan(count: n, from: c, genStart: genStart)
        return plan.batches == nil && plan.s > c && canCaptureInPass
    }

    public struct PromptPlan { public let s: Int; public let batches: [Range<Int>]?; public let head: Int }

    /// Alignment of the starts from which reading by prefill gives the same as without cache (the chunked GDN groups
    /// by 64; the prefix-cache entries are cut at multiples of it).
    static let exactCutAlign = 64

    public func promptPlan(count n: Int, from c: Int, genStart: Int?) -> PromptPlan {
        let R = AttentionLayer.maxRows
        var s = n - 1
        if let g = genStart, g > c, g < n, n - g <= R { s = g }
        if s < c { s = c }
        guard s > c else { return PromptPlan(s: s, batches: nil, head: 0) }
        // From a start aligned to 64 past 0 (a prefix-cache entry, or a snapshot at such a point) the span goes
        // through prefill, as without cache: the prefill split at an aligned point is bit-identical to the whole one,
        // while the verification path sums in another order (the real Muse, 32 rows past its entry at 4672
        // through two batches, gave last-position logits max |Δ| 0.162 against without cache).
        guard c == 0 || c % Self.exactCutAlign != 0 else { return PromptPlan(s: s, batches: nil, head: 0) }
        let len = s - c
        let rem = len % R, merges = rem > 0 && rem + (n - s) <= R
        guard (len + R - 1) / R + (merges ? 0 : 1) <= shortPassesMax else {
            return PromptPlan(s: s, batches: nil, head: 0)
        }
        var batches: [Range<Int>] = [], head = 0, i = c
        while i < s {
            let k = min(R, s - i)
            if i + k == s && k + (n - s) <= R { head = k; break }
            batches.append(i..<(i + k))
            i += k
        }
        return PromptPlan(s: s, batches: batches, head: head)
    }

    /// `ids` (≤ 16) at positions P... in one go through the verification path: logits and argmax of each row,
    /// drafter context written and the rows on the GDN tape (nacc = ids.count; folded by the next round
    /// or `foldPending`).
    public func stepRows(_ ids: [UInt32], drafter dr: DFlash2Drafter?) throws {
        let k = ids.count
        precondition(k >= 1 && k <= AttentionLayer.maxRows && nacc == 0 && P + k <= maxCtx)
        let tp = tok.contents().bindMemory(to: UInt32.self, capacity: AttentionLayer.maxRows)
        for (i, t) in ids.enumerated() { tp[i] = t }
        let p0 = P
        guard let cb = ctx.queue.makeCommandBuffer(), let e0 = cb.makeComputeCommandEncoder() else {
            throw SafetensorsError.io("cb")
        }
        let e = try encodeVerify(e0, rows: k, drafter: dr, cb: cb)
        if let dr { try dr.encodeContext(e, rows: k, nvalid: k, pos0: p0) }
        e.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if let err = cb.error { throw err }
        commit(accepted: k)
    }
}
