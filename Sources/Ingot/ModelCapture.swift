import Foundation
import Metal

// Checkpoint of the GDN inside the prefill pass (captures of the state after some rows) and logits of
// the prompt rows.

extension HybridModel {
    /// In-pass checkpoint, on by default in every model that supports it (`canCaptureInPass`;
    /// not LFM2). On the dense 27B it is exact (ingot-dev reuse, prefixcache, gen --check, also with the tree) and
    /// saves ~100 ms per prefilled turn (the separate pass for the tail cost ~110 ms: 32 new tokens
    /// over 3K, 338 → 240 ms; 300 tokens, 1099 → 1000 ms). Without it (LFM2), the tail goes through the
    /// verification path.
    /// Capturing in the pass works with chunked GDN and no short-convolution layers (LFM2 does not implement it).
    public var canCaptureInPass: Bool {
        gdnChunk != nil && options.bench.gdnChunk
            && !layers.contains {
                if case .conv = $0.mixer { return true }; return false
            }
    }

    /// Prefill of `ids` (≤ chunkRows, a single chunk) over the current state with captures of the state after rows
    /// `rows` in `targets` (and logits of the last row, like `prefill`).
    /// `slots` = the capture of `captureTargets` for each row (by default, the first ones); `rings` = those of
    /// these rows (indices into `rows`) whose drafter ring is also captured (the prefix-cache cuts: rows far from
    /// the end, which the following ring rows would overwrite).
    public func prefillCapturing(
        _ ids: [UInt32], drafter: DFlash2Drafter?, continuing: Bool, rows: [Int],
        targets: [[(state: MTLBuffer, hist: MTLBuffer)]], split: Int? = nil, cuts: [Int] = [], cutSlots: [Int] = [],
        slots: [Int]? = nil, rings: [Int] = []
    ) throws -> (token: UInt32, seconds: Double) {
        precondition(ids.count - (split ?? cuts.last ?? 0) <= chunkRows && rows.count == targets.count)
        precondition(slots.map { $0.count == rows.count } ?? true)
        if rows.isEmpty { pendingCapture = nil } else { pendingCapture = (rows: rows, targets: targets) }
        let sl = slots ?? Array(0..<rows.count)
        let caps = prefillCaps
        pendingConvCapture = caps.convCaptures && !rows.isEmpty ? captureConvTargets.map { c in sl.map { c[$0] } } : nil
        precondition(!caps.convCaptures || rows.isEmpty || pendingConvCapture?.count == rows.count)
        var drafterEnds: [Int] = []
        if !rings.isEmpty, drafter != nil, let dt = captureDraftTargets {
            precondition(caps.rowCuts)
            let last0 = split ?? cuts.last ?? 0
            pendingRingCapture = rings.map { (rows: rows[$0] + 1, targets: dt[sl[$0]]) }
            drafterEnds = rings.map { last0 + rows[$0] + 1 }
        }
        defer { pendingCapture = nil; pendingConvCapture = nil; pendingRingCapture = [] }
        return try prefill(
            ids, drafter: drafter, continuing: continuing, head: true, split: split, cuts: cuts, cutSlots: cutSlots,
            drafterEnds: drafterEnds)
    }

}
