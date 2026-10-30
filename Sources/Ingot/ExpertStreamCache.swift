import Foundation
import Metal

// SSD expert streaming: cache policy of ExpertStream (ExpertStream.swift): next-layer read-ahead,
// ARC slots and the `smap`/`smapR` tables.

extension ExpertStream {
    /// Read-ahead of layer `n`: top-k of each row of the prediction (the router of n over the FFN input of the previous
    /// layer) that are missing, without waiting. Called with the GPU already running: it can only evict experts of
    /// other layers (the GPU is now computing layer n − 1, protected, and stops at n before using it).
    func runPrefetch(
        _ n: Int, rows rowsForPred: Int, protect p0: Set<Int>, pred pbIn: MTLBuffer? = nil, topK: Int? = nil
    ) {
        guard let pb = pbIn ?? pred[n] else { return }
        if wideK > 0, wide.active {
            runWide(n, rows: rowsForPred, pred: pb)
            return
        }
        let E = experts
        var protect = p0
        let cand = predTop(pb, rows: rowsForPred, k: topK ?? prefetchK)
        var pref: [(key: Int, slot: Int)] = []
        // The GPU is already heading to layer n and may be computing its residents: none of n is evicted.
        arc.exclude = (n * E)..<((n + 1) * E)
        // Nor are those with read-ahead in flight (see `handle`).
        let busy = pendingPrefetch.map { $0 != nil }
        if busy.contains(true) { arc.excludePred = { busy[$0 / E] } }
        defer {
            arc.exclude = 0..<0
            arc.excludePred = nil
        }
        for e in cand.sorted() where slotOf[n * E + e] < 0 {
            let k = n * E + e
            protect.insert(k)
            pref.append((k, allocate(k, protect: protect)))
            prefetchedKeys.insert(k)
        }
        // The layer may already have reads in progress (requested two layers earlier): they join the same group.
        launchAhead(n, pref)
        statsLock.lock()
        stats.prefetchReads += pref.count
        stats.bytesRead += pref.count * expertBytes
        statsLock.unlock()
    }

    /// Slot for key `k` (which is missing): a free one or the one ARC releases (never one in `protect`).
    func allocate(_ k: Int, protect: Set<Int>) -> Int {
        if let v = arc.insert(k, protect: protect) {
            // v leaves the cache: its slot goes to k
            let s = Int(slotOf[v])
            slotOf[v] = -1
            setResident(v, nil)
            prefetchedKeys.remove(v)
            ownerOf[s] = Int32(k)
            slotOf[k] = Int32(s)
            setMap(k, s)
            return s
        }
        let s = Int(freeSlots.removeLast())
        ownerOf[s] = Int32(k)
        slotOf[k] = Int32(s)
        setMap(k, s)
        return s
    }

    /// Publishes in `smapR` the experts of layer l's read-ahead (already read) that are still in the cache.
    func publishPending(_ l: Int) {
        for (k, s) in pendingKeys[l] where Int(slotOf[k]) == s { setResident(k, s) }
        pendingKeys[l] = []
    }

    func setResident(_ k: Int, _ s: Int?) {
        smapR.contents().storeBytes(of: s.map { UInt32($0) } ?? 0xffff_ffff, toByteOffset: k * 4, as: UInt32.self)
    }

    func setMap(_ k: Int, _ s: Int) {
        smap.contents().storeBytes(of: UInt32(s), toByteOffset: k * 4, as: UInt32.self)
    }
}
