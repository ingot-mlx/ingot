import Foundation
import Metal

// Flash-Next, lossy: off in the engine and in ingot-dev, rank 5 in ingot serve (EngineOptions.serve), where the speed it
// gains is worth its small loss. In rounds, experts missing from the cache that do not go past a given rank in
// any row are not waited for (weight 0 and a zeroed slot) and are read in the background for the following rounds.
extension ExpertStream {
    func fetchSkipped(protect p0: Set<Int>) {
        guard !skipBacklog.isEmpty else { return }
        var protect = p0
        for (lb, keys) in skipBacklog {
            var pref: [(key: Int, slot: Int)] = []
            for k in keys where slotOf[k] < 0 {
                protect.insert(k)
                pref.append((k, allocate(k, protect: protect)))
            }
            guard !pref.isEmpty else { continue }
            launchAhead(lb, pref)
            statsLock.lock()
            stats.skipFetched += pref.count
            stats.bytesRead += pref.count * expertBytes
            statsLock.unlock()
        }
        skipBacklog.removeAll(keepingCapacity: true)
    }

    /// Drops from `needed` the layer-l experts that are missing (after read-ahead) and whose best rank across the
    /// `rows` rows is ≥ skipRank; their slot becomes the zero slot and their weights 0 (the rest are not renormalized:
    /// lower KL). The GPU is stopped at the barrier and the combine comes later: it may change `wts`. With
    /// resident-first compute, skipped experts are not in `smapR`: they go to the tiles of the missing, with `smap`.
    func applySkip(layer l: Int, needed: [Int], inds: MTLBuffer, wts: MTLBuffer, rows r: Int) -> [Int] {
        guard let sr = skipRank, r > 0 else { return needed }
        let E = experts, K = topk
        let ip = inds.contents().bindMemory(to: UInt32.self, capacity: r * K)
        let wp = wts.contents().bindMemory(to: Float.self, capacity: r * K)
        var best = [Int: Int]()
        for i in 0..<(r * K) {
            let e = Int(ip[i])
            if slotOf[l * E + e] < 0 && !(wideK > 0 && wide.slotOf[l * E + e] >= 0) {
                best[e] = min(best[e] ?? 99, i % K)
            }
        }
        let skip = Set(best.filter { $0.value >= sr }.map(\.key))
        guard !skip.isEmpty else { return needed }
        if zeroSlot < 0 { zeroSlot = reserveZeroSlot() }
        var pairs = 0
        for i in 0..<(r * K) where skip.contains(Int(ip[i])) {
            wp[i] = 0
            pairs += 1
        }
        for e in skip { setMap(l * E + e, Int(zeroSlot)) }
        skipBacklog.append((l, skip.map { l * E + $0 }))
        statsLock.lock()
        stats.skipped += skip.count
        stats.skippedPairs += pairs
        statsLock.unlock()
        return needed.filter { !skip.contains($0) }
    }

    /// A cache slot that stays zeroed (leaves ARC for good).
    func reserveZeroSlot() -> Int32 {
        var s: Int32
        if let f = freeSlots.popLast() {
            s = f
        } else {
            guard let v = arc.lru(1, protect: []) ?? arc.lru(2, protect: []) else {
                preconditionFailure("skip: no free slot")
            }
            arc.unlink(v)
            s = slotOf[v]
            slotOf[v] = -1
            setResident(v, nil)
            prefetchedKeys.remove(v)
            ownerOf[Int(s)] = -1
        }
        arc.c -= 1
        for (i, b) in pieceBytes.enumerated() { memset(arenas[i].contents() + Int(s) * b, 0, b) }
        return s
    }

}
