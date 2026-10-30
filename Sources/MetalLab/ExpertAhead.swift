import Foundation
import Metal

// Flash-Next decode: less SSD wait in the rounds. Exact: it only changes what is read and when; the kernels read the
// same weights.
// - Partial wait: on reaching layer l, the CPU waits only for the read-ahead experts the round requests (before, for
//   the whole read-ahead of the layer, including those the prediction missed). The rest stay in flight and their layer
//   stays busy for eviction (as before, nobody is evicted with a read in flight) until they finish.
extension ExpertStream {
    /// Launches the read-ahead of `pref` (experts of layer n): one group per expert in addition to the layer's.
    func launchAhead(_ n: Int, _ pref: [(key: Int, slot: Int)]) {
        guard !pref.isEmpty else { return }
        let g = pendingPrefetch[n] ?? DispatchGroup()
        for (k, s) in pref {
            let kg = DispatchGroup()
            for i in 0..<pieceBytes.count {
                g.enter()
                kg.enter()
                readQueue.async { [self] in
                    readPieceSync(key: k, slot: s, piece: i)
                    kg.leave()
                    g.leave()
                }
            }
            aheadGroup[k] = kg
        }
        pendingPrefetch[n] = g
        pendingKeys[n] += pref
    }

    /// The read-ahead of layer l must have arrived: all of it (prefill, or without partial wait) or only that of the
    /// experts in `needed` (round).
    func waitAhead(_ l: Int, needed: [Int], partial: Bool) {
        guard let g = pendingPrefetch[l] else { return }
        if partial {
            let E = experts
            for e in needed { aheadGroup[l * E + e]?.wait() }
            if g.wait(timeout: .now()) != .success {
                var rest: [(key: Int, slot: Int)] = []
                var left = 0
                for (k, s) in pendingKeys[l] {
                    if let kg = aheadGroup[k], kg.wait(timeout: .now()) != .success {
                        rest.append((k, s))
                        left += 1
                        continue
                    }
                    aheadGroup[k] = nil
                    if Int(slotOf[k]) == s { setResident(k, s) }
                }
                pendingKeys[l] = rest
                statsLock.lock()
                stats.aheadLeft += left
                statsLock.unlock()
                return
            }
        } else {
            g.wait()
        }
        finishAhead(l)
    }

    /// Layer l with all its read-ahead finished: it stops being busy and its experts are published.
    private func finishAhead(_ l: Int) {
        pendingPrefetch[l] = nil
        for (k, _) in pendingKeys[l] { aheadGroup[k] = nil }
        publishPending(l)
    }

    /// Layers whose read-ahead has already finished (with the partial wait, some may be left over from earlier rounds).
    func reapAhead() {
        for l in 0..<numLayers {
            if let g = pendingPrefetch[l], g.wait(timeout: .now()) == .success { finishAhead(l) }
        }
    }

    /// The top-k of each of the `rows` rows of a prediction (logits [rows, predStride] fp32).
    func predTop(_ pb: MTLBuffer, rows: Int, k K: Int) -> Set<Int> {
        let E = experts, ps = predStride
        let p = pb.contents().bindMemory(to: Float.self, capacity: MoELayer.maxRows * ps)
        var cand = Set<Int>()
        guard K > 0 else { return cand }
        var bv = [Float](repeating: 0, count: K), be = [Int](repeating: 0, count: K)
        for r in 0..<rows {
            // partial top-k: insertion into a sorted list of K
            var cnt = 0
            for e in 0..<E {
                let v = p[r * ps + e]
                if cnt == K && v <= bv[K - 1] { continue }
                var j = cnt == K ? K - 1 : cnt
                while j > 0 && bv[j - 1] < v {
                    bv[j] = bv[j - 1]
                    be[j] = be[j - 1]
                    j -= 1
                }
                bv[j] = v
                be[j] = e
                if cnt < K { cnt += 1 }
            }
            for j in 0..<cnt { cand.insert(be[j]) }
        }
        return cand
    }
}
