import Foundation
import Metal

// Wide read-ahead into staging buffers, splash style (ExpertStore.cpp: reads ahead into its staging area and only
// "repacks" into the cache slot what the round routes). Exact: it only changes where the kernels read the same
// weights from.
//
// - Staging buffers: the slots of the prefill staging area (the ring, Z slots of the same arena), which nobody uses in
//   decode. When the first round arrives they become `free` and return to the ring before the next prefill.
// - Read-ahead: after answering layer l, the top-`wideK` of each row of the prediction for l + 1 that are not in
//   the cache go to a staging slot, in rank order (all the top-1, then the top-2…), without touching ARC: what the
//   prediction misses evicts no resident (which top-14 used to lose with plain read-ahead).
// - Priority for demand: the read-ahead readers (own threads) do not start a read while demand reads are in
//   progress; on reaching the layer, what was requested and is still queued is read as demand and what was not
//   requested and is still queued is cancelled.
// - Policy on use: the expert enters ARC as an on-demand miss, but without copying: its staging slot becomes the
//   cache slot and the slot ARC releases (or a free one) becomes a staging slot (swap). So the cache ends up the same as
//   if it had been read on demand, without waiting for the SSD (without this the cache does not learn and the round
//   took twice as long at 512).
// - The slot ARC releases is safe for the staging area for the same reason reusing it for a miss is safe today: it is
//   never in the current layer nor in one with a cache read in flight, and it is no longer in `smapR`; the GPU would
//   only read it again if the layer asks for it, and then it is a miss with a new slot.

/// Read-ahead queue with priority for demand (own threads; per-key states).
final class WideReader: @unchecked Sendable {
    enum State: UInt8 { case none = 0, queued, reading, done }
    private let cond = NSCondition()
    private var queue: [Int] = []
    private var head = 0
    private var state: [UInt8]
    private var slot: [Int32]
    private var demand = 0
    private let read: (Int, Int) -> Void

    init(keys: Int, threads n: Int, read: @escaping (Int, Int) -> Void) {
        state = [UInt8](repeating: 0, count: keys)
        slot = [Int32](repeating: -1, count: keys)
        self.read = read
        for i in 0..<n {
            let t = Thread { [unowned self] in self.loop() }
            t.qualityOfService = .userInteractive
            t.name = "expert.ahead\(i)"
            t.start()
        }
    }

    private func loop() {
        cond.lock()
        while true {
            while demand > 0 || head >= queue.count { cond.wait() }
            let k = queue[head]
            head += 1
            if head == queue.count {
                queue.removeAll(keepingCapacity: true)
                head = 0
            }
            guard state[k] == State.queued.rawValue else { continue }
            state[k] = State.reading.rawValue
            let s = Int(slot[k])
            cond.unlock()
            read(k, s)
            cond.lock()
            state[k] = State.done.rawValue
            cond.broadcast()
        }
    }

    func enqueue(_ list: [(key: Int, slot: Int)]) {
        guard !list.isEmpty else { return }
        cond.lock()
        for (k, s) in list {
            state[k] = State.queued.rawValue
            slot[k] = Int32(s)
            queue.append(k)
        }
        cond.broadcast()
        cond.unlock()
    }

    /// If `k` is still queued, removes it (it becomes `none`) and returns `queued`; otherwise its state.
    func take(_ k: Int) -> State {
        cond.lock()
        defer { cond.unlock() }
        let s = State(rawValue: state[k])!
        if s == .queued { state[k] = State.none.rawValue }
        return s
    }

    func status(_ k: Int) -> State {
        cond.lock()
        defer { cond.unlock() }
        return State(rawValue: state[k])!
    }

    /// Waits for the read of `k` (in progress) to finish and leaves it at `none`.
    func finish(_ k: Int) {
        cond.lock()
        while state[k] == State.reading.rawValue { cond.wait() }
        state[k] = State.none.rawValue
        cond.unlock()
    }

    func beginDemand() {
        cond.lock(); demand += 1; cond.unlock()
    }

    func endDemand() {
        cond.lock()
        demand -= 1
        cond.broadcast()
        cond.unlock()
    }
}

/// Wide read state (only touched by the CPU thread that serves the GPU).
final class WideState {
    var active = false
    /// Free staging slots.
    var free: [Int32] = []
    /// Staging slot of each key with read-ahead (−1 if none).
    var slotOf: [Int32]
    /// Keys with read-ahead requested for each layer.
    var layerKeys: [[Int]]
    /// Keys not requested by their layer whose read is still in progress (released when done).
    var drain: [Int] = []
    /// Keys read ahead that their layer did not request: they stay in their staging slot (oldest to newest) until the
    /// slot is needed; if a round requests them, they are used like anything else read ahead.
    var stash: [Int] = []
    let reader: WideReader

    init(keys: Int, layers: Int, reader: WideReader) {
        slotOf = [Int32](repeating: -1, count: keys)
        layerKeys = [[Int]](repeating: [], count: layers)
        self.reader = reader
    }
}

extension ExpertStream {
    /// Direct read of an expert record (16 KiB-aligned in the file and in the arena) into its staging slot.
    func wideReadDirect(key k: Int, slot s: Int) {
        if pieceBytes.count == 1, src[k / experts][0].off % 16384 == 0, pieceBytes[0] % 16384 == 0 {
            let l = k / experts, e = k % experts, len = pieceBytes[0]
            let (fd, toff) = src[l][0]
            let dst = arenas[0].contents() + s * len
            var done = 0
            while done < len {
                let r = pread(fd, dst + done, len - done, off_t(toff + e * len + done))
                if r <= 0 { break }
                done += r
            }
            precondition(done == len, "wide read: short SSD read")
            return
        }
        for i in 0..<pieceBytes.count { readPieceSync(key: k, slot: s, piece: i) }
    }

    func makeWide() -> WideState {
        let r = WideReader(keys: numLayers * experts, threads: 4) { [unowned self] k, s in
            wideReadDirect(key: k, slot: s)
        }
        return WideState(keys: numLayers * experts, layers: numLayers, reader: r)
    }

    /// Before a round: takes the prefill ring slots (if they are all free and have no loans).
    func wideEnsurePool() -> Bool {
        let w = wide
        if w.active { return true }
        let done = doneEvent.signaledValue
        while let f = ringUsed.first, f.value <= done {
            ringFree.append(contentsOf: f.slots)
            ringUsed.removeFirst()
        }
        guard ringUsed.isEmpty, borrowed.isEmpty else { return false }
        w.free = ringFree
        ringFree = []
        w.active = true
        return true
    }

    /// Before using the prefill ring: finishes or cancels all wide reads and returns the slots to it.
    func wideReturnPool() {
        guard wideK > 0, wide.active else { return }
        let w = wide
        for l in 0..<numLayers {
            for k in w.layerKeys[l] { wideDrop(k) }
            w.layerKeys[l] = []
        }
        for k in w.drain {
            w.reader.finish(k)
            w.free.append(w.slotOf[k])
            w.slotOf[k] = -1
        }
        w.drain = []
        for k in w.stash {
            w.free.append(w.slotOf[k])
            w.slotOf[k] = -1
        }
        w.stash = []
        ringFree += w.free
        w.free = []
        w.active = false
    }

    /// Releases the wide read of `k` (not requested): cancels if still queued, waits if in progress.
    private func wideDrop(_ k: Int) {
        let w = wide
        let s = w.slotOf[k]
        guard s >= 0 else { return }
        if w.reader.take(k) != .queued { w.reader.finish(k) }
        w.free.append(s)
        w.slotOf[k] = -1
    }

    /// Unrequested keys whose read has finished release their slot.
    func wideReap() {
        guard wideK > 0, wide.active, !wide.drain.isEmpty else { return }
        let w = wide
        w.drain.removeAll { k in
            guard w.reader.status(k) == .done else { return false }
            w.reader.finish(k)
            w.stash.append(k)
            return true
        }
    }

    /// Work of layer l in a round with wide read: separates from `needed` what has a staging slot, releases what was not
    /// requested and returns (queued keys to read now, keys already read or in progress).
    func wideSplit(_ l: Int, needed: [Int]) -> (claim: [(key: Int, slot: Int)], wait: [Int]) {
        let w = wide, E = experts
        var claim: [(key: Int, slot: Int)] = []
        var waitK: [Int] = []
        var want = Set<Int>()
        for e in needed {
            let k = l * E + e
            guard w.slotOf[k] >= 0 else { continue }
            if slotOf[k] >= 0 { continue }  // already resident (via another path): released below
            want.insert(k)
            if let i = w.drain.firstIndex(of: k) { w.drain.remove(at: i) }  // released earlier, read in progress
            if let i = w.stash.firstIndex(of: k) {
                w.stash.remove(at: i)
                statsLock.lock(); stats.wideStashHits += 1; statsLock.unlock()
            }
            if w.reader.take(k) == .queued { claim.append((k, Int(w.slotOf[k]))) } else { waitK.append(k) }
        }
        var cancelled = 0, wasted = 0
        for k in w.layerKeys[l] where !want.contains(k) && w.slotOf[k] >= 0 {
            switch w.reader.take(k) {
            case .queued:
                cancelled += 1
                w.free.append(w.slotOf[k])
                w.slotOf[k] = -1
            case .done:
                wasted += 1
                w.reader.finish(k)
                w.stash.append(k)
            default:
                wasted += 1
                w.drain.append(k)
            }
        }
        w.layerKeys[l] = []
        statsLock.lock()
        stats.wideCancelled += cancelled
        stats.bytesRead -= cancelled * expertBytes
        stats.wideWasted += wasted
        statsLock.unlock()
        return (claim, waitK)
    }

    /// After reading what was claimed and waiting for what was in progress: the layer uses those staging slots (they enter
    /// the cache by swapping slots).
    func wideUse(_ keys: [Int], protect: Set<Int>) {
        let w = wide
        for k in keys {
            let sp = Int(w.slotOf[k])
            w.slotOf[k] = -1
            let s = allocate(k, protect: protect)  // k enters ARC with slot s (free, or the one of the evicted entry)
            ownerOf[s] = -1
            ownerOf[sp] = Int32(k)
            slotOf[k] = Int32(sp)
            setMap(k, sp)
            setResident(k, sp)
            w.free.append(Int32(s))
        }
    }

    /// Wide read-ahead of layer `n`: top-`wideK` of each row, by rank, into staging slots.
    func runWide(_ n: Int, rows: Int, pred pb: MTLBuffer) {
        let w = wide, E = experts, ps = predStride, K = wideK
        guard w.active, rows > 0 else { return }
        let p = pb.contents().bindMemory(to: Float.self, capacity: MoELayer.maxRows * ps)
        // top-K of each row (highest to lowest) and then all by rank
        var tops: [[Int]] = []
        for r in 0..<rows {
            var bv: [Float] = [], be: [Int] = []
            for e in 0..<E {
                let v = p[r * ps + e]
                if bv.count == K && v <= bv[K - 1] { continue }
                var j = bv.count
                if j == K {
                    j = K - 1
                    bv.removeLast()
                    be.removeLast()
                }
                while j > 0 && bv[j - 1] < v { j -= 1 }
                bv.insert(v, at: j)
                be.insert(e, at: j)
            }
            tops.append(be)
        }
        var list: [(key: Int, slot: Int)] = []
        var short = 0
        for rank in 0..<K {
            for t in tops where rank < t.count {
                let k = n * E + t[rank]
                if slotOf[k] >= 0 { continue }
                if w.slotOf[k] >= 0 {
                    // already requested and released by its layer with the read in progress: back to the layer's list
                    if let i = w.drain.firstIndex(of: k) {
                        w.drain.remove(at: i)
                        w.layerKeys[n].append(k)
                    }
                    continue
                }
                if w.free.isEmpty, !w.stash.isEmpty {
                    let old = w.stash.removeFirst()
                    w.free.append(w.slotOf[old])
                    w.slotOf[old] = -1
                }
                guard let s = w.free.popLast() else {
                    short += 1
                    continue
                }
                w.slotOf[k] = s
                w.layerKeys[n].append(k)
                list.append((k, Int(s)))
            }
        }
        w.reader.enqueue(list)
        statsLock.lock()
        stats.wideIssued += list.count
        stats.wideShort += short
        stats.bytesRead += list.count * expertBytes
        statsLock.unlock()
    }
}
