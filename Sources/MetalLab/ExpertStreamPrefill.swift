import Foundation
import Metal

// SSD expert streaming, prefill: ExpertStream's prefill (ExpertStream.swift): layers fetched ahead of the GPU into the
// pass-through area (a ring) and cache slots lent to the ring for the duration of the prefill.

extension ExpertStream {
    /// Prefill ahead of the GPU: all the layer's experts that are not in the cache, into the ring.
    func handleBulk(_ job: Job) {
        let l = job.layer, E = experts
        if let g = pendingPrefetch[l] {
            g.wait()
            pendingPrefetch[l] = nil
        }
        publishPending(l)
        wideReturnPool()
        let t0 = DispatchTime.now().uptimeNanoseconds
        // The same layer of the previous group must have finished before its slot table is changed.
        waitDone(lastPfValue[l])
        lastPfValue[l] = job.pf
        let keys = (0..<E).map { l * E + $0 }.filter { slotOf[$0] < 0 }
        let ring = ringAlloc(keys.count, value: job.pf)
        let t1 = DispatchTime.now().uptimeNanoseconds
        var miss: [(key: Int, slot: Int)] = []
        miss.reserveCapacity(keys.count)
        for (k, s) in zip(keys, ring) {
            miss.append((k, Int(s)))
            setMap(k, Int(s))
        }
        statsLock.lock()
        stats.bulkJobs += 1
        stats.prefillReads += miss.count
        stats.bulkRingWaitNs += t1 - t0
        statsLock.unlock()
        ensureBigStaging()
        let v = job.pf, list = miss
        pf.register(v)
        bulkQueue.async { [self] in
            if !list.isEmpty { readNow(list) }
            let t2 = DispatchTime.now().uptimeNanoseconds
            statsLock.lock()
            stats.bulkReadNs += t2 - t1
            statsLock.unlock()
            pf.complete(v)
        }
    }

    /// Waits (spinning with pauses) until the GPU signals `done` ≥ v.
    func waitDone(_ v: UInt64) {
        while doneEvent.signaledValue < v { usleep(20) }
    }

    /// n ring slots for prefill job `v` (waits for the GPU to release those of earlier layers).
    func ringAlloc(_ n: Int, value v: UInt64) -> [Int32] {
        guard n > 0 else { return [] }
        while true {
            let done = doneEvent.signaledValue
            while let f = ringUsed.first, f.value <= done {
                ringFree.append(contentsOf: f.slots)
                ringUsed.removeFirst()
            }
            if ringFree.count >= n { break }
            // The ring grows by borrowing cache slots (the least used) for the duration of the prefill.
            if borrowed.count < maxBorrow, let b = borrowSlot() {
                borrowed.append(b)
                ringFree.append(b)
                continue
            }
            precondition(!ringUsed.isEmpty, "SSD expert streaming: prefill ring too small (\(ringFree.count) < \(n))")
            usleep(20)
        }
        let s = Array(ringFree.suffix(n))
        ringFree.removeLast(n)
        ringUsed.append((v, s))
        return s
    }

    /// A cache slot for the ring: evicts the least-used resident expert (without a ghost) of a layer the GPU can no
    /// longer be using (its last prefill job finished; later jobs will see it as non-resident) and lowers the ARC
    /// capacity.
    func borrowSlot() -> Int32? {
        let done = doneEvent.signaledValue, E = experts
        let busy = pendingPrefetch.map { $0 != nil }
        arc.excludePred = { [lastPfValue] k in lastPfValue[k / E] > done || busy[k / E] }
        defer { arc.excludePred = nil }
        if let s = freeSlots.popLast() {
            arc.c -= 1
            return s
        }
        guard let v = arc.lru(1, protect: []) ?? arc.lru(2, protect: []) else { return nil }
        arc.unlink(v)
        arc.c -= 1
        let sl = slotOf[v]
        slotOf[v] = -1
        setResident(v, nil)
        prefetchedKeys.remove(v)
        ownerOf[Int(sl)] = -1
        return sl
    }

    /// Returns the lent slots to the cache (with the ring already free: the GPU finished the prefill).
    func returnBorrowed() {
        guard !borrowed.isEmpty else { return }
        let done = doneEvent.signaledValue
        while let f = ringUsed.first, f.value <= done {
            ringFree.append(contentsOf: f.slots)
            ringUsed.removeFirst()
        }
        guard ringUsed.isEmpty else { return }
        let set = Set(borrowed)
        ringFree.removeAll { set.contains($0) }
        freeSlots.append(contentsOf: borrowed)
        arc.c += borrowed.count
        borrowed = []
    }
}
