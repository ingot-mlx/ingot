import Foundation
import Metal

// SSD expert streaming: the CPU thread of ExpertStream (ExpertStream.swift). It serves the job queue and, at each layer
// with the GPU stalled, fetches what is missing (decode into the ARC cache; prefill without read-ahead into staging).

extension ExpertStream {
    func serve() {
        while true {
            // Preload advances in batches while there is no other job (and none in the last 50 ms:
            // within a request, the gaps between layers are shorter).
            if preloadPos < preloadKeys.count {
                if preloadIdle() {
                    preloadStep()
                    continue
                }
                if jobSem.wait(timeout: .now() + .milliseconds(5)) == .timedOut { continue }
            } else {
                jobSem.wait()
            }
            lastJobNs = DispatchTime.now().uptimeNanoseconds
            jobLock.lock()
            let job = jobs.removeFirst()
            jobLock.unlock()
            if case .preload(let keys) = job.kind {
                startPreload(keys)
                continue
            }
            if case .sync(let work) = job.kind {
                work()
                continue
            }
            // Layers with read-ahead from earlier, already finished rounds stop being busy.
            reapAhead()
            wideReap()
            if case .bulk = job.kind {
                // The read runs in the background and `ready` is signaled in order on completion (PfTracker); at most one layer
                // ahead (with two in flight, 2% less prefill throughput in decode: kept at one).
                pf.waitPending(atMost: 0)
                handleBulk(job)
                pf.waitPending(atMost: 0)
                continue
            }
            if case .custom(let work, true) = job.kind {
                pf.register(job.pf)
                work()
                pf.complete(job.pf)
                continue
            }
            // Everything else (decode, prefill without read-ahead) runs with the prefill reads already finished.
            pf.waitPending(atMost: 0)
            while event.signaledValue < job.value {}  // spin: ~64 µs per layer versus ~100 with wait
            let m0 = traceOn ? mach_absolute_time() : 0
            let t0 = DispatchTime.now().uptimeNanoseconds
            handle(job)
            let t1 = DispatchTime.now().uptimeNanoseconds
            statsLock.lock()
            if traceOn { traceMarks.append((job.value, job.layer, m0, mach_absolute_time())) }
            if case .custom = job.kind {
                stats.customJobs += 1
                stats.customNs += t1 - t0
            } else if job.pf > 0 {
                stats.prefillJobs += 1
                stats.prefillStallNs += t1 - t0
            } else {
                stats.jobs += 1
                stats.stallNs += t1 - t0
            }
            statsLock.unlock()
            event.signaledValue = job.value + 1
            if let d = deferredPrefetch {
                deferredPrefetch = nil
                runPrefetch(d.layer, rows: d.rows, protect: d.protect)
            }
        }
    }

    func handle(_ job: Job) {  // swiftlint:disable:this cyclomatic_complexity - hot path: every layer and round
        let l = job.layer, E = experts
        var needed: [Int] = []
        var rowsForPred = 0
        var next: Int?
        switch job.kind {
        case .verify(let moe, let rowState, let rows, let nx):
            returnBorrowed()
            let args = moe.args.contents().bindMemory(to: UInt32.self, capacity: 8)
            let U = Int(args[1]) - 1
            let en = moe.entries.contents().bindMemory(to: UInt32.self, capacity: (U + 1) * 4)
            needed.reserveCapacity(U)
            for i in 0..<U { needed.append(Int(en[4 * i])) }
            rowsForPred =
                rowState.map { min(Int($0.contents().load(fromByteOffset: 24, as: UInt32.self)), rows) } ?? rows
            next = nx
        case .prefill(let pmoe, let rows):
            needed = uniqueExperts(pmoe.inds, count: rows * topk)
        case .verifyInds(let inds, let wts, let rowState, let rows, let nx):
            returnBorrowed()
            let r = rowState.map { min(Int($0.contents().load(fromByteOffset: 24, as: UInt32.self)), rows) } ?? rows
            if skipRank != nil, let w = wts { skipJob = (inds, w, r) }
            needed = uniqueExperts(inds, count: r * topk)
            let ip = inds.contents().bindMemory(to: UInt32.self, capacity: r * topk)
            for i in 0..<(r * topk) {
                let k = l * E + Int(ip[i])
                if slotOf[k] < 0 { rankOf[k] = min(rankOf[k] ?? 99, i % topk) }
            }
            rowsForPred = r
            next = nx
        case .prefillInds(let inds, let rows):
            needed = uniqueExperts(inds, count: rows * topk)
        case .custom(let work, _):
            work()
            return
        case .bulk, .preload, .sync:
            preconditionFailure()
        }
        // This layer's read-ahead (requested in the previous one) must have arrived (in a round, with the partial wait,
        // only that of the experts the round requests).
        var prefillRows = 0
        if case .prefill(_, let rows) = job.kind { prefillRows = rows }
        if case .prefillInds(_, let rows) = job.kind { prefillRows = rows }
        waitAhead(l, needed: needed, partial: prefillRows == 0)
        reapAhead()
        if let (inds, w, r) = skipJob {
            skipJob = nil
            fetchSkipped(protect: Set(needed.map { l * E + $0 }))
            needed = applySkip(layer: l, needed: needed, inds: inds, wts: w, rows: r)
        }
        if prefillRows > 0 {
            // Prefill: whatever is missing goes to the staging area (it does not enter the cache or evict anyone).
            let keys = needed.map { l * E + $0 }.filter { slotOf[$0] < 0 }
            wideReturnPool()
            waitDone(lastPfValue[l])
            lastPfValue[l] = job.pf
            let ring = ringAlloc(keys.count, value: job.pf)
            var miss: [(key: Int, slot: Int)] = []
            for (k, s) in zip(keys, ring) {
                miss.append((k, Int(s)))
                setMap(k, Int(s))
            }
            if !miss.isEmpty { readNow(miss) }
            statsLock.lock()
            stats.prefillReads += miss.count
            statsLock.unlock()
            return
        }
        var protect = Set<Int>()
        for e in needed { protect.insert(l * E + e) }
        // Never evict an expert with its read-ahead in flight (from another layer: with deeper read-ahead there can be
        // reads for l + 2 or more pending); its slot would go to another expert while the old read is still writing.
        let busy = pendingPrefetch.map { $0 != nil }
        if busy.contains(true) { arc.excludePred = { busy[$0 / E] } }
        defer { arc.excludePred = nil }
        var miss: [(key: Int, slot: Int)] = []
        var hits = 0, useful = 0
        // Whatever has a staging slot is not a miss (nor a hit): it is read from there.
        var isWide = false
        if wideK > 0, case .verifyInds = job.kind { isWide = wideEnsurePool() }
        var claim: [(key: Int, slot: Int)] = [], waitK: [Int] = []
        if isWide { (claim, waitK) = wideSplit(l, needed: needed) }
        let claimKeys = claim.map { $0.key }
        let wideKeys = Set(claimKeys + waitK)
        for e in needed {
            let k = l * E + e
            if wideKeys.contains(k) { continue }
            if slotOf[k] >= 0 {
                hits += 1
                arc.touch(k)
                if prefetchedKeys.remove(k) != nil { useful += 1 }
            } else {
                miss.append((k, allocate(k, protect: protect)))
                if let rk = rankOf.removeValue(forKey: k) {
                    statsLock.lock(); stats.missRank[min(rk, 15)] += 1; statsLock.unlock()
                }
            }
        }
        rankOf.removeAll(keepingCapacity: true)
        let tr = DispatchTime.now().uptimeNanoseconds
        if isWide { wide.reader.beginDemand() }
        if !miss.isEmpty || !claim.isEmpty { readNow(miss + claim) }
        if isWide { wide.reader.endDemand() }
        for (k, s) in miss { setResident(k, s) }
        var wns: UInt64 = 0
        if isWide {
            let tw = DispatchTime.now().uptimeNanoseconds
            for k in waitK { wide.reader.finish(k) }
            wns = DispatchTime.now().uptimeNanoseconds - tw
            wideUse(claimKeys + waitK, protect: protect)
        }
        let rns = DispatchTime.now().uptimeNanoseconds - tr
        // The next layer's read-ahead is decided after answering the GPU (before, it was done with the GPU stalled);
        // layer l stays protected until then.
        if let n = next, prefetchK > 0, rowsForPred > 0 {
            deferredPrefetch = (n, rowsForPred, protect)
        }
        statsLock.lock()
        if prefillRows > 0 {
            stats.prefillReads += miss.count
        } else {
            stats.requests += needed.count
            stats.hits += hits
            stats.demandReads += miss.count
            stats.prefetchUseful += useful
            stats.bytesRead += miss.count * expertBytes
            stats.wideClaimed += claim.count
            stats.wideUsed += waitK.count
            stats.wideWaitNs += wns
            stats.readNs += rns
        }
        statsLock.unlock()
    }

    /// Distinct experts (in order of appearance) among the first `count` indices of `inds`.
    func uniqueExperts(_ inds: MTLBuffer, count: Int) -> [Int] {
        let E = experts
        let ip = inds.contents().bindMemory(to: UInt32.self, capacity: count)
        var seen = [Bool](repeating: false, count: E)
        var out: [Int] = []
        for i in 0..<count {
            let e = Int(ip[i])
            if e < E && !seen[e] { seen[e] = true; out.append(e) }
        }
        return out
    }
}
