import Foundation

// Expert cache preload at `lab serve` startup.
//
// Prefill does not put what it reads into the cache (it goes to the staging area), so after startup the cache is
// empty and each request without a decode reads everything it routes from the SSD: with 498 tokens, 18.7 GB and 3.5 s
// cold versus 11 GB and 2.3 s once a decode has filled it. On shutdown, `lab serve` saves the resident keys in order
// of value (ARC T2 then T1, MRU to LRU); at startup, the CPU thread reads them back into the free slots, in batches
// and only when it has no other work (requests and warm-up go first). Exact: only where the weights are read from
// changes.
extension ExpertStream {
    /// Resident keys, most valuable first (T2 then T1, MRU to LRU). On the CPU thread.
    public func residentKeysByValue() -> [Int] {
        var out: [Int] = []
        let sem = DispatchSemaphore(value: 0)
        enqueue(
            Job(
                value: 0, layer: 0,
                kind: .sync(work: { [unowned self] in
                    self.preloadKeys = []
                    for w in [2, 1] {
                        var k = self.arc.head[w]
                        while k >= 0 {
                            if self.slotOf[Int(k)] >= 0 { out.append(Int(k)) }
                            k = self.arc.next[Int(k)]
                        }
                    }
                    sem.signal()
                })))
        sem.wait()
        return out
    }

    /// Asks to preload `keys` (in order) into the free slots; returns immediately.
    public func preload(_ keys: [Int]) {
        enqueue(Job(value: 0, layer: 0, kind: .preload(keys: keys.filter { $0 >= 0 && $0 < numLayers * experts })))
    }

    func startPreload(_ keys: [Int]) {
        preloadKeys = keys
        preloadPos = 0
        preloadRead = 0
        preloadReadNs = 0
        preloadT0 = DispatchTime.now().uptimeNanoseconds
    }

    /// No queued jobs and no prefill reads in flight.
    func preloadIdle() -> Bool {
        jobLock.lock()
        let empty = jobs.isEmpty
        jobLock.unlock()
        return empty && pf.idle && DispatchTime.now().uptimeNanoseconds - lastJobNs > 50_000_000
    }

    /// One batch of ~32 MB: free slots only (evicts nobody, so it never touches a slot the GPU is using).
    func preloadStep() {
        // Slots the last prefill borrowed for the staging area go back to the cache first (the GPU has finished:
        // no work for 50 ms; returnBorrowed checks it against the ring).
        returnBorrowed()
        let batch = max(1, (32 << 20) / expertBytes)
        var list: [(key: Int, slot: Int)] = []
        while list.count < batch && preloadPos < preloadKeys.count && !freeSlots.isEmpty && arc.resident < arc.c {
            let k = preloadKeys[preloadPos]
            preloadPos += 1
            if slotOf[k] >= 0 { continue }
            list.append((k, allocate(k, protect: [])))
        }
        if list.isEmpty {
            finishPreload()
            return
        }
        let t0 = DispatchTime.now().uptimeNanoseconds
        readNow(list)
        preloadReadNs += DispatchTime.now().uptimeNanoseconds - t0
        for (k, s) in list { setResident(k, s) }
        preloadRead += list.count
        statsLock.lock()
        stats.warmReads += list.count
        statsLock.unlock()
        if preloadPos >= preloadKeys.count { finishPreload() }
    }

    private func finishPreload() {
        let ms = Double(DispatchTime.now().uptimeNanoseconds - preloadT0) / 1e6
        preloadLog?(
            String(
                format: "experts: %d preloaded (%.2f GB) in %.0f ms (%.0f ms reading)", preloadRead,
                Double(preloadRead * expertBytes) / 1e9, ms, Double(preloadReadNs) / 1e6))
        preloadKeys = []
        preloadPos = 0
    }

    /// Fingerprint of the expert layout: if it does not match, the keys file is not valid.
    var preloadFingerprint: String { "\(numLayers)x\(experts)x\(expertBytes)" }

    /// Saves the resident keys (for the next startup's preload). Returns how many.
    @discardableResult
    public func savePreloadKeys(to url: URL, model: String) throws -> Int {
        let keys = residentKeysByValue()
        guard !keys.isEmpty else { return 0 }
        let head = try JSONSerialization.data(withJSONObject: [
            "version": 1, "layout": preloadFingerprint, "model": model, "keys": keys.count,
        ])
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        let body = keys.map { UInt32($0) }.withUnsafeBytes { Data($0) }
        try (head + Data([10]) + body).write(to: url, options: .atomic)
        return keys.count
    }

    /// Reads the keys saved by `savePreloadKeys` (empty if there is no file or it belongs to another model).
    public func loadPreloadKeys(from url: URL, model: String) -> [Int] {
        guard let data = try? Data(contentsOf: url), let nl = data.firstIndex(of: 10),
            let head = try? JSONSerialization.jsonObject(with: data[..<nl]) as? [String: Any],
            head["version"] as? Int == 1, head["layout"] as? String == preloadFingerprint,
            head["model"] as? String == model, let n = head["keys"] as? Int, data.count - nl - 1 == n * 4
        else { return [] }
        return data[(nl + 1)...].withUnsafeBytes { raw in
            (0..<n).map { Int(raw.loadUnaligned(fromByteOffset: $0 * 4, as: UInt32.self)) }
        }
    }
}
