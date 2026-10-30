import Foundation
import Metal

// SSD expert streaming: SSD reads of ExpertStream (ExpertStream.swift): 16 KiB-aligned `pread`s with F_NOCACHE into
// staging buffers and from there to the slots.

extension ExpertStream {
    func forEachTask(_ n: Int, _ body: @escaping @Sendable (Int) -> Void) {
        DispatchQueue.concurrentPerform(iterations: n, execute: body)
    }

    /// Reads now (in parallel) the missing experts.
    func readNow(_ list: [(key: Int, slot: Int)]) {
        if list.count > 1 { readCoalesced(list); return }
        let np = pieceBytes.count
        let n = list.count * np
        forEachTask(n) { [self] i in
            let (k, s) = list[i / np]
            readPieceSync(key: k, slot: s, piece: i % np)
        }
    }

    /// Like `readNow`, but consecutive experts of a layer (up to `coalesce`) go in a single aligned `pread` per chunk,
    /// into a large staging buffer, and from there each one to its slot.
    func readCoalesced(_ list0: [(key: Int, slot: Int)]) {
        let list = list0.sorted { $0.key < $1.key }
        var runs0: [Range<Int>] = []
        var a = 0
        for i in 1...list.count {
            if i == list.count || list[i].key != list[i - 1].key + 1 || i - a >= coalesce
                || list[i].key / experts != list[a].key / experts
            {
                runs0.append(a..<i); a = i
            }
        }
        let runs = runs0
        ensureBigStaging()
        // Tasks: (run, chunk), each with its whole read.
        var tasks0: [(run: Int, piece: Int, p0: Int, p1: Int)] = []
        for (ri, run) in runs.enumerated() {
            for i in 0..<pieceBytes.count { tasks0.append((ri, i, 0, pieceBytes[i] * run.count)) }
        }
        let tasks = tasks0
        forEachTask(tasks.count) { [self] t in
            let tk = tasks[t]
            let run = runs[tk.run], i = tk.piece
            let k0 = list[run.lowerBound].key
            let l = k0 / experts, e0 = k0 % experts
            let (fd, toff) = src[l][i]
            let len = pieceBytes[i]
            let off = toff + e0 * len + tk.p0, total = tk.p1 - tk.p0
            let a0 = off & ~16383, a1 = (off + total + 16383) & ~16383
            bigSem!.wait()
            bigLock.lock()
            let bi = bigFree.removeLast()
            bigLock.unlock()
            let buf = bigStaging[bi]
            var done = 0
            while done < a1 - a0 {
                let r = pread(fd, buf + done, a1 - a0 - done, off_t(a0 + done))
                if r <= 0 { break }
                done += r
            }
            precondition(done >= off + total - a0, "SSD expert streaming: short SSD read")
            // Copy [p0, p1) of the run to the slots of its experts.
            var b = tk.p0
            while b < tk.p1 {
                let j = b / len, within = b % len
                let n = min(len - within, tk.p1 - b)
                (arenas[i].contents() + list[run.lowerBound + j].slot * len + within).copyMemory(
                    from: buf + (off - a0) + (b - tk.p0), byteCount: n)
                b += n
            }
            bigLock.lock()
            bigFree.append(bi)
            bigLock.unlock()
            bigSem!.signal()
        }
    }

    /// Buffers for the coalesced reads (created the first time, from the thread that serves the GPU).
    func ensureBigStaging() {
        bigLock.lock()
        defer { bigLock.unlock() }
        guard bigStaging.isEmpty else { return }
        let bigSize = coalesce * pieceBytes.max()! + 2 * 16384
        let n = 16
        bigStaging = (0..<n).map { _ in
            var p: UnsafeMutableRawPointer?
            posix_memalign(&p, 16384, bigSize)
            return p!
        }
        bigFree = Array(0..<n)
        bigSem = DispatchSemaphore(value: n)
    }

    /// One chunk: the 16 KiB-aligned span that contains it into the staging buffer and the useful part to the slot.
    func readPieceSync(key k: Int, slot s: Int, piece i: Int) {
        let l = k / experts, e = k % experts
        let (fd, toff) = src[l][i]
        let len = pieceBytes[i]
        let off = toff + e * len
        let a0 = off & ~16383, a1 = (off + len + 16383) & ~16383
        stagingSem.wait()
        stagingLock.lock()
        let si = stagingFree.removeLast()
        stagingLock.unlock()
        let buf = staging[si]
        var done = 0
        while done < a1 - a0 {
            let r = pread(fd, buf + done, a1 - a0 - done, off_t(a0 + done))
            if r <= 0 { break }
            done += r
        }
        precondition(done >= off + len - a0, "SSD expert streaming: short SSD read")
        (arenas[i].contents() + s * len).copyMemory(from: buf + (off - a0), byteCount: len)
        stagingLock.lock()
        stagingFree.append(si)
        stagingLock.unlock()
        stagingSem.signal()
    }
}
