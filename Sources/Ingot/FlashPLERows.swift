import Foundation

/// Cache of Flash-Next PLE table rows (ple.bin, 320 M rows of 100 B, 163 per 16 KiB page) in CPU memory:
/// set-associative (16 ways, the least recently used in the set is evicted), and misses are read from the SSD
/// with F_NOCACHE in 4 KiB-aligned blocks (a row never crosses a 16 KiB page; if it crosses a 4 KiB block, 8 KiB are
/// read). Splash 1.4.0 does the same with 511 MiB (`SparseRowStore`); it touched ~47 rows per round.
///
/// `fetch` copies the requested rows, in order, into a staging buffer read by `fn_ple_gather`; the GPU never reads the
/// cache, so evicting a row has no races with the GPU. It is called from a single thread (ExpertStream's, or the
/// prefill's); only the reads run in parallel.
public final class FlashPLERows: @unchecked Sendable {
    public let fmt: FlashNextFormat
    let fd: Int32
    static let ways = 16
    let sets: Int
    let setMask: Int
    var tags: [UInt32]
    var stamps: [UInt32]
    let data: UnsafeMutableRawPointer
    var clock: UInt32 = 0
    public let bytes: Int

    public struct Stats: Sendable {
        public var calls = 0, lookups = 0, hits = 0, reads = 0, readNs: UInt64 = 0, totalNs: UInt64 = 0
        public var hitRate: Double { lookups > 0 ? Double(hits) / Double(lookups) : 0 }
        public init() {}
    }
    public var stats = Stats()

    /// `bytes`: cache size (rounded to a power-of-two number of sets; 0 = no cache, everything from the SSD).
    public init(directory: URL, fmt: FlashNextFormat, bytes: Int) throws {
        self.fmt = fmt
        fd = open(directory.appendingPathComponent("ple.bin").path, O_RDONLY)
        guard fd >= 0 else { throw SafetensorsError.io("cannot open ple.bin") }
        _ = fcntl(fd, F_NOCACHE, 1)
        var s = 1
        while s * 2 * Self.ways * fmt.pleRowBytes <= bytes { s *= 2 }
        sets = bytes > 0 ? s : 0
        setMask = max(s - 1, 0)
        tags = [UInt32](repeating: .max, count: max(sets, 1) * Self.ways)
        stamps = [UInt32](repeating: 0, count: max(sets, 1) * Self.ways)
        self.bytes = sets * Self.ways * fmt.pleRowBytes
        data = UnsafeMutableRawPointer.allocate(byteCount: max(self.bytes, 16), alignment: 16384)
    }
    deinit {
        close(fd)
        data.deallocate()
    }

    @inline(__always) func setOf(_ r: UInt32) -> Int { Int((UInt64(r) &* 0x9E37_79B9_7F4A_7C15) >> 32) & setMask }

    /// Offset of row r in ple.bin.
    @inline(__always) func offset(_ r: UInt32) -> Int {
        (Int(r) / fmt.pleRowsPerPage) * fmt.plePage + (Int(r) % fmt.pleRowsPerPage) * fmt.pleRowBytes
    }

    /// Copies rows `idx[0..<n]` to `dst` (100 B each, in order), from the cache or the SSD.
    public func fetch(_ idx: UnsafePointer<UInt32>, count n: Int, into dst: UnsafeMutableRawPointer) throws {
        let t0 = DispatchTime.now().uptimeNanoseconds
        clock &+= 1
        let RB = fmt.pleRowBytes, W = Self.ways
        var missRows: [UInt32: Int] = [:]  // row → index in `miss`
        var miss: [(row: UInt32, slot: Int, dsts: [Int])] = []
        for j in 0..<n {
            let r = idx[j]
            // First those already missing in this call (their slot has the tag, but the data is not there yet).
            if let m = missRows[r] {
                miss[m].dsts.append(j)
                continue
            }
            if sets > 0 {
                let b = setOf(r) * W
                var hit = -1
                for w in 0..<W where tags[b + w] == r {
                    hit = b + w
                    break
                }
                if hit >= 0 {
                    stamps[hit] = clock
                    (dst + j * RB).copyMemory(from: data + hit * RB, byteCount: RB)
                    stats.hits += 1
                    continue
                }
            }
            // Slot: the least recently used way in the set not assigned in this call (if all are, the
            // row does not enter the cache).
            var slot = -1
            if sets > 0 {
                let b = setOf(r) * W
                var best = UInt32.max
                for w in 0..<W where stamps[b + w] != clock && (tags[b + w] == .max ? 0 : stamps[b + w] &+ 1) < best {
                    best = tags[b + w] == .max ? 0 : stamps[b + w] &+ 1
                    slot = b + w
                }
                if slot >= 0 {
                    tags[slot] = r
                    stamps[slot] = clock
                }
            }
            missRows[r] = miss.count
            miss.append((r, slot, [j]))
        }
        stats.calls += 1
        stats.lookups += n
        if !miss.isEmpty {
            let tr = DispatchTime.now().uptimeNanoseconds
            let list = miss
            let dataAddr = UInt(bitPattern: data), dstAddr = UInt(bitPattern: dst), fdc = fd
            let offs = list.map { offset($0.row) }
            let bad = UnsafeMutablePointer<Int32>.allocate(capacity: list.count)
            defer { bad.deallocate() }
            let badAddr = UInt(bitPattern: bad)
            DispatchQueue.concurrentPerform(iterations: list.count) { i in
                let o = offs[i]
                let a0 = o & ~4095, a1 = (o + RB + 4095) & ~4095
                let buf = UnsafeMutableRawPointer.allocate(byteCount: 8192, alignment: 16384)
                defer { buf.deallocate() }
                let got = pread(fdc, buf, a1 - a0, off_t(a0))
                UnsafeMutablePointer<Int32>(bitPattern: badAddr)![i] = got == a1 - a0 ? 0 : 1
                let src = buf + (o - a0)
                if list[i].slot >= 0 {
                    UnsafeMutableRawPointer(bitPattern: dataAddr)!.advanced(by: list[i].slot * RB)
                        .copyMemory(from: src, byteCount: RB)
                }
                for j in list[i].dsts {
                    UnsafeMutableRawPointer(bitPattern: dstAddr)!.advanced(by: j * RB).copyMemory(
                        from: src, byteCount: RB)
                }
            }
            let failed = (0..<list.count).reduce(0) { $0 + Int(bad[$1]) }
            if failed > 0 {
                for m in list where m.slot >= 0 { tags[m.slot] = .max }
                throw SafetensorsError.io("pread of ple.bin failed (\(failed) rows)")
            }
            stats.reads += list.count
            stats.readNs += DispatchTime.now().uptimeNanoseconds - tr
        }
        stats.totalNs += DispatchTime.now().uptimeNanoseconds - t0
    }

    /// Diagnostic (reads that bypass the page cache): of the 16 KiB pages of these rows, how many
    /// are in the macOS file cache according to `mincore` (with F_NOCACHE and aligned reads they should be 0).
    public func residentPages(rows: [UInt32]) -> (resident: Int, checked: Int) {
        var st = stat()
        guard fstat(fd, &st) == 0 else { return (0, 0) }
        let size = Int(st.st_size)
        guard let base = mmap(nil, size, PROT_READ, MAP_SHARED, fd, 0), base != MAP_FAILED else { return (0, 0) }
        defer { munmap(base, size) }
        let pg = Int(getpagesize())
        var resident = 0
        var vec = [CChar](repeating: 0, count: 1)
        for r in rows {
            let o = offset(r) & ~(pg - 1)
            if mincore(base + o, pg, &vec) == 0, vec[0] & 1 != 0 { resident += 1 }
        }
        return (resident, rows.count)
    }
}
