import Foundation
import Metal

// SSD expert streaming: MoE experts read from the SSD with an in-memory slot cache.
//
// - Store: `slots` global slots (for all 40 layers), each holding the 9 chunks of one expert (gate, up and down:
//   weights, scales and biases) in 9 shared buffers with the same per-expert stride as the safetensors tensors.
//   The MoE kernels read expert e of layer l from slot `smap[l][e]` (in normal mode, the identity).
// - ARC replacement (as in splash 1.4.0) for decode. Prefill does not enter the cache: its missing experts go to
//   a pass-through area of E slots (one layer), so it does not evict what decode uses (with 6.8 GB, a 501-token prompt
//   read 11.6 GB and left the next decode with a 77% hit rate).
// - Synchronization: at every MoE layer, after the router, the encoder is cut with an MTLSharedEvent: the GPU signals
//   and waits; a CPU thread looks at the routing (the expert union of the verification or the prefill top-k), fetches
//   the missing ones from the SSD (`pread` with F_NOCACHE, 16 KiB-aligned, into a staging buffer, then copy to the
//   slot), writes the slot table and answers. Measured: ~64 µs per layer with the CPU spinning.
// - Read-ahead (verification): the router of layer l + 1 is applied to the FFN input of layer l; when answering
//   layer l, the CPU requests the top-k of each row of that prediction that are missing, without waiting.
// - Exact: the kernels do the same math with the same weights; only where they read them from changes.
// - Files: this one holds the state and loading; ExpertStreamEncode (encoder cuts), ExpertStreamServe (CPU thread),
//   ExpertStreamCache (read-ahead and ARC slots), ExpertStreamPrefill (prefill ring), ExpertStreamIO
//   (SSD reads).
public final class ExpertStream: @unchecked Sendable {
    public let ctx: MetalContext
    public let experts: Int, topk: Int, numLayers: Int
    /// Bytes of each of the 9 chunks of an expert and of the whole expert.
    public let pieceBytes: [Int]
    public let expertBytes: Int
    public let slots: Int
    /// Slots managed by ARC (the first Z are the prefill pass-through area: a ring).
    public let cacheSlots: Int
    /// 9 buffers [slots, …]: gate (w, s, b), up (w, s, b), down (w, s, b).
    public let arenas: [MTLBuffer]
    /// [numLayers, experts] uint32: slot of each expert (only valid for the resident ones).
    public let smap: MTLBuffer
    /// [numLayers, experts] uint32: slot of each cached expert that is already read, or 0xffffffff. The GPU reads it
    /// before stopping at the layer (to compute the residents while the CPU fetches what is missing), so the CPU only
    /// sets a slot once the expert is read, and clears it (on eviction) only if the GPU cannot be using it:
    /// when the GPU stops at layer m, never for an expert in the union of m; during read-ahead (the GPU keeps going
    /// towards layer n), never for one of layer n.
    public let smapR: MTLBuffer
    /// Router logits of layer l over the FFN input of layer l − 1 ([16, E] fp32), for read-ahead.
    /// adelantada.
    public let pred: [MTLBuffer?]
    /// Partial wait (ExpertAhead.swift); group of each expert with its read-ahead in flight.
    var aheadGroup: [Int: DispatchGroup] = [:]
    public var prefetchK: Int
    /// Stride of the `pred` rows (E in Qwen3.6; the padded router outputs in Flash-Next).
    public let predStride: Int
    let event: MTLSharedEvent
    var nextValue: UInt64 = 0
    /// Prefill: the CPU signals `ready` (v) when the layer of job v has its experts; the GPU signals `done` (v)
    /// when it has finished the MoE of that layer (its ring slots become free).
    let readyEvent: MTLSharedEvent, doneEvent: MTLSharedEvent
    var pfSeq: UInt64 = 0
    // Pass-through ring (touched only by the CPU thread): free slots and, in order, the occupied ones with the `done`
    // value that frees them.
    var ringFree: [Int32]
    var ringUsed: [(value: UInt64, slots: [Int32])] = []
    var lastPfValue: [UInt64]

    // SSD source: per layer and chunk, (descriptor, tensor offset in the file).
    let src: [[(fd: Int32, off: Int)]]
    let fds: [Int32]

    // Cache state (touched only by the CPU thread that serves the GPU).
    var slotOf: [Int32]  // key (layer·E + expert) → slot or −1
    var ownerOf: [Int32]  // slot → key or −1
    var freeSlots: [Int32]
    var arc: ARCList
    var pendingPrefetch: [DispatchGroup?]
    /// Keys (and slots) of each layer's read-ahead in progress: published in `smapR` when the layer is reached.
    var pendingKeys: [[(key: Int, slot: Int)]]
    /// Preload (ExpertPreload.swift): keys in value order and the next one to look at.
    var preloadKeys: [Int] = [], preloadPos = 0, preloadT0: UInt64 = 0, preloadRead = 0, lastJobNs: UInt64 = 0
    var preloadReadNs: UInt64 = 0
    /// One line when the preload finishes (from the CPU thread).
    public var preloadLog: (@Sendable (String) -> Void)?

    // Reads: aligned staging buffers.
    let stagingSize: Int
    let staging: [UnsafeMutableRawPointer]
    let stagingLock = NSLock()
    var stagingFree: [Int]
    let stagingSem: DispatchSemaphore
    let readQueue = DispatchQueue(label: "expert.read", qos: .userInteractive, attributes: .concurrent)

    // Job queue (one per MoE layer and pass) and the thread that serves it.
    public enum JobKind {
        /// Verification: the expert union of `moe` (entries/args); `rowState` (st[6]) or `rows` rows for the
        /// prediction of the next layer.
        case verify(moe: MoELayer, rowState: MTLBuffer?, rows: Int, next: Int?)
        /// Prefill: the top-k of the first `rows` rows of `pmoe.inds`.
        case prefill(pmoe: PrefillMoE, rows: Int)
        /// Prefill ahead of the GPU: all the layer's experts that are not in the cache.
        case bulk
        /// Flash-Next, verification: the top-k of the rows in `inds` ([rows, topk] uint32): `rowState` (st[6]) or
        /// `rows` rows; `next`: layer of the prediction.
        /// `wts`: weights [rows, topk] fp32 (lossy expert skipping changes them before the combination).
        case verifyInds(inds: MTLBuffer, wts: MTLBuffer?, rowState: MTLBuffer?, rows: Int, next: Int?)
        /// Flash-Next, prefill without fetching ahead: the top-k of the first `rows` rows of `inds`.
        case prefillInds(inds: MTLBuffer, rows: Int)
        /// Someone else's work (Flash-Next's PLE table) on the same thread, with the GPU stopped at the barrier (or, with
        /// `ahead`, ahead of the GPU: answered with `readyEvent` like `.bulk`).
        case custom(work: () -> Void, ahead: Bool)
        /// Preload these keys into the free slots when there is no other work (ExpertPreload.swift).
        case preload(keys: [Int])
        /// CPU-thread work outside the passes (without touching the GPU events).
        case sync(work: () -> Void)
    }
    struct Job {
        let value: UInt64
        let layer: Int
        let kind: JobKind
        /// Prefill: `done` value that frees its ring slots (0 in decode).
        var pf: UInt64 = 0
    }
    var jobs: [Job] = []
    let jobLock = NSLock()
    let jobSem = DispatchSemaphore(value: 0)
    var thread: Thread?

    /// Counters (since the last `resetStats`).
    public struct Stats: Sendable {
        public var jobs = 0, requests = 0, hits = 0, demandReads = 0, prefetchReads = 0, prefetchUseful = 0
        public var bytesRead = 0
        /// CPU time with the GPU stopped at a layer barrier (from seeing it arrive until answering).
        public var stallNs: UInt64 = 0, readNs: UInt64 = 0
        /// The same, for prefill layers only (not included in the above).
        public var prefillJobs = 0, prefillReads = 0, prefillStallNs: UInt64 = 0
        /// Prefill ahead: layers, CPU read time and wait for a free ring slot.
        public var bulkJobs = 0, bulkReadNs: UInt64 = 0, bulkRingWaitNs: UInt64 = 0
        /// Experts read by the `lab serve` preload (ExpertPreload.swift).
        public var warmReads = 0
        /// Other people's jobs (Flash-Next PLE) with the GPU stopped.
        public var customJobs = 0, customNs: UInt64 = 0
        /// Flash-Next (diagnostics for lossy expert skipping): on-demand misses of the round by their best rank (0 =
        /// top-1) among the rows.
        public var missRank = [Int](repeating: 0, count: 16)
        /// Lossy expert skipping: missing experts that were not waited for (keys per layer and round) and (row, k) pairs
        /// whose weight was dropped.
        public var skipped = 0, skippedPairs = 0, skipFetched = 0
        /// Read-ahead experts still in flight on reaching their layer and not waited for.
        public var aheadLeft = 0
        /// Wide read (ExpertWide.swift): requested, used on reaching its layer (already read or in progress), claimed
        /// (still queued: read as demand), cancelled, read unused, without a pass-through slot, and wait for those still
        /// in progress.
        public var wideIssued = 0, wideUsed = 0, wideClaimed = 0, wideCancelled = 0, wideWasted = 0, wideShort = 0
        public var wideWaitNs: UInt64 = 0
        /// Wide read: of those used, how many came from what was read ahead in earlier rounds and not requested then.
        public var wideStashHits = 0
        public var hitRate: Double { requests > 0 ? Double(hits) / Double(requests) : 0 }
    }
    public internal(set) var stats = Stats()
    let statsLock = NSLock()
    var prefetchedKeys = Set<Int>()
    /// Top-k of each row in the wide read into pass-through buffers (0 = the previous read-ahead into the cache;
    /// ExpertWide.swift).
    public var wideK = 0
    lazy var wide: WideState = makeWide()

    /// Lossy (nil = off): minimum rank (0 = top-1) from which a missing expert is not waited for.
    public var skipRank: Int?
    /// (`--round-trace`): encoders with timestamps in the round being recorded and, on the thread, (value, layer, mach
    /// when the signal is seen, mach when answered) of each job the GPU waits for.
    public var roundTrace: RoundTrace? { didSet { traceOn = roundTrace != nil } }
    var traceOn = false
    var traceMarks: [(UInt64, Int, UInt64, UInt64)] = []
    /// The thread's marks with an event value in [lo, hi]; also drops the earlier ones (unrecorded rounds).
    public func takeTraceMarks(from lo: UInt64, upTo hi: UInt64) -> [(UInt64, Int, UInt64, UInt64)] {
        statsLock.lock()
        defer { statsLock.unlock() }
        let r = traceMarks.filter { $0.0 >= lo && $0.0 <= hi }
        traceMarks.removeAll { $0.0 <= hi }
        return r
    }
    /// Zero slot: skipped experts point to it (output exactly 0) until they are really read.
    var zeroSlot: Int32 = -1
    var skipJob: (inds: MTLBuffer, wts: MTLBuffer, rows: Int)?
    /// Skipped experts are read anyway, in the background (otherwise one of rank ≥ N is never read as long as it does
    /// not come out with rank < N). They are requested in the next job (its layer is already computed and its `smap`,
    /// which points to the zero slot, can be changed) and published in `smapR` on returning to its layer.
    var skipBacklog: [(layer: Int, keys: [Int])] = []

    /// Diagnostics: best rank of each missing expert in the job in progress.
    var rankOf: [Int: Int] = [:]

    /// Pending read-ahead (layer, prediction rows, protected keys), after answering the GPU.
    var deferredPrefetch: (layer: Int, rows: Int, protect: Set<Int>)?

    let bulkQueue = DispatchQueue(label: "expert.bulk", qos: .userInteractive, attributes: .concurrent)
    /// Jobs ahead (`bulk` and `custom` with `ahead`) in flight; signals `readyEvent` in order.
    lazy var pf = PfTracker(event: readyEvent)

    /// Cache slots lent to the prefill ring (returned at the first decode job) and the maximum.
    var borrowed: [Int32] = []
    let maxBorrow = 128

    /// Consecutive experts (same layer) read with a single `pread` per chunk (measured: 8).
    let coalesce = 8
    /// Large staging buffers for the coalesced reads.
    var bigStaging: [UnsafeMutableRawPointer] = []
    var bigFree: [Int] = []
    let bigLock = NSLock()
    var bigSem: DispatchSemaphore?

    /// The experts of a Qwen3.5/3.6 MoE in their MLX safetensors: 9 chunks per expert (gate, up and down: weights,
    /// scales and biases), each with its per-expert stride inside the tensor.
    public convenience init(
        ctx: MetalContext, store: SafetensorsStore, config cfg: ModelConfig, bytes: Int, prefetchK: Int = 4
    ) throws {
        let sp = try cfg.requiredSpec
        guard sp.expertLayout == .mlxSwitchMLP, let E = cfg.numExperts, let K = cfg.expertsPerToken else {
            throw SafetensorsError.format("SSD streaming: \(sp.family.rawValue) experts cannot be read from the SSD")
        }
        // Shard descriptors (F_NOCACHE: reads bypass the page cache when aligned).
        var fds: [Int32] = []
        for f in store.files {
            let fd = open(f.path, O_RDONLY)
            guard fd >= 0 else { throw SafetensorsError.io("cannot open \(f.path)") }
            _ = fcntl(fd, F_NOCACHE, 1)
            fds.append(fd)
        }
        var src: [[(Int32, Int)]] = []
        var pb: [Int] = []
        for l in 0..<cfg.numLayers {
            let pre = try store.layerPrefix(l) + sp.names.ffn + sp.names.experts
            var row: [(Int32, Int)] = []
            var bytesRow: [Int] = []
            for proj in ["gate_proj", "up_proj", "down_proj"] {
                for part in ["weight", "scales", "biases"] {
                    let t = try store.info(pre + proj + "." + part)
                    guard t.byteCount % E == 0 else { throw SafetensorsError.format("SSD expert streaming: \(t.name)") }
                    row.append((fds[t.shard], t.fileOffset))
                    bytesRow.append(t.byteCount / E)
                }
            }
            if pb.isEmpty { pb = bytesRow }
            guard bytesRow == pb else { throw SafetensorsError.format("SSD streaming: expert sizes differ by layer") }
            src.append(row)
        }
        try self.init(
            ctx: ctx, experts: E, topk: K, layers: cfg.numLayers, src: src, pieceBytes: pb, fds: fds, bytes: bytes,
            prefetchK: prefetchK, predStride: E)
    }

    /// Flash-Next (scripts/flashnext): a single record per expert in experts.bin ([layer][expert], 16 KiB-aligned):
    /// one chunk of `fmt.recordBytes` at offset (layer·E + e)·record.
    public convenience init(
        ctx: MetalContext, directory: URL, flashNext fmt: FlashNextFormat, topk K: Int, bytes: Int,
        prefetchK: Int = 4, predStride: Int
    ) throws {
        let fd = open(directory.appendingPathComponent("experts.bin").path, O_RDONLY)
        guard fd >= 0 else { throw SafetensorsError.io("cannot open experts.bin") }
        _ = fcntl(fd, F_NOCACHE, 1)
        let E = fmt.experts, R = fmt.recordBytes
        try self.init(
            ctx: ctx, experts: E, topk: K, layers: fmt.layers, src: (0..<fmt.layers).map { [(fd, $0 * E * R)] },
            pieceBytes: [R], fds: [fd], bytes: bytes, prefetchK: prefetchK, predStride: predStride)
    }

    init(
        ctx: MetalContext, experts E: Int, topk K: Int, layers numLayers: Int, src: [[(fd: Int32, off: Int)]],
        pieceBytes pb: [Int], fds: [Int32], bytes: Int, prefetchK: Int, predStride: Int
    ) throws {
        self.ctx = ctx
        experts = E; topk = K; self.numLayers = numLayers
        self.prefetchK = prefetchK
        self.predStride = predStride
        self.fds = fds
        self.src = src
        pieceBytes = pb
        expertBytes = pb.reduce(0, +)
        // Chunks need not be multiples of 16 KiB: each read covers the 16 KiB-aligned span that contains the chunk
        // (readPieceSync, readCoalesced), so a small chunk (the 4 KiB scales of toy-qwen36-moe) only reads more bytes.
        let ns = min(bytes / pb.reduce(0, +), numLayers * E)
        slots = ns
        // Prefill pass-through area: the experts of one layer (a fixed ring of 384 lost on decode).
        let Z = E
        // The cache must be able to hold at once the experts of one layer (≤ 16 rows × top-k) and the read-ahead ones of
        // the next.
        guard ns >= Z + E else {
            throw SafetensorsError.format("SSD streaming: --expert-cache-gb too small (\(ns) slots; minimum \(Z + E))")
        }
        cacheSlots = ns - Z
        let dev = ctx.device
        arenas = try pb.map { b in
            guard let buf = dev.makeBuffer(length: b * ns, options: .storageModeShared) else {
                throw SafetensorsError.io("SSD expert streaming: cannot create the store (\(b * ns) bytes)")
            }
            return buf
        }
        let sm = ctx.buffer(length: numLayers * E * 4)
        memset(sm.contents(), 0, numLayers * E * 4)
        smap = sm
        let sr = ctx.buffer(length: numLayers * E * 4)
        memset(sr.contents(), 0xff, numLayers * E * 4)
        smapR = sr
        pred = (0..<numLayers).map { $0 == 0 ? nil : ctx.buffer(length: MoELayer.maxRows * predStride * 4) }
        event = ctx.device.makeSharedEvent()!
        event.signaledValue = 0
        readyEvent = ctx.device.makeSharedEvent()!
        readyEvent.signaledValue = 0
        doneEvent = ctx.device.makeSharedEvent()!
        doneEvent.signaledValue = 0
        // The pass-through area goes in the first Z slots and the cache in the ones above.
        ringFree = (0..<Int32(Z)).reversed()
        lastPfValue = [UInt64](repeating: 0, count: numLayers)
        let keys = numLayers * E
        slotOf = [Int32](repeating: -1, count: keys)
        ownerOf = [Int32](repeating: -1, count: ns)
        freeSlots = (Int32(Z)..<Int32(ns)).reversed()
        arc = ARCList(keys: keys, capacity: ns - Z)
        pendingPrefetch = [DispatchGroup?](repeating: nil, count: numLayers)
        pendingKeys = [[(key: Int, slot: Int)]](repeating: [], count: numLayers)
        // A chunk fits in the staging buffer with 16 KiB extra on each side.
        stagingSize = (pb.max()! + 2 * 16384)
        let nStaging = 32
        staging = (0..<nStaging).map { _ in
            var p: UnsafeMutableRawPointer?
            posix_memalign(&p, 16384, (pb.max()! + 2 * 16384))
            return p!
        }
        stagingFree = Array(0..<nStaging)
        stagingSem = DispatchSemaphore(value: nStaging)
        let t = Thread { [unowned self] in self.serve() }
        t.qualityOfService = .userInteractive
        t.name = "expert.stream"
        thread = t
        t.start()
    }

    deinit {
        for fd in fds { close(fd) }
        for p in staging { free(p) }
    }

    public var arenaBytes: Int { expertBytes * slots }

    /// One line with the counters since the last `resetStats`.
    public func summary(rounds: Int) -> String {
        statsLock.lock()
        let s = stats
        statsLock.unlock()
        let R = Double(max(rounds, 1))
        return String(
            format:
                "SSD expert streaming: %d slots (%.2f GB); "
                + "decode: hits %.1f%%, per round %.1f misses + %.1f read ahead (useful %d), %.0f MB read, GPU stopped %.2f ms (read %.2f); "
                + "prefill: %d experts read (%.2f GB), GPU stopped %.0f ms; "
                + "preloaded %d; ahead %d layers, read %.0f ms, ring wait %.0f ms",
            slots, Double(arenaBytes) / 1e9, s.hitRate * 100, Double(s.demandReads) / R, Double(s.prefetchReads) / R,
            s.prefetchUseful, Double(s.bytesRead) / 1e6 / R, Double(s.stallNs) / 1e6 / R, Double(s.readNs) / 1e6 / R,
            s.prefillReads, Double(s.prefillReads * expertBytes) / 1e9, Double(s.prefillStallNs) / 1e6, s.warmReads,
            s.bulkJobs,
            Double(s.bulkReadNs) / 1e6, Double(s.bulkRingWaitNs) / 1e6)
            + (wideK > 0
                ? String(
                    format:
                        "; wide top-%d: per round %.1f requested, %.1f used, %.1f claimed, "
                        + "%.1f cancelled, %.1f unused, %.1f without slot, wait %.2f ms, %.1f from the stash",
                    wideK, Double(s.wideIssued) / R, Double(s.wideUsed) / R, Double(s.wideClaimed) / R,
                    Double(s.wideCancelled) / R, Double(s.wideWasted) / R, Double(s.wideShort) / R,
                    Double(s.wideWaitNs) / 1e6 / R, Double(s.wideStashHits) / R) : "")
            + String(format: "; not waited for, per round %.1f", Double(s.aheadLeft) / R)
            + (skipRank.map {
                String(
                    format:
                        "; lossy skipping (rank ≥ %d): %.1f experts skipped per round (%.1f pairs; %.1f read in the background)",
                    $0 + 1, Double(s.skipped) / R, Double(s.skippedPairs) / R, Double(s.skipFetched) / R)
            } ?? "")

    }

    public func resetStats() {
        statsLock.lock(); stats = Stats(); statsLock.unlock()
    }
}
