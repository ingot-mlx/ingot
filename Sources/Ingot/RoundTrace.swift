import Foundation
import Metal

/// GPU timestamps inside the real rounds of `decodeSpec` (`--round-trace N`; off by default). Each compute encoder
/// of the round carries a mark at the start and another at the end (MTLCounterSampleBuffer at stage boundaries,
/// the only thing the M6 supports). With level 1 only the encoders the round already has are marked (the cuts of
/// ExpertStream in each layer); with 2, the encoder is also cut between pieces (drafter, HC, QSA, GDN, attention,
/// router, MoE...), which adds a little overhead of its own. The ExpertStream thread records when it sees each
/// signal and when it answers, in CPU time. Everything is written per round to a JSONL (scripts/roundtrace.py
/// splits it up). Without the option, `roundTrace` is nil and the engine does not change.
public final class RoundTrace {
    /// What separates one encoder from the previous one: start of the round, signal to the CPU (`signalOnly`), wait
    /// for its answer (`waitAnswer`), notice (`notify`) or diagnostic cut (level 2).
    public enum Edge: String { case start, signal, wait, notify, cut }
    struct Enc {
        let label: String
        let layer: Int
        let edge: Edge
        let value: UInt64
    }
    struct Slot {
        let buf: Int
        var encs: [Enc] = []
        var cb: MTLCommandBuffer?
        /// Level 3: copied bitmaps (layer, rows, words per row, offset in `bmBufs[buf]` in words).
        var bms: [(layer: Int, rows: Int, nw: Int, off: Int, G: Int)] = []
    }

    public let level: Int
    public var cuts: Bool { level >= 2 && cur != nil }
    /// Level 3: also cuts inside the QSA (indexer, preparation, scoring, selection) and the attention
    /// (projection, prework, attention, reduction, o_proj), and copies the QSA bitmap of each layer to output in the
    /// JSONL how many words of 32 blocks (steps of 128 keys) the attention walks with real text.
    public var fine: Bool { level >= 3 && cur != nil }
    let device: MTLDevice
    let bufs: [MTLCounterSampleBuffer]
    static let maxEnc = 2048
    /// Level 3: copy of the round's QSA bitmaps (one per slot of `bufs`), created on first use.
    var bmBufs: [MTLBuffer] = []
    static let bmWords = 12 * 16 * 2048
    var cur: Slot?
    var inflight: [Slot] = []
    var nextBuf = 0
    let out: FileHandle
    /// GPU time (marks) <-> CPU time (mach_absolute_time) correlation and the mach-to-seconds step.
    var cpu0: MTLTimestamp = 0, gpu0: MTLTimestamp = 0
    let machToS: Double
    /// Units of the CPU time of `sampleTimestamps` in seconds (mach or ns depending on the system; checked on start).
    let cpuToS: Double
    public var label = ""
    public private(set) var rounds = 0

    public init(device: MTLDevice, level: Int, path: String) throws {
        guard let set = device.counterSets?.first(where: { $0.name == MTLCommonCounterSet.timestamp.rawValue }) else {
            throw SafetensorsError.io("round-trace: no timestamp counter")
        }
        guard device.supportsCounterSampling(.atStageBoundary) else {
            throw SafetensorsError.io("round-trace: the GPU does not support stage-boundary timestamps")
        }
        self.device = device
        self.level = level
        bufs = try (0..<4).map { _ in
            let d = MTLCounterSampleBufferDescriptor()
            d.counterSet = set
            d.sampleCount = 2 * Self.maxEnc
            d.storageMode = .shared
            return try device.makeCounterSampleBuffer(descriptor: d)
        }
        FileManager.default.createFile(atPath: path, contents: nil)
        guard let fh = FileHandle(forWritingAtPath: path) else { throw SafetensorsError.io("round-trace: \(path)") }
        out = fh
        var tb = mach_timebase_info_data_t()
        mach_timebase_info(&tb)
        machToS = Double(tb.numer) / Double(tb.denom) / 1e9
        (cpu0, gpu0) = device.sampleTimestamps()
        let now = Double(mach_absolute_time()) * machToS
        cpuToS = abs(Double(cpu0) * machToS - now) < 1 ? machToS : 1e-9
        print(
            String(
                format: "round-trace: level %d, Metal CPU time in %@ (%.3f s difference from mach)", level,
                cpuToS == machToS ? "mach units" : "ns", Double(cpu0) * cpuToS - now))
    }

    deinit { try? out.close() }

    /// First encoder of the round (starts recording).
    public func begin(_ cb: MTLCommandBuffer, label l: String) -> MTLComputeCommandEncoder {
        precondition(cur == nil && inflight.count < bufs.count)
        cur = Slot(buf: nextBuf, cb: cb)
        nextBuf = (nextBuf + 1) % bufs.count
        return make(cb, l, layer: -1, edge: .start, value: 0)!
    }

    /// New encoder with marks if `cb` is the round being recorded (otherwise nil: the caller creates a normal one).
    public func make(_ cb: MTLCommandBuffer, _ l: String, layer: Int, edge: Edge, value: UInt64 = 0)
        -> MTLComputeCommandEncoder?
    {
        guard var s = cur, s.cb === cb, s.encs.count < Self.maxEnc else { return nil }
        let i = s.encs.count
        let pd = MTLComputePassDescriptor()
        pd.sampleBufferAttachments[0].sampleBuffer = bufs[s.buf]
        pd.sampleBufferAttachments[0].startOfEncoderSampleIndex = 2 * i
        pd.sampleBufferAttachments[0].endOfEncoderSampleIndex = 2 * i + 1
        s.encs.append(Enc(label: l, layer: layer, edge: edge, value: value))
        cur = s
        return cb.makeComputeCommandEncoder(descriptor: pd)
    }

    /// Level 2: cuts the encoder before piece `l` (with level 1 or outside the recorded round, the same encoder).
    public func cut(_ cb: MTLCommandBuffer, _ e: MTLComputeCommandEncoder, _ l: String, layer: Int)
        -> MTLComputeCommandEncoder
    {
        guard level >= 2, let s = cur, s.cb === cb else { return e }
        e.endEncoding()
        return make(cb, l, layer: layer, edge: .cut)!
    }

    /// The round is already encoded (before `commit`).
    public func endEncode() {
        if let s = cur { inflight.append(s) }
        cur = nil
    }

    /// After `waitUntilCompleted` of the oldest round in flight: writes its line, with the marks of the
    /// ExpertStream thread for its jobs (event value, layer, mach on seeing the signal, mach on answering).
    public func finish(_ cb: MTLCommandBuffer, rows: Int, accepted: Int, P: Int, stream: ExpertStream?) {
        guard !inflight.isEmpty, inflight[0].cb === cb else { return }
        let s = inflight.removeFirst()
        let vals = s.encs.map(\.value).filter { $0 > 0 }
        let cpuMarks = stream?.takeTraceMarks(from: vals.min() ?? 1, upTo: vals.max() ?? 0) ?? []
        let n = s.encs.count
        guard n > 0, let data = try? bufs[s.buf].resolveCounterRange(0..<(2 * n)) else { return }
        let ts = data.withUnsafeBytes { Array($0.bindMemory(to: UInt64.self)) }
        let (c1, g1) = device.sampleTimestamps()
        // GPU -> mach (line through the two correlations; the M6 counter runs in the same units, but it is adjusted).
        let k = g1 > gpu0 ? Double(c1 - cpu0) / Double(g1 - gpu0) : 1
        func sec(gpu t: UInt64) -> Double { (Double(cpu0) + (Double(t) - Double(gpu0)) * k) * cpuToS }
        let base = cb.gpuStartTime
        func ms(_ t: Double) -> Double { ((t - base) * 1e6).rounded() / 1e3 }
        var encs: [[Any]] = []
        for (i, e) in s.encs.enumerated() {
            let a = ts[2 * i], b = ts[2 * i + 1]
            // An empty encoder may leave no mark (0 or ~0): it is recorded as -1.
            let ok = a != 0 && b != 0 && a != UInt64.max && b != UInt64.max
            encs.append([
                e.label, e.layer, e.edge.rawValue, ok ? ms(sec(gpu: a)) : -1, ok ? ms(sec(gpu: b)) : -1, Int(e.value),
            ])
        }
        let cpu = cpuMarks.map { [Int($0.0), $0.1, ms(Double($0.2) * machToS), ms(Double($0.3) * machToS)] as [Any] }
        var o: [String: Any] = [
            "label": label, "round": rounds, "level": level, "rows": rows, "accepted": accepted, "P": P,
            "gpu_ms": ((cb.gpuEndTime - cb.gpuStartTime) * 1e6).rounded() / 1e3, "cb_start_s": cb.gpuStartTime,
            "encs": encs, "cpu": cpu,
        ]
        if !s.bms.isEmpty { o["qsa_bm"] = bitmapStats(s, rows: rows, P: P) }
        rounds += 1
        if let d = try? JSONSerialization.data(withJSONObject: o) { out.write(d + Data("\n".utf8)) }
    }

    /// Level 3: copies the `rows` rows of the QSA bitmap of layer `layer` (`nw` words per row) for the round's line
    /// (`G`: queries per KV head of that layer). It goes in a blit without marks; the encoder that follows is called
    /// "after.bm" (empty: the next cut closes it).
    public func captureBitmap(
        _ cb: MTLCommandBuffer, _ e: MTLComputeCommandEncoder, _ bm: MTLBuffer, rows: Int, nw: Int, layer: Int,
        G: Int
    ) -> MTLComputeCommandEncoder {
        guard level >= 3, var s = cur, s.cb === cb else { return e }
        if bmBufs.isEmpty {
            bmBufs = (0..<bufs.count).map { _ in
                device.makeBuffer(length: Self.bmWords * 4, options: .storageModeShared)!
            }
        }
        let off = s.bms.last.map { $0.off + $0.rows * $0.nw } ?? 0
        guard off + rows * nw <= Self.bmWords else { return e }
        s.bms.append((layer, rows, nw, off, G))
        cur = s
        e.endEncoding()
        let b = cb.makeBlitCommandEncoder()!
        b.copy(from: bm, sourceOffset: 0, to: bmBufs[s.buf], destinationOffset: off * 4, size: rows * nw * 4)
        b.endEncoding()
        return make(cb, "after.bm", layer: layer, edge: .cut)!
    }

    /// Per layer: [layer, rows, closed words of row 0, bits per row (mean), words with some bit per row
    /// (mean), steps walked per KV head (sum over the threadgroups of dattn_i8s, closed words with some
    /// bit in the OR of their rows, plus those of the tail), empty 16-key fragments inside those closed steps,
    /// words with some bit in the OR of all rows]. Threadgroups as in AttentionLayer.encode (groups of 16
    /// queries, RG = 2).
    func bitmapStats(_ s: Slot, rows R: Int, P: Int) -> [[Int]] {
        let w = bmBufs[s.buf].contents().bindMemory(to: UInt32.self, capacity: Self.bmWords)
        var out: [[Int]] = []
        for c in s.bms {
            let rr = min(R, c.rows), Q = c.G * rr, groups = (Q + 15) / 16, rg = min(groups, 2),
                nz = (groups + rg - 1) / rg
            var bits = 0, act = 0, steps = 0, empty16 = 0
            for r in 0..<rr {
                for k in 0..<c.nw {
                    let x = w[c.off + r * c.nw + k]
                    bits += x.nonzeroBitCount
                    if x != 0 { act += 1 }
                }
            }
            for z in 0..<nz {
                let row0 = z * 16 * rg, rlast = min(row0 + 16 * rg - 1, Q - 1), sr0 = row0 / c.G, sr1 = rlast / c.G
                let sclosed = ((P + sr0 + 1) / 4) * 4, kend = P + rr
                var k0 = 0
                while k0 < kend {
                    let wi = k0 / 128
                    if k0 + 128 <= sclosed {
                        var any: UInt32 = 0
                        for r in sr0...sr1 where wi < c.nw { any |= w[c.off + r * c.nw + wi] }
                        if any != 0 {
                            steps += 1
                            for f in 0..<8 where (any >> (4 * f)) & 0xF == 0 { empty16 += 1 }
                        }
                    } else {
                        steps += 1
                    }
                    k0 += 128
                }
            }
            var uni = 0
            for k in 0..<c.nw {
                var any: UInt32 = 0
                for r in 0..<rr { any |= w[c.off + r * c.nw + k] }
                if any != 0 { uni += 1 }
            }
            out.append([c.layer, rr, ((P + 1) / 4) / 32, bits / max(rr, 1), act / max(rr, 1), steps, empty16, uni])
        }
        return out
    }

    /// Rounds that were encoded and are not recorded (the extra one that is rolled back at the end).
    public func drop(_ cb: MTLCommandBuffer) {
        if !inflight.isEmpty, inflight[0].cb === cb { inflight.removeFirst() }
    }
}

extension HybridModel {
    /// Diagnostic cut before piece `l` (only with `--round-trace 2` and in the round being recorded).
    @inline(__always) func tcut(_ cb: MTLCommandBuffer, _ e: MTLComputeCommandEncoder, _ l: String, _ layer: Int)
        -> MTLComputeCommandEncoder
    {
        guard let t = roundTrace, t.cuts else { return e }
        return t.cut(cb, e, l, layer: layer)
    }
}

extension HybridModel.Mixer {
    /// Name of the piece in the `--round-trace` timeline.
    var traceName: String {
        switch self {
        case .gdn: return "gdn"
        case .attn: return "attn"
        case .conv: return "conv"
        case .none: return "none"
        }
    }
}
