import Foundation
import Metal

// SSD-streamed experts, phase 1: trace of the MoE routing in each decode round, to simulate offline an
// expert cache (LRU, ARC, frequency, read-ahead) with the real rows of each round (those chosen by the row selector or
// those of a copy round). Diagnostic only: with the trace on, rounds are not chained (each waits for the
// previous one) and each layer also computes the router of the next layer; tokens do not change.
//
// Little-endian binary file, a sequence of records:
//   session: u32 'RTS1', u32 seq, u32 n, n bytes of UTF-8 label (zero-padded up to a multiple of 4)
//   round:   u32 'RTR1', u32 seq, u32 P (anchor position), u32 routed rows, u32 accepted (m),
//            u32 flags (bit 0: copy round; bit 1: with sampling), u32 L (MoE layers), u32 topk;
//            then, per MoE layer and per row: u8 ids[topk] (by rank), f16 weights[topk] (renormalized softmax),
//            u8 pred[16] (top-16 of that layer's router applied to the previous layer's FFN input, by logit;
//            0xFF in the first layer).
// Consumed by an offline cache simulator.
public final class RouteTrace {
    public static let magicSession: UInt32 = 0x3153_5452  // "RTS1"
    public static let magicRound: UInt32 = 0x3152_5452  // "RTR1"
    public static let predK = 16

    let fh: FileHandle
    /// Per layer: fp32 logits [16, E] of the layer's router over the previous layer's FFN input.
    let pred: [MTLBuffer?]
    var seq: UInt32 = 0
    public private(set) var rounds = 0

    public init(path: String, model m: HybridModel) throws {
        if !FileManager.default.fileExists(atPath: path) {
            FileManager.default.createFile(atPath: path, contents: nil)
        }
        fh = try FileHandle(forWritingTo: URL(fileURLWithPath: path))
        fh.seekToEndOfFile()
        pred = m.layers.enumerated().map { i, l in
            guard i > 0, let moe = l.moeOpt else { return nil }
            return m.ctx.buffer(length: MoELayer.maxRows * moe.experts * 4)
        }
    }

    deinit { try? fh.close() }

    /// Label of the next session (started by each decodeSpec; without a label, "decode").
    public var nextLabel: String?

    /// Starts a session (a lab-dev gen prompt or a lab serve request).
    public func begin(label: String) {
        seq += 1
        var d = Data()
        let b = Array(label.utf8)
        for v in [Self.magicSession, seq, UInt32(b.count)] {
            withUnsafeBytes(of: v.littleEndian) { d.append(contentsOf: $0) }
        }
        d.append(contentsOf: b)
        d.append(contentsOf: [UInt8](repeating: 0, count: (4 - b.count % 4) % 4))
        fh.write(d)
    }

    /// In encodeVerify, after the norm of layer i's FFN: router of layer i + 1 over that input.
    func encodePrediction(_ e: MTLComputeCommandEncoder, model m: HybridModel, layer i: Int, x: MTLBuffer, rows: Int)
        throws
    {
        guard i + 1 < m.layers.count, let next = m.layers[i + 1].moeOpt, let pb = pred[i + 1] else { return }
        try next.vq.encode(e, x: x, next.router, y: pb, rows: rows)
    }

    /// After completing a round (the GPU no longer writes to the MoE buffers): `rows` routed rows.
    func record(model m: HybridModel, P: Int, rows: Int, accepted: Int, copy: Bool, sampled: Bool) {
        let moes = m.layers.compactMap(\.moeOpt)
        guard let first = moes.first else { return }
        let topk = first.topk, E = first.experts
        var d = Data()
        d.reserveCapacity(36 + moes.count * rows * (topk * 3 + Self.predK))
        let hdr: [UInt32] = [
            Self.magicRound, seq, UInt32(P), UInt32(rows), UInt32(accepted),
            (copy ? 1 : 0) | (sampled ? 2 : 0),
            UInt32(moes.count), UInt32(topk),
        ]
        for v in hdr { withUnsafeBytes(of: v.littleEndian) { d.append(contentsOf: $0) } }
        var order = [Int](0..<E)
        for (i, l) in m.layers.enumerated() {
            guard let moe = l.moeOpt else { continue }
            let ip = moe.inds.contents().bindMemory(to: UInt32.self, capacity: MoELayer.maxRows * topk)
            let wp = moe.wts.contents().bindMemory(to: Float.self, capacity: MoELayer.maxRows * topk)
            let pp = pred[i].map { $0.contents().bindMemory(to: Float.self, capacity: MoELayer.maxRows * E) }
            for r in 0..<rows {
                for k in 0..<topk { d.append(UInt8(truncatingIfNeeded: ip[r * topk + k])) }
                for k in 0..<topk {
                    withUnsafeBytes(of: Float16(wp[r * topk + k]).bitPattern.littleEndian) { d.append(contentsOf: $0) }
                }
                if let pp {
                    let base = r * E
                    order.sort { pp[base + $0] > pp[base + $1] || (pp[base + $0] == pp[base + $1] && $0 < $1) }
                    for k in 0..<Self.predK { d.append(UInt8(order[k])) }
                } else {
                    d.append(contentsOf: [UInt8](repeating: 0xFF, count: Self.predK))
                }
            }
        }
        fh.write(d)
        rounds += 1
    }
}
