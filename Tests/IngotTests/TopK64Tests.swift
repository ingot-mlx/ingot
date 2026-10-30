import Foundation
import Metal
import Testing

@testable import Ingot

/// TopK64 (topk64_part + topk64_merge): the 64 candidates, bit for bit, are those of a full sort of the row (value
/// descending, ties by lower id), the max is exact and Z is Σ exp((x − max)/T) to float rounding.
@Suite struct TopK64Tests {
    static let K = SamplingParams.candidates

    enum Shape: String, CaseIterable {
        case random, ties, ascending, descending, negInf
    }

    /// Row r of `rows` rows of `V` logits with the given shape.
    static func logits(_ shape: Shape, V: Int, rows: Int, seed: UInt64) -> [Float] {
        var g = SplitMix64(seed: seed)
        var x = [Float](repeating: 0, count: rows * V)
        for r in 0..<rows {
            for i in 0..<V {
                let u = g.uniform()
                switch shape {
                case .random: x[r * V + i] = u * 30 - 15
                case .ties: x[r * V + i] = Float(Int(u * 12)) * 0.5  // 12 levels: many ties at the cut
                case .ascending: x[r * V + i] = Float(i) * 1e-3 + Float(r)
                case .descending: x[r * V + i] = -Float(i) * 1e-3 - Float(r)
                case .negInf: x[r * V + i] = u < 0.5 ? -.infinity : u * 8
                }
            }
        }
        return x
    }

    static func check(_ x: [Float], V: Int, rows: Int, params: [SamplingParams]) throws {
        let ctx = try MetalContext()
        let kp = params.map { SampleKernelParams($0, vocab: V) }
        let lb = ctx.buffer(x), idx = ctx.buffer(length: rows * K * 4), val = ctx.buffer(length: rows * K * 4)
        let z = ctx.buffer(length: rows * 8)
        try ctx.run { e in
            try TopK64.encode(
                e, ctx: ctx, x: lb, xOffset: 0, rows: rows, params: kp, out: (idx, val, z),
                scratch: ctx.buffer(length: TopK64.scratchLength(rows: rows)))
        }
        let ip = idx.contents().bindMemory(to: UInt32.self, capacity: rows * K)
        let vp = val.contents().bindMemory(to: Float.self, capacity: rows * K)
        let zp = z.contents().bindMemory(to: Float.self, capacity: rows * 2)
        for r in 0..<rows {
            let row = Array(x[(r * V)..<((r + 1) * V)])
            // Candidates as the kernel defines them: x > -FLT_MAX or (x == -FLT_MAX), padded with (-FLT_MAX, ~0).
            let order = row.indices.filter { row[$0] >= -Float.greatestFiniteMagnitude }.sorted {
                row[$0] > row[$1] || (row[$0] == row[$1] && $0 < $1)
            }
            var wantI = order.prefix(K).map { UInt32($0) }, wantV = order.prefix(K).map { row[$0] }
            while wantI.count < K { wantI.append(0xffff_ffff); wantV.append(-.greatestFiniteMagnitude) }
            let gotI = (0..<K).map { ip[r * K + $0] }, gotV = (0..<K).map { vp[r * K + $0] }
            #expect(gotI == wantI, "row \(r): ids")
            #expect(gotV.map(\.bitPattern) == wantV.map(\.bitPattern), "row \(r): values")
            let mx = row.max()!
            #expect(zp[2 * r] == mx, "row \(r): max")
            let it = 1 / Double(params[params.count > 1 ? r : 0].temperature)
            let Z = row.reduce(0.0) { $0 + exp(Double($1 - mx) * it) }
            #expect(abs(Double(zp[2 * r + 1]) / Z - 1) < 1e-5, "row \(r): Z \(zp[2 * r + 1]) vs \(Z)")
        }
    }

    /// A 262K vocabulary (Gemma 4) and smaller ones, down to fewer than 64 tokens (padding) and fewer than one per
    /// chunk.
    @Test(arguments: Shape.allCases)
    func candidatesMatchFullSort(shape: Shape) throws {
        for (V, rows) in [(262_144, 1), (151_936, 3), (3000, 2), (100, 1), (40, 2), (7, 1)] {
            let x = Self.logits(shape, V: V, rows: rows, seed: UInt64(V))
            try Self.check(x, V: V, rows: rows, params: [SamplingParams(temperature: 0.7)])
        }
    }

    /// The `_each` variant: one temperature per row.
    @Test func eachRowItsOwnParams() throws {
        let V = 50_000, rows = 4
        let x = Self.logits(.random, V: V, rows: rows, seed: 9)
        let ps = [0.3, 1, 1.5, 0.8].map { SamplingParams(temperature: Float($0)) }
        try Self.check(x, V: V, rows: rows, params: ps)
    }
}
