import Foundation
import Testing

@testable import Ingot

/// RoPE frequency tables against mlx-lm's `initialize_rope` (ref/rope_ref.py → ref/rope/rope_ref.json:
/// yarn-truncate, gpt-oss, yarn-mscale, linear, default). No GPU.
@Suite struct RopeTableTests {
    static let refURL = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        .deletingLastPathComponent().appendingPathComponent("ref/rope/rope_ref.json")

    /// Distance in units in the last place between two finite floats of the same sign.
    static func ulps(_ a: Float, _ b: Float) -> Int {
        abs(Int(Int32(bitPattern: a.bitPattern)) - Int(Int32(bitPattern: b.bitPattern)))
    }

    @Test func matchesMlxLm() throws {
        let data = try Data(contentsOf: Self.refURL)
        let rows = try #require(try JSONSerialization.jsonObject(with: data) as? [[String: Any]])
        #expect(rows.count >= 5)
        for row in rows {
            let name = try #require(row["name"] as? String)
            let theta = try #require((row["theta"] as? NSNumber)?.doubleValue)
            let dims = try #require((row["dims"] as? NSNumber)?.intValue)
            let params = try #require(row["params"] as? [String: Any])
            let want = try #require(row["inv_freq"] as? [NSNumber]).map(\.floatValue)
            let want64 = try #require(row["inv_freq_f64"] as? [NSNumber]).map(\.floatValue)
            let mscale = try #require((row["mscale"] as? NSNumber)?.floatValue)
            let tab = RopeTable.make(theta: theta, rotaryDims: dims, scaling: try RopeScaling.parse(params))
            try #require(tab.count == dims / 2 + 1, "\(name): table size")
            // Same formula in float64 (rounded once, like ours): at most 1 ulp. mlx-lm rounds every float32 step (its
            // pow of a large base is off by a few ulp): at most 8 (3 from float64 in yarn-truncate).
            let worst64 = zip(tab, want64).map { Self.ulps($0, $1) }.max() ?? 0
            #expect(worst64 <= 1, "\(name): inv_freq off by \(worst64) ulp from the float64 formula")
            let worst = zip(tab, want).map { Self.ulps($0, $1) }.max() ?? 0
            #expect(worst <= 8, "\(name): inv_freq off by \(worst) ulp from mlx-lm")
            #expect(Self.ulps(tab[dims / 2], mscale) <= 1, "\(name): mscale \(tab[dims / 2]) vs \(mscale)")
        }
    }

    @Test func defaultTableIsTheKernelFormula() throws {
        // theta^(-i / rh) and mscale 1: the table of the unscaled layers of a model with LAB_AT_ROPETAB.
        let tab = RopeTable.make(theta: 1e7, rotaryDims: 64, scaling: nil)
        #expect(tab.count == 33 && tab[32] == 1 && tab[0] == 1)
        for i in 0..<32 { #expect(Self.ulps(tab[i], Float(pow(1e7, -Double(i) / 32))) <= 1) }
    }

    /// Gemma 4's full layers (mlx-lm ProportionalRoPE, dims 512, partial 0.25, θ 1e6): the first 64 pairs with
    /// period θ^(2i/512) (× factor), the other 192 with angle 0; mscale 1.
    @Test func proportionalTable() throws {
        let s = try #require(try RopeScaling.parse(["rope_type": "proportional", "partial_rotary_factor": 0.25]))
        #expect(s.kind == .proportional && s.factor == 1 && s.partial == 0.25)
        let tab = RopeTable.make(theta: 1e6, rotaryDims: 512, scaling: s)
        #expect(tab.count == 257 && tab[256] == 1)
        for i in 0..<64 { #expect(Self.ulps(tab[i], Float(1 / pow(1e6, Double(2 * i) / 512))) <= 1, "pair \(i)") }
        #expect(tab[64...255].allSatisfy { $0 == 0 })
        let f2 = try #require(try RopeScaling.parse(["rope_type": "proportional", "factor": 2.0]))
        let t2 = RopeTable.make(theta: 1e4, rotaryDims: 256, scaling: f2)
        for i in 0..<128 { #expect(Self.ulps(t2[i], Float(1 / (2 * pow(1e4, Double(2 * i) / 256)))) <= 1) }
    }

    @Test func unsupportedTypesThrow() throws {
        #expect(try RopeScaling.parse(["rope_type": "default", "rope_theta": 1e4]) == nil)
        #expect(try RopeScaling.parse(["type": "linear", "factor": 2.0])?.kind == .linear)
        for bad: [String: Any] in [
            ["rope_type": "dynamic", "factor": 2.0],
            ["rope_type": "longrope", "short_factor": [1.0], "long_factor": [1.0]],
            ["rope_type": "proportional", "partial_rotary_factor": 1.5],
            // llama3: only Llama (tier 2) uses it; removed.
            ["rope_type": "llama3", "factor": 8.0, "original_max_position_embeddings": 8192],
            ["rope_type": "yarn", "factor": 4.0],  // no original_max_position_embeddings
            ["rope_type": "yarn", "factor": 40.0, "original_max_position_embeddings": 4096, "mscale_all_dim": 1.0],
            ["rope_type": "linear"],
        ] {
            #expect(throws: SafetensorsError.self) { try RopeScaling.parse(bad) }
        }
    }
}
