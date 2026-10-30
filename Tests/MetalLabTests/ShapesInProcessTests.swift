import Foundation
import Metal
import Testing

@testable import MetalLab

/// The shape goes through MetalContext (no global ModelDims), so layers of different shapes coexist in one
/// process. Creates the attention layers of three shapes (head 256 with gate and GQA 8, LFM2.5's 64 without gate
/// and GQA 4, and Muse Glimmer's 128 with full RoPE, GQA 16 and a sliding window) before launching any, then launches each one (also in reverse order)
/// against its CPU reference. If one shape overwrote another, the layer created earlier would give
/// sizes or results of the last one. Goes with KernelLibraryTests (GPU tests).
extension KernelLibraryTests {
    @Test func severalShapesInOneProcess() throws {
        let models = ["qwen3.6-35b-a3b", "lfm2.5-24b-a2b", "muse-glimmer-30b"]
        var g = SplitMix64(seed: 2026_10_24)
        let fixtures = try models.map { m -> (String, AttnFixture) in
            let ctx = try MetalContext(dims: try Self.dims(m))
            return (m, try AttnFixture(ctx, positions: 300, &g))
        }
        #expect(Set(fixtures.map { $0.1.ctx.dims }).count == models.count)
        // The library is shared between contexts of the same shape (per-shape cache) and not between different shapes.
        let again = try MetalContext(dims: try Self.dims(models[0]))
        #expect(again.library === fixtures[0].1.ctx.library)
        #expect(fixtures[1].1.ctx.library !== fixtures[0].1.ctx.library)
        for (m, f) in fixtures + fixtures.reversed() {
            #expect(f.a.hd == f.ctx.dims.attnHD && f.a.kvHeads == f.ctx.dims.attnKVH && f.a.G == f.ctx.dims.attnG)
            let x = Syn.roundHalf(Syn.normal(&g, 4 * f.hidden))
            let r = Check(
                label: "\(m): round of 4 rows at 300", gpu: try f.round(x, rows: 4, P: 300),
                ref: f.reference(x, rows: 4, P: 300), tol: .maxRel(5e-3)
            ).evaluate()
            print("shapes \(r.text)")
            #expect(r.ok, "\(m): \(r.text)")
        }
    }
}
