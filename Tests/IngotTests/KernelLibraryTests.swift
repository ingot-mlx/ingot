import Foundation
import Metal
import Testing

@testable import Ingot

/// Every kernel instance of the engine must be able to create its pipeline with the dimensions of each supported
/// model (threadgroup memory ≤ `maxThreadgroupMemoryLength`, 32 KB on the M6). It runs nothing on the GPU: it only
/// compiles. It goes with the GPU tests (scripts/ci.sh --toy) because it uses the device.
@Suite struct KernelLibraryTests {

    /// The shape of each model, from its ModelSpec (the configs of ModelSpecTests).
    static func dims(_ name: String) throws -> ModelDims {
        // The router cases against mlx-lm (KernelCaseRouter.swift) compile their own router.
        if name.hasPrefix("router:") { return RouterVector.named(String(name.dropFirst(7))).dims }
        let cfg: ModelConfig
        switch name {
        case "qwen3.8-27b": cfg = try ModelSpecTests.q27()
        // Qwen3.5 0.8B: hidden 1024, GDN 16 k × 16 v (GDN_R = 1) and 8 q / 2 KV gated heads.
        case "qwen3.5-0.8b": cfg = try ModelSpecTests.q35small()
        case "bonsai-2-27b": cfg = try ModelSpecTests.bonsai27()
        case "qwen3.8-flash-next": cfg = try ModelSpecTests.flashNext()
        case "lfm2.5-24b-a2b": cfg = try ModelSpecTests.lfm2()
        case "lfm2.5-8b-a1b": cfg = try ModelSpecTests.lfm25_8b()
        // Granite 4.1 3B (ungated, no QK-norm, head 64, G = 5, g32 from its config: LAB_G32); -qkvb below adds the
        // q|k|v biases (toy-granite).
        case "granite-4.1-3b-qkvb": cfg = try ModelSpecTests.granite()
        // Muse Glimmer 30B: head 128 with full RoPE (θ 5e5) and NoPE layers, GQA 32/2, the gate (its gate_proj
        // interleaved at load), weightless QK-norm and the sliding window of 2048 with a ring KV (LAB_AT_SWA).
        case "muse-glimmer-30b": cfg = try ModelSpecTests.muse()
        // LFM3-1.7B: the short convolution of 7 taps (LAB_SC_L), head 128 with the scalar gate per head (repeated into
        // the gated q row), the zero-tail RoPE table (LAB_AT_ROPETAB) and the window of 256 with a ring KV.
        case "lfm3-1.7b": cfg = try ModelSpecTests.lfm3()
        // Gemma 4 31B: sliding layers (head 256, 16 kv, v_norm, RoPE table, window) and, with -full, its full layers'
        // own shape (head 512, 4 kv of K = V, proportional RoPE table).
        case "gemma4-31b", "gemma4-31b-full": cfg = try ModelSpecTests.gemma4("gemma4-31b")
        // gpt-oss 20b (MXFP4-Q8's quantization: the `_mxfp4` expert instances, LAB_MXFP4): router and expert biases,
        // swiglu_oai, hidden 2880 (LAB_MOE_H64), 64 q heads of 64, learned sinks (LAB_AT_SINKS), window and YaRN table
        // (LAB_AT_ROPETAB). The sliding window with a ring KV (LAB_AT_SWA) is also Gemma 4's (gemma4-31b).
        case "gpt-oss-20b": cfg = try ModelSpecTests.gptOss(quantization: ModelSpecTests.mxfp4Q8)
        // Nemotron-H: Mamba2 in the GDN kernels (LAB_SSM, LAB_GDN_DV), the Lightning's 64 heads × 64 and the Nano 4B's
        // 96 × 80, and relu² without gate (LAB_FFN_RELU2) in the Lightning's MoE and the Nano's dense MLP (hidden 3136).
        case "nemotron-3.5-lightning": cfg = try ModelSpecTests.lightning()
        case "nemotron-3-nano-4b": cfg = try ModelSpecTests.nano()
        // GLM-4.7-Flash: MLA's latent head (576, one KV head for 20 query heads) and the per-head matmuls (zm_mla.metal).
        case "glm-4.7-flash": cfg = try ModelSpecTests.glm()
        default: cfg = try ModelSpecTests.heretic()  // qwen3.6-35b-a3b (Heretic, Ornith)
        }
        let spec = try ModelSpec(cfg)
        var shape = name.hasSuffix("-full") ? try #require(spec.fullShape) : spec.shape
        // The `_gelu` instances of the gated FFN kernels (LAB_FFN_GELU).
        if name.contains("-gelu") { shape.ffnGelu = true }
        // q|k|v biases in the prework (LAB_AT_QKVB; ungated attention only).
        if name.contains("-qkvb") { shape.attnQKVB = true }
        return shape
    }

    @Test(arguments: [
        "qwen3.6-35b-a3b", "qwen3.8-27b", "lfm2.5-24b-a2b", "bonsai-2-27b", "qwen3.8-flash-next", "muse-glimmer-30b",
        "lfm2.5-8b-a1b", "qwen3.6-gelu",
        "granite-4.1-3b-qkvb", "qwen3.5-0.8b", "gemma4-31b", "gemma4-31b-full",
        "gpt-oss-20b", "nemotron-3.5-lightning", "nemotron-3-nano-4b", "glm-4.7-flash", "lfm3-1.7b",
    ])
    func everyKernelInstanceBuilds(model: String) throws {
        let ctx = try MetalContext(dims: try Self.dims(model))
        var broken: Set<String> = [], count = 0
        for name in ctx.library.functionNames {
            guard let f = ctx.library.makeFunction(name: name), f.functionType == .kernel else { continue }
            count += 1
            do {
                let p = try ctx.device.makeComputePipelineState(function: f)
                #expect(p.staticThreadgroupMemoryLength <= ctx.device.maxThreadgroupMemoryLength, "\(name)")
            } catch {
                broken.insert(name)
            }
        }
        #expect(count > 200)
        #expect(broken.isEmpty, "instances that do not create their pipeline: \(broken.sorted())")
    }
}
