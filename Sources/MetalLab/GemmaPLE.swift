import Foundation
import Metal

/// Per-layer inputs of Gemma 4 E2B/E4B (PLE; mlx-lm gemma4_text `_get_per_layer_inputs`, `_project_per_layer_inputs`
/// and the gate at the end of each layer). Per row and layer, n (256) values
///
///     ple = (N(W_p · e · hidden^-0.5) + T[token] · √n) · 2^-0.5
///
/// with e the scaled embedding (the residual stream right after the embedding), W_p `per_layer_model_projection`, N
/// the RMSNorm `per_layer_projection_norm` over each layer's n values and T `embed_tokens_per_layer` (4 bits, rows of
/// layers · n). Computed once per round or prefill chunk, in fp32 (mlx-lm runs it in bf16; the CI's reference is mlx-lm
/// fp32). At the end of layer i, after the FFN epilogue, the PLE branch
///
///     h = (h + N_i(W_o · (gelu_tanh(W_g · h) ⊙ ple_i))) · s_i
///
/// (`per_layer_input_gate`, `per_layer_projection`, `post_per_layer_input_norm`) carries the layer scalar s_i, which
/// then is not applied in the FFN epilogue (HybridModel.encodePLEBranch).
public final class GemmaPLE {
    let ctx: MetalContext
    /// Values per layer and layers (a table row holds layers · n).
    public let n: Int, layers: Int
    let hidden: Int, eps: Float
    let table, proj: DenseQ
    let projNorm: MTLBuffer
    /// Per layer: gate (hidden → n), projection (n → hidden) and post-norm weight.
    let gates: [DenseQ], projs: [DenseQ]
    public let postNorms: [MTLBuffer]
    let vq: VQMV

    /// Work of the round (16 rows) or of a prefill chunk: the inputs and the projection [rows, layers · n] fp32, the
    /// gate [rows, n] fp32 and the gated values [rows, n] half.
    struct Work { let ple, proj, gate, gated: MTLBuffer }
    let round, prefill: Work

    /// `root`: prefix of the model's tensors ("language_model.model."); `rows`: rows of a prefill chunk.
    init(ctx: MetalContext, store: SafetensorsStore, config cfg: ModelConfig, root: String, rows chunkRows: Int) throws
    {
        self.ctx = ctx
        let sp = try cfg.requiredSpec, d = ctx.device
        n = sp.pleDim; layers = cfg.numLayers; hidden = cfg.hiddenSize; eps = Float(cfg.rmsNormEps)
        vq = VQMV(ctx)
        table = try DenseQ(store: store, config: cfg, paths: [root + "embed_tokens_per_layer"], device: d)
        proj = try DenseQ(store: store, config: cfg, paths: [root + "per_layer_model_projection"], device: d)
        guard table.K == layers * n, table.N >= cfg.vocabSize, table.bits == 4, proj.N == layers * n,
            proj.K == hidden, proj.bits == 4
        else {
            throw SafetensorsError.format(
                "PLE: table \(table.N) × \(table.K) (\(table.bits) bits) and projection \(proj.N) × \(proj.K) "
                    + "(expected ≥ \(cfg.vocabSize) × \(layers * n) and \(layers * n) × \(hidden), 4 bits)")
        }
        let f32 = ctx.dims.normF32
        projNorm = try plainNormTensor(store, root + "per_layer_projection_norm.weight", d, f32: f32)
        var g: [DenseQ] = [], p: [DenseQ] = [], w: [MTLBuffer] = []
        for i in 0..<layers {
            let pre = try store.layerPrefix(i)
            g.append(try DenseQ(store: store, config: cfg, paths: [pre + "per_layer_input_gate"], device: d))
            p.append(try DenseQ(store: store, config: cfg, paths: [pre + "per_layer_projection"], device: d))
            w.append(try plainNormTensor(store, pre + "post_per_layer_input_norm.weight", d, f32: f32))
            guard g[i].N == n, g[i].K == hidden, p[i].N == hidden, p[i].K == n, g[i].bits == 4, p[i].bits == 4 else {
                throw SafetensorsError.format("\(pre): PLE gate/projection shapes (expected \(n) × \(hidden), 4 bits)")
            }
        }
        gates = g; projs = p; postNorms = w
        let width = layers * n, pn = n
        func work(_ r: Int) -> Work {
            Work(
                ple: ctx.buffer(length: r * width * 4), proj: ctx.buffer(length: r * width * 4),
                gate: ctx.buffer(length: r * pn * 4), gated: ctx.buffer(length: r * pn * 2))
        }
        round = work(HybridModel.decodeRows)
        prefill = work(chunkRows)
    }

    public var weightBytes: Int {
        table.bytes + proj.bytes + gates.reduce(0) { $0 + $1.bytes } + projs.reduce(0) { $0 + $1.bytes }
    }

    /// PLEParams of xb_block.metal.
    private func params(layer: Int) -> (UInt32, UInt32, UInt32, Float, Float, Float, Float) {
        (
            UInt32(n), UInt32(layers), UInt32(layer), eps, Float(1 / Double(hidden).squareRoot()),
            Float(Double(n).squareRoot()), Float(1 / 2.0.squareRoot())
        )
    }

    /// half copy of `count` fp32 values (the matrices take half inputs).
    private func cast(_ e: MTLComputeCommandEncoder, _ x: MTLBuffer, _ y: MTLBuffer, count: Int) throws {
        var c = UInt32(count)
        e.setComputePipelineState(try ctx.pipeline("cast_rows_h"))
        e.setBuffer(x, offset: 0, index: 0)
        e.setBuffer(y, offset: 0, index: 1)
        e.setBytes(&c, length: 4, index: 2)
        e.dispatchThreads(
            MTLSize(width: count, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    /// The inputs of `rows` rows: tokens of `tok` (from byte `tokOffset`) and the residual stream `resid` right after the
    /// embedding; `xh`: a half scratch of rows × hidden. `prefill`: the chunk's work (otherwise the round's).
    func encodeInputs(
        _ e: MTLComputeCommandEncoder, tok: MTLBuffer, tokOffset: Int, resid: MTLBuffer, xh: MTLBuffer, rows: Int,
        prefill pf: Bool
    ) throws {
        let w = pf ? prefill : round
        try table.encodeEmbed(ctx, e, tok: tok, tokOffset: tokOffset, outF: w.ple, rows: rows)
        try cast(e, resid, xh, count: rows * hidden)
        if pf {
            try proj.encodePrefillGemm(ctx, e, x: xh, y: w.proj, rows: rows, i8: nil)
        } else {
            try vq.encode(e, x: xh, proj, y: w.proj, rows: rows)
        }
        var p = params(layer: 0)
        e.setComputePipelineState(try ctx.pipeline("ple_inputs"))
        for (i, b) in [w.ple, w.proj, projNorm].enumerated() { e.setBuffer(b, offset: 0, index: i) }
        e.setBytes(&p, length: MemoryLayout.size(ofValue: p), index: 3)
        e.dispatchThreadgroups(
            MTLSize(width: layers, height: rows, depth: 1),
            threadsPerThreadgroup: MTLSize(width: n, height: 1, depth: 1))
    }

    /// Layer i's PLE branch over `resid` (`rows` rows) added into `out` (the branch buffer, at zero); the caller adds
    /// it to the residual stream with the post-norm and the layer scalar. `xh`: a half scratch of rows × hidden.
    func encodeLayer(
        _ e: MTLComputeCommandEncoder, layer i: Int, resid: MTLBuffer, xh: MTLBuffer, out: MTLBuffer, rows: Int,
        prefill pf: Bool
    ) throws {
        let w = pf ? prefill : round
        try cast(e, resid, xh, count: rows * hidden)
        if pf {
            try gates[i].encodePrefillGemm(ctx, e, x: xh, y: w.gate, rows: rows, i8: nil)
        } else {
            try vq.encode(e, x: xh, gates[i], y: w.gate, rows: rows)
        }
        var p = params(layer: i)
        e.setComputePipelineState(try ctx.pipeline("ple_gate_rows"))
        for (k, b) in [w.gate, w.ple, w.gated].enumerated() { e.setBuffer(b, offset: 0, index: k) }
        e.setBytes(&p, length: MemoryLayout.size(ofValue: p), index: 3)
        e.dispatchThreads(
            MTLSize(width: rows * n, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        if pf {
            try projs[i].encodePrefillGemm(ctx, e, x: w.gated, y: out, rows: rows, i8: nil, accumulate: true)
        } else {
            try vq.encode(e, x: w.gated, projs[i], y: out, rows: rows, accumulate: true)
        }
    }
}
