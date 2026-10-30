import Foundation
import Metal

// The block pieces around the layers that only run when the config asks for them (BlockSpec): branch
// epilogue (post-norm, residual multiplier, o_proj bias, layer scalar), embedding scale and norm, and logits
// post-processing. With the block of the current families none of them is dispatched (xb_block.metal).

extension HybridModel {
    /// Where a branch (mixer or FFN) of layer `l` accumulates: the residual stream itself, or the branch buffer that
    /// `encodeBranchEpilogue` adds to it.
    func branchOut(_ l: Layer, resid: MTLBuffer, prefill: Bool) -> MTLBuffer {
        l.post == nil ? resid : (prefill ? pbranch! : branch!)
    }

    /// After a branch of layer `l` accumulated into its branch buffer: resid = (resid + m · N(t + b)) · s per row
    /// (`branch_add_rows`; the buffer is left at zero). `ffn`: the FFN branch (its post-norm, and the layer scalar);
    /// otherwise the mixer (its post-norm and the o_proj bias). Does nothing for layers without an epilogue.
    func encodeBranchEpilogue(
        _ e: MTLComputeCommandEncoder, _ l: Layer, resid: MTLBuffer, rows: Int, ffn: Bool, prefill: Bool
    ) throws {
        guard let post = l.post else { return }
        let b = spec.block
        let normSpec = ffn ? b.ffnPostNorm : b.mixerPostNorm
        // With PLE the layer scalar goes after the PLE branch (encodePLEBranch), the last one of the layer.
        try encodeBranchAdd(
            e, resid: resid, rows: rows, prefill: prefill,
            norm: (ffn ? post.ffnNorm : post.mixerNorm, normSpec?.eps ?? eps), bias: ffn ? nil : post.oBias,
            mul: b.residualScale, scale: ffn && ple == nil ? post.scalar : 1)
    }

    /// Layer i's PLE branch (Gemma 4 E2B/E4B, GemmaPLE), after the FFN epilogue: gate and projection into the branch
    /// buffer, then resid = (resid + N(t)) · s with `post_per_layer_input_norm` and the layer scalar (mlx-lm adds the
    /// gate without the residual multiplier). Does nothing without PLE.
    func encodePLEBranch(_ e: MTLComputeCommandEncoder, layer i: Int, resid: MTLBuffer, rows: Int, prefill: Bool)
        throws
    {
        guard let ple, let post = layers[i].post else { return }
        try ple.encodeLayer(
            e, layer: i, resid: resid, xh: prefill ? pxn : xn, out: prefill ? pbranch! : branch!, rows: rows,
            prefill: prefill)
        try encodeBranchAdd(
            e, resid: resid, rows: rows, prefill: prefill, norm: (ple.postNorms[i], eps), bias: nil, mul: 1,
            scale: post.scalar)
    }

    /// resid = (resid + mul · N(t + bias)) · scale per row over the branch buffer t (`branch_add_rows`, which leaves t
    /// at zero); N: RMSNorm with weight `norm.w` and its epsilon, or the identity without a weight. t: `branch` /
    /// `pbranch` unless given.
    func encodeBranchAdd(
        _ e: MTLComputeCommandEncoder, resid: MTLBuffer, rows: Int, prefill: Bool, norm: (w: MTLBuffer?, eps: Float),
        bias: MTLBuffer?, mul: Float, scale: Float, t: MTLBuffer? = nil
    ) throws {
        let w = norm.w
        var p = (UInt32(hidden), norm.eps, mul, scale, UInt32((w != nil ? 1 : 0) | (bias != nil ? 2 : 0)))
        e.setComputePipelineState(try ctx.pipeline("branch_add_rows"))
        e.setBuffer(resid, offset: 0, index: 0)
        e.setBuffer(t ?? (prefill ? pbranch! : branch!), offset: 0, index: 1)
        e.setBuffer(w ?? resid, offset: 0, index: 2)
        e.setBuffer(bias ?? resid, offset: 0, index: 3)
        e.setBytes(&p, length: MemoryLayout.size(ofValue: p), index: 4)
        e.dispatchThreadgroups(
            MTLSize(width: rows, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: normThreads, height: 1, depth: 1))
    }

    /// After the embedding of `rows` rows of `resid`: the embedding multiplier and the embedding norm, if any.
    func encodeEmbedPost(_ e: MTLComputeCommandEncoder, resid: MTLBuffer, rows: Int) throws {
        let b = spec.block
        if b.embedScale != 1 {
            var p = (b.embedScale, UInt32(rows * hidden))
            e.setComputePipelineState(try ctx.pipeline("scale_rows"))
            e.setBuffer(resid, offset: 0, index: 0)
            e.setBytes(&p, length: 8, index: 1)
            e.dispatchThreads(
                MTLSize(width: rows * hidden, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
        if let n = b.embedNorm, let w = embedNormW {
            var p = (UInt32(hidden), n.eps, UInt32(hidden), UInt32(hidden))
            e.setComputePipelineState(try ctx.pipeline("rmsnorm_rows_f"))
            e.setBuffer(resid, offset: 0, index: 0)
            e.setBuffer(w, offset: 0, index: 1)
            e.setBytes(&p, length: 16, index: 2)
            e.dispatchThreadgroups(
                MTLSize(width: rows, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: normThreads, height: 1, depth: 1))
        }
    }

    /// After the lm_head of `rows` rows: logits · mul / div and the softcap, if any (before the argmax and the
    /// sampling, as mlx-lm).
    func encodeLogitsPost(_ e: MTLComputeCommandEncoder, rows: Int) throws {
        let b = spec.block
        guard b.logitsMul != 1 || b.logitsDiv != 1 || b.softcap != nil else { return }
        var p = (UInt32(rows * vocab), b.logitsMul, b.logitsDiv, b.softcap ?? 0)
        e.setComputePipelineState(try ctx.pipeline("logits_post"))
        e.setBuffer(logits, offset: 0, index: 0)
        e.setBytes(&p, length: 16, index: 1)
        e.dispatchThreads(
            MTLSize(width: rows * vocab, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
}
