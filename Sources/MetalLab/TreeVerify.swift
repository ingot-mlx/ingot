import Foundation
import Metal

// Tree verification in the dense model (zt_tree.metal): DFlash2 rounds verify a draft tree when options.treeNodes is
// set (Generate.swift chooses tree or chain per round); it can also measure the round cost and compare the logits
// with the chain of the path of each node.
extension HybridModel {
    public static let treeWords = 320
    /// The tree only with attention key chunks of decode of <= 1024 (prompts of < 16K): with 25K (chunk of 2048)
    /// exact row-wise attention lengthened the round by 10% and lost 3-6%.
    public static let treeMaxChunk = 1024

    /// Enables (or removes, with nil) tree mode in all layers: verification reads the topology of `b`.
    public func setTree(_ b: MTLBuffer?) {
        for l in layers {
            switch l.mixer {
            case .gdn(let g): g.treeBuf = b
            case .attn(let a): a.treeBuf = b
            case .conv, .none: precondition(b == nil, "tree verification: tree only in Qwen (GDN + attention)")
            }
        }
    }

    /// End of a tree round (after verification): greedy acceptance over the tree (spec_accept_tree: path,
    /// record in `rec` and round state in `state`) and compaction of the path into contiguous rows in the KV of each
    /// attention layer, the tape and history of each GDN layer and the drafter captures.
    public func encodeTreeFinish(
        _ e: MTLComputeCommandEncoder, tree: MTLBuffer, state: MTLBuffer, rec: MTLBuffer, ring: Int,
        drafter dr: DFlash2Drafter
    ) throws {
        var rg = UInt32(ring)
        e.setComputePipelineState(try ctx.pipeline("spec_accept_tree"))
        for (i, b) in [state, tok, top1, dr.dtok, rec].enumerated() { e.setBuffer(b, offset: 0, index: i) }
        e.setBytes(&rg, length: 4, index: 5)
        e.setBuffer(tree, offset: 0, index: 6)
        e.dispatchThreads(
            MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        for l in layers {
            switch l.mixer {
            case .attn(let a) where a.kvSource == nil:  // a KV-sharing layer's KV is its source's, compacted there
                var ld = UInt32(a.ldkv)
                e.setComputePipelineState(try a.ctx.pipeline("kv_compact_tree"))
                for (i, b) in [a.kc, a.vt, a.ks, a.vs, tree].enumerated() { e.setBuffer(b, offset: 0, index: i) }
                e.setBytes(&ld, length: 4, index: 5)
                e.dispatchThreads(
                    MTLSize(width: a.hd, height: a.kvHeads, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: a.hd, height: 1, depth: 1))
            case .gdn(let g):
                try g.encodeTreeCompact(e, tree: tree)
            case .attn, .conv, .none: break
            }
        }
        var nb = (UInt32(dr.ctxWidth), UInt32(0))
        e.setComputePipelineState(try ctx.pipeline("rows_compact_tree_h"))
        e.setBuffer(dr.caps, offset: 0, index: 0)
        e.setBuffer(tree, offset: 0, index: 1)
        e.setBytes(&nb, length: 8, index: 2)
        e.dispatchThreads(
            MTLSize(width: dr.ctxWidth, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
}
