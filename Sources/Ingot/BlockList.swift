import Foundation
import Metal

// Block lists: the contract between a sparse-attention selector and the kernel that walks them
// (Shaders/verify_attn_pl.metal). The attention layer only sees a BlockList, not where it came from.

/// Block lists for the next `AttentionLayer.encode` call (verification and decode). uint32 entry =
/// block << T | mask of T rows, blocks of `blockTokens` tokens in ascending order, no repeats, including the open block
/// of each row; `lists` [tiles · (perKVHead ? kvHeads : 1)][stride], `counts` [same index]. Chunk s walks
/// entries [s·perSplit, (s+1)·perSplit): the split depends only on the list (with T = 1, B = 1 and B = 16
/// give bit-identical results).
public struct BlockList {
    public let lists: MTLBuffer, counts: MTLBuffer
    public let stride: Int, tileRows: Int, blockTokens: Int, perSplit: Int
    public let perKVHead: Bool
    /// Verification chunks: constant (depends on neither position nor rows).
    public var nsplit: Int { (stride + perSplit - 1) / perSplit }
}

/// SparseListParams from verify_attn_pl.metal.
struct SparseListParams {
    var stride: UInt32, tiles: UInt32, perKVHead: UInt32, perSplit: UInt32
}

/// Generic conversion of a per-row bitmap ([rows, words]; bit b = full block b of 4 tokens) to lists with
/// T = 1 (blocklist_from_bitmap). The lists live here and are rewritten on every call.
public final class BitmapBlockLists {
    let ctx: MetalContext
    public let lists: MTLBuffer, counts: MTLBuffer
    public let stride: Int, perSplit: Int, maxRows: Int

    public init(ctx: MetalContext, stride: Int, perSplit: Int, maxRows: Int = AttentionLayer.maxRows) {
        precondition(stride > 0 && perSplit > 0)
        self.ctx = ctx
        self.stride = stride; self.perSplit = perSplit; self.maxRows = maxRows
        lists = ctx.buffer(length: maxRows * stride * 4)
        counts = ctx.buffer(length: maxRows * 4)
        memset(lists.contents(), 0, lists.length); memset(counts.contents(), 0, counts.length)
    }

    /// Lists for the `rows` rows of `bitmap` (from `offset`) at positions pos… (st[0] with `state`).
    public func encode(
        _ e: MTLComputeCommandEncoder, bitmap: MTLBuffer, offset: Int = 0, words nw: Int, rows: Int, pos: Int,
        state: MTLBuffer?
    ) throws -> BlockList {
        precondition(rows >= 1 && rows <= maxRows)
        var bp = (UInt32(nw), UInt32(pos), UInt32(stride), UInt32(state != nil ? 1 : 0))
        e.setComputePipelineState(try ctx.pipeline("blocklist_from_bitmap"))
        e.setBuffer(bitmap, offset: offset, index: 0); e.setBuffer(lists, offset: 0, index: 1)
        e.setBuffer(counts, offset: 0, index: 2)
        e.setBytes(&bp, length: 16, index: 3)
        e.setBuffer(state ?? counts, offset: 0, index: 4)
        e.dispatchThreadgroups(
            MTLSize(width: rows, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        return BlockList(
            lists: lists, counts: counts, stride: stride, tileRows: 1, blockTokens: 4, perSplit: perSplit,
            perKVHead: false)
    }
}

/// Bitmap selector: a per-row bitmap already written (the last QSA selection, qsa_m_select) plus the generic
/// conversion. It writes the lists of `rows` rows at positions pos… (st[0] with `state`). Another selector (DSA, NSA,
/// MoBA) would provide the same `encodeBlockList` with its own lists.
public struct BitmapSelector {
    public let bitmap: MTLBuffer, words: Int, lists: BitmapBlockLists
    public func encodeBlockList(_ e: MTLComputeCommandEncoder, rows: Int, pos: Int, state: MTLBuffer?) throws
        -> BlockList
    {
        try lists.encode(e, bitmap: bitmap, words: words, rows: rows, pos: pos, state: state)
    }
}
