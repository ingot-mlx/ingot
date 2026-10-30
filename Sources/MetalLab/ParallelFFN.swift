import Foundation
import Metal

// Gemma 4 26B-A4B's FFN branch (FFNKind.denseAndMoE; mlx-lm gemma4_text DecoderLayer with enable_moe):
//   h1 = post_feedforward_layernorm_1(mlp(pre_feedforward_layernorm(h)))
//   h2 = post_feedforward_layernorm_2(experts(pre_feedforward_layernorm_2(h), router(h)))
//   h  = h + post_feedforward_layernorm(h1 + h2)          (the usual branch epilogue, then the PLE branch if any)
// The router reads the un-normalized stream through its own RMSNorm (MoELayer.routerNorm). Each half accumulates into
// `branchPart` and `branch_add_rows` adds it, post-normed, to the FFN branch buffer (fp32 sum, so the order of the two
// halves does not change a bit).

extension HybridModel {
    /// The parallel FFN of layer `i` over the residual stream (`h` in the round, `ph` in the prefill) into the branch
    /// buffer; the caller runs the branch epilogue after it. Round: `rows` ≤ 16 and `rowLimit` as in the MoE. Prefill:
    /// `rows` is the padded chunk (the MoE takes its real rows from `pmoe.active`) and `i8` the lossy dense GEMMs.
    func encodeParallelFFN(
        _ e: MTLComputeCommandEncoder, layer i: Int, rows: Int, prefill: Bool, rowLimit: MTLBuffer? = nil,
        i8: DenseI8Scratch? = nil
    ) throws {
        let l = layers[i]
        guard let pn = l.parallel, let f = l.dense, let moe = l.moeOpt, let rn = moe.routerNorm else {
            preconditionFailure("layer \(i): not a parallel FFN layer")
        }
        let resid = prefill ? ph : h, x = prefill ? pxn : xn, part = prefill ? pbranchPart! : branchPart!
        let branch = branchOut(l, resid: resid, prefill: prefill), eps = spec.block.ffnNorm.eps
        func addPart(_ w: MTLBuffer) throws {
            try encodeBranchAdd(
                e, resid: branch, rows: rows, prefill: prefill, norm: (w, eps), bias: nil, mul: 1, scale: 1, t: part)
        }
        // Dense MLP on pre_feedforward_layernorm (n2).
        try norm(e, resid, l.n2, x, rows: rows, eps: eps)
        if prefill {
            try f.encodePrefill(e, x: x, out: part, rows: rows, s: pff!, i8: i8)
        } else {
            try f.encode(e, x: x, out: part, rows: rows)
        }
        try addPart(pn.densePost)
        // Router logits from its own norm of the stream, then the experts on pre_feedforward_layernorm_2.
        try norm(e, resid, rn, x, rows: rows, eps: eps)
        if prefill {
            let pm = pmoe!
            bindPrefillMoE(pm, layer: i)
            pm.accumulateInto = part
            try pm.encode(e, .router)
            try norm(e, resid, pn.moePre, x, rows: rows, eps: eps)
            for p in PrefillMoE.Piece.allCases where p != .router { try pm.encode(e, p) }
        } else {
            try moe.encode(e, x: x, out: part, rows: rows, accumulate: true, only: .router, rowLimit: rowLimit)
            try norm(e, resid, pn.moePre, x, rows: rows, eps: eps)
            for p in [MoELayer.Piece.route, .gateup, .down] {
                try moe.encode(e, x: x, out: part, rows: rows, accumulate: true, only: p, rowLimit: rowLimit)
            }
        }
        try addPart(pn.moePost)
    }
}
