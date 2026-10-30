import Metal

// Flash-Next prefill Q2 experts via the int8 path (zi_fn_i8.metal). Only with the INT8 prefill route
// (options.prefillInt8 and int8Experts, in the chunks that run in int8): xh and h become int8 per group of 64 of
// each row (the only loss: the (q − 1) weights are exact in int8) and gate|up and down use int8 × int8 matmul2d with
// per-group rescaling. Same tiles and outputs (hE half, yE fp32) as the fp16 NAX kernels.
extension FlashParts {
    func encodeExpertsI8(
        _ e: MTLComputeCommandEncoder, arena: MTLBuffer, smap: MTLBuffer, smapOff: Int, tiles tl: MTLBuffer,
        args ar: MTLBuffer, rows T: Int, nax: Int
    ) throws {
        // One group of 64 of K per step (with 2, BK 128, the expert took 593 ms versus 421: more registers).
        let ng = 1
        precondition([16, 32, 48, 64].contains(nax), "int8 experts: tiles of 16 to 64 rows")
        if xq8 == nil {
            let rows = xh.length / (hidden * 2)
            xq8 = (ctx.buffer(length: rows * hidden), ctx.buffer(length: rows * hidden / 16))
        }
        let (xq, xs) = xq8!
        // gate|up leaves h already in int8 (and its per-group-of-64 scales) in the third and fourth quarters of hE.
        let hq = hE.length / 2, hs = hE.length / 4 * 3
        let nx = T * hidden / 16
        try threads(e, "fn_q8g_perm", nx, [(xh, 0), (xq, 0), (xs, 0)], [UInt32(nx)])
        var qp = (
            UInt32(fmt.recordBytes), UInt32(fmt.gateCodes), UInt32(fmt.gateScales), UInt32(fmt.upCodes),
            UInt32(fmt.upScales), UInt32(fmt.downCodes), UInt32(fmt.downScales), UInt32(T)
        )
        let tpg = MTLSize(width: 128, height: 1, depth: 1)
        e.setComputePipelineState(try ctx.pipeline("fn_moe_gateup_i8_bm\(nax)_ng\(ng)"))
        for (k, (bf, o)) in [(xq, 0), (xs, 0), (arena, 0), (smap, smapOff), (tl, 0), (sorted, 0), (hE, hq), (hE, hs)]
            .enumerated()
        { e.setBuffer(bf, offset: o, index: k) }
        e.setBytes(&qp, length: 32, index: 8)
        e.dispatchThreadgroups(indirectBuffer: ar, indirectBufferOffset: 0, threadsPerThreadgroup: tpg)
        e.setComputePipelineState(try ctx.pipeline("fn_moe_down_i8_bm\(nax)_ng\(ng)"))
        for (k, (bf, o)) in [(hE, hq), (hE, hs), (arena, 0), (smap, smapOff), (tl, 0), (yE, 0)].enumerated() {
            e.setBuffer(bf, offset: o, index: k)
        }
        e.setBytes(&qp, length: 32, index: 6)
        e.dispatchThreadgroups(indirectBuffer: ar, indirectBufferOffset: 16, threadsPerThreadgroup: tpg)
    }
}
