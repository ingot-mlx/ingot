import Foundation
import Metal

/// Dense SwiGLU FFN of one layer of the full model (Qwen3.8-27B: 5120 → 17408 → 5120), with gate and up
/// concatenated by output rows at load time (a single [2·I, H] matrix: the output of each row is [gate | up]).
/// - Decode/verification (B ≤ 16): NAX qmv of gate|up (fp32) → SwiGLU to half → NAX qmv of down added to the residual.
/// - Prefill (M rows, multiple of 64): NAX v3 GEMM of gate|up with half output → SwiGLU → accumulated down GEMM.
/// Nemotron-H's relu² has no gate: `gu` is up_proj alone ([I, H]) and the `_relu2` activation kernels read it.
public final class DenseMLP {
    public let ctx: MetalContext
    public let layer: Int
    public let gu, down: DenseQ
    public let hidden: Int, inter: Int
    /// gate | up (false: relu² without gate, `gu` is up alone).
    let gated: Bool
    public var vq: VQMV, downVQ: VQMV
    /// Verification work buffers (shared between layers: used in series).
    nonisolated(unsafe) static var work: [ObjectIdentifier: (gu: MTLBuffer, h: MTLBuffer)] = [:]
    let guOut, hb: MTLBuffer

    public init(ctx: MetalContext, store: SafetensorsStore, config cfg: ModelConfig, layer: Int) throws {
        self.ctx = ctx
        self.layer = layer
        vq = VQMV(ctx)
        let sp = try cfg.requiredSpec, pre = try store.layerPrefix(layer) + sp.names.ffn
        gated = sp.block.activation != .relu2
        let n = sp.names
        gu = try DenseQ(
            store: store, config: cfg, paths: (gated ? [pre + n.denseGate] : []) + [pre + n.denseUp],
            device: ctx.device)
        down = try DenseQ(store: store, config: cfg, paths: [pre + n.denseDown], device: ctx.device)
        hidden = gu.K; inter = gated ? gu.N / 2 : gu.N
        // down (K = 17408 = 272 blocks of 64): does not exceed 90 % of the floor with any split tried.
        downVQ = VQMV(ctx)
        // 8 bits: fn_qmv8_nf1_sg16 in the round (VQMV; LFM3's down 2048×7680 at 140 GB/s, 130 with 32 columns per
        // threadgroup) and qmm_t_nax3_8 in prefill; outside the --int8 W8A8 (DenseQ.int8OK).
        guard [2, 4, 8].contains(gu.bits), down.bits == gu.bits, down.K == inter, down.N == hidden else {
            throw SafetensorsError.format("unsupported dense FFN shape")
        }
        // Shared by the layers of a context; a wider layer (Gemma 4 E2B's double-wide MLP) replaces them with its own
        // size for the layers that come after it (the earlier ones keep the narrower ones, enough for them).
        let key = ObjectIdentifier(ctx), R = MoELayer.maxRows, I = inter
        (guOut, hb) = MetalContext.scratchLock.withLock {
            if let w = Self.work[key], w.h.length >= R * I * 2 { return w }
            let w = (gu: ctx.buffer(length: R * 2 * I * 4), h: ctx.buffer(length: R * I * 2))
            Self.work[key] = w
            return w
        }
    }

    public var bytes: Int { gu.bytes + down.bytes }

    public enum Piece: String, CaseIterable { case gateup, swiglu, down }

    /// x: half [rows, hidden] → out float [rows, hidden] (+=).
    public func encode(_ e: MTLComputeCommandEncoder, x: MTLBuffer, out: MTLBuffer, rows: Int, only: Piece? = nil)
        throws
    {
        precondition(rows >= 1 && rows <= MoELayer.maxRows)
        if only == nil || only == .gateup { try vq.encode(e, x: x, gu, y: guOut, rows: rows) }
        if only == nil || only == .swiglu {
            var n = UInt32(inter)
            e.setComputePipelineState(try ctx.pipeline("dswiglu\(ctx.dims.ffnActSuffix)"))
            e.setBuffer(guOut, offset: 0, index: 0)
            e.setBuffer(hb, offset: 0, index: 1)
            e.setBytes(&n, length: 4, index: 2)
            e.dispatchThreads(
                MTLSize(width: rows * inter, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
        if only == nil || only == .down { try downVQ.encode(e, x: hb, down, y: out, rows: rows, accumulate: true) }
    }

    /// Prefill: `M` rows (multiple of 64). s.gu: half [M, 2·I]; s.h: half [M, I].
    public func encodePrefill(
        _ e: MTLComputeCommandEncoder, x: MTLBuffer, out: MTLBuffer, rows M: Int,
        s: (gu: MTLBuffer, h: MTLBuffer), i8: DenseI8Scratch? = nil
    ) throws {
        try gu.encodePrefillGemm(ctx, e, x: x, y: s.gu, rows: M, i8: i8, halfOut: true)
        if let i8, gu.int8OK, gated {
            // SwiGLU fused with the quantization of the down input (relu² goes through dswiglu_hh_relu2 and the
            // generic quantization of down's input).
            try i8.quantize(ctx, e, x: s.gu, rows: M, K: inter, swiglu: true)
            try down.encodeQmmI8(ctx, e, x: s.h, y: out, rows: M, scratch: i8, accumulate: true, quantizeX: false)
            return
        }
        var n = UInt32(inter)
        e.setComputePipelineState(try ctx.pipeline("dswiglu_hh\(ctx.dims.ffnActSuffix)"))
        e.setBuffer(s.gu, offset: 0, index: 0)
        e.setBuffer(s.h, offset: 0, index: 1)
        e.setBytes(&n, length: 4, index: 2)
        e.dispatchThreads(
            MTLSize(width: M * inter / 4, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        try down.encodePrefillGemm(ctx, e, x: s.h, y: out, rows: M, i8: i8, accumulate: true)
    }
}

/// Prefill work buffers for rotated W8A8 (xa_dense_i8.metal): converted weights [N, K] int8 + per-row scale,
/// activations [M, K] int8 + per-row scale. One per model (calls run in series).
public final class DenseI8Scratch {
    public let wq, sw, xq, sx: MTLBuffer
    /// K block of the int8 GEMM (the tile is 2 × WN simdgroups of 32 × 32, WN = 4 with K > 8192, otherwise 2:
    /// measured with M = 2048, down: 19.3 → 13.6 ms).
    public var bk = 128
    /// Hadamard of 32·2^3 (256) along K when K/32 is a multiple of 32 (otherwise none).
    private let rotLB = 3
    func lb(_ K: Int) -> Int { K / 32 % 32 == 0 ? rotLB : 0 }
    /// Per-row scales of the rotated weights, precomputed at load time (one fewer pass in the conversion): per matrix.
    public var rowScales: [ObjectIdentifier: MTLBuffer] = [:]
    public init(ctx: MetalContext, maxNK: Int, maxN: Int, maxRows: Int, maxK: Int) {
        wq = ctx.buffer(length: maxNK); sw = ctx.buffer(length: maxN * 4)
        xq = ctx.buffer(length: maxRows * maxK); sx = ctx.buffer(length: maxRows * 4)
    }

    /// Half activations [M, K] → xq, sx (rotated, int8 per row). With `swiglu`, x is gate | up [M, 2K].
    public func quantize(
        _ ctx: MetalContext, _ e: MTLComputeCommandEncoder, x: MTLBuffer, rows M: Int, K: Int,
        swiglu: Bool = false
    ) throws {
        var p = SIMD2<UInt32>(UInt32(M), UInt32(K))
        e.setComputePipelineState(
            try ctx.pipeline(
                "\(swiglu ? "swiglu_quant_rot_i8\(ctx.dims.ffnActSuffix)" : "quant_rows_rot_i8g")_l\(lb(K))"))
        for (i, b) in [x, xq, sx].enumerated() { e.setBuffer(b, offset: 0, index: i) }
        e.setBytes(&p, length: 8, index: 3)
        e.dispatchThreadgroups(
            MTLSize(width: (M + 3) / 4, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    }
}

extension DenseQ {
    /// Whether the rotated W8A8 of --int8 1 applies to this matrix (w4_i8rot converts from 4 bits, with groups of 64
    /// or 32): 8-bit ones go through the exact GEMM even with --int8 (they already have int8 precision; the
    /// conversion does not exist).
    public var int8OK: Bool { bits == 4 }

    /// Y[M, N] (+)= X·Wᵀ in rotated W8A8 (lossy). With `quantizeX` it first quantizes X (half [M, K]) into the scratch;
    /// otherwise it uses the X already quantized there (same input for several matrices).
    public func encodeQmmI8(
        _ ctx: MetalContext, _ e: MTLComputeCommandEncoder, x: MTLBuffer, y: MTLBuffer, rows M: Int,
        scratch sc: DenseI8Scratch, accumulate: Bool = false, halfOut: Bool = false,
        quantizeX: Bool = true
    ) throws {
        precondition(bits == 4 && K % 64 == 0 && N * K <= sc.wq.length && !(halfOut && accumulate))
        if quantizeX { try sc.quantize(ctx, e, x: x, rows: M, K: K) }
        var p2 = SIMD2<UInt32>(UInt32(N), UInt32(K))
        let pre = sc.rowScales[ObjectIdentifier(w)]
        e.setComputePipelineState(try ctx.pipeline("w4_i8rot\(pre != nil ? "_pre" : "")\(gsuf)_l\(sc.lb(K))"))
        for (i, b) in [w, s, b, sc.wq, pre ?? sc.sw].enumerated() { e.setBuffer(b, offset: 0, index: i) }
        e.setBytes(&p2, length: 8, index: 5)
        e.dispatchThreadgroups(
            MTLSize(width: (N + 3) / 4, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        let wm = 2
        var wn = K > 8192 ? 4 : 2
        if N % (32 * wn) != 0 { wn = 2 }
        precondition(N % (32 * wn) == 0 && M % (32 * wm) == 0)
        var p = QmmParams(M: M, N: N, K: K)
        var scale: Float = 1.0 / Float(32 << sc.lb(K))  // (H x)·(H w) = n x·w with the unnormalized Hadamard of size n
        e.setComputePipelineState(
            try ctx.pipeline(
                "qmm_i8_w\(wm)x\(wn)_bk\(K % sc.bk == 0 ? sc.bk : 64)_\(accumulate ? "f32acc" : halfOut ? "h" : "f32")")
        )
        for (i, b) in [sc.xq, sc.wq, sc.sx, pre ?? sc.sw, y].enumerated() { e.setBuffer(b, offset: 0, index: i) }
        e.setBytes(&p, length: MemoryLayout<QmmParams>.size, index: 5)
        e.setBytes(&scale, length: 4, index: 6)
        e.dispatchThreadgroups(
            MTLSize(width: N / (32 * wn), height: M / (32 * wm), depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32 * wm * wn, height: 1, depth: 1))
    }

    /// Ternary (Bonsai) in W8A8 (--int8 1): rotated x (the format's Hadamard) → int8 per row; ternary weights to int8
    /// with one scale per row in each chunk (tern_i8row, minimal loss, like the 4-bit path) and the regular qmm_i8:
    /// 435/473/454/434 tok/s at 512/4K/16K/32K versus 410/426/405/382 for the variant with exact weights and
    /// per-group scale at each K step (qmm_i8t, removed), with similar KL.
    public func encodeQmmI8T(
        _ ctx: MetalContext, _ e: MTLComputeCommandEncoder, x: MTLBuffer, y: MTLBuffer, rows M: Int,
        scratch sc: DenseI8Scratch, accumulate: Bool = false, halfOut: Bool = false
    ) throws {
        precondition(tern && K % 128 == 0 && M * K <= sc.xq.length && !(halfOut && accumulate))
        let xr = had != nil ? try encodeRotate(ctx, e, x: x, xOffset: 0, ldx: K, rows: M) : x
        var p2 = SIMD2<UInt32>(UInt32(M), UInt32(K))
        e.setComputePipelineState(try ctx.pipeline("quant_rows_i8"))
        for (i, b) in [xr, sc.xq, sc.sx].enumerated() { e.setBuffer(b, offset: 0, index: i) }
        e.setBytes(&p2, length: 8, index: 3)
        e.dispatchThreadgroups(
            MTLSize(width: (M + 3) / 4, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        precondition(N * K <= sc.wq.length)
        var p3 = SIMD2<UInt32>(UInt32(N), UInt32(K))
        e.setComputePipelineState(try ctx.pipeline("tern_i8row"))
        for (i, b) in [w, s, sc.wq, sc.sw].enumerated() { e.setBuffer(b, offset: 0, index: i) }
        e.setBytes(&p3, length: 8, index: 4)
        e.dispatchThreadgroups(
            MTLSize(width: (N + 3) / 4, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        let wm = 2
        var wn = K > 8192 ? 4 : 2
        if N % (32 * wn) != 0 { wn = 2 }
        var p = QmmParams(M: M, N: N, K: K)
        var one: Float = 1
        e.setComputePipelineState(
            try ctx.pipeline("qmm_i8_w\(wm)x\(wn)_bk128_\(accumulate ? "f32acc" : halfOut ? "h" : "f32")"))
        for (i, b) in [sc.xq, sc.wq, sc.sx, sc.sw, y].enumerated() { e.setBuffer(b, offset: 0, index: i) }
        e.setBytes(&p, length: MemoryLayout<QmmParams>.size, index: 5)
        e.setBytes(&one, length: 4, index: 6)
        e.dispatchThreadgroups(
            MTLSize(width: N / (32 * wn), height: M / (32 * wm), depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32 * wm * wn, height: 1, depth: 1))
    }

    /// Per-row scales of this rotated matrix (w4_rowscale), stored in the scratch: the conversion of each call then
    /// makes a single pass.
    public func precomputeRowScales(_ ctx: MetalContext, _ e: MTLComputeCommandEncoder, scratch sc: DenseI8Scratch)
        throws
    {
        let buf = ctx.buffer(length: N * 4)
        var p2 = SIMD2<UInt32>(UInt32(N), UInt32(K))
        e.setComputePipelineState(try ctx.pipeline("w4_rowscale\(gsuf)_l\(sc.lb(K))"))
        for (i, b) in [w, s, b, sc.wq, buf].enumerated() { e.setBuffer(b, offset: 0, index: i) }
        e.setBytes(&p2, length: 8, index: 5)
        e.dispatchThreadgroups(
            MTLSize(width: (N + 3) / 4, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        sc.rowScales[ObjectIdentifier(w)] = buf
    }

    /// Prefill GEMM: exact (encodeQmmF32) or, with `i8`, in rotated W8A8.
    public func encodePrefillGemm(
        _ ctx: MetalContext, _ e: MTLComputeCommandEncoder, x: MTLBuffer, y: MTLBuffer, rows M: Int,
        i8: DenseI8Scratch?, accumulate: Bool = false, halfOut: Bool = false
    ) throws {
        if let i8, tern {
            try encodeQmmI8T(ctx, e, x: x, y: y, rows: M, scratch: i8, accumulate: accumulate, halfOut: halfOut)
        } else if let i8, int8OK {
            try encodeQmmI8(ctx, e, x: x, y: y, rows: M, scratch: i8, accumulate: accumulate, halfOut: halfOut)
        } else {
            try encodeQmmF32(ctx, e, x: x, y: y, rows: M, accumulate: accumulate, halfOut: halfOut)
        }
    }
}
