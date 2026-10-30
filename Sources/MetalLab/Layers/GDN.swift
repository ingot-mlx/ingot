import Foundation
import Metal

// MARK: - GatedDeltaNet verification

// Portions adapted from splash (Apache-2.0; see THIRD_PARTY_NOTICES.md); changed: the tape is rewritten in Swift/Metal.

/// GDN (linear_attn) layer of Qwen3.6 with B ≤ 16 rows and the "tape": each round folds the `nacc` accepted rows of the
/// previous block, writes that state (the one from before its block) and saves k, v, g, beta and the raw conv rows of
/// its own block for the next round. This way the state is read and written once per round and no per-position
/// states are stored (idea from splash `decode/gdn.metal` and oMLX `qwen35_gdn_verify_fused.py`, Apache-2.0).
///
/// Also Nemotron-H's Mamba2 (`Kind.mamba2`): the same state, tape, ping-pong, conversation slots, snapshots and
/// prefix cache, with the recurrence of the GDN without the delta term (β = dt, k = B, v = x, q = C, plus `D·x`), a
/// conv with bias, the gated group norm in mlx-lm's order (`mamba_normgate`) and the sequential prefill only.
public final class GDNLayer {
    public enum Kind: Sendable { case gatedDelta, mamba2 }
    public static let maxRows = 16
    let kind: Kind
    /// k heads and their dimension and v dimension per head (Qwen: 16 × 128 and 128; Mamba2: groups, state and
    /// head_dim; ModelDims of the context).
    public let hk: Int, dk: Int, dv: Int
    /// v heads (32 in Qwen3.6, 48 in the 27B; ModelDims of the context): v width, conv channels (q | k | v) and
    /// qkv|z|b|a outputs.
    public let hv: Int, vDim: Int, convDim: Int, projN: Int
    /// Width of q and of k per row (hk · dk).
    let qkDim: Int
    public let ctx: MetalContext
    public let layer: Int
    public let proj, outProj: DenseQ
    /// Bonsai: in_proj_b | in_proj_a fp32 [2·hv, hidden] (columns convDim + vDim ... of the projection).
    public let baW: MTLBuffer?
    public let convW, aLog, dtBias, normW: MTLBuffer
    /// Mamba2: conv bias (LAB_NWT) and D (fp32) per head.
    let convB: MTLBuffer?, dSkip: MTLBuffer?
    public let eps: Float
    /// Ping-pong per round: state, conv history and tape (k, v, g, beta). One conversation's (`GDNSlot`): batched
    /// decoding swaps them with those of another (`exchangeSlot`).
    public private(set) var state: [MTLBuffer], hist: [MTLBuffer], tk: [MTLBuffer], tv: [MTLBuffer], tg: [MTLBuffer],
        tb: [MTLBuffer]
    public let q, projOut, y, o: MTLBuffer
    public var cur = 0
    /// Draft tree (`--tree`): with a tree buffer, the round verifies a tree of drafts (conv by ancestors,
    /// recurrence with slots and tape following the accepted path; see zt_tree.metal). Only without round state.
    public var treeBuf: MTLBuffer?
    let vq: VQMV
    /// Recurrence: 4 dv rows per simdgroup and 2 steps loaded at once (16 dv rows per threadgroup of 128 threads).
    static let recurKernel = "gdn_recur_dv4_g2", recurRowsPerTG = 16

    public init(ctx: MetalContext, store: SafetensorsStore, config cfg: ModelConfig, layer: Int) throws {
        self.ctx = ctx
        let dm = ctx.dims
        hv = dm.gdnHV; vDim = dm.gdnVDim; convDim = dm.gdnConvDim; projN = dm.gdnProjN
        hk = dm.gdnHK; dk = dm.gdnDK; dv = dm.gdnDV; qkDim = hk * dk
        kind = dm.ssm ? .mamba2 : .gatedDelta
        self.layer = layer
        vq = VQMV(ctx)
        let pre = try store.layerPrefix(layer) + (dm.ssm ? "mixer." : "linear_attn.")
        let d = ctx.device
        // qkv|z|b|a: 12352 rows in Qwen3.6, 16480 in the 27B (padded to a multiple of 64 for the prefill GEMMs).
        // Mamba2: one in_proj z|xBC|dt (10304 rows in the Lightning, 17504 in the Nano 4B), padded the same way.
        if dm.ssm {
            proj = try DenseQ(store: store, config: cfg, paths: [pre + "in_proj"], device: d, padTo: 64)
            baW = nil
        } else if store.tensors[pre + "in_proj_qkv.signs"] != nil {
            // Bonsai 2 (rotated ternary): ternary qkv|z (with zero rows up to the qkv|z|b|a width) and unrotated fp32 b|a,
            // added afterwards into their columns (rowdot_f32w).
            proj = try DenseQ(
                store: store, config: cfg, paths: ["in_proj_qkv", "in_proj_z"].map { pre + $0 }, device: d,
                totalRows: (projN + 63) / 64 * 64)
            let bt = try store.info(pre + "in_proj_b.weight"), at = try store.info(pre + "in_proj_a.weight")
            guard bt.dtype == .f32, at.dtype == .f32, bt.shape == [hv, cfg.hiddenSize], at.shape == bt.shape else {
                throw SafetensorsError.format("Bonsai in_proj_b/a: expected fp32 [hv, hidden]")
            }
            baW = ctx.buffer(store.array(bt, as: Float.self) + store.array(at, as: Float.self))
        } else {
            proj = try DenseQ(
                store: store, config: cfg,
                paths: ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"].map { pre + $0 },
                device: d, padTo: 64)
            baW = nil
        }
        outProj = try DenseQ(store: store, config: cfg, paths: [pre + "out_proj"], device: d)
        // Bits per matrix: qkv|z|b|a and out_proj at 4 bits, or ternary (Bonsai) in both. 8-bit GDN matrices (the
        // OptiQ mixed conversions) are refused at load: no tier-1 model carries them and nothing tests them since the
        // mixed toy left.
        let bitsOK = proj.tern ? outProj.tern : proj.bits == 4 && outProj.bits == 4
        guard proj.N == (projN + 63) / 64 * 64, outProj.K == vDim, bitsOK else {
            throw SafetensorsError.format(
                "unsupported GDN shape (N \(proj.N), bits \(proj.bits)/\(outProj.bits)): 4 bits per matrix, or ternary")
        }
        try Self.checkSmallWeights(store, prefix: pre, f32: ctx.dims.normF32, mamba: dm.ssm)
        if dm.ssm {
            try Self.checkMambaShapes(store, prefix: pre, hv: hv, vDim: vDim, convDim: convDim)
            convB = try alignedTensor(store, pre + "conv1d.bias", d)
            dSkip = try floatTensor(store, pre + "D", d)
        } else {
            convB = nil; dSkip = nil
        }
        convW = try alignedTensor(store, pre + "conv1d.weight", d)
        // A_log always in fp32 (the kernels read `const device float*`): mlx-lm does not cast it when converting, so
        // the mlx-community packs of Qwen3.5 0.8B/2B keep the original F32, while the calibrated ones carry BF16.
        // bf16/fp16 → fp32 is exact, so the BF16 models give the same bits as before.
        aLog = try floatTensor(store, pre + "A_log", d)
        dtBias = try alignedTensor(store, pre + "dt_bias", d)
        normW = try alignedTensor(store, pre + "norm.weight", d)
        eps = Float(cfg.rmsNormEps)
        let R = Self.maxRows
        let fresh = Self.slotBuffers(ctx, hv: hv, vDim: vDim, convDim: convDim, qkDim: qkDim, dv: dv, dk: dk)
        state = fresh.state; hist = fresh.hist; tk = fresh.tk; tv = fresh.tv; tg = fresh.tg; tb = fresh.tb
        q = ctx.buffer(length: R * qkDim * 4)
        projOut = ctx.buffer(length: R * proj.N * 4)
        y = ctx.buffer(length: R * vDim * 4)
        o = ctx.buffer(length: R * vDim * 2)
    }

    /// A layer from buffers (KernelCase `mamba2`): the shape is the context's and the kind follows it (Mamba2 with
    /// `ssm`); `proj` only gives the projection's row stride (the test writes the projected rows itself).
    init(
        ctx: MetalContext, proj: DenseQ, outProj: DenseQ, convW: MTLBuffer, convB: MTLBuffer?, aLog: MTLBuffer,
        dtBias: MTLBuffer, dSkip: MTLBuffer?, normW: MTLBuffer, eps: Float
    ) {
        self.ctx = ctx
        let dm = ctx.dims
        hv = dm.gdnHV; vDim = dm.gdnVDim; convDim = dm.gdnConvDim; projN = dm.gdnProjN
        hk = dm.gdnHK; dk = dm.gdnDK; dv = dm.gdnDV; qkDim = hk * dk
        kind = dm.ssm ? .mamba2 : .gatedDelta
        layer = 0
        vq = VQMV(ctx)
        self.proj = proj; self.outProj = outProj; baW = nil
        self.convW = convW; self.convB = convB; self.aLog = aLog; self.dtBias = dtBias; self.dSkip = dSkip
        self.normW = normW; self.eps = eps
        let R = Self.maxRows
        let fresh = Self.slotBuffers(ctx, hv: hv, vDim: vDim, convDim: convDim, qkDim: qkDim, dv: dv, dk: dk)
        state = fresh.state; hist = fresh.hist; tk = fresh.tk; tv = fresh.tv; tg = fresh.tg; tb = fresh.tb
        for b in state + hist { memset(b.contents(), 0, b.length) }
        q = ctx.buffer(length: R * qkDim * 4)
        projOut = ctx.buffer(length: R * proj.N * 4)
        y = ctx.buffer(length: R * vDim * 4)
        o = ctx.buffer(length: R * vDim * 2)
    }

    /// conv, dt_bias and the norm are read as they are, in the type of the small weights (LAB_NWT: fp32 with
    /// `normF32`, bf16 otherwise), and A_log is converted to fp32 from fp32, bf16 or fp16: any other dtype is a load
    /// error naming the tensor, never a silent reinterpretation (like Granite's fp16 scales).
    /// Mamba2 also has the conv bias (LAB_NWT) and D (to fp32 like A_log).
    static func checkSmallWeights(_ store: SafetensorsStore, prefix pre: String, f32: Bool, mamba: Bool = false) throws
    {
        let nwt: DType = f32 ? .f32 : .bf16
        for n in ["conv1d.weight", "dt_bias", "norm.weight"] + (mamba ? ["conv1d.bias"] : []) {
            let t = try store.info(pre + n)
            guard t.dtype == nwt else {
                throw SafetensorsError.format(
                    "\(pre + n): \(t.dtype.rawValue), expected \(nwt.rawValue) (small GDN weights)")
            }
        }
        for n in ["A_log"] + (mamba ? ["D"] : []) {
            let a = try store.info(pre + n)
            guard [.f32, .bf16, .f16].contains(a.dtype) else {
                throw SafetensorsError.format("\(pre)\(n): \(a.dtype.rawValue), expected F32, BF16 or F16")
            }
        }
    }

    /// Mamba2's small tensors with the shapes the kernels index: conv [convDim, 4, 1] and its bias, the gated norm
    /// [d_inner] and A_log, dt_bias and D per head; conv channels in whole blocks of 128 (the prework's grid).
    static func checkMambaShapes(_ store: SafetensorsStore, prefix pre: String, hv: Int, vDim: Int, convDim: Int)
        throws
    {
        let want: [(String, [Int])] = [
            ("conv1d.weight", [convDim, 4, 1]), ("conv1d.bias", [convDim]), ("norm.weight", [vDim]),
            ("A_log", [hv]), ("dt_bias", [hv]), ("D", [hv]),
        ]
        for (n, sh) in want {
            let t = try store.info(pre + n)
            guard t.shape == sh else { throw SafetensorsError.format("\(pre + n): shape \(t.shape), expected \(sh)") }
        }
        guard convDim % 128 == 0 else {
            throw SafetensorsError.format("Mamba2 conv of \(convDim) channels (a multiple of 128 only)")
        }
    }

    public var weightBytes: Int {
        proj.bytes + outProj.bytes + convDim * 4 * 2 + (baW?.length ?? 0) + (convB?.length ?? 0)
    }

    /// Bonsai: y[:, convDim + vDim ..< +2·hv] = x · [b; a]ᵀ (x unrotated; y float or half with stride proj.N).
    func encodeBA(_ e: MTLComputeCommandEncoder, x: MTLBuffer, xOffset: Int = 0, y: MTLBuffer, rows: Int, half: Bool)
        throws
    {
        guard let baW else { return }
        let col = convDim + vDim, K = ctx.dims.hidden
        var p = (UInt32(rows), UInt32(2 * hv), UInt32(K), UInt32(K), UInt32(proj.N), UInt32(0))
        e.setComputePipelineState(try ctx.pipeline(half ? "rowdot_f32w_h" : "rowdot_f32w_f"))
        e.setBuffer(x, offset: xOffset, index: 0)
        e.setBuffer(baW, offset: 0, index: 1)
        e.setBuffer(y, offset: col * (half ? 2 : 4), index: 2)
        e.setBytes(&p, length: 24, index: 3)
        e.dispatchThreadgroups(
            MTLSize(width: (2 * hv + 7) / 8, height: rows, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    // MARK: Conversation slots

    /// The recurrent state of one conversation (batched decoding, `HybridModel.generateBatch`): what `exchangeSlot`
    /// swaps with the live fields of the layer. Slot 0 is the layer's own buffers.
    struct GDNSlot {
        var state, hist, tk, tv, tg, tb: [MTLBuffer]
        var cur: Int
    }

    private static func slotBuffers(
        _ ctx: MetalContext, hv: Int, vDim: Int, convDim: Int, qkDim: Int, dv: Int, dk: Int
    ) -> GDNSlot {
        let R = maxRows
        func pair(_ n: Int) -> [MTLBuffer] { [ctx.buffer(length: n), ctx.buffer(length: n)] }
        return GDNSlot(
            state: pair(hv * dv * dk * 4), hist: pair((3 + R) * convDim * 4), tk: pair(R * qkDim * 4),
            tv: pair(R * vDim * 4), tg: pair(R * hv * 4), tb: pair(R * hv * 4), cur: 0)
    }

    /// A fresh slot: zero state and conv history, as after `HybridModel.reset`.
    func makeSlot() -> GDNSlot {
        let s = Self.slotBuffers(ctx, hv: hv, vDim: vDim, convDim: convDim, qkDim: qkDim, dv: dv, dk: dk)
        for b in s.state + s.hist { memset(b.contents(), 0, b.length) }
        return s
    }

    /// Swaps the live state, history, tape and ping-pong side with those of `s` (only references: nothing is copied).
    func exchangeSlot(_ s: inout GDNSlot) {
        swap(&state, &s.state); swap(&hist, &s.hist)
        swap(&tk, &s.tk); swap(&tv, &s.tv); swap(&tg, &s.tg); swap(&tb, &s.tb)
        swap(&cur, &s.cur)
    }

    public enum Piece: String, CaseIterable { case proj, prework, recur, normgate, outproj }

    /// One round: x half [rows, 2048] → out float [rows, 2048] (with `accumulate`, out += GDN). The `nacc` rows of the block
    /// before are folded first. Calls `advance()` after each round (flips the ping-pong).
    /// `rowOffset` (batched decoding): the state pieces (prework, recur) of these `rows` read and write rows
    /// `rowOffset…` of `projOut` and `y`, the rows of this conversation in a round of several.
    public func encode(
        _ enc: MTLComputeCommandEncoder, x: MTLBuffer, xOffset: Int = 0, out: MTLBuffer, rows: Int,
        nacc: Int, accumulate: Bool = false, only: Piece? = nil, roundState: MTLBuffer? = nil, rowLimit: Bool = false,
        rowOffset: Int = 0
    ) throws {
        precondition(rows >= 0 && rows <= Self.maxRows && nacc >= 0 && nacc <= Self.maxRows)
        precondition(!rowLimit || roundState != nil)
        precondition(
            rowOffset == 0
                || (rowOffset + rows <= Self.maxRows && roundState == nil && treeBuf == nil
                    && (only == .prework || only == .recur)),
            "rowOffset: state pieces of the plain round only")
        let po = rowOffset * proj.N * 4, yo = rowOffset * vDim * 4
        let a = cur, b = 1 - cur
        // dyn: bit 0 = nacc of the round state; bit 1 = the recurrence only walks the first st[6] rows.
        var gp = (
            UInt32(rows), UInt32(nacc), UInt32(proj.N), eps, UInt32(0),
            UInt32(roundState == nil ? 0 : 1) | (rowLimit ? 2 : 0)
        )
        let st = roundState ?? projOut
        if rows > 0 && (only == nil || only == .proj) {
            try vq.encode(enc, x: x, xOffset: xOffset, proj, y: projOut, rows: rows)
            try encodeBA(enc, x: x, xOffset: xOffset, y: projOut, rows: rows, half: false)
        }
        if let trb = treeBuf {
            precondition(!rowLimit && baW == nil && kind == .gatedDelta, "tree verification: Qwen's GDN only")
            if only == nil || only == .prework, rows > 0 {
                enc.setComputePipelineState(try ctx.pipeline("gdn_prework_tree"))
                for (i, buf) in [projOut, convW, hist[a], aLog, dtBias, hist[b], q, tk[b], tv[b], tg[b], tb[b]]
                    .enumerated()
                {
                    enc.setBuffer(buf, offset: 0, index: i)
                }
                enc.setBytes(&gp, length: 24, index: 11)
                enc.setBuffer(trb, offset: 0, index: 12)
                enc.setBuffer(st, offset: 0, index: 13)
                enc.dispatchThreadgroups(
                    MTLSize(width: convDim / 128, height: rows, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            }
            if only == nil || only == .recur {
                enc.setComputePipelineState(try ctx.pipeline("gdn_recur_tree_s\(rows <= 5 ? 1 : rows <= 11 ? 2 : 3)"))
                for (i, buf) in [state[a], state[b], tk[a], tv[a], tg[a], tb[a], q, tk[b], tv[b], tg[b], tb[b], y]
                    .enumerated()
                {
                    enc.setBuffer(buf, offset: 0, index: i)
                }
                enc.setBytes(&gp, length: 24, index: 12)
                enc.setBuffer(trb, offset: 0, index: 13)
                enc.setBuffer(st, offset: 0, index: 14)
                enc.dispatchThreadgroups(
                    MTLSize(width: hv * dv / Self.recurRowsPerTG, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            }
        }
        if treeBuf == nil, only == nil || only == .prework, rows > 0 {
            enc.setComputePipelineState(try ctx.pipeline("gdn_prework"))
            for (i, buf) in [projOut, convW, hist[a], aLog, dtBias, hist[b], q, tk[b], tv[b], tg[b], tb[b]].enumerated()
            {
                enc.setBuffer(buf, offset: i == 0 ? po : 0, index: i)
            }
            enc.setBytes(&gp, length: 24, index: 11)
            enc.setBuffer(st, offset: 0, index: 12)
            bindConvBias(enc)
            enc.dispatchThreadgroups(
                MTLSize(width: convDim / 128, height: rows, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        }
        if treeBuf == nil, only == nil || only == .recur {
            enc.setComputePipelineState(try ctx.pipeline(Self.recurKernel))
            for (i, buf) in [state[a], state[b], tk[a], tv[a], tg[a], tb[a], q, tk[b], tv[b], tg[b], tb[b], y]
                .enumerated()
            {
                enc.setBuffer(buf, offset: i == 11 ? yo : 0, index: i)
            }
            enc.setBytes(&gp, length: 24, index: 12)
            enc.setBuffer(st, offset: 0, index: 13)
            bindSkip(enc)
            enc.dispatchThreadgroups(
                MTLSize(width: hv * dv / Self.recurRowsPerTG, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        }
        if rows > 0 && (only == nil || only == .normgate) {
            try encodeNormGate(enc, y: y, proj: projOut, o: o, rows: rows, gp: &gp, half: false)
        }
        if rows > 0 && (only == nil || only == .outproj) {
            try vq.encode(enc, x: o, outProj, y: out, rows: rows, accumulate: accumulate)
        }
    }

    public func advance() { cur = 1 - cur }

    /// Mamba2's extra arguments: the conv bias of the prework (buffer 13) and D of the recurrence (buffer 14).
    private func bindConvBias(_ e: MTLComputeCommandEncoder) {
        if let convB { e.setBuffer(convB, offset: 0, index: 13) }
    }
    private func bindSkip(_ e: MTLComputeCommandEncoder) { if let dSkip { e.setBuffer(dSkip, offset: 0, index: 14) } }

    /// Gated norm of `rows` rows: per v head (`gdn_normgate`, or its `_h` with a half projection) or, in Mamba2, per
    /// group of channels with the gate first (`mamba_normgate`, fp32 projection).
    private func encodeNormGate(
        _ e: MTLComputeCommandEncoder, y: MTLBuffer, proj: MTLBuffer, o: MTLBuffer, rows: Int,
        gp: inout (UInt32, UInt32, UInt32, Float, UInt32, UInt32), half: Bool
    ) throws {
        let mamba = kind == .mamba2
        precondition(!mamba || !half, "Mamba2: fp32 projection only (sequential prefill)")
        e.setComputePipelineState(try ctx.pipeline(mamba ? "mamba_normgate" : half ? "gdn_normgate_h" : "gdn_normgate"))
        for (i, buf) in [y, proj, normW, o].enumerated() { e.setBuffer(buf, offset: 0, index: i) }
        e.setBytes(&gp, length: 24, index: 4)
        e.dispatchThreadgroups(
            MTLSize(width: (rows * (mamba ? hk : hv) + 3) / 4, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    }

    /// After accepting a tree (and after `advance()`), leaves the accepted path in contiguous rows of the tape and of the
    /// conv history on the side the round wrote.
    func encodeTreeCompact(_ e: MTLComputeCommandEncoder, tree: MTLBuffer) throws {
        let c = cur
        e.setComputePipelineState(try ctx.pipeline("gdn_compact_tree"))
        for (i, b) in [tk[c], tv[c], tg[c], tb[c], hist[c], tree].enumerated() { e.setBuffer(b, offset: 0, index: i) }
        let n = qkDim + vDim + 2 * hv + convDim
        e.dispatchThreads(
            MTLSize(width: n, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    /// State after folding the first `nacc` rows of the tape, written to `stateOut` (and the conv window to
    /// `histOut`) without touching the live state (snapshot mid-block).
    func encodeFold(_ e: MTLComputeCommandEncoder, nacc: Int, stateOut: MTLBuffer, histOut: MTLBuffer) throws {
        precondition(nacc >= 0 && nacc <= Self.maxRows)
        let a = cur
        var gp = (UInt32(0), UInt32(nacc), UInt32(proj.N), eps, UInt32(0), UInt32(0))
        e.setComputePipelineState(try ctx.pipeline(Self.recurKernel))
        for (i, buf) in [
            state[a], stateOut, tk[a], tv[a], tg[a], tb[a], q, tk[1 - a], tv[1 - a], tg[1 - a], tb[1 - a], y,
        ].enumerated() {
            e.setBuffer(buf, offset: 0, index: i)
        }
        e.setBytes(&gp, length: 24, index: 12)
        e.setBuffer(projOut, offset: 0, index: 13)
        bindSkip(e)
        e.dispatchThreadgroups(
            MTLSize(width: hv * dv / Self.recurRowsPerTG, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        var n4 = UInt32(3 * convDim / 4)
        e.setComputePipelineState(try ctx.pipeline("copy_f32"))
        e.setBuffer(hist[a], offset: nacc * convDim * 4, index: 0)
        e.setBuffer(histOut, offset: 0, index: 1)
        e.setBytes(&n4, length: 4, index: 2)
        e.dispatchThreads(
            MTLSize(width: Int(n4), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    /// Folds the first `nacc` rows of the tape into the state (no new block) and leaves the conv window
    /// aligned: afterwards the state is the one after those rows, with no pending tape, as after a prefill.
    func encodeFold(_ e: MTLComputeCommandEncoder, nacc: Int) throws {
        precondition(nacc > 0 && nacc <= Self.maxRows)
        let a = cur, b = 1 - cur
        var gp = (UInt32(0), UInt32(nacc), UInt32(proj.N), eps, UInt32(0), UInt32(0))
        e.setComputePipelineState(try ctx.pipeline(Self.recurKernel))
        for (i, buf) in [state[a], state[b], tk[a], tv[a], tg[a], tb[a], q, tk[b], tv[b], tg[b], tb[b], y].enumerated()
        {
            e.setBuffer(buf, offset: 0, index: i)
        }
        e.setBytes(&gp, length: 24, index: 12)
        e.setBuffer(projOut, offset: 0, index: 13)
        bindSkip(e)
        e.dispatchThreadgroups(
            MTLSize(width: hv * dv / Self.recurRowsPerTG, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        // The 3 conv rows before the new position: rows nacc..nacc+2 of the current history.
        var n4 = UInt32(3 * convDim / 4)
        e.setComputePipelineState(try ctx.pipeline("copy_f32"))
        e.setBuffer(hist[a], offset: nacc * convDim * 4, index: 0)
        e.setBuffer(hist[b], offset: 0, index: 1)
        e.setBytes(&n4, length: 4, index: 2)
        e.dispatchThreads(
            MTLSize(width: Int(n4), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        advance()
    }

}

// MARK: - Prefill (chunks of many rows)

extension GDNLayer {
    /// Prefill chunk: `rows` real rows (the projections use `padded`, a multiple of 64). The recurrence
    /// writes the final state (no tape) and the conv window ends up in rows 0..2 of the next history.
    /// `x` nil: the projected rows are already in `s.proj`; `out` nil: no out_proj (KernelCase `mamba2`).
    func encodePrefill(
        _ e: MTLComputeCommandEncoder?, x: MTLBuffer?, out: MTLBuffer?, rows r: Int, padded M: Int,
        s: (
            proj: MTLBuffer, hcur: MTLBuffer, q: MTLBuffer, k: MTLBuffer, v: MTLBuffer, g: MTLBuffer, b: MTLBuffer,
            y: MTLBuffer, o: MTLBuffer
        ),
        chunk: GDNChunk? = nil, i8: DenseI8Scratch? = nil, i8Out: DenseI8Scratch? = nil, tl: GPUTimeline? = nil,
        capture: (rows: [Int], targets: [(state: MTLBuffer, hist: MTLBuffer)])? = nil
    ) throws {
        let a = cur, b = 1 - cur
        precondition(capture == nil || chunk != nil, "capture without chunked GDN")
        precondition(kind == .gatedDelta || chunk == nil, "Mamba2: sequential prefill only")
        let capS = capture.map { (rows: $0.rows, states: $0.targets.map(\.state)) }
        var gp = (UInt32(r), UInt32(0), UInt32(proj.N), eps, UInt32(1), UInt32(0))
        // With chunked GDN: the projection in half and q, k, v in half (gdn_prework_hh / gdn_cprep_h / gdn_normgate_h);
        // with the sequential recurrence (reference), everything in fp32.
        let hh = chunk != nil
        if let x {
            try tseg(tl, e, "gdn.proj") {
                try proj.encodePrefillGemm(ctx, $0, x: x, y: s.proj, rows: M, i8: i8, halfOut: hh)
                try encodeBA($0, x: x, y: s.proj, rows: M, half: hh)
            }
        }
        try tseg(tl, e, "gdn.conv") { e in
            e.setComputePipelineState(try ctx.pipeline(hh ? "gdn_prework_hh" : "gdn_prework"))
            for (i, buf) in [s.proj, convW, hist[a], aLog, dtBias, s.hcur, s.q, s.k, s.v, s.g, s.b].enumerated() {
                e.setBuffer(buf, offset: 0, index: i)
            }
            var gpw = gp
            if let cap = capture, let lo = cap.rows.min(), let hi = cap.rows.max() {
                // First and last captured row + 1 (the kernel saves the conv history of the rows in between).
                precondition(cap.rows.count <= GDNChunk.maxCaptures && hi + 1 < 4096)
                gpw.5 = UInt32(lo + 1) << 8 | UInt32(hi + 1) << 20
            }
            e.setBytes(&gpw, length: 24, index: 11)
            e.setBuffer(s.proj, offset: 0, index: 12)
            bindConvBias(e)
            e.dispatchThreadgroups(
                MTLSize(width: convDim / 128, height: r, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        }
        if let ch = chunk {
            if let tl {
                try tl.seg("gdn.cprep") {
                    try ch.encode(
                        $0, q: s.q, k: s.k, v: s.v, gate: s.g, beta: s.b, sin: state[a], sout: state[b], y: s.y,
                        rows: r, only: 1, halfIn: true)
                }
                try tl.seg("gdn.cscan") {
                    try ch.encode(
                        $0, q: s.q, k: s.k, v: s.v, gate: s.g, beta: s.b, sin: state[a], sout: state[b], y: s.y,
                        rows: r, only: 2, halfIn: true, capture: capS)
                }
            } else {
                try ch.encode(
                    e!, q: s.q, k: s.k, v: s.v, gate: s.g, beta: s.b, sin: state[a], sout: state[b], y: s.y, rows: r,
                    halfIn: true, capture: capS)
            }
        } else {
            try tseg(tl, e, "gdn.recur") { e in
                e.setComputePipelineState(try ctx.pipeline(Self.recurKernel))
                for (i, buf) in [state[a], state[b], tk[a], tv[a], tg[a], tb[a], s.q, s.k, s.v, s.g, s.b, s.y]
                    .enumerated()
                {
                    e.setBuffer(buf, offset: 0, index: i)
                }
                e.setBytes(&gp, length: 24, index: 12)
                e.setBuffer(s.proj, offset: 0, index: 13)
                bindSkip(e)
                e.dispatchThreadgroups(
                    MTLSize(width: hv * dv / Self.recurRowsPerTG, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            }
        }
        try tseg(tl, e, "gdn.normgate") { e in
            var n4 = UInt32(3 * convDim / 4)
            e.setComputePipelineState(try ctx.pipeline("copy_f32"))
            e.setBuffer(s.hcur, offset: r * convDim * 4, index: 0)
            e.setBuffer(hist[b], offset: 0, index: 1)
            e.setBytes(&n4, length: 4, index: 2)
            e.dispatchThreads(
                MTLSize(width: Int(n4), height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            // Conv history after each captured row (its 3 entries: rows j+1…j+3 of hcur).
            if let cap = capture {
                for (j, t) in zip(cap.rows, cap.targets) {
                    e.setBuffer(s.hcur, offset: (j + 1) * convDim * 4, index: 0)
                    e.setBuffer(t.hist, offset: 0, index: 1)
                    e.dispatchThreads(
                        MTLSize(width: Int(n4), height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                }
            }
            try encodeNormGate(e, y: s.y, proj: s.proj, o: s.o, rows: r, gp: &gp, half: hh)
        }
        if let out {
            try tseg(tl, e, "gdn.outproj") {
                try outProj.encodePrefillGemm(ctx, $0, x: s.o, y: out, rows: M, i8: i8Out, accumulate: true)
            }
        }
        advance()
    }
}

/// Buffers and launch of the chunked prefill GDN (yz_gdn_chunk.metal).
public final class GDNChunk {
    public static let C = 64
    let ctx: MetalContext
    public let maxRows: Int
    let W, U, Q, P, K, g: MTLBuffer
    /// K̃' and γ' of the captures (gdn_capprep) and a filler buffer for the arguments that are unused.
    let kcap, gcap, dummy: MTLBuffer
    /// v heads of the GDN (ModelDims of the context).
    let hv: Int
    /// Captures within one pass: points (a) and (b) and up to two prefix-cache cuts captured as rows of the last
    /// chunk (system and conversation).
    public static let maxCaptures = 4
    public init(ctx: MetalContext, maxRows: Int) {
        self.ctx = ctx
        self.maxRows = maxRows
        hv = ctx.dims.gdnHV
        let nch = (maxRows + Self.C - 1) / Self.C, H = hv
        W = ctx.buffer(length: H * nch * 64 * 128 * 2); U = ctx.buffer(length: H * nch * 128 * 64 * 4)
        Q = ctx.buffer(length: H * nch * 64 * 128 * 2); P = ctx.buffer(length: H * nch * 64 * 64 * 2)
        K = ctx.buffer(length: H * nch * 128 * 64 * 2); g = ctx.buffer(length: H * nch * 4)
        kcap = ctx.buffer(length: Self.maxCaptures * H * 128 * 64 * 2)
        gcap = ctx.buffer(length: Self.maxCaptures * H * 4)
        dummy = ctx.buffer(length: 16)
    }
    /// q, k, v, g, beta: outputs of gdn_prework for `rows` rows; y: [rows, 4096]; state sin → sout.
    public func encode(  // swiftlint:disable:this function_parameter_count - hot path: chunked prefill
        _ e: MTLComputeCommandEncoder, q: MTLBuffer, k: MTLBuffer, v: MTLBuffer, gate: MTLBuffer, beta: MTLBuffer,
        sin: MTLBuffer, sout: MTLBuffer, y: MTLBuffer, rows: Int, only: Int = 0, halfIn: Bool = false,
        capture: (rows: [Int], states: [MTLBuffer])? = nil
    ) throws {
        // only: 1 = preparation only, 2 = scan only (for timing).
        precondition(rows > 0 && rows <= maxRows)
        let nch = (rows + Self.C - 1) / Self.C
        var p = (UInt32(rows), UInt32(nch))
        if only != 2 {
            e.setComputePipelineState(try ctx.pipeline(halfIn ? "gdn_cprep_h" : "gdn_cprep"))
            for (i, b) in [q, k, v, gate, beta, W, U, Q, P, K, g].enumerated() { e.setBuffer(b, offset: 0, index: i) }
            e.setBytes(&p, length: 8, index: 11)
            e.dispatchThreadgroups(
                MTLSize(width: nch, height: hv, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
        if only == 1 { return }
        // State after the rows `capture.rows` (indices within the pass) into `capture.states`.
        // GdnCapParams: n, c[4] (chunk of each capture; 0xFFFFFFFF = none), j[4] (row within the chunk).
        var cp = (
            UInt32(0), UInt32(0), UInt32(0), UInt32(0), UInt32(0), UInt32(0), UInt32(0), UInt32(0), UInt32(0)
        )
        if let cap = capture, !cap.rows.isEmpty {
            precondition(
                cap.rows.count <= Self.maxCaptures && cap.rows.count == cap.states.count
                    && cap.rows.allSatisfy { $0 >= 0 && $0 < rows - 1 })
            let r = cap.rows
            let c = (0..<4).map { $0 < r.count ? UInt32(r[$0] / Self.C) : 0xFFFF_FFFF }
            let j = (0..<4).map { $0 < r.count ? UInt32(r[$0] % Self.C) : 0 }
            cp = (UInt32(r.count), c[0], c[1], c[2], c[3], j[0], j[1], j[2], j[3])
            e.setComputePipelineState(try ctx.pipeline(halfIn ? "gdn_capprep_h" : "gdn_capprep"))
            for (i, b) in [k, gate, kcap, gcap].enumerated() { e.setBuffer(b, offset: 0, index: i) }
            e.setBytes(&cp, length: 36, index: 4)
            e.dispatchThreadgroups(
                MTLSize(width: hv, height: r.count, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            e.setComputePipelineState(try ctx.pipeline("gdn_cscan_cap"))
            e.setBuffer(kcap, offset: 0, index: 11)
            e.setBuffer(gcap, offset: 0, index: 12)
            e.setBuffer(cap.states[0], offset: 0, index: 13)
            e.setBuffer(cap.states.count > 1 ? cap.states[1] : dummy, offset: 0, index: 14)
            e.setBuffer(cap.states.count > 2 ? cap.states[2] : dummy, offset: 0, index: 15)
            e.setBuffer(cap.states.count > 3 ? cap.states[3] : dummy, offset: 0, index: 16)
        } else {
            e.setComputePipelineState(try ctx.pipeline("gdn_cscan"))
            for i in 11...16 { e.setBuffer(dummy, offset: 0, index: i) }
        }
        for (i, b) in [sin, sout, W, U, Q, P, K, g, y].enumerated() { e.setBuffer(b, offset: 0, index: i) }
        e.setBytes(&p, length: 8, index: 9)
        e.setBytes(&cp, length: 36, index: 10)
        e.dispatchThreadgroups(
            MTLSize(width: hv * 4, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    }
}
