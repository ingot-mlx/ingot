import Foundation
import Metal

// Qwen3.8-Flash-Next inside HybridModel: the same
// computations as the standalone model (FlashNextModel, retired), but with what HybridModel gives:
// speculative decoding chained with DFlash2, prompt preparation with snapshots, prefix cache and
// `ingot serve`. What is specific to Flash-Next:
// - the residual stream is 4 fp32 streams (hyper-connections, `HyperConn`): before each block, the mixing of the
//   streams; after it, the injection;
// - layer 1 carries the n-gram PLE table: row indices by the hash of the previous two tokens (on the
//   CPU in prefill; on the GPU in the rounds, which may be chained) and rows of ple.bin through the row cache
//   (`FlashPLERows`) on the ExpertStream thread, hidden behind layer 0. The PLE conv carries 9 history rows with
//   ping-pong and tape, like the GDN;
// - the MoE (512 experts top-10, Q2) reads experts from the SSD via ExpertStream with the single record of experts.bin:
//   prefill by groups with the experts fetched ahead, and residents-before-waiting in the rounds.

/// Hyper-connection before a block: 1 + w of its norm (fp32 [4·D]), down|inject [320 (+4), 4·D] and up [4·D, 320]
/// (8 bits g64; BF16 as the exactness reference).
public struct FlashHyperConn {
    public let norm: MTLBuffer
    public let down: MTLBuffer?, up: MTLBuffer?
    public let downQ: DenseQ?, upQ: DenseQ?
    public let inject: Bool
    /// `upQ` with rows reordered (d·4 + s): g comes out with the 4 streams of each d together (fused prefill HC).
    public var interleaved: Bool { upQ != nil }
    public var bytes: Int {
        norm.length + (down?.length ?? 0) + (up?.length ?? 0) + (downQ?.bytes ?? 0) + (upQ?.bytes ?? 0)
    }
}

/// PLE weights and state (layer 1): projections, norms, conv and the conv history (9 rows + the round
/// tape, ping-pong like the GDN).
public final class FlashPLE {
    public let keyProj: DenseQ, valueProj: DenseQ
    public let wk: MTLBuffer, wq: MTLBuffer, wc: MTLBuffer, conv: MTLBuffer
    public let hist: [MTLBuffer]
    public var cur = 0
    public static let window = 9
    /// Width of a history row: HC streams × hidden (ModelDims of the context).
    let W: Int
    init(
        keyProj: DenseQ, valueProj: DenseQ, wk: MTLBuffer, wq: MTLBuffer, wc: MTLBuffer, conv: MTLBuffer,
        hist: [MTLBuffer], width: Int
    ) {
        self.keyProj = keyProj; self.valueProj = valueProj; self.wk = wk; self.wq = wq; self.wc = wc
        self.conv = conv; self.hist = hist; W = width
    }
    public func advance() { cur = 1 - cur }
    public func reset() {
        cur = 0
        memset(hist[0].contents(), 0, hist[0].length)
    }
    public var windowBytes: Int { Self.window * W * 4 }
    public var weightBytes: Int { keyProj.bytes + valueProj.bytes + wk.length + wq.length + wc.length + conv.length }

    /// Window after folding the first `nacc` rows of the tape (rows nacc..nacc + 8 of the current history), written
    /// to `histOut` (snapshot) or, without it, to the next ping-pong (and advances): as after a prefill.
    func encodeFold(_ ctx: MetalContext, _ e: MTLComputeCommandEncoder, nacc: Int, histOut: MTLBuffer? = nil) throws {
        var n4 = UInt32(Self.window * W / 4)
        e.setComputePipelineState(try ctx.pipeline("copy_f32"))
        e.setBuffer(hist[cur], offset: nacc * W * 4, index: 0)
        e.setBuffer(histOut ?? hist[1 - cur], offset: 0, index: 1)
        e.setBytes(&n4, length: 4, index: 2)
        e.dispatchThreads(
            MTLSize(width: Int(n4), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        if histOut == nil { advance() }
    }
}

public final class FlashParts {
    public struct LayerParts {
        public let hcA: FlashHyperConn, hcF: FlashHyperConn
        /// 8-bit router with the shared expert gate in row E (padded to 576 rows); shared gate|up and down.
        public let router: DenseQ, sGU: DenseQ, sDown: DenseQ
    }
    /// Low rank of the hyper-connections (hc_lowrank).
    public let R: Int
    /// Hyper-connection streams (hc_count), PLE heads and width of its embedding (ModelDims).
    public let hcS: Int, pleH: Int, pleDim: Int
    public let fmt: FlashNextFormat
    public let parts: [LayerParts]
    public let finalHC: FlashHyperConn
    public let ple: FlashPLE
    public let pleRows: FlashPLERows
    /// QSA of each attention layer (by layer index).
    public let qsa: [Int: FlashQSA]
    /// Those of `qsa` in layer order (snapshots, rollback, folding).
    public var qsaList: [FlashQSA] { qsa.keys.sorted().map { qsa[$0]! } }
    /// Position of each `qsa` layer in `qsaList` (their windows in the convolution captures come after the
    /// PLE's).
    let qsaIndex: [Int: Int]
    /// RoPE θ (the indexer's is the attention's).
    public let theta: Float
    /// QSA: indexer projection [C, 640], queries [C, 4, 128], raw keys [C, 128], scores of a batch
    /// of rows and bitmap [C, words].
    let qsaQK, qsaQ, qsaRK, qsaS, qsaBM: MTLBuffer
    /// Block lists of the round (verification and decode; prefill keeps the bit mask).
    let qsaLists: BitmapBlockLists
    let qsaNBMax: Int
    public let E: Int, K: Int, I: Int, routerN: Int, hidden: Int, eps: Float
    public var W: Int { hcS * hidden }
    let C: Int
    let ctx: MetalContext
    var vq: VQMV
    /// Streams: `sPool[c]` is the one of chunk c of a prefill group (the first, also the rounds').
    var sPool: [MTLBuffer]
    var S: MTLBuffer
    /// PLE rows of each chunk of a group (prow[c]) and indices (pidx).
    var prowPool: [MTLBuffer]
    // Work buffers (rows ≤ C).
    let nrm, nrmH, dp, low, lowH, inj, g, xh, br, rr: MTLBuffer
    let logitsR, inds, wts, sgate, offsets, sorted, pos, tiles, args, hE, yE, gu, hs, sh: MTLBuffer
    let tilesR, tilesM, argsR, argsM: MTLBuffer
    let pidx, pslot, pemb, kf, vf, gated, normd: MTLBuffer
    /// xh in permuted int8 and its scales per group of 64 (created in the first chunk that uses it).
    var xq8: (q: MTLBuffer, s: MTLBuffer)?

    init(
        ctx: MetalContext, store: SafetensorsStore, config cfg: ModelConfig, chunkRows: Int, layers nl: Int,
        pleCacheBytes: Int, maxCtx: Int, pool: KVPool? = nil
    ) throws {
        self.ctx = ctx
        fmt = try FlashNextFormat(cfg)
        hidden = cfg.hiddenSize
        eps = Float(cfg.rmsNormEps)
        theta = try cfg.requiredSpec.rope.theta
        vq = VQMV(ctx)
        // 6-bit head on NAX (2 fragments, 8 split-K simdgroups).
        vq.nax6 = true
        let dims = ctx.dims
        hcS = dims.hcS; pleH = dims.pleH; pleDim = dims.pleDim
        let R = (cfg.text["hc_lowrank"] as? NSNumber)?.intValue ?? 320
        self.R = R
        precondition(
            fmt.pleHeads == dims.pleH, "Flash-Next: PLE heads differ from those of the context ModelDims")
        let d = ctx.device, W = dims.hcS * cfg.hiddenSize
        E = cfg.numExperts ?? 512; K = cfg.expertsPerToken ?? 10; I = cfg.moeIntermediateSize ?? 640
        routerN = (E + 1 + 63) / 64 * 64
        precondition(E % 32 == 0 && E <= 512 && K <= 16 && I % 64 == 0 && E == fmt.experts)
        let root = String(try store.layerPrefix(0).dropLast("layers.0.".count))
        func f32(_ n: String) throws -> [Float] {
            let t = try store.info(n)
            guard t.dtype == .f32 else { throw SafetensorsError.format("\(n): se esperaba fp32") }
            return store.array(t, as: Float.self)
        }
        func onePlus(_ n: String) throws -> MTLBuffer { try onePlusF32Tensor(ctx, store, n) }
        func bf16(_ names: [String], rows: Int, cols: Int) throws -> MTLBuffer {
            let buf = d.makeBuffer(length: rows * cols * 2, options: .storageModeShared)!
            var off = 0
            for n in names {
                let t = try store.info(n)
                guard t.dtype == .bf16, t.shape.count == 2, t.shape[1] == cols else {
                    throw SafetensorsError.format("\(n): se esperaba bf16 [·, \(cols)]")
                }
                buf.contents().advanced(by: off).copyMemory(from: store.bytes(t).baseAddress!, byteCount: t.byteCount)
                off += t.byteCount
            }
            precondition(off == rows * cols * 2)
            return buf
        }
        func hc(_ pre: String, inject: Bool) throws -> FlashHyperConn {
            let down = [pre + "input_mix_weight_down"] + (inject ? [pre + "block_inject_weight"] : [])
            if store.tensors[pre + "input_mix_weight_down.scales"] != nil {
                let dq = try DenseQ(store: store, config: cfg, paths: down, device: d, padTo: 64)
                let uq = Self.interleaveRows(
                    try DenseQ(store: store, config: cfg, paths: [pre + "input_mix_weight_up"], device: d), groups: 4,
                    device: d)
                guard dq.bits == 8, uq.bits == 8 else { throw SafetensorsError.format("HyperConn de \(dq.bits) bits") }
                return FlashHyperConn(
                    norm: try onePlus(pre + "hc_norm.weight"), down: nil, up: nil, downQ: dq, upQ: uq, inject: inject)
            }
            return FlashHyperConn(
                norm: try onePlus(pre + "hc_norm.weight"),
                down: try bf16(down.map { $0 + ".weight" }, rows: R + (inject ? dims.hcS : 0), cols: W),
                up: try bf16([pre + "input_mix_weight_up.weight"], rows: W, cols: R), downQ: nil, upQ: nil,
                inject: inject)
        }
        var ps: [LayerParts] = []
        var plePart: FlashPLE?
        var qs: [Int: FlashQSA] = [:]
        let Cmax = max(chunkRows, HybridModel.decodeRows)
        C = Cmax
        for i in 0..<nl {
            let pre = try store.layerPrefix(i)
            let m = pre + "mlp."
            let router = try DenseQ(
                store: store, config: cfg, paths: [m + "gate", m + "shared_expert_gate"], device: d, padTo: 64)
            guard router.N == routerN, router.bits == 8 else { throw SafetensorsError.format("Flash-Next router") }
            if cfg.layerTypes[i] == "full_attention" {
                qs[i] = try FlashQSA(ctx: ctx, store: store, prefix: pre, layer: i, maxCtx: maxCtx, pool: pool)
            }
            if i == fmt.pleLayer {
                let p = pre + "ple."
                plePart = FlashPLE(
                    keyProj: try DenseQ(store: store, config: cfg, paths: [p + "key_proj"], device: d),
                    valueProj: try DenseQ(store: store, config: cfg, paths: [p + "value_proj"], device: d),
                    wk: try onePlus(p + "norm_key.weight"), wq: try onePlus(p + "norm_query.weight"),
                    wc: try onePlus(p + "norm_conv.weight"), conv: ctx.buffer(try f32(p + "conv1d.weight")),
                    hist: [
                        ctx.buffer(length: (FlashPLE.window + HybridModel.decodeRows) * W * 4),
                        ctx.buffer(length: (FlashPLE.window + HybridModel.decodeRows) * W * 4),
                    ], width: W)
            }
            ps.append(
                LayerParts(
                    hcA: try hc(pre + "attn_hyper_connection.", inject: true),
                    hcF: try hc(pre + "mlp_hyper_connection.", inject: true), router: router,
                    sGU: try DenseQ(
                        store: store, config: cfg, paths: [m + "shared_expert.gate_proj", m + "shared_expert.up_proj"],
                        device: d),
                    sDown: try DenseQ(store: store, config: cfg, paths: [m + "shared_expert.down_proj"], device: d)))
        }
        parts = ps
        qsa = qs
        qsaIndex = Dictionary(uniqueKeysWithValues: qs.keys.sorted().enumerated().map { ($1, $0) })
        guard let pl = plePart else { throw SafetensorsError.format("Flash-Next without the PLE layer") }
        ple = pl
        for h in ple.hist { memset(h.contents(), 0, h.length) }
        finalHC = try hc(root + "hyper_connection_mixer.", inject: false)
        pleRows = try FlashPLERows(directory: store.directory, fmt: fmt, bytes: pleCacheBytes)
        let Pp = Cmax * K
        func b(_ n: Int) -> MTLBuffer { ctx.buffer(length: n) }
        S = b(Cmax * W * 4)
        sPool = [S]
        prowPool = [b(Cmax * pleH * fmt.pleRowBytes)]
        nrm = b(Cmax * W * 4); nrmH = b(Cmax * W * 2); dp = b(Cmax * 384 * 4); low = b(Cmax * R * 4)
        rr = b(Cmax * 4 * 4)
        lowH = b(Cmax * R * 2); inj = b(Cmax * 4 * 4)
        g = b(Cmax * W * 4); xh = b(Cmax * hidden * 2); br = b(Cmax * hidden * 4)
        logitsR = b(Cmax * routerN * 4); inds = b(Pp * 4); wts = b(Pp * 4); sgate = b(Cmax * 4)
        offsets = b((E + 1) * 4); sorted = b(Pp * 4); pos = b(Pp * 4); tiles = b((Pp + E) * 16); args = b(32)
        tilesR = b((Pp + E) * 16); tilesM = b((Pp + E) * 16); argsR = b(32); argsM = b(32)
        hE = b(Pp * I * 4); yE = b(Pp * hidden * 4); gu = b(Cmax * 2 * I * 4); hs = b(Cmax * I * 2)
        sh = b(Cmax * hidden * 4)
        pidx = b(Cmax * pleH * 4)
        pslot = ctx.buffer((0..<(Cmax * pleH)).map { UInt32($0) })
        pemb = b(Cmax * pleDim * 2); kf = b(Cmax * W * 4); vf = b(Cmax * hidden * 4); gated = b(Cmax * W * 4)
        normd = b(Cmax * W * 4)
        qsaNBMax = (maxCtx + HybridModel.decodeRows) / 4 + 2
        qsaQK = b(Cmax * 640 * 4); qsaQ = b(Cmax * 4 * 128 * 4); qsaRK = b(Cmax * 128 * 4)
        qsaS = b(max(16 * qsaNBMax * 4, 64_000_000))
        qsaBM = b(Cmax * ((qsaNBMax + 31) / 32) * 4)
        qsaLists = BitmapBlockLists(ctx: ctx, stride: Self.qsaListStride, perSplit: Self.qsaListPerSplit)
    }

    /// Rows of W reordered from [g][n] to [n][g] (g < groups): row n·groups + g is the original g·(N/groups) + n.
    static func interleaveRows(_ m: DenseQ, groups: Int, device: MTLDevice) -> DenseQ {
        let N = m.N, per = N / groups
        func perm(_ src: MTLBuffer) -> MTLBuffer {
            let rowBytes = src.length / N
            let dst = device.makeBuffer(length: src.length, options: .storageModeShared)!
            for g in 0..<groups {
                for n in 0..<per {
                    (dst.contents() + (n * groups + g) * rowBytes).copyMemory(
                        from: src.contents() + (g * per + n) * rowBytes, byteCount: rowBytes)
                }
            }
            return dst
        }
        return DenseQ(w: perm(m.w), s: perm(m.s), b: perm(m.b), N: N, K: m.K, bits: m.bits)
    }

    public var weightBytes: Int {
        parts.reduce(0) { $0 + $1.hcA.bytes + $1.hcF.bytes + $1.router.bytes + $1.sGU.bytes + $1.sDown.bytes }
            + finalHC.bytes + ple.weightBytes + qsa.values.reduce(0) { $0 + $1.weightBytes }
    }

    /// Streams and PLE rows of `n` chunks of a group.
    func ensureParts(_ n: Int) {
        while sPool.count < n { sPool.append(ctx.buffer(length: C * W * 4)) }
        while prowPool.count < n { prowPool.append(ctx.buffer(length: C * pleH * fmt.pleRowBytes)) }
    }

    // MARK: - Pieces

    /// Tiles of the per-row reduction kernels (fn_gnorm, fn_ple_gate, fn_bf16_mm): FN_TG threads and FN_NSG
    /// simdgroups of zf_flashnext.metal.
    static let tg = 256, nsg = tg / 32

    func threads(
        _ e: MTLComputeCommandEncoder, _ name: String, _ n: Int, _ bufs: [(MTLBuffer, Int)], _ params: [UInt32]
    ) throws {
        e.setComputePipelineState(try ctx.pipeline(name))
        for (i, (bf, o)) in bufs.enumerated() { e.setBuffer(bf, offset: o, index: i) }
        params.withUnsafeBytes { e.setBytes($0.baseAddress!, length: $0.count, index: bufs.count) }
        e.dispatchThreads(
            MTLSize(width: n, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    /// Y = X·Wᵀ with quantized W: round qmv (≤ 16 rows) or prefill GEMM (M = rows padded to 64).
    func mm(_ e: MTLComputeCommandEncoder, _ m: DenseQ, x: MTLBuffer, y: MTLBuffer, rows: Int, prefill: Bool) throws {
        if !prefill {
            try vq.encode(e, x: x, m, y: y, rows: rows)
        } else if m.bits == 4 {
            try m.encodeQmmF32(ctx, e, x: x, y: y, rows: rows)
        } else {
            precondition(m.bits == 8 && m.N % 64 == 0 && rows % 64 == 0)
            var qp = QmmParams(M: rows, N: m.N, K: m.K)
            e.setComputePipelineState(try ctx.pipeline("qmm_t_nax_f32_b8_g64"))
            for (i, bf) in [x, m.w, m.s, m.b, y].enumerated() { e.setBuffer(bf, offset: 0, index: i) }
            e.setBytes(&qp, length: MemoryLayout<QmmParams>.size, index: 5)
            e.dispatchThreadgroups(
                MTLSize(width: m.N / 64, height: rows / 64, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        }
    }

    // swiftlint:disable:next function_parameter_count - hot path: wrapper of the fn_bf16_mm kernel (its constants)
    func bf16mm(
        _ e: MTLComputeCommandEncoder, x: MTLBuffer, w: MTLBuffer, y: MTLBuffer, M: Int, N: Int, K: Int, ldx: Int,
        ldy: Int
    ) throws {
        var p = (UInt32(M), UInt32(N), UInt32(K), UInt32(ldx), UInt32(ldy), UInt32(0))
        e.setComputePipelineState(try ctx.pipeline("fn_bf16_mm"))
        e.setBuffer(x, offset: 0, index: 0); e.setBuffer(w, offset: 0, index: 1)
        e.setBuffer(y, offset: 0, index: 2)
        e.setBytes(&p, length: 24, index: 3)
        e.dispatchThreadgroups(
            MTLSize(width: (N + Self.nsg - 1) / Self.nsg, height: (M + 15) / 16, depth: 1),
            threadsPerThreadgroup: MTLSize(width: Self.tg, height: 1, depth: 1))
    }

    func gnorm(_ e: MTLComputeCommandEncoder, x: MTLBuffer, xOffset: Int, w: MTLBuffer, o: MTLBuffer, rows: Int) throws
    {
        var p = (UInt32(hidden), eps, UInt32(W), UInt32(W))
        e.setComputePipelineState(try ctx.pipeline("fn_gnorm"))
        e.setBuffer(x, offset: xOffset, index: 0); e.setBuffer(w, offset: 0, index: 1)
        e.setBuffer(o, offset: 0, index: 2)
        e.setBytes(&p, length: 16, index: 3)
        e.setBuffer(nrmH, offset: 0, index: 4)
        e.dispatchThreadgroups(
            MTLSize(width: 4, height: rows, depth: 1),
            threadsPerThreadgroup: MTLSize(width: Self.tg, height: 1, depth: 1))
    }

    /// Hyper-connection before a block: streams `S` (rows from `rowOffset`) → xh half [rows, D] (+ inj if it injects).
    /// `padded`: rows padded to 64 for the prefill 8-bit GEMMs (nil = round).
    /// `injectFirst`: first does the injection of the previous block (fn_hc_inject) inside the norm (exact fusion,
    /// fnx4). `zeroBr`: leaves br at zero in the epilogue if not written in fp32 (replaces fn_zero). Returns whether br
    /// ended at zero.
    @discardableResult
    func hcMix(
        _ e0: MTLComputeCommandEncoder?, _ h: FlashHyperConn, rows: Int, rowOffset: Int = 0, padded: Int? = nil,
        f32: Bool = false, injectFirst: Bool = false, zeroBr: Bool = false
    ) throws -> Bool {
        // Each launch of the prefill fused HC carries its name (hc.norm, hc.down, hc.act, hc.up).
        func run(_: String, _ f: (MTLComputeCommandEncoder) throws -> Void) throws { try f(e0!) }
        let fusable =
            h.downQ != nil && h.interleaved && rowOffset == 0 && (padded != nil ? true : rows <= 16)
        if injectFirst && !fusable {
            try inject(e0!, rows: rows)
        }
        let injF = injectFirst && fusable
        // Prefill with 8-bit HC: fused (no g or n in memory; fn_hc_upmix_nax_r).
        if let M = padded, let dq = h.downQ, let uq = h.upQ, h.interleaved, rowOffset == 0 {
            try run("hc.norm") { e in
                if injF {
                    try self.injNorm(e, h, out: self.rr, rows: rows, rrMode: true)
                } else {
                    var np = (UInt32(self.hidden), self.eps, UInt32(self.W), UInt32(self.W))
                    e.setComputePipelineState(try self.ctx.pipeline("fn_gnorm_rr"))
                    e.setBuffer(self.S, offset: 0, index: 0); e.setBuffer(h.norm, offset: 0, index: 1)
                    e.setBuffer(self.rr, offset: 0, index: 2)
                    e.setBytes(&np, length: 16, index: 3)
                    e.setBuffer(self.nrmH, offset: 0, index: 4)
                    e.dispatchThreadgroups(
                        MTLSize(width: 4, height: rows, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                }
            }
            try run("hc.down") { e in try self.hcDown(e, dq, M: M) }
            let nd = self.R + (h.inject ? self.hcS : 0)
            try run("hc.act") { e in
                try self.threads(
                    e, "fn_hc_act", rows * nd, [(self.dp, 0), (self.low, 0), (self.inj, 0), (self.lowH, 0)],
                    [UInt32(rows), UInt32(R), UInt32(dq.N), UInt32(h.inject ? 1 : 0)])
            }
            let zb = zeroBr && !f32
            try run("hc.up") { e in
                var up = (UInt32(M), UInt32(rows), UInt32(uq.K), UInt32(f32 ? 1 : 0) | (zb ? 2 : 0))
                e.setComputePipelineState(try self.ctx.pipeline("fn_hc_upmix_nax_r"))
                for (i, bf) in [self.lowH, uq.w, uq.s, uq.b, self.S, self.rr, h.norm, self.xh, self.br].enumerated() {
                    e.setBuffer(bf, offset: 0, index: i)
                }
                e.setBytes(&up, length: 16, index: 9)
                e.dispatchThreadgroups(
                    MTLSize(width: uq.N / 64, height: M / 64, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            }
            return zb
        }
        let e = e0!
        // Round with 8-bit HC (fnx4): norm (with the injection), down with 16 columns per threadgroup and up with the
        // activation and the mix inside: 3 launches instead of 5–7, same arithmetic.
        if padded == nil, let dq = h.downQ, let uq = h.upQ, h.interleaved, rowOffset == 0, rows <= 16 {
            if injF {
                try injNorm(e, h, out: nrm, rows: rows, rrMode: false)
            } else {
                try gnorm(e, x: S, xOffset: 0, w: h.norm, o: nrm, rows: rows)
            }
            var vp = VParams(
                M: UInt32(rows), N: UInt32(dq.N), K: UInt32(dq.K), flags: 0, ldx: UInt32(dq.K), ldy: UInt32(dq.N))
            e.setComputePipelineState(try ctx.pipeline("fn_qmv8_nf1_sg16"))
            for (i, bf) in [nrmH, dq.w, dq.s, dq.b, dp].enumerated() { e.setBuffer(bf, offset: 0, index: i) }
            e.setBytes(&vp, length: MemoryLayout<VParams>.size, index: 5)
            e.dispatchThreadgroups(
                MTLSize(width: dq.N / 16, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32 * 16, height: 1, depth: 1))
            let zb = zeroBr && !f32
            var up = (
                UInt32(rows), UInt32(uq.N), UInt32(uq.K), UInt32(dq.N),
                UInt32(f32 ? 1 : 0) | (zb ? 2 : 0) | (h.inject ? 4 : 0)
            )
            e.setComputePipelineState(try ctx.pipeline("fn_hc_upmix_v_nf2_sg5"))
            for (i, bf) in [dp, uq.w, uq.s, uq.b, nrm, xh, br, inj].enumerated() {
                e.setBuffer(bf, offset: 0, index: i)
            }
            e.setBytes(&up, length: 20, index: 8)
            e.dispatchThreadgroups(
                MTLSize(width: uq.N / 32, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32 * 5, height: 1, depth: 1))
            return zb
        }
        try gnorm(e, x: S, xOffset: rowOffset * W * 4, w: h.norm, o: nrm, rows: rows)
        let nd = R + (h.inject ? hcS : 0)
        let ldd = h.downQ?.N ?? 324
        if let dq = h.downQ {
            try mm(e, dq, x: nrmH, y: dp, rows: padded ?? rows, prefill: padded != nil)
        } else {
            try bf16mm(e, x: nrm, w: h.down!, y: dp, M: rows, N: nd, K: W, ldx: W, ldy: ldd)
        }
        try threads(
            e, "fn_hc_act", rows * nd, [(dp, 0), (low, 0), (inj, 0), (lowH, 0)],
            [UInt32(rows), UInt32(R), UInt32(ldd), UInt32(h.inject ? 1 : 0)])
        if let uq = h.upQ {
            try mm(e, uq, x: lowH, y: g, rows: padded ?? rows, prefill: padded != nil)
        } else {
            try bf16mm(e, x: low, w: h.up!, y: g, M: rows, N: W, K: R, ldx: R, ldy: W)
        }
        try threads(
            e, "fn_hc_mix", rows * hidden, [(nrm, 0), (g, 0), (xh, 0), (br, 0)],
            [UInt32(rows), (f32 ? 1 : 0) | (h.interleaved ? 2 : 0)])
        return false
    }

    /// Prefill HC down: dp fp32 [M, dq.N] = nrmH · downᵀ (8 bits).
    /// With the v3 body (X in threadgroup memory, WM × WN grid, `hcDownV3`); fn_qmm8_f32_nax
    /// is the path for shapes that do not fit the grid. Same order of sums per element: bit-exact.
    func hcDown(_ e: MTLComputeCommandEncoder, _ dq: DenseQ, M: Int) throws {
        var qp = QmmParams(M: M, N: dq.N, K: dq.K)
        do {
            var (wm, wn, bk) = hcDownV3
            if (M % (32 * wm)) != 0 { (wm, wn, bk) = (2, 2, 64) }
            if dq.N % (32 * wn) == 0 {
                e.setComputePipelineState(try ctx.pipeline("qmm_t_nax3_8_w\(wm)x\(wn)_bk\(bk)_f32"))
                for (i, bf) in [nrmH, dq.w, dq.s, dq.b, dp].enumerated() { e.setBuffer(bf, offset: 0, index: i) }
                e.setBytes(&qp, length: MemoryLayout<QmmParams>.size, index: 5)
                e.dispatchThreadgroups(
                    MTLSize(width: dq.N / (32 * wn), height: M / (32 * wm), depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 32 * wm * wn, height: 1, depth: 1))
                return
            }
        }
        e.setComputePipelineState(try ctx.pipeline("fn_qmm8_f32_nax"))
        for (i, bf) in [nrmH, dq.w, dq.s, dq.b, dp].enumerated() { e.setBuffer(bf, offset: 0, index: i) }
        e.setBytes(&qp, length: MemoryLayout<QmmParams>.size, index: 5)
        e.dispatchThreadgroups(
            MTLSize(width: dq.N / 64, height: M / 64, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    }

    /// fn_hc_inj_gnorm: x += br·inj and the HC norm (prefill rr or round nrm) with the half copy.
    func injNorm(_ e: MTLComputeCommandEncoder, _ h: FlashHyperConn, out: MTLBuffer, rows: Int, rrMode: Bool) throws {
        var np = (UInt32(hidden), eps, UInt32(W), UInt32(W), UInt32(1 | (rrMode ? 2 : 0)))
        e.setComputePipelineState(try ctx.pipeline("fn_hc_inj_gnorm"))
        for (i, bf) in [S, h.norm, out].enumerated() { e.setBuffer(bf, offset: 0, index: i) }
        e.setBytes(&np, length: 20, index: 3)
        for (i, bf) in [nrmH, br, inj].enumerated() { e.setBuffer(bf, offset: 0, index: 4 + i) }
        e.dispatchThreadgroups(
            MTLSize(width: 4, height: rows, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    func zeroBr(_ e: MTLComputeCommandEncoder, rows: Int) throws {
        var n4 = UInt32(rows * hidden / 4)
        e.setComputePipelineState(try ctx.pipeline("fn_zero"))
        e.setBuffer(br, offset: 0, index: 0)
        e.setBytes(&n4, length: 4, index: 1)
        e.dispatchThreads(
            MTLSize(width: Int(n4), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    func inject(_ e: MTLComputeCommandEncoder, rows: Int) throws {
        try threads(e, "fn_hc_inject", rows * W, [(S, 0), (br, 0), (inj, 0)], [UInt32(rows)])
    }

    /// Embedding of `rows` rows (tokens in `tb` from `tokOffset`) into the 4 streams of `S`.
    func embedStreams(_ e: MTLComputeCommandEncoder, embed: DenseQ, tb: MTLBuffer, tokOffset: Int, rows: Int) throws {
        var ep = (UInt32(hidden), UInt32(1), UInt32(W), UInt32(0))
        e.setComputePipelineState(try ctx.pipeline(embed.embedKernel))
        for s in 0..<4 {
            for (k, (bf, o)) in [
                (tb, tokOffset), (embed.w, 0), (embed.s, 0), (embed.b, 0), (S, s * hidden * 4), (S, 0),
            ]
            .enumerated() { e.setBuffer(bf, offset: o, index: k) }
            e.setBytes(&ep, length: 16, index: 6)
            e.dispatchThreadgroups(
                MTLSize(width: rows, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
    }

    /// PLE indices of a round on the GPU (and the round's tokens into the history by position).
    func encodePleRound(
        _ e: MTLComputeCommandEncoder, tok: MTLBuffer, hist: MTLBuffer, rows: Int, pos: Int, state: MTLBuffer?
    ) throws {
        var p = FnPleRoundParamsSwift(fmt: fmt, rows: rows, pos: pos, dyn: state != nil)
        e.setComputePipelineState(try ctx.pipeline("fn_ple_round"))
        e.setBuffer(tok, offset: 0, index: 0); e.setBuffer(hist, offset: 0, index: 1)
        e.setBuffer(pidx, offset: 0, index: 2)
        e.setBytes(&p, length: MemoryLayout<FnPleRoundParamsSwift>.size, index: 3)
        e.setBuffer(state ?? pidx, offset: 0, index: 4)
        e.dispatchThreads(
            MTLSize(width: rows * pleH, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
    }

    /// PLE over the streams of `T` rows with rows already in `prow`: gather, projections, gate and conv (with tape in
    /// the rounds; `state`: round nacc in st[1]).
    func encodePLE(
        _ e: MTLComputeCommandEncoder, prow: MTLBuffer, rows T: Int, padded M: Int, prefill: Bool, nacc: Int,
        state: MTLBuffer?
    ) throws {
        let p = ple
        try threads(e, "fn_ple_gather", T * pleDim, [(prow, 0), (pslot, 0), (pemb, 0)], [UInt32(T)])
        try mm(e, p.keyProj, x: pemb, y: kf, rows: prefill ? M : T, prefill: prefill)
        try mm(e, p.valueProj, x: pemb, y: vf, rows: prefill ? M : T, prefill: prefill)
        var ep = eps
        e.setComputePipelineState(try ctx.pipeline("fn_ple_gate"))
        for (i, bf) in [kf, vf, S, p.wk, p.wq, p.wc, gated, normd].enumerated() { e.setBuffer(bf, offset: 0, index: i) }
        e.setBytes(&ep, length: 4, index: 8)
        e.dispatchThreadgroups(
            MTLSize(width: 4, height: T, depth: 1), threadsPerThreadgroup: MTLSize(width: Self.tg, height: 1, depth: 1))
        var cp = (UInt32(T), UInt32(prefill ? 0 : nacc), UInt32(prefill ? 0 : 1), UInt32(state != nil ? 1 : 0))
        let a = p.cur, b = 1 - p.cur
        e.setComputePipelineState(try ctx.pipeline("fn_ple_conv"))
        for (i, bf) in [normd, gated, p.conv, p.hist[a], p.hist[b], S].enumerated() {
            e.setBuffer(bf, offset: 0, index: i)
        }
        e.setBytes(&cp, length: 16, index: 6)
        e.setBuffer(state ?? pidx, offset: 0, index: 7)
        e.dispatchThreads(
            MTLSize(width: T * W, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        p.advance()
    }

    /// Split-K simdgroups of the vnax kernels (gate|up, down).
    let vnaxSG: (Int, Int) = (8, 5)
    /// Top-512 plus the open block (and one of margin).
    static let qsaListStride = FlashQSA.topk + 2
    /// Entries of the list per round chunk. Constant, so B = 1 and B = 16 give the same. With 64
    /// it meets fidelity against mlx-vlm at 4K, 16K and 32K and the round at 32K drops by 3.3%
    /// (measured on the M6).
    static let qsaListPerSplit = 64
    /// Grid (WM, WN, BK) of the prefill HC down.
    let hcDownV3: (Int, Int, Int) = (2, 4, 64)
}

// MARK: - MoE: router, experts from the SSD, combination and drafter captures

extension FlashParts {
    /// Router, routing, grouping and shared expert of a layer's MoE over xh (T real rows; M padded in the
    /// prefill). Leaves the shared expert output in `sh` and the tiles in `tiles`/`args`.
    func encodeMoEFront(
        _ e: MTLComputeCommandEncoder, _ l: LayerParts, rows T: Int, padded M: Int, prefill: Bool,
        rowState: MTLBuffer? = nil
    ) throws {
        try mm(e, l.router, x: xh, y: logitsR, rows: prefill ? M : T, prefill: prefill)
        try encodeRouteGroup(e, rows: T, prefill: prefill, rowState: rowState)
        try mm(e, l.sGU, x: xh, y: gu, rows: prefill ? M : T, prefill: prefill)
        try threads(e, "fn_swiglu", T * I, [(gu, 0), (hs, 0)], [UInt32(T), UInt32(I)])
        try mm(e, l.sDown, x: hs, y: sh, rows: prefill ? M : T, prefill: prefill)
    }

    /// Router top-k and shared expert gate of T rows (moe_route_topk_fn: σ of column E of the logits) and
    /// grouping by expert into tiles.
    func encodeRouteGroup(
        _ e: MTLComputeCommandEncoder, rows T: Int, prefill: Bool, rowState: MTLBuffer? = nil
    ) throws {
        var rp = (UInt32(T), UInt32(E), UInt32(routerN), UInt32(K))
        e.setComputePipelineState(try ctx.pipeline("moe_route_topk_fn"))
        // x and shared_expert_gate (1–4) are not read with the gate in the logits. A single call for the 8 buffers
        // (round-encoding CPU).
        e.setBuffers(
            [logitsR, logitsR, logitsR, logitsR, logitsR, inds, wts, sgate], offsets: [0, 0, 0, 0, 0, 0, 0, 0],
            range: 0..<8)
        e.setBytes(&rp, length: 16, index: 8)
        e.dispatchThreadgroups(
            MTLSize(width: (T + 3) / 4, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        // Grouping (moe_group, no shared expert): the same tiles for gate|up and down. Prefill: BM rows for the
        // NAX kernels (gate|up in blocks of 64 columns of I, down of 128 of D); round: ≤ 16 rows for the vnax
        // (blocks of 16 and 32 columns). Indirect grids in args[0..2] (gate|up) and args[4..6] (down).
        let bm = prefill ? naxBM(rows: T) : 0
        let tr = bm > 0 ? bm : 16, nxA = bm > 0 ? I / 64 : I / 16, nxB = bm > 0 ? hidden / 128 : hidden / 32
        // With `rowState`, only the pairs of the first st[6] rows (the others are never accepted).
        var gp = (
            UInt32(T * K), UInt32(E), UInt32(tr), UInt32(tr), UInt32(nxA), UInt32(nxB), UInt32(0),
            rowState != nil ? UInt32(2 | K << 8) : 0
        )
        e.setComputePipelineState(try ctx.pipeline("moe_group"))
        e.setBuffers(
            [inds, offsets, sorted, pos, tiles, tiles, args, args], offsets: [0, 0, 0, 0, 0, 0, 0, 16], range: 0..<8)
        e.setBytes(&gp, length: 32, index: 8)
        e.setBuffer(rowState ?? inds, offset: 0, index: 9)
        e.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
    }

    /// Rows per tile of the prefill NAX kernels (0 = the plain ones): ~T·K/E rows per expert.
    func naxBM(rows T: Int) -> Int {
        let per = T * K / E
        // Measured with fnprof (chunk of 2048, ~40 rows per expert): BM 16 671 ms, 32 559, 48 763, 64 896.
        return per >= 12 ? 32 : 16
    }

    /// gate|up and down of tiles `tl`/`ar` with the experts in `arena` through the `smap` table (from `smapOff`).
    /// `nax`: tiles of `nax` rows with the NAX kernels (prefill); h goes in half over `hE`. `i8` (with `nax`): int8 xh.
    func encodeExperts(
        _ e: MTLComputeCommandEncoder, arena: MTLBuffer, smap: MTLBuffer, smapOff: Int, tiles tl: MTLBuffer,
        args ar: MTLBuffer, rows T: Int, nax: Int = 0, i8: Bool = false
    ) throws {
        if nax > 0 && i8 {
            try encodeExpertsI8(e, arena: arena, smap: smap, smapOff: smapOff, tiles: tl, args: ar, rows: T, nax: nax)
            return
        }
        if nax > 0 {
            var qp = (
                UInt32(fmt.recordBytes), UInt32(fmt.gateCodes), UInt32(fmt.gateScales), UInt32(fmt.upCodes),
                UInt32(fmt.upScales), UInt32(fmt.downCodes), UInt32(fmt.downScales), UInt32(T)
            )
            e.setComputePipelineState(try ctx.pipeline("fn_moe_gateup_nax_bm\(nax)_bk64"))
            for (k, (bf, o)) in [(xh, 0), (arena, 0), (smap, smapOff), (tl, 0), (sorted, 0), (hE, 0)].enumerated() {
                e.setBuffer(bf, offset: o, index: k)
            }
            e.setBytes(&qp, length: 32, index: 6)
            e.dispatchThreadgroups(
                indirectBuffer: ar, indirectBufferOffset: 0,
                threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            e.setComputePipelineState(try ctx.pipeline("fn_moe_down_nax_bm\(nax)_bk64"))
            for (k, (bf, o)) in [(hE, 0), (arena, 0), (smap, smapOff), (tl, 0), (yE, 0)].enumerated() {
                e.setBuffer(bf, offset: o, index: k)
            }
            e.setBytes(&qp, length: 32, index: 5)
            e.dispatchThreadgroups(
                indirectBuffer: ar, indirectBufferOffset: 16,
                threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            return
        }
        var qp = (
            UInt32(fmt.recordBytes), UInt32(fmt.gateCodes), UInt32(fmt.gateScales), UInt32(fmt.upCodes),
            UInt32(fmt.upScales), UInt32(fmt.downCodes), UInt32(fmt.downScales), UInt32(T)
        )
        let tpgA = MTLSize(width: 32 * vnaxSG.0, height: 1, depth: 1)
        let tpgB = MTLSize(width: 32 * vnaxSG.1, height: 1, depth: 1)
        e.setComputePipelineState(
            try ctx.pipeline("fn_moe_gateup_vnax_sg\(vnaxSG.0)"))
        for (k, (bf, o)) in [(xh, 0), (arena, 0), (smap, smapOff), (tl, 0), (sorted, 0), (hE, 0)].enumerated() {
            e.setBuffer(bf, offset: o, index: k)
        }
        e.setBytes(&qp, length: 32, index: 6)
        e.dispatchThreadgroups(indirectBuffer: ar, indirectBufferOffset: 0, threadsPerThreadgroup: tpgA)
        e.setComputePipelineState(
            try ctx.pipeline("fn_moe_down_vnax_sg\(vnaxSG.1)"))
        for (k, (bf, o)) in [(hE, 0), (arena, 0), (smap, smapOff), (tl, 0), (yE, 0)].enumerated() {
            e.setBuffer(bf, offset: o, index: k)
        }
        e.setBytes(&qp, length: 32, index: 5)
        e.dispatchThreadgroups(indirectBuffer: ar, indirectBufferOffset: 16, threadsPerThreadgroup: tpgB)
    }

    /// Combination + injection (fnx4, exact: combination with the shared expert + fn_hc_inject, bypassing br).
    func encodeCombineInject(_ e: MTLComputeCommandEncoder, rows T: Int) throws {
        try threads(
            e, "fn_moe_combine_inj", T * hidden, [(yE, 0), (pos, 0), (wts, 0), (sh, 0), (sgate, 0), (S, 0), (inj, 0)],
            [UInt32(T)])
    }

    /// Tiles of the residents (smapR) and of those that are missing.
    func encodeSplit(_ e: MTLComputeCommandEncoder, smapR: MTLBuffer, smapOff: Int) throws {
        e.setComputePipelineState(try ctx.pipeline("fn_moe_split"))
        for (k, (bf, o)) in [(tiles, 0), (args, 0), (smapR, smapOff), (tilesR, 0), (tilesM, 0), (argsR, 0), (argsM, 0)]
            .enumerated()
        { e.setBuffer(bf, offset: o, index: k) }
        e.dispatchThreads(
            MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
    }

    /// Drafter capture: mean of the 4 streams of `rows` rows in half, columns of layer `slot` of `caps`.
    func encodeCapture(_ e: MTLComputeCommandEncoder, caps: MTLBuffer, width: Int, slot j: Int, rows: Int) throws {
        var p = SIMD2<UInt32>(UInt32(rows), UInt32(width))
        e.setComputePipelineState(try ctx.pipeline("fn_stream_mean"))
        e.setBuffer(S, offset: 0, index: 0)
        e.setBuffer(caps, offset: j * hidden * 2, index: 1)
        e.setBytes(&p, length: 8, index: 2)
        e.dispatchThreads(
            MTLSize(width: rows * hidden, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
}

/// FnPleRoundParams of zf_flashnext.metal.
struct FnPleRoundParamsSwift {
    var mult: (UInt64, UInt64, UInt64)
    var size:
        (
            UInt64, UInt64, UInt64, UInt64, UInt64, UInt64, UInt64, UInt64, UInt64, UInt64, UInt64, UInt64, UInt64,
            UInt64, UInt64, UInt64
        )
    var off:
        (
            UInt64, UInt64, UInt64, UInt64, UInt64, UInt64, UInt64, UInt64, UInt64, UInt64, UInt64, UInt64, UInt64,
            UInt64, UInt64, UInt64
        )
    var rows: UInt32, eos: UInt32, pos: UInt32, dyn: UInt32
    init(fmt: FlashNextFormat, rows: Int, pos: Int, dyn: Bool) {
        // 16 slots (FN_PH ≤ 16); the surplus ones, zeroed.
        let m = fmt.pleMult, s = fmt.pleSize + Array(repeating: 0, count: 16 - fmt.pleSize.count)
        let o = fmt.pleOff + Array(repeating: 0, count: 16 - fmt.pleOff.count)
        mult = (m[0], m[1], m[2])
        size = (s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8], s[9], s[10], s[11], s[12], s[13], s[14], s[15])
        off = (o[0], o[1], o[2], o[3], o[4], o[5], o[6], o[7], o[8], o[9], o[10], o[11], o[12], o[13], o[14], o[15])
        self.rows = UInt32(rows); eos = fmt.pleEos; self.pos = UInt32(pos); self.dyn = dyn ? 1 : 0
    }
}
