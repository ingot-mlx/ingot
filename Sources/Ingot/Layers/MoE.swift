import Foundation
import Metal

// MoE of Qwen3.6-35B-A3B (256 experts × 512, top-8, shared 512) and of LFM2.5 (64 × 1536, top-4, sigmoid router,
// no shared expert): MoELayer for decode and verification (B ≤ 16 rows, expert union); PrefillMoE, for prefill
// (GEMMs grouped by expert), is in MoEPrefill.swift. The router (RouterSpec) picks the kernels (MoEKernelSet). Kernels
// in Shaders/verify.metal, verify_nax.metal, qmm_nax_moe.metal and yw_lfm.metal. Flash-Next goes through FlashParts.

// MARK: - Kernels per router

/// Kernels of the Qwen3.6 and LFM2 MoE available for each router (RouterSpec) and the shapes they accept. A shape
/// without kernels gives a clear error at load. With softmax and a gated shared expert, E and top-k come from the
/// config (LAB_MOE_E and LAB_MOE_K in vmoe_route, moe_route_topk): E multiple of 32, ≤ 512, top-k ≤ 16, inter % 64.
struct MoEKernelSet {
    /// Routing of the round (≤ 16 rows) and top-k of the prefill.
    let route: String, prefillRoute: String
    /// Prefill combine without shared expert (with shared expert: moe_combine / moe_combine_acc).
    let combineNoShared: String?

    static func forRouter(_ r: RouterSpec, inter: Int) throws -> MoEKernelSet {
        // Logits read with the row stride of the router padded to 64 (MoELayer.routerLD): E multiple of 32.
        let shapeOK = r.experts % 32 == 0 && r.experts <= 512 && r.topK <= 16 && inter % 64 == 0
        switch (r.scoring, r.shared) {
        case (.softmax, _) where shapeOK && !r.selectBias:
            // Gated shared expert, ungated or none (LAB_MOE_SHARED): without it, the prefill combine
            // is the LFM2 one.
            return MoEKernelSet(
                route: "vmoe_route", prefillRoute: "moe_route_topk_cfg",
                combineNoShared: r.shared == .none ? "moe_combine_acc_ns" : nil)
        case (.sigmoid, .none) where shapeOK && r.selectBias && r.scale == 1 && r.normEps == 1e-6 && !r.logitBias:
            // LFM2: the common sigmoid router kernels (before lfm_moe_route and lfm_route_topk: same bits).
            return MoEKernelSet(
                route: "vmoe_route", prefillRoute: "moe_route_topk_cfg", combineNoShared: "moe_combine_acc_ns")
        case (.sigmoid, _) where shapeOK && !r.logitBias:
            // Any other sigmoid router (GLM-4.7, Nemotron-H): with or without shared expert,
            // selection bias, renormalization, scale and epsilon as macros (LAB_MOE_SCORE…) of the common kernels.
            return MoEKernelSet(
                route: "vmoe_route", prefillRoute: "moe_route_topk_cfg",
                combineNoShared: r.shared == .none ? "moe_combine_acc_ns" : nil)
        default:
            throw SafetensorsError.format(
                "unsupported MoE: router \(r.scoring) (selection bias \(r.selectBias)), shared expert \(r.shared), "
                    + "\(r.experts) experts top-\(r.topK) of \(inter) (there are kernels for softmax without a bias and "
                    + "for sigmoid, with E a multiple of 32 up to 512, top-k up to 16 and an intermediate multiple of 64)"
            )
        }
    }
}

// MARK: - Verification MoE

/// MoE block of one layer with B ≤ 16 rows: 8-bit or bf16 router (fp32 logits) → top-8 + expert union on the GPU →
/// gate+up+SwiGLU per distinct expert (each expert is read once for all its rows) → down + combine.
public final class MoELayer {
    /// NAX kernels (matmul2d with 16-row fragments): gate+up with 16 columns (nf1) and 16 split-K simdgroups;
    /// down per (expert × 16 columns) with 16 simdgroups and a separate sum of the 9 slots
    /// (vnax_moe_down_split + vmoe_combine9; B = 8: down 0.135 → 0.115 ms per layer).
    /// 16 columns and 16 split-K simdgroups in both for every family (it was 32 columns and 8
    /// simdgroups except for LFM2 and GLM-4.7 top-4). ingot-dev gen --mode plain, ABABAB, ms per token at 512/4K:
    /// gpt-oss 20b -5.0/-4.4 %, Nemotron 3.5 Lightning -5.1/-5.3 %, Gemma 4 26B-A4B -1.4/-1.2 %, Qwen3.6 MoE
    /// -1.4/-1.9 %; Qwen3.6 MoE DFlash2 rounds of 8 rows -0.5/-0.5 % and 16 rows +0.4/-1.5 %. It changes the order
    /// of the split-K sum (other bits; Gemma 26B's 4K greedy changes at a near-tie, token 71).
    private static let nGuNF = 1, nGuSGS = 16, sDnNF = 1, sDnSGS = 16
    private static let guCols = 16 * nGuNF

    public static let maxRows = 16
    public let ctx: MetalContext
    public let layer: Int
    public let router, sharedGate, sGate, sUp, sDown: DenseQ
    public let eGate, eUp, eDown: DeviceQuantizedLinear
    /// Experts and shared expert with a gate (false: relu² without gate, where sGate and eGate are sUp and eUp).
    let gated: Bool
    public let hidden: Int, inter: Int, experts: Int, topk: Int
    /// Intermediate of the shared expert (that of the experts if there is none).
    public let sinter: Int
    public let logits, entries, pairs, args, inds, wts, h: MTLBuffer
    /// SSD expert streaming: the union split into resident and missing experts, with their grids (only with `stream`).
    public let entriesR, argsR, entriesM, argsM: MTLBuffer?
    /// SSD expert streaming: table of this layer's residents in `stream.smapR`.
    public let smapR: (buffer: MTLBuffer, offset: Int)?
    /// Expert pass (SSD expert streaming): the residents (uncombined) or the missing ones (and the combine).
    public enum Pass { case resident, missing }
    /// Slots of the per-expert down [top-k + 1, 16, hidden] fp32 (shared across layers: used in series).
    nonisolated(unsafe) public static var partBuffers: [ObjectIdentifier: MTLBuffer] = [:]
    public let part: MTLBuffer
    public var vq: VQMV
    /// Router (ModelSpec). Without shared expert (LFM2), sGate, sUp, sDown and sharedGate are empty matrices the
    /// kernels do not read: no entry carries the shared expert id.
    public let routerSpec: RouterSpec
    let kernels: MoEKernelSet
    public let expertBias: MTLBuffer?
    /// Router linear bias [E] fp32 (RouterSpec.logitBias, gpt-oss; LAB_MOE_LBIAS): bound where the selection bias goes.
    let routerBias: MTLBuffer?
    /// Expert biases fp32 (RouterSpec.expertBiases, gpt-oss; LAB_MOE_EBIAS): gate and up [E, inter], down [E, hidden].
    let eBias: ExpertBiases?
    /// Gemma 4 (RouterSpec.ownNorm): RMSNorm weight of the router's own input, `router.scale` · hidden^-0.5 rounded as
    /// mlx-lm does (both factors in bf16); the model normalizes the residual stream with it before `.router`.
    public let routerNorm: MTLBuffer?
    /// Gemma 4 (RouterSpec.perExpertScale; LAB_MOE_PES): `per_expert_scale` [E] fp32, multiplied into the weight of
    /// each chosen expert after the renormalization.
    let pes: MTLBuffer?
    /// SSD expert streaming: expert → slot table read by the kernels (buffer, offset): identity, or this layer's in
    /// the `ExpertStream` store (the expert weights are then the store's buffers).
    public let smap: (buffer: MTLBuffer, offset: Int)
    public let stream: ExpertStream?

    /// The identity table (expert e in slot e), shared per context.
    nonisolated(unsafe) static var identityMaps: [ObjectIdentifier: MTLBuffer] = [:]
    public static func identityMap(_ ctx: MetalContext) -> MTLBuffer {
        let key = ObjectIdentifier(ctx)
        return MetalContext.scratchLock.withLock {
            if let b = identityMaps[key] { return b }
            let b = ctx.buffer((0..<1024).map { UInt32($0) })
            identityMaps[key] = b
            return b
        }
    }

    public init(
        ctx: MetalContext, store: SafetensorsStore, config cfg: ModelConfig, layer: Int, stream: ExpertStream? = nil
    ) throws {
        self.ctx = ctx
        self.layer = layer
        vq = VQMV(ctx)
        let sp = try cfg.requiredSpec
        guard let rs = sp.router else { throw SafetensorsError.format("layer \(layer): the model has no MoE") }
        routerSpec = rs
        let pre = try store.layerPrefix(layer) + (sp.names.moe ?? sp.names.ffn)
        let d = ctx.device
        // Experts and shared expert without gate (Nemotron-H's relu²): the gate is the up matrix itself (same buffers,
        // no extra memory) so the kernels get valid arguments; their `_relu2` instances do not read it.
        gated = sp.names.expertGate != nil
        guard gated == (sp.block.activation != .relu2), gated || stream == nil else {
            throw SafetensorsError.format(
                "layer \(layer): MoE experts \(gated ? "with" : "without") gate and activation \(sp.block.activation)"
                    + (stream != nil ? " (SSD expert streaming needs gated experts)" : ""))
        }
        let gateName = sp.names.expertGate ?? sp.names.expertUp
        router = try Self.loadRouter(store: store, config: cfg, path: pre + sp.names.router, device: d)
        if rs.shared == .none {
            let empty = DenseQ.empty(device: d, K: cfg.hiddenSize)
            sharedGate = DenseQ.empty(device: d, K: cfg.hiddenSize, bits: 8); sGate = empty; sUp = empty; sDown = empty
        } else {
            // Ungated: an empty gate nobody reads (vmoe_route with LAB_MOE_SHARED = 2 gives weight 1).
            sharedGate =
                rs.shared == .ungated
                ? DenseQ.empty(device: d, K: cfg.hiddenSize, bits: 8)
                : try DenseQ(store: store, config: cfg, paths: [pre + "shared_expert_gate"], device: d)
            let se = pre + sp.names.sharedExpert
            sUp = try DenseQ(store: store, config: cfg, paths: [se + "up_proj"], device: d)
            sGate = gated ? try DenseQ(store: store, config: cfg, paths: [se + "gate_proj"], device: d) : sUp
            sDown = try DenseQ(store: store, config: cfg, paths: [se + "down_proj"], device: d)
        }
        expertBias = rs.selectBias ? try Self.loadSelectBias(store: store, prefix: pre, device: d) : nil
        routerBias = rs.logitBias ? try floatTensor(store, pre + sp.names.router + ".bias", d) : nil
        eBias = rs.expertBiases ? try ExpertBiases(store: store, prefix: pre + sp.names.experts, device: d) : nil
        routerNorm =
            rs.ownNorm ? try Self.loadRouterNorm(store: store, name: pre + sp.names.routerScale, device: d) : nil
        pes = rs.perExpertScale ? try floatTensor(store, pre + sp.names.perExpertScale, d) : nil
        func load(_ n: String) throws -> DeviceQuantizedLinear {
            try DeviceQuantizedLinear(
                QuantizedLinear(store: store, path: pre + n, config: cfg, mxfp4: true), store: store, device: d)
        }
        /// SSD expert streaming: experts are not loaded; their weights are the store's buffers (chunks 3i, 3i+1, 3i+2).
        func streamed(_ n: String, _ i: Int, _ s: ExpertStream) throws -> DeviceQuantizedLinear {
            DeviceQuantizedLinear(
                synthetic: try QuantizedLinear(store: store, path: pre + n, config: cfg), w: s.arenas[3 * i],
                s: s.arenas[3 * i + 1], b: s.arenas[3 * i + 2])
        }
        self.stream = stream
        if let s = stream {
            eGate = try streamed(sp.names.experts + gateName, 0, s)
            eUp = try streamed(sp.names.experts + sp.names.expertUp, 1, s)
            eDown = try streamed(sp.names.experts + sp.names.expertDown, 2, s)
            smap = (s.smap, layer * s.experts * 4)
        } else {
            eUp = try load(sp.names.experts + sp.names.expertUp)
            eGate = gated ? try load(sp.names.experts + gateName) : eUp
            eDown = try load(sp.names.experts + sp.names.expertDown)
            smap = (Self.identityMap(ctx), 0)
        }
        hidden = eGate.layer.inFeatures; inter = eGate.layer.outFeatures; experts = eGate.layer.experts
        sinter = rs.shared == .none ? inter : sGate.N
        topk = cfg.expertsPerToken ?? 8
        kernels = try MoEKernelSet.forRouter(rs, inter: inter)
        guard router.bits == 8 || router.bits == 16, sharedGate.bits == 8, router.N == Self.routerLD(experts),
            experts == rs.experts,
            topk == rs.topK,
            hidden % 64 == 0, hidden >= 128, sinter % 64 == 0, eBias?.matches(experts, inter, hidden) ?? true,
            (pes?.length ?? Int.max) >= experts * 4, (routerNorm?.length ?? Int.max) >= hidden * 2,
            rs.shared == .none || (sinter == rs.sharedInter && sDown.K == sinter),
            eGate.layer.params.bits == 4, eUp.layer.params.bits == 4, eDown.layer.params.bits == 4
        else {
            throw SafetensorsError.format("unsupported MoE shape")
        }
        // The shared expert is read inside the experts' fused kernels (vnax_moe_gateup / vnax_moe_down_split and the
        // prefill tiles), at 4 bits: any other width would load and be read wrong.
        guard rs.shared == .none || [sGate, sUp, sDown].allSatisfy({ $0.bits == 4 }) else {
            throw SafetensorsError.format(Self.sharedBitsMessage(layer: layer, [sGate.bits, sUp.bits, sDown.bits]))
        }
        // MXFP4 experts (gpt-oss): the three of them, without a shared expert (the `_mxfp4` instances read the e8m0
        // scales of the experts only). Otherwise groups of 64 in the experts and the shared expert: the only affine
        // instances of the round and prefill kernels.
        let fp4 = [eGate, eUp, eDown].map(\.layer.isMXFP4)
        if fp4.contains(true) {
            guard !fp4.contains(false), rs.shared == .none else {
                throw SafetensorsError.format(
                    "layer \(layer): MXFP4 experts need gate, up and down in MXFP4 and no shared expert")
            }
        } else {
            let groups =
                [eGate.layer.params.groupSize, eUp.layer.params.groupSize, eDown.layer.params.groupSize]
                + (rs.shared == .none ? [] : [sGate.gs, sUp.gs, sDown.gs])
            guard groups.allSatisfy({ $0 == 64 }) else {
                throw SafetensorsError.format(
                    "layer \(layer): MoE in groups of \(groups.map(String.init).joined(separator: "/")) (experts gate/"
                        + "up/down, then shared): only groups of 64 are supported")
            }
        }
        let R = Self.maxRows, P = R * topk + R
        logits = ctx.buffer(length: R * router.N * 4)
        entries = ctx.buffer(length: (R * topk + 1) * 16)
        pairs = ctx.buffer(length: P * 8)
        args = ctx.buffer(length: 32)
        let key = ObjectIdentifier(ctx), partBytes = (topk + 1) * 16 * hidden * 4
        part = MetalContext.scratchLock.withLock {
            if let b = Self.partBuffers[key] { return b }
            let b = ctx.buffer(length: partBytes)
            Self.partBuffers[key] = b
            return b
        }
        inds = ctx.buffer(length: R * topk * 4)
        wts = ctx.buffer(length: R * topk * 4)
        h = ctx.buffer(length: (R * topk * inter + R * max(inter, sinter)) * 4)
        if let s = stream {
            entriesR = ctx.buffer(length: (R * topk + 1) * 16); entriesM = ctx.buffer(length: (R * topk + 1) * 16)
            argsR = ctx.buffer(length: 32); argsM = ctx.buffer(length: 32)
            smapR = (s.smapR, layer * s.experts * 4)
        } else {
            entriesR = nil; entriesM = nil; argsR = nil; argsM = nil; smapR = nil
        }
    }

    /// Load error for a shared expert that is not 4-bit (MoELayer and PrefillMoE).
    static func sharedBitsMessage(layer: Int?, _ bits: [Int]) -> String {
        (layer.map { "layer \($0): " } ?? "")
            + "shared expert at \(bits.map(String.init).joined(separator: "/")) bits (gate/up/down): only 4-bit is "
            + "supported, it is fused with the experts' kernels"
    }

    /// Router: 8-bit as is (vnax_qmv8 and qmm_t_nax_f32_b8_g64); bf16 as is (Nemotron-H and GLM ship it in bf16:
    /// vnax_qmvbf16 and qmm_t_nax_f32_bf16; quantized to 8 bits it changed the Lightning's top-6 in 3.9 % of the
    /// rows); fp32 (LFM2.5-8B) quantized to 8 bits at load, lossy versus mlx-lm, which applies it in fp32: model
    /// fidelity measures it. Any other width is a load error.
    static func loadRouter(store: SafetensorsStore, config cfg: ModelConfig, path: String, device d: MTLDevice) throws
        -> DenseQ
    {
        // Output rows zero-padded up to a multiple of 64 (the prefill router GEMMs use 64-column tiles: with 32
        // experts, LFM2.5-8B, the grid came out empty). The router kernels read the first E rows
        // with stride routerLD.
        if store.tensors[path + ".scales"] == nil {
            if try store.info(path + ".weight").dtype == .bf16 {
                return try DenseQ(bf16Rows: store, name: path + ".weight", padTo: 64, device: d)
            }
            return DenseQ.padRows(
                try DenseQ.quantize(store: store, name: path + ".weight", device: d, bits: 8), to: 64, device: d)
        }
        let q = try DenseQ(store: store, config: cfg, paths: [path], device: d, padTo: 64)
        guard q.bits == 8 else {
            throw SafetensorsError.format(
                "\(path): router at \(q.bits) bits: only 8-bit or unquantized routers are supported")
        }
        return q
    }

    /// Gemma 4's router norm weight: mlx-lm's `self.scale * self._root_size` in bf16 (the Python scalar hidden^-0.5 becomes
    /// bf16 first, then the product is rounded to bf16), bf16 [hidden] like the other norm weights (LAB_NWT).
    static func loadRouterNorm(store: SafetensorsStore, name: String, device d: MTLDevice) throws -> MTLBuffer {
        let t = try store.info(name)
        guard t.dtype == .bf16, t.shape.count == 1 else {
            throw SafetensorsError.format("\(name): expected bf16 [hidden]")
        }
        let root = bf16ToFloat(floatToBF16(1 / Float(t.shape[0]).squareRoot()))
        let w = store.array(t, as: UInt16.self).map { floatToBF16(bf16ToFloat($0) * root) }
        return w.withUnsafeBytes { d.makeBuffer(bytes: $0.baseAddress!, length: $0.count)! }
    }

    /// Selection bias of the router (RouterSpec.selectBias), fp32: `expert_bias` (LFM2) or `gate.e_score_correction_bias`
    /// (GLM, Nemotron-H).
    static func loadSelectBias(store: SafetensorsStore, prefix pre: String, device d: MTLDevice) throws -> MTLBuffer {
        let name = ["expert_bias", "gate.e_score_correction_bias"].first { store.tensors[pre + $0] != nil }
        return try floatTensor(store, pre + (name ?? "expert_bias"), d)
    }

    /// Row stride of the router logits: E rounded up to 64 (MOE_LD in ab_moe_route.metal).
    static func routerLD(_ experts: Int) -> Int { (experts + 63) / 64 * 64 }

    /// SSD expert streaming: splits the expert union (after `.route`) into resident and missing experts.
    public func encodeSplit(_ enc: MTLComputeCommandEncoder) throws {
        guard let eR = entriesR, let aR = argsR, let eM = entriesM, let aM = argsM, let sR = smapR else {
            preconditionFailure("expert streaming without stream")
        }
        var E = UInt32(experts)
        enc.setComputePipelineState(try ctx.pipeline("expert_split"))
        enc.setBuffer(entries, offset: 0, index: 0)
        enc.setBuffer(args, offset: 0, index: 1)
        enc.setBuffer(sR.buffer, offset: sR.offset, index: 2)
        enc.setBuffer(eR, offset: 0, index: 3)
        enc.setBuffer(aR, offset: 0, index: 4)
        enc.setBuffer(eM, offset: 0, index: 5)
        enc.setBuffer(aM, offset: 0, index: 6)
        enc.setBytes(&E, length: 4, index: 7)
        enc.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    /// Bytes of one expert (gate + up + down, weights + scales + biases) and of the rest the layer always reads.
    public var expertBytes: Int {
        func q(_ l: DeviceQuantizedLinear) -> Int {
            (l.layer.weight.byteCount + l.layer.scales.byteCount * (l.layer.isMXFP4 ? 1 : 2)) / experts
        }
        return (gated ? q(eGate) : 0) + q(eUp) + q(eDown)
    }
    public var fixedBytes: Int {
        router.bytes + sharedGate.bytes + (gated ? sGate.bytes : 0) + sUp.bytes + sDown.bytes
    }

    public enum Piece: String, CaseIterable { case router, route, gateup, down }

    /// Suffix of the expert kernel instances of the round: "_mxfp4" with MXFP4 experts (LAB_MXFP4), none if affine.
    var fp4Suffix: String { eGate.layer.isMXFP4 ? "_mxfp4" : "" }

    private func params(_ rows: Int, accumulate: Bool) -> (UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32) {
        // flags: bit 0 accumulate; 8–15 down column blocks; 16–23 top-k; 24–31 slots (top-k + shared).
        // Gate+up tiles: those of the larger of the two intermediates.
        (
            UInt32(hidden), UInt32(inter), UInt32(experts), UInt32(rows), UInt32(max(inter, sinter) / Self.guCols),
            (accumulate ? 1 : 0) | UInt32(hidden / (16 * Self.sDnNF)) << 8 | UInt32(topk) << 16 | UInt32(
                topk + (routerSpec.shared != .none ? 1 : 0)) << 24, UInt32(sinter)
        )
    }

    /// x: half [rows, hidden]; out: float [rows, hidden] (with `accumulate`, out += MoE).
    /// With `rowLimit` (chained-round state), only the first st[6] rows count (rows chosen on the
    /// GPU): routing, expert union and combine ignore the others.
    public func encode(
        _ enc: MTLComputeCommandEncoder, x: MTLBuffer, xOffset: Int = 0, out: MTLBuffer, outOffset: Int = 0,
        rows: Int, accumulate: Bool = false, only: Piece? = nil, rowLimit: MTLBuffer? = nil, pass: Pass? = nil
    ) throws {
        // With `pass`, gate+up and down run over the half of the union that applies (and its slot table); the
        // combine, only with the missing ones (the last).
        let ent = pass == .resident ? entriesR! : pass == .missing ? entriesM! : entries
        let ags = pass == .resident ? argsR! : pass == .missing ? argsM! : args
        let smp = pass == .resident ? smapR! : smap
        precondition(rows >= 1 && rows <= Self.maxRows)
        var mp = params(rows, accumulate: accumulate)
        if rowLimit != nil { mp.5 |= 2 }
        let rst = rowLimit ?? logits
        if only == nil || only == .router {
            try vq.encode(enc, x: x, xOffset: xOffset, router, y: logits, rows: rows)
        }
        if only == nil || only == .route {
            enc.setComputePipelineState(try ctx.pipeline(kernels.route))
            enc.setBuffer(logits, offset: 0, index: 0)
            enc.setBuffer(x, offset: xOffset, index: 1)
            enc.setBuffer(sharedGate.w, offset: 0, index: 2)
            enc.setBuffer(sharedGate.s, offset: 0, index: 3)
            enc.setBuffer(sharedGate.b, offset: 0, index: 4)
            for (i, b) in [entries, pairs, args, inds, wts].enumerated() { enc.setBuffer(b, offset: 0, index: 5 + i) }
            enc.setBytes(&mp, length: 28, index: 10)
            enc.setBuffer(rst, offset: 0, index: 11)
            if let b = expertBias ?? routerBias { enc.setBuffer(b, offset: 0, index: 12) }
            if let b = pes { enc.setBuffer(b, offset: 0, index: 13) }
            enc.dispatchThreadgroups(
                MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 512, height: 1, depth: 1))
        }
        if only == nil || only == .gateup {
            enc.setComputePipelineState(
                try ctx.pipeline("vnax_moe_gateup\(ctx.dims.ffnActSuffix)\(fp4Suffix)_nf\(Self.nGuNF)_sg\(Self.nGuSGS)")
            )
            enc.setBuffer(x, offset: xOffset, index: 0)
            var i = 1
            for l in [eGate, eUp] {
                for t in [l.w, l.s, l.b] { enc.setBuffer(t.buffer, offset: t.offset, index: i); i += 1 }
            }
            for m in [sGate, sUp] { for b in [m.w, m.s, m.b] { enc.setBuffer(b, offset: 0, index: i); i += 1 } }
            enc.setBuffer(ent, offset: 0, index: 13)
            enc.setBuffer(pairs, offset: 0, index: 14)
            enc.setBuffer(h, offset: 0, index: 15)
            enc.setBytes(&mp, length: 28, index: 16)
            enc.setBuffer(smp.buffer, offset: smp.offset, index: 17)
            if let eb = eBias {
                enc.setBuffer(eb.gate, offset: 0, index: 18)
                enc.setBuffer(eb.up, offset: 0, index: 19)
            }
            enc.dispatchThreadgroups(
                indirectBuffer: ags, indirectBufferOffset: 0,
                threadsPerThreadgroup: MTLSize(width: 32 * Self.nGuSGS, height: 1, depth: 1))
        }
        if only == nil || only == .down {
            enc.setComputePipelineState(
                try ctx.pipeline("vnax_moe_down_split\(fp4Suffix)_nf\(Self.sDnNF)_sg\(Self.sDnSGS)"))
            enc.setBuffer(h, offset: 0, index: 0)
            for (i, t) in [eDown.w, eDown.s, eDown.b].enumerated() {
                enc.setBuffer(t.buffer, offset: t.offset, index: 1 + i)
            }
            for (i, b) in [sDown.w, sDown.s, sDown.b].enumerated() { enc.setBuffer(b, offset: 0, index: 4 + i) }
            enc.setBuffer(ent, offset: 0, index: 7)
            enc.setBuffer(pairs, offset: 0, index: 8)
            enc.setBuffer(part, offset: 0, index: 10)
            enc.setBytes(&mp, length: 28, index: 11)
            enc.setBuffer(inds, offset: 0, index: 12)
            enc.setBuffer(smp.buffer, offset: smp.offset, index: 13)
            if let eb = eBias { enc.setBuffer(eb.down, offset: 0, index: 14) }
            enc.dispatchThreadgroups(
                indirectBuffer: ags, indirectBufferOffset: 16,
                threadsPerThreadgroup: MTLSize(width: 32 * Self.sDnSGS, height: 1, depth: 1))
            if pass == .resident { return }
            enc.setComputePipelineState(try ctx.pipeline("vmoe_combine9"))
            enc.setBuffer(part, offset: 0, index: 0)
            enc.setBuffer(out, offset: outOffset, index: 1)
            enc.setBytes(&mp, length: 28, index: 2)
            enc.setBuffer(rst, offset: 0, index: 3)
            enc.dispatchThreads(
                MTLSize(width: rows * hidden / 4, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
    }
}
