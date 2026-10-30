import Foundation
import Metal

// PrefillMoE: the MoE block for the prefill (GEMMs grouped by expert), and the experts' biases (ExpertBiases) that it
// shares with MoELayer (MoE.swift). Kernels in Shaders/qmm_nax_moe.metal.

/// Biases of the experts (gpt-oss: `experts.{gate,up,down}_proj.bias`, bf16 in the checkpoint) as fp32 buffers: gate and
/// up [E, inter], added before the activation; down [E, hidden], added to each expert's output before its weight.
struct ExpertBiases {
    let gate, up, down: MTLBuffer

    init(store: SafetensorsStore, prefix: String, device d: MTLDevice) throws {
        gate = try floatTensor(store, prefix + "gate_proj.bias", d)
        up = try floatTensor(store, prefix + "up_proj.bias", d)
        down = try floatTensor(store, prefix + "down_proj.bias", d)
    }

    /// The three have the experts' shapes (each at least as long as its [E, n] fp32).
    func matches(_ experts: Int, _ inter: Int, _ hidden: Int) -> Bool {
        gate.length >= experts * inter * 4 && up.length >= experts * inter * 4 && down.length >= experts * hidden * 4
    }
}

// MARK: - MoE block on the GPU

/// MoE block of one Qwen3.6 layer (qwen3_5_moe) with our kernels, for the prefill (chunks of many rows):
/// router (8-bit qmm → top-k) → grouping → grouped gate+up+SwiGLU → grouped down → combine. The shared expert
/// goes as expert E inside the grouped GEMMs (splash's idea). Exact.
public final class PrefillMoE {
    /// Tiles of the grouped GEMMs (v2 kernels of qmm_nax_moe.metal): BM rows per tile, FN fragment pairs of
    /// columns per simdgroup (BN = 64·FN in gate+up, 128·FN in down), K step and weight prefetch steps.
    public struct Tiles: CustomStringConvertible {
        public var bmA: Int, fnA: Int, bmB: Int, fnB: Int
        public var bkA = 64, bkB = 64, pfA = 1, pfB = 1
        public init(bmA: Int = 32, fnA: Int = 1, bmB: Int = 32, fnB: Int = 1, bkA: Int = 64, bkB: Int = 64) {
            self.bmA = bmA; self.fnA = fnA; self.bmB = bmB; self.fnB = fnB
            self.bkA = bkA; self.bkB = bkB
        }
        public var bnA: Int { 64 * fnA }
        public var bnB: Int { 128 * fnB }
        /// Affine 4-bit in groups of 64 (the only affine instances) or MXFP4 (`mxfp4`; LAB_MXFP4, only the tiles of
        /// ModelPrefill.configurePrefillMoE). `act`: suffix of the activation instance (ModelDims.ffnActSuffix;
        /// "" = SiLU).
        public func gateUpKernel(act: String = "", mxfp4: Bool = false) -> String {
            "moe_gateup_swiglu2\(act)_\(mxfp4 ? "mxfp4" : "b4_g64")_bm\(bmA)_fn\(fnA)_bk\(bkA)_pf\(pfA)"
        }
        public func downKernel(mxfp4: Bool = false) -> String {
            "moe_down2_\(mxfp4 ? "mxfp4" : "b4_g64")_bm\(bmB)_fn\(fnB)_bk\(bkB)_pf\(pfB)"
        }
        public var description: String {
            "gate+up BM\(bmA)×BN\(bnA) bk\(bkA) pf\(pfA), down BM\(bmB)×BN\(bnB) bk\(bkB) pf\(pfB)"
        }
    }

    public enum Piece: String, CaseIterable { case router, group, gateup, down, combine }

    public let ctx: MetalContext
    /// Rows of the pass (multiple of 64, ≤ `capacity`); the full model changes it on every prefill chunk.
    public var rows: Int
    /// Real rows of the pass (≤ rows; nil = all). The padding ones up to the multiple of 64 go through the router
    /// (its tiles are 64 rows) but are neither grouped nor combined: so they do not drag in experts nobody needs
    /// (a 65-row chunk read the experts of 63 padding rows, +26 ms). Each row is computed separately in the
    /// grouped GEMMs, so the real ones come out bit-exact.
    public var active: Int?
    var liveRows: Int { min(active ?? rows, rows) }
    public let capacity: Int
    public let hidden: Int, inter: Int, sharedInter: Int, experts: Int, topk: Int
    public var tiles: Tiles

    /// Weights (the full model changes them per layer before encoding; the work buffers are shared).
    public var router, sharedGate, sGate, sUp, sDown, eGate, eUp, eDown: DeviceQuantizedLinear
    /// SSD expert streaming: expert → slot table of the layer (the identity if all experts are in memory).
    public var smap: (buffer: MTLBuffer, offset: Int)
    /// Input half [rows, hidden] (by default, its own buffer).
    public var x: MTLBuffer
    /// If not nil, the combine adds the result (fp32) to this [rows, hidden] buffer instead of writing `out`.
    public var accumulateInto: MTLBuffer?
    public let logits, inds, wts, extInds, extWts, sgate, offsets, rowMap, pos, tilesA, tilesB, argsA, argsB: MTLBuffer
    public let h, yd, out: MTLBuffer
    public let loadSeconds: Double
    public let copiedBytes: Int
    /// If true, grouping and combine use the routing set by the caller (`setRouting`, e.g. the reference's fp32 top-8)
    /// instead of the one `router` computes (which still runs, for the shared expert's gate).
    public var externalRouting = false
    private var useInds: MTLBuffer { externalRouting ? extInds : inds }
    private var useWts: MTLBuffer { externalRouting ? extWts : wts }

    /// With weights already loaded: router, shared expert gate, shared expert (gate, up, down), experts (gate/up/down).
    /// Selection bias of the router (RouterSpec.selectBias: LFM2's expert_bias, DeepSeek's e_score_correction_bias);
    /// without a shared expert, the sGate… layers are empty and not read.
    public var expertBias: MTLBuffer?
    /// Router linear bias and expert biases of the layer (gpt-oss; MoELayer.routerBias and eBias).
    var routerBias: MTLBuffer?
    var eBias: ExpertBiases?
    /// Per-expert scale of the layer (Gemma 4; MoELayer.pes).
    var pes: MTLBuffer?
    /// The selection bias, router bias, expert biases and per-expert scale of a layer (the full model sets them per
    /// layer).
    func setBiases(_ l: MoELayer) {
        expertBias = l.expertBias; routerBias = l.routerBias; eBias = l.eBias; pes = l.pes
    }
    public let routerSpec: RouterSpec
    let kernels: MoEKernelSet

    public init(
        ctx: MetalContext, weights w: [DeviceQuantizedLinear], config: ModelConfig, rows: Int,
        tiles: Tiles = Tiles(), loadSeconds: Double = 0, copiedBytes: Int = 0
    ) throws {
        guard let rs = try config.requiredSpec.router, let E = config.numExperts, let K = config.expertsPerToken,
            w.count == 8
        else {
            throw SafetensorsError.format("the model is not MoE")
        }
        routerSpec = rs
        self.ctx = ctx
        self.tiles = tiles
        router = w[0]; sharedGate = w[1]; sGate = w[2]; sUp = w[3]; sDown = w[4]; eGate = w[5]; eUp = w[6]; eDown = w[7]
        smap = (MoELayer.identityMap(ctx), 0)
        self.loadSeconds = loadSeconds
        self.copiedBytes = copiedBytes

        self.rows = rows
        capacity = rows
        experts = E; topk = K
        hidden = eGate.layer.inFeatures
        inter = eGate.layer.outFeatures
        sharedInter = sGate.layer.outFeatures
        kernels = try MoEKernelSet.forRouter(rs, inter: inter)
        // Affine 4-bit g64 or MXFP4 experts (MoELayer checks that the three agree and that MXFP4 has no shared expert).
        guard eGate.layer.experts == E, eGate.layer.params.bits == 4,
            eGate.layer.params.groupSize == 64 || (eGate.layer.isMXFP4 && rs.shared == .none),
            router.layer.outFeatures == MoELayer.routerLD(E),
            router.layer.params.bits == 8 || router.layer.params.bits == 16,
            E % 32 == 0, rs.shared == .none || sharedGate.layer.params.bits == 8, rows % 64 == 0,
            hidden % 64 == 0, hidden >= tiles.bnB, rs.shared == .none || sharedInter % 64 == 0
        else {
            throw SafetensorsError.format("unsupported MoE shape (E \(E), k \(K), rows \(rows))")
        }
        let sharedBits = [sGate, sUp, sDown].map(\.layer.params.bits)
        guard rs.shared == .none || sharedBits.allSatisfy({ $0 == 4 }) else {
            throw SafetensorsError.format(MoELayer.sharedBitsMessage(layer: nil, sharedBits))
        }
        let P = rows * K
        x = ctx.buffer(length: rows * hidden * 2)
        logits = ctx.buffer(length: rows * MoELayer.routerLD(E) * 4)
        inds = ctx.buffer(length: P * 4)
        wts = ctx.buffer(length: P * 4)
        extInds = ctx.buffer(length: P * 4)
        extWts = ctx.buffer(length: P * 4)
        sgate = ctx.buffer(length: rows * 4)
        offsets = ctx.buffer(length: (E + 1) * 4)
        rowMap = ctx.buffer(length: (P + rows) * 4)
        pos = ctx.buffer(length: P * 4)
        tilesA = ctx.buffer(length: (P / 16 + min(E, P) + rows / 16 + 1) * 16)
        tilesB = ctx.buffer(length: (P / 16 + min(E, P) + rows / 16 + 1) * 16)
        argsA = ctx.buffer(length: 16)
        argsB = ctx.buffer(length: 16)
        h = ctx.buffer(length: (P * inter + rows * max(inter, sharedInter)) * 2)
        // MXFP4 experts write Yd in fp32 (moe_down2_mxfp4, moe_combine_acc_ns_f32): gpt-oss's can pass half's range.
        yd = ctx.buffer(length: (P + rows) * hidden * (eDown.layer.isMXFP4 ? 4 : 2))
        out = ctx.buffer(length: rows * hidden * 2)
    }

    private func bind(_ enc: MTLComputeCommandEncoder, _ l: DeviceQuantizedLinear, at i: Int) {
        enc.setBuffer(l.w.buffer, offset: l.w.offset, index: i)
        enc.setBuffer(l.s.buffer, offset: l.s.offset, index: i + 1)
        enc.setBuffer(l.b.buffer, offset: l.b.offset, index: i + 2)
    }

    /// Intermediate of the shared expert in the grouped GEMMs (sNK; that of the experts if there is no shared expert).
    var sNK: Int { routerSpec.shared == .none ? inter : sGate.layer.outFeatures }

    public func encode(_ enc: MTLComputeCommandEncoder, _ piece: Piece) throws {
        let T = rows, E = experts, R = liveRows, P = R * topk, tg128 = MTLSize(width: 128, height: 1, depth: 1)
        switch piece {
        case .router:
            // fp32 logits = x · Wrᵀ (8 bits, or bf16 as the checkpoint ships it), and top-k + shared expert gate.
            let NL = MoELayer.routerLD(E)  // router rows padded up to 64 (MoELayer.loadRouter)
            var qp = QmmParams(M: T, N: NL, K: hidden)
            enc.setComputePipelineState(
                try ctx.pipeline(router.layer.params.bits == 16 ? "qmm_t_nax_f32_bf16" : "qmm_t_nax_f32_b8_g64"))
            enc.setBuffer(x, offset: 0, index: 0)
            bind(enc, router, at: 1)
            enc.setBuffer(logits, offset: 0, index: 4)
            enc.setBytes(&qp, length: MemoryLayout<QmmParams>.size, index: 5)
            enc.dispatchThreadgroups(MTLSize(width: NL / 64, height: T / 64, depth: 1), threadsPerThreadgroup: tg128)
            var rp = (UInt32(T), UInt32(E), UInt32(hidden), UInt32(topk))
            enc.setComputePipelineState(try ctx.pipeline(kernels.prefillRoute))
            enc.setBuffer(logits, offset: 0, index: 0)
            enc.setBuffer(x, offset: 0, index: 1)
            bind(enc, sharedGate, at: 2)
            enc.setBuffer(inds, offset: 0, index: 5)
            enc.setBuffer(wts, offset: 0, index: 6)
            enc.setBuffer(sgate, offset: 0, index: 7)
            enc.setBytes(&rp, length: 16, index: 8)
            if let b = expertBias ?? routerBias { enc.setBuffer(b, offset: 0, index: 9) }
            if let b = pes { enc.setBuffer(b, offset: 0, index: 10) }
            enc.dispatchThreadgroups(MTLSize(width: (T + 3) / 4, height: 1, depth: 1), threadsPerThreadgroup: tg128)
        case .group:
            // The shared expert enters as expert E with its T rows, with its tiles first.
            var gp = (
                UInt32(P), UInt32(E), UInt32(tiles.bmA), UInt32(tiles.bmB),
                // Down column tiles: the last one shifted back when hidden is not a multiple of BN (LAB_MOE_H64).
                UInt32(max(inter, sNK) / tiles.bnA), UInt32((hidden + tiles.bnB - 1) / tiles.bnB),
                UInt32(routerSpec.shared != .none ? R : 0),
                UInt32(1)
            )
            enc.setComputePipelineState(try ctx.pipeline("moe_group"))
            for (i, b) in [useInds, offsets, rowMap, pos, tilesA, tilesB, argsA, argsB].enumerated() {
                enc.setBuffer(b, offset: 0, index: i)
            }
            enc.setBytes(&gp, length: 32, index: 8)
            enc.setBuffer(useInds, offset: 0, index: 9)  // st: unused (flags bit 1 off)
            enc.dispatchThreadgroups(
                MTLSize(width: 1, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
        case .gateup:
            var mp = (UInt32(inter), UInt32(hidden), UInt32(topk), UInt32(E), UInt32(sNK), UInt32(P))
            enc.setComputePipelineState(
                try ctx.pipeline(tiles.gateUpKernel(act: ctx.dims.ffnActSuffix, mxfp4: eGate.layer.isMXFP4)))
            enc.setBuffer(x, offset: 0, index: 0)
            bind(enc, sGate, at: 11); bind(enc, sUp, at: 14)
            bind(enc, eGate, at: 1)
            bind(enc, eUp, at: 4)
            enc.setBuffer(h, offset: 0, index: 7)
            enc.setBuffer(rowMap, offset: 0, index: 8)
            enc.setBuffer(tilesA, offset: 0, index: 9)
            enc.setBytes(&mp, length: 24, index: 10)
            enc.setBuffer(smap.buffer, offset: smap.offset, index: 17)
            if let eb = eBias {
                enc.setBuffer(eb.gate, offset: 0, index: 18)
                enc.setBuffer(eb.up, offset: 0, index: 19)
            }
            enc.dispatchThreadgroups(indirectBuffer: argsA, indirectBufferOffset: 0, threadsPerThreadgroup: tg128)
        case .down:
            var mp = (UInt32(hidden), UInt32(inter), UInt32(topk), UInt32(E), UInt32(sNK), UInt32(P))
            enc.setComputePipelineState(try ctx.pipeline(tiles.downKernel(mxfp4: eDown.layer.isMXFP4)))
            bind(enc, sDown, at: 7)
            enc.setBuffer(h, offset: 0, index: 0)
            bind(enc, eDown, at: 1)
            enc.setBuffer(yd, offset: 0, index: 4)
            enc.setBuffer(tilesB, offset: 0, index: 5)
            enc.setBytes(&mp, length: 24, index: 6)
            enc.setBuffer(smap.buffer, offset: smap.offset, index: 10)
            if let eb = eBias { enc.setBuffer(eb.down, offset: 0, index: 11) }
            enc.dispatchThreadgroups(indirectBuffer: argsB, indirectBufferOffset: 0, threadsPerThreadgroup: tg128)
        case .combine:
            // The shared expert output is rows P..P+T−1 of yd.
            var cp = (UInt32(hidden), UInt32(topk))
            if let ns = kernels.combineNoShared {
                precondition(accumulateInto != nil)
                enc.setComputePipelineState(try ctx.pipeline(ns + (eDown.layer.isMXFP4 ? "_f32" : "")))
                for (i, b) in [yd, pos, useWts].enumerated() { enc.setBuffer(b, offset: 0, index: i) }
                enc.setBuffer(accumulateInto!, offset: 0, index: 5)
                enc.setBytes(&cp, length: 8, index: 6)
                enc.dispatchThreadgroups(
                    MTLSize(width: R, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                return
            }
            enc.setComputePipelineState(try ctx.pipeline(accumulateInto == nil ? "moe_combine" : "moe_combine_acc"))
            for (i, b) in [yd, pos, useWts, yd, sgate, accumulateInto ?? out].enumerated() {
                enc.setBuffer(b, offset: 0, index: i)
            }
            enc.setBuffer(yd, offset: P * hidden * 2, index: 3)
            enc.setBytes(&cp, length: 8, index: 6)
            enc.dispatchThreadgroups(
                MTLSize(width: R, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
    }

    /// The whole block in order (serial encoder).
    public func encodeAll(_ enc: MTLComputeCommandEncoder) throws {
        for p in Piece.allCases { try encode(enc, p) }
    }

}
