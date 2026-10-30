import Foundation
import Metal

// Dense quantized matrices (DenseQ) and their launchers: few-row qmv (VQMV, decode and verification) and the prefill
// GEMM (encodeQmmF32: v3 on NAX). Kernels in Shaders/verify.metal, verify_nax.metal, qmm_nax*.metal.

public struct QmmParams {
    public var M: UInt32, N: UInt32, K: UInt32
    public init(M: Int, N: Int, K: Int) { self.M = UInt32(M); self.N = UInt32(N); self.K = UInt32(K) }
}

/// Read bandwidth measured on this Mac (microbenchmark), for the memory floors.
public let measuredReadGBps = 157.0

/// Dense quantized matrix in aligned buffers. Several layers with the same K and the same bits can be concatenated
/// by output rows at load time (a single qmv for qkv|z|b|a, as mlx-serve PR #911 does with its weights).
public struct DenseQ {
    public let w: MTLBuffer, s: MTLBuffer, b: MTLBuffer
    public let N: Int, K: Int, bits: Int
    /// Group size of the scale and bias: 64, or 32 with 4 bits (from the config, `group_size` per layer); 128 for
    /// the ternary.
    public let gs: Int
    /// Suffix of the kernels with groups of 32 (instantiated only with LAB_G32, ModelDims.quantG32); none with 64.
    var gsuf: String { gs == 32 ? "_g32" : "" }
    /// bits = 2: Bonsai 2 ternary (w = (q − 1)·s, groups of 128, only the fp16 scale in `s`; `b` = `s`).
    public var bytes: Int {
        bits == 16 ? N * K * 2 : bits == 2 ? N * K / 4 + N * (K / 128) * 2 : N * K * bits / 8 + 2 * N * (K / gs) * 2
    }
    /// Signs (float [K]) of the Hadamard rotation of the input (Bonsai prism_hadamard format): before the matrix,
    /// x ← H₁₀₂₄(x ⊙ signs) / 32 (the launchers apply it in a work buffer); in the embedding, the inverse.
    public var had: MTLBuffer?
    public var tern: Bool { bits == 2 }

    /// Unquantized bf16 table [N, K] (embeddings of the 27B MLX-Serve pack): bits = 16, s and b empty.
    public init(bf16Table store: SafetensorsStore, name: String, device: MTLDevice) throws {
        let t = try store.info(name)
        guard t.dtype == .bf16, t.shape.count == 2 else { throw SafetensorsError.format("\(name) is not 2D bf16") }
        N = t.shape[0]; K = t.shape[1]; bits = 16; gs = 64
        w = device.makeBuffer(length: t.byteCount, options: .storageModeShared)!
        w.contents().copyMemory(from: store.bytes(t).baseAddress!, byteCount: t.byteCount)
        s = device.makeBuffer(length: 16, options: .storageModeShared)!
        b = s
    }

    /// Unquantized bf16 matrix [N, K] with zero output rows up to a multiple of `m` (bits = 16, s and b empty): the MoE
    /// router that Nemotron-H and GLM ship in bf16, applied as is by vnax_qmvbf16 and qmm_t_nax_f32_bf16.
    init(bf16Rows store: SafetensorsStore, name: String, padTo m: Int, device: MTLDevice) throws {
        let t = try store.info(name)
        guard t.dtype == .bf16, t.shape.count == 2 else { throw SafetensorsError.format("\(name) is not 2D bf16") }
        K = t.shape[1]; N = (t.shape[0] + m - 1) / m * m; bits = 16; gs = 64
        w = device.makeBuffer(length: N * K * 2, options: .storageModeShared)!
        memset(w.contents(), 0, N * K * 2)
        w.contents().copyMemory(from: store.bytes(t).baseAddress!, byteCount: t.byteCount)
        s = device.makeBuffer(length: 16, options: .storageModeShared)!
        b = s
    }

    /// Row-gather kernel of this table (4, 6, 8 bits or bf16; all take tok, w, s, b, outF, outH and EmbedParams).
    var embedKernel: String {
        precondition((bits == 4 || bits == 6 || bits == 8 || bits == 16) && (gs == 64 || bits == 4))
        return bits == 16
            ? "embed_rows_bf16" : bits == 6 ? "embed_rows_b6" : bits == 8 ? "embed_rows_b8" : "embed_rows\(gsuf)"
    }

    /// Rows tok[r] of the table (embedding) → float [rows, K] (flags 1) and/or half (flags 2) with stride `ldo`.
    public func encodeEmbed(
        _ ctx: MetalContext, _ e: MTLComputeCommandEncoder, tok: MTLBuffer, tokOffset: Int = 0,
        outF: MTLBuffer?, outH: MTLBuffer? = nil, rows: Int
    ) throws {
        if tern {
            precondition(had != nil && K % 1024 == 0)
            var ep = (UInt32(K), UInt32((outF != nil ? 1 : 0) | (outH != nil ? 2 : 0)), UInt32(K), UInt32(0))
            e.setComputePipelineState(try ctx.pipeline("embed_rows_t"))
            let o = outF ?? outH!
            for (i, buf) in [tok, w, s, had!, outF ?? o, outH ?? o].enumerated() {
                e.setBuffer(buf, offset: i == 0 ? tokOffset : 0, index: i)
            }
            e.setBytes(&ep, length: 16, index: 6)
            e.dispatchThreadgroups(
                MTLSize(width: rows, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
            return
        }
        var ep = (UInt32(K), UInt32((outF != nil ? 1 : 0) | (outH != nil ? 2 : 0)), UInt32(K), UInt32(0))
        e.setComputePipelineState(try ctx.pipeline(embedKernel))
        let o = outF ?? outH!
        for (i, buf) in [tok, w, s, b, outF ?? o, outH ?? o].enumerated() {
            e.setBuffer(buf, offset: i == 0 ? tokOffset : 0, index: i)
        }
        e.setBytes(&ep, length: 16, index: 6)
        e.dispatchThreadgroups(
            MTLSize(width: rows, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    /// Empty matrix (1 row of zeros): stands in for the shared expert in models that have none (LFM2).
    public static func empty(device: MTLDevice, K: Int, bits: Int = 4) -> DenseQ {
        let w = device.makeBuffer(length: max(16, K * bits / 8), options: .storageModeShared)!
        let s = device.makeBuffer(length: max(16, K / 64 * 2), options: .storageModeShared)!
        let b = device.makeBuffer(length: max(16, K / 64 * 2), options: .storageModeShared)!
        for x in [w, s, b] { memset(x.contents(), 0, x.length) }
        return DenseQ(w: w, s: s, b: b, N: 1, K: K, bits: bits)
    }

    /// `padTo`: extra output rows (zero) up to a multiple of padTo (the prefill GEMMs need N % 64 = 0).
    /// `multiLinear`: the paths are per-head MultiLinear tensors [heads, out, in], each one heads · out rows
    /// (QuantizedLinear).
    public init(
        store: SafetensorsStore, config: ModelConfig, paths: [String], device: MTLDevice, padTo: Int = 1,
        totalRows: Int? = nil, multiLinear: Bool = false
    ) throws {
        let ls = try paths.map {
            try QuantizedLinear(store: store, path: $0, config: config, multiLinear: multiLinear)
        }
        if let f = ls.first, f.params.bits == 2, f.params.groupSize == 128 {
            self = try DenseQ(ternary: store, layers: ls, device: device, totalRows: totalRows)
            return
        }
        // Pieces at 4 and 8 bits (an MLX checkpoint with per-path bits, e.g. qkv|z at 8 and a|b at 4) are
        // concatenated at 8 bits: the 4-bit ones are widened losslessly (widen4to8Words). Groups of 64 only (the
        // 8-bit kernels read g64).
        let mixed = Set(ls.map(\.params.bits)) == [4, 8]
        guard let f = ls.first,
            ls.allSatisfy({
                $0.inFeatures == f.inFeatures && $0.experts == 1
                    && (mixed
                        ? $0.params.groupSize == 64 && $0.params.mode == f.params.mode : $0.params == f.params)
            }),
            [4, 6, 8].contains(f.params.bits)
        else {
            throw SafetensorsError.format(
                "cannot concatenate \(paths) (\(ls.map(\.params.description)); mixed 4/8 bits only with groups of 64)")
        }
        // Groups of 32: only with 4 bits (the 6- and 8-bit kernels read groups of 64).
        guard f.params.groupSize == 64 || (f.params.groupSize == 32 && f.params.bits == 4) else {
            throw SafetensorsError.format(
                "\(f.path): \(f.params.bits) bits with groups of \(f.params.groupSize) not supported (groups of 64, or "
                    + "32 only with 4 bits)")
        }
        K = f.inFeatures; bits = mixed ? 8 : f.params.bits; gs = f.params.groupSize
        // Scales and biases must be bf16 (the kernels read bf16; fp16 packs are not converted, so they fail here).
        for l in ls where l.scales.dtype != .bf16 || (l.biases?.dtype ?? .bf16) != .bf16 {
            throw SafetensorsError.format(
                "\(l.path): scales/biases are \(l.scales.dtype.rawValue)/\((l.biases?.dtype ?? .bf16).rawValue), "
                    + "expected bf16")
        }
        let n0 = ls.reduce(0) { $0 + $1.outFeatures }
        N = (n0 + padTo - 1) / padTo * padTo
        let pad = Double(N) / Double(n0)
        // Pieces widened from 4 to 8 bits: their packed weights double and their scales are divided by 16 (exact in
        // bf16).
        let widen = ls.map { mixed && $0.params.bits == 4 }
        enum Part { case weight, scales, biases }
        func cat(_ infos: [TensorInfo], _ part: Part) -> MTLBuffer {
            let sizes = zip(infos, widen).map { part == .weight && $1 ? $0.byteCount * 2 : $0.byteCount }
            let total = sizes.reduce(0, +)
            let buf = device.makeBuffer(
                length: max(Int((Double(total) * pad).rounded(.up)), 16), options: .storageModeShared)!
            memset(buf.contents(), 0, buf.length)
            var off = 0
            for (i, t) in infos.enumerated() {
                let dst = buf.contents().advanced(by: off)
                if part == .weight, widen[i] {
                    Self.widen4to8Words(
                        UnsafeRawPointer(store.bytes(t).baseAddress!), dst, sourceWords: t.byteCount / 4)
                } else if part == .scales && widen[i] {
                    let d16 = dst.bindMemory(to: UInt16.self, capacity: t.byteCount / 2)
                    for (j, v) in store.array(t, as: UInt16.self).enumerated() {
                        d16[j] = floatToBF16(bf16ToFloat(v) / 16)
                    }
                } else {
                    dst.copyMemory(from: store.bytes(t).baseAddress!, byteCount: t.byteCount)
                }
                off += sizes[i]
            }
            return buf
        }
        w = cat(ls.map(\.weight), .weight); s = cat(ls.map(\.scales), .scales)
        b = cat(
            try ls.map { l in
                guard let bi = l.biases else { throw SafetensorsError.missing(l.path + ".biases") }
                return bi
            }, .biases)
    }

    /// Rows `rows` of `src` (4 or 8 bits), in that order: the draft's lm_head with a chosen vocabulary.
    public init(gather src: DenseQ, rows: [Int], device: MTLDevice) {
        precondition(src.bits == 4 || src.bits == 8)
        N = rows.count; K = src.K; bits = src.bits; gs = src.gs
        let wr = src.K * src.bits / 8, sr = src.K / src.gs * 2, n = rows.count
        func g(_ from: MTLBuffer, _ rb: Int) -> MTLBuffer {
            let buf = device.makeBuffer(length: n * rb, options: .storageModeShared)!
            for (i, r) in rows.enumerated() {
                buf.contents().advanced(by: i * rb).copyMemory(
                    from: from.contents().advanced(by: r * rb), byteCount: rb)
            }
            return buf
        }
        let gw = g(src.w, wr), gsb = g(src.s, sr), gb = g(src.b, sr)
        w = gw; s = gsb; b = gb
    }

}

/// Aligned copy of a small tensor (norms, conv, A_log...).
func alignedTensor(_ store: SafetensorsStore, _ name: String, _ device: MTLDevice) throws -> MTLBuffer {
    try store.deviceTensor(name, device: device).buffer
}

/// A plain RMSNorm weight as stored, which must have the dtype of the small weights (LAB_NWT: fp32 with `normF32`,
/// bf16 otherwise): `alignedTensor` copies bytes without looking at the dtype, so anything else (fp16) is a load error
/// naming the tensor instead of a silent misread.
func plainNormTensor(_ store: SafetensorsStore, _ name: String, _ device: MTLDevice, f32: Bool) throws -> MTLBuffer {
    try checkNormDtype(store, name, f32: f32)
    return try alignedTensor(store, name, device)
}

func checkNormDtype(_ store: SafetensorsStore, _ name: String, f32: Bool) throws {
    let want: DType = f32 ? .f32 : .bf16, got = try store.info(name).dtype
    guard got == want else {
        throw SafetensorsError.format("\(name): norm weight is \(got.rawValue), expected \(want.rawValue)")
    }
}

/// An RMSNorm weight stored as `w` and applied as `x·(1 + w)` (Muse), folded at load time in the type of
/// the small weights (LAB_NWT): bf16(1 + w) rounded to nearest even (the value mlx-lm computes with `1.0 + weight` in
/// bf16), or 1 + w in fp32 with fp32 small weights.
func onePlusNormTensor(_ ctx: MetalContext, _ store: SafetensorsStore, _ name: String) throws -> MTLBuffer {
    let t = try store.info(name)
    let w: [Float]
    switch t.dtype {
    case .f32: w = store.array(t, as: Float.self)
    case .bf16: w = store.array(t, as: UInt16.self).map(bf16ToFloat)
    case .f16: w = store.array(t, as: UInt16.self).map { Float(Float16(bitPattern: $0)) }
    default: throw SafetensorsError.format("\(name): expected fp32, bf16 or fp16")
    }
    return ctx.dims.normF32 ? ctx.buffer(w.map { 1 + $0 }) : ctx.buffer(w.map { floatToBF16(1 + $0) })
}

/// An fp32 norm weight applied as `x·(1 + w)` (Flash-Next), folded at load time as 1 + w in fp32.
func onePlusF32Tensor(_ ctx: MetalContext, _ store: SafetensorsStore, _ name: String) throws -> MTLBuffer {
    let t = try store.info(name)
    guard t.dtype == .f32 else { throw SafetensorsError.format("\(name): expected fp32") }
    return ctx.buffer(store.array(t, as: Float.self).map { 1 + $0 })
}

/// An fp32 or bf16 vector of the store as an fp32 buffer (expert_bias: fp32 in LFM2.5-24B, bf16 in LFM2.5-8B).
func floatTensor(_ store: SafetensorsStore, _ name: String, _ device: MTLDevice) throws -> MTLBuffer {
    let t = try store.info(name)
    if t.dtype == .f32 { return try alignedTensor(store, name, device) }
    guard t.dtype == .bf16 || t.dtype == .f16 else {
        throw SafetensorsError.format("\(name): expected fp32, bf16 or fp16")
    }
    let v = store.bytes(t).bindMemory(to: UInt16.self).map {
        t.dtype == .f16 ? Float(Float16(bitPattern: $0)) : bf16ToFloat($0)
    }
    return v.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: max($0.count, 16))! }
}

public struct VParams {
    public var M: UInt32, N: UInt32, K: UInt32, flags: UInt32, ldx: UInt32, ldy: UInt32
}

/// Launcher of `vqmv`: Y[M, N] (+)= X[M, K] · Wᵀ with few rows.
public struct VQMV {
    public let ctx: MetalContext
    private let nc = 4, sgs = 4
    /// Rows per pass (NR); nil = depends on M.
    public var nr: Int?
    /// matmul2d (NAX) from this many rows (4 bits only); 0 = never. NF fragments of 16 columns and SGS simdgroups
    /// (split-K) per threadgroup.
    /// nf1 sg32 (16 columns and 32 split-K simdgroups per threadgroup): +2–13 % over nf2 sg16 in all the shapes of
    /// the round (2048×4096: 0.038 → 0.033 ms; 248320×2048: 1.83 → 1.78).
    public var naxFromRows = 1
    /// 6 bits on NAX (vnax_qmv6_nf2_sg8: 2 fragments, 8 split-K simdgroups) instead of qmv6. Set by Flash-Next.
    public var nax6 = false
    private let naxNF = 1, naxSGS = 32
    /// 8 bits on NAX with 16 split-K simdgroups (fn_qmv8_nf1_sg16).
    private let nax8SGS = 16
    /// Ternary (Bonsai): 1 fragment and split-K. Measured with ingot-dev gen --profile (B = 8, 27B): NF 1 with 8
    /// simdgroups (gate|up 21.6 ms, 89 % of the floor) vs NF 2 (27.9) and NF 4 (40); with large K (down, 17408) 16
    /// simdgroups.
    private let ternNF = 1, ternSGS = 8, ternSGSBigK = 16
    public init(_ ctx: MetalContext) { self.ctx = ctx }

    public static func defaultNR(_ rows: Int) -> Int { rows >= 8 ? 8 : rows >= 4 ? 4 : rows >= 2 ? 2 : 1 }

    public func encode(
        _ enc: MTLComputeCommandEncoder, x: MTLBuffer, xOffset: Int = 0, xFloat: Bool = false,
        _ m: DenseQ, y: MTLBuffer, yOffset: Int = 0, rows: Int, ldx: Int? = nil, ldy: Int? = nil,
        accumulate: Bool = false, nPrefix: Int? = nil
    ) throws {
        var x = x, xOffset = xOffset, ldx = ldx
        if m.had != nil {
            precondition(!xFloat)
            x = try m.encodeRotate(ctx, enc, x: x, xOffset: xOffset, ldx: ldx ?? m.K, rows: rows)
            xOffset = 0; ldx = m.K
        }
        if m.tern {
            // Ternary (xc_tern.metal): groups of 128, 16·NF columns per threadgroup.
            let nf = ternNF, sg = m.K > 8192 ? ternSGSBigK : ternSGS
            let np = nPrefix ?? m.N
            precondition(!xFloat && rows <= 16 && np % (16 * nf) == 0)
            var p = VParams(
                M: UInt32(rows), N: UInt32(np), K: UInt32(m.K), flags: accumulate ? 1 : 0,
                ldx: UInt32(ldx ?? m.K), ldy: UInt32(ldy ?? np))
            enc.setComputePipelineState(try ctx.pipeline("vnax_qmvt_nf\(nf)_sg\(sg)"))
            for (i, b) in [x, m.w, m.s, y].enumerated() {
                enc.setBuffer(b, offset: i == 0 ? xOffset : i == 3 ? yOffset : 0, index: i == 3 ? 4 : i)
            }
            enc.setBytes(&p, length: MemoryLayout<VParams>.size, index: 5)
            enc.dispatchThreadgroups(
                MTLSize(width: np / (16 * nf), height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32 * sg, height: 1, depth: 1))
            return
        }
        // 6 bits (tied lm_head of LFM2.5): qmv6, 4 rows of W per simdgroup.
        if m.bits == 6 {
            precondition(!xFloat && nPrefix == nil && rows <= 16 && m.N % 32 == 0 && m.K % 64 == 0)
            var p = VParams(
                M: UInt32(rows), N: UInt32(m.N), K: UInt32(m.K), flags: accumulate ? 1 : 0,
                ldx: UInt32(ldx ?? m.K), ldy: UInt32(ldy ?? m.N))
            if nax6 {
                // On NAX (vnax_qmv6): Flash-Next, 6-bit head with any number of rows (the same path for 1 and
                // for 16: each row gives the same result).
                precondition(m.N % 32 == 0)
                enc.setComputePipelineState(try ctx.pipeline("vnax_qmv6_nf2_sg8"))
                for (i, b) in [x, m.w, m.s, m.b, y].enumerated() {
                    enc.setBuffer(b, offset: i == 0 ? xOffset : i == 4 ? yOffset : 0, index: i)
                }
                enc.setBytes(&p, length: MemoryLayout<VParams>.size, index: 5)
                enc.dispatchThreadgroups(
                    MTLSize(width: m.N / 32, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 32 * 8, height: 1, depth: 1))
                return
            }
            // Any number of rows through qmv6 (the rows in a loop with the same body): each row gives the bits of
            // the one-row pass. qmv6m (W read once for all rows) summed in another order under fast math (logits
            // |Δ| ~1e-6 between a round of k rows and k steps of one, rowscheck on toy-lfm2) and was no
            // faster: 128000 × 2048, 2–8 rows within 3 %, 16 rows 6.7 ms against 7.3.
            enc.setComputePipelineState(try ctx.pipeline("qmv6_nw4"))
            for (i, b) in [x, m.w, m.s, m.b, y].enumerated() {
                enc.setBuffer(b, offset: i == 0 ? xOffset : i == 4 ? yOffset : 0, index: i)
            }
            enc.setBytes(&p, length: MemoryLayout<VParams>.size, index: 5)
            enc.dispatchThreadgroups(
                MTLSize(width: m.N / 32, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            return
        }
        // nPrefix: only the first rows of W (outputs [0, nPrefix)), with y compact (ldy = nPrefix).
        if let np = nPrefix {
            precondition(m.bits == 4 && !xFloat && rows <= 16 && np % (16 * naxNF) == 0 && np <= m.N)
            let pso = try ctx.pipeline("vnax_qmv\(m.gsuf)_nf\(naxNF)_sg\(naxSGS)")
            var p = VParams(
                M: UInt32(rows), N: UInt32(np), K: UInt32(m.K), flags: accumulate ? 1 : 0,
                ldx: UInt32(ldx ?? m.K), ldy: UInt32(ldy ?? np))
            enc.setComputePipelineState(pso)
            for (i, b) in [x, m.w, m.s, m.b, y].enumerated() {
                enc.setBuffer(b, offset: i == 0 ? xOffset : i == 4 ? yOffset : 0, index: i)
            }
            enc.setBytes(&p, length: MemoryLayout<VParams>.size, index: 5)
            enc.dispatchThreadgroups(
                MTLSize(width: np / (16 * naxNF), height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32 * naxSGS, height: 1, depth: 1))
            return
        }
        if naxFromRows > 0, rows >= naxFromRows, m.bits == 4, !xFloat, rows <= 16, m.N % (16 * naxNF) == 0 {
            let pso = try ctx.pipeline("vnax_qmv\(m.gsuf)_nf\(naxNF)_sg\(naxSGS)")
            var p = VParams(
                M: UInt32(rows), N: UInt32(m.N), K: UInt32(m.K), flags: accumulate ? 1 : 0,
                ldx: UInt32(ldx ?? m.K), ldy: UInt32(ldy ?? m.N))
            enc.setComputePipelineState(pso)
            enc.setBuffer(x, offset: xOffset, index: 0)
            enc.setBuffer(m.w, offset: 0, index: 1)
            enc.setBuffer(m.s, offset: 0, index: 2)
            enc.setBuffer(m.b, offset: 0, index: 3)
            enc.setBuffer(y, offset: yOffset, index: 4)
            enc.setBytes(&p, length: MemoryLayout<VParams>.size, index: 5)
            enc.dispatchThreadgroups(
                MTLSize(width: m.N / (16 * naxNF), height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32 * naxSGS, height: 1, depth: 1))
            return
        }
        if m.bits == 16 {
            // Unquantized bf16 (the bf16 MoE router): only on NAX, with the 8-bit router's split-K.
            guard !xFloat, rows <= 16, m.N % 32 == 0, m.K % 64 == 0 else {
                throw SafetensorsError.format("vqmv: bf16 matrix \(m.N)×\(m.K) with \(rows) rows not supported")
            }
            let sg = nax8SGS
            var p = VParams(
                M: UInt32(rows), N: UInt32(m.N), K: UInt32(m.K), flags: accumulate ? 1 : 0,
                ldx: UInt32(ldx ?? m.K), ldy: UInt32(ldy ?? m.N))
            enc.setComputePipelineState(try ctx.pipeline("vnax_qmvbf16_nf2_sg\(sg)"))
            enc.setBuffer(x, offset: xOffset, index: 0)
            enc.setBuffer(m.w, offset: 0, index: 1)
            enc.setBuffer(y, offset: yOffset, index: 4)
            enc.setBytes(&p, length: MemoryLayout<VParams>.size, index: 5)
            enc.dispatchThreadgroups(
                MTLSize(width: m.N / 32, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32 * sg, height: 1, depth: 1))
            return
        }
        if m.bits == 8, !xFloat, rows <= 16, m.N % 32 == 0 {
            // 8 bits on NAX (every 8-bit matrix of the round: LFM3's dense part, MoE routers, 8-bit heads): split-K
            // across nax8SGS simdgroups. With 16, 16 columns per threadgroup (fn_qmv8_nf1_sg16, 16 KB of threadgroup
            // memory; the same summation order as vnax_qmv8_nf2_sg16, bit for bit): every LFM3 piece reads faster
            // than with 32 columns (32 KB), not only the narrow ones (down 2048×7680: 130 → 140 GB/s; gate|up
            // 15360×2048: 153 → 162; head: 158 → 165), and LFM3's write goes from 13.72 to 13.03 ms/token at 512
            // and from 13.84 to 13.08 at 4K (only N ≤ 4096: 13.47 / 13.54); Qwen3.6 MoE (router N = 256) unchanged
            // within noise.
            let pso = try ctx.pipeline("fn_qmv8_nf1_sg16")
            var p = VParams(
                M: UInt32(rows), N: UInt32(m.N), K: UInt32(m.K), flags: accumulate ? 1 : 0,
                ldx: UInt32(ldx ?? m.K), ldy: UInt32(ldy ?? m.N))
            enc.setComputePipelineState(pso)
            for (i, b) in [x, m.w, m.s, m.b, y].enumerated() {
                enc.setBuffer(b, offset: i == 0 ? xOffset : i == 4 ? yOffset : 0, index: i)
            }
            enc.setBytes(&p, length: MemoryLayout<VParams>.size, index: 5)
            enc.dispatchThreadgroups(
                MTLSize(width: m.N / 16, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32 * nax8SGS, height: 1, depth: 1))
            return
        }
        let ncv = m.bits == 8 ? min(nc, 4) : nc
        let nrv = nr ?? Self.defaultNR(rows)
        guard m.N % (ncv * sgs) == 0 else {
            throw SafetensorsError.format("vqmv: N \(m.N) is not a multiple of \(ncv * sgs)")
        }
        let pso = try ctx.pipeline("vqmv_b\(m.bits)\(m.gsuf)_nc\(ncv)_nr\(nrv)_sg\(sgs)_\(xFloat ? "f" : "h")")
        var p = VParams(
            M: UInt32(rows), N: UInt32(m.N), K: UInt32(m.K), flags: accumulate ? 1 : 0,
            ldx: UInt32(ldx ?? m.K), ldy: UInt32(ldy ?? m.N))
        enc.setComputePipelineState(pso)
        enc.setBuffer(x, offset: xOffset, index: 0)
        enc.setBuffer(m.w, offset: 0, index: 1)
        enc.setBuffer(m.s, offset: 0, index: 2)
        enc.setBuffer(m.b, offset: 0, index: 3)
        enc.setBuffer(y, offset: yOffset, index: 4)
        enc.setBytes(&p, length: MemoryLayout<VParams>.size, index: 5)
        enc.dispatchThreadgroups(
            MTLSize(width: m.N / (ncv * sgs), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32 * sgs, height: 1, depth: 1))
    }
}

extension DenseQ {
    public init(w: MTLBuffer, s: MTLBuffer, b: MTLBuffer, N: Int, K: Int, bits: Int, gs: Int = 64) {
        self.w = w; self.s = s; self.b = b; self.N = N; self.K = K; self.bits = bits
        self.gs = bits == 2 ? 128 : gs
    }

    /// Bonsai 2 ternary matrices (MLX affine 2 bits, groups of 128, fp16 scale and bias with bias = −scale,
    /// Hadamard rotation by `.signs`) concatenated by output rows; `totalRows` pads with zero rows.
    /// Checks that the bias is exactly −scale and that all the matrices share the signs (same input).
    init(ternary store: SafetensorsStore, layers ls: [QuantizedLinear], device: MTLDevice, totalRows: Int?) throws {
        guard let f = ls.first,
            ls.allSatisfy({ $0.inFeatures == f.inFeatures && $0.params == f.params && $0.experts == 1 })
        else {
            throw SafetensorsError.format("cannot concatenate \(ls.map(\.path))")
        }
        K = f.inFeatures; bits = 2; gs = 128
        let n0 = ls.reduce(0) { $0 + $1.outFeatures }
        N = totalRows ?? n0
        precondition(N >= n0)
        let G = K / 128
        let wb = device.makeBuffer(length: N * K / 4, options: .storageModeShared)!
        let sb = device.makeBuffer(length: N * G * 2, options: .storageModeShared)!
        memset(wb.contents(), 0, wb.length); memset(sb.contents(), 0, sb.length)
        var signs: [Float]?
        var row = 0
        for l in ls {
            guard l.weight.dtype == .u32, l.scales.dtype == .f16, let bi = l.biases, bi.dtype == .f16 else {
                throw SafetensorsError.format("\(l.path): expected u32 + fp16 scale/bias")
            }
            let sc = store.array(l.scales, as: UInt16.self), bs = store.array(bi, as: UInt16.self)
            for i in 0..<sc.count where bs[i] != sc[i] ^ 0x8000 && !(sc[i] & 0x7FFF == 0 && bs[i] & 0x7FFF == 0) {
                throw SafetensorsError.format("\(l.path): bias ≠ −scale in group \(i) (not ternary)")
            }
            let sgT = try store.info(l.path + ".signs")
            guard sgT.dtype == .f32, sgT.shape == [K] else { throw SafetensorsError.format("\(l.path).signs") }
            let sg = store.array(sgT, as: Float.self)
            guard sg.allSatisfy({ $0 == 1 || $0 == -1 }) else {
                throw SafetensorsError.format("\(l.path): signs not ±1")
            }
            if let s0 = signs, s0 != sg {
                throw SafetensorsError.format("\(l.path): different signs in the concatenation")
            }
            signs = sg
            wb.contents().advanced(by: row * K / 4).copyMemory(
                from: store.bytes(l.weight).baseAddress!, byteCount: l.weight.byteCount)
            sc.withUnsafeBytes {
                sb.contents().advanced(by: row * G * 2).copyMemory(from: $0.baseAddress!, byteCount: $0.count)
            }
            row += l.outFeatures
        }
        w = wb; s = sb; b = sb
        had = signs!.withUnsafeBytes {
            device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)!
        }
    }

    /// Work buffer of the input rotation (half [rows, K]), one per context, grows as needed.
    nonisolated(unsafe) static var hadScratch: [ObjectIdentifier: MTLBuffer] = [:]
    static func hadBuffer(_ ctx: MetalContext, bytes: Int) -> MTLBuffer {
        let key = ObjectIdentifier(ctx)
        return MetalContext.scratchLock.withLock {
            if let b = hadScratch[key], b.length >= bytes { return b }
            let b = ctx.buffer(length: max(bytes, 1 << 20))
            hadScratch[key] = b
            return b
        }
    }

    /// x half [rows, K] (stride ldx, from xOffset) → H(x ⊙ signs)/32 in the work buffer (stride K).
    public func encodeRotate(
        _ ctx: MetalContext, _ e: MTLComputeCommandEncoder, x: MTLBuffer, xOffset: Int, ldx: Int, rows: Int
    ) throws -> MTLBuffer {
        precondition(K % 1024 == 0)
        let out = Self.hadBuffer(ctx, bytes: rows * K * 2)
        var p = (UInt32(K), UInt32(ldx), UInt32(K), UInt32(0))
        e.setComputePipelineState(try ctx.pipeline("had_rows_h"))
        e.setBuffer(x, offset: xOffset, index: 0)
        e.setBuffer(had!, offset: 0, index: 1)
        e.setBuffer(out, offset: 0, index: 2)
        e.setBytes(&p, length: 16, index: 3)
        e.dispatchThreadgroups(
            MTLSize(width: K / 1024, height: rows, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
        return out
    }

    /// Affine 4-bit quantization with groups of 64 (MLX format: low nibble first, bf16 scale and bias) of a bf16
    /// [N, K] matrix on CPU. Scale (max − min)/15 and bias = min, rounded to bf16 before quantizing.
    public static func quantize4(bf16 src: UnsafePointer<UInt16>, N: Int, K: Int, device: MTLDevice, bits: Int = 4)
        -> DenseQ
    {
        nonisolated(unsafe) let src = src
        return quantizeAffine(N: N, K: K, device: device, bits: bits) { bf16ToFloat(src[$0]) }
    }

    /// A 4- or 8-bit matrix (groups of 64) dequantized and quantized again to `bits` (an 8-bit lm_head to 4 bits for
    /// a drafter's proposals): w = scale·q + bias in fp32, then the same affine quantization.
    public static func requantize(_ m: DenseQ, bits: Int, device: MTLDevice) -> DenseQ {
        precondition((m.bits == 4 || m.bits == 8) && m.gs == 64 && m.K % 64 == 0)
        let mask = UInt32((1 << m.bits) - 1), per = 32 / m.bits, gpr = m.K / 64
        nonisolated(unsafe) let w = UnsafePointer(m.w.contents().bindMemory(to: UInt32.self, capacity: m.N * m.K / per))
        nonisolated(unsafe) let s = UnsafePointer(m.s.contents().bindMemory(to: UInt16.self, capacity: m.N * gpr))
        nonisolated(unsafe) let b = UnsafePointer(m.b.contents().bindMemory(to: UInt16.self, capacity: m.N * gpr))
        let sb = m.bits
        return quantizeAffine(N: m.N, K: m.K, device: device, bits: bits) { i in
            let g = i / 64, q = (w[i / per] >> UInt32(sb * (i % per))) & mask
            return bf16ToFloat(s[g]) * Float(q) + bf16ToFloat(b[g])
        }
    }

    /// The same from an fp32 [N, K] matrix (the fp32 router of LFM2.5-8B: to 8 bits at load).
    public static func quantize(f32 src: UnsafePointer<Float>, N: Int, K: Int, device: MTLDevice, bits: Int) -> DenseQ {
        nonisolated(unsafe) let src = src
        return quantizeAffine(N: N, K: K, device: device, bits: bits) { src[$0] }
    }

    /// The same matrix with zero output rows up to a multiple of `m` (4 or 8 bits, with their groups).
    public static func padRows(_ q: DenseQ, to m: Int, device: MTLDevice) -> DenseQ {
        let N = (q.N + m - 1) / m * m
        if N == q.N { return q }
        func pad(_ b: MTLBuffer, _ used: Int, _ total: Int) -> MTLBuffer {
            let o = device.makeBuffer(length: total, options: .storageModeShared)!
            memset(o.contents(), 0, total)
            o.contents().copyMemory(from: b.contents(), byteCount: used)
            return o
        }
        let wr = q.K * q.bits / 8, sr = q.K / q.gs * 2
        return DenseQ(
            w: pad(q.w, q.N * wr, N * wr), s: pad(q.s, q.N * sr, N * sr), b: pad(q.b, q.N * sr, N * sr), N: N, K: q.K,
            bits: q.bits, gs: q.gs)
    }

    /// A 4-bit matrix (groups of 64 or 32, kept) widened to 8 bits losslessly: q₈ = 16·q₄ and scale₈ = scale₄ / 16
    /// (exact in bf16: only the exponent changes), same bias. So w = scale·q + bias comes out equal with the 8-bit
    /// kernels. Only for tests (an 8-bit shared expert in shared-bits-guard); the 4-bit pieces concatenated with
    /// 8-bit ones in DenseQ.init use widen4to8Words.
    static func widen4to8(_ q: DenseQ, device: MTLDevice) -> DenseQ {
        precondition(q.bits == 4 && q.K % q.gs == 0)
        let n = q.N * q.K, groups = q.N * q.K / q.gs
        let w = device.makeBuffer(length: n, options: .storageModeShared)!
        let s = device.makeBuffer(length: groups * 2, options: .storageModeShared)!
        widen4to8Words(UnsafeRawPointer(q.w.contents()), w.contents(), sourceWords: n / 8)
        let ss = q.s.contents().bindMemory(to: UInt16.self, capacity: groups)
        let sd = s.contents().bindMemory(to: UInt16.self, capacity: groups)
        for g in 0..<groups { sd[g] = floatToBF16(bf16ToFloat(ss[g]) / 16) }
        let b = device.makeBuffer(length: groups * 2, options: .storageModeShared)!
        b.contents().copyMemory(from: q.b.contents(), byteCount: groups * 2)
        return DenseQ(w: w, s: s, b: b, N: q.N, K: q.K, bits: 8, gs: q.gs)
    }

    /// Packed MLX 4-bit words (8 values each, low nibble first) to 8-bit words with each value times 16: `sourceWords`
    /// words in `src` (any alignment, e.g. straight from the safetensors mmap) give 2·sourceWords words in `dst`.
    static func widen4to8Words(_ src: UnsafeRawPointer, _ dst: UnsafeMutableRawPointer, sourceWords n: Int) {
        let d = dst.bindMemory(to: UInt32.self, capacity: 2 * n)
        for i in 0..<n {
            let x = src.loadUnaligned(fromByteOffset: 4 * i, as: UInt32.self)
            var lo: UInt32 = 0, hi: UInt32 = 0
            for j in 0..<4 {
                lo |= ((x >> UInt32(4 * j)) & 0xF) << UInt32(8 * j + 4)
                hi |= ((x >> UInt32(4 * j + 16)) & 0xF) << UInt32(8 * j + 4)
            }
            d[2 * i] = lo; d[2 * i + 1] = hi
        }
    }

    /// An fp32 or bf16 matrix of the store quantized (affine, groups of 64) to `bits`.
    public static func quantize(store: SafetensorsStore, name: String, device: MTLDevice, bits: Int) throws -> DenseQ {
        let t = try store.info(name)
        guard t.shape.count == 2, t.dtype == .f32 || t.dtype == .bf16 else {
            throw SafetensorsError.format("\(name): expected an fp32 or bf16 matrix")
        }
        let N = t.shape[0], K = t.shape[1], raw = store.bytes(t)
        if t.dtype == .bf16 {
            let a = [UInt16](raw.bindMemory(to: UInt16.self))
            return a.withUnsafeBufferPointer {
                quantize4(bf16: $0.baseAddress!, N: N, K: K, device: device, bits: bits)
            }
        }
        var a = [Float](repeating: 0, count: N * K)
        a.withUnsafeMutableBytes { $0.copyMemory(from: raw) }
        return a.withUnsafeBufferPointer { quantize(f32: $0.baseAddress!, N: N, K: K, device: device, bits: bits) }
    }

    private static func quantizeAffine(
        N: Int, K: Int, device: MTLDevice, bits: Int, read: @Sendable (Int) -> Float
    ) -> DenseQ {
        precondition(K % 64 == 0 && (bits == 4 || bits == 8))
        let levels = Float((1 << bits) - 1), per = 32 / bits
        let w = device.makeBuffer(length: N * K * bits / 8, options: .storageModeShared)!
        let s = device.makeBuffer(length: N * K / 64 * 2, options: .storageModeShared)!
        let b = device.makeBuffer(length: N * K / 64 * 2, options: .storageModeShared)!
        // Each block writes different rows: the pointers are shared across threads without races.
        nonisolated(unsafe) let wp = w.contents().bindMemory(to: UInt32.self, capacity: N * K / per)
        nonisolated(unsafe) let sp = s.contents().bindMemory(to: UInt16.self, capacity: N * K / 64)
        nonisolated(unsafe) let bp = b.contents().bindMemory(to: UInt16.self, capacity: N * K / 64)
        let blocks = (N + 31) / 32
        DispatchQueue.concurrentPerform(iterations: blocks) { blk in
            var v = [Float](repeating: 0, count: 64)
            for n in (blk * 32)..<min(N, blk * 32 + 32) {
                for g in 0..<(K / 64) {
                    let base = n * K + g * 64
                    var mn = Float.infinity, mx = -Float.infinity
                    for j in 0..<64 { let f = read(base + j); v[j] = f; mn = min(mn, f); mx = max(mx, f) }
                    let sb = floatToBF16((mx - mn) / levels), bb = floatToBF16(mn)
                    let sc = bf16ToFloat(sb), bi = bf16ToFloat(bb)
                    let gi = n * (K / 64) + g
                    sp[gi] = sb; bp[gi] = bb
                    for wi in 0..<(64 / per) {
                        var word: UInt32 = 0
                        for jj in 0..<per {
                            let q = sc > 0 ? min(levels, max(0, ((v[wi * per + jj] - bi) / sc).rounded())) : 0
                            word |= UInt32(q) << UInt32(bits * jj)
                        }
                        wp[base / per + wi] = word
                    }
                }
            }
        }
        return DenseQ(w: w, s: s, b: b, N: N, K: K, bits: bits)
    }

    /// Several bf16 matrices of the store concatenated by output rows and quantized to 4 bits.
    public static func quantize4(store: SafetensorsStore, names: [String], device: MTLDevice, bits: Int = 4) throws
        -> DenseQ
    {
        let infos = try names.map { try store.info($0) }
        guard let K = infos.first?.shape.last,
            infos.allSatisfy({ $0.dtype == .bf16 && $0.shape.count == 2 && $0.shape[1] == K })
        else { throw SafetensorsError.format("cannot quantize \(names)") }
        let N = infos.reduce(0) { $0 + $1.shape[0] }
        var all = [UInt16](repeating: 0, count: N * K)
        var off = 0
        all.withUnsafeMutableBytes { dst in
            for t in infos {
                dst.baseAddress!.advanced(by: off).copyMemory(from: store.bytes(t).baseAddress!, byteCount: t.byteCount)
                off += t.byteCount
            }
        }
        return all.withUnsafeBufferPointer { quantize4(bf16: $0.baseAddress!, N: N, K: K, device: device, bits: bits) }
    }

    /// Prefill GEMM Y[M, N] (+)= X[M, K] · Wᵀ (4 bits) with v3 (qmm_nax_v3.metal: X through threadgroup memory;
    /// bit-identical to qmm_t_nax): K ≥ 4096 with 64×128 tiles and BK 64 (2048×4096: 2.45 → 1.97 ms), otherwise
    /// 64×64 with BK 32 (12352×2048: 5.50 → 5.01 ms). Output fp32, accumulated fp32 or half.
    public func encodeQmmF32(
        _ ctx: MetalContext, _ enc: MTLComputeCommandEncoder, x: MTLBuffer, xOffset: Int = 0,
        y: MTLBuffer, rows M: Int, accumulate: Bool = false, halfOut: Bool = false
    ) throws {
        precondition(M % 64 == 0 && N % 64 == 0 && [2, 4, 8].contains(bits) && !(halfOut && accumulate))
        var x = x, xOffset = xOffset
        if had != nil { x = try encodeRotate(ctx, enc, x: x, xOffset: xOffset, ldx: K, rows: M); xOffset = 0 }
        var p = QmmParams(M: M, N: N, K: K)
        let wn = K >= 4096 && N % 128 == 0 ? 4 : 2, bk = wn == 4 ? 64 : 32
        enc.setComputePipelineState(
            try ctx.pipeline(
                "qmm_t_nax3\(tern ? "t" : bits == 8 ? "_8" : gsuf)_w2x\(wn)_bk\(bk)_"
                    + (accumulate ? "f32acc" : halfOut ? "h" : "f32")))
        enc.setBuffer(x, offset: xOffset, index: 0)
        enc.setBuffer(w, offset: 0, index: 1)
        enc.setBuffer(s, offset: 0, index: 2)
        enc.setBuffer(b, offset: 0, index: 3)
        enc.setBuffer(y, offset: 0, index: 4)
        enc.setBytes(&p, length: MemoryLayout<QmmParams>.size, index: 5)
        enc.dispatchThreadgroups(
            MTLSize(width: N / (32 * wn), height: M / 64, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 64 * wn, height: 1, depth: 1))
    }
}

extension DenseQ {
    /// View as DeviceQuantizedLinear (for the prefill kernels, which take layers).
    public func asDQL() -> DeviceQuantizedLinear {
        let wi = TensorInfo(
            name: "w", dtype: .u32, shape: [N, K * bits / 32], shard: 0, fileOffset: 0, byteCount: N * K * bits / 8)
        let si = TensorInfo(
            name: "s", dtype: .bf16, shape: [N, K / gs], shard: 0, fileOffset: 0, byteCount: N * K / gs * 2)
        let bi = TensorInfo(
            name: "b", dtype: .bf16, shape: [N, K / gs], shard: 0, fileOffset: 0, byteCount: N * K / gs * 2)
        guard let l = try? QuantizedLinear(synthetic: "dense", weight: wi, scales: si, biases: bi, groupSize: gs) else {
            preconditionFailure("invalid synthetic layer (N: \(N), K: \(K), bits: \(bits))")
        }
        return DeviceQuantizedLinear(synthetic: l, w: w, s: s, b: b)
    }
}
