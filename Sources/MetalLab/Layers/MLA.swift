import Foundation
import Metal

// GLM-4.7-Flash's multi-head latent attention (MLA) in the absorbed form over a latent cache, as mlx-lm's
// decode (`Glm4MoeLiteAttention`, L == 1) computes it, in every pass (decode, verification and prefill):
// - the cache holds one KV head per token, `[k_pe | c]` (qk_rope_head_dim + kv_lora_rank = 576), int8 in `kc` as
//   `[H₆₄ k_pe | H₅₁₂ c]` with two scales per token (k_pe's in `ks`, c's in `vs`), read by the 20 query heads (G = 20);
//   c is also V: P·V reads it from `kc` (no `vt`, 584 bytes per token and layer) and the output is rotated back by
//   H₅₁₂ (vattn_reduce in decode and verification, mla_out_unrotate after the prefill attention; both belong to the
//   attention layer's latent KV, `AttentionLayer.latentKV`);
// - the query of head h is `[q_pe_h | embed_q_h · q_nope_h]`, so q·k = q_pe·k_pe + q_nope·(embed_qᵀ c) is mlx-lm's
//   score (scale (nope + rope)^-0.5, BlockSpec.attnScale);
// - the output of each head (576; dims 0…63 unused) goes through `unembed_out_h` (its dims 64…575, the latent part)
//   and then o_proj.
// The prework, dattn_i8/pattn_i8 and the reduce are the existing kernels with a 576 head (verify_attn*.metal); the
// per-head matmuls and the layout are zm_mla.metal. The decompressed form is gone (one path per family).

/// The `.proj` and `.oproj` pieces of an MLA `AttentionLayer` (Layers/Attention.swift).
/// - `.proj`, x → `projOut` with the layout of the prework (`[q (H × [q_pe | q_lat]) | k ([k_pe | c]) | v (unused)]`,
///   fp32, each head of 576; the v head keeps the shape's width but nothing writes or reads it):
///   - `qkvA`: `[q_a_proj | kv_a_proj_with_mqa]` (one matmul over x), with `k_pe`'s rows de-interleaved
///     (MLASpec.kvARowOrder);
///   - the two RMSNorms (`q_a_layernorm`, `kv_a_layernorm`) with `rmsnorm_rows`, to half;
///   - `qB`: `q_b_proj` with its rows reordered so each head comes out as `[q_pe de-interleaved | q_nope]`
///     (MLASpec.qBRowOrder): the prework rotates the first `qk_rope_head_dim` dimensions of q and k by halves;
///   - `mla_split` lays out q_pe and k and q_nope in half; `embedQ` per head (q_nope → q_lat, 4 bits as in the
///     checkpoint, K = nope, N = kv_lora_rank) writes q_lat into the q heads.
/// - `.oproj`: `unembedOut` per head (dims 64…575 of each head of the attention output → v_head_dim, half), then the layer's
///   o_proj.
final class MLAProjection {
    let spec: MLASpec
    let qkvA, qB: DenseQ
    /// The per-head MultiLinear halves of `kv_b_proj`, as stored ([H · out, in], 4 bits g64).
    let embedQ, unembedOut: DenseQ
    let qANorm, kvANorm: MTLBuffer
    let eps: Float
    /// Width of the attention head (the latent one: rope + kv_lora_rank).
    var headDim: Int { spec.latentHeadDim }
    /// Work buffers (fp32 outputs of the matmuls, half norms, q_nope and the unembedded output), shared by the layers of
    /// a context and grown on demand (decode: ≤ 16 rows; prefill: the padded chunk).
    final class Scratch {
        var rows = 0
        var a, qn, cn, q, qNope, out: MTLBuffer?
        /// Decompressed prefill attention (`encodeDecompressed`): W' of the current layer, one block of k_nope and
        /// v, and the running state of the rows (`decRows` rows); allocated on the first long chunk.
        var wp, kn, vv, st, ml: MTLBuffer?
        var decRows = 0
    }
    let scratch: Scratch

    init(
        ctx: MetalContext, store: SafetensorsStore, config cfg: ModelConfig, prefix pre: String, spec m: MLASpec,
        names n: MLANames, shared: MLAProjection?
    ) throws {
        let d = ctx.device
        spec = m
        guard let qRank = m.qLoraRank else { throw SafetensorsError.format("MLA without q_lora_rank") }
        guard ctx.dims.attnHD == m.latentHeadDim, ctx.dims.attnKVH == 1, ctx.dims.attnQH == m.heads else {
            throw SafetensorsError.format("MLA: the attention shape is not the latent one (one KV head of rope + rank)")
        }
        let a = try DenseQ(store: store, config: cfg, paths: [pre + n.qA, pre + n.kvA], device: d)
        guard a.N == qRank + m.kvAWidth, [4, 8].contains(a.bits) else {
            throw SafetensorsError.format("\(pre)\(n.qA)|\(n.kvA): N \(a.N), \(a.bits) bits (4 or 8)")
        }
        qkvA = DenseQ(gather: a, rows: Array(0..<qRank) + m.kvARowOrder.map { qRank + $0 }, device: d)
        let b = try DenseQ(store: store, config: cfg, paths: [pre + n.qB], device: d)
        guard b.N == m.heads * m.qHeadDim, b.K == qRank, [4, 8].contains(b.bits) else {
            throw SafetensorsError.format("\(pre)\(n.qB): N \(b.N), K \(b.K), \(b.bits) bits")
        }
        qB = DenseQ(gather: b, rows: m.qBRowOrder, device: d)
        // The per-head kernels (vnax_qmv_heads, qmm_t_nax3_heads) take 4 bits in groups of 64.
        func perHead(_ path: String, out: Int, in k: Int) throws -> DenseQ {
            let q = try DenseQ(store: store, config: cfg, paths: [pre + path], device: d, multiLinear: true)
            guard q.N == m.heads * out, q.K == k, q.bits == 4, q.gs == 64 else {
                throw SafetensorsError.format(
                    "\(pre)\(path): N \(q.N), K \(q.K), \(q.bits) bits g\(q.gs) (expected [\(m.heads), \(out), \(k)] "
                        + "at 4 bits g64)")
            }
            return q
        }
        embedQ = try perHead(n.embedQ, out: m.kvLoraRank, in: m.qkNopeHeadDim)
        unembedOut = try perHead(n.unembedOut, out: m.vHeadDim, in: m.kvLoraRank)
        qANorm = try plainNormTensor(store, pre + n.qANorm, d, f32: ctx.dims.normF32)
        kvANorm = try plainNormTensor(store, pre + n.kvANorm, d, f32: ctx.dims.normF32)
        eps = Float(cfg.rmsNormEps)
        scratch = shared?.scratch ?? Scratch()
    }

    var bytes: Int { qkvA.bytes + qB.bytes + embedQ.bytes + unembedOut.bytes }

    /// Input width of o_proj: H · v_head_dim (the attention output is H · 576).
    var oProjK: Int { spec.heads * spec.vHeadDim }

    private func buffers(_ ctx: MetalContext, rows: Int) -> Scratch {
        let s = scratch, H = spec.heads
        if s.rows < rows {
            let r = max(rows, AttentionLayer.maxRows)
            s.a = ctx.buffer(length: r * qkvA.N * 4)
            s.qn = ctx.buffer(length: r * qB.K * 2)
            s.cn = ctx.buffer(length: r * spec.kvLoraRank * 2)
            s.q = ctx.buffer(length: r * qB.N * 4)
            s.qNope = ctx.buffer(length: r * H * spec.qkNopeHeadDim * 2)
            s.out = ctx.buffer(length: r * oProjK * 2)
            s.rows = r
        }
        return s
    }

    /// One side of a per-head matmul: buffer and byte offset, row stride and offset between two heads (elements), and
    /// whether it is half (an input always is; an output is fp32 otherwise).
    struct HeadIO {
        var buf: MTLBuffer, offset = 0, ld: Int, perHead: Int, half = false
    }

    /// Y_h = X_h · W_hᵀ for every head h in one dispatch (zm_mla.metal): `w` is [H · N, K]; X_h starts `x.perHead`
    /// elements after X_{h−1} (rows of `x.ld`), Y_h `y.perHead` after Y_{h−1}. Decode and verification (`rows` ≤ 16):
    /// vnax_qmv_heads; prefill (`prefill`, rows a multiple of 64): qmm_t_nax3_heads.
    static func encodeHeads(
        _ ctx: MetalContext, _ e: MTLComputeCommandEncoder, _ w: DenseQ, heads H: Int, x: HeadIO, y: HeadIO, rows: Int,
        prefill: Bool
    ) throws {
        precondition(w.bits == 4 && w.gs == 64 && w.N % H == 0)
        let N = w.N / H, K = w.K
        let ldx = x.ld, xh = x.perHead, ldy = y.ld, yh = y.perHead
        e.setBuffer(x.buf, offset: x.offset, index: 0)
        e.setBuffer(w.w, offset: 0, index: 1)
        e.setBuffer(w.s, offset: 0, index: 2)
        e.setBuffer(w.b, offset: 0, index: 3)
        e.setBuffer(y.buf, offset: y.offset, index: 4)
        let ot = y.half ? "h" : "f32"
        if prefill {
            precondition(rows % 64 == 0 && N % 64 == 0 && K % 32 == 0)
            let bk = K % 64 == 0 ? 64 : 32
            var p = (
                UInt32(rows), UInt32(N), UInt32(K), UInt32(ldx), UInt32(ldy), UInt32(xh), UInt32(yh)
            )
            e.setComputePipelineState(try ctx.pipeline("qmm_t_nax3_heads_w2x2_bk\(bk)_\(ot)"))
            e.setBytes(&p, length: MemoryLayout.size(ofValue: p), index: 5)
            e.dispatchThreadgroups(
                MTLSize(width: N / 64, height: rows / 64, depth: H),
                threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            return
        }
        precondition(rows >= 1 && rows <= 16 && N % 32 == 0 && K % 64 == 0)
        // embed_q (K 192: 3 blocks of 64) splits K over 4 simdgroups; unembed_out (K 512) over 8.
        let nf = 2, sgs = K >= 512 ? 8 : 4
        var p = VParams(
            M: UInt32(rows), N: UInt32(N), K: UInt32(K), flags: 0, ldx: UInt32(ldx), ldy: UInt32(ldy))
        var hp = (UInt32(xh), UInt32(yh))
        e.setComputePipelineState(try ctx.pipeline("vnax_qmv_heads_nf\(nf)_sg\(sgs)_\(ot)"))
        e.setBytes(&p, length: MemoryLayout<VParams>.size, index: 5)
        e.setBytes(&hp, length: MemoryLayout.size(ofValue: hp), index: 6)
        e.dispatchThreadgroups(
            MTLSize(width: N / (16 * nf), height: H, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32 * sgs, height: 1, depth: 1))
    }

    /// `.proj`: x (half [rows, K], from `xOffset`) → `proj` (fp32 [rows, projN]). Decode and verification: qmv (`vq`);
    /// prefill (`prefill`, rows padded to 64): the GEMMs, with `i8` (lab serve's W8A8) only on `qkvA`; the inner
    /// matmuls (K = q_lora_rank, nope) stay exact.
    func encode(
        _ ctx: MetalContext, _ e: MTLComputeCommandEncoder, vq: VQMV, x: MTLBuffer, xOffset: Int = 0, proj: MTLBuffer,
        projN: Int, rows: Int, prefill: Bool = false, i8: DenseI8Scratch? = nil
    ) throws {
        let s = buffers(ctx, rows: rows), qRank = qB.K, R = spec.kvLoraRank, H = spec.heads, hd = headDim
        let rope = spec.qkRopeHeadDim, nope = spec.qkNopeHeadDim
        precondition(projN == (H + 2) * hd)
        if prefill {
            precondition(xOffset == 0)
            try qkvA.encodePrefillGemm(ctx, e, x: x, y: s.a!, rows: rows, i8: i8)
        } else {
            try vq.encode(e, x: x, xOffset: xOffset, qkvA, y: s.a!, rows: rows)
        }
        e.setComputePipelineState(try ctx.pipeline("rmsnorm_rows"))
        for (w, off, D, o) in [(qANorm, 0, qRank, s.qn!), (kvANorm, qRank * 4, R, s.cn!)] {
            var np = (UInt32(D), eps, UInt32(qkvA.N), UInt32(D))
            e.setBuffer(s.a!, offset: off, index: 0)
            e.setBuffer(w, offset: 0, index: 1)
            e.setBuffer(o, offset: 0, index: 2)
            e.setBytes(&np, length: 16, index: 3)
            e.dispatchThreadgroups(
                MTLSize(width: rows, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
        if prefill {
            try qB.encodeQmmF32(ctx, e, x: s.qn!, y: s.q!, rows: rows)
        } else {
            try vq.encode(e, x: s.qn!, qB, y: s.q!, rows: rows)
        }
        var p = (UInt32(rope + nope), UInt32(rope), UInt32(nope), UInt32(qkvA.N), UInt32(qRank + R))
        e.setComputePipelineState(try ctx.pipeline("mla_split"))
        for (i, b) in [s.a!, s.q!, s.cn!, s.qNope!, proj].enumerated() { e.setBuffer(b, offset: 0, index: i) }
        e.setBytes(&p, length: MemoryLayout.size(ofValue: p), index: 5)
        e.dispatchThreadgroups(
            MTLSize(width: (H * (rope + nope) + hd + 255) / 256, height: rows, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        // q_lat of head h → columns h·hd + rope… of the projection row.
        try Self.encodeHeads(
            ctx, e, embedQ, heads: H, x: HeadIO(buf: s.qNope!, ld: H * nope, perHead: nope, half: true),
            y: HeadIO(buf: proj, offset: rope * 4, ld: projN, perHead: hd), rows: rows, prefill: prefill)
    }

    // MARK: - Decompressed prefill attention

    /// Chunks with at least `decMinRows` rows over at least `decMinKeys` keys (context + chunk) take the
    /// decompressed attention (zn_mla_dec.metal): 512 MAC per row, key and head instead of 1,088, plus decompressing
    /// every key once per chunk (4.59 M MAC per key), which a chunk of 1,024 rows already repays.
    /// Measured (MLX's SDPA, chunks of 2,048 at 16K/32K): 0.47-0.51 × the absorbed attention; 0.53-0.56 with 1,024 rows;
    /// 0.94 with 512.
    static let decMinRows = 1024, decMinKeys = 8192
    /// Keys decompressed at a time (k_nope and v: 448 halves per key and head, 73 MB for the 20 heads).
    static let decBlock = 4096
    /// Groups per threadgroup and 16-row fragments per simdgroup of pattn_mla_dec.
    static let decTile = (3, 2)

    /// Whether a prefill chunk of `rows` rows ending at `keys` keys takes the decompressed attention. The path
    /// depends on the chunk's rows, so a prompt split at other cuts may change bits (splitcheck of GLM compares with
    /// a tolerance); within the path a row gives the same bits whatever its chunk (blocks of keys at fixed positions).
    func decompressed(rows: Int, keys: Int) -> Bool { rows >= Self.decMinRows && keys >= Self.decMinKeys }

    /// Allocates the decompressed path's buffers of `s` (W', one block of k_nope and v of `block` keys, the running
    /// state of `rows` rows) for `heads` heads of the GLM shape.
    static func decBuffers(_ ctx: MetalContext, _ s: Scratch, heads H: Int, rows: Int, block: Int = decBlock) {
        if s.wp == nil {
            s.wp = ctx.buffer(length: H * 448 * 512 * 2)
            s.kn = ctx.buffer(length: H * block * 192 * 2)
            s.vv = ctx.buffer(length: H * block * 256 * 2)
        }
        if s.decRows < rows {
            s.st = ctx.buffer(length: H * rows * 256 * 4)
            s.ml = ctx.buffer(length: H * rows * 8)
            s.decRows = rows
        }
    }

    /// What the decompressed attention reads of a layer and a chunk: the prework's q (`q`: H₆₄ q_pe in dims 0…63 of
    /// each head), the latent cache already written for the chunk (`kc`, `ks`, `vs`, stride `ldkv`), the `rows` rows
    /// of the chunk after `pos` keys and the score scale.
    struct DecInputs {
        var q, kc, ks, vs: MTLBuffer
        var ldkv, pos, rows: Int
        var scale: Float
    }

    /// Decompressed attention of a prefill chunk (q_nope comes from this layer's `.proj`). The output (half
    /// [rows, H · v_head_dim]) is o_proj's input, returned (no unembed_out).
    func encodeDecompressed(_ ctx: MetalContext, _ e: MTLComputeCommandEncoder, _ x: DecInputs) throws -> MTLBuffer {
        precondition(spec.qkNopeHeadDim == 192 && spec.vHeadDim == 256 && spec.kvLoraRank == 512)
        Self.decBuffers(ctx, scratch, heads: spec.heads, rows: x.rows)
        try Self.encodeDecompressed(
            ctx, e, embedQ: embedQ, unembedOut: unembedOut, scratch: scratch, heads: spec.heads, x)
        return scratch.out!
    }

    /// The kernels of `encodeDecompressed` over `s` (decBuffers done; `s.qNope` holds q_nope [rows, H · 192] and the
    /// output goes to `s.out`), in blocks of `block` keys (a multiple of 128 no larger than decBuffers').
    static func encodeDecompressed(
        _ ctx: MetalContext, _ e: MTLComputeCommandEncoder, embedQ: DenseQ, unembedOut: DenseQ, scratch s: Scratch,
        heads H: Int, _ x: DecInputs, block: Int = decBlock
    ) throws {
        let r = x.rows, pos = x.pos
        precondition(block % 128 == 0 && r <= s.decRows)
        let w = 448
        e.setComputePipelineState(try ctx.pipeline("mla_dec_weights"))
        for (i, b) in [embedQ.w, embedQ.s, embedQ.b, unembedOut.w, unembedOut.s, unembedOut.b, s.wp!].enumerated() {
            e.setBuffer(b, offset: 0, index: i)
        }
        e.dispatchThreadgroups(
            MTLSize(width: w, height: H, depth: 1), threadsPerThreadgroup: MTLSize(width: 576, height: 1, depth: 1))
        let end = (pos + r + 127) / 128 * 128, (rg, rf) = decTile
        var kb0 = 0
        while kb0 < end {
            let kb1 = min(kb0 + block, end)
            var p = (
                UInt32(r), UInt32(pos), UInt32(kb0), UInt32(kb1), UInt32(block),
                UInt32((kb0 == 0 ? 1 : 0) | (kb1 == end ? 2 : 0)), UInt32(x.ldkv), x.scale
            )
            e.setComputePipelineState(try ctx.pipeline("mla_dec_kv"))
            for (i, b) in [x.kc, x.vs, s.wp!, s.kn!, s.vv!].enumerated() { e.setBuffer(b, offset: 0, index: i) }
            e.setBytes(&p, length: MemoryLayout.size(ofValue: p), index: 5)
            e.dispatchThreadgroups(
                MTLSize(width: w / 64, height: (kb1 - kb0) / 64, depth: H),
                threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            e.setComputePipelineState(try ctx.pipeline("pattn_mla_dec_\(rg)_\(rf)"))
            for (i, b) in [s.qNope!, x.q, x.kc, x.ks, s.kn!, s.vv!, s.st!, s.ml!, s.out!].enumerated() {
                e.setBuffer(b, offset: 0, index: i)
            }
            e.setBytes(&p, length: MemoryLayout.size(ofValue: p), index: 9)
            e.dispatchThreadgroups(
                MTLSize(width: (r + 16 * rg * rf - 1) / (16 * rg * rf), height: H, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 128 * rg, height: 1, depth: 1))
            kb0 = kb1
        }
    }

    /// `.oproj`, first half: the attention output (half [rows, H · hd], from `xOffset`) → unembed_out per head (half
    /// [rows, H · v_head_dim], returned) for o_proj.
    func encodeOut(
        _ ctx: MetalContext, _ e: MTLComputeCommandEncoder, attn x: MTLBuffer, xOffset: Int = 0, rows: Int,
        prefill: Bool = false
    ) throws -> MTLBuffer {
        let s = buffers(ctx, rows: rows), H = spec.heads, hd = headDim, v = spec.vHeadDim
        try Self.encodeHeads(
            ctx, e, unembedOut, heads: H,
            x: HeadIO(buf: x, offset: xOffset + spec.qkRopeHeadDim * 2, ld: H * hd, perHead: hd, half: true),
            y: HeadIO(buf: s.out!, ld: H * v, perHead: v, half: true), rows: rows, prefill: prefill)
        return s.out!
    }
}
