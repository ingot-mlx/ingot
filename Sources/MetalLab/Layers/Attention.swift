import Foundation
import Metal

// MARK: - Full verification attention

struct AttnParams {
    var B: UInt32, ctx: UInt32, C: UInt32, nsplit: UInt32, ldkv: UInt32
    var scale: Float, eps: Float, theta: Float
    var dyn: UInt32 = 0
    /// Sliding window (keys a row sees, counting its own; 0 = full attention). Read only by the windowed kernels
    /// (`pattn_i8w_*`, `dattn_i8w_*`, `pattn_q8w_*`).
    var window: UInt32 = 0
    /// Ring KV of a sliding-window layer (slot = position % ring; 0 = linear).
    var ring: UInt32 = 0
    init(B: UInt32, ctx: UInt32, C: UInt32, nsplit: UInt32, ldkv: UInt32, scale: Float, eps: Float, theta: Float) {
        self.B = B; self.ctx = ctx; self.C = C; self.nsplit = nsplit; self.ldkv = ldkv
        self.scale = scale; self.eps = eps; self.theta = theta
    }
}

/// Full attention layer (Qwen3.5/3.6, Flash-Next, LFM2) with B ≤ 16 rows over a KV cache of `ctx` positions.
public final class AttentionLayer {
    public static let maxRows = 16
    /// Head dimension (ModelDims of the context: 256 in Qwen3.5/3.6, 64 in LFM2, 512 in Gemma 4's full layers, 576 in
    /// GLM's latent MLA; the kernels support 64, 128, 256, 512 and 576)
    /// and whether q_proj carries a per-head gate.
    public let hd: Int
    public let gated: Bool
    /// KV and query heads (2 / 16 in Qwen3.6, 4 / 24 in the 27B), queries per KV head, width of the q|gate|k|v
    /// projection and of the output.
    public let kvHeads: Int, qHeads: Int, G: Int, projN: Int, outDim: Int
    public let ctx: MetalContext
    public let layer: Int
    /// `proj`: the q|k|v projection (q_proj padded in a KV-sharing layer; with MLA, its first matmul [q_a | kv_a]).
    public let proj, oProj: DenseQ
    /// MLA (GLM-4.7-Flash, latent cache): the projections that build the q|k|v row in the `.proj` piece and the prefill,
    /// and unembed_out before o_proj in `.oproj` (nil = `proj` writes the row and o_proj reads the attention output).
    private(set) var mla: MLAProjection?
    public let qNorm, kNorm: MTLBuffer
    /// int8 KV, the only one (kc and vt hold int8 with the same layout; ks, vs: scale per token and head). K is stored
    /// rotated by the orthonormal Hadamard (H k, vattn_prework_i8), and the queries are rotated alike.
    /// MLA's latent head (`latentKV`): kc alone holds `[H₆₄ k_pe | H₅₁₂ c]`, ks the scale of k_pe and vs that of c, and
    /// V is read from kc's latent part; vt is a 16-byte placeholder that no kernel reads or writes.
    public private(set) var kc, vt: MTLBuffer
    /// Flash-Next QSA: per-row block mask ([rows, nw] uint32, qsa_m_select) for the next call to `encode` or
    /// `encodePrefill` (nil = dense attention). FlashQSA sets and clears it.
    public var qsaMask: (bm: MTLBuffer, nw: Int)?
    /// Block lists for the next call to `encode` (verification and decode; prefill keeps using `qsaMask`).
    /// With lists, dattn_i8l_1_4 replaces dattn_i8s and `nsplit` is the constant of the list.
    public var blockList: BlockList?
    public private(set) var ks, vs: MTLBuffer
    public let maxCtx: Int
    /// MLA's latent KV (head of 576, GLM-4.7-Flash): K only, with two scales; no vt (see `kc`).
    public let latentKV: Bool
    public private(set) var ldkv: Int
    public var pos = 0  // positions in the cache before the block
    /// KV-sharing layer (Gemma 4 E2B/E4B, ModelSpec.kvSource): the layer whose KV this one reads (its kc/vt/ks/vs,
    /// stride and ring are the same buffers and values); the prework only writes q (grid of `qHeads`), and the
    /// projection is q_proj alone padded with zero rows to the shape's width. nil = its own KV.
    public let kvSource: AttentionLayer?
    /// Draft tree (`--tree`): with a tree buffer, RoPE by depth and ancestor mask (zt_tree.metal).
    public var treeBuf: MTLBuffer?
    /// Fixed chunk (keys per chunk) for every B: a row then gives exactly the same result with B = 1 as with B = 8
    /// (greedy speculation has to match normal decoding bit for bit). nil: depends on B (128·B keys up to B = 8, 512
    /// with B = 16; used by the benches).
    public var fixedChunk: Int?
    public let eps: Float, theta: Float
    /// Attention score scale (BlockSpec.attnScale from the loader; 1/√hd by default).
    public var scale: Float
    /// Bias of q|k|v (fp32 [projN], the layout of `projOut`), added in the prework (LAB_AT_QKVB); nil = none.
    public var qkvBias: MTLBuffer?
    /// Layer without RoPE (`nope` bit of `AttnParams.dyn`: the prework rotates neither q nor k).
    public let nope: Bool
    /// Sliding window of the layer (keys a row sees, counting its own; 0 = full attention; ModelSpec.windows).
    public let window: Int
    /// Learned sinks per query head (fp32 [qHeads]; LAB_AT_SINKS): read where the softmax closes (prefill
    /// epilogue of pattn_i8, vattn_reduce). nil without sinks.
    public let sinks: MTLBuffer?
    /// RoPE table of the layer (fp32 [rotaryDims/2 + 1]: inv_freq per pair, then mscale; RopeTable): read by the
    /// prework kernels at buffer 11 when the library has LAB_AT_ROPETAB. nil without it.
    public let ropeTable: MTLBuffer?
    /// Ring KV of a sliding-window layer: `ring` slots per KV head (ldkv = ring), the key of position p in slot
    /// p % ring; 0 = linear KV (`ringSize`).
    public let ring: Int
    /// Largest prefill chunk the ring was sized for (encodePrefill refuses longer ones).
    public let ringRows: Int
    /// Highest position written + 1 since the KV last restarted from 0 (host side; with the chained decode, its upper
    /// bound). With a ring, the keys a restart at position c needs ([c − W + 1, c)) are intact while no write reached
    /// c − W + 1 + ring (`ringIntact`).
    public internal(set) var ringHigh = 0
    /// Lowest position whose key the ring holds: 0 after a restart from 0; P − W after restoring a prefix cache
    /// entry of P tokens, which only carries the window of P (`entryWindow`).
    var ringLow = 0

    /// Whether the KV can restart at position c (prefix cache entry, snapshot, `reuse`): always with a linear KV; with
    /// a ring, only if the window of c ([c − W + 1, c)) is still in the ring: not overwritten by a later write and not
    /// below what a restored entry brought back.
    public func ringIntact(_ c: Int) -> Bool {
        ring == 0 || (ringHigh <= c - window + ring && max(0, c - window + 1) >= ringLow)
    }

    func noteWrite(rows: Int) {
        if pos == 0 { ringHigh = 0; ringLow = 0 }
        ringHigh = max(ringHigh, pos + rows)
    }

    /// Positions of the KV a prefix cache entry of P tokens keeps for this layer: all of 0..<P with a linear
    /// KV; with a ring, the last W ([P − W, P); a restart at P reads from P − W + 1).
    func entryWindow(_ P: Int) -> (lo: Int, n: Int) {
        let lo = ring > 0 ? max(0, P - window) : 0
        return (lo, P - lo)
    }

    /// The ring after restoring an entry of P tokens: only `entryWindow(P)` is valid.
    func ringRestored(at P: Int) {
        guard ring > 0 else { return }
        ringLow = entryWindow(P).lo
        ringHigh = P
    }

    public let projOut, q, opart, ml, attnOut: MTLBuffer
    var vq: VQMV
    static let maxSplit = 1024
    /// Decode with int8 KV (dattn_i8): at most 2 groups of 16 queries per threadgroup and the rest in tg.z (B = 8:
    /// 32K 2.93 → 2.82 ms; 4K 0.50 → 0.41).
    static let decodeRG = 2
    /// Query groups per decode threadgroup. One token with more than 16 queries per KV head (GLM's latent attention:
    /// G = 20, two groups) and few key chunks: one group per threadgroup, twice the threadgroups for a grid that would
    /// leave most of the GPU idle (4 at 512 of context). Bit for bit the same (each group's rows are computed alone).
    /// GLM-4.7-Flash-4bit-h8 (lab-dev gen --mode plain): 19.80 -> 19.44 ms per token at 700 of context,
    /// 19.92 -> 19.51 at 1000, 19.92 -> 19.76 at 1500, even at 2000 (17 chunks) and +0.6 % at 4K (22): up to 16.
    ///
    /// Several rows in exactly 3 groups (6-8 rows with G 6: Qwen3.8-27B; 7-9 with G 5: Granite 4.1, Nemotron Nano):
    /// one group per threadgroup, so no threadgroup carries an empty group (RG 2 would launch 4 groups for 3). Bit for
    /// bit the same (the split of groups across threadgroups does not change any row's sum). Qwen3.8-27B 32K chain of
    /// 8 rows: attn.attn 10.25 -> 8.36 ms; same session (3 + 3) 130.65 -> 128.97 ms per round. With
    /// 5 or 7 groups RG 2 stays (12 rows, 16K tree: RG 1 is +0.9 ms per round).
    func decodeRG(rows: Int, splits: Int) -> Int {
        if rows == 1 && G > 16 && splits <= 16 { return 1 }
        if rows > 1 && (G * rows + 15) / 16 == 3 { return 1 }
        return Self.decodeRG
    }

    /// `shared`: another layer whose work buffers (projections, q, partials) are reused. `kvSource`: the layer whose
    /// KV this one reads (ModelSpec.kvSource; it has no k/v projections nor k_norm).
    public convenience init(
        ctx: MetalContext, store: SafetensorsStore, config cfg: ModelConfig, layer: Int, maxCtx: Int = 8192,
        shared: AttentionLayer? = nil, pool: KVPool? = nil, ringRows: Int = 2048, kvSource: AttentionLayer? = nil
    ) throws {
        // Family names (LFM2: out_proj, q_layernorm and k_layernorm; Nemotron-H: under `mixer.`).
        let sp = try cfg.requiredSpec
        let pre = try store.layerPrefix(layer) + sp.names.attn
        let d = ctx.device
        // K = V (Gemma 4's full layers, no v_proj): k_proj twice, so the v heads of the prework read the raw k
        // projection (normalized by v_norm, not rotated). Exact; it costs one more k matrix per full layer (a
        // choice that needs no kernel change).
        let keqv = sp.kEqV && cfg.layerTypes[layer] == "full_attention"
        // MLA (GLM-4.7-Flash, Layers/MLA.swift): its low-rank projections build the q|k|v row; `proj` is their first
        // matmul ([q_a | kv_a] over x), the one lab serve's W8A8 prefill may take.
        let mla = try sp.mla.map { m in
            guard let names = sp.names.mla else { throw SafetensorsError.format("MLA without its weight names") }
            return try MLAProjection(
                ctx: ctx, store: store, config: cfg, prefix: pre, spec: m, names: names, shared: shared?.mla)
        }
        // A KV-sharing layer: q_proj with zero rows up to the q|k|v width (the prework reads rows of AT_QP; only the q
        // heads run).
        let proj =
            if let mla { mla.qkvA } else if kvSource != nil {
                try DenseQ(store: store, config: cfg, paths: [pre + "q_proj"], device: d, padTo: ctx.dims.attnProjN)
            } else {
                try Self.qkvProjection(
                    store: store, config: cfg, prefix: pre, gate: sp.names.attnGate, keqv: keqv,
                    qHeads: ctx.dims.attnQH, headDim: ctx.dims.attnHD, device: d)
            }
        let oProj = try DenseQ(store: store, config: cfg, paths: [pre + sp.names.oProj], device: d)
        // Bits per matrix: 4 or 8 each (mixed q|k|v pieces come concatenated at 8 from DenseQ), or ternary;
        // the prefill GEMM has no 6-bit variant.
        // MLA: o_proj reads the unembedded heads (H · v_head_dim), not the attention output (H · 576).
        guard mla != nil || proj.N == ctx.dims.attnProjN, oProj.K == mla?.oProjK ?? ctx.dims.attnOutDim,
            [proj.bits, oProj.bits].allSatisfy({ [2, 4, 8].contains($0) })
        else {
            throw SafetensorsError.format(
                "unsupported attention shape (N \(proj.N), bits \(proj.bits)/\(oProj.bits): 4 or 8 per matrix)")
        }
        let rope = sp.ropes[layer] ?? sp.rope
        // QK-norm weight by its convention (BlockSpec.qkNorm): the tensor (`plain`), x·(1 + w) folded at load time as
        // the layer norms (no family uses it today), or a buffer of ones (`unit`: x · 1 is exact; also without QK-norm, where the
        // kernels skip it, LAB_AT_QKN = 0, and the buffer is not read).
        func qk(_ n: String) throws -> MTLBuffer {
            switch sp.block.qkNorm {
            case .plain: try plainNormTensor(store, pre + n, d, f32: ctx.dims.normF32)
            case .onePlus: try onePlusNormTensor(ctx, store, pre + n)
            case .unit, nil: Self.unitNorm(ctx)
            }
        }
        try self.init(
            ctx: ctx, layer: layer, maxCtx: maxCtx, proj: proj, oProj: oProj,
            qNorm: try qk(sp.names.qNorm), kNorm: kvSource != nil ? Self.unitNorm(ctx) : try qk(sp.names.kNorm),
            eps: Float(cfg.rmsNormEps), theta: rope.theta, nope: rope.nope, window: sp.windows[layer],
            sinks: ctx.dims.attnSinks ? try floatTensor(store, pre + sp.names.sinks, d) : nil,
            ropeTable: ctx.dims.attnRopeTab ? ctx.buffer(rope.table) : nil, shared: shared,
            pool: pool, ringRows: ringRows, kvSource: kvSource)
        self.mla = mla
        scale = sp.block.attnScale
        if sp.block.qkvBias {
            var b: [Float] = []
            for p in ["q_proj", "k_proj", "v_proj"] {
                let t = try store.info(pre + p + ".bias")
                switch t.dtype {
                case .f32: b += store.array(t, as: Float.self)
                case .bf16: b += store.array(t, as: UInt16.self).map(bf16ToFloat)
                default: throw SafetensorsError.format("\(pre)\(p).bias: expected fp32 or bf16")
                }
            }
            guard b.count == projN else { throw SafetensorsError.format("\(pre)q|k|v biases: \(b.count) ≠ \(projN)") }
            qkvBias = ctx.buffer(b)
        }
    }

    /// The q|k|v projection row as the prework reads it. With the gate inside q_proj (Qwen3.5/3.6, Flash-Next) or no
    /// gate, q_proj|k_proj|v_proj concatenated. With the gate in its own projection (`gate`, Muse Glimmer's gate_proj;
    /// LFM3's, one row per head, `headGateRowOrder`),
    /// q_proj|gate_proj|k_proj|v_proj concatenated and then its rows reordered (`gateRowOrder`) into Qwen's row,
    /// `[q_h | gate_h]` per query head and k, v behind: exact, whole rows move with their scales and biases, and the
    /// prework and its kernels do not change.
    static func qkvProjection(
        store: SafetensorsStore, config cfg: ModelConfig, prefix pre: String, gate: String?, keqv: Bool, qHeads: Int,
        headDim: Int, device d: MTLDevice
    ) throws -> DenseQ {
        let v = keqv ? "k_proj" : "v_proj"
        guard let gate else {
            return try DenseQ(store: store, config: cfg, paths: ["q_proj", "k_proj", v].map { pre + $0 }, device: d)
        }
        let qRows = qHeads * headDim
        let qOut = try QuantizedLinear(store: store, path: pre + "q_proj", config: cfg).outFeatures
        let gateOut = try QuantizedLinear(store: store, path: pre + gate, config: cfg).outFeatures
        // A gate per dimension (Muse: qHeads · headDim rows) or one scalar per head (LFM3: qHeads rows).
        guard qOut == qRows, gateOut == qRows || gateOut == qHeads else {
            let (p, rows) = qOut != qRows ? ("q_proj", qOut) : (gate, gateOut)
            throw SafetensorsError.format(
                "\(pre)\(p): \(rows) rows, expected \(qHeads) heads × \(headDim)\(p == gate ? " or \(qHeads)" : "") (the "
                    + "separate gate interleaves with q_proj head by head)")
        }
        let paths = ["q_proj", gate, "k_proj", v].map { pre + $0 }
        let cat = try DenseQ(store: store, config: cfg, paths: paths, device: d)
        let order =
            gateOut == qHeads
            ? headGateRowOrder(qHeads: qHeads, headDim: headDim, kvRows: cat.N - qRows - qHeads)
            : gateRowOrder(qHeads: qHeads, headDim: headDim, kvRows: cat.N - 2 * qRows)
        return DenseQ(gather: cat, rows: order, device: d)
    }

    /// Row order that turns `[q_proj | gate_proj | k_proj | v_proj]` with a scalar gate per head (LFM3: `qHeads` gate
    /// rows, `out_h · σ(gate_proj(x))_h`) into the same gated q row: for each head h, its q rows and then its gate row
    /// repeated `headDim` times, and k|v as they are. Exact: every repeated row gives the same value, so `at_gate` applies
    /// the same σ to the head's `headDim` dimensions.
    static func headGateRowOrder(qHeads: Int, headDim hd: Int, kvRows: Int) -> [Int] {
        let qRows = qHeads * hd
        return (0..<qHeads).flatMap { h in Array(h * hd..<(h + 1) * hd) + Array(repeating: qRows + h, count: hd) }
            + Array(qRows + qHeads..<qRows + qHeads + kvRows)
    }

    /// Row order that turns `[q_proj | gate_proj | k_proj | v_proj]` (`qHeads · headDim` rows each for q and the gate,
    /// `kvRows` for k and v) into the gated q row of the prework: for each head h, q rows h·hd..<(h+1)·hd and then the
    /// gate's, as Qwen's q_proj carries them (`AT_QS = 2 · AT_HD`, `at_gate`), and k|v as they are.
    static func gateRowOrder(qHeads: Int, headDim hd: Int, kvRows: Int) -> [Int] {
        let qRows = qHeads * hd
        return (0..<qHeads).flatMap { h in Array(h * hd..<(h + 1) * hd) + Array(qRows + h * hd..<qRows + (h + 1) * hd) }
            + Array(2 * qRows..<2 * qRows + kvRows)
    }

    /// Ring slots of a sliding-window layer of window `w` for prefill chunks of up to `rows` rows: the visible keys of
    /// a chunk ([pos − W + 1, pos + rows)) and of a verification block must be distinct slots, and a rejected draft row
    /// must not overwrite a key a later row still sees: W + max(rows, maxRows) + maxRows, rounded up to 128 (a 128-key
    /// step of pattn_i8 is then one contiguous block of slots). 0 (linear KV) if not smaller than the linear one.
    static func ringSize(window w: Int, rows: Int, maxCtx: Int) -> Int {
        guard w > 0 else { return 0 }
        let r = (w + max(rows, maxRows) + maxRows + 127) / 128 * 128
        return r < ldkv(for: maxCtx) ? r : 0
    }

    /// Weight of a weightless QK-norm: AT_HD ones in the type of the small weights (LAB_NWT: fp32 or bf16).
    static func unitNorm(_ ctx: MetalContext) -> MTLBuffer {
        let hd = ctx.dims.attnHD
        return ctx.dims.normF32
            ? ctx.buffer([Float](repeating: 1, count: hd)) : ctx.buffer([UInt16](repeating: 0x3F80, count: hd))
    }

    /// With matrices and norms already on the GPU (the loader above and per-kernel tests, with synthetic weights).
    init(
        ctx: MetalContext, layer: Int, maxCtx: Int, proj: DenseQ, oProj: DenseQ, qNorm: MTLBuffer, kNorm: MTLBuffer,
        eps: Float, theta: Float, nope: Bool = false, window: Int = 0, sinks: MTLBuffer? = nil,
        ropeTable: MTLBuffer? = nil, shared: AttentionLayer? = nil, pool: KVPool? = nil, ringRows: Int = 2048,
        kvSource: AttentionLayer? = nil
    ) throws {
        precondition(window == 0 || ctx.dims.attnSWA, "sliding window without LAB_AT_SWA in the library")
        precondition((sinks != nil) == ctx.dims.attnSinks, "attention sinks and LAB_AT_SINKS go together")
        precondition((ropeTable != nil) == ctx.dims.attnRopeTab, "RoPE table and LAB_AT_ROPETAB go together")
        guard ropeTable.map({ $0.length >= (ctx.dims.attnRot / 2 + 1) * 4 }) ?? true else {
            throw SafetensorsError.format("RoPE table: expected \(ctx.dims.attnRot / 2 + 1) values per layer")
        }
        guard sinks.map({ $0.length >= ctx.dims.attnQH * 4 }) ?? true else {
            throw SafetensorsError.format("attention sinks: expected \(ctx.dims.attnQH) values per layer")
        }
        self.ctx = ctx
        let dims = ctx.dims
        hd = dims.attnHD; gated = dims.attnGate; kvHeads = dims.attnKVH; qHeads = dims.attnQH; G = dims.attnG
        projN = dims.attnProjN; outDim = dims.attnOutDim
        self.layer = layer
        self.maxCtx = maxCtx
        vq = VQMV(ctx)
        self.proj = proj
        self.oProj = oProj
        self.qNorm = qNorm
        self.kNorm = kNorm
        self.eps = eps
        self.theta = theta
        self.nope = nope
        self.window = window
        self.sinks = sinks
        self.ropeTable = ropeTable
        self.ringRows = ringRows
        scale = 1 / Float(dims.attnHD).squareRoot()
        latentKV = Self.latent(hd: dims.attnHD)
        self.kvSource = kvSource
        if let src = kvSource {
            // The source's KV itself (same shape, window and stride): nothing of its own.
            precondition(
                pool == nil && src.ctx.dims == dims && src.window == window && src.maxCtx == maxCtx
                    && src.kvSource == nil, "KV-sharing layer with another shape than its source")
            ring = src.ring; ldkv = src.ldkv
            kc = src.kc; vt = src.vt; ks = src.ks; vs = src.vs
        } else {
            ring = pool == nil ? Self.ringSize(window: window, rows: ringRows, maxCtx: maxCtx) : 0
            // KV that grows on demand (KVPool): buffers for the pool's current capacity (it reallocates them on growth).
            ldkv = ring > 0 ? ring : Self.ldkv(for: pool?.capacity ?? maxCtx)
            kc = ctx.buffer(length: kvHeads * ldkv * hd)
            vt = ctx.buffer(length: latentKV ? Self.placeholderBytes : kvHeads * ldkv * hd)
            memset(kc.contents(), 0, kc.length); memset(vt.contents(), 0, vt.length)
            ks = ctx.buffer(length: kvHeads * ldkv * 4); vs = ctx.buffer(length: kvHeads * ldkv * 4)
            memset(ks.contents(), 0, ks.length); memset(vs.contents(), 0, vs.length)
        }
        let R = Self.maxRows
        if let o = shared {
            projOut = o.projOut; q = o.q; opart = o.opart; ml = o.ml; attnOut = o.attnOut
        } else {
            projOut = ctx.buffer(length: R * projN * 4)
            q = ctx.buffer(length: kvHeads * G * R * hd * 2)
            opart = ctx.buffer(length: kvHeads * Self.maxSplit * G * R * hd * 4 / 4)
            ml = ctx.buffer(length: kvHeads * Self.maxSplit * G * R * 8)
            attnOut = ctx.buffer(length: R * outDim * 2)
        }
        if let pool {
            pool.bytesPerPosition += Self.kvBytesPerPosition(kvHeads: kvHeads, hd: hd)
            pool.resizers.append { [unowned self, unowned pool] cap in self.resize(capacity: cap, pool: pool) }
        }
    }

    /// Whether a head of `hd` is MLA's latent one (K only with two scales, V read from K; `latentKV`).
    static func latent(hd: Int) -> Bool { hd == 576 }
    /// Size of vt with a latent KV (a placeholder: Metal wants a buffer bound at every index the kernels declare).
    static let placeholderBytes = 16

    /// int8 KV bytes per position of a layer: K and transposed V with one fp32 scale each per KV head (2·(hd + 4));
    /// MLA's latent head keeps K alone with two scales (hd + 8: 584 instead of 1,160 bytes).
    static func kvBytesPerPosition(kvHeads: Int, hd: Int) -> Int {
        kvHeads * (latent(hd: hd) ? hd + 8 : 2 * (hd + 4))
    }

    /// KV stride for `n` positions: margin of one maximum chunk (2048 keys).
    static func ldkv(for n: Int) -> Int { ((n + maxRows + 2047) / 2048 + 1) * 2048 }

    /// KVPool: reallocates the int8 KV for `cap` positions, copying the old contents with the new stride (GPU idle).
    func resize(capacity cap: Int, pool: KVPool) {
        let ld0 = ldkv, ld1 = Self.ldkv(for: cap), H = kvHeads, D = hd, m = min(ld0, ld1)
        func fresh(_ n: Int) -> MTLBuffer { ctx.zeroedBuffer(length: n) }
        let k1 = fresh(H * ld1 * D), ks1 = fresh(H * ld1 * 4), vs1 = fresh(H * ld1 * 4)
        let v1 = latentKV ? vt : fresh(H * ld1 * D)
        pool.blit { b in
            KVPool.copyRows(b, kc, k1, rows: H, n: m * D, ss: ld0 * D, ds: ld1 * D)
            if !latentKV { KVPool.copyRows(b, vt, v1, rows: H * D, n: m, ss: ld0, ds: ld1) }
            KVPool.copyRows(b, ks, ks1, rows: H, n: m * 4, ss: ld0 * 4, ds: ld1 * 4)
            KVPool.copyRows(b, vs, vs1, rows: H, n: m * 4, ss: ld0 * 4, ds: ld1 * 4)
        }
        kc = k1; vt = v1; ks = ks1; vs = vs1
        ldkv = ld1
    }

    public func chunk(rows: Int) -> Int { fixedChunk ?? (rows >= 16 ? 512 : min(2048, max(128, 128 * rows))) }
    public func splits(rows: Int) -> Int { (pos + rows + chunk(rows: rows) - 1) / chunk(rows: rows) }

    public enum Piece: String, CaseIterable { case proj, prework, attn, reduce, oproj }

    /// With `state` (chained decode), the real position is read on the GPU (st[0]) and `pos` is only an upper bound
    /// for the number of chunks (empty chunks give (−∞, 0) and the reduction skips them: the result does not change).
    /// `rowOffset` (batched decoding): the state pieces (prework, attn, reduce) of these `rows` work on rows
    /// `rowOffset…` of `projOut` and `attnOut`, the rows of this conversation in a round of several.
    public func encode(
        _ enc: MTLComputeCommandEncoder, x: MTLBuffer, xOffset: Int = 0, out: MTLBuffer, outOffset: Int = 0, rows: Int,
        accumulate: Bool = false, only: Piece? = nil, state: MTLBuffer? = nil, rowOffset: Int = 0
    ) throws {
        precondition(rows >= 1 && rows <= Self.maxRows && pos + rows <= maxCtx + Self.maxRows)
        precondition(
            rowOffset == 0
                || (rowOffset + rows <= Self.maxRows && state == nil && treeBuf == nil && blockList == nil
                    && [.prework, .attn, .reduce].contains(only)),
            "rowOffset: state pieces of the plain round only")
        let po = rowOffset * projN * 4, ao = rowOffset * outDim * 2
        let bl = treeBuf == nil ? blockList : nil
        let C = chunk(rows: rows), ns = bl?.nsplit ?? splits(rows: rows)
        precondition(ns * rows <= Self.maxSplit * 4 && (bl != nil || ring > 0 || ns * C <= ldkv))
        // With a window, the splits that end before the window start of the lowest possible row hold no visible
        // key; they are not launched (the first launched split goes in dyn >> 16, and nsplit counts only the launched
        // ones). `pos` is exact without `state` and at most maxRows − 1 above the real position with it. Bit for bit
        // the same: a skipped split only added (−inf, 0) to the reduction.
        let s0 = window > 0 && bl == nil && treeBuf == nil ? max(0, pos - Self.maxRows - window + 1) / C : 0
        var ap = AttnParams(
            B: UInt32(rows), ctx: UInt32(pos), C: UInt32(C), nsplit: UInt32(ns - s0), ldkv: UInt32(ldkv),
            scale: scale, eps: eps, theta: theta)
        ap.dyn = (state == nil ? 0 : Self.dynState) | dynNope | UInt32(s0) << 16
        ap.window = UInt32(window)
        ap.ring = UInt32(ring)
        if only == nil || only == .prework { noteWrite(rows: rows) }
        precondition(
            window == 0 || (treeBuf == nil && qsaMask == nil && bl == nil), "sliding window: no tree, mask or lists")
        let st = state ?? projOut
        if only == nil || only == .proj {
            if let mla {
                try mla.encode(ctx, enc, vq: vq, x: x, xOffset: xOffset, proj: projOut, projN: projN, rows: rows)
            } else {
                try vq.encode(enc, x: x, xOffset: xOffset, proj, y: projOut, rows: rows)
            }
        }
        if only == nil || only == .prework {
            enc.setComputePipelineState(try ctx.pipeline(treeBuf != nil ? "vattn_prework_i8t" : "vattn_prework_i8"))
            if let tb = treeBuf { enc.setBuffer(tb, offset: 0, index: 10) }
            if let qb = qkvBias { enc.setBuffer(qb, offset: 0, index: 12) }
            for (i, b) in [projOut, qNorm, kNorm, q, kc, ks].enumerated() {
                enc.setBuffer(b, offset: i == 0 ? po : 0, index: i)
            }
            enc.setBytes(&ap, length: MemoryLayout<AttnParams>.size, index: 6)
            enc.setBuffer(vt, offset: 0, index: 7)
            enc.setBuffer(vs, offset: 0, index: 8); enc.setBuffer(st, offset: state == nil ? po : 0, index: 9)
            if let rt = ropeTable { enc.setBuffer(rt, offset: 0, index: 11) }
            enc.dispatchThreadgroups(
                MTLSize(width: preworkSlots, height: rows, depth: 1),
                threadsPerThreadgroup: MTLSize(width: hd, height: 1, depth: 1))
        }
        if only == nil || only == .attn {
            if let bl {
                // One list per row (T = 1), grid (chunk, KV head, row) of 128 threads (verify_attn_pl.metal).
                precondition(bl.tileRows == 1 && bl.blockTokens == 4 && G <= 16)
                var lp = SparseListParams(
                    stride: UInt32(bl.stride), tiles: UInt32(rows), perKVHead: bl.perKVHead ? 1 : 0,
                    perSplit: UInt32(bl.perSplit))
                enc.setComputePipelineState(try ctx.pipeline("dattn_i8l_1_4"))
                for (i, b) in [q, kc, vt, ks, vs].enumerated() { enc.setBuffer(b, offset: 0, index: i) }
                enc.setBytes(&ap, length: MemoryLayout<AttnParams>.size, index: 7)
                enc.setBuffer(opart, offset: 0, index: 8); enc.setBuffer(ml, offset: 0, index: 9)
                enc.setBuffer(st, offset: 0, index: 10)
                enc.setBuffer(bl.lists, offset: 0, index: 14); enc.setBuffer(bl.counts, offset: 0, index: 15)
                enc.setBytes(&lp, length: MemoryLayout<SparseListParams>.size, index: 16)
                enc.dispatchThreadgroups(
                    MTLSize(width: ns, height: kvHeads, depth: rows),
                    threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            } else if hd == 576, treeBuf == nil, qsaMask == nil, s0 == 0, C % 256 == 0 {
                // MLA's latent head: mattn_i8 (verify_attn_pm.metal), 8 simdgroups per group of 16 queries
                // and steps of 256 keys; one group per threadgroup, the groups in tg.z, for any number of rows.
                // GLM-4.7-Flash-4bit-h8, decode --copy off against dattn_i8 in the same session (medians of 4 + 4):
                // 512 17.90 -> 17.20 ms per token, 4K 19.97 -> 18.23, 16K 24.89 -> 22.56, 20.8K 28.34 -> 27.53.
                enc.setComputePipelineState(try ctx.pipeline("mattn_i8"))
                for (i, b) in [q, kc, vt, ks, vs].enumerated() { enc.setBuffer(b, offset: 0, index: i) }
                enc.setBytes(&ap, length: MemoryLayout<AttnParams>.size, index: 7)
                enc.setBuffer(opart, offset: 0, index: 8); enc.setBuffer(ml, offset: 0, index: 9)
                enc.setBuffer(st, offset: state == nil ? po : 0, index: 10)
                enc.dispatchThreadgroups(
                    MTLSize(width: ns, height: kvHeads, depth: (G * rows + 15) / 16),
                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            } else {
                // All rows of the block in one threadgroup per chunk (verify_attn_pf.metal: dattn_i8).
                // rg groups of 16 queries; at most decodeRG(rows:splits:) per threadgroup (the rest in tg.z).
                let groups = (G * rows + 15) / 16, rg = min(groups, decodeRG(rows: rows, splits: ns - s0))
                let nz = (groups + rg - 1) / rg
                // With a tree, the "per-row" threadgroups (rows / rg) come before the nz regular ones; they handle the
                // chunks that touch the block (verify_attn_pf.metal).
                let nzRows = treeBuf != nil ? (rows + rg - 1) / rg : 0
                if treeBuf != nil { ap.dyn |= UInt32(nzRows) << 8 }
                precondition((qsaMask == nil && blockList == nil) || treeBuf == nil)
                enc.setComputePipelineState(
                    try ctx.pipeline(
                        treeBuf != nil
                            ? "dattn_i8t_\(rg)" : qsaMask != nil ? "dattn_i8s_\(rg)" : "dattn_i8\(swa)_\(rg)"))
                if let tb = treeBuf { enc.setBuffer(tb, offset: 0, index: 13) }
                if let qm = qsaMask {
                    var nw = UInt32(qm.nw)
                    enc.setBuffer(qm.bm, offset: 0, index: 14)
                    enc.setBytes(&nw, length: 4, index: 15)
                }
                for (i, b) in [q, kc, vt, ks, vs, projOut, attnOut].enumerated() {
                    enc.setBuffer(b, offset: i == 5 ? po : i == 6 ? ao : 0, index: i)
                }
                enc.setBytes(&ap, length: MemoryLayout<AttnParams>.size, index: 7)
                enc.setBuffer(opart, offset: 0, index: 8); enc.setBuffer(ml, offset: 0, index: 9)
                enc.setBuffer(st, offset: state == nil ? po : 0, index: 10)
                enc.dispatchThreadgroups(
                    MTLSize(width: ns - s0, height: kvHeads, depth: nz + nzRows),
                    threadsPerThreadgroup: MTLSize(width: 128 * rg, height: 1, depth: 1))
            }
        }
        if only == nil || only == .reduce {
            enc.setComputePipelineState(try ctx.pipeline("vattn_reduce"))
            for (i, b) in [opart, ml, projOut, attnOut].enumerated() {
                enc.setBuffer(b, offset: i == 2 ? po : i == 3 ? ao : 0, index: i)
            }
            enc.setBytes(&ap, length: MemoryLayout<AttnParams>.size, index: 4)
            if let sinks { enc.setBuffer(sinks, offset: 0, index: 5) }
            enc.dispatchThreadgroups(
                MTLSize(width: G * rows, height: kvHeads, depth: 1),
                threadsPerThreadgroup: MTLSize(width: hd, height: 1, depth: 1))
        }
        if only == nil || only == .oproj {
            let x = try mla?.encodeOut(ctx, enc, attn: attnOut, rows: rows) ?? attnOut
            try vq.encode(enc, x: x, oProj, y: out, yOffset: outOffset, rows: rows, accumulate: accumulate)
        }
    }

    // MARK: Conversation slots

    /// The KV of one conversation and its counters (batched decoding, `HybridModel.generateBatch`): what
    /// `exchangeSlot` swaps with the live fields of the layer. Slot 0 is the layer's own buffers.
    struct KVSlot {
        var kc, vt, ks, vs: MTLBuffer
        var ldkv, pos: Int
        var fixedChunk: Int?
        var ringHigh, ringLow: Int
    }

    /// A fresh slot for `positions` positions (int8 KV; a ring layer keeps its ring, whose size does not depend on
    /// the context).
    func makeSlot(positions n: Int) -> KVSlot {
        let ld = ring > 0 ? ring : Self.ldkv(for: n)
        func zeroed(_ len: Int) -> MTLBuffer { ctx.zeroedBuffer(length: len) }
        return KVSlot(
            kc: zeroed(kvHeads * ld * hd), vt: zeroed(latentKV ? Self.placeholderBytes : kvHeads * ld * hd),
            ks: zeroed(kvHeads * ld * 4),
            vs: zeroed(kvHeads * ld * 4), ldkv: ld, pos: 0, fixedChunk: fixedChunk, ringHigh: 0, ringLow: 0)
    }

    /// Swaps the live KV and counters with those of `s` (only references: O(1), nothing is copied).
    func exchangeSlot(_ s: inout KVSlot) {
        swap(&kc, &s.kc); swap(&vt, &s.vt); swap(&ks, &s.ks); swap(&vs, &s.vs)
        swap(&ldkv, &s.ldkv); swap(&pos, &s.pos); swap(&fixedChunk, &s.fixedChunk)
        swap(&ringHigh, &s.ringHigh); swap(&ringLow, &s.ringLow)
    }

    /// Bits of `AttnParams.dyn` (verify_attn.metal): position of the round state (st[0]) and layer without RoPE.
    static let dynState: UInt32 = 1, dynNopeBit: UInt32 = 2
    var dynNope: UInt32 { nope ? Self.dynNopeBit : 0 }
    /// Heads the prework runs: q, k and v (q and k with a latent KV: V is K's latent part); only q in a KV-sharing
    /// layer (the source already wrote this row's KV).
    var preworkSlots: Int { qHeads + (kvSource == nil ? (latentKV ? 1 : 2) * kvHeads : 0) }
    /// Kernel name suffix of the windowed attention (verify_attn_pf.metal, SWA).
    var swa: String { window > 0 ? "w" : "" }

}

// MARK: - Prefill (chunks of many rows)

extension AttentionLayer {
    /// Prefill chunk: causal attention of `rows` rows over the cache, with the new K/V written to the cache: pattn_i8
    /// (single-pass flash, 2 groups of 16 queries per threadgroup; pattn64_i8 for the exact full layers with 64-wide
    /// heads).
    func encodePrefill(
        _ e0: MTLComputeCommandEncoder?, x: MTLBuffer, out: MTLBuffer, rows r: Int, padded M: Int,
        s: (proj: MTLBuffer, q: MTLBuffer, out: MTLBuffer),
        i8: DenseI8Scratch? = nil, i8Out: DenseI8Scratch? = nil, qk8: (q: MTLBuffer, s: MTLBuffer)? = nil,
        tl: GPUTimeline? = nil
    ) throws {
        precondition(pos + r <= maxCtx)
        var ap = AttnParams(
            B: UInt32(r), ctx: UInt32(pos), C: UInt32(pos + r), nsplit: 1, ldkv: UInt32(ldkv),
            scale: scale, eps: eps, theta: theta)
        ap.dyn = dynNope
        ap.window = UInt32(window)
        ap.ring = UInt32(ring)
        precondition(window == 0 || qsaMask == nil, "sliding window: no QSA mask")
        precondition(ring == 0 || r <= ringRows, "prefill chunk of \(r) rows over the ring sized for \(ringRows)")
        noteWrite(rows: r)
        try tseg(tl, e0, "attn.proj") {
            if let mla {
                try mla.encode(ctx, $0, vq: vq, x: x, proj: s.proj, projN: projN, rows: M, prefill: true, i8: i8)
            } else {
                try proj.encodePrefillGemm(ctx, $0, x: x, y: s.proj, rows: M, i8: i8)
            }
        }
        try tseg(tl, e0, "attn.prework") { e in
            e.setComputePipelineState(try ctx.pipeline("vattn_prework_i8"))
            if let qb = qkvBias { e.setBuffer(qb, offset: 0, index: 12) }
            for (i, b) in [s.proj, qNorm, kNorm, s.q, kc, ks].enumerated() { e.setBuffer(b, offset: 0, index: i) }
            e.setBytes(&ap, length: MemoryLayout<AttnParams>.size, index: 6)
            e.setBuffer(vt, offset: 0, index: 7)
            e.setBuffer(vs, offset: 0, index: 8); e.setBuffer(s.proj, offset: 0, index: 9)
            if let rt = ropeTable { e.setBuffer(rt, offset: 0, index: 11) }
            e.dispatchThreadgroups(
                MTLSize(width: preworkSlots, height: r, depth: 1),
                threadsPerThreadgroup: MTLSize(width: hd, height: 1, depth: 1))
        }
        // GLM, long chunk: decompressed attention (zn_mla_dec.metal), straight into o_proj's input.
        if let mla, mla.decompressed(rows: r, keys: pos + r) {
            var x: MTLBuffer?
            try tseg(tl, e0, "attn.attn") {
                x = try mla.encodeDecompressed(
                    ctx, $0, .init(q: s.q, kc: kc, ks: ks, vs: vs, ldkv: ldkv, pos: pos, rows: r, scale: scale))
            }
            try tseg(tl, e0, "attn.oproj") {
                try oProj.encodePrefillGemm(ctx, $0, x: x!, y: out, rows: M, i8: i8Out, accumulate: true)
            }
            pos += r
            return
        }
        // Single-pass flash attention (verify_attn_pf.metal): normalized output with the gate applied, in s.out.
        let rg = 2
        try tseg(tl, e0, "attn.attn") { e in
            if let qk8 {
                var n = UInt32(kvHeads * G * r)
                e.setComputePipelineState(try ctx.pipeline("q_quant_i8"))
                for (i, b) in [s.q, qk8.q, qk8.s].enumerated() { e.setBuffer(b, offset: 0, index: i) }
                e.setBytes(&n, length: 4, index: 3)
                e.dispatchThreadgroups(
                    MTLSize(width: (Int(n) + 3) / 4, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            }
            // 64-wide heads, full layers, exact Q·Kᵀ: pattn64_i8 (verify_attn_pf.metal), with two query fragments
            // per simdgroup from 12K keys on (same bits either way).
            let p64 = hd == 64 && window == 0 && qsaMask == nil && qk8 == nil && mla == nil
            let tq = pos + r >= 12_288 ? 2 : 1
            e.setComputePipelineState(
                try ctx.pipeline(
                    p64
                        ? "pattn64_i8_t\(tq)"
                        : qk8 != nil
                            ? (qsaMask != nil ? "pattn_q8s_\(rg)" : "pattn_q8\(swa)_\(rg)")
                            : qsaMask != nil ? "pattn_i8s_\(rg)" : "pattn_i8\(swa)_\(rg)"))
            if let qm = qsaMask {
                var nw = UInt32(qm.nw)
                e.setBuffer(qm.bm, offset: 0, index: 14)
                e.setBytes(&nw, length: 4, index: 15)
            }
            for (i, b) in [s.q, kc, vt, ks, vs, s.proj, s.out].enumerated() { e.setBuffer(b, offset: 0, index: i) }
            if let sinks { e.setBuffer(sinks, offset: 0, index: 16) }
            if let qk8 { e.setBuffer(qk8.q, offset: 0, index: 11); e.setBuffer(qk8.s, offset: 0, index: 12) }
            e.setBytes(&ap, length: MemoryLayout<AttnParams>.size, index: 7)
            if p64 {
                e.dispatchThreadgroups(
                    MTLSize(width: (G * r + 64 * tq - 1) / (64 * tq), height: kvHeads, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            } else {
                e.dispatchThreadgroups(
                    MTLSize(width: (G * r + 16 * rg - 1) / (16 * rg), height: kvHeads, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 128 * rg, height: 1, depth: 1))
            }
            if latentKV {
                // V was read rotated from K (H₅₁₂ c): rotate the latent output back in place (zm_mla.metal).
                e.setComputePipelineState(try ctx.pipeline("mla_out_unrotate"))
                e.setBuffer(s.out, offset: 0, index: 0)
                e.dispatchThreadgroups(
                    MTLSize(width: qHeads, height: r, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: hd, height: 1, depth: 1))
            }
        }
        try tseg(tl, e0, "attn.oproj") {
            let x = try mla?.encodeOut(ctx, $0, attn: s.out, rows: M, prefill: true) ?? s.out
            try oProj.encodePrefillGemm(ctx, $0, x: x, y: out, rows: M, i8: i8Out, accumulate: true)
        }
        pos += r
    }
}
