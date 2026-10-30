import Foundation
import Metal

/// Complete Qwen3.5/3.6 with ingot's kernels: Qwen3.6-35B-A3B (40 layers: 30 GDN + 10 attention, all with
/// MoE) or dense Qwen3.8-27B (64 layers: 48 GDN + 16 attention, dense FFN); 4-bit embeddings and lm_head. Prefill in
/// chunks (≤ 2048 rows) and decode with B rows (1 = normal, 8 = DFlash2 verification). Each decode round is one
/// command buffer with a single CPU wait at the end. The dimensions come from ModelDims (MetalContext). The
/// generation loops are in Generate.swift and the per-layer prefill in Layers/. Here: loading, state and the shared
/// pieces; prefill in ModelPrefill.swift, captures in ModelCapture.swift and the round in ModelRound.swift.
///
/// Also LFM2.5-24B-A2B (lfm2_moe): 40 layers (30 short-convolution + 10 attention with 64-wide heads), the first two
/// with dense FFN and the rest 64-expert top-4 MoE with sigmoid router; tied 6-bit embedding that acts
/// as lm_head. No drafter.
public final class HybridModel {
    /// `none`: a layer without mixer (Nemotron-H's MLP and MoE blocks: norm, FFN, residual).
    public enum Mixer { case gdn(GDNLayer), attn(AttentionLayer), conv(ShortConvLayer), none }
    public struct Layer {
        public let n1: MTLBuffer, n2: MTLBuffer, mixer: Mixer
        /// MoE (Qwen3.6) or dense FFN (27B): one of the two, both with `parallel` (Gemma 4 26B-A4B), or neither
        /// (Nemotron-H's Mamba2 and attention blocks: norm, mixer, residual).
        public let moeOpt: MoELayer?, moeW: [DeviceQuantizedLinear], dense: DenseMLP?
        public var moe: MoELayer { moeOpt! }
        var hasFFN: Bool { moeOpt != nil || dense != nil }
        var hasMixer: Bool {
            if case .none = mixer { return false }
            return true
        }
        /// Branch epilogue of the layer (nil = the branches add straight into the residual stream).
        public var post: LayerPost?
        /// Norms of the parallel FFN (FFNKind.denseAndMoE; nil otherwise): `n2` is the dense MLP's pre-norm.
        public var parallel: ParallelNorms?
    }

    /// Gemma 4 26B-A4B's parallel FFN (ParallelFFN.swift): the MoE's pre-norm and the post-norms of the dense MLP and of
    /// the MoE, applied before both outputs are summed into the FFN branch.
    public struct ParallelNorms {
        public let moePre: MTLBuffer, densePost: MTLBuffer, moePost: MTLBuffer
    }

    /// What a layer applies to its branch outputs before adding them to the residual stream
    /// (`branch_add_rows`): post-norm weights of the mixer and FFN branches, o_proj bias (fp32) and layer scalar.
    public struct LayerPost {
        public let mixerNorm: MTLBuffer?, ffnNorm: MTLBuffer?, oBias: MTLBuffer?
        public let scalar: Float
    }

    public let ctx: MetalContext
    /// Context of the full attention layers when their shape differs from the model's (ModelSpec.fullShape); nil
    /// otherwise.
    public let fullCtx: MetalContext?
    public let cfg: ModelConfig
    /// Family, weight names, RoPE, router... (ModelSpec.swift).
    public let spec: ModelSpec
    public let layers: [Layer]
    public let embed, lmHead: DenseQ
    public let finalNorm: MTLBuffer
    public let hidden: Int, vocab: Int, eps: Float
    public var isMoE: Bool { cfg.isMoE }
    public let maxCtx: Int, chunkRows: Int
    public let loadSeconds: Double

    // Decode (B ≤ 16).
    public let tok, h, xn, logits, top1: MTLBuffer
    /// Branch outputs (fp32, zero between branches) of the round and of the prefill, if some layer has a branch
    /// epilogue; weight of the embedding norm, if the block has one.
    let branch: MTLBuffer?, pbranch: MTLBuffer?
    /// Output of one half of the parallel FFN (dense MLP or MoE, fp32, zero between uses) before its post-norm adds it
    /// to `branch` / `pbranch` (Gemma 4 26B-A4B; nil without it).
    let branchPart: MTLBuffer?, pbranchPart: MTLBuffer?
    let embedNormW: MTLBuffer?
    public var vqHead: VQMV
    // Prefill (≤ chunkRows rows).
    let ptok, pxn: MTLBuffer
    /// Residual stream of the current chunk; in a group of chunks, each chunk has its own (`phPool`, the first is the
    /// usual one).
    var ph: MTLBuffer
    var phPool: [MTLBuffer]
    /// Drafter captures of the second chunk with drafter of a group (chunkRows rows; created on first use).
    var capsAlt: MTLBuffer?
    /// Flash-Next: those of the third and later chunks with drafter of a group (up to `flashDraftParts`).
    var capsExtra: [MTLBuffer] = []
    /// Chunks with drafter per group in Flash-Next (with a prefix-cache cut in the pass, the windows
    /// of the cut and of the prompt end usually touch 3 chunks; each extra one costs 52 MB with 2048-row chunks).
    static let flashDraftParts = 3
    /// Short convolution of the prefill (LFM2): in_proj fp32 [C, 3D] and y half [C, D].
    let scs: (proj: MTLBuffer, y: MTLBuffer)
    /// Engine options (EngineOptions.swift): set by whoever creates the model; read each time they are used.
    public var options: EngineOptions
    /// Calibration of the verification rows by the drafter's confidence (SpecRows.swift), from the family file
    /// or the model's (`SpecRowsConfig.load`). nil = no calibration: each DFlash2 round verifies all 8 rows
    /// (dense 27B). `options.bench.specRowsFixed` overrides both.
    public var rowsConfig: SpecRowsConfig?
    /// State of the draft tree (`options.treeNodes`; TreeVerify.swift).
    public lazy var treeState: MTLBuffer = ctx.buffer(length: Self.treeWords * 4)
    /// Diagnostics (`ingot-dev fncheck`): prefill and verification stop after this layer (the residual stream stays
    /// in ph / h).
    public var debugStopAfter: Int?
    /// MoE routing trace of each decode round (RouteTrace.swift); with it the rounds are not chained.
    public var routeTrace: RouteTrace?
    /// GPU timestamps inside the `decodeSpec` rounds (`--round-trace N`).
    public var roundTrace: RoundTrace? { didSet { expertStream?.roundTrace = roundTrace } }
    /// SSD expert streaming: experts read from the SSD with a memory cache (`options.expertCacheBytes`).
    public let expertStream: ExpertStream?
    /// Qwen3.8-Flash-Next (FlashParts.swift): hyper-connections, PLE and MoE with Q2 experts from the SSD.
    public let fn: FlashParts?
    /// Gemma 4 E2B/E4B per-layer inputs (GemmaPLE.swift); nil in the other models.
    public let ple: GemmaPLE?
    /// Maximum rows of a verification round.
    public static let decodeRows = 16
    /// End tokens (generation_config.json; by default those of Qwen3.5/3.6).
    public var eosSet: Set<UInt32> = HybridModel.eos
    let gs:
        (
            proj: MTLBuffer, hcur: MTLBuffer, q: MTLBuffer, k: MTLBuffer, v: MTLBuffer, g: MTLBuffer, b: MTLBuffer,
            y: MTLBuffer, o: MTLBuffer
        )
    let ats: (proj: MTLBuffer, q: MTLBuffer, out: MTLBuffer)
    public let pmoe: PrefillMoE?
    /// Prefill dense FFN: gate|up half [C, 2·I] and SwiGLU half [C, I].
    let pff: (gu: MTLBuffer, h: MTLBuffer)?
    /// Prefill GDN in chunks of 64 (WY form on NAX); nil in LFM2 (no GDN).
    let gdnChunk: GDNChunk?

    /// Tokens in the cache (position of the next) and accepted rows of the previous round that the GDN has to fold.
    public internal(set) var P = 0
    public var nacc = 0
    /// Slots of the other conversations of a batched decode (BatchDecode.swift); nil until the first one.
    var convSlots: ConversationSlots?
    /// ingot serve batching: asked before each extra chained round of `decodeSpec`/`decodePlain`; true stops
    /// without enqueueing it, so the last emitted token stays outside the cache (`decodeYielded` says so) and the
    /// conversation can go on in the joint rounds.
    var decodeYield: (() -> Bool)?
    var decodeYielded = false
    /// Key chunk of the decode attention: fixed for a whole generation (the same for B = 1 and B = 8, so each
    /// row gives the same result) and chosen by prompt length so there are enough chunks: 512 below 8K,
    /// 1024 up to 16K and 2048 above (measured with ingot-dev gen --profile).
    public static func decodeChunk(promptTokens n: Int) -> Int {
        n < 8192 ? 512 : n < 16384 ? 1024 : 2048
    }

    /// Decode key chunk of the latent (MLA) layers of GLM-4.7-Flash, fixed per generation like
    /// `decodeChunk(promptTokens:)` and the same for 1…16 rows. With the 576 head one step of 128 keys of a threadgroup
    /// takes ~16 µs whatever the grid, so with one row the time follows the steps of the longest chunk: 256 below 1536
    /// tokens (more threadgroups for few keys), 512 up to 8K and 1024 from there on (2048 left the GPU half empty at
    /// 20.8K). GLM-4.7-Flash-4bit-h8 against the previous policy (512/1024/2048), rounds in one command buffer: 1 row
    /// 19.34 -> 17.76 ms at 563 positions, 19.50 -> 18.11 at 1600 and 33.47 -> 28.21 at 20951; 16 rows no worse
    /// (53.97 -> 53.38, 58.99 -> 57.89, 134.23 -> 122.71). Up to ~2.1K positions C 256 with 8 and 16 rows stays
    /// within +0.4 % of C 512 without the latent P·V; further on its opart reduction (chunks × rows) loses
    /// (measured on the M6).
    static func latentDecodeChunk(promptTokens n: Int) -> Int {
        n < 1536 ? 256 : n < 8192 ? 512 : 1024
    }

    /// Decode chunk of an attention layer: the latent policy for MLA layers, `decodeChunk(promptTokens:)` otherwise.
    /// A latent chunk grows (×2) when the longest generation the cache allows would pass the opart limit of
    /// `AttentionLayer.encode` (chunks × rows ≤ 4 · maxSplit with 16 rows).
    static func decodeChunk(promptTokens n: Int, latent: Bool, maxCtx: Int) -> Int {
        guard latent else { return decodeChunk(promptTokens: n) }
        var c = latentDecodeChunk(promptTokens: n)
        let rows = AttentionLayer.maxRows
        while (maxCtx + rows + c - 1) / c * rows > AttentionLayer.maxSplit * 4 { c *= 2 }
        return c
    }

    /// int8 KV cache (scale per token and head).
    public init(
        ctx: MetalContext, store: SafetensorsStore, config cfg: ModelConfig, maxCtx: Int, chunkRows: Int = 2048,
        options: EngineOptions = EngineOptions(), progress: ((Int) -> Void)? = nil
    ) throws {
        let t0 = Date()
        self.ctx = ctx
        self.options = options
        rowsConfig = try SpecRowsConfig.load(modelDirectory: store.directory, config: cfg)
        self.cfg = cfg
        let sp = try cfg.requiredSpec
        spec = sp
        self.maxCtx = maxCtx
        self.chunkRows = chunkRows
        let d = ctx.device
        eps = Float(cfg.rmsNormEps)
        hidden = cfg.hiddenSize
        precondition(ctx.dims == ModelDims(cfg), "MetalContext created with other dimensions")
        var ls: [Layer] = []
        var firstAttn: AttentionLayer?
        // Full attention layers of their own shape (Gemma 4: heads of 512): a second context on the same queue, and
        // their own shared work buffers (sized for that shape).
        let fullCtx = try sp.fullShape.map { try MetalContext(sharing: ctx, dims: $0) }
        self.fullCtx = fullCtx
        var firstFull: AttentionLayer?
        let flash = sp.family == .flashNext
        // LFM2 has no GDN; only LFM2 has short convolution. Nemotron-H's Mamba2 layers are GDN layers (Kind.mamba2),
        // without the chunked prefill.
        let hasDelta = cfg.layerTypes.contains("linear_attention"), hasConv = cfg.layerTypes.contains("conv")
        // KV that grows on demand (Flash-Next; options.kvGrow): the KV and the QSA keys are reserved for one
        // span and reallocated when growing (KVPool).
        let kvp = flash && options.kvGrow ? KVPool(ctx: ctx, maxPositions: maxCtx, allowance: options.kvAllowance) : nil
        kvPool = kvp
        // Flash-Next: experts always from the SSD (34 GB); without --expert-cache-gb, an 8 GB cache.
        let fparts =
            flash
            ? try FlashParts(
                ctx: ctx, store: store, config: cfg, chunkRows: chunkRows, layers: cfg.numLayers,
                pleCacheBytes: options.pleCacheBytes, maxCtx: maxCtx, pool: kvp) : nil
        fn = fparts
        let stream = try Self.makeExpertStream(ctx: ctx, store: store, config: cfg, options: options, flash: fparts)
        expertStream = stream
        let dummy = ctx.buffer(length: 16)
        for i in 0..<cfg.numLayers {
            let pre = try store.layerPrefix(i)
            let mixer: Mixer
            if flash {
                // No layer norms (the hyper-connections) nor Qwen3.6 MoE (FlashParts).
                if cfg.layerTypes[i] == "full_attention" {
                    let at = try Self.flashAttention(
                        ctx: ctx, store: store, config: cfg, layer: i, maxCtx: maxCtx, shared: firstAttn, pool: kvp)
                    firstAttn = firstAttn ?? at
                    mixer = .attn(at)
                } else {
                    mixer = .gdn(try GDNLayer(ctx: ctx, store: store, config: cfg, layer: i))
                }
                ls.append(Layer(n1: dummy, n2: dummy, mixer: mixer, moeOpt: nil, moeW: [], dense: nil))
                progress?(i)
                continue
            }
            try Self.checkBlockTensors(store, prefix: pre, spec: sp)
            if cfg.layerTypes[i] == "conv" {
                mixer = .conv(try ShortConvLayer(ctx: ctx, store: store, config: cfg, layer: i))
            } else if cfg.layerTypes[i] == "ffn" {
                mixer = .none
            } else if ModelSpec.attentionLayerTypes.contains(cfg.layerTypes[i]) {
                let own = fullCtx != nil && cfg.layerTypes[i] == "full_attention"
                // Gemma 4 E2B/E4B: the last layers read the KV of an earlier one (ModelSpec.kvSource).
                let a = try AttentionLayer(
                    ctx: own ? fullCtx! : ctx, store: store, config: cfg, layer: i, maxCtx: maxCtx,
                    shared: own ? firstFull : firstAttn, ringRows: chunkRows,
                    kvSource: sp.kvSource[i].map { Self.attention(ls[$0]) })
                a.fixedChunk = Self.decodeChunk(promptTokens: 0, latent: a.mla != nil, maxCtx: maxCtx)
                if own { firstFull = firstFull ?? a } else { firstAttn = firstAttn ?? a }
                mixer = .attn(a)
            } else {
                mixer = .gdn(try GDNLayer(ctx: ctx, store: store, config: cfg, layer: i))
            }
            let (moe, w, dense, parallel) = try Self.ffnParts(ctx, store, cfg, layer: i, prefix: pre, stream: stream)
            let blk = sp.block
            ls.append(
                Layer(
                    n1: try Self.normWeight(ctx, store, pre + sp.names.inputNorm, blk.mixerNorm),
                    n2: try Self.normWeight(ctx, store, pre + sp.names.postNorm, blk.ffnNorm),
                    mixer: mixer, moeOpt: moe, moeW: w, dense: dense,
                    post: blk.needsBranchEpilogue ? try Self.layerPost(ctx, store, prefix: pre, spec: sp) : nil,
                    parallel: parallel))
            progress?(i)
        }
        layers = ls
        let root = String(try store.layerPrefix(0).dropLast("layers.0.".count))  // "language_model.model."
        finalNorm = flash ? dummy : try Self.normWeight(ctx, store, root + sp.names.finalNorm, sp.block.finalNorm)
        embedNormW = try sp.block.embedNorm.map { n in
            try Self.normWeight(ctx, store, sp.names.embedNorm.map { root + $0 }, n)
        }
        ple = sp.pleDim > 0 ? try GemmaPLE(ctx: ctx, store: store, config: cfg, root: root, rows: chunkRows) : nil
        let embName = root + sp.names.embedding
        embed = try Self.embeddingTable(store, cfg, embName, d)
        // LFM2.5: tied embedding (6 bits) as lm_head.
        let lmName = Self.lmHeadName(root: root)
        let tied =
            store.tensors[lmName + ".weight"] == nil
            && (cfg.raw["tie_word_embeddings"] as? Bool ?? (sp.family == .gemma4))
        lmHead = tied ? embed : try DenseQ(store: store, config: cfg, paths: [lmName], device: d)
        // The real vocabulary (rows of the stored table): what the logits, the argmax and the sampling see.
        vocab = try store.info((tied ? embName : lmName) + ".weight").shape[0]
        // The lm_head (qmv and prefill GEMM instances) needs N % 64 = 0; every tier-1 vocabulary is a multiple of 64
        // (ModelSpec checks vocab_size; this checks the stored table).
        guard vocab % 64 == 0 else { throw SafetensorsError.format(ModelSpec.vocabGap(vocab)) }
        // 8 bits: embed_rows_b8 and vnax_qmv8 in the head.
        guard [2, 4, 6, 8, 16].contains(embed.bits), [2, 4, 6, 8].contains(lmHead.bits), embed.K == hidden else {
            throw SafetensorsError.format("embed/lm_head not supported")
        }
        // Without generation_config.json, those of Qwen3.5/3.6 (the toys rely on them).
        if let eos = Self.generationEOS(store.directory) {
            eosSet = eos
        }
        vqHead = VQMV(ctx)
        let R = 16
        tok = ctx.buffer(length: R * 4)
        h = ctx.buffer(length: R * hidden * 4)
        xn = ctx.buffer(length: R * hidden * 2)
        logits = ctx.buffer(length: R * vocab * 4)
        top1 = ctx.buffer(length: R * 4)
        let C = chunkRows
        let needsBranch = ls.contains { $0.post != nil }
        branch = needsBranch ? ctx.zeroedBuffer(length: R * hidden * 4) : nil
        pbranch = needsBranch ? ctx.zeroedBuffer(length: C * hidden * 4) : nil
        let needsPart = ls.contains { $0.parallel != nil }
        branchPart = needsPart ? ctx.zeroedBuffer(length: R * hidden * 4) : nil
        pbranchPart = needsPart ? ctx.zeroedBuffer(length: C * hidden * 4) : nil
        ptok = ctx.buffer(length: (maxCtx + C) * 4)  // each chunk in its own zone (enqueued without waiting)
        ph = ctx.buffer(length: C * hidden * 4)
        phPool = [ph]
        pxn = ctx.buffer(length: C * hidden * 2)
        let VD = ctx.dims.gdnVDim, HV = ctx.dims.gdnHV, QK = ctx.dims.gdnHK * ctx.dims.gdnDK
        // Prefill GDN buffers (LFM2 has no GDN: minimal) and short-convolution buffers (LFM2 only).
        let G = hasDelta || cfg.layerTypes.contains("mamba") ? C : 0
        gs = (
            ctx.buffer(length: G * (ctx.dims.gdnProjN + 63) / 64 * 64 * 4),
            ctx.buffer(length: (3 + G) * ctx.dims.gdnConvDim * 4),
            ctx.buffer(length: G * QK * 4), ctx.buffer(length: G * QK * 4), ctx.buffer(length: G * VD * 4),
            ctx.buffer(length: G * HV * 4), ctx.buffer(length: G * HV * 4), ctx.buffer(length: G * VD * 4),
            ctx.buffer(length: G * VD * 2)
        )
        scs =
            hasConv
            ? (ctx.buffer(length: C * 3 * hidden * 4), ctx.buffer(length: C * hidden * 2))
            : (ctx.buffer(length: 16), ctx.buffer(length: 16))
        ats = Self.prefillAttentionBuffers(ctx, spec: sp, rows: C)
        gdnChunk =
            hasDelta ? GDNChunk(ctx: ctx, maxRows: C) : nil
        if let first = ls.first(where: { $0.moeOpt != nil }) {
            let pm = try PrefillMoE(ctx: ctx, weights: first.moeW, config: cfg, rows: C)
            pm.x = pxn
            pm.accumulateInto = ph
            pm.setBiases(first.moe)
            pmoe = pm
        } else {
            pmoe = nil
        }
        if let I = ls.compactMap({ $0.dense?.inter }).max() {
            pff = (ctx.buffer(length: C * 2 * I * 2), ctx.buffer(length: C * I * 2))
        } else {
            pff = nil
        }
        loadSeconds = Date().timeIntervalSince(t0)
    }

    /// FFN of layer `i` by its FFNKind: the MoE (with its weights for PrefillMoE), the dense MLP, both (Gemma 4 26B-A4B,
    /// with the norms of the parallel FFN), or none (FFNKind.none).
    static func ffnParts(
        _ ctx: MetalContext, _ store: SafetensorsStore, _ cfg: ModelConfig, layer i: Int, prefix pre: String,
        stream: ExpertStream?
    ) throws -> (MoELayer?, [DeviceQuantizedLinear], DenseMLP?, ParallelNorms?) {
        let sp = try cfg.requiredSpec, kind = sp.ffnKinds[i]
        var moe: MoELayer?, w: [DeviceQuantizedLinear] = []
        if kind == .moe || kind == .denseAndMoE {
            let m = try MoELayer(ctx: ctx, store: store, config: cfg, layer: i, stream: stream)
            m.vq.nr = 8  // same router instance for any B
            w = [m.router, m.sharedGate, m.sGate, m.sUp, m.sDown].map { $0.asDQL() } + [m.eGate, m.eUp, m.eDown]
            moe = m
        }
        let dense =
            kind == .dense || kind == .denseAndMoE ? try DenseMLP(ctx: ctx, store: store, config: cfg, layer: i) : nil
        let n = sp.block.ffnNorm
        let parallel =
            kind == .denseAndMoE
            ? ParallelNorms(
                moePre: try normWeight(ctx, store, pre + sp.names.moePreNorm, n),
                densePost: try normWeight(ctx, store, pre + sp.names.densePostNorm, n),
                moePost: try normWeight(ctx, store, pre + sp.names.moePostNorm, n)) : nil
        return (moe, w, dense, parallel)
    }

    /// The attention layer of `l` (a KV source: ModelSpec guarantees the type).
    static func attention(_ l: Layer) -> AttentionLayer {
        guard case .attn(let a) = l.mixer else { preconditionFailure("KV source of another type than attention") }
        return a
    }

    /// Prefill attention buffers (projection, q, output) of `rows` rows for the larger of the model's attention shapes
    /// (q: query heads × head per row).
    static func prefillAttentionBuffers(_ ctx: MetalContext, spec sp: ModelSpec, rows C: Int) -> (
        proj: MTLBuffer, q: MTLBuffer, out: MTLBuffer
    ) {
        let shapes = [ctx.dims] + (sp.fullShape.map { [$0] } ?? [])
        return (
            ctx.buffer(length: C * shapes.map(\.attnProjN).max()! * 4),
            ctx.buffer(length: C * shapes.map { $0.attnQH * $0.attnHD }.max()! * 2),
            ctx.buffer(length: C * shapes.map(\.attnOutDim).max()! * 2)
        )
    }

    /// ExpertStream of the experts on the SSD (Flash-Next always, 8 GB cache without --expert-cache-gb; the MoEs
    /// with --expert-cache-gb), with the read options of `options`.
    private static func makeExpertStream(
        ctx: MetalContext, store: SafetensorsStore, config cfg: ModelConfig, options: EngineOptions,
        flash fparts: FlashParts?
    ) throws -> ExpertStream? {
        return try (fparts != nil ? (options.expertCacheBytes ?? 8_000_000_000) : options.expertCacheBytes).map {
            if let fp = fparts {
                return try ExpertStream(
                    ctx: ctx, directory: store.directory, flashNext: fp.fmt, topk: fp.K, bytes: $0,
                    prefetchK: options.expertPrefetch ?? 6, predStride: fp.routerN)
            }
            // ExpertStream reads the experts of all layers: with dense layers in the MoE, not yet.
            guard try !cfg.requiredSpec.ffnKinds.contains(where: { $0 == .dense || $0 == .denseAndMoE }) else {
                throw SafetensorsError.format("--expert-cache-gb: not yet with dense layers in the MoE")
            }
            return try ExpertStream(
                ctx: ctx, store: store, config: cfg, bytes: $0, prefetchK: options.expertPrefetch ?? 4)
        }.map { s in
            s.wideK = fparts != nil ? options.expertWide ?? 10 : 0
            s.skipRank = fparts != nil ? options.expertSkipRank : nil
            return s
        }
    }

    /// Flash-Next attention layer: no layer norms (the hyper-connections), q and k norms as 1 + w (fp32).
    private static func flashAttention(
        ctx: MetalContext, store: SafetensorsStore, config cfg: ModelConfig, layer i: Int, maxCtx: Int,
        shared: AttentionLayer?, pool: KVPool?
    ) throws -> AttentionLayer {
        let sp = try cfg.requiredSpec
        let rope = sp.ropes[i] ?? sp.rope
        let a = try store.layerPrefix(i) + "self_attn."
        func onePlus(_ n: String) throws -> MTLBuffer { try onePlusF32Tensor(ctx, store, n) }
        let at = try AttentionLayer(
            ctx: ctx, layer: i, maxCtx: maxCtx,
            proj: try DenseQ(
                store: store, config: cfg, paths: ["q_proj", "k_proj", "v_proj"].map { a + $0 }, device: ctx.device),
            oProj: try DenseQ(store: store, config: cfg, paths: [a + "o_proj"], device: ctx.device),
            qNorm: try onePlus(a + "q_norm.weight"), kNorm: try onePlus(a + "k_norm.weight"),
            eps: Float(cfg.rmsNormEps),
            theta: rope.theta, nope: rope.nope, shared: shared, pool: pool)
        at.fixedChunk = Self.decodeChunk(promptTokens: 0)
        return at
    }

    /// The embedding table `name`: quantized, or bf16 copied as it is (the 27B MLX-Serve pack leaves it in bf16, 2.5 GB).
    static func embeddingTable(_ store: SafetensorsStore, _ cfg: ModelConfig, _ name: String, _ d: MTLDevice) throws
        -> DenseQ
    {
        store.tensors[name + ".scales"] != nil
            ? try DenseQ(store: store, config: cfg, paths: [name], device: d)
            : try DenseQ(bf16Table: store, name: name + ".weight", device: d)
    }

    /// The lm_head next to the model root: `lm_head` beside `model.` (`language_model.lm_head` for
    /// `language_model.model.`), and at the top of the file for any other root (Nemotron-H's `backbone.`; before, the
    /// root lost its last 6 characters whatever they were).
    static func lmHeadName(root: String) -> String {
        (root.hasSuffix("model.") ? String(root.dropLast("model.".count)) : "") + "lm_head"
    }

    /// Weight of an RMSNorm with its convention: the tensor as is (`plain`, the path of before), `1 + w` folded
    /// at load time (bf16 rounded to nearest even, the value mlx-lm computes with `1.0 + weight` in bf16; fp32 with
    /// fp32 small weights) or ones (`unit`; also when `name` is nil). LAB_NWT type.
    static func normWeight(_ ctx: MetalContext, _ store: SafetensorsStore, _ name: String?, _ n: NormSpec) throws
        -> MTLBuffer
    {
        let d = ctx.dims.hidden, f32 = ctx.dims.normF32
        switch (n.weight, name) {
        case (.plain, let name?):
            return try plainNormTensor(store, name, ctx.device, f32: f32)
        case (.onePlus, let name?):
            return try onePlusNormTensor(ctx, store, name)
        case (.plain, nil), (.onePlus, nil):
            throw SafetensorsError.format("norm with weight but no weight name in this family")
        case (.unit, _):
            return f32 ? ctx.buffer([Float](repeating: 1, count: d)) : ctx.buffer([UInt16](repeating: 0x3F80, count: d))
        }
    }

    /// Post-norm weights, o_proj bias and layer scalar of layer `pre` (only for layers with a branch epilogue).
    static func layerPost(_ ctx: MetalContext, _ store: SafetensorsStore, prefix pre: String, spec sp: ModelSpec)
        throws -> LayerPost
    {
        let b = sp.block
        var scalar: Float = 1
        if b.layerScalar {
            let t = try store.info(pre + "layer_scalar")
            let v: [Float] =
                t.dtype == .f32 ? store.array(t, as: Float.self) : store.array(t, as: UInt16.self).map(bf16ToFloat)
            guard v.count == 1 else { throw SafetensorsError.format("\(pre)layer_scalar: expected one value") }
            scalar = v[0]
        }
        return LayerPost(
            mixerNorm: try b.mixerPostNorm.map {
                try normWeight(ctx, store, sp.names.mixerPostNorm.map { pre + $0 }, $0)
            },
            ffnNorm: try b.ffnPostNorm.map { try normWeight(ctx, store, sp.names.ffnPostNorm.map { pre + $0 }, $0) },
            oBias: b.oBias ? try floatTensor(store, pre + sp.names.attn + sp.names.oProj + ".bias", ctx.device) : nil,
            scalar: scalar)
    }

    /// Tensors of a layer that the block spec does not account for (biases of the attention projections, a separate
    /// attention gate or a `layer_scalar`) are a load error, not weights ignored silently (Qwen2-style checkpoints carry q/k/v biases
    /// without declaring them in the config).
    static func checkBlockTensors(_ store: SafetensorsStore, prefix pre: String, spec sp: ModelSpec) throws {
        let b = sp.block, a = sp.names.attn
        var extra: [String] = []
        for p in ["q_proj", "k_proj", "v_proj"] where !b.qkvBias && store.tensors[pre + a + p + ".bias"] != nil {
            extra.append(a + p + ".bias")
        }
        if !b.oBias && store.tensors[pre + a + sp.names.oProj + ".bias"] != nil {
            extra.append(a + sp.names.oProj + ".bias")
        }
        if !b.layerScalar && store.tensors[pre + "layer_scalar"] != nil { extra.append("layer_scalar") }
        // A separate attention gate (Muse Glimmer's gate_proj) only loads with a family that declares it (WeightNames.attnGate).
        if sp.names.attnGate == nil && store.tensors[pre + a + "gate_proj.weight"] != nil {
            extra.append(a + "gate_proj")
        }
        guard extra.isEmpty else {
            throw SafetensorsError.format(
                "\(pre): tensors not declared in config.json: \(extra.joined(separator: ", ")) (not supported)")
        }
    }

    /// End tokens of generation_config.json (nil if missing or without them).
    private static func generationEOS(_ directory: URL) -> Set<UInt32>? {
        guard
            let gc = try? JSONSerialization.jsonObject(
                with: Data(contentsOf: directory.appendingPathComponent("generation_config.json"))) as? [String: Any]
        else { return nil }
        return eosIds(gc)
    }

    /// `eos_token_id` (one or a list) of a config JSON.
    private static func eosIds(_ json: [String: Any]) -> Set<UInt32>? {
        if let a = json["eos_token_id"] as? [NSNumber] { return Set(a.map { UInt32($0.intValue) }) }
        if let x = json["eos_token_id"] as? NSNumber { return [UInt32(x.intValue)] }
        return nil
    }

    public var weightBytes: Int {
        // A tied lm_head is the embedding itself (same buffers): counted once.
        var n = embed.bytes + (lmHead.w === embed.w ? 0 : lmHead.bytes)
        for l in layers {
            switch l.mixer {
            case .gdn(let g): n += g.weightBytes
            case .attn(let a): n += (a.mla?.bytes ?? a.proj.bytes) + a.oProj.bytes
            case .conv(let c): n += c.weightBytes
            case .none: break
            }
            if let m = l.moeOpt { n += m.fixedBytes + (expertStream == nil ? m.experts * m.expertBytes : 0) }
            if let f = l.dense { n += f.bytes }
        }
        return n + (expertStream?.arenaBytes ?? 0) + (fn?.weightBytes ?? 0) + (ple?.weightBytes ?? 0)
    }

    // MARK: - Common pieces

    /// `eps`: the norm's own epsilon (BlockSpec; nil = the model's).
    public func norm(
        _ e: MTLComputeCommandEncoder, _ src: MTLBuffer, _ w: MTLBuffer, _ dst: MTLBuffer, rows: Int,
        rowOffset: Int = 0,
        eps normEps: Float? = nil
    ) throws {
        var p = (UInt32(hidden), normEps ?? eps, UInt32(hidden), UInt32(hidden))
        e.setComputePipelineState(try ctx.pipeline("rmsnorm_rows"))
        e.setBuffer(src, offset: rowOffset * hidden * 4, index: 0)
        e.setBuffer(w, offset: 0, index: 1)
        e.setBuffer(dst, offset: 0, index: 2)
        e.setBytes(&p, length: 16, index: 3)
        e.dispatchThreadgroups(
            MTLSize(width: rows, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: normThreads, height: 1, depth: 1))
    }
    /// Threads per RMSNorm row: 1024 with hidden 5120 (27B: 129 norms per round, 1.10 → 0.72 ms); 256 with 2048.
    lazy var normThreads = hidden >= 4096 ? 1024 : 256

    public func embedRows(
        _ e: MTLComputeCommandEncoder, tok t: MTLBuffer, tokOffset: Int = 0, out: MTLBuffer, rows: Int
    ) throws {
        try embed.encodeEmbed(ctx, e, tok: t, tokOffset: tokOffset, outF: out, rows: rows)
    }

    /// `width`: width of a row of `caps` (DFlash2Drafter.ctxWidth of the drafter that reads them).
    func capture(
        _ e: MTLComputeCommandEncoder, _ src: MTLBuffer, into caps: MTLBuffer, width: Int, slot j: Int, rows: Int
    ) throws {
        var p = (UInt32(hidden), UInt32(hidden), UInt32(width), UInt32(0))
        e.setComputePipelineState(try ctx.pipeline("capture_rows"))
        e.setBuffer(src, offset: 0, index: 0)
        e.setBuffer(caps, offset: j * hidden * 2, index: 1)
        e.setBytes(&p, length: 16, index: 2)
        e.dispatchThreadgroups(
            MTLSize(width: rows, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    func argmax(_ e: MTLComputeCommandEncoder, rows: Int) throws {
        var n = UInt32(vocab)
        e.setComputePipelineState(try ctx.pipeline("argmax_rows"))
        e.setBuffer(logits, offset: 0, index: 0)
        e.setBuffer(top1, offset: 0, index: 1)
        e.setBytes(&n, length: 4, index: 2)
        e.dispatchThreadgroups(
            MTLSize(width: rows, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
    }

    // MARK: - Prefill and capture state

    /// Block the last chained generation went back to (ingot-dev reuse diagnostics).
    public internal(set) var lastRewind = ""
    /// Temperature sampling buffers (Sampling.swift).
    lazy var sampleBufs = SampleBuffers(ctx)

    /// Scales and buffers of the W8A8 prefill (options.prefillInt8), created in the first chunk that uses it.
    var denseI8: DenseI8Scratch?
    /// KV that grows on demand (Flash-Next in `ingot serve`); nil = the whole max-context KV reserved at load.
    public let kvPool: KVPool?
    /// Prefill chunks that ran in W8A8 and with Q·Kᵀ in int8 (to check in the log that the option applies).
    public internal(set) var prefillI8Chunks = 0, prefillQK8Chunks = 0

    /// When preparing a prompt, the span up to point (a) is read through the verification path (batches of ≤ 16 rows,
    /// the last one with the final span if they fit) instead of prefill if that gives at most this many batches,
    /// counting the final span's (Prompt.swift). Measured on a second turn over 3K tokens: a short prefill
    /// costs about 3 batches (Qwen3.6: batch of ≤ 16 rows ~30–35 ms, prefill of 20–50 rows ~110–125 ms; 27B: ~110 vs
    /// ~340 ms); with 3 batches the MoE still wins (101–115 vs 122 ms) and the dense no longer (350 vs 337 ms).
    /// With the checkpoint inside the pass (no separate pass for the final span nor experts of the padding rows),
    /// a prefill of 17–48 rows already costs less than 2–3 batches (second turn over 3K, Heretic: 20 new tokens
    /// 71 → 62 ms, 32 → 106 → 74 ms, 40 → 120 → 79 ms): in the MoE only what fits in one goes by batches.
    /// Never from a start aligned to 64 past 0 (prefix-cache entries): that span goes through prefill, as without cache.
    public lazy var shortPassesMax = isMoE ? (canCaptureInPass ? 1 : 3) : 2

    var atq8: (q: MTLBuffer, s: MTLBuffer)?

    /// GDN state captures inside the next prefill chunk: rows (indices within the chunk) and, per
    /// capture, the buffers (state, conv history) of each GDN layer. Set and cleared by `prefillCapturing`.
    var pendingCapture: (rows: [Int], targets: [[(state: MTLBuffer, hist: MTLBuffer)]])?
    /// Destinations for the `preparePrompt` captures (up to 2; set by the engine with its free snapshots). nil =
    /// no captures: the final span goes through the verification path as before.
    public var captureTargets: [[(state: MTLBuffer, hist: MTLBuffer)]]?
    /// In Flash-Next, also, per capture, the window buffers (`StateSnapshot.conv`: the PLE one and the QSA ones in
    /// layer order). Set by the engine together with `captureTargets`.
    public var captureConvTargets: [[MTLBuffer]]?
    /// Per capture, the drafter ring (`StateSnapshot.draft`); only used by the cut captures.
    public var captureDraftTargets: [[(kc: MTLBuffer, vt: MTLBuffer)]]?
    var pendingConvCapture: [[MTLBuffer]]?
    /// The drafter ring after the first `rows` rows of the last chunk, one per prefix-cache cut captured as one more
    /// row.
    var pendingRingCapture: [(rows: Int, targets: [(kc: MTLBuffer, vt: MTLBuffer)])] = []

    /// Real rows of the last prefill chunk (their final states stay in `ph`).
    public internal(set) var lastChunkRows = 0

    // MARK: - Context

    /// Positions the KV admits now: the max context or, with KVPool, its current capacity.
    public var ctxLimit: Int { kvPool.map { min(maxCtx, $0.capacity) } ?? maxCtx }

    /// With KVPool: KV for `n` positions (up to maxCtx), reallocating if needed. With the GPU stopped. Returns false
    /// if `n` exceeds the max context. Without KVPool it does nothing.
    @discardableResult
    public func ensureContext(_ n: Int) throws -> Bool {
        guard let p = kvPool else { return true }
        return p.ensure(positions: min(maxCtx, n))
    }

    /// With KVPool: trims the KV to `n` positions if at least two spans are spare. With the GPU stopped and only if
    /// nothing live uses those positions.
    public func trimContext(_ n: Int) throws {
        kvPool?.trim(positions: n)
    }

    public func reset() {
        P = 0; nacc = 0
        for l in layers {
            switch l.mixer {
            case .gdn(let g):
                g.cur = 0
                memset(g.state[0].contents(), 0, g.state[0].length)
                memset(g.hist[0].contents(), 0, g.hist[0].length)
            case .attn(let a): a.pos = 0
            case .conv(let c): c.reset()
            case .none: break
            }
        }
        fn?.ple.reset()
        for q in fn?.qsaList ?? [] { q.reset() }
    }

    /// Decode attention key chunk by total prompt length (fixed during a generation; the latent MLA layers follow
    /// `latentDecodeChunk(promptTokens:)`).
    public func setDecodeChunk(promptTokens n: Int) {
        for l in layers {
            if case .attn(let a) = l.mixer {
                a.fixedChunk = Self.decodeChunk(promptTokens: n, latent: a.mla != nil, maxCtx: maxCtx)
            }
        }
    }
}

/// "0-9,12" → {0…9, 12}.
public func parseLayerSet(_ s: String) -> Set<Int> {
    var out = Set<Int>()
    for part in s.split(separator: ",") {
        let ab = part.split(separator: "-").compactMap { Int($0) }
        if ab.count == 2 { out.formUnion(ab[0]...ab[1]) } else if let a = ab.first { out.insert(a) }
    }
    return out
}
