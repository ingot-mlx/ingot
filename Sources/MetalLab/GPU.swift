import Foundation
import Metal

/// Device, queue and kernel library (the `.metal` files of `Shaders/` are compiled at run time, concatenated
/// in alphabetical order).
public final class MetalContext: @unchecked Sendable {
    public let device: MTLDevice
    public let queue: MTLCommandQueue
    public let library: MTLLibrary
    private var pipelines: [String: MTLComputePipelineState] = [:]

    /// Model shape of this context (LAB_* macros of its library). The layers, the model and the drafter read it from
    /// here (or copy it on creation): there is no global shape, so two contexts of different shape coexist.
    public let dims: ModelDims

    /// `dims`: model dimensions that the kernels take as compile-time constants (LAB_* macros); by
    /// default, those of Qwen3.6-35B-A3B. `extraSources`: Metal code compiled after the engine kernels, in
    /// the same library (kernels of external benches). The compiled library is kept
    /// per (shape, extra sources): another context with the same shape reuses it without compiling.
    public init(dims: ModelDims = ModelDims(), extraSources: [String] = []) throws {
        self.dims = dims
        guard let d = MTLCreateSystemDefaultDevice(), let q = d.makeCommandQueue() else {
            throw SafetensorsError.io("no Metal device")
        }
        try MachineRequirements.verify(device: d)
        device = d
        queue = q
        library = try Self.library(device: d, dims: dims, extraSources: extraSources)
    }

    /// Another shape on the device and queue of `base` (its own library, compiled or reused per shape): the full
    /// attention layers of a model whose full and sliding layers have different heads (Gemma 4: 512 and 256). Its
    /// pipelines go in the same command buffers as the model's.
    public init(sharing base: MetalContext, dims: ModelDims) throws {
        self.dims = dims
        device = base.device
        queue = base.queue
        library = try Self.library(device: base.device, dims: dims, extraSources: [])
    }

    /// Libraries compiled per shape and extra sources (one per shape in the process; compiling costs ~1.5 s).
    private struct LibraryKey: Hashable {
        let dims: ModelDims
        let extra: [String]
    }
    /// One slot per key: its lock serializes the compilation of that key only, so two different keys compile at
    /// the same time and the same key compiles once (the other callers wait on the slot and reuse its library).
    private final class LibrarySlot: @unchecked Sendable {
        let lock = NSLock()
        var library: MTLLibrary?
        /// Successful compilations of this key (tests check that concurrent callers share one).
        var compiles = 0
    }
    /// Guarded by `slotsLock`, which is held only to find or create a slot, never while compiling.
    nonisolated(unsafe) private static var slots: [LibraryKey: LibrarySlot] = [:]
    private static let slotsLock = NSLock()

    private static func slot(_ key: LibraryKey) -> LibrarySlot {
        slotsLock.withLock {
            if let s = slots[key] { return s }
            let s = LibrarySlot()
            slots[key] = s
            return s
        }
    }

    private static func library(device d: MTLDevice, dims: ModelDims, extraSources: [String]) throws -> MTLLibrary {
        let s = slot(LibraryKey(dims: dims, extra: extraSources))
        s.lock.lock()
        defer { s.lock.unlock() }
        if let lib = s.library, lib.device.registryID == d.registryID { return lib }
        let lib = try compileLibrary(device: d, dims: dims, extraSources: extraSources)
        s.library = lib
        s.compiles += 1
        return lib
    }

    /// Times the library of this key has been compiled in the process (0 if never requested).
    static func libraryCompiles(dims: ModelDims, extraSources: [String] = []) -> Int {
        let key = LibraryKey(dims: dims, extra: extraSources)
        guard let s = slotsLock.withLock({ slots[key] }) else { return 0 }
        return s.lock.withLock { s.compiles }
    }

    /// Guards the per-context scratch buffer tables of the layers (`MoELayer.partBuffers`, `MoELayer.identityMaps`,
    /// `DenseMLP.work`, `DenseQ.hadScratch`): contexts of different tests are built at the same time.
    static let scratchLock = NSLock()

    private static func compileLibrary(device d: MTLDevice, dims: ModelDims, extraSources: [String]) throws
        -> MTLLibrary
    {
        guard let dir = Bundle.module.url(forResource: "Shaders", withExtension: nil) else {
            throw SafetensorsError.io("cannot find Shaders/ in the bundle")
        }
        let files = try FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "metal" }.sorted { $0.path < $1.path }
        let src = (try files.map { try String(contentsOf: $0, encoding: .utf8) } + extraSources).joined(separator: "\n")
        let opts = MTLCompileOptions()
        opts.mathMode = .fast
        opts.languageVersion = .version4_1
        opts.preprocessorMacros = dims.macros
        return try d.makeLibrary(source: src, options: opts)
    }

    public func pipeline(_ name: String) throws -> MTLComputePipelineState {
        if let p = pipelines[name] { return p }
        guard let f = library.makeFunction(name: name) else { throw SafetensorsError.missing("kernel \(name)") }
        let p = try device.makeComputePipelineState(function: f)
        pipelines[name] = p
        Self.usedLock.lock()
        Self.used.insert(name)
        Self.usedLock.unlock()
        return p
    }

    /// Kernels requested in this process (all contexts): `--pipelines FILE` of lab and lab-dev writes them to the file.
    public static var usedPipelines: Set<String> {
        usedLock.lock()
        defer { usedLock.unlock() }
        return used
    }
    nonisolated(unsafe) private static var used: Set<String> = []
    private static let usedLock = NSLock()

    public func buffer(length: Int) -> MTLBuffer {
        device.makeBuffer(length: max(length, 16), options: .storageModeShared)!
    }

    /// A buffer of `length` bytes (at least 16, as `buffer(length:)`) whose first `length` bytes are set to zero.
    func zeroedBuffer(length: Int) -> MTLBuffer {
        let b = buffer(length: length)
        memset(b.contents(), 0, length)
        return b
    }

    public func buffer<T>(_ a: [T]) -> MTLBuffer {
        a.withUnsafeBytes {
            device.makeBuffer(bytes: $0.baseAddress!, length: max($0.count, 16), options: .storageModeShared)!
        }
    }

    /// Runs a command buffer and returns (GPU ms, wall ms).
    @discardableResult
    public func run(concurrent: Bool = false, _ encode: (MTLComputeCommandEncoder) throws -> Void) throws -> (
        gpu: Double, wall: Double
    ) {
        guard let cb = queue.makeCommandBuffer(),
            let enc = concurrent
                ? cb.makeComputeCommandEncoder(dispatchType: .concurrent) : cb.makeComputeCommandEncoder()
        else {
            throw SafetensorsError.io("cannot create the command buffer")
        }
        let t0 = DispatchTime.now().uptimeNanoseconds
        try encode(enc)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        let t1 = DispatchTime.now().uptimeNanoseconds
        if let e = cb.error { throw e }
        return ((cb.gpuEndTime - cb.gpuStartTime) * 1e3, Double(t1 - t0) / 1e6)
    }
}

/// Shape of the model (ModelSpec.shape; alias ModelShape) and of its DFlash2 drafter that the kernels fix at compile
/// time (LAB_* macros, one library per shape; each MetalContext has its own). Default: Qwen3.6-35B-A3B (hidden 2048,
/// GDN 16 k × 32 v, attention 16 q × 2 KV); dense Qwen3.8-27B has hidden 5120, GDN 16 k × 48 v and attention
/// 24 q × 4 KV (GQA 6). The macro defaults are the earlier numbers, so the same values give the same code.
public struct ModelDims: Sendable, Hashable {
    public var hidden = 2048
    public var gdnHV = 32
    public var attnQH = 16
    public var attnKVH = 2
    /// Attention head dimension and gate: 256 with gate in Qwen3.5/3.6; 64 without gate in LFM2 (the kernels
    /// support 64, 128 and 256, with and without gate).
    public var attnHD = 256
    public var attnGate = true
    /// QK-norm (RMSNorm per q and k head, with or without weight) or none (BlockSpec.qkNorm; LAB_AT_QKN).
    public var attnQKN = true
    /// q|k|v projections with bias (BlockSpec.qkvBias): the prework adds it (LAB_AT_QKVB).
    var attnQKVB = false
    /// Small weights (norms, conv, dt_bias) in fp32 instead of bf16 (Bonsai 2: prism_hadamard_qwen35). A_log is always
    /// loaded in fp32.
    public var normF32 = false
    /// Taps of the short convolution (`conv_L_cache`: 3 in LFM2, 7 in LFM3): LAB_SC_L, passed only when it is not 3, so
    /// the LFM2 library does not change.
    public var convL = 3
    /// GDN norm gate sigmoid instead of SiLU (Flash-Next: `output_gate_type: sigmoid`).
    public var gdnSigmoid = false
    /// MoE of the generic kernels (zf_flashnext.metal): experts, top-k and intermediate (default: those of Qwen3.6).
    public var moeE = 256, moeK = 8, moeI = 512
    /// Shared expert of the MoE (LAB_MOE_SHARED in vmoe_route and moe_route_topk_cfg): 0 none, 1 gated, 2 ungated
    /// (ModelSpec.sharedKind).
    public var moeShared = 1
    /// Router of vmoe_route and moe_route_topk_cfg (ModelSpec.routerKeys): sigmoid scoring (LAB_MOE_SCORE),
    /// selection bias (LAB_MOE_BIAS), routed_scaling_factor (LAB_MOE_RSCALE) and renormalization epsilon
    /// (LAB_MOE_EPS). With Qwen's (softmax, no bias, 1, 0) the macros are not emitted and the library is unchanged.
    public var moeSigmoid = false, moeBias = false
    public var moeRScale: Float = 1, moeEps: Float = 0
    /// Flash-Next: hyper-connection streams, PLE heads (all and the 2-gram ones) and width of its embedding.
    public var hcS = 4, pleH = 16, pleHPN = 8, pleDim = 2560
    /// Dimensions rotated by the attention RoPE (head_dim · partial_rotary_factor; LFM2, the whole head).
    public var attnRot = 64
    /// GDN k heads and their dimension (the kernels assume 16 × 128: only checked).
    public var gdnHK = 16, gdnDK = 128
    /// GDN v dimension per head (`linear_value_head_dim`, 128: only checked; Nemotron-H's Mamba2 `mamba_head_dim`).
    public var gdnDV = 128
    /// The recurrent layers are Nemotron-H's Mamba2: v heads `mamba_num_heads`, k heads `n_groups`, k dimension
    /// `ssm_state_size` and v `mamba_head_dim`, with a single `dt` per head in the projection. The GDN kernels run it
    /// with LAB_SSM and LAB_GDN_DV (no delta term, D·x skip, gated group norm); without it the library is the previous
    /// one.
    var ssm = false
    /// There are 4-bit affine layers with groups of 32 (`quantization.group_size` global or per layer):
    /// instantiates their kernels (LAB_G32). With everything in groups of 64, the library is the previous one. MXFP4's
    /// fixed groups of 32 do not count: they are another format with their own kernels (`moeMXFP4`).
    var quantG32 = false
    /// MoE experts in MXFP4 (e2m1 codes, one e8m0 scale per group of 32; gpt-oss): instantiates the `_mxfp4` expert
    /// kernels (LAB_MXFP4). Without it, the library is the previous one.
    var moeMXFP4 = false
    /// There are sliding-window attention layers: instantiates the windowed attention kernels (LAB_AT_SWA).
    /// Without them, the library is the previous one.
    var attnSWA = false
    /// Learned attention sinks per query head (ModelSpec.attnSinks): the pattn_i8 prefill epilogue and vattn_reduce
    /// read them (LAB_AT_SINKS). Without them, the library is the previous one.
    var attnSinks = false
    /// Some attention layer has a scaled RoPE (ModelSpec.hasRopeTable): the prework kernels read a per-layer
    /// table (inv_freq + mscale, buffer 11) instead of `pow(theta, …)` (LAB_AT_ROPETAB). Without it, the library is the
    /// previous one.
    var attnRopeTab = false
    /// The gated FFN uses GELU-tanh instead of SiLU (BlockSpec.activation): compiles the `_gelu` instances
    /// (LAB_FFN_GELU) and the model picks them (`ffnActSuffix`). The drafter keeps the SiLU ones.
    var ffnGelu = false
    /// The MoE uses gpt-oss's swiglu_oai with this limit (BlockSpec.activation): compiles the `_oai` instances
    /// (LAB_FFN_OAI, LAB_FFN_LIMIT) and the model picks them (`ffnActSuffix`). nil: the library is the previous one.
    var ffnLimit: Float?
    /// The FFN is Nemotron-H's relu² without gate (BlockSpec.activation): compiles the `_relu2` instances
    /// (LAB_FFN_RELU2) and the model picks them (`ffnActSuffix`). Without it, the library is the previous one.
    var ffnRelu2 = false
    /// Router linear with bias and experts with biases (gpt-oss; RouterSpec.logitBias and expertBiases): LAB_MOE_LBIAS
    /// and LAB_MOE_EBIAS. MoE hidden a multiple of 64 but not of 128 (gpt-oss's 2880): LAB_MOE_H64. Without them, the
    /// library is the previous one.
    var moeLBias = false, moeEBias = false, moeH64 = false
    /// Per-expert scale of the router weights (Gemma 4; RouterSpec.perExpertScale): LAB_MOE_PES. Without it, the library
    /// is the previous one.
    var moePES = false
    /// Weightless per-head RMSNorm of V before it is stored (Gemma 4's v_norm; BlockSpec.vNorm): the prework kernels
    /// normalize the v heads like q and k (LAB_AT_VN). Without it, the library is the previous one.
    var attnVN = false
    public init() {}
    public init(_ cfg: ModelConfig) {
        hidden = cfg.hiddenSize
        gdnHV = (cfg.text["linear_num_value_heads"] as? NSNumber)?.intValue ?? 32
        attnQH = (cfg.text["num_attention_heads"] as? NSNumber)?.intValue ?? 16
        attnKVH = (cfg.text["num_key_value_heads"] as? NSNumber)?.intValue ?? 2
        if cfg.family == .lfm2 || cfg.family == .granite {
            // LFM2 and Granite (mlx-lm granite.py ignores head_dim): the head is hidden / heads, rotated whole.
            attnHD = hidden / attnQH
            attnGate = false
            attnRot = attnHD
        } else if cfg.family == .glm4MoeLite, let m = MLASpec(cfg.text) {
            // GLM-4.7-Flash (MLA, GLM.swift, Layers/MLA.swift), in the absorbed form over a latent cache: one KV head of
            // `[k_pe | c]` (qk_rope_head_dim + kv_lora_rank = 576) read by every query head (G = heads), the queries
            // `[q_pe | embed_q·q_nope]` per head (`num_key_value_heads` is not read, as in mlx-lm), and the RoPE on the
            // first `qk_rope_head_dim` (64) dimensions once the loader moves them there (MLASpec.qBRowOrder,
            // kvARowOrder), not on head_dim · partial_rotary_factor (mlx-lm ignores that key here). The attention scale
            // stays (nope + rope)^-0.5 (BlockSpec.attnScale).
            attnHD = m.latentHeadDim
            attnKVH = 1
            attnGate = false
            attnRot = m.qkRopeHeadDim
        } else {
            attnHD = (cfg.text["head_dim"] as? NSNumber)?.intValue ?? 256
            attnGate = ModelSpec.attnGated(cfg.family)
            // LFM3's partial_rotary_factor is a zero tail over the whole head (the proportional table), not a narrower
            // rotation.
            let partial =
                cfg.family == .lfm3
                ? 1
                : ((cfg.text["rope_parameters"] as? [String: Any])?["partial_rotary_factor"] as? NSNumber)?.doubleValue
                    ?? (cfg.text["partial_rotary_factor"] as? NSNumber)?.doubleValue
                    ?? ModelSpec.defaultPartialRotary(cfg.family)
            attnRot = Int((Double(attnHD) * partial).rounded())
        }
        gdnHK = (cfg.text["linear_num_key_heads"] as? NSNumber)?.intValue ?? 16
        gdnDK = (cfg.text["linear_key_head_dim"] as? NSNumber)?.intValue ?? 128
        gdnDV = (cfg.text["linear_value_head_dim"] as? NSNumber)?.intValue ?? 128
        if cfg.family == .nemotronH {
            func int(_ k: String) -> Int? { (cfg.text[k] as? NSNumber)?.intValue }
            ssm = true
            gdnHV = int("mamba_num_heads") ?? 0; gdnHK = int("n_groups") ?? 0
            gdnDK = int("ssm_state_size") ?? 0; gdnDV = int("mamba_head_dim") ?? 0
        }
        convL = (cfg.text["conv_L_cache"] as? NSNumber)?.intValue ?? 3
        // Bonsai 2 and Flash-Next (our format): norms, conv and dt_bias in fp32.
        normF32 = ModelFamily.smallWeightsF32(cfg)
        moeE = cfg.numExperts ?? 256; moeK = cfg.expertsPerToken ?? 8; moeI = cfg.moeIntermediateSize ?? 512
        let rk = ModelSpec.routerKeys(cfg.text, family: cfg.family)
        moeSigmoid = rk.scoring == .sigmoid; moeBias = rk.selectBias; moeRScale = rk.scale; moeEps = rk.normEps
        switch ModelSpec.sharedKind(cfg) {
        case .none: moeShared = 0
        case .gated: moeShared = 1
        case .ungated: moeShared = 2
        }
        attnSWA = cfg.layerTypes.contains("sliding_attention")
        attnSinks = ModelSpec.attnSinks(cfg)
        attnRopeTab = ModelSpec.hasScaledRope(cfg)
        let block = BlockSpec.read(
            text: cfg.text, family: cfg.family, eps: Float(cfg.rmsNormEps), headDim: attnHD, layers: cfg.numLayers
        ).spec
        ffnGelu = block.activation == .geluTanh
        if case .swigluOAI(let lim) = block.activation { ffnLimit = lim }
        ffnRelu2 = block.activation == .relu2
        moeLBias = cfg.isMoE && ModelSpec.moeBiases(cfg.family)
        moeEBias = moeLBias
        moeH64 = cfg.isMoE && hidden % 128 != 0
        moePES = cfg.isMoE && cfg.family == .gemma4
        attnQKN = block.qkNorm != nil
        attnVN = block.vNorm != nil
        attnQKVB = block.qkvBias
        let quants = [cfg.quantDefault].compactMap { $0 } + cfg.quantOverrides.values
        quantG32 = quants.contains { $0.groupSize == 32 && $0.mode == "affine" }
        moeMXFP4 = cfg.isMoE && quants.contains { $0.mode == "mxfp4" }
        gdnSigmoid = cfg.family == .flashNext && (cfg.text["output_gate_type"] as? String ?? "sigmoid") == "sigmoid"
        if cfg.family == .flashNext {
            func int(_ k: String) -> Int? { (cfg.text[k] as? NSNumber)?.intValue }
            hcS = int("hc_count") ?? 4
            pleHPN = int("heads_per_ngram") ?? 8
            pleH = pleHPN * ((int("ngram_size") ?? 3) - 1)
            pleDim = int("ple_embed_dim") ?? 2560
        }
    }
    /// Macros the kernels are compiled with (aa_dims.metal).
    public var macros: [String: NSObject] {
        var m: [String: NSObject] = [
            "LAB_HIDDEN": NSNumber(value: hidden), "LAB_GDN_HV": NSNumber(value: gdnHV),
            "LAB_AT_QH": NSNumber(value: attnQH), "LAB_AT_KVH": NSNumber(value: attnKVH),
            "LAB_AT_HD": NSNumber(value: attnHD), "LAB_AT_GATE": NSNumber(value: attnGate ? 1 : 0),
            "LAB_AT_QKN": NSNumber(value: attnQKN ? 1 : 0),
            "LAB_NWT": (normF32 ? "float" : "bfloat") as NSString, "LAB_GDN_SIG": NSNumber(value: gdnSigmoid ? 1 : 0),
            "LAB_MOE_E": NSNumber(value: moeE), "LAB_MOE_K": NSNumber(value: moeK), "LAB_MOE_I": NSNumber(value: moeI),
            "LAB_MOE_SHARED": NSNumber(value: moeShared),
            "LAB_AT_ROT": NSNumber(value: attnRot), "LAB_GDN_HK": NSNumber(value: gdnHK),
            "LAB_GDN_DK": NSNumber(value: gdnDK), "LAB_HC_S": NSNumber(value: hcS), "LAB_PLE_H": NSNumber(value: pleH),
            "LAB_PLE_HPN": NSNumber(value: pleHPN), "LAB_PLE_DIM": NSNumber(value: pleDim),
        ]
        if ssm {
            m["LAB_SSM"] = NSNumber(value: 1)
            m["LAB_GDN_DV"] = NSNumber(value: gdnDV)
        }
        if quantG32 { m["LAB_G32"] = NSNumber(value: 1) }
        if convL != 3 { m["LAB_SC_L"] = NSNumber(value: convL) }
        if attnSWA { m["LAB_AT_SWA"] = NSNumber(value: 1) }
        if attnSinks { m["LAB_AT_SINKS"] = NSNumber(value: 1) }
        if attnRopeTab { m["LAB_AT_ROPETAB"] = NSNumber(value: 1) }
        if ffnGelu { m["LAB_FFN_GELU"] = NSNumber(value: 1) }
        if ffnRelu2 { m["LAB_FFN_RELU2"] = NSNumber(value: 1) }
        if attnQKVB { m["LAB_AT_QKVB"] = NSNumber(value: 1) }
        if attnVN { m["LAB_AT_VN"] = NSNumber(value: 1) }
        if moeSigmoid { m["LAB_MOE_SCORE"] = NSNumber(value: 1) }
        if moeBias { m["LAB_MOE_BIAS"] = NSNumber(value: 1) }
        if moeRScale != 1 { m["LAB_MOE_RSCALE"] = "\(moeRScale)f" as NSString }
        if moeEps != 0 { m["LAB_MOE_EPS"] = "\(moeEps)f" as NSString }
        if let lim = ffnLimit {
            m["LAB_FFN_OAI"] = NSNumber(value: 1)
            m["LAB_FFN_LIMIT"] = "\(lim)f" as NSString
        }
        if moeLBias { m["LAB_MOE_LBIAS"] = NSNumber(value: 1) }
        if moeEBias { m["LAB_MOE_EBIAS"] = NSNumber(value: 1) }
        if moeH64 { m["LAB_MOE_H64"] = NSNumber(value: 1) }
        if moePES { m["LAB_MOE_PES"] = NSNumber(value: 1) }
        if moeMXFP4 { m["LAB_MXFP4"] = NSNumber(value: 1) }
        return m
    }

    /// Shape of the full attention layers when it differs from the sliding ones' (nil: one shape per model). Gemma 4:
    /// `global_head_dim` (512) and, with K = V (`attention_k_eq_v`), `num_global_key_value_heads`, as mlx-lm's
    /// gemma4_text (without K = V the full layers keep `num_key_value_heads`). The RoPE then rotates the whole head
    /// (the proportional table carries zero frequencies past its partial factor), and the rest of the macros are the
    /// model's.
    public static func fullAttention(_ cfg: ModelConfig) -> ModelDims? {
        let t = cfg.text
        guard cfg.layerTypes.contains("full_attention"), let ghd = (t["global_head_dim"] as? NSNumber)?.intValue
        else { return nil }
        let base = ModelDims(cfg)
        let keqv = (t["attention_k_eq_v"] as? Bool) ?? false
        let kv = keqv ? (t["num_global_key_value_heads"] as? NSNumber)?.intValue ?? base.attnKVH : base.attnKVH
        guard ghd != base.attnHD || kv != base.attnKVH else { return nil }
        var d = base
        d.attnHD = ghd
        d.attnKVH = kv
        d.attnRot = ghd
        return d
    }

    /// Suffix of the gated FFN kernel instances of the model's activation ("" = SiLU, "_gelu" = GELU-tanh, "_oai" =
    /// swiglu_oai, MoE only, "_relu2" = relu² without gate).
    var ffnActSuffix: String { ffnGelu ? "_gelu" : ffnLimit != nil ? "_oai" : ffnRelu2 ? "_relu2" : "" }

    /// Attention: queries per KV head, width of the q|gate|k|v projection and of the output.
    public var attnG: Int { attnQH / attnKVH }
    public var attnProjN: Int { attnQH * (attnGate ? 2 : 1) * attnHD + 2 * attnKVH * attnHD }
    public var attnOutDim: Int { attnQH * attnHD }
    /// GDN (16 × 128 k heads and v dimension 128): v width, conv channels (q | k | v) and outputs of
    /// qkv|z|b|a. Mamba2 (`ssm`): d_inner, conv channels (x | B | C) and outputs of z|xBC|dt (one dt per head).
    public var gdnVDim: Int { gdnHV * gdnDV }
    public var gdnConvDim: Int { 2 * gdnHK * gdnDK + gdnVDim }
    public var gdnProjN: Int { gdnConvDim + gdnVDim + (ssm ? 1 : 2) * gdnHV }
}

/// Read of a half buffer into [Float].
public func readHalf(_ b: MTLBuffer, count: Int, offset: Int = 0) -> [Float] {
    let p = (b.contents() + offset).bindMemory(to: Float16.self, capacity: count)
    return (0..<count).map { Float(p[$0]) }
}
