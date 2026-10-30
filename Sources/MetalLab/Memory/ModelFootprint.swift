import Foundation

/// What each model family takes on the GPU besides the weights and the KV cache: the fixed state (live recurrent
/// state, decode buffers, the part of the KV and drafter that does not scale) and the prefill work, which grows
/// with the chunk rows. Measured once per family with `currentAllocatedSize` (see `lab plan`); unmeasured
/// families carry an estimate and the plan says so.
public struct FamilyCalibration: Sendable, Equatable {
    /// Fixed bytes (no weights, KV, drafter or snapshots).
    public var fixedBytes: Int
    /// Bytes per row of the prefill chunk.
    public var perChunkRowBytes: Int
    public var measured: Bool
    public var source: String

    public init(fixedBytes: Int, perChunkRowBytes: Int, measured: Bool, source: String) {
        self.fixedBytes = fixedBytes
        self.perChunkRowBytes = perChunkRowBytes
        self.measured = measured
        self.source = source
    }

    /// The one for a family (FamilyTraits, ModelSpec.swift).
    public static func forFamily(_ f: ModelFamily) -> FamilyCalibration { f.traits.calibration }

    /// Qwen3.6-35B-A3B (Heretic, Ornith): measured on the M6 (`lab serve --prefix-disk 0 --snapshots 0
    /// --prefix-cache-gb 0 --max-ctx 32768`): without drafter and 2048 chunks, 20.84 GB = weights 19.51 + KV 0.34 +
    /// 0.99; with 1024 and 512 chunks (with drafter), 0.31 and 0.15 GB less -> ~300 KB per row and 0.38 GB fixed.
    public static let qwen36MoE = FamilyCalibration(
        fixedBytes: 380_000_000, perChunkRowBytes: 300_000, measured: true,
        source: "measured (lab serve, chunks of 2048/1024/512)")
    /// Qwen3.5/3.8 dense (27B): measured the same way: without drafter and 2048 chunks, 17.97 GB = weights 15.13 +
    /// KV 1.09 + 1.75; with 1024 chunks, 0.48 GB less -> ~470 KB per row and 0.79 GB fixed.
    public static let qwen35Dense = FamilyCalibration(
        fixedBytes: 790_000_000, perChunkRowBytes: 470_000, measured: true,
        source: "measured (lab serve, chunks of 2048/1024)")
    /// LFM2 / LFM2.5 (no drafter): unmeasured; like Qwen3.6 (same hidden 2048, MoE) with a 25 % margin.
    public static let lfm2 = FamilyCalibration(
        fixedBytes: 475_000_000, perChunkRowBytes: 375_000, measured: false,
        source: "estimated: Qwen3.6's (hidden 2048, MoE) + 25 %")
    /// Qwen3.8-Flash-Next (FlashParts): streams of 4 × 2560 fp32 per chunk of a prefill group (4) and MoE buffers
    /// of 10 experts per row (yE: 10 × 2560 fp32), PLE and prefill GDN/attention: ~800 KB per row; fixed: round
    /// buffers, PLE history and read rings. Estimated from the code, unmeasured.
    public static let flashNext = FamilyCalibration(
        fixedBytes: 700_000_000, perChunkRowBytes: 800_000, measured: false,
        source: "estimated from the code (FlashParts), unmeasured")
    /// gpt-oss (20b MXFP4-Q8): measured on the M6 like Qwen3.6 (`lab serve --prefix-disk 0 --snapshots 0
    /// --prefix-cache-gb 0 --max-ctx 32768`): with 2048 chunks, 13.06 GB = weights 12.06 + KV 0.47 + 0.53; with 1024
    /// chunks, 0.18 GB less -> ~180 KB per row and 0.18 GB fixed (at 131072, 14.34 GB, as predicted).
    public static let gptOss = FamilyCalibration(
        fixedBytes: 180_000_000, perChunkRowBytes: 180_000, measured: true,
        source: "measured (lab serve, chunks of 2048/1024)")
    /// Gemma 4: measured on the M6 like Qwen3.6 (`lab serve --prefix-disk 0 --snapshots 0 --prefix-cache-gb 0
    /// --max-ctx 32768`, chunks of 2048 and 1024; the sliding-window ring grows with the chunk and is taken out):
    /// 26B-A4B 15.57 / 15.26 GB = weights 14.20 + KV 0.34 + ring 0.33 / 0.23 + 0.70 / 0.49 -> ~205 KB per row and
    /// 0.28 GB fixed; 31B 21.30 / 20.54 GB = weights 17.27 + KV 1.35 + ring 1.33 / 0.91 + 1.35 / 1.01 -> ~332 KB per
    /// row and 0.67 GB fixed (at 131072 with 1024 chunks, 24.60 GB, as predicted). One value per family: the 31B's,
    /// the larger (it overestimates the 26B-A4B by ~0.65 GB with 2048 chunks).
    public static let gemma4 = FamilyCalibration(
        fixedBytes: 670_000_000, perChunkRowBytes: 332_000, measured: true,
        source: "measured (lab serve, 26B-A4B and 31B, chunks of 2048/1024; the 31B's)")
    /// Nemotron-H: measured on the M6 with the Lightning 30B-A3B 4bit like Qwen3.6 (`lab serve --prefix-disk 0
    /// --snapshots 0 --prefix-cache-gb 0 --max-ctx 32768`): with 2048 chunks, 18.64 GB = weights 17.77 + KV 0.10 + 0.77;
    /// with 1024 chunks, 0.24 GB less -> ~235 KB per row and 0.29 GB fixed (live Mamba2 state included; at 262144,
    /// 19.37 GB, as predicted). The Nano 4B (hybrid dense) is unmeasured and shares it.
    public static let nemotronH = FamilyCalibration(
        fixedBytes: 285_000_000, perChunkRowBytes: 235_000, measured: true,
        source: "measured (lab serve, Lightning 30B-A3B, chunks of 2048/1024)")
    /// GLM-4.7-Flash: measured on the M6 with the mlx-community 4bit like Qwen3.6 (`lab serve --prefix-disk 0
    /// --snapshots 0 --prefix-cache-gb 0 --max-ctx 32768`): with 2048 chunks, 19.70 GB = weights 16.85 + latent KV 1.79 +
    /// 1.06; with 1024 chunks, 0.22 GB less -> ~215 KB per row and 0.62 GB fixed (the attention's shared 189 MB opart,
    /// G 20 × head 576, included; at 131072 with 1024 chunks, 24.84 GB, as predicted). Plus, from the code, the
    /// decompressed prefill attention of long chunks (MLAProjection.decBuffers): 83 MB fixed (W' and one block of
    /// k_nope and v) and 20.6 KB per row (the running state of 20 heads × 256 in fp32).
    public static let glm4MoeLite = FamilyCalibration(
        fixedBytes: 703_000_000, perChunkRowBytes: 236_000, measured: true,
        source: "measured (lab serve, chunks of 2048/1024)")
    /// Muse Glimmer 30B: measured on the M6 with the mlx-community 4bit like Qwen3.6 (`lab serve --prefix-disk 0
    /// --snapshots 0 --prefix-cache-gb 0 --max-ctx 32768`): without drafter and 2048 chunks, 18.29 GB = weights 15.67 +
    /// KV 0.23 + ring 0.09 + 2.30; with the drafter (incoai DFlash2, 2.00 GB in the plan) 20.33 / 20.07 GB with 2048 /
    /// 1024 chunks (ring 0.09 / 0.07) -> ~235 KB per row; fixed 1.86 GB, taken with the drafter (the default) so that
    /// the plan matches it (it overestimates the run without drafter by ~0.04 GB). At 131072 with the drafter, 21.01 GB,
    /// as predicted.
    public static let museGlimmer = FamilyCalibration(
        fixedBytes: 1_860_000_000, perChunkRowBytes: 235_000, measured: true,
        source: "measured (lab serve, chunks of 2048/1024, with drafter)")
    /// LFM3: measured on the M6 with LiquidAI's LFM3-1.7B-MLX-8bit like Qwen3.6 (`lab serve --prefix-disk 0
    /// --snapshots 0 --prefix-cache-gb 0 --max-ctx 32768`): with 2048 chunks, 2.52 GB = weights 1.89 + KV 0.21 + ring
    /// 0.01 + 0.41; with 1024 chunks, 0.13 GB less -> ~127 KB per row (0.26 GB with 2048) and 0.15 GB fixed (conv
    /// history included). At 262144, 3.97 GB, as predicted.
    public static let lfm3 = FamilyCalibration(
        fixedBytes: 150_000_000, perChunkRowBytes: 127_000, measured: true,
        source: "measured (lab serve, LFM3-1.7B-MLX-8bit, chunks of 2048/1024)")
    /// Uncalibrated family: the most expensive of the known ones, with a 25 % margin.
    public static let unknown = FamilyCalibration(
        fixedBytes: 990_000_000, perChunkRowBytes: 590_000, measured: false,
        source: "uncalibrated family: the largest known + 25 %")
}

/// A model's GPU footprint, computed without loading the weights: `config.json` and the safetensors headers
/// (of the model and the drafter). All in bytes.
public struct ModelFootprint: Sendable, Equatable {
    public var name: String
    public var modelType: String
    /// Bytes of the tensors the engine loads (no MTP or towers: `SafetensorsStore.skippedParts`).
    public var weightBytes: Int
    /// The largest tensor (must fit in `maxBufferLength`).
    public var largestTensorBytes: Int
    /// int8 KV per token: attention layers × 2 × KV heads × (head dim + 4), plus 8 for the ids.
    public var kvBytesPerToken: Int
    /// KV of a single layer per token (one layer's buffer must fit in `maxBufferLength`).
    public var kvBytesPerTokenPerLayer: Int
    /// `max_position_embeddings`.
    public var nativeContext: Int
    /// Loaded drafter (weights and buffers); nil = the model has no drafter.
    public var drafterBytes: Int?
    /// Each snapshot: recurrent state (GDN or convolution) and, with a drafter, its context ring.
    public var snapshotBaseBytes: Int
    public var snapshotDrafterBytes: Int
    public var calibration: FamilyCalibration
    /// Experts read from the SSD (Flash-Next: always; experts.bin): the expert cache gets what is left of the
    /// budget (`fixedCacheBytes` pins it, --expert-cache-gb), with a minimum of `minCacheBytes` (one layer's
    /// slots and the prefill ring). `cpuCacheBytes`: the PLE table's row cache (CPU memory).
    public struct StreamedExperts: Sendable, Equatable {
        public var totalBytes: Int, recordBytes: Int, minCacheBytes: Int, cpuCacheBytes: Int
        public var fixedCacheBytes: Int?
        /// Flash-Next grouped prefill (`EngineOptions.flashPrefillGroup`, --expert-pf-group): each chunk over 4 (what
        /// the calibration counts) carries its streams (4 × hidden fp32 per row: 84 MB with 2048 chunks) and each
        /// chunk with a drafter over one, its captures (target layers × hidden half per row: 52 MB).
        public var prefillGroup = 8, streamBytesPerRow = 0, draftCapsBytesPerRow = 0
        /// The Qwen3.5/3.6 MoE's grouped prefill (`EngineOptions.expertPrefillGroup`): a fixed amount, counted with
        /// chunks of 2048 (the maximum), instead of Flash-Next's per-row one.
        public var fixedGroupBytes: Int?
        /// KV that grows on demand (KVPool): Flash-Next only; the others reserve the whole context's KV at load.
        public var onDemandKV = true
        /// Set by the planner when it moves the experts of a model that does not fit in memory to the SSD
        /// (`ModelFootprint.streamable`): the expert cache goes before a prefix cache over 0.25 GB and a context
        /// over 64K.
        public var automatic = false
        public func groupBytes(chunkRows: Int, drafter: Bool) -> Int {
            if let g = fixedGroupBytes { return g }
            return chunkRows
                * (max(0, prefillGroup - 4) * streamBytesPerRow
                    + (drafter ? (HybridModel.flashDraftParts - 1) * draftCapsBytesPerRow : 0))
        }
        public init(totalBytes: Int, recordBytes: Int, minCacheBytes: Int, cpuCacheBytes: Int) {
            self.totalBytes = totalBytes; self.recordBytes = recordBytes; self.minCacheBytes = minCacheBytes
            self.cpuCacheBytes = cpuCacheBytes
        }
    }
    public var streamed: StreamedExperts?
    /// Experts that can be read from the SSD (Qwen3.5/3.6 MoE in MLX safetensors, `ExpertStream`), counted in
    /// `weightBytes`: the planner moves them to the SSD (`streamed`) when the model does not fit in memory with the
    /// useful context and its drafter. nil = it cannot, or `--expert-cache-gb` already set it.
    public var streamable: StreamedExperts?
    /// Sliding-window layers with a ring KV: they do not grow with the context (not in `kvBytesPerToken`); each
    /// one holds `AttentionLayer.ringSize` slots, which depends on the prefill chunk.
    public var windowLayers = 0, window = 0
    /// Fixed KV of the sliding-window layers for prefill chunks of `chunkRows` rows (0 without them).
    public func kvFixedBytes(chunkRows: Int) -> Int {
        guard windowLayers > 0 else { return 0 }
        let ring = AttentionLayer.ringSize(window: window, rows: chunkRows, maxCtx: Int.max / 4)
        return windowLayers * kvBytesPerTokenPerLayer * ring
    }

    public init(
        name: String, modelType: String, weightBytes: Int, largestTensorBytes: Int, kvBytesPerToken: Int,
        kvBytesPerTokenPerLayer: Int, nativeContext: Int, drafterBytes: Int?, snapshotBaseBytes: Int,
        snapshotDrafterBytes: Int, calibration: FamilyCalibration
    ) {
        self.name = name
        self.modelType = modelType
        self.weightBytes = weightBytes
        self.largestTensorBytes = largestTensorBytes
        self.kvBytesPerToken = kvBytesPerToken
        self.kvBytesPerTokenPerLayer = kvBytesPerTokenPerLayer
        self.nativeContext = nativeContext
        self.drafterBytes = drafterBytes
        self.snapshotBaseBytes = snapshotBaseBytes
        self.snapshotDrafterBytes = snapshotDrafterBytes
        self.calibration = calibration
    }

    public var hasDrafter: Bool { drafterBytes != nil }

    public func prefillBytes(chunkRows: Int) -> Int {
        calibration.fixedBytes + calibration.perChunkRowBytes * chunkRows
    }
    public func snapshotBytes(drafter: Bool) -> Int { snapshotBaseBytes + (drafter ? snapshotDrafterBytes : 0) }

    /// From data already read (tests use this with synthetic configs).
    /// - Parameters:
    ///   - drafterConfig: the DFlash2 drafter's `config.json` (nil = no drafter).
    ///   - drafterWeightBytes: its weights as the engine loads them (`drafterWeightBytes(tensors:)`).
    public init(
        name: String, config cfg: ModelConfig, weightBytes: Int, largestTensorBytes: Int,
        drafterConfig: [String: Any]?, drafterWeightBytes: Int?, calibration: FamilyCalibration? = nil
    ) {
        let dims = ModelDims(cfg)
        let t = cfg.text
        func int(_ d: [String: Any], _ k: String) -> Int? { (d[k] as? NSNumber)?.intValue }
        let spec = try? ModelSpec(cfg)
        // Only the layers with their own KV (Gemma 4 E2B/E4B: the KV-sharing ones read another layer's).
        let own = cfg.layerTypes.indices.filter { i in spec.map { $0.kvSource[i] == nil } ?? true }
        let windowed = own.filter { (spec?.windows[$0] ?? 0) > 0 }.count
        let attnLayers = own.filter { ModelSpec.attentionLayerTypes.contains(cfg.layerTypes[$0]) }.count - windowed
        // KV int8 per token of a layer (K and V^T with a scale per head; MLA's latent head, K alone with two scales):
        // the sliding layers' shape (their ring) and the full layers' own one if it differs (Gemma 4).
        let kvLayer = AttentionLayer.kvBytesPerPosition(kvHeads: dims.attnKVH, hd: dims.attnHD)
        let kvFull =
            ModelDims.fullAttention(cfg).map { AttentionLayer.kvBytesPerPosition(kvHeads: $0.attnKVH, hd: $0.attnHD) }
            ?? kvLayer
        // GDN and Nemotron-H's Mamba2 (a GDN variant) share the recurrent state's layout.
        let gdnLayers = cfg.layerTypes.filter { $0 == "linear_attention" || $0 == "mamba" }.count
        let convLayers = cfg.layerTypes.filter { $0 == "conv" }.count
        // GDN (GDNLayer): hv × dv × dk fp32 state and a 3-row window of convDim = 2·hk·dk + hv·dv (Mamba2: H × P × N
        // and x | B | C).
        let gdnState = dims.gdnHV * dims.gdnDV * dims.gdnDK * 4 + 3 * dims.gdnConvDim * 4
        let cal = calibration ?? (cfg.family.map(FamilyCalibration.forFamily) ?? .unknown)
        let flash = cfg.family == .flashNext
        var dBytes: Int?
        var dSnap = 0
        var capsRow = 0
        if let dc = drafterConfig, let dw = drafterWeightBytes {
            // Drafter (Drafter.swift): 4-bit weights, captures of 2048 rows × (target layers · H) half, fc fp32,
            // enc half, kvout 2048 × 2048 fp32, each layer's kc|vt ring (KV heads × slots × head dim half)
            // and (block − 1)-row logits.
            let H = int(dc, "hidden_size") ?? cfg.hiddenSize
            let dfc = dc["dflash_config"] as? [String: Any]
            let targets = (dfc?["target_layer_ids"] as? [Any])?.count ?? 8
            let draftRows = ((dfc?["block_size"] as? NSNumber)?.intValue ?? 8) - 1
            let layers = int(dc, "num_hidden_layers") ?? 6
            // DFlash v1 (no selector in its config): its full-attention layer has a ring of 8192 (Drafter.swift).
            let wins = DFlash2Drafter.layerWindows(config: dc, layers: layers, v1: dfc?["selector_rank"] == nil)
            let slots = wins.reduce(0) { $0 + DFlash2Drafter.slots(window: $1) }
            let ring = 2 * (int(dc, "num_key_value_heads") ?? 8) * (int(dc, "head_dim") ?? 128) * 2
            let cap = 2048
            let buffers =
                cap * targets * H * 2 + cap * H * 6 + cap * 2048 * 4 + slots * ring + draftRows * cfg.vocabSize * 4
            dBytes = dw + buffers
            dSnap = slots * ring
            capsRow = targets * H * 2
        }
        self.init(
            name: name, modelType: cfg.modelType, weightBytes: weightBytes, largestTensorBytes: largestTensorBytes,
            // Flash-Next: also each attention layer's QSA block keys (512 B per block of 4).
            kvBytesPerToken: attnLayers * kvFull + 8 + (flash ? attnLayers * 128 : 0),
            kvBytesPerTokenPerLayer: kvLayer,
            nativeContext: int(t, "max_position_embeddings") ?? 262_144, drafterBytes: dBytes,
            snapshotBaseBytes: gdnLayers * gdnState + convLayers * (dims.convL - 1) * cfg.hiddenSize * 4
                + (flash ? 9 * 4 * cfg.hiddenSize * 4 : 0),
            snapshotDrafterBytes: dSnap, calibration: cal)
        windowLayers = windowed
        window = spec?.windows.max() ?? 0
        if flash, let lab = cfg.raw["metal_lab"] as? [String: Any],
            let ex = lab["experts"] as? [String: Any], let R = (ex["record_bytes"] as? NSNumber)?.intValue,
            let E = (ex["experts"] as? NSNumber)?.intValue, let L = (ex["layers"] as? NSNumber)?.intValue
        {
            streamed = StreamedExperts(
                totalBytes: R * E * L, recordBytes: R, minCacheBytes: 2 * E * R, cpuCacheBytes: 512_000_000)
            streamed!.streamBytesPerRow = 4 * cfg.hiddenSize * 4
            streamed!.draftCapsBytesPerRow = capsRow
        }
    }

    /// From the model directory (and the drafter's, if any), reading only `config.json` and the headers.
    public init(directory: URL, drafterDirectory: URL?, name: String? = nil) throws {
        let cfg = try ModelConfig(directory: directory)
        let w = try Self.tensorBytes(directory: directory)
        var dc: [String: Any]?
        var db: Int?
        if let dd = drafterDirectory {
            dc =
                try JSONSerialization.jsonObject(with: Data(contentsOf: dd.appendingPathComponent("config.json")))
                as? [String: Any]
            db = try Self.drafterWeightBytes(tensors: Self.headerTensors(directory: dd))
        }
        let gates = cfg.family == .lfm3 ? Self.repeatedGateBytes(try Self.headerTensors(directory: directory), cfg) : 0
        self.init(
            name: name ?? directory.lastPathComponent, config: cfg, weightBytes: w.total + gates,
            largestTensorBytes: w.largest, drafterConfig: dc, drafterWeightBytes: db)
        streamable = try Self.streamableExperts(cfg, tensors: Self.headerTensors(directory: directory))
    }

    /// The experts `ExpertStream` can read from the SSD (MLX `switch_mlp` layout, no dense FFN layers, one size per
    /// layer), or nil. Grouped prefill: each chunk past the first carries its residual stream (fp32) and a second
    /// chunk the drafter captures (half, 5 layers), counted with chunks of 2048.
    public static func streamableExperts(
        _ cfg: ModelConfig, tensors: [(name: String, shape: [Int], bytes: Int)],
        prefillGroup: Int = EngineOptions().expertPrefillGroup
    ) -> StreamedExperts? {
        guard let sp = try? cfg.requiredSpec, sp.expertLayout == .mlxSwitchMLP,
            !sp.ffnKinds.contains(where: { $0 == .dense || $0 == .denseAndMoE }), let E = cfg.numExperts, E > 0
        else { return nil }
        let bytes = tensors.filter { $0.name.contains("switch_mlp") && !SafetensorsStore.isSkipped($0.name) }
            .reduce(0) { $0 + $1.bytes }
        let L = cfg.numLayers
        guard bytes > 0, bytes % (E * L) == 0 else { return nil }
        let R = bytes / (E * L)
        var s = StreamedExperts(totalBytes: bytes, recordBytes: R, minCacheBytes: 2 * E * R, cpuCacheBytes: 0)
        let H = cfg.hiddenSize
        s.prefillGroup = prefillGroup
        s.fixedGroupBytes = prefillGroup > 1 ? 2048 * ((prefillGroup - 1) * H * 4 + 5 * H * 2) : 0
        s.onDemandKV = false
        return s
    }

    /// LFM3's scalar gate per head is repeated over the head at load time (AttentionLayer.headGateRowOrder): head_dim − 1
    /// more copies of each gate_proj row (codes, scales and biases) than the checkpoint stores.
    static func repeatedGateBytes(_ tensors: [(name: String, shape: [Int], bytes: Int)], _ cfg: ModelConfig) -> Int {
        let hd = ModelDims(cfg).attnHD
        return tensors.filter { $0.name.contains("self_attn.gate_proj.") }.reduce(0) { $0 + $1.bytes * (hd - 1) }
    }

    /// DFlash2 drafter weights as `DFlash2Drafter` loads them: bf16 matrices quantized to 4 bits on load
    /// (0.275 bytes per bf16 byte: 4 bits plus scale and bias per group; k_proj and v_proj count twice, in qkv and
    /// in kv) and the selector's codebooks in bf16. Measured: 0.48 GB with Qwen3.6's (0.49 predicted) and 1.27 GB
    /// with the 27B's (1.27).
    public static func drafterWeightBytes(tensors: [(name: String, shape: [Int], bytes: Int)]) -> Int {
        var total = 0.0
        for t in tensors {
            if t.name.contains("codebook") {
                total += Double(t.bytes)
            } else if t.shape.count == 2 && t.name.hasSuffix(".weight") && !t.name.contains("norm") {
                let dup = t.name.contains("self_attn.k_proj") || t.name.contains("self_attn.v_proj")
                total += Double(t.bytes) * 0.275 * (dup ? 2 : 1)
            }
        }
        return Int(total)
    }

    /// Sum of the tensor bytes of a directory's `.safetensors` (no MTP or towers, `SafetensorsStore.skippedParts`,
    /// like the loader), reading
    /// only the headers.
    public static func tensorBytes(directory: URL) throws -> (total: Int, largest: Int) {
        var total = 0, largest = 0
        for t in try headerTensors(directory: directory) where !SafetensorsStore.isSkipped(t.name) {
            total += t.bytes
            largest = max(largest, t.bytes)
        }
        return (total, largest)
    }

    /// The tensors of all the `.safetensors` headers in a directory.
    public static func headerTensors(directory: URL) throws -> [(name: String, shape: [Int], bytes: Int)] {
        let files = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "safetensors" }
        guard !files.isEmpty else { throw SafetensorsError.io("no .safetensors in \(directory.path)") }
        return try files.flatMap { try headerTensors($0) }
    }

    /// Name, shape and bytes of each tensor in a safetensors header (u64 length and the JSON), without reading
    /// the data.
    public static func headerTensors(_ url: URL) throws -> [(name: String, shape: [Int], bytes: Int)] {
        let fh = try FileHandle(forReadingFrom: url)
        defer { try? fh.close() }
        guard let lenData = try fh.read(upToCount: 8), lenData.count == 8 else {
            throw SafetensorsError.format("\(url.path) too short")
        }
        let n = lenData.withUnsafeBytes { Int(UInt64(littleEndian: $0.loadUnaligned(as: UInt64.self))) }
        guard n > 0, n < 100_000_000, let json = try fh.read(upToCount: n), json.count == n,
            let obj = try JSONSerialization.jsonObject(with: json) as? [String: Any]
        else { throw SafetensorsError.format("unreadable header in \(url.path)") }
        return obj.compactMap { name, v in
            guard name != "__metadata__", let d = v as? [String: Any], let o = d["data_offsets"] as? [Int], o.count == 2
            else { return nil }
            return (name, (d["shape"] as? [Int]) ?? [], o[1] - o[0])
        }
    }
}
