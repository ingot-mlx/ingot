import Foundation
import Testing

@testable import MetalLab

/// Memory planner (MetalLab/Memory) per memory tier, no GPU: synthetic 16/24/32/48 GB machines built from the
/// values measured on the 32 GB M6 and footprints from trimmed real configs (only the fields it uses).
struct MemoryPlannerTests {
    // The M6: physicalMemory, recommendedMaxWorkingSetSize and maxBufferLength.
    static let m6 = MachineInfo(
        physicalMemory: 34_359_738_368, recommendedWorkingSet: 26_800_603_136, maxBufferLength: 20_100_448_256,
        name: "M6")
    static func mac(_ gb: Double) -> MachineInfo { gb == 32 ? m6 : m6.simulating(memoryGB: gb) }

    static func dict(_ json: String) -> [String: Any] {
        (try? JSONSerialization.jsonObject(with: Data(json.utf8))) as? [String: Any] ?? [:]
    }
    static func cfg(_ json: String) -> ModelConfig {
        guard let c = try? ModelConfig(json: dict(json)) else { fatalError("malformed test config") }
        return c
    }
    static func layers(_ pattern: [String], _ n: Int) -> String {
        "[" + (0..<n).map { "\"\(pattern[$0 % pattern.count])\"" }.joined(separator: ",") + "]"
    }

    // Qwen3.6-35B-A3B Heretic: weights and largest tensor from its headers; DFlash2 drafter from incoai
    // (drafterWeightBytes from its headers: codebooks 254,279,680 + matrices 797,966,336 and k/v 50,331,648 at 0.275).
    static let hereticConfig = cfg(
        """
        {"model_type":"qwen3_5_moe","text_config":{"model_type":"qwen3_5_moe_text","hidden_size":2048,"num_hidden_layers":40,
         "layer_types":\(layers(["linear_attention", "linear_attention", "linear_attention", "full_attention"], 40)),
         "num_attention_heads":16,"num_key_value_heads":2,"head_dim":256,"linear_num_key_heads":16,"linear_num_value_heads":32,
         "linear_key_head_dim":128,"linear_value_head_dim":128,"max_position_embeddings":262144,"num_experts":256,"num_experts_per_tok":8,
         "moe_intermediate_size":512,"shared_expert_intermediate_size":512,"vocab_size":248320}}
        """)
    static let heretic = ModelFootprint(
        name: "Heretic",
        config: hereticConfig,
        weightBytes: 19_508_787_456, largestTensorBytes: 254_279_680,
        drafterConfig: dict(
            """
            {"hidden_size":2048,"num_hidden_layers":6,"num_key_value_heads":8,"head_dim":128,"dflash_config":{"target_layer_ids":[1,6,11,16,22,27,32,37]}}
            """),
        drafterWeightBytes: 254_279_680 + Int(Double(797_966_336 + 50_331_648) * 0.275))

    // Qwen3.8-27B calibrated (without the 29 MTP) with its DFlash2 drafter (5 layers, hidden 5120).
    static let q27 = ModelFootprint(
        name: "27B",
        config: cfg(
            """
            {"model_type":"qwen3_5","text_config":{"model_type":"qwen3_5_text","hidden_size":5120,"num_hidden_layers":64,
             "layer_types":\(layers(["linear_attention", "linear_attention", "linear_attention", "full_attention"], 64)),
             "num_attention_heads":24,"num_key_value_heads":4,"head_dim":256,"linear_num_key_heads":16,"linear_num_value_heads":48,
             "linear_key_head_dim":128,"linear_value_head_dim":128,"max_position_embeddings":262144,"intermediate_size":17408,"vocab_size":248320}}
            """),
        weightBytes: 15_132_802_048, largestTensorBytes: 635_699_200,
        drafterConfig: dict(
            """
            {"hidden_size":5120,"num_hidden_layers":5,"num_key_value_heads":8,"head_dim":128,"dflash_config":{"target_layer_ids":[5,19,33,47,61]}}
            """),
        drafterWeightBytes: 254_279_680 + Int(Double(3_593_994_240 + 104_857_600) * 0.275))

    // LFM2.5-24B-A2B calibrated (size of its safetensors; no drafter).
    static let lfm = ModelFootprint(
        name: "LFM2.5",
        config: cfg(
            """
            {"model_type":"lfm2_moe","hidden_size":2048,"num_hidden_layers":40,"num_dense_layers":2,
             "layer_types":\(layers(["conv", "conv", "full_attention", "conv"], 40)),
             "num_attention_heads":32,"num_key_value_heads":8,"max_position_embeddings":131072,"num_experts":64,"vocab_size":128000}
            """),
        weightBytes: 13_552_000_000, largestTensorBytes: 300_000_000, drafterConfig: nil, drafterWeightBytes: nil)

    // Synthetic: a ~9B dense model at 4 bits (Qwen3.5-9B shape, 5.5 GB), the candidate for 16 GB.
    static let small9B = ModelFootprint(
        name: "9B",
        config: cfg(
            """
            {"model_type":"qwen3_5_text","hidden_size":4096,"num_hidden_layers":32,
             "layer_types":\(layers(["linear_attention", "linear_attention", "linear_attention", "full_attention"], 32)),
             "num_attention_heads":16,"num_key_value_heads":4,"head_dim":256,"linear_num_value_heads":32,"max_position_embeddings":262144,"vocab_size":248320}
            """),
        weightBytes: 5_500_000_000, largestTensorBytes: 600_000_000, drafterConfig: nil, drafterWeightBytes: nil)

    func plan(_ m: MachineInfo, _ f: ModelFootprint, _ o: MemoryOptions = MemoryOptions()) throws -> MemoryPlan {
        try MemoryPlanner.plan(machine: m, model: f, options: o)
    }

    @Test func footprintsMatchMeasurements() {
        // KV per token: 10.4 KB (Qwen3.6) and 33.3 KB (27B), as in defaultMaxCtx.
        #expect(Self.heretic.kvBytesPerToken == 10 * 2 * 2 * 260 + 8)
        #expect(Self.q27.kvBytesPerToken == 16 * 2 * 4 * 260 + 8)
        // Snapshots measured with lab serve: 0.12 GB (0.07 without drafter) and 0.20 GB (0.16).
        #expect(abs(Double(Self.heretic.snapshotBytes(drafter: true)) / 1e9 - 0.12) < 0.005)
        #expect(abs(Double(Self.heretic.snapshotBytes(drafter: false)) / 1e9 - 0.07) < 0.005)
        #expect(abs(Double(Self.q27.snapshotBytes(drafter: true)) / 1e9 - 0.20) < 0.005)
        // Loaded drafter (Metal memory difference with and without drafter): 0.66 and 1.53 GB.
        #expect(abs(Double(Self.heretic.drafterBytes!) / 1e9 - 0.66) < 0.03)
        #expect(abs(Double(Self.q27.drafterBytes!) / 1e9 - 1.53) < 0.05)
        // Measured at startup with 32K, no drafter or snapshots: 20.84 and 17.97 GB.
        let h =
            Self.heretic.weightBytes + Self.heretic.prefillBytes(chunkRows: 2048) + 32768 * Self.heretic.kvBytesPerToken
        let q = Self.q27.weightBytes + Self.q27.prefillBytes(chunkRows: 2048) + 32768 * Self.q27.kvBytesPerToken
        #expect(abs(Double(h) / 1e9 - 20.84) < 0.03)
        #expect(abs(Double(q) / 1e9 - 17.97) < 0.03)
    }

    /// DFlash v1 drafter (Gemma 4 26B-A4B's, z-lab's: no selector): the full-attention layer gets a ring of 8192 + 32
    /// slots and the four sliding ones 2048 + 32 (Drafter.swift); a DFlash2 keeps 2048 everywhere. The snapshots copy
    /// every ring: (4 · 2080 + 8224) slots × K and V of 8 heads × 128 half, 67.8 MB (42.6 with the 2048 ring alone).
    @Test func drafterV1FullRing() {
        let types = Array(repeating: "sliding_attention", count: 4) + ["full_attention"]
        let dc: [String: Any] = [
            "hidden_size": 4096, "num_hidden_layers": 5, "num_key_value_heads": 8, "head_dim": 128,
            "layer_types": types, "dflash_config": ["target_layer_ids": [1, 6, 11, 17, 22, 27]],
        ]
        #expect(DFlash2Drafter.layerWindows(config: dc, layers: 5, v1: true) == [2048, 2048, 2048, 2048, 8192])
        #expect(DFlash2Drafter.layerWindows(config: dc, layers: 5, v1: false) == Array(repeating: 2048, count: 5))
        let f = ModelFootprint(
            name: "v1",
            config: Self.cfg(
                """
                {"model_type":"qwen3_5_text","hidden_size":4096,"num_hidden_layers":4,"layer_types":\(Self.layers(["full_attention"], 4)),
                 "num_attention_heads":16,"num_key_value_heads":4,"head_dim":256,"max_position_embeddings":262144,"vocab_size":1024}
                """),
            weightBytes: 1_000_000_000, largestTensorBytes: 100_000_000, drafterConfig: dc, drafterWeightBytes: 0)
        #expect(f.snapshotDrafterBytes == (4 * 2080 + 8224) * 2 * 8 * 128 * 2)
    }

    /// 32 GB (the M6): the usual values, 128K with Qwen3.6 and 64K with the 27B, nothing trimmed.
    @Test func mac32() throws {
        #expect(MemoryPlanner.budget(Self.m6) == 24_300_603_136)
        let h = try plan(Self.m6, Self.heretic)
        #expect(
            h.context == 131_072 && h.drafter && h.prefixCacheBytes == 1_000_000_000 && h.snapshots == 4
                && h.chunkRows == 2048)
        let q = try plan(Self.m6, Self.q27)
        #expect(
            q.context == 65_536 && q.drafter && q.prefixCacheBytes == 1_000_000_000 && q.snapshots == 4
                && q.chunkRows == 2048)
        let l = try plan(Self.m6, Self.lfm)
        #expect(l.context == 131_072 && !l.drafter)
        // The 128K case was measured: 22.76 GB at startup with 2 snapshots created during warmup.
        #expect(abs(Double(h.expectedAtLoad + 2 * Self.heretic.snapshotBytes(drafter: true)) / 1e9 - 22.76) < 0.05)
    }

    /// Tiers of other Macs (simulated with this one's ratio): 48 GB, native context (256K) with all defaults;
    /// 24 GB, the 4-bit Heretic does not fit, the 27B is trimmed to the useful minimum and LFM2.5 fits whole;
    /// 16 GB, only a 4-bit 9B (128K), and the refusal explains what it asks for and what the Mac leaves.
    @Test func otherMacs() throws {
        let h48 = try plan(Self.mac(48), Self.heretic), q48 = try plan(Self.mac(48), Self.q27)
        #expect(h48.context == 262_144 && h48.drafter && h48.prefixCacheBytes == 1_000_000_000)
        #expect(q48.context == 262_144 && q48.drafter)

        #expect(throws: MemoryPlanError.self) { try plan(Self.mac(24), Self.heretic) }
        let q24 = try plan(Self.mac(24), Self.q27)
        #expect(
            q24.context == 16_384 && !q24.drafter && q24.prefixCacheBytes == 0 && q24.snapshots == 2
                && q24.chunkRows == 1024)
        #expect(q24.warnings.contains { $0.contains("no drafter") })
        let l24 = try plan(Self.mac(24), Self.lfm)
        #expect(l24.context == 131_072 && l24.prefixCacheBytes == 1_000_000_000)

        for f in [Self.q27, Self.lfm] { #expect(throws: MemoryPlanError.self) { try plan(Self.mac(16), f) } }
        let s16 = try plan(Self.mac(16), Self.small9B)
        #expect(s16.context == 131_072 && s16.fits)
        do {
            _ = try plan(Self.mac(16), Self.heretic)
            Issue.record("should have refused")
        } catch {
            let s = "\(error)"
            #expect(s.contains("does not fit") && s.contains("19.51 GB") && s.contains("simulating a 16 GB Mac"))
        }
    }

    /// Lowering the budget step by step, trims come in order: prefix cache → snapshots → chunk → context →
    /// drafter, and the plan always fits.
    @Test func trimOrder() throws {
        var rank = 0
        var sawDrafterOff = false
        for i in 0..<400 {
            let gb = 48 - Double(i) * 0.05
            guard let p = try? plan(Self.mac(gb), Self.heretic) else { break }
            #expect(p.fits)
            let r: Int
            if !p.drafter {
                r = 6
            } else if p.context < 32768 {
                r = 5
            } else if p.chunkRows < 2048 {
                r = 4
            } else if p.snapshots < 4 {
                r = 3
            } else if p.prefixCacheBytes == 0 {
                r = 2
            } else if p.prefixCacheBytes < 1_000_000_000 {
                r = 1
            } else {
                r = 0
            }
            #expect(r >= rank, "at \(gb) GB the trim goes backwards (\(r) < \(rank))")
            rank = max(rank, r)
            // Each step implies the previous ones.
            if p.snapshots < 4 { #expect(p.prefixCacheBytes == 0) }
            if p.chunkRows < 2048 { #expect(p.snapshots == 2) }
            if p.context < 32768 { #expect(p.chunkRows == 1024) }
            if !p.drafter { sawDrafterOff = true; #expect(p.chunkRows == 1024 && p.prefixCacheBytes == 0) }
        }
        #expect(rank == 6 && sawDrafterOff)
    }

    /// What the user sets wins: the rest is adjusted and, if it does not fit, a warning is given.
    @Test func userValuesWin() throws {
        // 256K on 32 GB: fits by dropping the prefix cache and two snapshots.
        let a = try plan(Self.m6, Self.heretic, MemoryOptions(maxContext: 262_144))
        #expect(a.context == 262_144 && a.fits && a.prefixCacheBytes < 1_000_000_000)
        // No prefix cache leaves memory over, but context stays what fits (128K: 256K does not fit with 4 snapshots).
        let b = try plan(Self.m6, Self.heretic, MemoryOptions(prefixCacheBytes: 0))
        #expect(b.prefixCacheBytes == 0 && b.context == 131_072)
        // --drafter none.
        let c = try plan(Self.m6, Self.q27, MemoryOptions(drafter: false))
        #expect(!c.drafter && c.context >= 65_536)
        // Settings that do not fit: used anyway, with a warning.
        let d = try plan(
            Self.mac(28), Self.heretic, MemoryOptions(maxContext: 262_144, prefixCacheBytes: 4_000_000_000))
        #expect(d.context == 262_144 && d.prefixCacheBytes == 4_000_000_000 && !d.fits)
        #expect(d.warnings.contains { $0.contains("does not fit") })
        // But if the model does not fit even at the minimum, it refuses even when the user sets values.
        #expect(throws: MemoryPlanError.self) { try plan(Self.mac(16), Self.heretic, MemoryOptions(maxContext: 8192)) }
    }

    @Test func maxBufferIsChecked() {
        var m = Self.m6
        m.maxBufferLength = 200_000_000
        #expect(throws: MemoryPlanError.self) { try plan(m, Self.heretic) }
    }

    /// Safetensors headers are read without loading data, and mtp and visual are dropped.
    @Test func headerParsing() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("memplan-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let header = """
            {"__metadata__":{"format":"mlx"},"model.layers.0.w.weight":{"dtype":"U32","shape":[4,8],"data_offsets":[0,128]},
             "mtp.layers.0.w.weight":{"dtype":"BF16","shape":[8,8],"data_offsets":[128,256]},
             "model.visual.x":{"dtype":"BF16","shape":[16],"data_offsets":[256,288]},
             "model.norm.weight":{"dtype":"BF16","shape":[512],"data_offsets":[288,1312]}}
            """
        var data = Data()
        var n = UInt64(header.utf8.count).littleEndian
        data.append(Data(bytes: &n, count: 8))
        data.append(Data(header.utf8))
        data.append(Data(count: 1312))
        try data.write(to: dir.appendingPathComponent("model.safetensors"))
        let t = try ModelFootprint.tensorBytes(directory: dir)
        #expect(t.total == 128 + 1024 && t.largest == 1024)
        // Drafter weights: 2D matrices at 0.275 (k/v twice), codebooks whole, norms excluded.
        let w = ModelFootprint.drafterWeightBytes(tensors: [
            ("layers.0.self_attn.q_proj.weight", [8, 8], 1000), ("layers.0.self_attn.k_proj.weight", [8, 8], 1000),
            ("candidate_selector.predecessor_codebook", [8, 8], 500), ("layers.0.input_layernorm.weight", [8], 16),
        ])
        #expect(w == 275 + 550 + 500)
    }

    /// Flash-Next (experts on SSD) with on-demand KV (KVPool, default): native context and more expert cache than
    /// the previous plan (whole 128K KV at load); --kv-pool 0 restores the previous behavior.
    @Test func flashNextKVPool() throws {
        var f = ModelFootprint(
            name: "Flash-Next", modelType: "qwen4_exp_text", weightBytes: 3_330_000_000,
            largestTensorBytes: 800_000_000,
            kvBytesPerToken: 12 * 1040 + 8 + 12 * 128, kvBytesPerTokenPerLayer: 1040, nativeContext: 262_144,
            drafterBytes: 680_000_000, snapshotBaseBytes: 160_000_000, snapshotDrafterBytes: 0,
            calibration: .unknown)
        f.streamed = .init(
            totalBytes: 48 * 512 * 1_392_640, recordBytes: 1_392_640, minCacheBytes: 2 * 512 * 1_392_640,
            cpuCacheBytes: 512_000_000)
        let p = try MemoryPlanner.plan(machine: Self.m6, model: f)
        #expect(p.context == 262_144 && p.kvGrow && p.kvAllowance == MemoryPlanner.poolAllowance)
        #expect(p.fits)
        // KV beyond the allowance fits by giving up the prefix cache, 2 snapshots and the slack.
        let extra = f.kvBytesPerToken * (p.context - p.kvAllowance)
        #expect(extra <= p.prefixCacheBytes + (p.snapshots - 2) * f.snapshotBytes(drafter: p.drafter) + 1_500_000_000)
        var o = MemoryOptions()
        o.kvPool = false
        let q = try MemoryPlanner.plan(machine: Self.m6, model: f, options: o)
        #expect(!q.kvGrow && q.context == 131_072)
        #expect(p.expertCacheBytes! > q.expertCacheBytes! + 700_000_000)
        // The PLE row cache is CPU memory: it counts against physical, not Metal (+0.51 GB of expert cache), and on
        // the m6 physical does not bind, so the budget does not change.
        #expect(MemoryPlanner.budget(Self.m6, cpuBytes: 512_000_000) == MemoryPlanner.budget(Self.m6))
        #expect(p.total <= p.budget)
    }

    /// The Heretic with its experts streamable (as `ModelFootprint(directory:)` reads them from the headers): 40
    /// layers × 256 experts × 1,769,472 bytes (gate, up and down at 4 bits with their scales and biases) = 18.12 GB.
    static let hereticSSD: ModelFootprint = {
        var f = heretic
        f.streamable = ModelFootprint.streamableExperts(
            hereticConfig, tensors: [("model.layers.0.mlp.switch_mlp.gate_proj.weight", [], 40 * 256 * 1_769_472)])
        return f
    }()

    /// Qwen3.6 MoE that does not fit in memory: the plan moves its experts to the SSD and gives the expert cache what
    /// is left with 64K counted, a 0.25 GB prefix cache, 4 snapshots and the drafter (16 GB: 6.37 GB, the ~6.4 of
    /// RG-42 T0's sweep; before, `--expert-cache-gb 6` by hand). What fits in memory does not change (32 and 48 GB).
    @Test func streamedWhenItDoesNotFit() throws {
        let st = try #require(Self.hereticSSD.streamable)
        #expect(st.totalBytes == 18_119_393_280 && st.minCacheBytes == 2 * 256 * 1_769_472 && !st.onDemandKV)
        for gb in [32.0, 48] {
            let a = try plan(Self.mac(gb), Self.hereticSSD), b = try plan(Self.mac(gb), Self.heretic)
            #expect(a.expertCacheBytes == nil && a.context == b.context && a.total == b.total)
            #expect(a.prefixCacheBytes == b.prefixCacheBytes && a.snapshots == b.snapshots && a.drafter == b.drafter)
        }
        func ssd(_ m: MachineInfo, _ cache: ClosedRange<Double>) throws -> MemoryPlan {
            let p = try plan(m, Self.hereticSSD)
            let x = Double(try #require(p.expertCacheBytes)) / 1e9
            #expect(cache.contains(x), "expert cache \(x) GB")
            #expect(p.context == 65_536 && p.prefixCacheBytes == 250_000_000 && p.snapshots == 4)
            #expect(p.chunkRows == 2048 && p.drafter && !p.kvGrow && p.fits)
            // The cache takes the whole budget.
            #expect(p.budget - p.total < 1_000_000)
            #expect(p.explanation.contains { $0.contains("read from the SSD") })
            return p
        }
        let p16 = try ssd(Self.mac(16), 6.3...6.45)
        _ = try ssd(Self.mac(24), 13.0...13.2)
        _ = try ssd(Self.m6.simulating(memoryGB: 16, rwsGB: 10.7), 3.6...3.75)
        // The manual plan with --expert-cache-gb 6 (weights 7.48 GB with its prefill group; MemPlan.swift) picks the
        // same values and leaves 0.37 GB unused: the automatic cache is 6 GB plus that.
        var six = Self.heretic
        six.weightBytes += 6_000_000_000 - st.totalBytes + st.fixedGroupBytes!
        let m6 = try plan(Self.mac(16), six)
        #expect(m6.context == p16.context && m6.prefixCacheBytes == p16.prefixCacheBytes && m6.snapshots == 4)
        #expect(abs(p16.expertCacheBytes! - (6_000_000_000 + m6.budget - m6.total)) < 1_000_000)
        // What the user sets wins and the cache gets the rest: a 1 GB prefix cache leaves 0.75 GB less.
        let q = try plan(Self.mac(16), Self.hereticSSD, MemoryOptions(prefixCacheBytes: 1_000_000_000))
        #expect(abs(p16.expertCacheBytes! - q.expertCacheBytes! - 750_000_000) < 1_000_000)
        // A context set in memory that does not fit (24 GB, 8K): to the SSD too, with that context.
        let r = try plan(Self.mac(24), Self.hereticSSD, MemoryOptions(maxContext: 8192))
        #expect(r.context == 8192 && r.expertCacheBytes != nil && r.fits)
        // --drafter none: the drafter's memory goes to the cache.
        let n = try plan(Self.mac(16), Self.hereticSSD, MemoryOptions(drafter: false))
        #expect(!n.drafter && n.expertCacheBytes! > p16.expertCacheBytes! + 600_000_000)
        // Lowering memory, the plan never refuses while the minimum cache fits (16 GB at 10.7 down to 8 GB).
        for i in 0..<20 {
            let gb = 16 - Double(i) * 0.4
            guard let p = try? plan(Self.mac(gb), Self.hereticSSD) else { break }
            #expect(p.fits && p.expertCacheBytes! >= st.minCacheBytes)
        }
    }

    /// lab serve batching: the extra slots fit in what the plan leaves, up to 4, most slots first.
    @Test func batchSlots() throws {
        for gb in [16.0, 24, 32, 48] {
            let p = try MemoryPlanner.plan(machine: Self.mac(gb), model: Self.small9B)
            let b = MemoryPlanner.batchSlots(plan: p, model: Self.small9B)
            #expect(b.slots >= 1 && b.slots <= 4)
            #expect(
                b.slots == 1
                    ? b.positions == 0 && b.bytes == 0 : b.positions >= 16_384 && b.positions <= min(p.context, 32_768))
            #expect(p.total + b.bytes <= p.budget)
        }
        let p = try MemoryPlanner.plan(machine: Self.m6, model: Self.small9B)
        let b = MemoryPlanner.batchSlots(plan: p, model: Self.small9B)
        #expect(b.slots == 4)
        // One more slot or the next longer context would not fit.
        let one = b.bytes / 3
        #expect(
            b.positions == min(p.context, 32_768)
                || p.total + 3 * (one + Self.small9B.kvBytesPerToken * b.positions) > p.budget)
    }

    /// With a drafter each slot also takes its drafter ring, and its snapshots carry it. On the M6 the
    /// Heretic's default plan (context 131072, prefix cache 1 GB, 4 snapshots) leaves 0.31 GB: no extra slot fits.
    /// With a 32K context, a 0.25 GB prefix cache and 2 snapshots it keeps 4 slots of 32K.
    @Test func batchSlotsWithDrafter() throws {
        let p = try MemoryPlanner.plan(machine: Self.m6, model: Self.heretic)
        #expect(p.drafter && p.snapshots == 4)
        let none = MemoryPlanner.batchSlots(plan: p, model: Self.heretic)
        #expect(none.slots == 1 && none.bytes == 0)
        let q = try MemoryPlanner.plan(
            machine: Self.m6, model: Self.heretic,
            options: MemoryOptions(maxContext: 32_768, prefixCacheBytes: 250_000_000, snapshots: 2))
        let b = MemoryPlanner.batchSlots(plan: q, model: Self.heretic)
        #expect(q.total + b.bytes <= q.budget)
        let ring = Self.heretic.snapshotDrafterBytes
        #expect(ring > 0)
        func slot(_ snaps: Int) -> Int {
            Self.heretic.kvBytesPerToken * b.positions + (snaps + 1) * Self.heretic.snapshotBytes(drafter: true) + ring
        }
        #expect(b.slots == 4 && b.positions == 32_768 && b.snapshots == 2)
        #expect(b.bytes == 3 * slot(2) + ring)
    }
}
