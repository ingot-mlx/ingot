import Foundation
import LabBench
import MetalLab
import Testing

/// Which drafter each model runs with (DrafterFit.swift), with the real configs of ref/toy/real/ and the
/// drafters' config.json written to a temporary directory: no weights, no GPU. The family's default drafter (Qwen3.6's)
/// fits the Heretic and not Qwen3.5 0.8B (hidden) or 2B (target layers past its 24); `--drafter none` gives none.
@Suite struct DrafterFitTests {
    /// The drafters' config.json, trimmed to what the check reads (copied from the HF cache and the 27B pack).
    static func drafter(hidden: Int, targets: [Int], layers: Int) -> [String: Any] {
        [
            "hidden_size": hidden, "num_target_layers": layers, "head_dim": 128, "num_attention_heads": 32,
            "num_key_value_heads": 8, "sliding_window": 2048,
            "dflash_config": [
                "block_size": 8, "conv_group_size": 16, "conv_kernel_size": 2, "selector_rank": 256,
                "selector_top_k": 16, "target_layer_ids": targets,
            ] as [String: Any],
        ]
    }
    static var q36: [String: Any] { drafter(hidden: 2048, targets: [1, 6, 11, 16, 22, 27, 32, 37], layers: 40) }
    static var q27: [String: Any] { drafter(hidden: 5120, targets: [5, 19, 33, 47, 61], layers: 64) }
    static var flash: [String: Any] { drafter(hidden: 2560, targets: [5, 15, 25, 35, 45], layers: 48) }

    static func real(_ name: String) throws -> [String: Any] { try ToyReal.load(name, root: ModelSpecTests.repo).json }
    static func dims(_ name: String) throws -> (hidden: Int, layers: Int) {
        let c = try ModelConfig(json: try real(name))
        return (c.hiddenSize, c.numLayers)
    }

    @Test func mismatchByConfig() throws {
        func check(_ d: [String: Any], _ model: String) throws -> String? {
            let m = try Self.dims(model)
            return DFlash2Drafter.mismatch(config: d, hidden: m.hidden, layers: m.layers, v1: false)
        }
        #expect(try check(Self.q36, "qwen3.6-35b-a3b") == nil)
        #expect(try check(Self.q27, "qwen3.8-27b") == nil)
        #expect(try check(Self.flash, "qwen3.8-flash-next") == nil)
        #expect(try check(Self.q36, "qwen3.8-27b") == "hidden 2048 vs 5120")
        #expect(try check(Self.q36, "qwen3.5-0.8b") == "hidden 2048 vs 1024")
        // Same hidden as the 2B: it would pass the old guard and capture layers that do not exist.
        #expect(try check(Self.q36, "qwen3.5-2b") == "target layer 27 ≥ 24 layers")
        var n = Self.q36
        n["num_target_layers"] = 48
        #expect(try check(n, "qwen3.6-35b-a3b") == "num_target_layers 48 vs 40")
        var heads = Self.q36
        heads["num_attention_heads"] = 16
        #expect(try check(heads, "qwen3.6-35b-a3b")?.hasPrefix("heads 16/8") == true)
        // DFlash2 blocks of 8 and 16 (Muse Glimmer's); any other is a named error.
        var block = Self.q36
        var bdc = try #require(block["dflash_config"] as? [String: Any])
        bdc["block_size"] = 16
        block["dflash_config"] = bdc
        #expect(try check(block, "qwen3.6-35b-a3b") == nil)
        bdc["block_size"] = 12
        block["dflash_config"] = bdc
        #expect(try check(block, "qwen3.6-35b-a3b") == "DFlash2 block, window, conv or selector not supported")
        // DFlash v1 does not need the DFlash2 block; unknown v1/v2 (no headers) skips that check.
        #expect(DFlash2Drafter.mismatch(config: block, hidden: 2048, layers: 40, v1: true) == nil)
        #expect(DFlash2Drafter.mismatch(config: block, hidden: 2048, layers: 40, v1: nil) == nil)
    }

    /// A temporary tree with model and drafter directories (config.json only).
    final class Tree {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("drafterfit-\(UUID().uuidString)")
        func dir(_ name: String, _ json: [String: Any]) throws -> URL {
            let d = root.appendingPathComponent(name)
            try FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
            try JSONSerialization.data(withJSONObject: json).write(to: d.appendingPathComponent("config.json"))
            return d
        }
        deinit { try? FileManager.default.removeItem(at: root) }
    }

    @Test func resolution() throws {
        let t = Tree()
        // The HF cache: the family drafters, by name.
        let cache = [
            "incoai/Qwen3.6-35B-A3B-DFlash2": try t.dir("hf-q36", Self.q36),
            "incoai/Qwen3.8-Flash-Next-DFlash2": try t.dir("hf-flash", Self.flash),
        ]
        func locate(_ s: String) throws -> URL {
            guard let u = cache[s] else { throw SafetensorsError.io("cannot find \(s)") }
            return u
        }
        var logged: [String] = []
        func resolve(_ dir: URL, option: String? = nil, toyPack: Bool = true) throws -> URL? {
            try DFlash2Drafter.resolve(
                modelDir: dir, config: try ModelConfig(directory: dir), option: option, toyPack: toyPack,
                locate: locate
            ) { logged.append($0) }
        }
        let heretic = try t.dir("heretic", try Self.real("qwen3.6-35b-a3b"))
        let q27 = try t.dir("q27", try Self.real("qwen3.8-27b"))
        _ = try t.dir("q27/drafter", Self.q27)
        let flash = try t.dir("flash", try Self.real("qwen3.8-flash-next"))
        let q08 = try t.dir("q08", try Self.real("qwen3.5-0.8b"))
        let q2 = try t.dir("q2", try Self.real("qwen3.5-2b"))
        let granite = try t.dir("granite", try Self.real("granite-4.1-3b"))
        var toyJSON = try Self.real("granite-4.1-3b")
        toyJSON["metal_lab_toy"] = ["arch": "granite", "seed": 1] as [String: Any]
        let toyGranite = try t.dir("toy-granite", toyJSON)
        _ = try t.dir("toy-granite/drafter", Self.drafter(hidden: 2560, targets: [0, 1, 2, 3], layers: 40))

        // The same as before for the models whose drafter fits.
        #expect(try resolve(heretic) == cache["incoai/Qwen3.6-35B-A3B-DFlash2"])
        #expect(try resolve(q27) == q27.appendingPathComponent("drafter"))
        #expect(try resolve(flash) == cache["incoai/Qwen3.8-Flash-Next-DFlash2"])
        #expect(logged.isEmpty)
        // Qwen3.5 0.8B and 2B: the family's does not fit, so no drafter and one line saying why.
        #expect(try resolve(q08) == nil)
        #expect(try resolve(q2) == nil)
        #expect(logged.count == 2)
        #expect(logged.first?.contains("hidden 2048 vs 1024") == true)
        #expect(logged.last?.contains("target layer 27 ≥ 24 layers") == true)
        // --drafter none: none, also in a family that has one (lab-dev gen, reuse, prefixcache, splitcheck).
        for d in [heretic, q27, flash, q08] { #expect(try resolve(d, option: "none") == nil) }
        // --drafter with a name: that one, and an error if it does not fit.
        let q36Name = "incoai/Qwen3.6-35B-A3B-DFlash2"
        #expect(try resolve(heretic, option: q36Name) == cache[q36Name])
        #expect(throws: SafetensorsError.self) { try resolve(q08, option: q36Name) }
        #expect(throws: SafetensorsError.self) { try resolve(q2, option: q36Name) }
        // A family without drafter: none; the toy's `drafter/` only with toyPack (gen uses it only with --drafter).
        #expect(try resolve(granite) == nil)
        #expect(try resolve(toyGranite, toyPack: false) == nil)
        #expect(try resolve(toyGranite) == toyGranite.appendingPathComponent("drafter"))
        // Gemma 4: z-lab's DFlash v1 by default only on the 26B-A4B (hidden 2816), an error if it is missing from the
        // cache; E2B and 31B run without one and do not look it up (locate would throw).
        let g26 = try t.dir("g26", try Self.real("gemma4-26b-a4b"))
        let gemmaV1: [String: Any] = [
            "hidden_size": 2816, "num_target_layers": 30, "head_dim": 128, "num_attention_heads": 32,
            "num_key_value_heads": 8, "sliding_window": 2048, "block_size": 16,
            "dflash_config": ["mask_token_id": 4, "target_layer_ids": [1, 6, 11, 17, 22, 27]] as [String: Any],
        ]
        let gName = "z-lab/gemma-4-26B-A4B-it-DFlash"
        let gDir = try t.dir("hf-gemma", gemmaV1)
        #expect(throws: SafetensorsError.self) { try resolve(g26) }
        #expect(
            try DFlash2Drafter.resolve(
                modelDir: g26, config: try ModelConfig(directory: g26), option: nil, toyPack: true,
                locate: { $0 == gName ? gDir : try locate($0) }, log: { _ in }) == gDir)
        #expect(try resolve(g26, option: "none") == nil)
        for name in ["gemma4-e2b", "gemma4-e4b", "gemma4-31b"] {
            #expect(try resolve(try t.dir(name, try Self.real(name))) == nil, "\(name)")
        }
        // A drafter missing from the cache is still an error (lab plan and lab info go without it: plannedDrafterDir).
        let lost = try t.dir("lost", try Self.real("qwen3.6-35b-a3b"))
        #expect(throws: SafetensorsError.self) {
            try DFlash2Drafter.resolve(
                modelDir: lost, config: try ModelConfig(directory: lost), option: nil, toyPack: true,
                locate: { throw SafetensorsError.io("cannot find \($0)") }, log: { _ in })
        }
    }
}
