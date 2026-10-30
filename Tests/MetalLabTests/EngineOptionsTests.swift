import Foundation
import Testing

@testable import MetalLab

/// Default engine option values per architecture (the table of what each model enables):
/// they must not change when cleaning up the flags. No GPU.
@Suite struct EngineOptionsTests {
    static func config(_ textType: String, experts: Int?) throws -> ModelConfig {
        var t: [String: Any] = ["model_type": textType, "hidden_size": 2048, "num_hidden_layers": 40]
        if let experts { t["num_experts"] = experts }
        return try ModelConfig(json: ["model_type": "x", "text_config": t])
    }

    @Test func defaultsByArchitecture() {
        let o = EngineOptions()
        #expect(o.copy == .auto)
        #expect(o.copyEnabled(moe: true, drafter: true))  // Heretic / Ornith
        #expect(o.copyEnabled(moe: true, drafter: false))  // LFM2.5 (and the MoEs with --drafter none)
        #expect(!o.copyEnabled(moe: false, drafter: true))  // 27B
        #expect(o.copyEnabled(moe: false, drafter: false))
        var on = o, off = o
        on.copy = .on; off.copy = .off
        #expect(on.copyEnabled(moe: false, drafter: true) && !off.copyEnabled(moe: true, drafter: true))
        // Draft tree of 12 rows.
        #expect(o.treeNodes == 12)
    }

    @Test func prefillExactExceptServeOnMoE() {
        let gen = EngineOptions()
        #expect(!gen.prefillInt8 && !gen.prefillQK8)
        let moe = EngineOptions.serve(moe: true), dense = EngineOptions.serve(moe: false)
        #expect(moe.prefillInt8 && moe.prefillQK8)
        #expect(!dense.prefillInt8 && !dense.prefillQK8)
        // The settings that were decided.
        let i = gen.int8
        #expect(i.parts == ["gp", "go", "ap", "ao"] && i.layers == nil)
        #expect(i.minPos == nil && i.qk8MinCtx == 8192)
    }

    /// Rows by confidence in the Qwen3.6 MoEs (there is a family calibration) and a fixed 8 in the 27B (there is none).
    @Test func rowsCalibrationByFamily() throws {
        let moe = try Self.config("qwen3_5_moe_text", experts: 256),
            dense = try Self.config("qwen3_5_text", experts: nil)
        #expect(SpecRowsConfig.family(moe) == "qwen3_5_moe" && SpecRowsConfig.family(dense) == "qwen3_5")
        let empty = FileManager.default.temporaryDirectory.appendingPathComponent("metal-lab-rows-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: empty, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: empty) }
        #expect(try SpecRowsConfig.load(modelDirectory: empty, config: moe) != nil)
        #expect(try SpecRowsConfig.load(modelDirectory: empty, config: dense) == nil)
        #expect(try SpecRowsConfig.load(modelDirectory: empty, config: try Self.config("lfm2_moe", experts: 64)) == nil)
        // A file in the model folder takes precedence over the family's.
        let own = """
            {"platt": [[1, 0], [1, 0], [1, 0], [1, 0]], "cost": [{"ctx": 1000, "ms": [1, 2, 3, 4, 5, 6, 7]}], "kmin": 3}
            """
        try Data(own.utf8).write(to: empty.appendingPathComponent("metal-lab-rows.json"))
        let c = try #require(try SpecRowsConfig.load(modelDirectory: empty, config: dense))
        #expect(c.kmin == 3 && c.kmax == 8 && c.cost.count == 1)
    }
    /// The defaults that the help of the engine options (the doc comment of `engineOptions` in LabCLI/EngineCLI.swift)
    /// prints are those of `EngineOptions()` and `EngineOptions.serve`. Reads the source: no GPU, no LabCLI target.
    @Test func helpDefaultsMatchEngineOptions() throws {
        let src = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("Sources/LabCLI/EngineCLI.swift")
        let lines = try String(contentsOf: src, encoding: .utf8).split(separator: "\n").map(String.init)
        // The help line of an option and its continuation lines.
        func help(_ opt: String) throws -> String {
            let i = try #require(lines.firstIndex { $0.hasPrefix("///   \(opt) ") }, "no help line for \(opt)")
            var text = lines[i]
            for l in lines[(i + 1)...] {
                guard l.hasPrefix("///  ") && !l.hasPrefix("///   --") else { break }
                text += " " + l.dropFirst(3).trimmingCharacters(in: .whitespaces)
            }
            return text
        }
        let o = EngineOptions(), serve = EngineOptions.serve(moe: true)
        #expect(try help("--tree").contains("(\(o.treeNodes) by default; 0 = the chain)"))
        #expect(
            try help("--expert-pf-group").contains(
                "(\(o.expertPrefillGroup) by default, \(o.flashPrefillGroup) on Flash-Next"))
        #expect(try help("--expert-pf-bulk").hasSuffix("(\(o.expertBulkMinRows))"))
        #expect(try help("--expert-fn-bulk").hasSuffix("; \(o.flashBulkMinRows))"))
        #expect(try help("--ple-cache-mb").hasSuffix("(\(o.pleCacheBytes / 1_000_000))"))
        #expect(o.expertSkipRank == nil)
        let skip = try #require(serve.expertSkipRank)
        #expect(
            try help("--expert-skip-rank").contains(
                "(\(skip) = ranks \(skip + 1)–10, the default in `lab serve`; 0 = off)"))
    }
}
