import Foundation
import Metal
import Testing

@testable import MetalLab

/// Reading an MLX quantized layer from a safetensors file with a misaligned header (like the real models):
/// nibble order, scales and per-group biases. No GPU.
struct FormatTests {
    /// Writes a small safetensors file with an odd-length header (like the real models).
    func writeSafetensors(_ dir: URL, tensors: [(String, String, [Int], Data)], pad: Int = 3) throws {
        var hdr: [String: Any] = [:]
        var blob = Data()
        for (name, dt, shape, data) in tensors {
            hdr[name] = ["dtype": dt, "shape": shape, "data_offsets": [blob.count, blob.count + data.count]]
            blob.append(data)
        }
        var json = try JSONSerialization.data(withJSONObject: hdr, options: [.sortedKeys])
        while (8 + json.count) % 4 != pad { json.append(0x20) }
        var file = Data()
        var n = UInt64(json.count).littleEndian
        file.append(Data(bytes: &n, count: 8))
        file.append(json)
        file.append(blob)
        try file.write(to: dir.appendingPathComponent("model.safetensors"))
    }

    func tmpDir() throws -> URL {
        let d = FileManager.default.temporaryDirectory.appendingPathComponent("metal-lab-test-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
        return d
    }

    @Test func safetensorsQuantizedLayer() throws {
        let dir = try tmpDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        // 4-bit layer: 2 rows × 128 inputs → weight [2, 16] u32, scales/biases [2, 2] bf16.
        let N = 2, K = 128
        let q = (0..<(N * K)).map { UInt32(($0 * 7 + 3) % 16) }
        let w = packQ(q, bits: 4)
        // MLX: element j in bits 4j..4j+3 of the uint32 (low nibble first).
        #expect(packQ([1, 2, 3, 4, 5, 6, 7, 8], bits: 4) == [0x8765_4321])
        let s: [UInt16] = [0.5, 0.25, 1, 2].map(floatToBF16)
        let b: [UInt16] = [-1, 0, 0.5, 3].map(floatToBF16)
        func data<T>(_ a: [T]) -> Data { a.withUnsafeBytes { Data($0) } }
        try writeSafetensors(
            dir,
            tensors: [
                ("model.layers.0.mlp.gate_proj.weight", "U32", [N, K / 8], data(w)),
                ("model.layers.0.mlp.gate_proj.scales", "BF16", [N, K / 64], data(s)),
                ("model.layers.0.mlp.gate_proj.biases", "BF16", [N, K / 64], data(b)),
            ])
        let cfgJSON: [String: Any] = [
            "hidden_size": K, "num_hidden_layers": 1, "intermediate_size": N,
            "quantization": ["group_size": 64, "bits": 4],
        ]
        let cfg = try ModelConfig(json: cfgJSON)
        let st = try SafetensorsStore(directory: dir)
        #expect(st.tensors.count == 3)
        let l = try QuantizedLinear(store: st, path: "model.layers.0.mlp.gate_proj", config: cfg)
        #expect(l.params.bits == 4 && l.inFeatures == K && l.outFeatures == N && l.experts == 1)
        #expect(st.alignment(l.weight) < 4)  // deliberately misaligned
        #expect(try st.layerPrefix(0) == "model.layers.0.")
        let deq = st.dequantizeRows(l, rows: 0..<N)
        for r in 0..<N {
            for j in 0..<K {
                let g = r * 2 + j / 64
                let expected = bf16ToFloat(s[g]) * Float(q[r * K + j]) + bf16ToFloat(b[g])
                #expect(deq[r * K + j] == expected)
            }
        }
    }

    /// The layer prefix skips the towers and the MTP head and prefers the language model, deterministically: the Gemma 4
    /// E2B pack has `audio_tower.layers.0.` (shorter than the language model's) and the 31B ties
    /// `vision_tower.encoder.layers.0.` with `language_model.model.layers.0.` (30 characters each).
    @Test func layerPrefixPrefersLanguageModel() throws {
        let dir = try tmpDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let one = Data(count: 2)
        try writeSafetensors(
            dir,
            tensors: [
                "audio_tower.layers.0.w", "vision_tower.encoder.layers.0.w", "vision_tower.blocks.0.w",
                "language_model.model.layers.0.input_layernorm.weight", "mtp.layers.0.w", "model.visual.layers.0.w",
            ].map { ($0, "BF16", [1], one) })
        let st = try SafetensorsStore(directory: dir)
        #expect(try st.layerPrefix(0) == "language_model.model.layers.0.")
        #expect(throws: SafetensorsError.self) { try st.layerPrefix(1) }
        // Without the skip list, a tie is broken by `language_model`, then length, then lexicographic order.
        #expect(
            SafetensorsStore.preferredPrefix(["vision_tower.encoder.layers.0.", "language_model.model.layers.0."])
                == "language_model.model.layers.0.")
        #expect(SafetensorsStore.preferredPrefix(["b.layers.0.", "a.layers.0.", "model.layers.0."]) == "a.layers.0.")
        #expect(SafetensorsStore.isSkipped("vision_tower.blocks.0.attn.qkv.weight"))
        #expect(!SafetensorsStore.isSkipped("language_model.model.layers.0.linear_attn.A_log"))
    }

    /// The small GDN weights must have the dtype the kernels read (A_log: fp32, bf16 or fp16, converted to fp32);
    /// anything else is a load error instead of reading wrong bytes (Qwen3.5 0.8B from mlx-community: F32 A_log).
    @Test func gdnSmallWeightDtypes() throws {
        func store(_ dtypes: [String: String]) throws -> (SafetensorsStore, URL) {
            let dir = try tmpDir()
            let names = ["conv1d.weight", "dt_bias", "norm.weight", "A_log"]
            try writeSafetensors(
                dir,
                tensors: names.map { n in
                    let dt = dtypes[n] ?? "BF16"
                    return (
                        "model.layers.0.linear_attn." + n, dt, [1], Data(count: ["F32", "U32"].contains(dt) ? 4 : 2)
                    )
                })
            return (try SafetensorsStore(directory: dir), dir)
        }
        let pre = "model.layers.0.linear_attn."
        for (dtypes, f32, ok) in [
            ([:], false, true), (["A_log": "F32"], false, true), (["A_log": "F16"], false, true),
            (["dt_bias": "F32"], false, false), (["norm.weight": "F16"], false, false),
            (["A_log": "U32"], false, false),
            (["conv1d.weight": "F32", "dt_bias": "F32", "norm.weight": "F32", "A_log": "F32"], true, true),
            (["conv1d.weight": "F32", "dt_bias": "F32", "A_log": "F32"], true, false),
        ] as [([String: String], Bool, Bool)] {
            let (st, dir) = try store(dtypes)
            defer { try? FileManager.default.removeItem(at: dir) }
            if ok {
                try GDNLayer.checkSmallWeights(st, prefix: pre, f32: f32)
            } else {
                #expect(throws: SafetensorsError.self, "\(dtypes)") {
                    try GDNLayer.checkSmallWeights(st, prefix: pre, f32: f32)
                }
            }
        }
    }

    /// A plain norm weight must be bf16 (fp32 with `normF32`): fp16 (older MLX packs) is a load error naming the
    /// tensor, not a bf16 misread of its bytes.
    @Test func normDtypeGuard() throws {
        let dir = try tmpDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        try writeSafetensors(
            dir, tensors: ["BF16", "F16", "F32"].map { ("norm." + $0, $0, [4], Data(count: $0 == "F32" ? 16 : 8)) })
        let st = try SafetensorsStore(directory: dir)
        for (dt, f32, ok) in [
            ("BF16", false, true), ("F16", false, false), ("F32", false, false),
            ("F32", true, true), ("BF16", true, false), ("F16", true, false),
        ] as [(String, Bool, Bool)] {
            if ok {
                try checkNormDtype(st, "norm." + dt, f32: f32)
            } else {
                #expect(throws: SafetensorsError.self, "\(dt) f32=\(f32)") {
                    try checkNormDtype(st, "norm." + dt, f32: f32)
                }
            }
        }
    }

    /// Scales and biases in fp16 (older MLX packs) are a load error naming the matrix.
    @Test func fp16ScalesRefused() throws {
        let dir = try tmpDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        try writeSafetensors(
            dir,
            tensors: [
                ("l.weight", "U32", [8, 8], Data(count: 256)), ("l.scales", "F16", [8, 1], Data(count: 16)),
                ("l.biases", "F16", [8, 1], Data(count: 16)),
            ])
        let cfg = try ModelConfig(json: [
            "hidden_size": 64, "num_hidden_layers": 1, "quantization": ["group_size": 64, "bits": 4],
        ])
        let st = try SafetensorsStore(directory: dir)
        #expect(throws: SafetensorsError.self) {
            try DenseQ(store: st, config: cfg, paths: ["l"], device: MTLCreateSystemDefaultDevice()!)
        }
    }
}
