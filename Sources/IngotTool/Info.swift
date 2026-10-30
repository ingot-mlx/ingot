import Foundation
import Ingot
import IngotCLI
import Metal

/// ingot info --model org/repo|dir [--layer 10]: files, config, quantization and alignment of one layer, and the GPU.
func cmdInfo() throws {
    let model = try args.required("model", modelHelp)
    let dir = try ModelPaths.resolve(model)
    let t0 = Date()
    let store = try SafetensorsStore(directory: dir)
    let cfg = try ModelConfig(directory: dir)
    print("model: \(dir.path)")
    print(
        String(
            format: "safetensors: %d shards, %d tensors, headers read in %.0f ms",
            store.files.count, store.tensors.count, Date().timeIntervalSince(t0) * 1e3))
    for f in store.files {
        let hdr = Int(f.base.loadUnaligned(as: UInt64.self).littleEndian)
        print(
            "  \(URL(fileURLWithPath: f.path).lastPathComponent): \(fmtBytes(f.fileSize)), data from byte \(8 + hdr) (mod 16 = \((8 + hdr) % 16))"
        )
    }
    print(
        "config: \(cfg.modelType), hidden \(cfg.hiddenSize), layers \(cfg.numLayers), "
            + (cfg.isMoE
                ? "MoE \(cfg.numExperts!)×\(cfg.moeIntermediateSize ?? 0) top-\(cfg.expertsPerToken ?? 0), shared \(cfg.sharedExpertIntermediateSize ?? 0)"
                : "intermediate \(cfg.intermediateSize ?? 0)"))
    print(
        "default quantization: \(cfg.quantDefault?.description ?? "none"), \(cfg.quantOverrides.count) overrides"
    )
    let layer = args.int("layer", 10)
    let d = try describeLayer(store, config: cfg, layer: layer)
    print("layer \(layer) (\(d.type)), prefix \(d.prefix), MLP \(d.mlp)")
    for q in d.quantized {
        let shape = q.weight.shape.map(String.init).joined(separator: "×")
        print(
            "  \(q.path.dropFirst(d.prefix.count)): \(q.params), \(q.experts > 1 ? "\(q.experts) experts × " : "")\(q.outFeatures)×\(q.inFeatures) "
                + "(weight \(shape)), alignment \(store.alignment(q.weight)) B"
        )
    }
    if let device = MTLCreateSystemDefaultDevice() {
        print(
            "GPU: \(device.name), maxBufferLength \(fmtBytes(device.maxBufferLength)), "
                + "recommendedMaxWorkingSetSize \(fmtBytes(Int(device.recommendedMaxWorkingSetSize)))"
        )
    }
}

/// ingot plan [--model org/repo|dir] [--mem-limit-gb N] [--rws-gb X] [--max-ctx N] ...
/// The memory plan without loading the model; without --model, that of every model in the directory named by
/// INGOT_MODELS.
func cmdPlan() throws {
    let device = MTLCreateSystemDefaultDevice()
    let m = planMachine(device: device)
    let gb = { (b: Int) in String(format: "%.2f GB", Double(b) / 1e9) }
    print(
        "Mac: \(m.name), memory \(gb(m.physicalMemory)), GPU \(gb(m.recommendedWorkingSet)) (recommendedMaxWorkingSetSize), "
            + "max buffer \(gb(m.maxBufferLength))\(m.availableMemory.map { ", free or reclaimable now \(gb($0))" } ?? "")"
    )
    if let s = m.simulated { print(s) }
    var models: [String] = []
    if let one = args.opts["model"] {
        models = [one]
    } else {
        guard let env = ProcessInfo.processInfo.environment["INGOT_MODELS"], !env.isEmpty else {
            throw UsageError("ingot plan without --model needs INGOT_MODELS (a directory of models), or pass --model")
        }
        let root = (env as NSString).expandingTildeInPath
        models = ((try? FileManager.default.contentsOfDirectory(atPath: root)) ?? []).sorted()
            .filter { FileManager.default.fileExists(atPath: root + "/" + $0 + "/config.json") }.map { root + "/" + $0 }
        if models.isEmpty { throw UsageError("INGOT_MODELS=\(root): no model directories (with config.json)") }
    }
    for name in models {
        print("\n== \(name)")
        let dir = try ModelPaths.resolve(name)
        let cfg = try ModelConfig(directory: dir)
        let dd = try? plannedDrafterDir(dir, cfg)
        let fp = try plannedFootprint(dir: dir, cfg: cfg, drafterDir: dd)
        print(
            "footprint: weights \(gb(fp.weightBytes)), KV \(String(format: "%.1f", Double(fp.kvBytesPerToken) / 1000)) KB/token, "
                + "drafter \(fp.drafterBytes.map(gb) ?? "no"), snapshot \(gb(fp.snapshotBytes(drafter: fp.hasDrafter))), native context \(fp.nativeContext)"
        )
        do {
            let plan = try MemoryPlanner.plan(machine: m, model: fp, options: memOptions())
            print(plan.describe(), terminator: "")
        } catch {
            print("DOES NOT FIT: \(error)")
        }
    }
}
