import Foundation

/// MLP kind of a layer, deduced from the tensors present.
public enum MLPKind: CustomStringConvertible {
    case dense(intermediate: Int)
    case moe(experts: Int, intermediate: Int, shared: Int?)
    public var description: String {
        switch self {
        case .dense(let i): "dense (intermediate \(i))"
        case .moe(let e, let i, let s): "MoE (\(e) experts × \(i)\(s.map { ", shared \($0)" } ?? ""))"
        }
    }
}

public struct LayerDescription {
    public let index: Int
    public let prefix: String
    public let type: String
    public let mlp: MLPKind
    public let quantized: [QuantizedLinear]
}

/// Describes a layer: type (full or linear attention), MLP and all its quantized layers.
public func describeLayer(_ store: SafetensorsStore, config: ModelConfig, layer: Int) throws -> LayerDescription {
    let prefix = try store.layerPrefix(layer)
    let mods = Set(
        store.tensors.keys.filter { $0.hasPrefix(prefix) && $0.hasSuffix(".scales") }
            .map { String($0.dropLast(".scales".count)) })
    // gpt-oss names its experts `mlp.experts.` and stores them in MXFP4 (the only layers with MXFP4 kernels).
    let ql = try mods.sorted().map {
        try QuantizedLinear(store: store, path: $0, config: config, mxfp4: $0.contains("mlp.experts."))
    }
    let mlp: MLPKind
    let expertGates = ["mlp.switch_mlp.gate_proj", "mlp.experts.gate_proj"]
    if let sw = ql.first(where: { q in expertGates.contains { q.path.hasSuffix($0) } }) {
        let sh = ql.first(where: { $0.path.hasSuffix("mlp.shared_expert.gate_proj") })
        mlp = .moe(experts: sw.experts, intermediate: sw.outFeatures, shared: sh?.outFeatures)
    } else if let g = ql.first(where: { $0.path.hasSuffix("mlp.gate_proj") }) {
        mlp = .dense(intermediate: g.outFeatures)
    } else {
        throw SafetensorsError.missing("MLP of layer \(layer)")
    }
    let type = layer < config.layerTypes.count ? config.layerTypes[layer] : "?"
    return LayerDescription(index: layer, prefix: prefix, type: type, mlp: mlp, quantized: ql)
}
