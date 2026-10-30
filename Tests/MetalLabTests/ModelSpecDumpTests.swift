import Foundation
import LabBench
import Testing

@testable import MetalLab

/// The ModelSpec of every real config in ref/toy/real/, field by field, against ref/toy/model-spec-dump.json: a dump
/// of the specs made with the code before the family table, so moving what the family decides into
/// FamilyTraits cannot change a value unnoticed. A config that does not load records its error instead. No GPU.
/// Adding or changing a family on purpose: regenerate the file with `record` once and review its diff.
@Suite struct ModelSpecDumpTests {
    static let dumpPath = "ref/toy/model-spec-dump.json"
    /// true writes the current dump over the file (then back to false).
    static let record = false

    /// Every value of the spec as JSON: structs and enum payloads as objects keyed by field, optionals as null, leaves
    /// with `String(describing:)` (floats keep their shortest exact form). Sets and dictionaries sorted.
    static func dump(_ value: Any) -> Any {
        let m = Mirror(reflecting: value)
        switch m.displayStyle {
        case .optional:
            return m.children.first.map { dump($0.value) } ?? NSNull()
        case .collection, .tuple:
            return m.children.map { dump($0.value) }
        case .set:
            return m.children.map { "\(dump($0.value))" }.sorted()
        case .dictionary:
            return m.children.map { "\(dump($0.value))" }.sorted()
        case .struct, .class:
            var out: [String: Any] = [:]
            for (i, c) in m.children.enumerated() { out[c.label ?? "\(i)"] = dump(c.value) }
            // Computed properties the engine reads as fields.
            if let s = value as? ModelSpec {
                out["hasDrafter"] = "\(s.hasDrafter)"
                out["calibrationName"] = s.family.calibrationName
            }
            return out
        case .enum:
            guard let payload = m.children.first else { return "\(value)" }
            return [payload.label ?? "\(value)": dump(payload.value)]
        default:
            return "\(value)"
        }
    }

    /// The dump of each real config by name; the load error if it does not load.
    static func current() throws -> [String: Any] {
        let dir = ModelSpecTests.repo.appendingPathComponent("ref/toy/real")
        let names = try FileManager.default.contentsOfDirectory(atPath: dir.path).filter { $0.hasSuffix(".json") }
            .map { String($0.dropLast(5)) }.sorted()
        var out: [String: Any] = [:]
        for name in names {
            let json = try ToyReal.load(name, root: ModelSpecTests.repo).json
            do {
                var d = try #require(dump(try ModelSpecTests.spec(json)) as? [String: Any])
                // What the family decides outside the spec (read from the config by the loader).
                let cfg = try ModelConfig(json: json)
                d["smallWeightsF32"] = "\(ModelFamily.smallWeightsF32(cfg))"
                d["attnSinks"] = "\(ModelSpec.attnSinks(cfg))"
                d["hasScaledRope"] = "\(ModelSpec.hasScaledRope(cfg))"
                out[name] = d
            } catch {
                out[name] = ["loadError": "\(error)"]
            }
        }
        return out
    }

    /// Paths where two dumps differ ("" for the root), at most `limit`.
    static func differences(_ a: Any, _ b: Any, path: String = "", limit: Int = 20) -> [String] {
        switch (a, b) {
        case (let x as [String: Any], let y as [String: Any]):
            var out: [String] = []
            for k in Set(x.keys).union(y.keys).sorted() where out.count < limit {
                guard let xv = x[k], let yv = y[k] else {
                    out.append("\(path).\(k): only in \(x[k] == nil ? "current" : "dump")")
                    continue
                }
                out += differences(xv, yv, path: "\(path).\(k)", limit: limit - out.count)
            }
            return out
        case (let x as [Any], let y as [Any]):
            guard x.count == y.count else { return ["\(path): \(x.count) vs \(y.count) elements"] }
            var out: [String] = []
            for (i, (xv, yv)) in zip(x, y).enumerated() where out.count < limit {
                out += differences(xv, yv, path: "\(path)[\(i)]", limit: limit - out.count)
            }
            return out
        default:
            return "\(a)" == "\(b)" ? [] : ["\(path): \(a) (dump) vs \(b) (current)"]
        }
    }

    @Test func realConfigsMatchDump() throws {
        let url = ModelSpecTests.repo.appendingPathComponent(Self.dumpPath)
        // Round-trip through JSON so both sides hold the same types.
        let data = try JSONSerialization.data(
            withJSONObject: try Self.current(), options: [.sortedKeys, .prettyPrinted])
        let now = try JSONSerialization.jsonObject(with: data)
        if Self.record { try data.write(to: url) }
        let saved = try JSONSerialization.jsonObject(with: Data(contentsOf: url))
        let diffs = Self.differences(saved, now)
        #expect(diffs.isEmpty, "\(diffs.joined(separator: "\n"))")
    }
}
