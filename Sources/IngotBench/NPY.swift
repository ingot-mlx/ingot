import Foundation
import Ingot

/// Minimal reading and writing of `.npy` files (v1/v2/v3, little-endian, C order).
public struct NPYArray {
    public enum Kind: String { case f2 = "<f2", f4 = "<f4", u4 = "<u4", i4 = "<i4", u2 = "<u2" }
    public let kind: Kind
    public let shape: [Int]
    public let data: Data

    public init(kind: Kind, shape: [Int], data: Data) {
        self.kind = kind; self.shape = shape; self.data = data
    }

    public init(contentsOf url: URL) throws {
        let d = try Data(contentsOf: url, options: .alwaysMapped)
        guard d.count >= 10, d.prefix(6) == Data([0x93] + Array("NUMPY".utf8)) else {
            throw SafetensorsError.format("\(url.lastPathComponent) no es .npy")
        }
        let major = d[6]
        let hlen: Int, hstart: Int
        if major == 1 {
            hlen = Int(d[8]) | Int(d[9]) << 8; hstart = 10
        } else {
            hlen = Int(d[8]) | Int(d[9]) << 8 | Int(d[10]) << 16 | Int(d[11]) << 24; hstart = 12
        }
        let header = String(decoding: d[hstart..<hstart + hlen], as: UTF8.self)
        func field(_ k: String) -> String? {
            guard let r = header.range(of: "'\(k)':") else { return nil }
            return String(header[r.upperBound...]).trimmingCharacters(in: .whitespaces)
        }
        guard let descr = field("descr"), let q1 = descr.firstIndex(of: "'"),
            let q2 = descr[descr.index(after: q1)...].firstIndex(of: "'"),
            let kind = Kind(rawValue: String(descr[descr.index(after: q1)..<q2]))
        else { throw SafetensorsError.format("unsupported .npy dtype: \(header)") }
        if let fo = field("fortran_order"), fo.hasPrefix("True") {
            throw SafetensorsError.format(".npy in Fortran order")
        }
        guard let sh = field("shape"), let o = sh.firstIndex(of: "("), let c = sh.firstIndex(of: ")") else {
            throw SafetensorsError.format("shape .npy ilegible")
        }
        let dims = sh[sh.index(after: o)..<c].split(separator: ",").compactMap {
            Int($0.trimmingCharacters(in: .whitespaces))
        }
        self.kind = kind
        self.shape = dims
        self.data = d.subdata(in: hstart + hlen..<d.count)
    }

    public func write(to url: URL) throws {
        let shapeStr = shape.count == 1 ? "(\(shape[0]),)" : "(" + shape.map(String.init).joined(separator: ", ") + ")"
        var h = "{'descr': '\(kind.rawValue)', 'fortran_order': False, 'shape': \(shapeStr), }"
        let total = 10 + h.utf8.count + 1
        h += String(repeating: " ", count: (64 - total % 64) % 64) + "\n"
        var out = Data([0x93] + Array("NUMPY".utf8) + [1, 0])
        let n = UInt16(h.utf8.count)
        out.append(contentsOf: [UInt8(n & 0xFF), UInt8(n >> 8)])
        out.append(contentsOf: Array(h.utf8))
        out.append(data)
        try out.write(to: url)
    }

    /// Values as Float (converts from f2).
    public func floats() -> [Float] {
        switch kind {
        case .f4:
            return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        case .f2:
            return data.withUnsafeBytes { $0.bindMemory(to: Float16.self).map(Float.init) }
        default:
            fatalError("floats() sobre \(kind)")
        }
    }
}
