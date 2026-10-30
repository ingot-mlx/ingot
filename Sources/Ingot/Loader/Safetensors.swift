import Foundation
import Metal

/// safetensors dtypes that we use.
public enum DType: String, Sendable {
    case bf16 = "BF16", f16 = "F16", f32 = "F32", u32 = "U32", i32 = "I32", u8 = "U8"

    public var size: Int {
        switch self {
        case .bf16, .f16: 2
        case .f32, .u32, .i32: 4
        case .u8: 1
        }
    }
}

public struct TensorInfo: Sendable {
    public let name: String
    public let dtype: DType
    public let shape: [Int]
    public let shard: Int
    /// Absolute offset in the file (header included).
    public let fileOffset: Int
    public let byteCount: Int

    public var count: Int { shape.reduce(1, *) }

    public init(name: String, dtype: DType, shape: [Int], shard: Int, fileOffset: Int, byteCount: Int) {
        self.name = name; self.dtype = dtype; self.shape = shape
        self.shard = shard; self.fileOffset = fileOffset; self.byteCount = byteCount
    }
}

public enum SafetensorsError: Error, CustomStringConvertible {
    case io(String), format(String), missing(String)
    public var description: String {
        switch self {
        case .io(let s): "I/O: \(s)"
        case .format(let s): "format: \(s)"
        case .missing(let s): "no such tensor \(s)"
        }
    }
}

/// A file mapped into memory with `mmap` (read-only, no copy).
/// The mapped length is rounded up to a page so it can be wrapped in a `MTLBuffer` without copying.
public final class MappedFile: @unchecked Sendable {
    public let path: String
    public let base: UnsafeMutableRawPointer
    public let fileSize: Int
    public let mappedLength: Int

    public init(path: String) throws {
        self.path = path
        let fd = open(path, O_RDONLY)
        guard fd >= 0 else { throw SafetensorsError.io("cannot open \(path)") }
        defer { close(fd) }
        var st = stat()
        guard fstat(fd, &st) == 0 else { throw SafetensorsError.io("fstat \(path)") }
        fileSize = Int(st.st_size)
        let page = Int(getpagesize())
        mappedLength = (fileSize + page - 1) / page * page
        // PROT_READ + MAP_PRIVATE: the pages are the system cache's (no copy until someone writes).
        guard let p = mmap(nil, mappedLength, PROT_READ, MAP_PRIVATE, fd, 0), p != MAP_FAILED else {
            throw SafetensorsError.io("mmap \(path)")
        }
        base = p
    }

    deinit { munmap(base, mappedLength) }
}

/// Safetensors loader (one or several shards with `model.safetensors.index.json`).
public final class SafetensorsStore: @unchecked Sendable {
    public let directory: URL
    public let files: [MappedFile]
    public private(set) var tensors: [String: TensorInfo] = [:]

    public init(directory: URL) throws {
        self.directory = directory
        let fm = FileManager.default
        let index = directory.appendingPathComponent("model.safetensors.index.json")
        var shardNames: [String]
        if fm.fileExists(atPath: index.path) {
            let data = try Data(contentsOf: index)
            guard let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                let map = obj["weight_map"] as? [String: String]
            else { throw SafetensorsError.format("index.json without weight_map") }
            shardNames = Array(Set(map.values)).sorted()
        } else {
            shardNames = ["model.safetensors"]
        }
        files = try shardNames.map { try MappedFile(path: directory.appendingPathComponent($0).path) }
        for (i, f) in files.enumerated() {
            try parseHeader(f, shard: i)
        }
    }

    /// Header: little-endian u64 with the length N, N bytes of JSON, and then the data.
    private func parseHeader(_ f: MappedFile, shard: Int) throws {
        guard f.fileSize >= 8 else { throw SafetensorsError.format("\(f.path) too short") }
        let n = Int(f.base.loadUnaligned(as: UInt64.self).littleEndian)
        guard 8 + n <= f.fileSize else { throw SafetensorsError.format("header out of range") }
        let json = Data(bytes: f.base + 8, count: n)
        guard let obj = try JSONSerialization.jsonObject(with: json) as? [String: Any] else {
            throw SafetensorsError.format("header is not a JSON object")
        }
        let dataStart = 8 + n
        for (name, v) in obj where name != "__metadata__" {
            guard let d = v as? [String: Any],
                let dt = d["dtype"] as? String, let dtype = DType(rawValue: dt),
                let shape = d["shape"] as? [Int], let offs = d["data_offsets"] as? [Int], offs.count == 2
            else { throw SafetensorsError.format("malformed entry: \(name)") }
            let info = TensorInfo(
                name: name, dtype: dtype, shape: shape, shard: shard,
                fileOffset: dataStart + offs[0], byteCount: offs[1] - offs[0])
            guard info.byteCount == info.count * dtype.size, dataStart + offs[1] <= f.fileSize else {
                throw SafetensorsError.format("inconsistent size in \(name)")
            }
            tensors[name] = info
        }
    }

    public func info(_ name: String) throws -> TensorInfo {
        guard let t = tensors[name] else { throw SafetensorsError.missing(name) }
        return t
    }

    /// Pointer to the tensor's bytes inside the mmap (may be unaligned).
    public func bytes(_ t: TensorInfo) -> UnsafeRawBufferPointer {
        UnsafeRawBufferPointer(start: files[t.shard].base + t.fileOffset, count: t.byteCount)
    }

    /// Natural alignment (power of 2, max 4096) of the tensor's address.
    public func alignment(_ t: TensorInfo) -> Int {
        let a = Int(bitPattern: files[t.shard].base) + t.fileOffset
        return min(a & -a, 4096)
    }
}

public struct DeviceTensor {
    public let info: TensorInfo
    public let buffer: MTLBuffer
    public let offset: Int
}

extension SafetensorsStore {
    /// Copies the tensor into a new page-aligned shared `MTLBuffer`.
    public func deviceTensor(_ name: String, device: MTLDevice) throws -> DeviceTensor {
        let t = try info(name)
        let src = bytes(t)
        guard let buf = device.makeBuffer(length: max(t.byteCount, 16), options: .storageModeShared) else {
            throw SafetensorsError.io("makeBuffer(length:) failed for \(name)")
        }
        buf.contents().copyMemory(from: src.baseAddress!, byteCount: t.byteCount)
        return DeviceTensor(info: t, buffer: buf, offset: 0)
    }
}

extension SafetensorsStore {
    /// Parts of a pack the engine never loads: the MTP head and the vision/audio towers that mlx-lm and mlx-vlm drop
    /// for text (`visual` in Qwen3.6, `vision_tower` in the mlx-community Qwen3.5, the Gemma 4 towers and embedders, Muse Glimmer's
    /// tower, adapter and projection).
    /// The loader (`layerPrefix`) and the memory plan (`ModelFootprint`) skip the same list.
    static let skippedParts = [
        "mtp", "visual", "vision_tower", "audio_tower", "embed_vision", "embed_audio", "vision_embedder",
        "multi_modal_projector", "vision_adapter", "vision_projection",
    ]

    /// The tensor belongs to a part the engine never loads (`skippedParts`).
    static func isSkipped(_ name: String) -> Bool { skippedParts.contains { name.contains($0) } }

    /// Finds the module prefix of layer `layer` (e.g. `language_model.model.layers.10.`), without assuming
    /// the architecture: works for `model.layers.N.` and `language_model.model.layers.N.`. Deterministic and the
    /// language model first: skips the towers and the MTP head (`skippedParts`), prefers a prefix under
    /// `language_model`, then the shortest, then the first in order.
    public func layerPrefix(_ layer: Int) throws -> String {
        let key = "layers.\(layer)."
        let cands = Set(
            tensors.keys.compactMap { name -> String? in
                guard let r = name.range(of: key), !Self.isSkipped(name) else { return nil }
                return String(name[..<r.upperBound])
            })
        guard let p = Self.preferredPrefix(cands) else { throw SafetensorsError.missing("layer \(layer)") }
        return p
    }

    /// The prefix `layerPrefix` picks among the candidates: under `language_model` first, then the shortest, then
    /// the first in lexicographic order (never the order of a `Set`).
    static func preferredPrefix(_ cands: Set<String>) -> String? {
        func key(_ s: String) -> (Int, Int) { (s.contains("language_model") ? 0 : 1, s.count) }
        return cands.min { key($0) != key($1) ? key($0) < key($1) : $0 < $1 }
    }
}
