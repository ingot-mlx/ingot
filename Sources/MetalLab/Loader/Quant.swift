import Foundation
import Metal

/// MLX affine format (see `affine_quantize`/`affine_dequantize` in mlx/backend/metal/kernels/quantized.h, MIT):
///   - `weight`: uint32 `[..., out, in * bits / 32]`. With 4 bits each uint32 holds 8 values, element `j` in the
///     bits `4*j ... 4*j+3` (low nibble first; in bytes: byte `j/2`, low nibble if `j` is even).
///   - `scales`, `biases`: `[..., out, in / group_size]` (bf16 in these models).
///   - `w = scale * q + bias`.
/// MLX mxfp4 format (`mode: "mxfp4"`, mlx/backend/metal/kernels/fp4.h and fp_quantized.h, MIT): the same 4-bit packing,
/// but each code is an e2m1 float (sign in bit 3, magnitudes 0, 0.5, 1, 1.5, 2, 3, 4, 6), `scales` are uint8 e8m0
/// exponents `[..., out, in / 32]` (`2^(s − 127)`) and there are no `biases`: `w = 2^(s − 127) · fp4(q)`.
/// The leading dimensions (`...`) are the stacked experts of `switch_mlp` in the MoEs.
public struct QuantizedLinear: Sendable {
    public let path: String
    public let weight: TensorInfo
    public let scales: TensorInfo
    public let biases: TensorInfo?
    public let params: QuantParams
    /// Number of stacked experts (1 for a normal layer).
    public let experts: Int
    public let outFeatures: Int
    public let inFeatures: Int
    /// MXFP4 layer (`params.mode == "mxfp4"`): uint8 e8m0 scales, no biases.
    public var isMXFP4: Bool { params.mode == "mxfp4" }

    /// Builds the layer by reading the shapes; the bits are deduced from them and checked against the config.
    /// `mxfp4`: the caller has MXFP4 kernels (the MoE experts); elsewhere an MXFP4 layer is a load error.
    /// `multiLinear`: a per-head MultiLinear of mlx-lm (GLM's `embed_q`, `unembed_out`), quantized as [heads, out, in]
    /// with the groups along `in`, read as one matrix of heads · out rows (head h's rows h·out ..< (h + 1)·out), which is
    /// the same bytes; without it, the leading dimensions are stacked experts.
    public init(
        store: SafetensorsStore, path: String, config: ModelConfig?, mxfp4: Bool = false, multiLinear: Bool = false
    ) throws {
        self.path = path
        weight = try store.info(path + ".weight")
        scales = try store.info(path + ".scales")
        biases = store.tensors[path + ".biases"]
        guard weight.dtype == .u32, weight.shape.count >= 2, scales.shape.count == weight.shape.count else {
            throw SafetensorsError.format("\(path) does not look like an MLX quantized layer")
        }
        let packed = weight.shape.last!, groups = scales.shape.last!
        if multiLinear {
            guard weight.shape.count == 3 else {
                throw SafetensorsError.format(
                    "\(path): a MultiLinear is a 3D quantized tensor [heads, out, in] (shape \(weight.shape))")
            }
            outFeatures = weight.shape[0] * weight.shape[1]
            experts = 1
        } else {
            outFeatures = weight.shape[weight.shape.count - 2]
            experts = weight.shape.dropLast(2).reduce(1, *)
        }
        let declared = config?.quant(for: path)
        // Affine, or mxfp4 where the caller accepts it: another mode would be read as affine with bf16 scales without
        // a word.
        if let mode = declared?.mode, mode != "affine" && !(mode == "mxfp4" && mxfp4) {
            throw SafetensorsError.format(Self.modeMessage(path: path, mode: mode))
        }
        if declared?.mode == "mxfp4" {
            inFeatures = groups * 32
            guard declared?.bits == 4, declared?.groupSize == 32, packed * 8 == inFeatures, scales.dtype == .u8,
                biases == nil
            else {
                throw SafetensorsError.format(
                    "\(path): not an MLX mxfp4 layer (4 bits g32, uint8 scales, no biases; packed \(packed), groups "
                        + "\(groups), scales \(scales.dtype))")
            }
            params = QuantParams(bits: 4, groupSize: 32, mode: "mxfp4")
            return
        }
        let gs = declared?.groupSize ?? 64
        inFeatures = groups * gs
        let bits = packed * 32 / inFeatures
        guard packed * 32 == bits * inFeatures, [2, 3, 4, 5, 6, 8].contains(bits) else {
            throw SafetensorsError.format("\(path): bits do not add up (packed \(packed), groups \(groups))")
        }
        if let declared, declared.bits != bits {
            throw SafetensorsError.format("\(path): config says \(declared.bits) bits, the shapes say \(bits)")
        }
        params = QuantParams(bits: bits, groupSize: gs)
    }

    /// Load error of a quantization mode the engine cannot read here.
    static func modeMessage(path: String, mode: String) -> String {
        "\(path): quantization mode «\(mode)» not supported (affine only"
            + (mode == "mxfp4" ? "; MXFP4 runs only in the MoE experts)" : ")")
    }
}

extension ModelSpec {
    /// Quantization modes the engine cannot run: anything but affine goes by key, whatever the family, as its weights
    /// would be read as affine. MXFP4 (4-bit e2m1 codes with e8m0 scales per group of 32) runs in the
    /// gpt-oss experts only (the MLX conversions put it there and nowhere else; another layer in MXFP4 is a load error
    /// in QuantizedLinear).
    static func quantModeGaps(_ cfg: ModelConfig) -> [String] {
        var out: [String] = []
        let quants = [cfg.quantDefault].compactMap { $0 } + cfg.quantOverrides.values
        for m in Set(quants.map(\.mode)).subtracting(["affine"]).sorted() {
            guard m == "mxfp4" else {
                out.append("quantization mode «\(m)» (affine only)")
                continue
            }
            let odd = quants.filter { $0.mode == m && ($0.bits != 4 || $0.groupSize != 32) }
            if cfg.family == .nemotronH {
                out.append(
                    "MXFP4 quantization in nemotron_h (not supported: only the gpt-oss experts run it)")
            } else if cfg.family != .gptOss {
                out.append("MXFP4 quantization outside gpt-oss (only its experts have MXFP4 kernels)")
            } else if let q = odd.first {
                out.append("MXFP4 quantization with \(q.bits) bits g\(q.groupSize) (it is 4 bits g32)")
            }
        }
        return out
    }
}

// MARK: - CPU utilities (reference and tests)

@inline(__always) public func bf16ToFloat(_ v: UInt16) -> Float { Float(bitPattern: UInt32(v) << 16) }

@inline(__always) public func floatToBF16(_ f: Float) -> UInt16 {
    // Round to nearest even (NaN is not handled: it does not appear in weights).
    let u = f.bitPattern
    return UInt16(truncatingIfNeeded: (u &+ 0x7FFF &+ ((u >> 16) & 1)) >> 16)
}

/// Quantized value `j` (0..<in) of a packed row (bits ∈ {2,4,8}; 3, 5 and 6 as a continuous little-endian
/// bit stream, which is how MLX packs: value j in bits bits·j … of the row).
@inline(__always) public func unpackQ(_ row: UnsafePointer<UInt32>, _ j: Int, bits: Int) -> UInt32 {
    if 32 % bits != 0 {
        let bit = bits * j, w = bit / 32, o = bit % 32
        var v = UInt64(row[w]) >> UInt64(o)
        if o + bits > 32 { v |= UInt64(row[w + 1]) << UInt64(32 - o) }
        return UInt32(v & UInt64((1 << bits) - 1))
    }
    let per = 32 / bits
    return (row[j / per] >> UInt32(bits * (j % per))) & UInt32((1 << bits) - 1)
}

/// Packs integer values into uint32 in MLX order (for tests).
public func packQ(_ q: [UInt32], bits: Int) -> [UInt32] {
    let per = 32 / bits
    precondition(q.count % per == 0)
    var out = [UInt32](repeating: 0, count: q.count / per)
    for (j, v) in q.enumerated() { out[j / per] |= (v & UInt32((1 << bits) - 1)) << UInt32(bits * (j % per)) }
    return out
}

extension SafetensorsStore {
    /// Copies a tensor into an aligned array (the mmap data may be unaligned).
    public func array<T>(_ t: TensorInfo, as _: T.Type) -> [T] {
        precondition(t.byteCount % MemoryLayout<T>.stride == 0)
        let n = t.byteCount / MemoryLayout<T>.stride
        let src = bytes(t)
        return [T](unsafeUninitializedCapacity: n) { buf, count in
            UnsafeMutableRawBufferPointer(buf).copyMemory(from: src)
            count = n
        }
    }

    /// Dequantizes on the CPU rows `rows` of one expert of a layer (fp32), affine or mxfp4. For tests and checks.
    public func dequantizeRows(_ l: QuantizedLinear, expert: Int = 0, rows: Range<Int>) -> [Float] {
        let w = array(l.weight, as: UInt32.self)
        let s8 = l.isMXFP4 ? array(l.scales, as: UInt8.self) : []
        let s = l.isMXFP4 ? [] : array(l.scales, as: UInt16.self)
        let b = l.biases.map { array($0, as: UInt16.self) }
        let packedPerRow = l.inFeatures * l.params.bits / 32
        let groupsPerRow = l.inFeatures / l.params.groupSize
        var out = [Float](repeating: 0, count: rows.count * l.inFeatures)
        w.withUnsafeBufferPointer { wp in
            for (ri, r) in rows.enumerated() {
                let gr = expert * l.outFeatures + r
                let row = wp.baseAddress! + gr * packedPerRow
                for j in 0..<l.inFeatures {
                    let g = gr * groupsPerRow + j / l.params.groupSize
                    let q = unpackQ(row, j, bits: l.params.bits)
                    if l.isMXFP4 {
                        out[ri * l.inFeatures + j] = e8m0ToFloat(s8[g]) * fp4E2M1ToFloat(q)
                        continue
                    }
                    let sc = bf16ToFloat(s[g]), bi = b.map { bf16ToFloat($0[g]) } ?? 0
                    out[ri * l.inFeatures + j] = sc * Float(q) + bi
                }
            }
        }
        return out
    }
}

/// e2m1 code (low 4 bits: sign in bit 3, then 2 exponent bits and 1 mantissa bit, exponent bias 1) as a float:
/// ±{0, 0.5, 1, 1.5, 2, 3, 4, 6} (mlx fp4.h `fp4_e2m1`).
@inline(__always) public func fp4E2M1ToFloat(_ q: UInt32) -> Float {
    let e = Int((q >> 1) & 3), m = Float(q & 1)
    let mag = e == 0 ? 0.5 * m : Float(sign: .plus, exponent: e - 1, significand: 1 + 0.5 * m)
    return q & 8 != 0 ? -mag : mag
}

/// e8m0 scale (an exponent with bias 127, no sign or mantissa) as a float: `2^(s − 127)` (255 is NaN; 0 is 2⁻¹²⁷,
/// which fp32 holds only as a subnormal).
@inline(__always) public func e8m0ToFloat(_ s: UInt8) -> Float {
    s == 255 ? .nan : Float(sign: .plus, exponent: Int(s) - 127, significand: 1)
}

/// Weights of a quantized layer on the GPU (aligned copies).
public struct DeviceQuantizedLinear {
    public let layer: QuantizedLinear
    public let w: DeviceTensor
    public let s: DeviceTensor
    public let b: DeviceTensor

    public init(_ l: QuantizedLinear, store: SafetensorsStore, device: MTLDevice) throws {
        layer = l
        if l.isMXFP4 {
            // No biases: `b` is the scales tensor again, bound where the kernels' bias buffer goes and never read.
            let w = try store.deviceTensor(l.weight.name, device: device)
            let s = try store.deviceTensor(l.scales.name, device: device)
            self.w = w; self.s = s; b = s
            return
        }
        guard let bi = l.biases else { throw SafetensorsError.missing(l.path + ".biases") }
        // fp16 scales/biases are converted to bf16 only in DenseQ; here they would be read as bf16.
        guard l.params.bits == 2 || (l.scales.dtype == .bf16 && bi.dtype == .bf16) else {
            throw SafetensorsError.format(
                "\(l.path): \(l.scales.dtype) scales/biases are not supported here (bf16 only)")
        }
        let w = try store.deviceTensor(l.weight.name, device: device)
        let s = try store.deviceTensor(l.scales.name, device: device)
        let b = try store.deviceTensor(bi.name, device: device)
        self.w = w; self.s = s; self.b = b
    }
}

// MARK: - Synthetic layers (tests and selftest)

extension QuantizedLinear {
    /// A layer with no file behind it, with groups of 64.
    public init(
        synthetic path: String, weight: TensorInfo, scales: TensorInfo, biases: TensorInfo?, groupSize: Int = 64
    ) throws {
        self.path = path
        self.weight = weight; self.scales = scales; self.biases = biases
        outFeatures = weight.shape[weight.shape.count - 2]
        experts = weight.shape.dropLast(2).reduce(1, *)
        inFeatures = scales.shape.last! * groupSize
        let bits = weight.shape.last! * 32 / inFeatures
        params = QuantParams(bits: bits, groupSize: groupSize)
    }
}

extension DeviceQuantizedLinear {
    public init(synthetic l: QuantizedLinear, w: MTLBuffer, s: MTLBuffer, b: MTLBuffer) {
        layer = l
        self.w = DeviceTensor(info: l.weight, buffer: w, offset: 0)
        self.s = DeviceTensor(info: l.scales, buffer: s, offset: 0)
        self.b = DeviceTensor(info: l.biases ?? l.scales, buffer: b, offset: 0)
    }
}
