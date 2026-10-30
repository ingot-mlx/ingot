import Foundation

// Pieces of the toy-model generator (ToyModels.swift): deterministic pseudo-random generator, safetensors writer
// in MLX's layout (U32 weights + BF16 scales and biases) and a small BPE tokenizer trained on
// ref/agent_dump. Everything comes from the seed: same seed and same code, same bytes.

/// xoshiro256** with the seed expanded by splitmix64. Each tensor has its own generator (seed ⊕ hash of the
/// name), so a tensor's content does not depend on the order in which the others are written.
struct ToyRNG {
    var s0: UInt64, s1: UInt64, s2: UInt64, s3: UInt64

    init(seed: UInt64) {
        var x = seed
        func sm() -> UInt64 {
            x &+= 0x9E37_79B9_7F4A_7C15
            var z = x
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            return z ^ (z >> 31)
        }
        s0 = sm(); s1 = sm(); s2 = sm(); s3 = sm()
    }

    init(seed: UInt64, name: String) {
        var h: UInt64 = 0xCBF2_9CE4_8422_2325  // FNV-1a
        for b in name.utf8 { h = (h ^ UInt64(b)) &* 0x100_0000_01B3 }
        self.init(seed: seed ^ h)
    }

    mutating func next() -> UInt64 {
        let r = ((s1 &* 5) << 7 | (s1 &* 5) >> 57) &* 9
        let t = s1 << 17
        s2 ^= s0; s3 ^= s1; s1 ^= s2; s0 ^= s3; s2 ^= t
        s3 = s3 << 45 | s3 >> 19
        return r
    }

    /// Uniform in [0, 1).
    mutating func uniform() -> Float { Float(next() >> 40) * (1.0 / 16_777_216.0) }

    /// Standard normal (Box-Muller).
    mutating func normal() -> Float {
        let u = max(uniform(), 1e-7), v = uniform()
        return (-2 * log(u)).squareRoot() * cos(2 * Float.pi * v)
    }

    mutating func sign() -> Float { next() & 1 == 0 ? 1 : -1 }

    /// Fills `p` with random bytes (quantized weight codes: any bit pattern is valid).
    mutating func fill(_ p: UnsafeMutableRawBufferPointer) {
        let n = p.count / 8
        let w = p.baseAddress!.bindMemory(to: UInt64.self, capacity: n)
        for i in 0..<n { w[i] = next() }
        for i in (n * 8)..<p.count { p[i] = UInt8(truncatingIfNeeded: next()) }
    }
}

/// Float → bfloat16 with round-to-nearest-even (like MLX).
func toyBF16(_ f: Float) -> UInt16 {
    let b = f.bitPattern
    return UInt16((b &+ 0x7FFF &+ ((b >> 16) & 1)) >> 16)
}

/// Writer of a safetensors file (JSON header with `{"format": "mlx"}` and the data right after, no gaps). Each
/// tensor is generated as it is written, one at a time, so the whole file is never held in memory.
final class ToySafetensors {
    enum DType: String {
        case u32 = "U32", bf16 = "BF16", f32 = "F32", u8 = "U8"
        var size: Int {
            switch self {
            case .u8: 1
            case .bf16: 2
            case .u32, .f32: 4
            }
        }
    }
    struct Entry {
        let name: String, dtype: DType, shape: [Int]
        let fill: (UnsafeMutableRawBufferPointer) -> Void
        var bytes: Int { shape.reduce(1, *) * dtype.size }
    }
    let seed: UInt64
    private(set) var entries: [Entry] = []
    /// Group size of the 4-bit matrices without an explicit `group` (32 in toy-granite, as in the real one); for 8-bit ones, 64.
    var group4 = 64
    init(seed: UInt64) { self.seed = seed }

    func add(_ name: String, _ dtype: DType, _ shape: [Int], _ fill: @escaping (UnsafeMutableRawBufferPointer) -> Void)
    {
        precondition(!entries.contains { $0.name == name }, "duplicate tensor: \(name)")
        entries.append(Entry(name: name, dtype: dtype, shape: shape, fill: fill))
    }

    /// Floating-point tensor with each value drawn from `gen` (the tensor's own generator).
    func float(_ name: String, _ dtype: DType, _ shape: [Int], _ gen: @escaping (inout ToyRNG) -> Float) {
        let seed = seed
        add(name, dtype, shape) { p in
            var r = ToyRNG(seed: seed, name: name)
            let n = shape.reduce(1, *)
            if dtype == .bf16 {
                let q = p.baseAddress!.bindMemory(to: UInt16.self, capacity: n)
                for i in 0..<n { q[i] = toyBF16(gen(&r)) }
            } else {
                let q = p.baseAddress!.bindMemory(to: Float.self, capacity: n)
                for i in 0..<n { q[i] = gen(&r) }
            }
        }
    }

    /// MLX affine-quantized linear: `name.weight` U32 [E?, N, K·bits/32], `name.scales` and `name.biases` BF16
    /// [E?, N, K/group]. Uniform codes; the scale gives weights with deviation ≈ gain/√K and the bias centres them
    /// (−scale·(2^bits − 1)/2, like a symmetric MLX quantization).
    func quant(
        _ name: String, rows n: Int, cols k: Int, bits: Int = 4, group: Int? = nil, experts e: Int? = nil,
        gain: Float = 1, boost: (rows: Set<Int>, factor: Float)? = nil
    ) {
        let group = group ?? (bits == 4 ? group4 : 64)
        precondition(k % group == 0 && (k * bits) % 32 == 0, "\(name): K = \(k) does not fit \(bits) bits g\(group)")
        let lead = e.map { [$0] } ?? []
        let seed = seed
        add(name + ".weight", .u32, lead + [n, k * bits / 32]) { p in
            var r = ToyRNG(seed: seed, name: name + ".weight")
            r.fill(p)
        }
        let levels = Float((1 << bits) - 1)
        let span: Float = (levels + 1) * (levels + 1) - 1
        let lvlStd: Float = (span / 12).squareRoot()
        let base = gain / Float(k).squareRoot() / lvlStd
        let g = (e ?? 1) * n * (k / group)
        // The biases depend on the scales: both are drawn from the scales' generator.
        let makeScales = { () -> [Float] in
            var r = ToyRNG(seed: seed, name: name + ".scales")
            let gpr = k / group
            return (0..<g).map { i in
                let s = base * max(0.25, 1 + 0.25 * r.normal())
                guard let b = boost, b.rows.contains((i / gpr) % n) else { return s }
                return s * b.factor
            }
        }
        let shape = lead + [n, k / group]
        add(name + ".scales", .bf16, shape) { p in
            let q = p.baseAddress!.bindMemory(to: UInt16.self, capacity: g)
            for (i, s) in makeScales().enumerated() { q[i] = toyBF16(s) }
        }
        add(name + ".biases", .bf16, shape) { p in
            var r = ToyRNG(seed: seed, name: name + ".biases")
            let q = p.baseAddress!.bindMemory(to: UInt16.self, capacity: g)
            for (i, s) in makeScales().enumerated() { q[i] = toyBF16(-s * (levels / 2 + 0.3 * r.normal())) }
        }
    }

    /// MLX mxfp4-quantized linear (gpt-oss experts): `name.weight` U32 [E?, N, K/8] of e2m1 codes and `name.scales` U8
    /// [E?, N, K/32] of e8m0 exponents, no biases. Written directly, without a quantizer: uniform codes (deviation of
    /// the e2m1 values ≈ 2.93) and per-group exponents around the one that gives weights of deviation ≈ gain/√K, ±1
    /// at random, clamped to the range of the real gpt-oss-20b's scales (2⁻¹¹ … 2⁸).
    func mxfp4(_ name: String, rows n: Int, cols k: Int, experts e: Int? = nil, gain: Float = 1) {
        precondition(k % 32 == 0, "\(name): K = \(k) is not a multiple of 32")
        let lead = e.map { [$0] } ?? []
        let seed = seed
        add(name + ".weight", .u32, lead + [n, k / 8]) { p in
            var r = ToyRNG(seed: seed, name: name + ".weight")
            r.fill(p)
        }
        let center = (log2(gain / Float(k).squareRoot() / 2.93)).rounded()
        let g = (e ?? 1) * n * (k / 32)
        add(name + ".scales", .u8, lead + [n, k / 32]) { p in
            var r = ToyRNG(seed: seed, name: name + ".scales")
            for i in 0..<g {
                let x = center + Float(Int(r.next() % 3) - 1)
                p[i] = UInt8(127 + min(max(x, -11), 8))
            }
        }
    }

    func write(to url: URL) throws {
        var off = 0
        var parts: [String] = [#""__metadata__":{"format":"mlx"}"#]
        for e in entries {
            let shape = e.shape.map(String.init).joined(separator: ",")
            parts.append(
                #""\#(e.name)":{"dtype":"\#(e.dtype.rawValue)","shape":[\#(shape)],"data_offsets":[\#(off),\#(off + e.bytes)]}"#
            )
            off += e.bytes
        }
        var header = Array(("{" + parts.joined(separator: ",") + "}").utf8)
        while header.count % 8 != 0 { header.append(0x20) }
        FileManager.default.createFile(atPath: url.path, contents: nil)
        let fh = try FileHandle(forWritingTo: url)
        defer { try? fh.close() }
        var len = UInt64(header.count).littleEndian
        try fh.write(contentsOf: Data(bytes: &len, count: 8))
        try fh.write(contentsOf: Data(header))
        for e in entries {
            let buf = UnsafeMutableRawBufferPointer.allocate(byteCount: e.bytes, alignment: 16)
            defer { buf.deallocate() }
            e.fill(buf)
            try fh.write(contentsOf: Data(bytesNoCopy: buf.baseAddress!, count: e.bytes, deallocator: .none))
        }
    }
}

/// Byte-level BPE tokenizer (like Qwen's: Split with its regex + ByteLevel) trained on a small corpus,
/// with Qwen3.6's special tokens at the same positions relative to the end of the vocabulary.
enum ToyTokenizer {
    /// Qwen3.6's added tokens (ids 248044…248076) and whether they are special.
    static let qwenAdded: [(String, Bool)] = [
        ("<|endoftext|>", true), ("<|im_start|>", true), ("<|im_end|>", true), ("<|object_ref_start|>", true),
        ("<|object_ref_end|>", true), ("<|box_start|>", true), ("<|box_end|>", true), ("<|quad_start|>", true),
        ("<|quad_end|>", true), ("<|vision_start|>", true), ("<|vision_end|>", true), ("<|vision_pad|>", true),
        ("<|image_pad|>", true), ("<|video_pad|>", true), ("<tool_call>", false), ("</tool_call>", false),
        ("<|fim_prefix|>", false), ("<|fim_middle|>", false), ("<|fim_suffix|>", false), ("<|fim_pad|>", false),
        ("<|repo_name|>", false), ("<|file_sep|>", false), ("<tool_response>", false), ("</tool_response>", false),
        ("<think>", false), ("</think>", false), ("<|audio_start|>", true), ("<|audio_end|>", true),
        ("<tts_pad>", true), ("<tts_text_bos>", true), ("<tts_text_eod>", true), ("<tts_text_bos_single>", true),
        ("<|audio_pad|>", true),
    ]
    /// Toy id for a Qwen3.6 id ≥ 248044 (same distance to the end of a 248320 vocabulary).
    static func id(qwen q: Int, vocab: Int) -> Int { q - 248_320 + vocab }

    static let splitRegex =
        #"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#

    /// GPT-2 tables: byte → character.
    static let byteChar: [String] = {
        var out = [String](repeating: "", count: 256)
        var n = 0
        for b in 0..<256 {
            let keep = (33...126).contains(b) || (161...172).contains(b) || (174...255).contains(b)
            out[b] = String(Character(Unicode.Scalar(UInt32(keep ? b : 256 + n))!))
            if !keep { n += 1 }
        }
        return out
    }()

    /// tokenizer.json with `merges` merges learned over `corpus` and padding up to `vocab` ids.
    static func json(corpus: String, merges nMerges: Int, vocab: Int) throws -> Data {
        // Pre-tokenized words and their frequency.
        let re = try NSRegularExpression(pattern: splitRegex)
        let ns = corpus as NSString
        var freq: [String: Int] = [:]
        for m in re.matches(in: corpus, range: NSRange(location: 0, length: ns.length)) {
            freq[ns.substring(with: m.range), default: 0] += 1
        }
        var words: [[Int32]] = [], counts: [Int] = []
        for (w, c) in freq.sorted(by: { $0.key < $1.key }) {
            words.append(Array(w.utf8).map { Int32($0) }); counts.append(c)
        }
        var tok = byteChar
        var merges: [String] = []
        for _ in 0..<nMerges {
            var pc: [UInt64: Int] = [:]
            for (w, c) in zip(words, counts) where w.count > 1 {
                for i in 0..<(w.count - 1) {
                    pc[UInt64(UInt32(w[i])) << 32 | UInt64(UInt32(w[i + 1])), default: 0] += c
                }
            }
            // The most frequent pair; on a tie, the one with the lowest ids (deterministic).
            guard let best = pc.max(by: { $0.value < $1.value || ($0.value == $1.value && $0.key > $1.key) }),
                best.value >= 2
            else { break }
            let a = Int32(best.key >> 32), b = Int32(best.key & 0xFFFF_FFFF), nid = Int32(tok.count)
            merges.append(tok[Int(a)] + " " + tok[Int(b)])
            tok.append(tok[Int(a)] + tok[Int(b)])
            for j in words.indices where words[j].count > 1 {
                var w = words[j], o: [Int32] = [], i = 0
                guard zip(w, w.dropFirst()).contains(where: { $0 == a && $1 == b }) else { continue }
                while i < w.count {
                    if i + 1 < w.count && w[i] == a && w[i + 1] == b {
                        o.append(nid); i += 2
                    } else {
                        o.append(w[i]); i += 1
                    }
                }
                w = o; words[j] = w
            }
        }
        let firstAdded = id(qwen: 248_044, vocab: vocab)
        precondition(tok.count <= firstAdded, "toy vocabulary too small")
        var vmap: [String: Int] = [:]
        for (i, t) in tok.enumerated() { vmap[t] = i }
        // Padding: ids without merges (the model can generate them; they decode as text).
        let addedIds = Set((0..<qwenAdded.count).map { firstAdded + $0 })
        for i in tok.count..<vocab where !addedIds.contains(i) { vmap["<toy_\(i)>"] = i }
        let added: [[String: Any]] = qwenAdded.enumerated().map { i, t in
            [
                "id": firstAdded + i, "content": t.0, "single_word": false, "lstrip": false, "rstrip": false,
                "normalized": false, "special": t.1,
            ]
        }
        let root: [String: Any] = [
            "version": "1.0", "added_tokens": added, "normalizer": ["type": "NFC"],
            "pre_tokenizer": [
                "type": "Sequence",
                "pretokenizers": [
                    ["type": "Split", "pattern": ["Regex": splitRegex], "behavior": "Isolated", "invert": false],
                    ["type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false],
                ],
            ],
            "decoder": ["type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false],
            "model": [
                "type": "BPE", "continuing_subword_prefix": "", "end_of_word_suffix": "", "fuse_unk": false,
                "byte_fallback": false, "ignore_merges": false, "vocab": vmap, "merges": merges,
            ],
        ]
        return try JSONSerialization.data(withJSONObject: root, options: [.sortedKeys])
    }
}
