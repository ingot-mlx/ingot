import Foundation

/// Reading `tokenizer.json` for `BPETokenizer`: JSON parsing, vocabulary, merges, decoder, normalizer and
/// pretokenizer. Anything that is not supported is an error at load.
extension BPETokenizer {
    /// A normalizer step (tokenizers' `NFC`, `Replace` with a string pattern and `Prepend`), applied in order.
    enum NormStep {
        case nfc
        case replace(String, String)
        case prepend(String)
    }

    /// The vocabulary from the NSDictionary (its keys compare literally), keyed by UTF-8 bytes: every token, also those
    /// Swift's `String` equality merges.
    static func readVocabulary(_ vd: NSDictionary, fix: (String) -> String) throws -> [[UInt8]: Int32] {
        var bb: [[UInt8]: Int32] = [:]
        bb.reserveCapacity(vd.count)
        for (k, x) in vd {
            guard let t = k as? String, let i = x as? Int, i >= 0 else {
                throw TokError.unsupported("vocabulary entry")
            }
            bb[Array(fix(t).utf8)] = Int32(i)
        }
        return bb
    }

    /// The mode comes from the decoder: `ByteLevel` or none → byte-level; a `Sequence` of `Replace` (string
    /// pattern), `ByteFallback` and `Fuse` → SentencePiece. Anything else (`Strip`, `Metaspace`...) is an error.
    static func readDecoder(_ dec: [String: Any]?) throws -> (
        mode: Mode, replace: [(String, String)], byteFallback: Bool
    ) {
        guard let dec else { return (.byteLevel, [], false) }
        let steps: [[String: Any]]
        if (dec["type"] as? String) == "Sequence" {
            guard let list = dec["decoders"] as? [[String: Any]] else {
                throw TokError.unsupported("decoder Sequence without decoders")
            }
            steps = list
        } else {
            steps = [dec]
        }
        if steps.count == 1, (steps[0]["type"] as? String) == "ByteLevel" { return (.byteLevel, [], false) }
        var replaces: [(String, String)] = [], byteFallback = false
        for d in steps {
            switch d["type"] as? String {
            case "Replace":
                guard let from = (d["pattern"] as? [String: Any])?["String"] as? String,
                    let to = d["content"] as? String
                else { throw TokError.unsupported("decoder Replace without a string pattern") }
                replaces.append((from, to))
            case "ByteFallback": byteFallback = true
            case "Fuse": break
            default: throw TokError.unsupported("decoder \(d["type"] ?? "?")")
            }
        }
        return (.sentencePiece, replaces, byteFallback)
    }

    /// GPT-2 tables: printable bytes map to themselves and the rest to 256 + n.
    static func gpt2Tables() -> ([Character], [Character: UInt8]) {
        var bc = [Character](repeating: " ", count: 256)
        var n = 0
        for b in 0..<256 {
            let keep = (33...126).contains(b) || (161...172).contains(b) || (174...255).contains(b)
            bc[b] = Character(Unicode.Scalar(UInt32(keep ? b : 256 + n))!)
            if !keep { n += 1 }
        }
        var cb: [Character: UInt8] = [:]
        for (b, c) in bc.enumerated() { cb[c] = UInt8(b) }
        return (bc, cb)
    }

    /// SentencePiece lookup tables (empty in byte-level mode): `<0xNN>` ids for encoding (with `byte_fallback`) and
    /// decoding (with the `ByteFallback` decoder), single-scalar token ids and the scalar pairs inside tokens.
    static func sentencePieceTables(
        _ bb: [[UInt8]: Int32], enabled: Bool, byteFallback: Bool, decodeBytes: Bool
    ) -> (fallbackIds: [Int32], fallbackByte: [Int: UInt8], scalarIds: [UInt32: Int32], innerPairs: Set<UInt64>) {
        var fb = [Int32](repeating: -1, count: 256), fbByte: [Int: UInt8] = [:]
        var sc: [UInt32: Int32] = [:], ip = Set<UInt64>()
        guard enabled else { return (fb, fbByte, sc, ip) }
        for b in 0..<256 {
            guard let t = bb[Array(String(format: "<0x%02X>", b).utf8)] else { continue }
            if byteFallback { fb[b] = t }
            if decodeBytes { fbByte[Int(t)] = UInt8(b) }
        }
        for (k, i) in bb {
            let u = String(decoding: k, as: UTF8.self).unicodeScalars
            if u.count == 1 { sc[u.first!.value] = i }
            var prev: UInt32?
            for c in u {
                if let p = prev { ip.insert(UInt64(p) << 32 | UInt64(c.value)) }
                prev = c.value
            }
        }
        return (fb, fbByte, sc, ip)
    }

    /// As in tokenizers: a merge with a token that is not in the vocabulary is an error, and if a pair repeats the last
    /// one wins. Keyed by the pair of ids; the value is (rank, id of the merged token).
    static func readMerges(
        _ raw: [Any], vocabulary bb: [[UInt8]: Int32], fix: (String) -> String
    ) throws -> [UInt64: (rank: Int32, id: Int32)] {
        func id(_ t: String) -> Int32? { bb[Array(t.utf8)] }
        var mg: [UInt64: (rank: Int32, id: Int32)] = [:]
        mg.reserveCapacity(raw.count)
        for (r, m) in raw.enumerated() {
            let pair: [String]
            if let s = m as? String {
                pair = fix(s).split(separator: " ", maxSplits: 1, omittingEmptySubsequences: false).map(String.init)
            } else if let a = m as? [String] {
                pair = a.map(fix)
            } else {
                throw TokError.unsupported("merge \(r) ilegible")
            }
            guard pair.count == 2, let a = id(pair[0]), let b = id(pair[1]), let ab = id(pair[0] + pair[1]) else {
                throw TokError.unsupported("merge \(r) with tokens outside the vocabulary: \(pair)")
            }
            mg[UInt64(UInt32(a)) << 32 | UInt64(UInt32(b))] = (Int32(r), ab)
        }
        return mg
    }

    /// Added tokens are cut as is, without lstrip/rstrip/single_word or matching on the normalized text: a
    /// `normalized` one is only accepted for a special token with an NFC-only normalizer in byte-level mode.
    static func checkAddedTokens(_ tokens: [[String: Any]], normalizer: [NormStep], mode: Mode) throws {
        let onlyNFC = normalizer.allSatisfy {
            if case .nfc = $0 { return true }
            return false
        }
        for t in tokens {
            for k in ["lstrip", "rstrip", "single_word"] where (t[k] as? Bool) == true {
                throw TokError.unsupported("added token with \(k): \(t["content"] ?? "?")")
            }
            guard !normalizer.isEmpty, (t["normalized"] as? Bool) == true else { continue }
            if !onlyNFC || mode == .sentencePiece || (t["special"] as? Bool) != true {
                throw TokError.unsupported("normalized added token with normalizer: \(t["content"] ?? "?")")
            }
        }
    }

    /// `JSONSerialization` drops a leading U+FEFF from every string (Gemma has "\u{FEFF}", "\u{FEFF}\u{FEFF}" and
    /// "\u{FEFF}//" tokens: they would become "", "\u{FEFF}" and a second "//"). It is swapped in the raw JSON (as
    /// UTF-8 or as a `\ufeff` escape) for the noncharacter U+FDD2, which must not appear in the file, and the caller
    /// swaps it back (`unswapBOM`) in the strings it reads. A literal backslash before "ufeff" would turn into an
    /// invalid escape: an error at load, never a silent change. Returns whether anything was swapped.
    static func parseJSON(_ data: Data) throws -> (Any, swapped: Bool) {
        let ph = Data([0xEF, 0xB7, 0x92])
        if data.range(of: ph) != nil { throw TokError.unsupported("U+FDD2 in tokenizer.json") }
        var d = data, swapped = false
        for pat in [Data([0xEF, 0xBB, 0xBF]), Data("\\ufeff".utf8), Data("\\uFEFF".utf8)] {
            guard d.range(of: pat) != nil else { continue }
            var out = Data(capacity: d.count)
            var pos = d.startIndex
            while let r = d.range(of: pat, in: pos..<d.endIndex) {
                out += d[pos..<r.lowerBound]
                out += ph
                pos = r.upperBound
            }
            out += d[pos..<d.endIndex]
            d = out
            swapped = true
        }
        return (try JSONSerialization.jsonObject(with: d), swapped)
    }

    static let bomPlaceholder = String(Unicode.Scalar(0xFDD2)!)  // noncharacters cannot be written in literals

    static func unswapBOM(_ s: String) -> String {
        s.replacingOccurrences(of: bomPlaceholder, with: "\u{FEFF}", options: .literal)
    }

    /// `unswapBOM` in every string and key of a (small) JSON value, through NSDictionary, whose keys compare literally
    /// (a Swift dictionary would merge canonical equivalents).
    static func unswapBOM(in x: Any) -> Any {
        if let s = x as? String { return unswapBOM(s) }
        if let a = x as? [Any] { return a.map { unswapBOM(in: $0) } }
        if let d = x as? NSDictionary {
            let o = NSMutableDictionary(capacity: d.count)
            for (k, v) in d { o[(k as? String).map { unswapBOM($0) } ?? k] = unswapBOM(in: v) }
            return o
        }
        return x
    }

    /// Reads the `normalizer` (a `Sequence` or a single step): NFC, `Replace` with a string pattern and `Prepend`.
    static func normalizer(_ n: [String: Any]?) throws -> [NormStep] {
        guard let n else { return [] }
        let list: [[String: Any]]
        if (n["type"] as? String) == "Sequence" {
            guard let l = n["normalizers"] as? [[String: Any]] else {
                throw TokError.unsupported("normalizer Sequence without normalizers")
            }
            list = l
        } else {
            list = [n]
        }
        return try list.map { s in
            switch s["type"] as? String {
            case "NFC": return .nfc
            case "Replace":
                guard let from = (s["pattern"] as? [String: Any])?["String"] as? String,
                    let to = s["content"] as? String
                else { throw TokError.unsupported("normalizer Replace without a string pattern") }
                return .replace(from, to)
            case "Prepend":
                guard let p = s["prepend"] as? String else { throw TokError.unsupported("Prepend without text") }
                return .prepend(p)
            default: throw TokError.unsupported("normalizador \(s["type"] ?? "?")")
            }
        }
    }

    func normalize(_ s: String) -> String {
        var t = s
        for step in normalizer {
            switch step {
            case .nfc: t = t.precomposedStringWithCanonicalMapping
            case .replace(let a, let b): t = t.replacingOccurrences(of: a, with: b, options: .literal)
            case .prepend(let p): if !t.isEmpty { t = p + t }
            }
        }
        return t
    }

    // MARK: - Pretokenizador

    /// Behavior of a `Split` with the matching chunk (tokenizers' `SplitDelimiterBehavior`).
    enum SplitBehavior: String {
        case isolated = "Isolated", removed = "Removed", mergedWithPrevious = "MergedWithPrevious"
        case mergedWithNext = "MergedWithNext", contiguous = "Contiguous"
    }

    /// A `pre_tokenizer` step: they apply in order and each one splits the chunks the previous one leaves.
    enum PreStep {
        case split(NSRegularExpression, SplitBehavior, invert: Bool)
        /// Always the last step: bytes → GPT-2 characters come afterwards, in `bpe`.
        case byteLevel(addPrefixSpace: Bool, regex: NSRegularExpression?)
    }

    /// The `ByteLevel` expression with `use_regex` (GPT-2's).
    static let gpt2Pattern = "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"

    /// Reads the `pre_tokenizer` (a `Sequence`, which may nest others, or a single step) like HF tokenizers. Any
    /// step that is not known is an error at load. In byte-level mode `ByteLevel` must appear exactly once and at the
    /// end; in SentencePiece mode it must not appear (and the pretokenizer may be missing).
    static func preTokenizer(_ pt: [String: Any]?, byteLevel: Bool = true) throws -> [PreStep] {
        guard let pt else {
            if byteLevel { throw TokError.unsupported("no pre_tokenizer (ByteLevel is required)") }
            return []
        }
        func flatten(_ p: [String: Any]) throws -> [[String: Any]] {
            guard (p["type"] as? String) == "Sequence" else { return [p] }
            guard let list = p["pretokenizers"] as? [[String: Any]] else {
                throw TokError.unsupported("Sequence without pretokenizers")
            }
            return try list.flatMap(flatten)
        }
        let steps: [PreStep] = try flatten(pt).map { p in
            let type = p["type"] as? String ?? "?"
            switch type {
            case "Split":
                let pat = p["pattern"] as? [String: Any]
                let rx: String
                if let r = pat?["Regex"] as? String {
                    rx = icuPattern(r)
                } else if let lit = pat?["String"] as? String {
                    rx = NSRegularExpression.escapedPattern(for: lit)
                } else {
                    throw TokError.unsupported("Split without pattern")
                }
                guard let b = SplitBehavior(rawValue: p["behavior"] as? String ?? "") else {
                    throw TokError.unsupported("Split with behavior \(p["behavior"] ?? "?")")
                }
                return .split(try NSRegularExpression(pattern: rx), b, invert: (p["invert"] as? Bool) ?? false)
            case "ByteLevel":
                let useRegex = (p["use_regex"] as? Bool) ?? true
                return .byteLevel(
                    addPrefixSpace: (p["add_prefix_space"] as? Bool) ?? true,
                    regex: useRegex ? try NSRegularExpression(pattern: icuPattern(gpt2Pattern)) : nil)
            default:
                throw TokError.unsupported("pretokenizador \(type)")
            }
        }
        let byteLevelSteps = steps.indices.filter {
            if case .byteLevel = steps[$0] { return true }
            return false
        }
        if byteLevel ? byteLevelSteps != [steps.count - 1] : !byteLevelSteps.isEmpty {
            throw TokError.unsupported(
                byteLevel
                    ? "pretokenizer without a ByteLevel at the end" : "ByteLevel pretokenizer in SentencePiece mode")
        }
        return steps
    }

    /// Chunks of `s` after all pretokenizer steps, in order and with no empty chunks.
    func preTokenize(_ s: String) -> [String] {
        var pieces = [s]
        for step in preSteps {
            var next: [String] = []
            next.reserveCapacity(pieces.count * 2)
            for p in pieces {
                switch step {
                case .split(let re, let b, let invert):
                    Self.split(p, re, b, invert: invert, into: &next)
                case .byteLevel(let prefix, let re):
                    // By scalars: `hasPrefix` compares graphemes and " \u{200D}" does not start with " " for Swift.
                    let q = prefix && p.unicodeScalars.first != " " ? " " + p : p
                    if let re { Self.split(q, re, .isolated, invert: false, into: &next) } else { next.append(q) }
                }
            }
            pieces = next
        }
        return pieces
    }

    /// tokenizers' `NormalizedString::split`: the whole text ends up in chunks that match or not (with `invert`, the opposite)
    /// and the behavior decides what gets merged or dropped. Empty chunks are discarded.
    static func split(
        _ s: String, _ re: NSRegularExpression, _ b: SplitBehavior, invert: Bool, into out: inout [String]
    ) {
        let ns = s as NSString
        var parts: [(lo: Int, hi: Int, match: Bool)] = []
        var prev = 0
        for m in re.matches(in: s, range: NSRange(location: 0, length: ns.length)) {
            let r = m.range
            if prev != r.location { parts.append((prev, r.location, invert)) }
            parts.append((r.location, r.location + r.length, !invert))
            prev = r.location + r.length
        }
        if prev != ns.length { parts.append((prev, ns.length, invert)) }
        var res: [(lo: Int, hi: Int)] = []
        switch b {
        case .isolated:
            res = parts.map { ($0.lo, $0.hi) }
        case .removed:
            res = parts.filter { !$0.match }.map { ($0.lo, $0.hi) }
        case .mergedWithPrevious:
            var prevMatch = false
            for p in parts {
                if p.match && !prevMatch && !res.isEmpty {
                    res[res.count - 1].hi = p.hi
                } else {
                    res.append((p.lo, p.hi))
                }
                prevMatch = p.match
            }
        case .mergedWithNext:
            var prevMatch = false
            for p in parts.reversed() {
                if p.match && !prevMatch && !res.isEmpty {
                    res[res.count - 1].lo = p.lo
                } else {
                    res.append((p.lo, p.hi))
                }
                prevMatch = p.match
            }
            res.reverse()
        case .contiguous:
            var prevMatch: Bool?
            for p in parts {
                if p.match == prevMatch && !res.isEmpty {
                    res[res.count - 1].hi = p.hi
                } else {
                    res.append((p.lo, p.hi))
                }
                prevMatch = p.match
            }
        }
        for r in res where r.hi > r.lo { out.append(ns.substring(with: NSRange(location: r.lo, length: r.hi - r.lo))) }
    }

    /// The tokenizers expression (Rust regex crate/Oniguruma) in ICU: `\s` in Rust is Unicode White_Space, which
    /// in ICU is [\t-\r\x{85}\p{Z}] (ICU's `\s` leaves out \v and U+0085).
    static func icuPattern(_ p: String) -> String {
        p.replacingOccurrences(of: "\\s", with: "[\\t-\\r\\x{85}\\p{Z}]")
            .replacingOccurrences(of: "\\S", with: "[^\\t-\\r\\x{85}\\p{Z}]")
    }

}
