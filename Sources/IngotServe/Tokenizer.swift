import Foundation

/// BPE tokenizer from Hugging Face's `tokenizer.json`, in one of two modes decided by the file:
/// - byte-level (Qwen2/3.5/3.6, LFM2.5, GLM, gpt-oss...): added tokens (cut first, without normalizing), NFC
///   normalizer, the full `pre_tokenizer` (`Sequence` of `Split` and `ByteLevel`, like HF tokenizers),
///   bytes → GPT-2 characters and merges by rank;
/// - SentencePiece-style BPE (Gemma 3 and 4): `Replace(" "→"▁")`/`Prepend` normalizer, `Split` steps, initial
///   symbols by Unicode scalar with `byte_fallback` to `<0xNN>` tokens (or `unk`), and the
///   `Replace(▁→" ")`+`ByteFallback`+`Fuse` decoder.
/// Anything it does not support is an error at load time, never a silently different tokenization. No Python or
/// dependencies.
public final class BPETokenizer: @unchecked Sendable {
    /// The model vocabulary. Swift `String` keys compare by canonical equivalence, so tokens that only differ in
    /// composition ("é" and "e\u{301}", both in Gemma's vocabulary) share a key here: lookups that must be exact go
    /// through `idsByBytes`.
    public let vocab: [String: Int]
    /// The model vocabulary keyed by the UTF-8 bytes of each token (exact, for merges and initial symbols).
    let idsByBytes: [[UInt8]: Int32]
    public let idToToken: [String]
    /// Added tokens (special or not: `<|im_end|>`, `<think>`, `<tool_call>`...) by content and id.
    public let added: [String: Int]
    public let addedById: [Int: String]
    public let specialIds: Set<Int>
    let merges: [UInt64: (rank: Int32, id: Int32)]
    /// Byte-level: byte → id of the single-character token, or -1 if the vocabulary lacks it (Granite: HF drops the byte,
    /// or uses `unk`).
    let byteToId: [Int32]
    let byteChar: [Character]  // byte → GPT-2 character
    let charByte: [Character: UInt8]
    let preSteps: [PreStep]
    let addedRegex: NSRegularExpression?
    /// The special tokens and U+FDD0 (to escape them in content: `escapeSpecials`).
    let specialRegex: NSRegularExpression?
    let normalizer: [NormStep]
    public let mode: Mode
    /// SentencePiece: id of each single-scalar vocabulary token (the initial symbols), and the `<0xNN>` token of each
    /// byte (-1 if missing; all -1 without `byte_fallback`).
    let scalarIds: [UInt32: Int32]
    /// SentencePiece: every pair of adjacent scalars inside some vocabulary token. Between two known scalars that form
    /// no such pair no merge can ever cross, so a word is cut there and each part goes through the BPE (and its cache)
    /// on its own, with the same result (Gemma's whole text between added tokens is a single word).
    let innerPairs: Set<UInt64>
    let fallbackIds: [Int32]
    /// `unk` id (when the model names one that is in the vocabulary) and `fuse_unk`.
    let unkId: Int32?
    let fuseUnk: Bool
    /// SentencePiece decoder: string replacements on each token (`Replace`), and whether `<0xNN>` tokens become bytes
    /// (`ByteFallback`), keyed by id.
    let decodeReplace: [(String, String)]
    let fallbackByte: [Int: UInt8]
    /// `ignore_merges` (LFM2.5): a pretokenized word that is whole in the vocabulary is a single token, without
    /// applying the merges (which could split it differently).
    let ignoreMerges: Bool
    private var cache: [[UInt8]: [Int32]] = [:]
    private let lock = NSLock()

    public enum Mode: Sendable { case byteLevel, sentencePiece }

    public enum TokError: Error, CustomStringConvertible {
        case unsupported(String)
        public var description: String {
            switch self {
            case .unsupported(let s): return "unsupported tokenizer.json: \(s)"
            }
        }
    }

    public convenience init(directory: URL) throws {
        try self.init(data: Data(contentsOf: directory.appendingPathComponent("tokenizer.json")))
    }

    public init(data: Data) throws {
        let (parsed, swapped) = try Self.parseJSON(data)
        guard var root = parsed as? [String: Any],
            var model = root["model"] as? [String: Any], (model["type"] as? String) == "BPE",
            let vd = model["vocab"] as? NSDictionary
        else { throw TokError.unsupported("non-BPE model") }
        // The big parts (vocabulary, merges) are unswapped where they are read; the rest here.
        let fix: (String) -> String = { swapped ? Self.unswapBOM($0) : $0 }
        if swapped {
            for k in ["added_tokens", "normalizer", "pre_tokenizer", "decoder"] {
                if let x = root[k] { root[k] = Self.unswapBOM(in: x) }
            }
            if let u = model["unk_token"] as? String { model["unk_token"] = Self.unswapBOM(u) }
        }
        let byteFallback = (model["byte_fallback"] as? Bool) == true
        ignoreMerges = (model["ignore_merges"] as? Bool) ?? false
        let bb = try Self.readVocabulary(vd, fix: fix)
        idsByBytes = bb
        var v: [String: Int] = [:]
        v.reserveCapacity(bb.count)
        var inv = [String](repeating: "", count: Int(bb.values.max() ?? 0) + 1)
        for (k, i) in bb {
            let t = String(decoding: k, as: UTF8.self)
            v[t] = Int(i)
            inv[Int(i)] = t
        }
        vocab = v
        func id(_ t: String) -> Int32? { bb[Array(t.utf8)] }
        let dec = try Self.readDecoder(root["decoder"] as? [String: Any])
        mode = dec.mode
        if byteFallback && mode == .byteLevel { throw TokError.unsupported("byte_fallback with a byte-level decoder") }
        decodeReplace = dec.replace
        fuseUnk = (model["fuse_unk"] as? Bool) ?? false
        unkId = (model["unk_token"] as? String).flatMap(id)
        (byteChar, charByte) = Self.gpt2Tables()
        byteToId = mode == .byteLevel ? byteChar.map { c in id(String(c)) ?? -1 } : []
        let sp = Self.sentencePieceTables(
            bb, enabled: mode == .sentencePiece, byteFallback: byteFallback, decodeBytes: dec.byteFallback)
        (fallbackIds, fallbackByte) = (sp.fallbackIds, sp.fallbackByte)
        (scalarIds, innerPairs) = (sp.scalarIds, sp.innerPairs)
        merges = try Self.readMerges(model["merges"] as? [Any] ?? [], vocabulary: bb, fix: fix)
        var ad: [String: Int] = [:], adi: [Int: String] = [:], spIds = Set<Int>()
        for t in root["added_tokens"] as? [[String: Any]] ?? [] {
            guard let c = t["content"] as? String, let id = t["id"] as? Int else { continue }
            ad[c] = id; adi[id] = c
            if (t["special"] as? Bool) == true { spIds.insert(id) }
            if id >= inv.count { inv += [String](repeating: "", count: id - inv.count + 1) }
            inv[id] = c
        }
        added = ad; addedById = adi; specialIds = spIds; idToToken = inv
        if !ad.isEmpty {
            let alts = ad.keys.sorted { $0.count > $1.count }.map { NSRegularExpression.escapedPattern(for: $0) }
            addedRegex = try NSRegularExpression(pattern: alts.joined(separator: "|"))
        } else {
            addedRegex = nil
        }
        let spAlts = ad.filter { spIds.contains($0.value) }.keys.sorted { $0.count > $1.count }.map {
            NSRegularExpression.escapedPattern(for: $0)
        }
        specialRegex = try NSRegularExpression(pattern: (spAlts + ["\\x{FDD0}"]).joined(separator: "|"))
        normalizer = try Self.normalizer(root["normalizer"] as? [String: Any])
        try Self.checkAddedTokens(root["added_tokens"] as? [[String: Any]] ?? [], normalizer: normalizer, mode: mode)
        for k in ["continuing_subword_prefix", "end_of_word_suffix"] {
            if let x = model[k] as? String, !x.isEmpty { throw TokError.unsupported("BPE with \(k)") }
        }
        if let d = model["dropout"] as? Double, d > 0 { throw TokError.unsupported("BPE with dropout") }
        preSteps = try Self.preTokenizer(root["pre_tokenizer"] as? [String: Any], byteLevel: mode == .byteLevel)
    }

    // MARK: - Codificar

    /// Like transformers' `tokenizer.encode(text, add_special_tokens=False)`: all added tokens that appear
    /// in `text` (special or not) are cut as tokens. For texts the program supplies (markers, the template);
    /// a chat prompt goes through `encodeRendered` (special tokens in the content do not count).
    public func encode(_ text: String) -> [Int] { encode(text, unescaping: false) }

    /// Ids of a rendered template with escaped content (`escapeSpecials`): the template's added tokens
    /// are cut as usual and the special tokens that came from the content become text again and are tokenized
    /// as normal text (like `split_special_tokens=True` on that chunk). Without escapes, same as `encode`.
    public func encodeRendered(_ text: String) -> [Int] { encode(text, unescaping: true) }

    func encode(_ text: String, unescaping: Bool) -> [Int] {
        var out: [Int] = []
        out.reserveCapacity(text.utf8.count / 3)
        let ns = text as NSString
        var pos = 0
        func ordinary(_ s: String) { encodeOrdinary(unescaping ? unescapeSpecials(s) : s, into: &out) }
        if let re = addedRegex {
            for m in re.matches(in: text, range: NSRange(location: 0, length: ns.length)) {
                if m.range.location > pos {
                    ordinary(ns.substring(with: NSRange(location: pos, length: m.range.location - pos)))
                }
                out.append(added[ns.substring(with: m.range)]!)
                pos = m.range.location + m.range.length
            }
        }
        if pos < ns.length { ordinary(ns.substring(from: pos)) }
        return out
    }

    // MARK: - Content specials

    /// Escape of the special tokens that appear in message content. It is escaped before
    /// rendering the template, so that cuts by added tokens only see the specials the template puts:
    /// `<|im_end|>` in a file or a tool result becomes `\u{FDD0}n\u{FDD1}` (n = its id) and `\u{FDD0}` itself
    /// becomes `\u{FDD0}\u{FDD1}`. U+FDD0/U+FDD1 are Unicode noncharacters (for internal use, absent from real
    /// text) that are not whitespace and do not change under NFC, so they survive the template's `trim`, `split`, `replace` and `tojson`,
    /// which put the content into strings and manipulate it. Added tokens that are not special (`<think>`,
    /// `<tool_call>`...) are left alone: the template looks for them in the content and transformers always cuts them.
    /// (Swift does not allow noncharacters in literals: they go by value.)
    static let mark = Unicode.Scalar(0xFDD0)!, markEnd = Unicode.Scalar(0xFDD1)!

    public func escapeSpecials(_ s: String) -> String {
        guard let re = specialRegex else { return s }
        let ns = s as NSString
        let matches = re.matches(in: s, range: NSRange(location: 0, length: ns.length))
        if matches.isEmpty { return s }
        var out = "", pos = 0
        for m in matches {
            out += ns.substring(with: NSRange(location: pos, length: m.range.location - pos))
            let id = added[ns.substring(with: m.range)].map(String.init) ?? ""  // "" = U+FDD0 itself
            out.unicodeScalars.append(Self.mark); out += id; out.unicodeScalars.append(Self.markEnd)
            pos = m.range.location + m.range.length
        }
        return out + ns.substring(from: pos)
    }

    /// Rolls back `escapeSpecials` (in the rendered text, every U+FDD0 comes from an escape).
    public func unescapeSpecials(_ s: String) -> String {
        guard s.unicodeScalars.contains(Self.mark) else { return s }
        let u = Array(s.unicodeScalars)
        var out = String.UnicodeScalarView(), i = 0
        while i < u.count {
            guard u[i] == Self.mark else { out.append(u[i]); i += 1; continue }
            var j = i + 1, n = 0
            while j < u.count, (48...57).contains(u[j].value) { n = n * 10 + Int(u[j].value - 48); j += 1 }
            if j < u.count, u[j] == Self.markEnd {
                if j == i + 1 {
                    out.append(Self.mark)
                } else if let t = addedById[n] {
                    out.append(contentsOf: t.unicodeScalars)
                }
                i = j + 1
            } else {
                out.append(u[i]); i += 1
            }  // incomplete escape (should not happen): left as is
        }
        return String(out)
    }

    func encodeOrdinary(_ s: String, into out: inout [Int]) {
        let t = normalize(s)
        for piece in preTokenize(t) {
            for id in bpe(piece) { out.append(Int(id)) }
        }
    }

    /// BPE of a pretokenized word. SentencePiece: after the `ignore_merges` check on the whole word, the word is cut
    /// where no merge can cross (`innerPairs`) and each part is merged on its own.
    func bpe(_ piece: String) -> [Int32] {
        guard mode == .sentencePiece else { return bpeWord(piece) }
        if ignoreMerges, let id = idsByBytes[Array(piece.utf8)] { return [id] }
        var out: [Int32] = []
        var seg = String.UnicodeScalarView()
        var prevKnown: UInt32?
        for c in piece.unicodeScalars {
            let known = scalarIds[c.value] != nil
            if known, let p = prevKnown, !innerPairs.contains(UInt64(p) << 32 | UInt64(c.value)) {
                out += bpeWord(String(seg))
                seg = String.UnicodeScalarView()
            }
            seg.append(c)
            prevKnown = known ? c.value : nil
        }
        if !seg.isEmpty { out += bpeWord(String(seg)) }
        return out
    }

    /// BPE of a word like tokenizers' `Word::merge_all`: each time **one** pair is merged, the one with lowest rank (and, at
    /// equal rank, the leftmost), and the new pairs it forms enter the queue. Merging all
    /// occurrences of the best pair at once is not the same when a newly formed pair has a lower rank (it happens with the merges
    /// of LFM2.5: 30 newlines give 10+10+10 and not 8+8+8+6). The cache is keyed by bytes: Swift `String` keys
    /// compare by canonical equivalence and would mix composed and decomposed "é" without a normalizer.
    func bpeWord(_ piece: String) -> [Int32] {
        let bytes = Array(piece.utf8)
        lock.lock()
        if let c = cache[bytes] { lock.unlock(); return c }
        lock.unlock()
        if ignoreMerges, mode == .byteLevel, let id = idsByBytes[Array(String(bytes.map { byteChar[Int($0)] }).utf8)] {
            return [id]
        }
        var sym = initialSymbols(piece)
        let n = sym.count
        if n > 1 {
            func merge(_ a: Int32, _ b: Int32) -> (rank: Int32, id: Int32)? {
                merges[UInt64(UInt32(a)) << 32 | UInt64(UInt32(b))]
            }
            var prev = Array(-1..<(n - 1)), next = Array(1...n), alive = [Bool](repeating: true, count: n)
            next[n - 1] = -1
            var heap = MergeHeap()
            for i in 0..<(n - 1) {
                if let m = merge(sym[i], sym[i + 1]) { heap.push(.init(rank: m.rank, pos: Int32(i), id: m.id)) }
            }
            while let t = heap.pop() {
                let i = Int(t.pos), j = next[i]
                // Expired entries: the symbol was already merged with the previous one or the pair has changed.
                guard alive[i], j >= 0, let m = merge(sym[i], sym[j]), m.id == t.id else { continue }
                sym[i] = m.id; alive[j] = false
                next[i] = next[j]
                if next[j] >= 0 { prev[next[j]] = i }
                if prev[i] >= 0, let pm = merge(sym[prev[i]], sym[i]) {
                    heap.push(.init(rank: pm.rank, pos: Int32(prev[i]), id: pm.id))
                }
                if next[i] >= 0, let nm = merge(sym[i], sym[next[i]]) {
                    heap.push(.init(rank: nm.rank, pos: Int32(i), id: nm.id))
                }
            }
            sym = (0..<n).filter { alive[$0] }.map { sym[$0] }
        }
        if bytes.count < 256 {
            lock.lock(); if cache.count > 200_000 { cache.removeAll() }; cache[bytes] = sym; lock.unlock()
        }
        return sym
    }

    /// Symbols before the merges, like tokenizers' `BPE::merge_word`: one per byte (byte-level, through the GPT-2
    /// characters) or per Unicode scalar (SentencePiece; Rust's `chars()`). A symbol missing from the vocabulary goes to
    /// its `<0xNN>` tokens with `byte_fallback`, else to `unk` (merged with the previous one with `fuse_unk`), else it is
    /// dropped. As in tokenizers, a pending `unk` is not flushed before byte-fallback tokens.
    func initialSymbols(_ piece: String) -> [Int32] {
        var sym: [Int32] = []
        var pendingUnk = false
        func known(_ id: Int32) {
            if pendingUnk, let u = unkId { sym.append(u); pendingUnk = false }
            sym.append(id)
        }
        func missing() {
            guard let u = unkId else { return }
            if pendingUnk && !fuseUnk { sym.append(u) }
            pendingUnk = true
        }
        switch mode {
        case .byteLevel:
            sym.reserveCapacity(piece.utf8.count)
            for b in piece.utf8 {
                let id = byteToId[Int(b)]
                if id >= 0 { known(id) } else { missing() }
            }
        case .sentencePiece:
            sym.reserveCapacity(piece.unicodeScalars.count)
            for c in piece.unicodeScalars {
                if let id = scalarIds[c.value] {
                    known(id)
                    continue
                }
                let fb = Array(String(c).utf8).map { fallbackIds[Int($0)] }
                if fb.allSatisfy({ $0 >= 0 }) { sym += fb } else { missing() }
            }
        }
        if pendingUnk, let u = unkId { sym.append(u) }
        return sym
    }

    /// Empties the word cache (to time cold tokenization in tests).
    func clearCache() {
        lock.lock(); cache.removeAll(); lock.unlock()
    }

    // MARK: - Decodificar

    /// Text of a token through the decoder, except `ByteFallback`: byte-level, its GPT-2 characters back to bytes (added
    /// tokens as is); SentencePiece, the decoder's `Replace` steps (also on added tokens, like HF).
    func tokenText(_ id: Int) -> String {
        guard let t = addedById[id] ?? (id >= 0 && id < idToToken.count ? idToToken[id] : nil) else { return "" }
        var s = t
        for (a, b) in decodeReplace { s = s.replacingOccurrences(of: a, with: b, options: .literal) }
        return s
    }

    /// Bytes of a token (for added ones in byte-level mode, their text as is).
    public func bytes(of id: Int) -> [UInt8] {
        if mode == .sentencePiece {
            if let b = fallbackByte[id], addedById[id] == nil { return [b] }
            return Array(tokenText(id).utf8)
        }
        if let a = addedById[id] { return Array(a.utf8) }
        guard id >= 0 && id < idToToken.count else { return [] }
        return idToToken[id].map { charByte[$0] ?? 0x3f }
    }

    public func decode(_ ids: [Int], skipSpecial: Bool = false) -> String {
        let kept = ids.filter { !(skipSpecial && specialIds.contains($0)) }
        guard mode == .sentencePiece else { return String(decoding: kept.flatMap(bytes(of:)), as: UTF8.self) }
        // tokenizers' `ByteFallback`: a run of `<0xNN>` tokens is one string if it is valid UTF-8 and one U+FFFD per
        // token otherwise; then `Fuse`.
        var out = "", run: [UInt8] = []
        func flush() {
            guard !run.isEmpty else { return }
            if let s = String(validating: run, as: UTF8.self) {
                out += s
            } else {
                out += String(repeating: "\u{FFFD}", count: run.count)
            }
            run = []
        }
        for id in kept {
            if let b = fallbackByte[id], addedById[id] == nil {
                run.append(b)
            } else {
                flush()
                out += tokenText(id)
            }
        }
        flush()
        return out
    }
}

/// Min-heap of merges by (rank, position): tokenizers' `BinaryHeap` from `Word::merge_all`.
struct MergeHeap {
    struct Entry {
        var rank: Int32, pos: Int32, id: Int32
        func before(_ o: Entry) -> Bool { rank != o.rank ? rank < o.rank : pos < o.pos }
    }
    private var a: [Entry] = []

    mutating func push(_ e: Entry) {
        a.append(e)
        var i = a.count - 1
        while i > 0 {
            let p = (i - 1) / 2
            if !a[i].before(a[p]) { break }
            a.swapAt(i, p); i = p
        }
    }

    mutating func pop() -> Entry? {
        guard let top = a.first else { return nil }
        let last = a.removeLast()
        if a.isEmpty { return top }
        a[0] = last
        var i = 0
        while true {
            let l = 2 * i + 1, r = l + 1
            var m = i
            if l < a.count && a[l].before(a[m]) { m = l }
            if r < a.count && a[r].before(a[m]) { m = r }
            if m == i { break }
            a.swapAt(i, m); i = m
        }
        return top
    }
}

/// Incremental detokenization: accumulates bytes and only releases the prefix that is already complete UTF-8 (a character can
/// arrive split across tokens).
public struct StreamingDetokenizer {
    let tok: BPETokenizer
    var pending: [UInt8] = []
    public init(_ t: BPETokenizer) { tok = t }

    public mutating func push(_ id: Int) -> String {
        pending += tok.bytes(of: id)
        // Last valid cut: do not split an incomplete multibyte sequence at the end.
        var end = pending.count
        var i = pending.count - 1, back = 0
        while i >= 0 && back < 4 {
            let c = pending[i]
            if c & 0xC0 == 0x80 { i -= 1; back += 1; continue }
            let need = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 1
            if pending.count - i < need { end = i }
            break
        }
        guard end > 0 else { return "" }
        let s = String(decoding: pending[0..<end], as: UTF8.self)
        pending.removeFirst(end)
        return s
    }

    public mutating func flush() -> String {
        defer { pending = [] }
        return pending.isEmpty ? "" : String(decoding: pending, as: UTF8.self)
    }
}
