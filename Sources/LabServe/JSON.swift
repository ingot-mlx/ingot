import Foundation

/// JSON value that keeps key order (the template writes tools with `tojson` in the order they
/// arrive, like `json.dumps` in transformers) and tells integers from reals.
public indirect enum JSONValue: Equatable, Sendable {
    case null
    case bool(Bool)
    case int(Int)
    case double(Double)
    case string(String)
    case array([JSONValue])
    case object([(String, JSONValue)])

    public static func == (a: JSONValue, b: JSONValue) -> Bool {
        switch (a, b) {
        case (.null, .null): return true
        case let (.bool(x), .bool(y)): return x == y
        case let (.int(x), .int(y)): return x == y
        case let (.double(x), .double(y)): return x == y
        case let (.string(x), .string(y)): return x == y
        case let (.array(x), .array(y)): return x == y
        case let (.object(x), .object(y)):
            return x.count == y.count && zip(x, y).allSatisfy { $0.0 == $1.0 && $0.1 == $1.1 }
        default: return false
        }
    }

    public subscript(key: String) -> JSONValue? {
        if case .object(let kv) = self { return kv.first { $0.0 == key }?.1 }
        return nil
    }
    public var string: String? { if case .string(let s) = self { return s }; return nil }
    public var array: [JSONValue]? { if case .array(let a) = self { return a }; return nil }
    public var object: [(String, JSONValue)]? { if case .object(let o) = self { return o }; return nil }
    public var bool: Bool? { if case .bool(let b) = self { return b }; return nil }
    public var number: Double? {
        switch self {
        case .int(let i): return Double(i)
        case .double(let d): return d
        default: return nil
        }
    }
    public var intValue: Int? {
        switch self {
        case .int(let i): return i
        case .double(let d) where d == d.rounded(): return Int(d)
        default: return nil
        }
    }
    public var isNull: Bool { if case .null = self { return true }; return false }

    // MARK: - Reading

    public static func parse(_ data: Data) throws -> JSONValue {
        var p = Parser(bytes: [UInt8](data))
        p.skipWS()
        let v = try p.value(depth: 0)
        p.skipWS()
        guard p.i == p.bytes.count else { throw JSONError.syntax("text after the value", p.i) }
        return v
    }
    public static func parse(_ s: String) throws -> JSONValue { try parse(Data(s.utf8)) }

    /// Tolerant reader for the arguments a model writes: inside a string, an unescaped quote
    /// only closes it if what follows (skipping spaces) is `,` `}` `]` `:` or the end; otherwise it is a literal quote
    /// (the typical failure: code with quotes inside a hand-written JSON, `"newText": "f"x""`). It also accepts
    /// literal newlines and tabs inside strings.
    public static func parseLenient(_ s: String) throws -> JSONValue {
        if let v = try? parse(s) { return v }
        let u = Array(s.unicodeScalars)
        var out = String.UnicodeScalarView()
        var i = 0, inStr = false
        while i < u.count {
            let c = u[i]
            if !inStr {
                if c == "\"" { inStr = true }
                out.append(c); i += 1; continue
            }
            if c == "\\" {
                out.append(c)
                if i + 1 < u.count { out.append(u[i + 1]) }
                i += 2; continue
            }
            if c == "\"" {
                var j = i + 1
                while j < u.count, u[j] == " " || u[j] == "\n" || u[j] == "\r" || u[j] == "\t" { j += 1 }
                if j == u.count || [",", "}", "]", ":"].contains(u[j]) {
                    inStr = false; out.append(c)
                } else {
                    out.append(contentsOf: "\\\"".unicodeScalars)
                }
                i += 1; continue
            }
            if c == "\n" {
                out.append(contentsOf: "\\n".unicodeScalars)
            } else if c == "\t" {
                out.append(contentsOf: "\\t".unicodeScalars)
            } else if c == "\r" {
                out.append(contentsOf: "\\r".unicodeScalars)
            } else {
                out.append(c)
            }
            i += 1
        }
        return try parse(String(out))
    }

    // MARK: - Writing

    /// Compact JSON (`{"a":1}`), for API responses.
    public func compact() -> String {
        var o = ""
        write(&o, item: ",", key: ":", ensureASCII: false)
        return o
    }

    func write(_ o: inout String, item: String, key: String, ensureASCII: Bool) {
        switch self {
        case .null: o += "null"
        case .bool(let b): o += b ? "true" : "false"
        case .int(let i): o += String(i)
        case .double(let d): o += Self.pythonRepr(d)
        case .string(let s): Self.quote(s, into: &o, ensureASCII: ensureASCII)
        case .array(let a):
            o += "["
            for (i, v) in a.enumerated() {
                if i > 0 { o += item }
                v.write(&o, item: item, key: key, ensureASCII: ensureASCII)
            }
            o += "]"
        case .object(let kv):
            o += "{"
            for (i, (k, v)) in kv.enumerated() {
                if i > 0 { o += item }
                Self.quote(k, into: &o, ensureASCII: ensureASCII)
                o += key
                v.write(&o, item: item, key: key, ensureASCII: ensureASCII)
            }
            o += "}"
        }
    }

    /// Python float repr() (the shortest that reads back the same; scientific notation outside [1e-4, 1e16)).
    static func pythonRepr(_ d: Double) -> String {
        if d.isNaN { return "NaN" }
        if d.isInfinite { return d > 0 ? "Infinity" : "-Infinity" }
        if d == d.rounded() && abs(d) < 1e16 { return String(format: "%.1f", d) }
        return "\(d)"  // Swift gives the shortest representation, with the exponent as in Python (1e-05)
    }

    static func quote(_ s: String, into o: inout String, ensureASCII: Bool) {
        o += "\""
        for u in s.unicodeScalars {
            switch u {
            case "\"": o += "\\\""
            case "\\": o += "\\\\"
            case "\n": o += "\\n"
            case "\r": o += "\\r"
            case "\t": o += "\\t"
            case "\u{08}": o += "\\b"
            case "\u{0C}": o += "\\f"
            default:
                if u.value < 0x20 || (ensureASCII && u.value > 0x7e) {
                    if u.value > 0xffff {
                        let v = u.value - 0x10000
                        o += String(format: "\\u%04x\\u%04x", 0xd800 + (v >> 10), 0xdc00 + (v & 0x3ff))
                    } else {
                        o += String(format: "\\u%04x", u.value)
                    }
                } else {
                    o.unicodeScalars.append(u)
                }
            }
        }
        o += "\""
    }
}

public enum JSONError: Error, CustomStringConvertible {
    case syntax(String, Int)
    public var description: String {
        switch self {
        case .syntax(let m, let i): return "invalid JSON (\(m), byte \(i))"
        }
    }
}

private struct Parser {
    let bytes: [UInt8]
    var i = 0

    mutating func skipWS() {
        while i < bytes.count, bytes[i] == 0x20 || bytes[i] == 0x0a || bytes[i] == 0x0d || bytes[i] == 0x09 { i += 1 }
    }

    mutating func value(depth: Int) throws -> JSONValue {
        guard depth < 512 else { throw JSONError.syntax("demasiado anidado", i) }
        guard i < bytes.count else { throw JSONError.syntax("fin inesperado", i) }
        switch bytes[i] {
        case UInt8(ascii: "{"):
            i += 1
            var kv: [(String, JSONValue)] = []
            skipWS()
            if i < bytes.count, bytes[i] == UInt8(ascii: "}") { i += 1; return .object(kv) }
            while true {
                skipWS()
                guard i < bytes.count, bytes[i] == UInt8(ascii: "\"") else {
                    throw JSONError.syntax("expected a key", i)
                }
                let k = try string()
                skipWS()
                guard i < bytes.count, bytes[i] == UInt8(ascii: ":") else {
                    throw JSONError.syntax("expected ':'", i)
                }
                i += 1
                skipWS()
                let v = try value(depth: depth + 1)
                if let j = kv.firstIndex(where: { $0.0 == k }) { kv[j].1 = v } else { kv.append((k, v)) }
                skipWS()
                guard i < bytes.count else { throw JSONError.syntax("fin inesperado", i) }
                if bytes[i] == UInt8(ascii: ",") { i += 1; continue }
                if bytes[i] == UInt8(ascii: "}") { i += 1; return .object(kv) }
                throw JSONError.syntax("expected ',' or '}'", i)
            }
        case UInt8(ascii: "["):
            i += 1
            var a: [JSONValue] = []
            skipWS()
            if i < bytes.count, bytes[i] == UInt8(ascii: "]") { i += 1; return .array(a) }
            while true {
                skipWS()
                a.append(try value(depth: depth + 1))
                skipWS()
                guard i < bytes.count else { throw JSONError.syntax("fin inesperado", i) }
                if bytes[i] == UInt8(ascii: ",") { i += 1; continue }
                if bytes[i] == UInt8(ascii: "]") { i += 1; return .array(a) }
                throw JSONError.syntax("expected ',' or ']'", i)
            }
        case UInt8(ascii: "\""): return .string(try string())
        case UInt8(ascii: "t"): try lit("true"); return .bool(true)
        case UInt8(ascii: "f"): try lit("false"); return .bool(false)
        case UInt8(ascii: "n"): try lit("null"); return .null
        default: return try number()
        }
    }

    mutating func lit(_ s: String) throws {
        let u = Array(s.utf8)
        guard i + u.count <= bytes.count, Array(bytes[i..<(i + u.count)]) == u else {
            throw JSONError.syntax("literal", i)
        }
        i += u.count
    }

    mutating func number() throws -> JSONValue {
        let s = i
        var isFloat = false
        if i < bytes.count, bytes[i] == UInt8(ascii: "-") { i += 1 }
        while i < bytes.count {
            let c = bytes[i]
            if c >= 0x30 && c <= 0x39 {
                i += 1
            } else if c == UInt8(ascii: ".") || c == UInt8(ascii: "e") || c == UInt8(ascii: "E")
                || c == UInt8(ascii: "+") || c == UInt8(ascii: "-")
            {
                isFloat = true; i += 1
            } else {
                break
            }
        }
        guard i > s, let str = String(bytes: bytes[s..<i], encoding: .ascii) else { throw JSONError.syntax("valor", s) }
        if !isFloat, let v = Int(str) { return .int(v) }
        guard let d = Double(str) else { throw JSONError.syntax("number", s) }
        return .double(d)
    }

    mutating func hex4() throws -> UInt32 {
        guard i + 4 <= bytes.count, let s = String(bytes: bytes[i..<(i + 4)], encoding: .ascii),
            let v = UInt32(s, radix: 16)
        else {
            throw JSONError.syntax("\\u", i)
        }
        i += 4
        return v
    }

    mutating func string() throws -> String {
        i += 1
        var out = [UInt8]()
        while true {
            guard i < bytes.count else { throw JSONError.syntax("unterminated string", i) }
            let c = bytes[i]
            if c == UInt8(ascii: "\"") { i += 1; break }
            if c != UInt8(ascii: "\\") { out.append(c); i += 1; continue }
            i += 1
            guard i < bytes.count else { throw JSONError.syntax("escape", i) }
            let e = bytes[i]; i += 1
            switch e {
            case UInt8(ascii: "n"): out.append(0x0a)
            case UInt8(ascii: "t"): out.append(0x09)
            case UInt8(ascii: "r"): out.append(0x0d)
            case UInt8(ascii: "b"): out.append(0x08)
            case UInt8(ascii: "f"): out.append(0x0c)
            case UInt8(ascii: "/"): out.append(0x2f)
            case UInt8(ascii: "\\"): out.append(0x5c)
            case UInt8(ascii: "\""): out.append(0x22)
            case UInt8(ascii: "u"):
                var v = try hex4()
                if v >= 0xd800 && v < 0xdc00, i + 6 <= bytes.count, bytes[i] == UInt8(ascii: "\\"),
                    bytes[i + 1] == UInt8(ascii: "u")
                {
                    let save = i
                    i += 2
                    let lo = try hex4()
                    if lo >= 0xdc00 && lo < 0xe000 {
                        v = 0x10000 + ((v - 0xd800) << 10) + (lo - 0xdc00)
                    } else {
                        i = save
                    }
                }
                let sc = Unicode.Scalar(v) ?? "\u{FFFD}"
                out.append(contentsOf: Array(String(sc).utf8))
            default: throw JSONError.syntax("escape \\\(Character(Unicode.Scalar(e)))", i)
            }
        }
        return String(decoding: out, as: UTF8.self)
    }
}
