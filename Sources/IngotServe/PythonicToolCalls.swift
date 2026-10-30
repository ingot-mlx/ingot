import Foundation

// Tool calls for LFM2/LFM2.5: the template writes them as a list of Python calls between special
// tokens, `<|tool_call_start|>[name(a='text', b=3, c=[1, 2])]<|tool_call_end|>` (single-quoted
// strings with \\, \', \n and \r escaped; lists and objects as JSON). Here they are read tolerantly: strings with
// single or double quotes, True/False/None and true/false/null, numbers, nested lists and dictionaries.

struct PythonicReader {
    let s: [Character]
    var i = 0
    init(_ text: String) { s = Array(text) }

    mutating func ws() { while i < s.count, s[i].isWhitespace { i += 1 } }
    mutating func eat(_ c: Character) -> Bool { ws(); if i < s.count, s[i] == c { i += 1; return true }; return false }

    mutating func identifier() -> String? {
        ws()
        let st = i
        while i < s.count, s[i].isLetter || s[i].isNumber || s[i] == "_" || s[i] == "." || s[i] == "-" { i += 1 }
        return i > st ? String(s[st..<i]) : nil
    }

    mutating func string() -> String? {
        ws()
        guard i < s.count, s[i] == "'" || s[i] == "\"" else { return nil }
        let q = s[i]
        i += 1
        var out = ""
        while i < s.count {
            let c = s[i]
            if c == q { i += 1; return out }
            if c == "\\", i + 1 < s.count {
                let n = s[i + 1]
                i += 2
                switch n {
                case "n": out.append("\n")
                case "r": out.append("\r")
                case "t": out.append("\t")
                case "u" where i + 4 <= s.count:
                    if let v = UInt32(String(s[i..<(i + 4)]), radix: 16), let u = Unicode.Scalar(v) {
                        out.unicodeScalars.append(u); i += 4
                    } else {
                        out.append("u")
                    }
                default: out.append(n)
                }
                continue
            }
            out.append(c)
            i += 1
        }
        return out  // unclosed: whatever there is
    }

    mutating func value() -> JSONValue? {
        ws()
        guard i < s.count else { return nil }
        let c = s[i]
        if c == "'" || c == "\"" { return string().map { .string($0) } }
        if c == "[" || c == "(" {
            let close: Character = c == "[" ? "]" : ")"
            i += 1
            var a: [JSONValue] = []
            if eat(close) { return .array(a) }
            while true {
                guard let v = value() else { return nil }
                a.append(v)
                if eat(",") { if eat(close) { return .array(a) }; continue }
                if eat(close) { return .array(a) }
                return nil
            }
        }
        if c == "{" {
            i += 1
            var o: [(String, JSONValue)] = []
            if eat("}") { return .object(o) }
            while true {
                ws()
                guard let k = string() ?? identifier(), eat(":"), let v = value() else { return nil }
                o.append((k, v))
                if eat(",") { if eat("}") { return .object(o) }; continue }
                if eat("}") { return .object(o) }
                return nil
            }
        }
        // Number or word.
        let st = i
        while i < s.count, !",)]}".contains(s[i]), !s[i].isWhitespace { i += 1 }
        let w = String(s[st..<i])
        switch w {
        case "True", "true": return .bool(true)
        case "False", "false": return .bool(false)
        case "None", "null": return .null
        default:
            if let n = Int(w) { return .int(n) }
            if let d = Double(w) { return .double(d) }
            return w.isEmpty ? nil : .string(w)
        }
    }

    /// `[f(a=1, b='x'), g()]` (the brackets are optional).
    mutating func calls() -> [(String, [(String, JSONValue)])]? {
        let bracket = eat("[")
        var out: [(String, [(String, JSONValue)])] = []
        while true {
            ws()
            if bracket, eat("]") { return out }
            if i >= s.count { return out.isEmpty ? nil : out }
            guard let name = identifier(), eat("(") else { return out.isEmpty ? nil : out }
            var args: [(String, JSONValue)] = []
            if !eat(")") {
                while true {
                    guard let k = identifier(), eat("="), let v = value() else {
                        return out.isEmpty && args.isEmpty ? nil : out + [(name, args)]
                    }
                    args.append((k, v))
                    if eat(",") { continue }
                    if eat(")") { break }
                    return out + [(name, args)]
                }
            }
            out.append((name, args))
            _ = eat(",")
        }
    }
}

/// Reads the calls in a `[f(a=1), …]` block (without the special tokens). nil if nothing is understood.
func parsePythonicCalls(_ text: String) -> [(String, [(String, JSONValue)])]? {
    var r = PythonicReader(text)
    return r.calls()
}

/// Fits an already typed value to the schema: if the parameter is declared as string only and something else arrives, its JSON as
/// text; if array/object is declared and a string with JSON arrives, that JSON.
func conformValue(_ v: JSONValue, types: Set<String>?) -> JSONValue {
    guard let types else { return v }
    if case .string(let s) = v {
        if !types.contains("string") { return convertParameter(s, types: types) }
        return v
    }
    if types == ["string"] { return .string(v.compact()) }
    return v
}
