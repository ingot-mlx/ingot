import Foundation

// Portions adapted from mlx-lm (MIT, Copyright (c) 2023 Apple Inc.; `tool_parsers/`; see THIRD_PARTY_NOTICES.md).
// Tool-call blocks of the GLM-4.7 XML, Hermes JSON and Gemma 4 formats (the text between `<tool_call>` and
// `</tool_call>`, without the tags). OutputParser cuts the blocks; these read the calls in them, with the values fitted
// to the tools' schemas. References: mlx-lm 0.32.2 `tool_parsers/glm47.py`, `json_tools.py` and `gemma4.py` (ref/tool_ref.py).

/// A parsed call: name and arguments in order.
typealias ParsedCall = (name: String, arguments: [(String, JSONValue)])

/// GLM-4.7 (GLM): `NAME<arg_key>k</arg_key><arg_value>v</arg_value>…`. The name is the text before the
/// first `<arg_key>`; each value is converted by the schema (anything that is not a string is read, stripped, as JSON
/// when it is valid JSON, as mlx-lm's `_deserialize`). A value that stays a string keeps its whitespace: the template
/// writes it back verbatim (`<arg_value>{{ v }}</arg_value>`), and a file the model ends in `\n` keeps it (mlx-lm
/// strips it). A block without `<arg_key>` is a JSON call or a bare name (mlx-lm's fallbacks). nil if nothing is
/// understood.
func parseGLMCall(_ block: String, schemas: ToolSchemas) -> [ParsedCall]? {
    guard let k = block.range(of: "<arg_key>") else {
        if let c = parseJSONCall(block, schemas: schemas) { return c }
        let name = block.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty, !name.contains(where: \.isWhitespace) else { return nil }
        return [(name, [])]
    }
    let name = block[..<k.lowerBound].trimmingCharacters(in: .whitespacesAndNewlines)
    guard !name.isEmpty else { return nil }
    var args: [(String, JSONValue)] = []
    var rest = block[k.lowerBound...]
    // <arg_key>K</arg_key> whitespace <arg_value>V</arg_value>, as mlx-lm's `_func_arg_regex` (non-greedy).
    while let ks = rest.range(of: "<arg_key>") {
        guard let ke = rest.range(of: "</arg_key>", range: ks.upperBound..<rest.endIndex) else { break }
        let key = rest[ks.upperBound..<ke.lowerBound].trimmingCharacters(in: .whitespacesAndNewlines)
        let after = rest[ke.upperBound...].drop { $0.isWhitespace }
        guard after.hasPrefix("<arg_value>"),
            let ve = after.range(of: "</arg_value>")
        else {
            rest = rest[ke.upperBound...]
            continue
        }
        let raw = String(after[after.index(after.startIndex, offsetBy: 11)..<ve.lowerBound])
        var value = convertParameter(
            raw.trimmingCharacters(in: .whitespacesAndNewlines), types: schemas.types(name, key))
        if value.string != nil { value = .string(raw) }
        if let i = args.firstIndex(where: { $0.0 == key }) { args.remove(at: i) }
        args.append((key, value))
        rest = after[ve.upperBound...]
    }
    return [(name, args)]
}

/// Hermes (Granite 4.1; mlx-lm `json_tools`) and GLM-4.7's JSON fallback (`glm47._parse_json_tool_call`): one JSON
/// object `{"name": …, "arguments": …}`, with the arguments as an object or as a JSON string of one; values fitted to
/// the schema as in the Pythonic format. nil if it is not such an object.
func parseJSONCall(_ block: String, schemas: ToolSchemas) -> [ParsedCall]? {
    guard let v = try? JSONValue.parse(block.trimmingCharacters(in: .whitespacesAndNewlines)),
        let name = v["name"]?.string, !name.isEmpty
    else { return nil }
    var args = v["arguments"] ?? .object([])
    if let s = args.string, let o = try? JSONValue.parse(s) { args = o }
    if args.isNull { args = .object([]) }
    guard let kv = args.object else { return nil }
    return [(name, kv.map { ($0.0, conformValue($0.1, types: schemas.types(name, $0.0))) })]
}

/// Gemma 4: one or more `call:NAME{key:value,…}` (mlx-lm `gemma4.py`): balanced braces, strings between `<|"|>` (braces
/// inside them do not count) and bare keys, turned into JSON as `_gemma4_args_to_json`; values fitted to the schema.
/// nil if no call is understood.
func parseGemmaCalls(_ block: String, schemas: ToolSchemas) -> [ParsedCall]? {
    let q = "<|\"|>"
    var out: [ParsedCall] = []
    var rest = Substring(block)
    while let c = rest.range(of: "call:") {
        var i = c.upperBound
        while i < rest.endIndex, isWordChar(rest[i]) || rest[i] == "-" { i = rest.index(after: i) }
        let name = String(rest[c.upperBound..<i])
        guard !name.isEmpty, i < rest.endIndex, rest[i] == "{" else { rest = rest[c.upperBound...]; continue }
        var depth = 0, j = i
        var end: String.Index?
        while j < rest.endIndex {
            if rest[j...].hasPrefix(q) {
                guard let close = rest.range(of: q, range: rest.index(j, offsetBy: q.count)..<rest.endIndex) else {
                    break
                }
                j = close.upperBound
                continue
            }
            if rest[j] == "{" { depth += 1 }
            if rest[j] == "}" {
                depth -= 1
                if depth == 0 { end = rest.index(after: j); break }
            }
            j = rest.index(after: j)
        }
        guard let e = end, let obj = try? JSONValue.parse(gemmaArgsToJSON(rest[i..<e])), let kv = obj.object else {
            rest = rest[c.upperBound...]
            continue
        }
        out.append((name, kv.map { ($0.0, conformValue($0.1, types: schemas.types(name, $0.0))) }))
        rest = rest[e...]
    }
    return out.isEmpty ? nil : out
}

func isWordChar(_ c: Character) -> Bool { c.isLetter || c.isNumber || c == "_" }

/// Gemma 4 arguments as JSON: `<|"|>…<|"|>` strings become JSON strings and a word right after `{` or `,` followed by
/// `:` becomes a quoted key (the rest is already JSON: numbers, true/false/null, brackets).
func gemmaArgsToJSON(_ s: Substring) -> String {
    let q = "<|\"|>"
    var out = ""
    var prev: Character = " "
    var i = s.startIndex
    while i < s.endIndex {
        if s[i...].hasPrefix(q), let close = s.range(of: q, range: s.index(i, offsetBy: q.count)..<s.endIndex) {
            out += JSONValue.string(String(s[s.index(i, offsetBy: q.count)..<close.lowerBound])).compact()
            prev = "\""
            i = close.upperBound
            continue
        }
        if prev == "{" || prev == ",", isWordChar(s[i]) {
            var k = i
            while k < s.endIndex, isWordChar(s[k]) { k = s.index(after: k) }
            if k < s.endIndex, s[k] == ":" {
                out += JSONValue.string(String(s[i..<k])).compact() + ":"
                prev = ":"
                i = s.index(after: k)
                continue
            }
        }
        out.append(s[i])
        prev = s[i]
        i = s.index(after: i)
    }
    return out
}
