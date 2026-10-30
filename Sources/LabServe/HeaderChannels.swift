import Foundation

// Portions adapted from mlx-lm (MIT, Copyright (c) 2023 Apple Inc.; `tool_parsers/muse_spark.py`; see
// THIRD_PARTY_NOTICES.md).
// Muse ATEM output: messages with a header, `<|start|>assistant to=DEST<|message|>BODY<|eom|>` (the first one after
// the generation prompt `<|start|>assistant`, so without its `<|start|>assistant`). `to=self` is reasoning, `to=user`
// (or no recipient) the answer, and any other recipient a tool call whose body is an `<atem:function_calls>` block.
// `<|eot|>` ends the turn (an end token: the parser never sees it).
//
// gpt-oss harmony has the same skeleton: `<|channel|>analysis<|message|>…<|end|>` is reasoning, `final` and
// `commentary` (the preambles) the answer, and a `to=functions.NAME` recipient, in the role part
// (`<|start|>assistant to=functions.NAME<|channel|>commentary json<|message|>`) or in the channel part
// (`<|channel|>commentary to=functions.NAME <|constrain|>json<|message|>`), a call whose body is the JSON arguments.
// `<|call|>` and `<|return|>` are end tokens.

extension OutputParser {
    enum HeaderState { case header, body(HeaderChannel) }
    enum HeaderChannel { case reasoning, content, tool(String) }

    static let messageMark = "<|message|>"
    static let bodyEnds = ["<|eom|>", "<|eot|>", "<|end|>", "<|start|>"]

    /// The channel a header sends its body to.
    func channel(header: Substring) -> HeaderChannel {
        var h = header.trimmingCharacters(in: .whitespacesAndNewlines)
        if h.hasPrefix("<|start|>") { h = String(h.dropFirst(9)) }
        if h.hasPrefix("assistant") { h = String(h.dropFirst(9)) }
        h = h.trimmingCharacters(in: .whitespaces)
        if format == .harmony { return harmonyChannel(header: h) }
        guard h.hasPrefix("to=") else { return .content }
        let dest = h.dropFirst(3).prefix { !$0.isWhitespace && $0 != "<" }
        if dest == "self" { return .reasoning }
        return dest == "user" || dest.isEmpty || !toolsEnabled ? .content : .tool(String(dest))
    }

    /// harmony: a `functions.NAME` recipient anywhere in the header is a call (with tools in the request); otherwise
    /// the `analysis` channel is reasoning and the rest (`final`, `commentary`) the answer.
    func harmonyChannel(header h: String) -> HeaderChannel {
        func word(after mark: String) -> Substring? {
            h.range(of: mark).map { h[$0.upperBound...].prefix { !$0.isWhitespace && $0 != "<" } }
        }
        if toolsEnabled, let dest = word(after: "to="), dest.hasPrefix("functions."), dest.count > 10 {
            return .tool(String(dest.dropFirst(10)))
        }
        return word(after: "<|channel|>") == "analysis" ? .reasoning : .content
    }

    mutating func feedHeaders(_ text: String, _ ev: inout [OutputEvent]) {
        headerBuffer += text
        while !headerBuffer.isEmpty {
            switch headerState {
            case .header:
                if let r = headerBuffer.range(of: Self.messageMark) {
                    headerState = .body(channel(header: headerBuffer[..<r.lowerBound]))
                    headerBuffer = String(headerBuffer[r.upperBound...])
                    continue
                }
                // Not a header (too long, or several lines): plain answer text.
                if headerBuffer.count > 256 || headerBuffer.contains("\n") { headerState = .body(.content); continue }
                return
            case .body(let ch):
                let ends = Self.bodyEnds.compactMap { m in headerBuffer.range(of: m).map { ($0, m) } }
                guard let (r, tag) = ends.min(by: { $0.0.lowerBound < $1.0.lowerBound }) else {
                    let (ready, held) = holdPartial(headerBuffer, Self.bodyEnds)
                    deliver(ch, ready, &ev)
                    headerBuffer = held
                    return
                }
                deliver(ch, String(headerBuffer[..<r.lowerBound]), &ev)
                headerBuffer = String(headerBuffer[(tag == "<|start|>" ? r.lowerBound : r.upperBound)...])
                endMessage(ch, &ev)
                headerState = .header
            }
        }
    }

    mutating func deliver(_ ch: HeaderChannel, _ text: String, _ ev: inout [OutputEvent]) {
        guard !text.isEmpty else { return }
        switch ch {
        case .reasoning: reasoningText += text; ev.append(.reasoning(text))
        case .content: emitContent(text, &ev)
        case .tool: pending += text
        }
    }

    mutating func endMessage(_ ch: HeaderChannel, _ ev: inout [OutputEvent]) {
        guard case .tool(let dest) = ch else { return }
        let body = pending
        pending = ""
        let cs =
            format == .harmony
            ? parseHarmonyCall(dest, body, schemas: schemas) : parseATEMCalls(body, schemas: schemas)
        if let cs {
            emitCalls(cs, &ev)
        } else if format == .harmony {
            // The recipient already says it is a call: broken JSON goes out as the call's raw arguments (as the
            // OpenAI API, whose arguments may be invalid JSON), so the client reports the error to the model and it
            // retries, instead of ending the turn with the call as answer text (pi with gpt-oss-20b).
            emitRawCall(dest, body.trimmingCharacters(in: .whitespacesAndNewlines), &ev)
        } else {
            emitContent(body, &ev)
        }
    }

    mutating func finishHeaders(_ ev: inout [OutputEvent]) {
        switch headerState {
        case .header:
            // A header cut off before its body is dropped; anything else is answer text.
            let h = headerBuffer.trimmingCharacters(in: .whitespaces)
            if !(h.isEmpty || h.hasPrefix("to=") || h.hasPrefix("<|start|>") || h.hasPrefix("<|channel|>")) {
                emitContent(headerBuffer, &ev)
            }
        case .body(let ch):
            deliver(ch, headerBuffer, &ev)
            endMessage(ch, &ev)
        }
        headerBuffer = ""
        headerState = .header
    }
}

/// harmony: the body of a `to=functions.NAME` message is the JSON object of arguments (read tolerantly, as a declared
/// object parameter), values fitted to the schema. nil if it is not an object (the caller sends it raw).
func parseHarmonyCall(_ name: String, _ body: String, schemas: ToolSchemas) -> [ParsedCall]? {
    let t = body.trimmingCharacters(in: .whitespacesAndNewlines)
    guard let kv = ((try? JSONValue.parse(t)) ?? (try? JSONValue.parseLenient(t)))?.object else { return nil }
    return [(name, kv.map { ($0.0, conformValue($0.1, types: schemas.types(name, $0.0))) })]
}

/// `<atem:invoke name="N"><atem:parameter name="P">V</atem:parameter>…</atem:invoke>` (mlx-lm `muse_spark.py`; an
/// invoke not closed runs to the end, a parameter not closed is dropped). The values are written as the template
/// writes them (scalars as they are, lists and objects in JSON) and read by the schema; unlike mlx-lm, a declared
/// string stays text and surrounding spaces are kept (the template says they are not stripped). nil if no invoke.
func parseATEMCalls(_ body: String, schemas: ToolSchemas) -> [ParsedCall]? {
    func attr(_ tag: Substring) -> String? {
        guard let a = tag.range(of: "name=\""), let e = tag[a.upperBound...].firstIndex(of: "\"") else { return nil }
        return String(tag[a.upperBound..<e])
    }
    var out: [ParsedCall] = []
    var rest = Substring(body)
    while let s = rest.range(of: "<atem:invoke") {
        guard let gt = rest[s.upperBound...].firstIndex(of: ">"), let name = attr(rest[s.upperBound..<gt]),
            !name.isEmpty
        else { rest = rest[s.upperBound...]; continue }
        let after = rest[rest.index(after: gt)...]
        let close = after.range(of: "</atem:invoke>")
        var inner = after[..<(close?.lowerBound ?? after.endIndex)]
        var args: [(String, JSONValue)] = []
        while let p = inner.range(of: "<atem:parameter") {
            guard let pgt = inner[p.upperBound...].firstIndex(of: ">"),
                let pe = inner[pgt...].range(of: "</atem:parameter>")
            else { break }
            if let key = attr(inner[p.upperBound..<pgt]) {
                let raw = String(inner[inner.index(after: pgt)..<pe.lowerBound])
                if let i = args.firstIndex(where: { $0.0 == key }) { args.remove(at: i) }
                args.append((key, convertParameter(raw, types: schemas.types(name, key))))
            }
            inner = inner[pe.upperBound...]
        }
        out.append((name, args))
        rest = close.map { after[$0.upperBound...] } ?? after[after.endIndex...]
    }
    return out.isEmpty ? nil : out
}
