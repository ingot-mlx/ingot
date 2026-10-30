import Foundation

// Incremental parsing of model output: stop strings, reasoning (`<think>…</think>`) and tool calls in
// Qwen3.6 XML format (`<tool_call><function=NAME><parameter=P>value</parameter>…`), the
// same one read by vLLM's `qwen3_coder` parser and splash.
//
// Modified Swift port of `server/output.py` from splash 1.3.2 (incoai/splash, Apache-2.0; see THIRD_PARTY_NOTICES.md and
// LICENSES/Apache-2.0.txt): holding back partial tags, converting values according to the schema, the reasoning split
// and the tool-call state machine. Changed: rewritten in Swift, with our own types and streaming interface.

/// Splits `text` before the longest tail that starts any of `markers` (which more text could complete).
func holdPartial(_ text: String, _ markers: [String]) -> (ready: String, held: String) {
    let longest = markers.map(\.count).max() ?? 0
    var n = min(text.count, longest - 1)
    while n > 0 {
        let tail = String(text.suffix(n))
        if markers.contains(where: { n < $0.count && $0.hasPrefix(tail) }) { return (String(text.dropLast(n)), tail) }
        n -= 1
    }
    return (text, "")
}

/// Cuts the output at the first stop string (which is not emitted) and holds back what could be the start of one.
public struct StopMatcher {
    let stops: [String]
    var held = ""
    public private(set) var stopped = false
    public init(_ stops: [String]) { self.stops = stops.filter { !$0.isEmpty } }

    public mutating func put(_ text: String) -> String {
        guard !stopped else { return "" }
        guard !stops.isEmpty else { return text }
        let all = held + text
        if let r = stops.compactMap({ all.range(of: $0) }).min(by: { $0.lowerBound < $1.lowerBound }) {
            stopped = true
            held = ""
            return String(all[..<r.lowerBound])
        }
        let (ready, h) = holdPartial(all, stops)
        held = h
        return ready
    }
    public mutating func finish() -> String { defer { held = "" }; return stopped ? "" : held }
}

/// Output event for the API.
public enum OutputEvent: Equatable {
    case reasoning(String)
    case content(String)
    /// Header of a call (index, id, name).
    case toolStart(Int, String, String)
    /// Arguments (JSON) of call `index`.
    case toolArguments(Int, String)
}

public struct ToolCall: Equatable {
    public var id: String, name: String, arguments: String
}

/// JSON types declared for each parameter of each tool (to convert the values).
public struct ToolSchemas {
    var types: [String: [String: Set<String>]] = [:]
    public init(tools: [JSONValue]?) {
        for t in tools ?? [] {
            guard let f = t["function"], let name = f["name"]?.string else { continue }
            var ps: [String: Set<String>] = [:]
            for (p, schema) in f["parameters"]?["properties"]?.object ?? [] {
                if let s = schema["type"]?.string {
                    ps[p] = [s]
                } else if let a = schema["type"]?.array {
                    ps[p] = Set(a.compactMap(\.string))
                } else if let any = schema["anyOf"]?.array {
                    ps[p] = Set(any.compactMap { $0["type"]?.string })
                }
            }
            types[name] = ps
        }
    }
    func types(_ fn: String, _ p: String) -> Set<String>? {
        guard let t = types[fn]?[p], !t.isEmpty else { return nil }
        return t
    }
}

/// Converts a parameter's text according to the declared types (like splash's `convert_value`): a string
/// stays as text; any other value, as the JSON it writes if it is of a declared type or if string is not declared.
func convertParameter(_ text: String, types: Set<String>?) -> JSONValue {
    func kind(_ v: JSONValue) -> String {
        switch v {
        case .null: return "null"
        case .bool: return "boolean"
        case .int: return "integer"
        case .double: return "number"
        case .string: return "string"
        case .array: return "array"
        case .object: return "object"
        }
    }
    // An array or object declared that is not valid JSON is read tolerantly (unescaped quotes
    // inside strings); vLLM (qwen3_coder) and splash would leave it as text.
    if let types, !types.contains("string"), types.contains("array") || types.contains("object"),
        (try? JSONValue.parse(text)) == nil, let v = try? JSONValue.parseLenient(text),
        v.array != nil || v.object != nil
    {
        return v
    }
    guard let v = try? JSONValue.parse(text) else {
        let py: [String: JSONValue] = ["True": .bool(true), "False": .bool(false), "None": .null]
        if let v = py[text.trimmingCharacters(in: .whitespacesAndNewlines)], let types, !types.contains("string"),
            types.contains(kind(v))
        {
            return v
        }
        return .string(text)
    }
    if types == nil || types!.contains("string") {
        let k = kind(v)
        let declared = types == nil || types!.contains(k) || (k == "integer" && types!.contains("number"))
        return declared && k != "string" ? v : .string(text)
    }
    return v
}

/// Separates reasoning from the answer and extracts tool calls, chunk by chunk.
public struct OutputParser {
    let format: ToolCallFormat?
    /// The template left `<think>` without a newline (LFM2.5), or the model opened the reasoning itself: newlines at
    /// the start of the reasoning are removed.
    var trimReasoningStart = false
    // Reasoning.
    var reasoning: Bool
    var rpending = ""
    var separator = false
    /// Marks that close the reasoning: `reasoningClose`, preceded first by the separator the template writes before
    /// it (Gemma 4: `'\n<channel|>'`), so the reasoning comes out as the template takes it back.
    let reasoningCloses: [String]
    let reasoningEnds: [String]
    /// The model may open the reasoning itself (Qwen3, Gemma 4): the start of the output is held while it
    /// can still be this mark (after whitespace).
    var awaitingOpen: String?
    var startHeld = ""
    /// Gemma 4: a reasoning channel was just opened with `<|channel>`; its `thought` label is still to be dropped.
    var labelPending = false
    var labelHeld = ""
    // Calls.
    let schemas: ToolSchemas
    let toolsEnabled: Bool
    let idPrefix: String
    enum State { case content, call, name, args, paramName, value, closing }
    var state = State.content
    var pending = ""
    var heldWS = ""  // content whitespace held back until visible text appears
    var visible = false
    /// Gemma 4's channel closing mark (`<channel|>`; nil for other dialects). Gemma 4 output is read like vLLM's
    /// `gemma4` parser: `<|channel>` anywhere in the answer opens reasoning, an output that starts with `thought\n`
    /// is reasoning without its opening token, and a stray `<channel|>` or `<tool_call|>` in the answer is dropped
    /// (`dropChannelMarks`).
    let channelClose: String?
    var channelHeld = ""
    /// No visible content has gone out yet: an empty channel may still lack its opening token.
    var channelStart = true
    var fn: String?
    var callId = ""
    var params: [(String, JSONValue)] = []
    var param = ""
    var value = ""
    // Messages with headers (ATEM, harmony; HeaderChannels.swift).
    var headerState = HeaderState.header
    var headerBuffer = ""
    public private(set) var calls: [ToolCall] = []
    public private(set) var content = ""
    public internal(set) var reasoningText = ""

    static let thinkEnd = "</think>", callOpen = "<tool_call>", callClose = "</tool_call>", fnOpen = "<function="
    static let fnClose = "</function>", paramOpen = "<parameter=", paramClose = "</parameter>"
    static let channelOpen = "<|channel>", channelLabel = "thought"

    /// `thinking`: the output starts reasoning (the template left `<think>\n` at the end of the prompt).
    /// `reasoningOpen`: if the output starts (after whitespace) with this mark, it is reasoning too (the model opened
    /// it). `reasoningClose` ends the reasoning; `closeSeparator` is what the template writes between the reasoning and
    /// that mark (`ChatDialect.reasoningCloseSeparator`), cut off with it. A format that is not parsed (or none) leaves
    /// calls as content.
    public init(
        thinking: Bool, tools: [JSONValue]?, idPrefix: String, format: ToolCallFormat? = .qwenXML,
        trimReasoningStart: Bool = false, reasoningOpen: String? = nil, reasoningClose: String = "</think>",
        closeSeparator: String = ""
    ) {
        reasoning = thinking
        toolsEnabled = !(tools ?? []).isEmpty && format?.parsed == true
        schemas = ToolSchemas(tools: tools)
        self.idPrefix = idPrefix
        self.format = format
        self.trimReasoningStart = thinking && trimReasoningStart
        reasoningCloses = closeSeparator.isEmpty ? [reasoningClose] : [closeSeparator + reasoningClose, reasoningClose]
        channelClose = reasoningOpen?.hasPrefix(Self.channelOpen) == true ? reasoningClose : nil
        // Gemma 4: any channel name opens reasoning (the `thought` label is dropped after it).
        awaitingOpen = thinking ? nil : channelClose != nil ? Self.channelOpen : reasoningOpen
        reasoningEnds = reasoningCloses + (toolsEnabled ? [format?.blockMarks?.open ?? Self.callOpen] : [])
    }

    public mutating func put(_ text: String) -> [OutputEvent] {
        var ev: [OutputEvent] = []
        if format == .atem || format == .harmony { feedHeaders(text, &ev); return ev }
        if let open = awaitingOpen {
            startHeld += text
            let t = startHeld.drop { $0.isWhitespace }
            if t.hasPrefix(open) {
                awaitingOpen = nil
                startHeld = ""
                reasoning = true
                trimReasoningStart = true
                labelPending = channelClose != nil
                return put(String(t.dropFirst(open.count)))
            }
            // Gemma 4 sometimes leaves out the opening token: an output that starts with "thought\n" is reasoning
            // (vLLM's `Gemma4Parser._preprocess_feed`).
            let bare = Self.channelLabel + "\n"
            if channelClose != nil, t.hasPrefix(bare) {
                awaitingOpen = nil
                startHeld = ""
                reasoning = true
                trimReasoningStart = true
                return put(String(t.dropFirst(Self.channelLabel.count)))
            }
            if open.hasPrefix(t) || (channelClose != nil && bare.hasPrefix(t)) { return ev }
            awaitingOpen = nil
            let held = startHeld
            startHeld = ""
            feed(held, &ev)
            return ev
        }
        if !reasoning { feed(text, &ev); return ev }
        var text = text
        if labelPending {
            labelHeld += text
            let label = Self.channelLabel
            if labelHeld.count < label.count && label.hasPrefix(labelHeld) { return ev }
            text = labelHeld.hasPrefix(label) ? String(labelHeld.dropFirst(label.count)) : labelHeld
            labelHeld = ""
            labelPending = false
        }
        if trimReasoningStart {
            let t = String(text.drop { $0 == "\n" })
            if t.isEmpty { return ev }
            trimReasoningStart = false
            rpending += t
        } else {
            rpending += text
        }
        let ends = reasoningEnds.compactMap { m in rpending.range(of: m).map { ($0, m) } }
        if let (r, m) = ends.min(by: { $0.0.lowerBound < $1.0.lowerBound }) {
            let rs = String(rpending[..<r.lowerBound])
            var rest = String(rpending[r.lowerBound...])
            if reasoningCloses.contains(m) { rest = String(rest.dropFirst(m.count)); separator = true }
            rpending = ""
            reasoning = false
            if !rs.isEmpty { reasoningText += rs; ev.append(.reasoning(rs)) }
            feed(rest, &ev)
            return ev
        }
        let (ready, held) = holdPartial(rpending, reasoningEnds)
        rpending = held
        if !ready.isEmpty { reasoningText += ready; ev.append(.reasoning(ready)) }
        return ev
    }

    mutating func feed(_ t: String, _ ev: inout [OutputEvent]) {
        var text = t
        if separator {
            text = String(text.drop { $0 == "\n" })
            if text.isEmpty { return }
            separator = false
        }
        guard toolsEnabled else {
            guard channelClose != nil else { emitContent(text, &ev); return }
            pending += text
            parseChannels(&ev, final: false)
            return
        }
        pending += text
        if let m = format?.blockMarks { parseBlocks(m, &ev, final: false) } else { parse(&ev, final: false) }
    }

    /// Block formats (LFM2, GLM-4.7, Hermes, Gemma 4): content up to the opening mark (`<|tool_call_start|>`,
    /// `<tool_call>`), the whole block up to the closing mark (or the end) and its calls at once.
    mutating func parseBlocks(_ marks: (open: String, close: String), _ ev: inout [OutputEvent], final: Bool) {
        while !pending.isEmpty {
            if state == .content {
                // Gemma 4: a channel opens reasoning; a stray `<tool_call|>` is dropped (vLLM's `gemma4` parser).
                let tags = [marks.open, Self.thinkEnd] + (channelClose == nil ? [] : [Self.channelOpen, marks.close])
                guard let (r, tag) = firstTag(tags) else {
                    if final { emitContent(pending, &ev, final: true); pending = ""; return }
                    let (ready, held) = holdPartial(pending, tags)
                    emitContent(ready, &ev)
                    pending = held
                    return
                }
                emitContent(String(pending[..<r.lowerBound]), &ev, final: true)
                pending = String(pending[r.upperBound...])
                if tag == Self.channelOpen { return openChannel(&ev) }
                if tag == marks.open { state = .call }
            } else {
                guard let r = pending.range(of: marks.close) else {
                    if final { emitBlockCalls(pending, &ev); pending = ""; state = .content }
                    return
                }
                emitBlockCalls(String(pending[..<r.lowerBound]), &ev)
                pending = String(pending[r.upperBound...])
                state = .content
            }
        }
    }

    /// The calls of a block, in the format's syntax; values already fitted to the schema.
    func blockCalls(_ block: String) -> [ParsedCall]? {
        switch format {
        case .lfmPythonic:
            return parsePythonicCalls(block)?.map { name, args in
                (name, args.map { ($0.0, conformValue($0.1, types: schemas.types(name, $0.0))) })
            }
        case .glm47: return parseGLMCall(block, schemas: schemas)
        case .hermes: return parseJSONCall(block, schemas: schemas)
        case .gemma4: return parseGemmaCalls(block, schemas: schemas)
        default: return nil
        }
    }

    mutating func emitBlockCalls(_ block: String, _ ev: inout [OutputEvent]) {
        guard let cs = blockCalls(block) else {
            // Not understood: left as text (with its tags).
            emitContent((format?.blockMarks?.open ?? "") + block, &ev)
            return
        }
        emitCalls(cs, &ev)
    }

    mutating func emitCalls(_ cs: [ParsedCall], _ ev: inout [OutputEvent]) {
        for (name, args) in cs {
            heldWS = ""
            let id = "call_\(idPrefix)_\(calls.count)"
            ev.append(.toolStart(calls.count, id, name))
            let a = JSONValue.object(args).compact()
            ev.append(.toolArguments(calls.count, a))
            calls.append(ToolCall(id: id, name: name, arguments: a))
        }
        visible = false
        heldWS = ""
    }

    /// A call whose arguments could not be read, sent as written (harmony: the recipient names the function).
    mutating func emitRawCall(_ name: String, _ arguments: String, _ ev: inout [OutputEvent]) {
        heldWS = ""
        let id = "call_\(idPrefix)_\(calls.count)"
        ev.append(.toolStart(calls.count, id, name))
        ev.append(.toolArguments(calls.count, arguments))
        calls.append(ToolCall(id: id, name: name, arguments: arguments))
        visible = false
    }

    mutating func emitContent(_ text: String, _ ev: inout [OutputEvent], final: Bool = false) {
        let s = channelClose == nil ? text : dropChannelMarks(text, final: final)
        guard !s.isEmpty else { return }
        // The whitespace around calls (the template separates with "\n\n") is held back until visible
        // text arrives; if none does, it is dropped.
        let all = heldWS + s
        let trimmedEnd = all.reversed().drop { $0.isWhitespace }.count
        var body = String(all.prefix(trimmedEnd))
        heldWS = String(all.dropFirst(trimmedEnd))
        if !visible { body = String(body.drop { $0.isWhitespace }); if body.isEmpty { return } }
        visible = true
        content += body
        ev.append(.content(body))
    }

    /// Gemma 4 without tools: the answer up to a `<|channel>`, which opens reasoning.
    mutating func parseChannels(_ ev: inout [OutputEvent], final: Bool) {
        guard let r = pending.range(of: Self.channelOpen) else {
            let (ready, held) = final ? (pending, "") : holdPartial(pending, [Self.channelOpen])
            emitContent(ready, &ev, final: final)
            pending = held
            return
        }
        emitContent(String(pending[..<r.lowerBound]), &ev, final: true)
        pending = String(pending[r.upperBound...])
        openChannel(&ev)
    }

    /// `<|channel>` in the answer: what is pending after it is reasoning (its `thought` label dropped). The rest of
    /// the output goes through `put` again, so the caller must not look at `pending` afterwards.
    mutating func openChannel(_ ev: inout [OutputEvent]) {
        let rest = pending
        pending = ""
        reasoning = true
        trimReasoningStart = true
        labelPending = true
        ev += put(rest)
    }

    enum ChannelMatch: Equatable { case whole(Int), partial, none }

    /// An empty channel without its opening token at the start of `s`: a name of lowercase ASCII letters and `-`, a
    /// newline and `close`. `whole(n)`: its length; `partial`: more text could complete it.
    static func matchEmptyChannel(_ s: Substring, close: String) -> ChannelMatch {
        let name = s.prefix { $0.isASCII && ($0.isLowercase || $0 == "-") }
        guard name.count <= 16 else { return .none }
        var rest = s.dropFirst(name.count)
        if rest.isEmpty { return .partial }
        guard !name.isEmpty, rest.first == "\n" else { return .none }
        rest = rest.dropFirst()
        if rest.hasPrefix(close) { return .whole(name.count + 1 + close.count) }
        return close.hasPrefix(rest) ? .partial : .none
    }

    /// Gemma 4 sometimes repeats empty reasoning channels at the start of its answer without the opening token
    /// (`thought\n<channel|>`, `think\n<channel|>`), and leaves a stray `<channel|>` in it (vLLM absorbs it too).
    /// Neither is text for the client, and the templates strip channels from the history anyway (`strip_thinking`),
    /// so both are dropped. Holds back what could still become one unless `final`.
    mutating func dropChannelMarks(_ text: String, final: Bool) -> String {
        guard let close = channelClose else { return text }
        var rest = Substring(channelHeld + text)
        channelHeld = ""
        var out = ""
        func pass(_ s: Substring) {
            if channelStart && s.contains(where: { !$0.isWhitespace }) { channelStart = false }
            out += s
        }
        while true {
            if channelStart {
                let body = rest.drop { $0.isWhitespace }
                switch Self.matchEmptyChannel(body, close: close) {
                case .whole(let n):
                    rest = body.dropFirst(n)
                    continue
                case .partial where !final:
                    channelHeld = String(rest)
                    return out
                default: break
                }
            }
            guard let r = rest.range(of: close) else {
                let (ready, held) = final ? (String(rest), "") : holdPartial(String(rest), [close])
                pass(Substring(ready))
                channelHeld = held
                return out
            }
            pass(rest[..<r.lowerBound])
            rest = rest[r.upperBound...]
        }
    }

    func firstTag(_ tags: [String]) -> (Range<String.Index>, String)? {
        tags.compactMap { t in pending.range(of: t).map { ($0, t) } }.min { $0.0.lowerBound < $1.0.lowerBound }
    }

    /// Name up to `>` (nil if it may still continue).
    mutating func takeName() -> String? {
        guard let i = pending.firstIndex(of: ">") else { return pending.count > 256 ? "" : nil }
        let n = String(pending[..<i])
        pending = String(pending[pending.index(after: i)...])
        return n.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    mutating func parse(_ ev: inout [OutputEvent], final: Bool) {
        while !pending.isEmpty {
            switch state {
            case .content:
                let tags = [Self.callOpen, Self.fnOpen, Self.thinkEnd]
                guard let (r, tag) = firstTag(tags) else {
                    if final { emitContent(pending, &ev); pending = ""; return }
                    let (ready, held) = holdPartial(pending, tags)
                    emitContent(ready, &ev)
                    pending = held
                    return
                }
                emitContent(String(pending[..<r.lowerBound]), &ev)
                pending = String(pending[r.upperBound...])
                state = tag == Self.callOpen ? .call : tag == Self.fnOpen ? .name : .content
            case .call:
                // After <tool_call>: <function=NAME> (the rest is discarded).
                guard let (r, tag) = firstTag([Self.fnOpen, Self.callClose]) else {
                    pending = holdPartial(pending, [Self.fnOpen, Self.callClose]).held
                    return
                }
                pending = String(pending[r.upperBound...])
                state = tag == Self.fnOpen ? .name : .content
            case .name:
                guard let n = takeName() else { return }
                beginCall(n, &ev)
            case .args:
                let tags = [Self.paramOpen, Self.fnClose, Self.callClose]
                guard let (r, tag) = firstTag(tags) else { pending = holdPartial(pending, tags).held; return }
                pending = String(pending[r.upperBound...])
                if tag == Self.paramOpen {
                    state = .paramName
                } else {
                    finishCall(&ev); state = tag == Self.callClose ? .content : .closing
                }
            case .paramName:
                guard let n = takeName() else { return }
                param = n; value = ""; state = .value
            case .value:
                // The value ends at </parameter> (or at the next tag if the model did not close it).
                let tags = [Self.paramClose, Self.paramOpen, Self.fnClose, Self.callClose]
                guard let (r, tag) = firstTag(tags) else {
                    let (ready, held) = holdPartial(pending, tags)
                    value += ready; pending = held
                    return
                }
                value += String(pending[..<r.lowerBound])
                pending = tag == Self.paramClose ? String(pending[r.upperBound...]) : String(pending[r.lowerBound...])
                endParam()
                state = .args
            case .closing:
                let rest = String(pending.drop { $0.isWhitespace })
                if rest.hasPrefix(Self.callClose) {
                    pending = String(rest.dropFirst(Self.callClose.count)); state = .content
                } else if Self.callClose.hasPrefix(rest) && !final {
                    return
                } else {
                    state = .content
                }
            }
        }
    }

    mutating func beginCall(_ name: String, _ ev: inout [OutputEvent]) {
        state = .args
        params = []
        fn = name.isEmpty ? nil : name
        guard let fn else { return }
        heldWS = ""
        callId = "call_\(idPrefix)_\(calls.count)"
        ev.append(.toolStart(calls.count, callId, fn))
    }

    mutating func endParam() {
        var v = value
        if v.hasPrefix("\n") { v.removeFirst() }
        if v.hasSuffix("\n") { v.removeLast() }
        if let fn, !param.isEmpty, !params.contains(where: { $0.0 == param }) {
            params.append((param, convertParameter(v, types: schemas.types(fn, param))))
        }
        value = ""; param = ""
    }

    mutating func finishCall(_ ev: inout [OutputEvent]) {
        if let fn {
            let args = JSONValue.object(params).compact()
            ev.append(.toolArguments(calls.count, args))
            calls.append(ToolCall(id: callId, name: fn, arguments: args))
        }
        fn = nil
        visible = false
        heldWS = ""
    }

    /// End of output: releases what was held. An open call is closed with what it has (if the output was cut off by
    /// length, `incomplete`).
    public mutating func finish(incomplete: Bool) -> [OutputEvent] {
        var ev: [OutputEvent] = []
        if format == .atem || format == .harmony { finishHeaders(&ev); return ev }
        if awaitingOpen != nil {
            awaitingOpen = nil
            let held = startHeld
            startHeld = ""
            // Gemma 4: an output that is only the label is an empty channel without its marks (vLLM drops it too).
            if channelClose == nil || held.trimmingCharacters(in: .whitespacesAndNewlines) != Self.channelLabel {
                feed(held, &ev)
            }
        }
        if reasoning {
            // A channel cut off inside what could still have been its label.
            if !labelHeld.isEmpty { rpending = labelHeld + rpending; labelHeld = "" }
            if !rpending.isEmpty { reasoningText += rpending; ev.append(.reasoning(rpending)) }
            rpending = ""
            return ev
        }
        if toolsEnabled, let m = format?.blockMarks {
            parseBlocks(m, &ev, final: true)
            pending = ""
            state = .content
        } else if toolsEnabled {
            parse(&ev, final: true)
            switch state {
            case .value: endParam(); finishCall(&ev)
            case .args, .paramName: finishCall(&ev)
            case .name:
                if !incomplete, !pending.isEmpty { beginCall(pending, &ev); finishCall(&ev) }
            default: break
            }
            pending = ""
            state = .content
        } else if channelClose != nil {
            parseChannels(&ev, final: true)
        }
        if !channelHeld.isEmpty { emitContent("", &ev, final: true) }
        return ev
    }
}
