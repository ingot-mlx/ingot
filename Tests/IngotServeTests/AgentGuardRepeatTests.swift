import Foundation
import Testing

@testable import IngotServe

/// Agent guards, repeated call: when the request is armed (`AgentGuard.suspect` over the history), the
/// streaks of pi sessions in Resources/guard/repeats.json, and the events with the guard against without it in qwenXML
/// (the held header), block formats and harmony.
@Suite struct AgentGuardRepeatTests {
    static let bash = AgentGuard.Call(name: "bash", arguments: #"{"command":"ls"}"#)

    static func msg(_ pairs: [(String, JSONValue)]) -> JSONValue { .object(pairs) }

    static func call(_ name: String, _ args: JSONValue) -> JSONValue {
        msg([
            ("role", .string("assistant")), ("content", .string("")),
            (
                "tool_calls",
                .array([
                    msg([
                        ("type", .string("function")),
                        ("function", msg([("name", .string(name)), ("arguments", args)])),
                    ])
                ])
            ),
        ])
    }

    static func result(_ text: String) -> JSONValue { msg([("role", .string("tool")), ("content", .string(text))]) }

    static let start: [JSONValue] = [
        msg([("role", .string("system")), ("content", .string("Agent."))]),
        msg([("role", .string("user")), ("content", .string("Do it."))]),
    ]

    static let ls = JSONValue.object([("command", .string("ls"))])

    /// Armed only by two identical single calls with the same result that end the history.
    @Test func arming() throws {
        let two =
            Self.start + [Self.call("bash", Self.ls), Self.result("a"), Self.call("bash", Self.ls), Self.result("a")]
        #expect(AgentGuard.suspect(messages: two) == Self.bash)
        // Arguments as text (what pi sends) or in another key order are the same call.
        let args = JSONValue.object([("b", .int(1)), ("a", .array([.object([("y", .null), ("x", .bool(true))])]))])
        let text = JSONValue.string(#"{"a":[{"x":true,"y":null}],"b":1}"#)
        let mixed = Self.start + [Self.call("f", args), Self.result("r"), Self.call("f", text), Self.result("r")]
        #expect(AgentGuard.suspect(messages: mixed)?.arguments == #"{"a":[{"x":true,"y":null}],"b":1}"#)
        // A streak of one, other results, other arguments, another name, a turn with two calls, a user message after
        // the last result, and a history that does not end in a tool result: not armed.
        let notArmed: [[JSONValue]] = [
            Self.start + [Self.call("bash", Self.ls), Self.result("a")],
            Self.start + [Self.call("bash", Self.ls), Self.result("a"), Self.call("bash", Self.ls), Self.result("b")],
            Self.start + [
                Self.call("bash", Self.ls), Self.result("a"),
                Self.call("bash", .object([("command", .string("ls -a"))])), Self.result("a"),
            ],
            Self.start + [Self.call("read", Self.ls), Self.result("a"), Self.call("bash", Self.ls), Self.result("a")],
            Self.start + [
                Self.call("bash", Self.ls), Self.result("a"),
                Self.msg([
                    ("role", .string("assistant")),
                    (
                        "tool_calls",
                        .array([
                            Self.call("bash", Self.ls)["tool_calls"]!.array![0],
                            Self.call("bash", Self.ls)["tool_calls"]!.array![0],
                        ])
                    ),
                ]),
                Self.result("a"), Self.result("a"),
            ],
            two + [Self.msg([("role", .string("user")), ("content", .string("Again."))])],
            Self.start + [Self.call("bash", Self.ls), Self.result("a"), Self.call("bash", Self.ls)],
        ]
        for (i, m) in notArmed.enumerated() { #expect(AgentGuard.suspect(messages: m) == nil, "case \(i)") }
    }

    /// Feeds the events in `rounds` through an armed guard as `Session` does; what goes out and the trip.
    static func filtered(_ rounds: [[OutputEvent]], suspect: AgentGuard.Call?) -> (
        sent: [OutputEvent], trip: AgentGuard.Trip?
    ) {
        var g = AgentGuard(suspect: suspect)
        var sent: [OutputEvent] = []
        for r in rounds {
            let (out, t) = g.filter(r)
            sent += out
            if let t { return (sent, t) }
        }
        return (sent + g.flush(), nil)
    }

    /// The same name with other arguments goes out unchanged; the suspect is dropped, after the calls before it.
    @Test func filterDecides() {
        let other: [[OutputEvent]] = [
            [.content("Let me look.")], [.toolStart(0, "c0", "bash")], [], [.toolArguments(0, #"{"command":"ls -a"}"#)],
        ]
        var r = Self.filtered(other, suspect: Self.bash)
        #expect(r.trip == nil && r.sent == Array(other.joined()))
        let repeated: [[OutputEvent]] = [
            [.toolStart(0, "c0", "read"), .toolArguments(0, #"{"path":"a"}"#)], [.toolStart(1, "c1", "bash")],
            [.toolArguments(1, #"{"command":"ls"}"#), .content("after")],
        ]
        r = Self.filtered(repeated, suspect: Self.bash)
        #expect(r.trip == .repeatedCall(name: "bash"))
        #expect(r.sent == [.toolStart(0, "c0", "read"), .toolArguments(0, #"{"path":"a"}"#)])
        // Not armed: everything as it is.
        r = Self.filtered(repeated, suspect: nil)
        #expect(r.trip == nil && r.sent == Array(repeated.joined()))
    }

    static func fixtures() throws -> [JSONValue] {
        let url = Bundle.module.url(forResource: "Resources", withExtension: nil)!
            .appendingPathComponent("guard/repeats.json")
        return try JSONValue.parse(Data(contentsOf: url))["cases"]?.array ?? []
    }

    /// The streaks of the pi corpus (`t2_corpus.py`): the third call of the 52 `write` of Qwen3.6, the 138 and
    /// 56 `bash` of LFM2.5-24B and the 3 `write` of gpt-oss trips; second calls of a streak and armed requests whose
    /// response changes the call do not.
    @Test func piStreaks() throws {
        let cases = try Self.fixtures()
        let qwen = ToolCallFormatTests.dialect(.qwenXML)
        var trips: [String] = []
        for c in cases {
            let src = c["source"]?.string ?? "?"
            let msgs = try ChatServer.normalize(c["messages"]?.array ?? [], dialect: qwen)
            let s = AgentGuard.suspect(messages: msgs)
            #expect((s != nil) == (c["armed"]?.bool == true), "\(src)")
            var events: [[OutputEvent]] = []
            for (i, rc) in (c["response_calls"]?.array ?? []).enumerated() {
                // The parser writes the arguments as compact JSON in the model's order.
                events.append([.toolStart(i, "c\(i)", rc["name"]?.string ?? "")])
                events.append([.toolArguments(i, (rc["arguments"] ?? .object([])).compact())])
            }
            let r = Self.filtered(events, suspect: s)
            #expect((r.trip != nil) == (c["trip"]?.bool == true), "\(src)")
            if r.trip != nil { trips.append(src) } else { #expect(r.sent == Array(events.joined()), "\(src)") }
        }
        #expect(
            trips.sorted() == [
                "agent/agent-lab-serve-q36-h1copy-r1-off-1.jsonl", "agent/agent-lfm4-r2.jsonl",
                "agent/agent-lfm4-r3.jsonl",
                "lab/2026-10-27-fm8-t7/agent-gptoss-fm8t7-pif2.jsonl",
            ])
        #expect(cases.count - trips.count >= 10)
    }

    /// Parses `text` in chunks of `size` as `Session` does (put per chunk, then finish), with and without the guard.
    static func parse(
        _ text: String, format: ToolCallFormat, promptTail: String, size: Int, suspect: AgentGuard.Call?
    ) -> (plain: [OutputEvent], guarded: [OutputEvent], trip: AgentGuard.Trip?) {
        let d = ToolCallFormatTests.dialect(format)
        let tools: [JSONValue] = [
            .object([
                ("type", .string("function")),
                (
                    "function",
                    .object([
                        ("name", .string("bash")),
                        (
                            "parameters",
                            .object([("properties", .object([("command", .object([("type", .string("string"))]))]))])
                        ),
                    ])
                ),
            ])
        ]
        var p = OutputParser(
            thinking: d.promptOpensReasoning(promptTail), tools: tools, idPrefix: "t", format: format,
            trimReasoningStart: d.reasoningOpen.map { promptTail.hasSuffix($0) } ?? false,
            reasoningOpen: d.reasoningOpen, reasoningClose: d.reasoningClose ?? "</think>",
            closeSeparator: d.reasoningCloseSeparator)
        var rounds: [[OutputEvent]] = []
        var i = text.startIndex
        while i < text.endIndex {
            let j = text.index(i, offsetBy: size, limitedBy: text.endIndex) ?? text.endIndex
            rounds.append(p.put(String(text[i..<j])))
            i = j
        }
        rounds.append(p.finish(incomplete: false))
        let r = filtered(rounds, suspect: suspect)
        return (Array(rounds.joined()), r.sent, r.trip)
    }

    /// qwenXML holds the header of a call with the suspect's name until its arguments: the events are the same when
    /// the call is not the repeat; when it is, what went before goes out and the call does not. Block formats and
    /// harmony send header and arguments together.
    @Test func eventsWithAndWithoutGuard() {
        let outputs: [(ToolCallFormat, String, String, String)] = [
            (
                .qwenXML, "<|im_start|>assistant\n<think>\n",
                "Reading.\n</think>\n\nLet me list.\n\n<tool_call>\n<function=bash>\n<parameter=command>\nCMD\n</parameter>\n</function>\n</tool_call>",
                "Let me list."
            ),
            (
                .hermes, "<|start_of_role|>assistant<|end_of_role|>",
                "Let me list.\n<tool_call>\n{\"name\": \"bash\", \"arguments\": {\"command\": \"CMD\"}}\n</tool_call>",
                "Let me list."
            ),
            (
                .harmony, "<|start|>assistant",
                "<|channel|>analysis<|message|>Reading.<|end|><|start|>assistant"
                    + "<|channel|>commentary to=functions.bash <|constrain|>json<|message|>{\"command\":\"CMD\"}",
                ""
            ),
        ]
        for (format, tail, template, before) in outputs {
            for size in [1, 2, 3, 7, 1000] {
                // Another command: identical events, no trip.
                var r = Self.parse(
                    template.replacingOccurrences(of: "CMD", with: "ls -a"), format: format, promptTail: tail,
                    size: size,
                    suspect: Self.bash)
                #expect(r.trip == nil && r.guarded == r.plain && r.plain.count >= 2, "\(format) \(size)")
                // The repeat: dropped, with everything before it sent.
                r = Self.parse(
                    template.replacingOccurrences(of: "CMD", with: "ls"), format: format, promptTail: tail, size: size,
                    suspect: Self.bash)
                #expect(r.trip == .repeatedCall(name: "bash"), "\(format) \(size)")
                let n = r.plain.firstIndex { if case .toolStart = $0 { true } else { false } } ?? r.plain.count
                #expect(r.guarded == Array(r.plain[..<n]), "\(format) \(size)")
                let content = r.guarded.compactMap { if case .content(let c) = $0 { c } else { nil } }.joined()
                #expect(content == before, "\(format) \(size)")
            }
        }
    }
}
