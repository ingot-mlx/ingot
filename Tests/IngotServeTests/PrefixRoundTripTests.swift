import Foundation
import Ingot
import Testing

@testable import IngotServe

/// The parser is the inverse of the template: after a turn, the next prompt (the history plus the assistant
/// message as ingot serve returned it, plus the tool results) starts with the previous prompt plus the text the model
/// wrote. Only then does the engine continue from its live state (`InferenceEngine.start`: the live state, prompt
/// and answer, must be a prefix of the new prompt); otherwise it reads the answer again (the Gemma 4 `\n` before
/// `<channel|>` and the stripped GLM values). The texts are what each template writes for a
/// message (what the model learns to write) and, per format, outputs as the models wrote them in pi sessions.
@Suite struct PrefixRoundTripTests {
    /// The templates of TokenizerTemplateTests plus the one mlx-community ships with Gemma 4 26B (the one ingot serve
    /// used in the pi sessions; it differs from google's), with google's tokenizer.
    static let mlxGemma = "mlx-community/gemma-4-26b-a4b-it-4bit"
    static let models = TokenizerTemplateTests.models + [mlxGemma]

    /// Cases that cannot hold, with the reason (each one must still fail: a known issue that stops happening fails).
    static let exceptions: [String: String] = [
        "google/gemma-4-26B-A4B-it content-call-think": gemmaContentAfterCall,
        "mlx-community/gemma-4-26b-a4b-it-4bit content-call-think": gemmaContentAfterCall,
        "openai/gpt-oss-20b observed":
            "harmony: the template writes the call back with another header (`to=` before the channel) and the "
            + "arguments through tojson; no parser can fix it (rolling the live state back)",
    ]

    static let gemmaContentAfterCall =
        "Gemma 4: the template writes a turn's text after its calls and, once the tool answers, puts the response "
        + "between them; the model ends its turn at the call, so the text is never written there (no parser fix)"

    static let tools = (try? ToolCallFormatTests.reference("GLM-4.7-Flash"))?["tools"]?.array ?? []

    static func call(_ id: String, _ name: String, _ args: JSONValue) -> JSONValue {
        .object([
            ("id", .string(id)), ("type", .string("function")),
            ("function", .object([("name", .string(name)), ("arguments", args)])),
        ])
    }

    /// Assistant turns the template renders (OpenAI shape; arguments as objects): reasoning plus calls with values
    /// that end in newlines, several calls, text before a call and values with surrounding spaces.
    static let turns: [(String, JSONValue)] = [
        (
            "call-newline",
            .object([
                ("role", .string("assistant")), ("content", .string("")),
                ("reasoning_content", .string("Miro el fichero.\nDos líneas.")),
                (
                    "tool_calls",
                    .array([
                        call(
                            "call_1", "bash",
                            .object([
                                ("command", .string("cat > f.txt <<EOF\nhola\nEOF\n")), ("timeout", .int(5)),
                            ]))
                    ])
                ),
            ])
        ),
        (
            "two-calls",
            .object([
                ("role", .string("assistant")), ("content", .string("")),
                ("reasoning_content", .string("Leo los dos.")),
                (
                    "tool_calls",
                    .array([
                        call("call_1", "read", .object([("path", .string("/a"))])),
                        call(
                            "call_2", "read",
                            .object([
                                ("path", .string("/tmp/«x» y.txt")), ("offset", .double(1.5)),
                                ("flags", .array([.string("a"), .string("b c")])),
                            ])),
                    ])
                ),
            ])
        ),
        (
            "content-call",
            .object([
                ("role", .string("assistant")), ("content", .string("Voy a mirar.")),
                ("reasoning_content", .string("Breve.")),
                ("tool_calls", .array([call("call_1", "bash", .object([("command", .string("ls -la"))]))])),
            ])
        ),
        (
            "spaces",
            .object([
                ("role", .string("assistant")), ("content", .string("")),
                ("reasoning_content", .string("Con espacios.")),
                (
                    "tool_calls",
                    .array([call("call_1", "bash", .object([("command", .string("  echo a  \n\n"))]))])
                ),
            ])
        ),
    ]

    /// Outputs as the models wrote them in pi sessions (after the generation prompt, without the end token).
    static let observed: [ToolCallFormat: String] = [
        // Gemma 4 26B, a pi session: the reasoning ends in "\n" and the model writes no other separator.
        .gemma4: "<|channel>thought\nVoy a crear el test.\n<channel|><|tool_call>call:bash{command:<|\"|>ls<|\"|>}"
            + "<tool_call|>",
        // GLM-4.7-Flash: a file written with its final newline.
        .glm47: "Lo escribo.</think><tool_call>bash<arg_key>command</arg_key><arg_value>cat > f.py <<'EOF'\n"
            + "print(1)\nEOF\n</arg_value></tool_call>",
        // gpt-oss-20b: `to=` after the channel and compact JSON.
        .harmony: "<|channel|>analysis<|message|>Miro.<|end|><|start|>assistant<|channel|>commentary "
            + "to=functions.bash <|constrain|>json<|message|>{\"command\":\"ls\"}",
    ]

    /// The repo's model info; the mlx-community Gemma template goes with google's tokenizer and configs.
    static func info(_ model: String) throws -> ChatModelInfo? {
        if model == mlxGemma {
            guard let g = TokenizerTemplateTests.repoDirectory("google/gemma-4-26B-A4B-it") else { return nil }
            let dir = FileManager.default.temporaryDirectory.appendingPathComponent("prefix-\(UUID().uuidString)")
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: dir) }
            for f in ["tokenizer.json", "tokenizer_config.json", "config.json", "generation_config.json"] {
                try? FileManager.default.createSymbolicLink(
                    at: dir.appendingPathComponent(f), withDestinationURL: g.appendingPathComponent(f))
            }
            let src = Bundle.module.url(forResource: "Resources", withExtension: nil)!
                .appendingPathComponent(ModelPaths.shortName(model)).appendingPathComponent("chat_template.jinja")
            try FileManager.default.copyItem(at: src, to: dir.appendingPathComponent("chat_template.jinja"))
            return try TokenizerTemplateTests.chatInfo(dir)
        }
        guard let dir = TokenizerTemplateTests.repoDirectory(model) else { return nil }
        return try TokenizerTemplateTests.chatInfo(dir)
    }

    /// The assistant message ingot serve returns for `text` (as `completionResponse` builds it), parsed as `Session`
    /// does after `prompt`, fed in chunks of `size` characters.
    static func parsed(_ text: String, prompt: String, info: ChatModelInfo, size: Int) -> JSONValue {
        let d = info.dialect
        var p = OutputParser(
            thinking: d.promptOpensReasoning(prompt), tools: tools, idPrefix: "t", format: d.toolFormat,
            trimReasoningStart: d.reasoningOpen.map { prompt.hasSuffix($0) } ?? false,
            reasoningOpen: d.reasoningOpen, reasoningClose: d.reasoningClose ?? "</think>",
            closeSeparator: d.reasoningCloseSeparator)
        var rest = Substring(text)
        while !rest.isEmpty {
            _ = p.put(String(rest.prefix(size)))
            rest = rest.dropFirst(size)
        }
        _ = p.finish(incomplete: false)
        var msg: [(String, JSONValue)] = [
            ("role", .string("assistant")),
            ("content", p.content.isEmpty && !p.calls.isEmpty ? .null : .string(p.content)),
        ]
        if !p.reasoningText.isEmpty { msg.append(("reasoning_content", .string(p.reasoningText))) }
        if !p.calls.isEmpty {
            msg.append(("tool_calls", .array(p.calls.map { call($0.id, $0.name, .string($0.arguments)) })))
        }
        return .object(msg)
    }

    static func body(_ messages: [JSONValue], _ kwargs: [(String, JSONValue)]) -> JSONValue {
        .object([("messages", .array(messages)), ("tools", .array(tools))] + kwargs)
    }

    /// Checks one output: the next prompt, with the parsed message and one tool result per call, starts with the
    /// prompt plus `text`, in text and in ids. Returns the failure, if any.
    static func check(
        _ text: String, history: [JSONValue], prompt p1: (text: String, ids: [Int]), kwargs: [(String, JSONValue)],
        info: ChatModelInfo
    ) throws -> String? {
        for size in [1_000_000, 1, 3] {
            let msg = parsed(text, prompt: p1.text, info: info, size: size)
            var next = history + [msg]
            for (i, c) in (msg["tool_calls"]?.array ?? []).enumerated() {
                next.append(
                    .object([
                        ("role", .string("tool")), ("tool_call_id", c["id"] ?? .string("")),
                        ("content", .string("resultado \(i)")),
                    ]))
            }
            if msg["tool_calls"] == nil {
                next.append(.object([("role", .string("user")), ("content", .string("Sigue."))]))
            }
            let p2 = try ChatServer.prompt(body: body(next, kwargs), info: info)
            let live = p1.text + text
            guard p2.text.hasPrefix(live) else {
                return "chunks of \(size): \(firstDiff(p2.text, live)) (message \(msg.compact()))"
            }
            guard p2.ids.starts(with: info.tokenizer.encode(live)) else { return "chunks of \(size): ids" }
        }
        return nil
    }

    static func firstDiff(_ a: String, _ b: String) -> String {
        let x = Array(a), y = Array(b)
        let i = (0..<min(x.count, y.count)).first { x[$0] != y[$0] } ?? min(x.count, y.count)
        return
            "character \(i) of \(y.count): next prompt \(String(x[max(0, i - 40)..<min(x.count, i + 20)]).debugDescription) "
            + "versus live \(String(y[max(0, i - 40)..<min(y.count, i + 20)]).debugDescription)"
    }

    /// No failure, or the failure an exception expects.
    static func expect(_ failure: String?, _ key: String) {
        if let reason = exceptions[key] {
            withKnownIssue(Comment(rawValue: reason)) { #expect(failure == nil, "\(key): \(failure ?? "")") }
        } else {
            #expect(failure == nil, "\(key): \(failure ?? "")")
        }
    }

    @Test(arguments: models) func nextPromptStartsWithLiveState(model: String) throws {
        guard let info = try Self.info(model) else { return }
        let ends = info.eosIds.map { info.tokenizer.decode([$0]) }.filter { !$0.isEmpty }
        let history: [JSONValue] = [
            .object([("role", .string("system")), ("content", .string("Eres un asistente."))]),
            .object([("role", .string("user")), ("content", .string("Mira los ficheros."))]),
        ]
        var checked = 0
        for (variant, kwargs) in [
            ("", [(String, JSONValue)]()), ("-think", [("enable_thinking", .bool(true))]),
            ("-nothink", [("enable_thinking", .bool(false))]),
        ] {
            let r1 = try ChatServer.prompt(body: Self.body(history, kwargs), info: info)
            let p1 = (text: r1.text, ids: r1.ids)
            for (name, turn) in Self.turns {
                var m = turn
                // Without reasoning the model writes none.
                if variant == "-nothink", case .object(let kv) = m {
                    m = .object(kv.filter { $0.0 != "reasoning_content" })
                }
                let msgs = try ChatServer.normalize(history + [m], dialect: info.dialect)
                let full = try info.prompt(
                    messages: msgs, tools: Self.tools, addGenerationPrompt: false, kwargs: kwargs
                )
                .text
                // A template whose generation prompt is not how it renders the turn (thinking off closes an empty
                // reasoning block the turn does not have): the model's text is not defined by it.
                guard full.hasPrefix(p1.text) else { continue }
                var text = String(full.dropFirst(p1.text.count))
                if let r = ends.compactMap({ text.range(of: $0) }).min(by: { $0.lowerBound < $1.lowerBound }) {
                    text = String(text[..<r.lowerBound])
                }
                let failure = try Self.check(text, history: history, prompt: p1, kwargs: kwargs, info: info)
                Self.expect(failure, "\(model) \(name)\(variant)")
                checked += 1
            }
            guard variant == "-think", let f = info.dialect.toolFormat, let text = Self.observed[f] else { continue }
            let failure = try Self.check(text, history: history, prompt: p1, kwargs: kwargs, info: info)
            Self.expect(failure, "\(model) observed")
        }
        #expect(checked >= 4, "\(model): only \(checked) cases rendered")
    }
}
