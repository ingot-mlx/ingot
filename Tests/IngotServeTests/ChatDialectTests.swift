import Foundation
import Ingot
import Testing

@testable import IngotServe

/// Chat dialects: what each family's template says (tool-call syntax, reasoning marks, `reasoning_effort`), and the
/// reasoning the model opens itself in the output.
@Suite struct ChatDialectTests {
    struct Expected: Sendable, CustomStringConvertible {
        let model: String
        let format: ToolCallFormat?
        let open: String?
        var xhigh = false
        var strength = false
        var warns = false
        var description: String { model }
    }

    static let think = "<think>"
    /// One template per tier 1 family (TokenizerTemplateTests.models; Flash-Next's is also the Qwen3.8 27B's), from the
    /// Hugging Face cache: under the CI a missing repo fails, elsewhere it is skipped.
    static let expected: [Expected] = [
        .init(model: TokenizerTemplateTests.heretic, format: .qwenXML, open: think),
        .init(model: "Qwen/Qwen3.8-Flash-Next", format: .qwenXML, open: think, xhigh: true),
        .init(model: "LiquidAI/LFM2.5-8B-A1B-MLX-4bit", format: .lfmPythonic, open: think),
        .init(model: "google/gemma-4-26B-A4B-it", format: .gemma4, open: "<|channel>thought"),
        .init(model: "meta-models/Muse-Glimmer-30B", format: .atem, open: nil, strength: true),
        .init(model: "openai/gpt-oss-20b", format: .harmony, open: nil),
        .init(model: "zai-org/GLM-4.7-Flash", format: .glm47, open: think),
        // Granite 4.1: Hermes JSON tool calls; `<think>` is in the vocabulary (the template does not reason).
        .init(model: "ibm-granite/granite-4.1-3b", format: .hermes, open: think),
        // Nemotron-H (Nano 4B): Qwen3-Coder XML tool calls in ChatML; the prompt opens `<think>\n` by default.
        .init(model: TokenizerTemplateTests.nemotron, format: .qwenXML, open: think),
        // LFM3: Qwen3-Coder XML tool calls in ChatML (the template never names <|tool_call_start|>); the prompt opens
        // `<think>` by default.
        .init(model: TokenizerTemplateTests.lfm3, format: .qwenXML, open: think),
    ]

    static func directory(_ model: String) -> URL? { TokenizerTemplateTests.repoDirectory(model) }

    @Test(arguments: expected) func detectsTemplateDialect(e: Expected) throws {
        guard let dir = Self.directory(e.model) else { return }
        let d = try ChatModelInfo(directory: dir).dialect
        #expect(d.toolFormat == e.format)
        #expect(d.reasoningOpen == e.open)
        #expect(d.reasoningClose == (e.open == nil ? nil : e.open == Self.think ? "</think>" : "<channel|>"))
        #expect(d.reasoningEffortXHigh == e.xhigh)
        #expect(d.reasoningStrength == e.strength)
        #expect(d.reasoningKwarg == (e.strength ? "reasoning_strength" : "reasoning_effort"))
        #expect((d.warning != nil) == e.warns, "\(d.warning ?? "no warning")")
    }

    /// Prefix cache markers: ChatML templates (Qwen, LFM2.5) give the ids the server always used; in every
    /// template's reference conversations (Resources/<model>/chat.json) the generation marker falls at the start of the
    /// generation prompt and the user marker at the start of the first user turn, in ids as in text.
    @Test(arguments: expected) func prefixCacheMarkers(e: Expected) throws {
        guard let dir = Self.directory(e.model) else { return }
        let info = try ChatModelInfo(directory: dir)
        let d = info.dialect, tok = info.tokenizer
        let gen = try #require(d.generationMarker), user = try #require(d.userMarker)
        if info.template.source.contains("<|im_start|>") {
            #expect(tok.encode(gen) == tok.encode("<|im_start|>assistant\n"))
            #expect(tok.encode(user) == tok.encode("<|im_start|>user"))
        }
        let gm = tok.encode(gen).map { UInt32($0) }, um = tok.encode(user).map { UInt32($0) }
        guard let cases = (try? TokenizerTemplateTests.resource(e.model, "chat.json"))?.array else { return }
        var checked = 0
        for c in cases where c["add_generation_prompt"]?.bool == true {
            let text = c["text"]!.string!, ids = c["ids"]!.array!.map { UInt32($0.intValue!) }
            let name = "\(e.model) \(c["name"]!.string!)"
            guard let g = text.range(of: gen, options: .backwards) else { continue }
            let gs = try #require(InferenceEngine.generationStart(ids, marker: gm), "\(name): no generation marker")
            #expect(
                tok.decode(ids[..<gs].map { Int($0) }) == String(text[..<g.upperBound]), "\(name): generation marker")
            if let u = text.range(of: user) {
                let us = try #require(Self.firstIndex(of: um, in: ids), "\(name): no user marker")
                #expect(tok.decode(ids[..<us].map { Int($0) }) == String(text[..<u.lowerBound]), "\(name): user marker")
            }
            checked += 1
        }
        #expect(checked >= 4, "\(e.model): \(checked) conversations with the markers")
    }

    /// The OpenAI `reasoning_effort` of a request reaches Muse's template as `reasoning_strength` (the template's
    /// default, high, when absent); an explicit `chat_template_kwargs.reasoning_strength` wins.
    @Test func museReasoningStrength() throws {
        let model = "meta-models/Muse-Glimmer-30B"
        guard let dir = Self.directory(model) else { return }
        let info = try ChatModelInfo(directory: dir)
        func render(_ extra: [(String, JSONValue)]) throws -> String {
            let msg = JSONValue.object([("role", .string("user")), ("content", .string("Hola"))])
            return try ChatServer.prompt(body: .object([("messages", .array([msg]))] + extra), info: info).text
        }
        #expect(try render([]).contains("Reasoning strength: high."))
        for (asked, shown) in [("low", "low"), ("medium", "medium"), ("xhigh", "xhigh"), ("minimal", "low")] {
            let text = try render([("reasoning_effort", .string(asked))])
            #expect(text.contains("Reasoning strength: \(shown)."), "\(asked)")
        }
        let kw = JSONValue.object([("reasoning_strength", .string("medium"))])
        let both = try render([("reasoning_effort", .string("low")), ("chat_template_kwargs", kw)])
        #expect(both.contains("Reasoning strength: medium."))
    }

    static func firstIndex(of m: [UInt32], in ids: [UInt32]) -> Int? {
        guard !m.isEmpty, ids.count >= m.count else { return nil }
        return (0...(ids.count - m.count)).first { Array(ids[$0..<($0 + m.count)]) == m }
    }

    @Test func reasoningEffortByTemplate() {
        func dialect(xhigh: Bool) -> ChatDialect {
            ChatDialect(
                toolFormat: .qwenXML, rendersTools: true, reasoningOpen: "<think>", reasoningClose: "</think>",
                reasoningEffortXHigh: xhigh)
        }
        #expect(dialect(xhigh: true).reasoningEffort("high") == "xhigh")
        #expect(dialect(xhigh: true).reasoningEffort("minimal") == "low")
        #expect(dialect(xhigh: true).reasoningEffort("medium") == "medium")
        #expect(dialect(xhigh: false).reasoningEffort("high") == "high")
        #expect(dialect(xhigh: false).reasoningEffort("minimal") == "minimal")
        let muse = ChatDialect(
            toolFormat: .atem, rendersTools: true, reasoningOpen: nil, reasoningClose: nil, reasoningEffortXHigh: false,
            reasoningStrength: true)
        #expect(muse.reasoningKwarg == "reasoning_strength")
        for v in ["low", "medium", "high", "xhigh"] { #expect(muse.reasoningEffort(v) == v) }
        #expect(muse.reasoningEffort("minimal") == "low" && muse.reasoningEffort("none") == "low")
        let d = dialect(xhigh: false)
        #expect(d.promptOpensReasoning("<|im_start|>assistant\n<think>\n"))
        #expect(d.promptOpensReasoning("<|im_start|>assistant\n<think>"))
        #expect(!d.promptOpensReasoning("<|im_start|>assistant\n<think>\n\n</think>\n\n"))
    }

    /// Output split into chunks of the given sizes (the last size repeats), through a parser that may see the model
    /// open its reasoning: (reasoning, content, calls).
    static func parse(
        _ out: String, sizes: [Int], open: String? = "<think>", close: String = "</think>",
        tools: [JSONValue]? = nil, format: ToolCallFormat? = .qwenXML
    ) -> (String, String, [ToolCall]) {
        var p = OutputParser(
            thinking: false, tools: tools, idPrefix: "t", format: format, reasoningOpen: open, reasoningClose: close)
        var r = "", c = ""
        func take(_ ev: [OutputEvent]) {
            for e in ev {
                switch e {
                case .reasoning(let s): r += s
                case .content(let s): c += s
                default: break
                }
            }
        }
        var rest = Substring(out), i = 0
        while !rest.isEmpty {
            let n = sizes[min(i, sizes.count - 1)]
            take(p.put(String(rest.prefix(n))))
            rest = rest.dropFirst(n)
            i += 1
        }
        take(p.finish(incomplete: false))
        return (r, c, p.calls)
    }

    static let splits: [[Int]] = [[1_000_000], [1], [2], [3], [5, 1, 7], [7, 2]]

    @Test func modelOpensReasoning() {
        for sizes in Self.splits {
            var (r, c, _) = Self.parse("<think>\nI reason.\n</think>\n\nThe answer.", sizes: sizes)
            #expect(r == "I reason.\n" && c == "The answer.", "\(sizes)")
            (r, c, _) = Self.parse("\n  <think>x</think>y", sizes: sizes)
            #expect(r == "x" && c == "y", "\(sizes)")
            // Not the mark: everything is content, whitespace included.
            (r, c, _) = Self.parse("<thinking about it", sizes: sizes)
            #expect(r.isEmpty && c == "<thinking about it", "\(sizes)")
            (r, c, _) = Self.parse("Plain answer <think>later</think>.", sizes: sizes)
            #expect(r.isEmpty && c == "Plain answer <think>later</think>.", "\(sizes)")
            // Cut off inside the mark or the reasoning.
            (r, c, _) = Self.parse("<thi", sizes: sizes)
            #expect(r.isEmpty && c == "<thi", "\(sizes)")
            (r, c, _) = Self.parse("<think>\nunfinished", sizes: sizes)
            #expect(r == "unfinished" && c.isEmpty, "\(sizes)")
            // Gemma 4's thought channel.
            (r, c, _) = Self.parse(
                "<|channel>thought\nPlan.<channel|>Done.", sizes: sizes, open: "<|channel>thought", close: "<channel|>")
            #expect(r == "Plan." && c == "Done.", "\(sizes)")
            // No mark in the dialect: as before.
            (r, c, _) = Self.parse("<think>x</think>y", sizes: sizes, open: nil)
            #expect(r.isEmpty && c == "<think>x</think>y", "\(sizes)")
        }
    }

    /// Calls after reasoning the model opened (Qwen XML), and calls with no known format: they stay as content.
    @Test func callsAfterOpenedReasoning() {
        let tools: [JSONValue] = [
            .object([
                ("type", .string("function")),
                (
                    "function",
                    .object([
                        ("name", .string("bash")),
                        (
                            "parameters",
                            .object([
                                ("type", .string("object")),
                                ("properties", .object([("command", .object([("type", .string("string"))]))])),
                            ])
                        ),
                    ])
                ),
            ])
        ]
        let xml =
            "<think>\nList it.\n</think>\n\n<tool_call>\n<function=bash>\n<parameter=command>\nls\n</parameter>\n"
            + "</function>\n</tool_call>"
        for sizes in Self.splits {
            let (r, c, calls) = Self.parse(xml, sizes: sizes, tools: tools)
            #expect(r == "List it.\n" && c.isEmpty, "\(sizes)")
            #expect(calls.map(\.name) == ["bash"] && calls.first?.arguments == #"{"command":"ls"}"#, "\(sizes)")
            let (_, nc, ncalls) = Self.parse(xml, sizes: sizes, tools: tools, format: nil)
            #expect(ncalls.isEmpty, "\(sizes)")
            #expect(nc == String(xml.split(separator: "\n\n", maxSplits: 1)[1]), "\(sizes)")
        }
    }
}
