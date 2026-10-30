import Foundation
import Testing

@testable import LabServe

/// Tool-call formats of the families with tool calls against ref/tool_ref.py (Resources/<repo>/tools.json, always present):
/// (a) round trip with the template: from what the template writes after the generation prompt, the parser recovers
/// the reasoning, the content and the calls of the message it came from, whole and split into chunks; (b) the same
/// calls mlx-lm's parser reads, there and in hand-written outputs, except where we differ on purpose (`differs`). mlx-lm
/// has no harmony parser ("mlx" null): gpt-oss has only (a), and its hand-written cases are in HarmonyTests.
@Suite struct ToolCallFormatTests {
    /// Repos with a tools.json reference.
    static let repos = [
        "GLM-4.7-Flash", "granite-4.1-3b", "gemma-4-26B-A4B-it", "Muse-Glimmer-30B", "gpt-oss-20b", "LFM3-1.7B",
    ]

    /// Hard cases where we differ from mlx-lm on purpose, with our calls (name, canonical JSON arguments). Hermes and
    /// Gemma 4 fit the values to the declared types, as the Pythonic and Qwen formats do (mlx-lm leaves them as
    /// written); ATEM keeps a declared string as text and its surrounding spaces (mlx-lm reads it as JSON or strips it);
    /// GLM-4.7 keeps a string value's whitespace, which its template writes back verbatim (mlx-lm strips it).
    static let differs: [String: [String: [(String, String)]]] = [
        "glm47": ["trailing-newline": [("bash", #"{"command":"cat > f.py <<EOF\nprint(1)\nEOF\n","timeout":7}"#)]],
        "hermes": ["number-as-string": [("bash", #"{"command":"42","timeout":7}"#)]],
        "gemma4": ["number-as-string": [("bash", #"{"command":"42","timeout":7}"#)]],
        "atem": [
            "number-as-string": [("bash", #"{"command":"42"}"#)],
            "spaces-kept": [("bash", #"{"command":"  echo \"a\"\n"}"#)],
        ],
    ]

    static let chunkings: [[Int]] = [[1_000_000], [1], [2], [3], [5, 1, 7], [11, 2], [4, 9, 1, 1, 13]]

    struct Out {
        var reasoning = "", content = ""
        var calls: [(String, String)] = []
    }

    /// The dialect the server would detect for this format (the reasoning marks of the templates in tools.json).
    static func dialect(_ format: ToolCallFormat) -> ChatDialect {
        let marks: (String?, String?) =
            switch format {
            case .gemma4: ("<|channel>thought", "<channel|>")
            case .atem, .harmony: (nil, nil)
            default: ("<think>", "</think>")
            }
        return ChatDialect(
            toolFormat: format, rendersTools: true, reasoningOpen: marks.0, reasoningClose: marks.1,
            reasoningEffortXHigh: false, reasoningCloseSeparator: format == .gemma4 ? "\n" : "")
    }

    /// Runs the parser as ChatServer builds it after a prompt ending in `promptTail`, feeding `text` in chunks.
    static func run(_ text: String, promptTail: String, format: ToolCallFormat, tools: [JSONValue], sizes: [Int])
        -> Out
    {
        let d = dialect(format)
        var p = OutputParser(
            thinking: d.promptOpensReasoning(promptTail), tools: tools, idPrefix: "t", format: format,
            trimReasoningStart: d.reasoningOpen.map { promptTail.hasSuffix($0) } ?? false,
            reasoningOpen: d.reasoningOpen, reasoningClose: d.reasoningClose ?? "</think>",
            closeSeparator: d.reasoningCloseSeparator)
        var o = Out()
        var names: [Int: String] = [:], args: [Int: String] = [:]
        func take(_ evs: [OutputEvent]) {
            for e in evs {
                switch e {
                case .reasoning(let s): o.reasoning += s
                case .content(let s): o.content += s
                case .toolStart(let i, _, let n): names[i] = n
                case .toolArguments(let i, let a): args[i, default: ""] += a
                }
            }
        }
        var rest = Substring(text), k = 0
        while !rest.isEmpty {
            let n = sizes[k % sizes.count]
            k += 1
            take(p.put(String(rest.prefix(n))))
            rest = rest.dropFirst(n)
        }
        take(p.finish(incomplete: false))
        o.calls = names.keys.sorted().map { (names[$0]!, canonical(args[$0] ?? "")) }
        #expect(o.calls.map(\.0) == p.calls.map(\.name))
        return o
    }

    /// JSON with sorted keys (Gemma 4 sorts the arguments; mlx-lm keeps the written order).
    static func canonical(_ v: JSONValue) -> String {
        switch v {
        case .object(let kv):
            return "{"
                + kv.sorted { $0.0 < $1.0 }.map { JSONValue.string($0.0).compact() + ":" + canonical($0.1) }
                .joined(separator: ",") + "}"
        case .array(let a): return "[" + a.map(canonical).joined(separator: ",") + "]"
        default: return v.compact()
        }
    }
    static func canonical(_ s: String) -> String { (try? JSONValue.parse(s)).map(canonical) ?? "invalid: \(s)" }

    static func calls(_ v: JSONValue?) -> [(String, String)] {
        (v?.array ?? []).map { ($0["name"]?.string ?? "", canonical($0["arguments"] ?? .object([]))) }
    }

    static func same(_ a: [(String, String)], _ b: [(String, String)]) -> Bool {
        a.count == b.count && zip(a, b).allSatisfy { $0.0 == $1.0 && $0.1 == $1.1 }
    }

    static func reference(_ repo: String) throws -> JSONValue {
        let url = Bundle.module.url(forResource: "Resources", withExtension: nil)!
            .appendingPathComponent(repo).appendingPathComponent("tools.json")
        return try JSONValue.parse(Data(contentsOf: url))
    }

    @Test(arguments: repos) func roundTripWithTemplate(repo: String) throws {
        let ref = try Self.reference(repo)
        let format = try #require(ToolCallFormat(rawValue: ref["format"]!.string!))
        let tools = ref["tools"]!.array!
        let spans = ref["spans"]!.array!
        #expect(spans.count >= 4)
        for s in spans {
            let name = s["name"]!.string!, text = s["text"]!.string!, tail = s["prompt_tail"]!.string!
            let e = s["expected"]!
            let want = Self.calls(e["calls"])
            if s["mlx"]?.isNull == false {
                #expect(Self.same(Self.calls(s["mlx"]), want), "\(repo) \(name): mlx-lm reads other calls")
            }
            for sizes in Self.chunkings {
                let o = Self.run(text, promptTail: tail, format: format, tools: tools, sizes: sizes)
                // Exact: the reasoning is rendered back as returned (Gemma 4 kept the "\n" before <channel|>).
                #expect(
                    o.reasoning == e["reasoning"]!.string!, "\(repo) \(name) \(sizes): \(o.reasoning.debugDescription)")
                #expect(o.content == e["content"]!.string!, "\(repo) \(name) \(sizes): \(o.content.debugDescription)")
                #expect(Self.same(o.calls, want), "\(repo) \(name) \(sizes): \(o.calls) versus \(want)")
            }
        }
    }

    @Test(arguments: repos) func hardCasesAsMLX(repo: String) throws {
        let ref = try Self.reference(repo)
        let fmt = ref["format"]!.string!
        let format = try #require(ToolCallFormat(rawValue: fmt))
        let tools = ref["tools"]!.array!
        for h in ref["hard"]!.array! {
            let name = h["name"]!.string!, text = h["text"]!.string!
            let want = Self.differs[fmt]?[name] ?? Self.calls(h["mlx"])
            for sizes in Self.chunkings {
                // After a generation prompt that leaves the reasoning open only if the case closes it unopened.
                let t = text.contains("</think>") && !text.contains("<think>") ? "<think>\n" : ""
                let o = Self.run(text, promptTail: t, format: format, tools: tools, sizes: sizes)
                #expect(Self.same(o.calls, want), "\(repo) \(name) \(sizes): \(o.calls) versus \(want)")
            }
        }
    }
}

/// ATEM message headers beyond the template's own turns: reasoning and answer, no tools, noise.
@Suite struct HeaderChannelTests {
    static let tools = (try? ToolCallFormatTests.reference("Muse-Glimmer-30B"))?["tools"]?.array ?? []

    @Test func headersRouteBodies() {
        let call =
            #" to=bash<|message|><atem:function_calls>\n<atem:invoke name="bash">\n<atem:parameter name="command">ls"#
            + "</atem:parameter>\n</atem:invoke>\n</atem:function_calls>"
        for sizes in ToolCallFormatTests.chunkings {
            var o = ToolCallFormatTests.run(
                " to=self<|message|>Breve.<|eom|><|start|>assistant to=user<|message|>Hola, ¿qué tal?",
                promptTail: "", format: .atem, tools: Self.tools, sizes: sizes)
            #expect(o.reasoning == "Breve." && o.content == "Hola, ¿qué tal?" && o.calls.isEmpty, "\(sizes)")
            // Without tools in the request a tool recipient is answer text.
            o = ToolCallFormatTests.run(call, promptTail: "", format: .atem, tools: [], sizes: sizes)
            #expect(o.calls.isEmpty && o.content.hasPrefix("<atem:function_calls>"), "\(sizes)")
            // No header at all: everything is the answer.
            o = ToolCallFormatTests.run(
                "Just text\nwith lines.", promptTail: "", format: .atem, tools: Self.tools, sizes: sizes)
            #expect(o.content == "Just text\nwith lines." && o.reasoning.isEmpty, "\(sizes)")
            // Cut off inside the reasoning, or inside a header.
            o = ToolCallFormatTests.run(
                " to=self<|message|>Pien", promptTail: "", format: .atem, tools: Self.tools, sizes: sizes)
            #expect(o.reasoning == "Pien" && o.content.isEmpty, "\(sizes)")
            o = ToolCallFormatTests.run(
                " to=self<|message|>x<|eom|><|start|>assistant to=us", promptTail: "", format: .atem,
                tools: Self.tools, sizes: sizes)
            #expect(o.reasoning == "x" && o.content.isEmpty, "\(sizes)")
        }
    }
}
