import Foundation
import Testing

@testable import IngotServe

/// gpt-oss harmony beyond the template's own turns (ToolCallFormatTests has the round trip): channels, recipients in the
/// role or in the channel, no tools, cut-off outputs, and the request messages `normalize` hands to the template. The
/// expectations follow the harmony format (mlx-lm 0.32.2 has no parser for it).
@Suite struct HarmonyTests {
    static let repo = "openai/gpt-oss-20b"
    static let tools = (try? ToolCallFormatTests.reference("gpt-oss-20b"))?["tools"]?.array ?? []

    static func run(_ text: String, tools: [JSONValue] = tools, sizes: [Int]) -> ToolCallFormatTests.Out {
        ToolCallFormatTests.run(
            text, promptTail: "<|start|>assistant", format: .harmony, tools: tools, sizes: sizes)
    }

    @Test func channelsAndRecipients() {
        for sizes in ToolCallFormatTests.chunkings {
            // Reasoning and answer.
            var o = Self.run(
                "<|channel|>analysis<|message|>Pienso.<|end|><|start|>assistant<|channel|>final<|message|>Hola.",
                sizes: sizes)
            #expect(o.reasoning == "Pienso." && o.content == "Hola." && o.calls.isEmpty, "\(sizes)")
            // A preamble (commentary without recipient) is answer text; the recipient in the channel part, with
            // `<|constrain|>json`, is a call, and its values are fitted to the schema.
            o = Self.run(
                "<|channel|>analysis<|message|>Miro.<|end|><|start|>assistant<|channel|>commentary<|message|>"
                    + "Voy a mirar.<|end|><|start|>assistant<|channel|>commentary to=functions.bash <|constrain|>json"
                    + #"<|message|>{"command":"ls -la","timeout":"30"}"#, sizes: sizes)
            #expect(o.reasoning == "Miro." && o.content == "Voy a mirar.", "\(sizes)")
            #expect(ToolCallFormatTests.same(o.calls, [("bash", #"{"command":"ls -la","timeout":30}"#)]), "\(sizes)")
            // The recipient in the role part (the template's own form), right after the generation prompt.
            o = Self.run(
                #" to=functions.read<|channel|>commentary json<|message|>{"path": "/a", "flags": ["x"]}"#, sizes: sizes)
            #expect(o.content.isEmpty && o.reasoning.isEmpty, "\(sizes)")
            #expect(ToolCallFormatTests.same(o.calls, [("read", #"{"flags":["x"],"path":"/a"}"#)]), "\(sizes)")
            // A recipient that is not a function (built-in tools) follows its channel.
            o = Self.run(#"<|channel|>analysis to=browser.search<|message|>{"q": "x"}"#, sizes: sizes)
            #expect(o.reasoning == #"{"q": "x"}"# && o.calls.isEmpty, "\(sizes)")
        }
    }

    @Test func noToolsAndBrokenOutputs() {
        let call = #" to=functions.bash<|channel|>commentary json<|message|>{"command": "ls"}"#
        for sizes in ToolCallFormatTests.chunkings {
            // Without tools in the request a call is answer text.
            var o = Self.run(call, tools: [], sizes: sizes)
            #expect(o.calls.isEmpty && o.content == #"{"command": "ls"}"#, "\(sizes)")
            // A body that is not a JSON object is still a call (the recipient says so), with its raw text as the
            // arguments (`canonical` marks it `invalid:`): the client reports the error to the model.
            // The gpt-oss-20b form seen with pi: recipient in the analysis channel, an edit object closed too early.
            o = Self.run(" to=functions.bash<|channel|>commentary json<|message|>not json", sizes: sizes)
            #expect(o.content.isEmpty && ToolCallFormatTests.same(o.calls, [("bash", "invalid: not json")]), "\(sizes)")
            let broken = #"{"path":"a.py","edits":[{"oldText":"x\n"},"newText":"y"}]}"#
            o = Self.run(
                "<|channel|>analysis<|message|>Fix it.<|end|><|start|>assistant<|channel|>analysis to=functions.edit code"
                    + "<|message|>" + broken, sizes: sizes)
            #expect(o.reasoning == "Fix it." && o.content.isEmpty, "\(sizes)")
            #expect(ToolCallFormatTests.same(o.calls, [("edit", "invalid: " + broken)]), "\(sizes)")
            // No header at all: everything is the answer.
            o = Self.run("Just text\nwith lines.", sizes: sizes)
            #expect(o.content == "Just text\nwith lines." && o.reasoning.isEmpty, "\(sizes)")
            // Cut off inside the reasoning, or inside the next header.
            o = Self.run("<|channel|>analysis<|message|>Pien", sizes: sizes)
            #expect(o.reasoning == "Pien" && o.content.isEmpty, "\(sizes)")
            o = Self.run("<|channel|>analysis<|message|>x<|end|><|start|>assistant<|channel|>fin", sizes: sizes)
            #expect(o.reasoning == "x" && o.content.isEmpty && o.calls.isEmpty, "\(sizes)")
        }
    }

    /// The request side: an OpenAI conversation (reasoning in `reasoning_content`, `reasoning_effort` at the top level)
    /// renders as transformers renders the same conversation with the reasoning in `thinking` (ref/tok_ref.py, chat.json).
    /// In a turn with tool calls the template rejects `content` next to `thinking`: the reasoning is kept.
    @Test func normalizeFeedsTemplate() throws {
        guard let dir = TokenizerTemplateTests.repoDirectory(Self.repo) else { return }
        let info = try TokenizerTemplateTests.chatInfo(dir)
        let cases = try TokenizerTemplateTests.resource(Self.repo, "chat.json").array!
        var checked = 0
        for c in cases where c["add_generation_prompt"]!.bool! {
            let name = c["name"]!.string!
            let msgs: [JSONValue] = c["messages"]!.array!.map { m in
                guard var kv = m.object, let i = kv.firstIndex(where: { $0.0 == "thinking" }) else { return m }
                kv[i].0 = "reasoning_content"
                if m["tool_calls"] != nil, let j = kv.firstIndex(where: { $0.0 == "content" }) {
                    kv[j].1 = .string("Voy a listarlo.")
                }
                return .object(kv)
            }
            var body: [(String, JSONValue)] = [("messages", .array(msgs))]
            if let t = c["tools"], !t.isNull { body.append(("tools", t)) }
            let kw = c["kwargs"]!.object!
            body.append(("chat_template_kwargs", .object(kw.filter { $0.0 != "reasoning_effort" })))
            if let re = kw.first(where: { $0.0 == "reasoning_effort" }) { body.append(re) }
            let p = try ChatServer.prompt(body: .object(body), info: info)
            #expect(
                p.text == c["text"]!.string!,
                "\(name): \(TokenizerTemplateTests().firstDiff(p.text, c["text"]!.string!))")
            checked += 1
        }
        #expect(checked >= 10)
    }
}
