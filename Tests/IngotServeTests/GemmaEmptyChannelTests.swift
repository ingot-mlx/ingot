import Foundation
import Testing

@testable import IngotServe

/// Gemma 4 26B with the mlx-community template, after a new user message: the model writes empty reasoning
/// channels in its answer, sometimes without the opening token. The outputs are the exact bytes a client received as
/// text, with the call each turn made; none may come back as content.
@Suite struct GemmaEmptyChannelTests {
    static let tools: [JSONValue] = [
        #"{"type":"function","function":{"name":"write","parameters":{"type":"object","properties":{"#
            + #""content":{"type":"string"},"path":{"type":"string"}}}}}"#,
        #"{"type":"function","function":{"name":"bash","parameters":{"type":"object","properties":{"#
            + #""command":{"type":"string"}}}}}"#,
    ].compactMap { try? JSONValue.parse($0) }

    static let write = #"<|tool_call>call:write{content:<|"|>x = 1<|"|>,path:<|"|>duration.py<|"|>}<tool_call|>"#
    static let writeCall = ("write", #"{"content":"x = 1","path":"duration.py"}"#)
    static let empty = "<|channel>thought\n<channel|>"

    /// Model output (text the parser sees) and the content it must return.
    static let cases: [(String, String, String)] = [
        ("open, after text", "---\n" + empty + write, "---"),
        ("open, after two lines", "---delete-instructions\n---\n" + empty + write, "---delete-instructions\n---"),
        ("no open token", "thought\n<channel|>" + write, ""),
        ("no open, then open", "thought\n<channel|>" + empty + write, ""),
        ("no open, think", "think\n<channel|>" + write, ""),
        ("repeated with open", String(repeating: empty, count: 28) + write, ""),
        ("repeated without open", String(repeating: "thought\n<channel|>", count: 34) + write, ""),
        ("without tools call", "thought\n<channel|>Done.", "Done."),
    ]

    static func run(_ text: String, promptTail: String, sizes: [Int], tools: [JSONValue]? = Self.tools)
        -> ToolCallFormatTests.Out
    {
        ToolCallFormatTests.run(text, promptTail: promptTail, format: .gemma4, tools: tools ?? [], sizes: sizes)
    }

    @Test func emptyChannelsAreNotContent() {
        // After a user message the prompt ends in "<|turn>model\n"; after a tool response, in "<tool_response|>".
        for tail in ["<|turn>model\n", "<tool_response|>"] {
            for (name, text, content) in Self.cases {
                for s in ToolCallFormatTests.chunkings {
                    let o = Self.run(text, promptTail: tail, sizes: s)
                    #expect(o.content == content, "\(name), chunks \(s)")
                    #expect(o.reasoning == "", "\(name), chunks \(s)")
                    if text.contains("<|tool_call>") {
                        #expect(o.calls.map(\.0) == [Self.writeCall.0], "\(name), chunks \(s)")
                        #expect(o.calls.first?.1 == Self.writeCall.1, "\(name), chunks \(s)")
                    }
                }
            }
        }
    }

    /// Reasoning first, then an empty channel before the call: the reasoning stays, the channel goes.
    @Test func emptyChannelAfterReasoning() {
        let text = "<|channel>thought\nI rewrite it.\n<channel|>" + Self.empty + Self.write
        for s in ToolCallFormatTests.chunkings {
            let o = Self.run(text, promptTail: "<|turn>model\n", sizes: s)
            #expect(o.reasoning == "I rewrite it.", "chunks \(s)")
            #expect(o.content == "", "chunks \(s)")
            #expect(o.calls.map(\.0) == ["write"], "chunks \(s)")
        }
    }

    /// Without tools in the request the same channels are dropped; text that only looks like one stays.
    @Test func noToolsAndLookalikes() {
        let kept = ["thought about it", "ok\nDone.", "thoughts on it", "Thought\nyes", "<|chan", "a <|channel b"]
        for s in ToolCallFormatTests.chunkings {
            #expect(
                Self.run(
                    "thought\n<channel|>" + Self.empty + "Hi.", promptTail: "<|turn>model\n", sizes: s,
                    tools: nil
                ).content == "Hi.", "chunks \(s)")
            for t in kept {
                let o = Self.run(t, promptTail: "<|turn>model\n", sizes: s, tools: nil)
                #expect(o.content == t && o.reasoning == "", "\(t), chunks \(s)")
            }
        }
    }

    /// Reasoning without its opening token, a channel in the middle of the answer and stray closing marks, read as
    /// vLLM's `gemma4` parser does: (output, reasoning, content), with and without tools.
    static let vllmCases: [(String, String, String)] = [
        ("thought", "", ""),
        ("thought\n", "", ""),
        ("thought\nThe user wants a fix.\n<channel|>Done.", "The user wants a fix.", "Done."),
        ("thought\nI check it.\n<channel|>" + write, "I check it.", ""),
        ("Hello <|channel>thought\nnot empty<channel|>Bye", "not empty", "Hello Bye"),
        ("<|channel>", "", ""),
        ("<|channel>h-thought\nx\n<channel|>Hi", "h-thought\nx", "Hi"),
        ("a <channel|> b", "", "a  b"),
        ("<tool_call|><|channel>thought\nFix it.\n<channel|>" + write, "Fix it.", ""),
    ]

    @Test func unopenedReasoningAsVLLM() {
        for (text, reasoning, content) in Self.vllmCases {
            for tools in [Self.tools, nil] {
                // Without tools the call stays as text.
                if tools == nil && text.contains("<|tool_call>") { continue }
                for s in ToolCallFormatTests.chunkings {
                    let o = Self.run(text, promptTail: "<|turn>model\n", sizes: s, tools: tools)
                    let where_ = "\(text.debugDescription), tools \(tools != nil), chunks \(s)"
                    #expect(o.reasoning == reasoning, "\(where_): \(o.reasoning.debugDescription)")
                    #expect(o.content == content, "\(where_): \(o.content.debugDescription)")
                    if text.contains("<|tool_call>") { #expect(o.calls.map(\.0) == ["write"], "\(where_)") }
                }
            }
        }
    }

    /// Other dialects are untouched: `thought\n<channel|>` is plain text for a `<think>` model.
    @Test func otherDialectsUntouched() {
        var p = OutputParser(
            thinking: false, tools: nil, idPrefix: "t", format: .qwenXML, reasoningOpen: "<think>",
            reasoningClose: "</think>")
        var content = ""
        for e in p.put("thought\n<channel|>Hi") + p.finish(incomplete: false) {
            if case .content(let c) = e { content += c }
        }
        #expect(content == "thought\n<channel|>Hi")
    }
}
