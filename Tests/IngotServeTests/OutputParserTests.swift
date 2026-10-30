import Foundation
import Testing

@testable import IngotServe

/// Output parser: reasoning, stop strings and Qwen3.6 tool calls, whole and split at random (streaming).
@Suite struct OutputParserTests {
    static let tools: [JSONValue] = [
        #"""
        {"type":"function","function":{"name":"bash","parameters":{"type":"object","properties":{
          "command":{"type":"string"},"timeout":{"type":"integer"},"flags":{"type":"array"},"verbose":{"type":"boolean"}}}}}
        """#,
        #"{"type":"function","function":{"name":"read","parameters":{"type":"object","properties":{"#
            + #""path":{"type":"string"},"offset":{"type":["number","null"]}}}}}"#,
    ].compactMap { try? JSONValue.parse($0) }

    struct Out: Equatable {
        var reasoning = "", content = "", calls: [ToolCall] = []
        var starts = 0
    }

    /// Feeds `text` to the parser in chunks of `sizes` characters (cycling) and collects the events.
    static func run(
        _ text: String, thinking: Bool, tools: [JSONValue]?, sizes: [Int], stops: [String] = [],
        format: ToolCallFormat = .qwenXML, trim: Bool = false
    ) -> Out {
        var p = OutputParser(thinking: thinking, tools: tools, idPrefix: "t", format: format, trimReasoningStart: trim)
        var stop = StopMatcher(stops)
        var o = Out()
        var argsByIndex: [Int: String] = [:], names: [Int: (String, String)] = [:]
        func take(_ evs: [OutputEvent]) {
            for e in evs {
                switch e {
                case .reasoning(let s): o.reasoning += s
                case .content(let s): o.content += s
                case .toolStart(let i, let id, let n): names[i] = (id, n); o.starts += 1
                case .toolArguments(let i, let a): argsByIndex[i, default: ""] += a
                }
            }
        }
        let chars = Array(text)
        var i = 0, k = 0
        while i < chars.count && !stop.stopped {
            let n = sizes[k % sizes.count]; k += 1
            let piece = String(chars[i..<min(chars.count, i + n)])
            i += n
            take(p.put(stop.put(piece)))
        }
        take(p.put(stop.finish()))
        take(p.finish(incomplete: false))
        o.calls = names.keys.sorted().map {
            ToolCall(id: names[$0]!.0, name: names[$0]!.1, arguments: argsByIndex[$0] ?? "")
        }
        #expect(o.calls == p.calls)
        #expect(o.content == p.content)
        return o
    }

    static let chunkings: [[Int]] = [[100000], [1], [2], [3], [5, 1, 7], [11, 2], [4, 9, 1, 1, 13]]

    @Test func reasoningThenContent() {
        let text = "I think a bit.\n</think>\n\nThe answer is 42."
        for s in Self.chunkings {
            let o = Self.run(text, thinking: true, tools: nil, sizes: s)
            #expect(o.reasoning == "I think a bit.\n")
            #expect(o.content == "The answer is 42.")
        }
        // No reasoning: everything is content (a stray </think> stays).
        let o = Self.run("Hello", thinking: false, tools: nil, sizes: [1])
        #expect(o.content == "Hello" && o.reasoning.isEmpty)
    }

    @Test func toolCallsStreamedAndWhole() {
        let text = """
            I will look.
            </think>

            First I list the directory.

            <tool_call>
            <function=bash>
            <parameter=command>
            ls -la /tmp | grep "a<b" && echo '</param'
            </parameter>
            <parameter=timeout>
            30
            </parameter>
            <parameter=flags>
            ["-x", 2]
            </parameter>
            <parameter=verbose>
            True
            </parameter>
            </function>
            </tool_call>
            <tool_call>
            <function=read>
            <parameter=path>
            /tmp/x y.txt
            </parameter>
            <parameter=offset>
            null
            </parameter>
            </function>
            </tool_call>
            """
        let calls = [
            ToolCall(
                id: "call_t_0", name: "bash",
                arguments:
                    #"{"command":"ls -la /tmp | grep \"a<b\" && echo '</param'","timeout":30,"flags":["-x",2],"verbose":true}"#
            ),
            ToolCall(id: "call_t_1", name: "read", arguments: #"{"path":"/tmp/x y.txt","offset":null}"#),
        ]
        for s in Self.chunkings {
            let o = Self.run(text, thinking: true, tools: Self.tools, sizes: s)
            #expect(o.reasoning == "I will look.\n", "chunks \(s)")
            #expect(o.content == "First I list the directory.", "chunks \(s)")
            #expect(o.calls == calls, "chunks \(s): \(o.calls)")
            #expect(o.starts == 2)
        }
    }

    @Test func toolCallInsideReasoningAndUnclosed() {
        // The model calls the tool without closing the reasoning; and the last call lacks </tool_call>.
        let text = "Necesito bash.\n<tool_call>\n<function=bash>\n<parameter=command>\npwd\n</parameter>\n</function>"
        for s in Self.chunkings {
            let o = Self.run(text, thinking: true, tools: Self.tools, sizes: s)
            #expect(o.reasoning == "Necesito bash.\n")
            #expect(o.content == "")
            #expect(o.calls == [ToolCall(id: "call_t_0", name: "bash", arguments: #"{"command":"pwd"}"#)])
        }
        // Without tools in the request, the XML is content.
        let o = Self.run(
            "<tool_call>\n<function=bash>\n</function>\n</tool_call>", thinking: false, tools: nil, sizes: [3])
        #expect(o.calls.isEmpty && o.content.hasPrefix("<tool_call>"))
    }

    /// LFM2.5: `<think>` without a newline in the prompt and calls as a Python list between special tokens.
    @Test func lfmPythonicToolCalls() {
        let text =
            "\nNeed to list.</think>I will look at it."
            + "<|tool_call_start|>[bash(command='ls -la \\'x\\'\\n', timeout=30, flags=[\"-a\", \"-l\"], verbose=True), "
            + "read(path=\"/tmp/a b\", offset=None)]<|tool_call_end|>"
        for sizes in [[1000], [1], [3, 7, 2], [5]] {
            let o = Self.run(text, thinking: true, tools: Self.tools, sizes: sizes, format: .lfmPythonic, trim: true)
            #expect(o.reasoning == "Need to list.")
            #expect(o.content == "I will look at it.")
            #expect(o.calls.map(\.name) == ["bash", "read"])
            #expect(
                o.calls.first?.arguments
                    == #"{"command":"ls -la 'x'\n","timeout":30,"flags":["-a","-l"],"verbose":true}"#)
            #expect(o.calls.last?.arguments == #"{"path":"/tmp/a b","offset":null}"#)
        }
        // Without tools in the request, the block stays as text; so does a block that cannot be parsed.
        let o2 = Self.run(
            "Hello<|tool_call_start|>not python<|tool_call_end|>", thinking: false, tools: Self.tools, sizes: [4],
            format: .lfmPythonic)
        #expect(o2.calls.isEmpty && o2.content.contains("not python"))
    }

    @Test func stopStrings() {
        for s in Self.chunkings {
            let o = Self.run("one two FIN three", thinking: false, tools: nil, sizes: s, stops: ["FIN", "zzz"])
            #expect(o.content == "one two", "chunks \(s)")
            let p = Self.run("one FI", thinking: false, tools: nil, sizes: s, stops: ["FIN"])
            #expect(p.content == "one FI")
        }
    }

    @Test func parameterConversion() {
        #expect(convertParameter("30", types: ["integer"]) == .int(30))
        #expect(convertParameter("30", types: ["string"]) == .string("30"))
        #expect(convertParameter("30", types: nil) == .int(30))
        #expect(convertParameter("hello", types: ["integer"]) == .string("hello"))
        #expect(convertParameter("False", types: ["boolean"]) == .bool(false))
        #expect(convertParameter("null", types: ["string", "null"]) == .null)
        #expect(convertParameter("{\"a\": 1}", types: ["object"]) == .object([("a", .int(1))]))
        #expect(convertParameter("\"x\"", types: nil) == .string("\"x\""))
        // The agent-task case: unescaped quotes inside newText.
        let bad =
            #"[{"oldText": "    if x:\n        pass", "newText": "    raise ValueError(f"Units out of order: {original!r}")\n"}]"#
        let v = convertParameter(bad, types: ["array"])
        #expect(
            v.array?.first?["newText"]?.string == #"    raise ValueError(f"Units out of order: {original!r}")"# + "\n")
        #expect(v.array?.first?["oldText"]?.string == "    if x:\n        pass")
        // With no array/object declared, or if it is not JSON anyway, the text stays.
        #expect(convertParameter(bad, types: ["string"]) == .string(bad))
        #expect(convertParameter("[{\"a\": ", types: ["array"]) == .string("[{\"a\": "))
        #expect(convertParameter("[1, 2]", types: ["array"]) == .array([.int(1), .int(2)]))
    }
}
