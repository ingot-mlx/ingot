import Foundation
import Testing

@testable import LabServe

/// Gemma 4 26B in pi sessions (mlx-community template): the 24 answers whose text reached the client with channel
/// marks. Resources/gemma4/agent-turns.json has each model output as written (reasoning, text and calls in order).
/// Read as vLLM's `gemma4` parser does, none leaves a mark or reasoning in the content, except two that no rule
/// fixes.
@Suite struct GemmaChannelTurnsTests {
    struct Turn: Decodable { let turn: String, finish: String, output: String, calls: [String] }

    static let tools =
        GemmaEmptyChannelTests.tools
        + [
            #"{"type":"function","function":{"name":"read","parameters":{"type":"object","properties":{"#
                + #""path":{"type":"string"}}}}}"#,
            #"{"type":"function","function":{"name":"edit","parameters":{"type":"object","properties":{"#
                + #""path":{"type":"string"},"edits":{"type":"array"}}}}}"#,
        ].compactMap { try? JSONValue.parse($0) }

    static func turns() throws -> [Turn] {
        let url = Bundle.module.url(forResource: "Resources", withExtension: nil)!
        let data = try Data(contentsOf: url.appendingPathComponent("gemma4/agent-turns.json"))
        return try JSONDecoder().decode([Turn].self, from: data)
    }

    /// Reasoning with no opening at all, closed by a stray `<channel|>`: by the time the mark arrives the text has
    /// been streamed (vLLM streams it as content too and drops the mark).
    static let unopened = "A3 #20"
    /// A loop cut by length that writes `<|channel|>`, which is not one of Gemma 4's marks.
    static let loop = "A2-followup #8"

    @Test func agentTurns() throws {
        let turns = try Self.turns()
        #expect(turns.count == 24)
        for t in turns {
            for s in ToolCallFormatTests.chunkings {
                let o = ToolCallFormatTests.run(
                    t.output, promptTail: "<|turn>model\n", format: .gemma4, tools: Self.tools, sizes: s)
                let at = "\(t.turn), chunks \(s)"
                #expect(o.calls.map(\.0) == t.calls, "\(at)")
                #expect(!o.reasoning.hasPrefix("thought"), "\(at)")
                switch t.turn {
                case Self.loop: continue
                case Self.unopened:
                    let text = t.output[..<t.output.range(of: "\n<channel|>")!.lowerBound]
                    #expect(o.content == text.trimmingCharacters(in: .whitespacesAndNewlines), "\(at)")
                default:
                    for mark in ["<|channel>", "<channel|>", "<tool_call|>", "thought\n"] {
                        #expect(!o.content.contains(mark), "\(at): \(o.content.prefix(200).debugDescription)")
                    }
                    #expect(["", "---", "---delete-instructions\n---"].contains(o.content), "\(at)")
                }
            }
        }
    }
}
