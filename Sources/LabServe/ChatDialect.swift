import Foundation

// Portions adapted from mlx-lm (MIT, Copyright (c) 2023 Apple Inc.; `tokenizer_utils.py`; see THIRD_PARTY_NOTICES.md).

/// Tool-call syntax a chat template writes, all parsed (OutputParser.swift, PythonicToolCalls.swift, ToolCallBlocks.swift,
/// HeaderChannels.swift).
public enum ToolCallFormat: String, Sendable {
    /// Qwen3.5/3.6 XML: `<tool_call>\n<function=NAME>\n<parameter=P>…` (mlx-lm `qwen3_coder`).
    case qwenXML
    /// LFM2 Python call list: `<|tool_call_start|>[f(a=1)]<|tool_call_end|>` (mlx-lm `pythonic`).
    case lfmPythonic
    /// GLM-4.7 XML: `<tool_call>NAME<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>` (GLM).
    case glm47
    /// Hermes JSON: `<tool_call>\n{"name": …, "arguments": …}\n</tool_call>` (Granite 4.1; mlx-lm `json_tools`).
    case hermes
    /// Gemma 4: `<|tool_call>call:NAME{key:value}<tool_call|>`.
    case gemma4
    /// Muse ATEM: `<atem:function_calls><atem:invoke name="…">…` (mlx-lm `muse_spark`).
    case atem
    /// gpt-oss harmony: `<|channel|>commentary to=functions.NAME <|constrain|>json<|message|>{…}` (read by headers, like
    /// ATEM).
    case harmony

    /// The server extracts these calls (the rest come out as content).
    public var parsed: Bool { self == .qwenXML || self == .atem || self == .harmony || blockMarks != nil }

    /// Marks around a block of calls that is read whole (nil: qwenXML, read as it streams; ATEM and harmony, read by
    /// headers).
    var blockMarks: (open: String, close: String)? {
        switch self {
        case .lfmPythonic: return ("<|tool_call_start|>", "<|tool_call_end|>")
        case .glm47, .hermes: return ("<tool_call>", "</tool_call>")
        case .gemma4: return ("<|tool_call>", "<tool_call|>")
        default: return nil
        }
    }
}

/// How a model talks, read from its chat template and vocabulary: the syntax of its tool calls, the marks around its
/// reasoning and the template parameter (and values) that carries `reasoning_effort`. The tool syntax follows mlx-lm's detection order
/// (`tokenizer_utils._infer_tool_parser`, 0.32.2) plus harmony; the reasoning marks, its `_infer_thinking`.
public struct ChatDialect: Equatable, Sendable {
    /// Tool-call syntax the template writes (nil: none that is known).
    public let toolFormat: ToolCallFormat?
    /// The template renders the request's tools.
    public let rendersTools: Bool
    /// Text that opens and closes the reasoning (`<think>`/`</think>`, or Gemma 4's `<|channel>thought`/`<channel|>`).
    /// Nil when the vocabulary has neither, and for harmony and ATEM, whose reasoning goes in channel headers.
    public let reasoningOpen: String?, reasoningClose: String?
    /// What the template writes between the reasoning and `reasoningClose` (`"\n"`: Gemma 4's
    /// `'\n<channel|>'`, ChatML's `'\n</think>'`); the parser cuts it off with the mark, so the reasoning it returns is
    /// rendered back as the model wrote it. Empty when the template writes the mark right after the reasoning.
    public let reasoningCloseSeparator: String
    /// The template takes `reasoning_effort` and knows `xhigh` (Qwen3.8 Flash-Next and 27B: xhigh, medium, low).
    public let reasoningEffortXHigh: Bool
    /// The template takes `reasoning_strength` instead of `reasoning_effort` (Muse: low, medium, high, xhigh, per its
    /// model card; the template writes it in the system block, `high` by default).
    public let reasoningStrength: Bool
    /// Start of the assistant's and the user's turns in the template, for the prefix cache (`InferenceEngine`'s
    /// `generationMarker`, point (a) of the snapshots, and `userMarker`, the end of the system block). Nil when the
    /// template's turns are not known: no such points.
    public let generationMarker: String?, userMarker: String?

    public init(
        toolFormat: ToolCallFormat?, rendersTools: Bool, reasoningOpen: String?, reasoningClose: String?,
        reasoningEffortXHigh: Bool, reasoningStrength: Bool = false, generationMarker: String? = nil,
        userMarker: String? = nil, reasoningCloseSeparator: String = ""
    ) {
        self.toolFormat = toolFormat
        self.rendersTools = rendersTools
        self.reasoningOpen = reasoningOpen
        self.reasoningClose = reasoningClose
        self.reasoningCloseSeparator = reasoningCloseSeparator
        self.reasoningEffortXHigh = reasoningEffortXHigh
        self.reasoningStrength = reasoningStrength
        self.generationMarker = generationMarker
        self.userMarker = userMarker
    }

    /// The dialect of a template; `hasToken` says whether a text is a token of the model (added or in the vocabulary).
    public static func detect(template t: String, hasToken: (String) -> Bool) -> ChatDialect {
        let format = toolFormat(template: t, hasToken: hasToken)
        var open: String?, close: String?
        if format != .harmony && format != .atem {
            if hasToken("<think>") && hasToken("</think>") {
                (open, close) = ("<think>", "</think>")
            } else if hasToken("<|channel>") && hasToken("<channel|>") {
                (open, close) = ("<|channel>thought", "<channel|>")
            }
        }
        let turns = turnMarkers(template: t)
        return ChatDialect(
            toolFormat: format, rendersTools: t.contains("tools"), reasoningOpen: open, reasoningClose: close,
            reasoningEffortXHigh: t.contains("reasoning_effort") && t.contains("xhigh"),
            reasoningStrength: t.contains("reasoning_strength") && !t.contains("reasoning_effort"),
            generationMarker: turns?.generation, userMarker: turns?.user,
            reasoningCloseSeparator: close.map { closeSeparator(template: t, close: $0) } ?? "")
    }

    /// `"\n"` when the template writes a newline right before the reasoning's closing mark (as `\n` in a string
    /// literal or as a literal newline), the separator the model learns to write there.
    static func closeSeparator(template t: String, close: String) -> String {
        t.contains("\\n" + close) || t.contains("\n" + close) ? "\n" : ""
    }

    /// The text that opens the assistant's turn (as the generation prompt writes it) and the user's, by the turn tokens
    /// the template uses. ChatML gives the markers the server always used.
    static func turnMarkers(template t: String) -> (generation: String, user: String)? {
        let known: [(String, String, String)] = [
            ("<|im_start|>", "<|im_start|>assistant\n", "<|im_start|>user"),  // ChatML: Qwen, LFM2.5
            ("<|turn>", "<|turn>model\n", "<|turn>user"),  // Gemma 4
            ("<start_of_turn>", "<start_of_turn>model\n", "<start_of_turn>user"),  // Gemma 3
            ("<|start|>", "<|start|>assistant", "<|start|>user"),  // harmony (gpt-oss), Muse
            ("<|assistant|>", "<|assistant|>", "<|user|>"),  // GLM
            ("<|start_of_role|>", "<|start_of_role|>assistant<|end_of_role|>", "<|start_of_role|>user<|end_of_role|>"),
        ]
        return known.first { t.contains($0.0) }.map { ($0.1, $0.2) }
    }

    static func toolFormat(template t: String, hasToken: (String) -> Bool) -> ToolCallFormat? {
        if t.contains("<|tool_call>") && t.contains("<tool_call|>") { return .gemma4 }
        if t.contains("<arg_key>") { return .glm47 }
        if t.contains("<|tool_list_start|>") || t.contains("<|tool_call_start|>") { return .lfmPythonic }
        if t.contains("<tool_call>\\n<function=") || t.contains("<tool_call>\n<function=") { return .qwenXML }
        if t.contains("<atem:function_calls>") { return .atem }
        if t.contains("<|channel|>") && t.contains("functions.") { return .harmony }
        if t.contains("<tool_call>") && t.contains("tool_call.name") { return .hermes }
        // No match in the template: the vocabulary (LFM2.5 conversions have the tokens without naming them).
        if hasToken("<|tool_call_start|>") && hasToken("<|tool_call_end|>") { return .lfmPythonic }
        return nil
    }

    /// Why tool calls will not be understood, for the log at load (nil if they will, or the template has no tools).
    public var warning: String? {
        guard rendersTools else { return nil }
        guard let f = toolFormat else {
            return "the chat template renders tools in a format lab serve does not know: tool calls come out as content"
        }
        return f.parsed ? nil : "tool calls in \(f.rawValue) format are not parsed yet: they come out as content"
    }

    /// The prompt ends inside the reasoning (Qwen leaves `<think>\n`; LFM2.5, `<think>`).
    public func promptOpensReasoning(_ prompt: String) -> Bool {
        guard let o = reasoningOpen else { return false }
        return prompt.hasSuffix(o) || prompt.hasSuffix(o + "\n")
    }

    /// Template parameter that receives the OpenAI API's `reasoning_effort`.
    public var reasoningKwarg: String { reasoningStrength ? "reasoning_strength" : "reasoning_effort" }

    /// `reasoning_effort` from the OpenAI API for the template: where it knows `xhigh`, "high" is its xhigh and
    /// "minimal" its low; as `reasoning_strength` (Muse: low, medium, high, xhigh), "minimal" and "none" are its low;
    /// elsewhere (gpt-oss: low, medium, high) as is.
    public func reasoningEffort(_ requested: String) -> String {
        if reasoningStrength { return ["minimal": "low", "none": "low"][requested] ?? requested }
        return reasoningEffortXHigh ? (["high": "xhigh", "minimal": "low"][requested] ?? requested) : requested
    }
}
