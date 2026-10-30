import Foundation
import Jinja

/// Chat template of the model (`chat_template.jinja` or `chat_template` from `tokenizer_config.json`), rendered with
/// swift-jinja with the same options as transformers (`trim_blocks`, `lstrip_blocks`, `tojson` not escaping
/// non-ASCII and keeping keys in order).
public final class ChatTemplate: @unchecked Sendable {
    let template: Template
    /// The template text (for `ChatDialect.detect`).
    public let source: String
    /// Special tokens that transformers passes to the template (bos_token, eos_token, ...).
    let specials: [String: String]
    /// The clock `strftime_now` reads (transformers: `datetime.now()`). Tests set a fixed instant so the templates
    /// that write today's date (gpt-oss) render the same on any day.
    var now: @Sendable () -> Date = { Date() }

    public init(directory: URL) throws {
        let jinja = directory.appendingPathComponent("chat_template.jinja")
        let cfg =
            (try? JSONSerialization.jsonObject(
                with: Data(contentsOf: directory.appendingPathComponent("tokenizer_config.json")))) as? [String: Any]
            ?? [:]
        var src: String
        if let s = try? String(contentsOf: jinja, encoding: .utf8) {
            src = s
        } else if let s = cfg["chat_template"] as? String {
            src = s
        } else {
            throw BPETokenizer.TokError.unsupported("the model has no chat template")
        }
        var sp: [String: String] = [:]
        for k in ["bos_token", "eos_token", "pad_token", "unk_token"] {
            if let s = cfg[k] as? String {
                sp[k] = s
            } else if let d = cfg[k] as? [String: Any], let s = d["content"] as? String {
                sp[k] = s
            }
        }
        specials = sp
        source = src
        template = try Template(
            Self.renameStrftimeNow(Self.applyWhitespaceControl(Self.stripGenerationTags(src))),
            with: .init(lstripBlocks: true, trimBlocks: true))
    }

    /// swift-jinja 2.5.2 does not strip the whitespace before `{%-`, `{{-` or `{#-` when the text before it ends in a
    /// literal `{` (Gemma 4's tool declarations came out as `parameters:{ properties:{ command:…` instead of
    /// `parameters:{properties:{command:…`). That `{` is written as the expression `{{ '{' }}` and the whitespace
    /// removed, which is what Jinja renders; nothing else in the template changes.
    static func applyWhitespaceControl(_ s: String) -> String {
        guard let re = try? NSRegularExpression(pattern: #"\{\s+(\{[%{#]-)"#) else { return s }
        return re.stringByReplacingMatches(
            in: s, range: NSRange(s.startIndex..., in: s), withTemplate: "{{ '{' }}$1")
    }

    /// `{% generation %}` / `{% endgeneration %}` (transformers extension to mark what the assistant writes;
    /// they do nothing when rendering; LFM2.5 uses them): swift-jinja does not know them and they are removed.
    static func stripGenerationTags(_ s: String) -> String {
        guard let re = try? NSRegularExpression(pattern: #"\{%-?\s*(end)?generation\s*-?%\}"#) else { return s }
        return re.stringByReplacingMatches(in: s, range: NSRange(s.startIndex..., in: s), withTemplate: "")
    }

    /// The name `render` gives its own `strftime_now`: swift-jinja 2.5.2 copies its built-ins over the render context
    /// (`Interpreter.interpret` builds a root `Environment`), so a context `strftime_now` would never be seen.
    static let strftimeNowName = "__ingot_strftime_now"

    /// Calls to `strftime_now(` go to `strftimeNowName`.
    static func renameStrftimeNow(_ s: String) -> String {
        guard let re = try? NSRegularExpression(pattern: #"\bstrftime_now(\s*\()"#) else { return s }
        return re.stringByReplacingMatches(
            in: s, range: NSRange(s.startIndex..., in: s), withTemplate: strftimeNowName + "$1")
    }

    /// `messages` and `tools` as JSON (keeping key order); extra `kwargs` (enable_thinking, ...).
    public func render(
        messages: [JSONValue], tools: [JSONValue]? = nil, addGenerationPrompt: Bool = true,
        kwargs: [(String, JSONValue)] = []
    ) throws -> String {
        var ctx: [String: Value] = [:]
        for (k, v) in specials { ctx[k] = .string(v) }
        ctx["messages"] = .array(messages.map(Self.value))
        if let tools { ctx["tools"] = .array(tools.map(Self.value)) }
        ctx["add_generation_prompt"] = .boolean(addGenerationPrompt)
        for (k, v) in kwargs { ctx[k] = Self.value(v) }
        let now = self.now
        ctx[Self.strftimeNowName] = .function { args, _, _ in
            guard args.count == 1, case .string(let format) = args[0] else {
                throw JinjaError.runtime("strftime_now takes one string argument")
            }
            return .string(Self.pythonStrftime(format, now()))
        }
        return try template.render(ctx)
    }

    /// `date` formatted as Python's `datetime.strftime` (C `strftime`, local time, "C" locale), the one transformers
    /// calls; swift-jinja's own `strftime_now` maps the codes to DateFormatter and gets some wrong (`%I`, `%w`, `%U`).
    static func pythonStrftime(_ format: String, _ date: Date) -> String {
        var t = time_t(date.timeIntervalSince1970.rounded(.down))
        var tm = tm()
        localtime_r(&t, &tm)
        var buf = [CChar](repeating: 0, count: 256 + 4 * format.utf8.count)
        let n = strftime(&buf, buf.count, format, &tm)  // the process never calls setlocale: "C" locale
        return String(decoding: buf.prefix(n).map { UInt8(bitPattern: $0) }, as: UTF8.self)
    }

    static func value(_ j: JSONValue) -> Value {
        switch j {
        case .null: return .null
        case .bool(let b): return .boolean(b)
        case .int(let i): return .int(i)
        case .double(let d): return .double(d)
        case .string(let s): return .string(s)
        case .array(let a): return .array(a.map(value))
        case .object(let kv):
            var o = OrderedDictionary<ObjectKey, Value>()
            for (k, v) in kv { o[.string(k)] = value(v) }
            return .object(o)
        }
    }
}

/// What the server needs from the model to talk: tokenizer, template, its dialect and end tokens.
public struct ChatModelInfo: @unchecked Sendable {
    public let tokenizer: BPETokenizer
    public let template: ChatTemplate
    /// Tool-call syntax, reasoning marks and `reasoning_effort` values of the template.
    public let dialect: ChatDialect
    /// End tokens: `eos_token_id` from generation_config.json and config.json, and the tokenizer's `eos_token`.
    public let eosIds: Set<Int>
    /// Default sampling of the model (generation_config.json).
    public let defaultTemperature: Float?, defaultTopP: Float?, defaultTopK: Int?
    /// min-p and penalties from generation_config.json (nil = neutral: 0, 1, 0, 0).
    public let defaultMinP: Float?, defaultRepetitionPenalty: Float?, defaultPresencePenalty: Float?,
        defaultFrequencyPenalty: Float?

    public init(directory dir: URL) throws {
        tokenizer = try BPETokenizer(directory: dir)
        template = try ChatTemplate(directory: dir)
        let tok = tokenizer
        dialect = ChatDialect.detect(template: template.source) { tok.added[$0] != nil || tok.vocab[$0] != nil }
        var eos = Set<Int>()
        func ids(_ v: Any?) -> [Int] { (v as? [Int]) ?? (v as? Int).map { [$0] } ?? [] }
        let gen =
            (try? JSONSerialization.jsonObject(
                with: Data(contentsOf: dir.appendingPathComponent("generation_config.json")))) as? [String: Any] ?? [:]
        let cfg =
            (try? JSONSerialization.jsonObject(with: Data(contentsOf: dir.appendingPathComponent("config.json"))))
            as? [String: Any] ?? [:]
        eos.formUnion(ids(gen["eos_token_id"]))
        eos.formUnion(ids(cfg["eos_token_id"]))
        eos.formUnion(ids((cfg["text_config"] as? [String: Any])?["eos_token_id"]))
        if let e = template.specials["eos_token"], let id = tokenizer.added[e] ?? tokenizer.vocab[e] { eos.insert(id) }
        eosIds = eos
        defaultTemperature = (gen["temperature"] as? NSNumber)?.floatValue
        defaultTopP = (gen["top_p"] as? NSNumber)?.floatValue
        defaultTopK = (gen["top_k"] as? NSNumber)?.intValue
        defaultMinP = (gen["min_p"] as? NSNumber)?.floatValue
        defaultRepetitionPenalty = (gen["repetition_penalty"] as? NSNumber)?.floatValue
        defaultPresencePenalty = (gen["presence_penalty"] as? NSNumber)?.floatValue
        defaultFrequencyPenalty = (gen["frequency_penalty"] as? NSNumber)?.floatValue
    }

    /// Chat prompt: special tokens only come from the template. Strings of messages and tools are escaped
    /// before rendering (`BPETokenizer.escapeSpecials`), so an `<|im_end|>` inside a file or a tool result is
    /// tokenized as text (like vLLM, or llama.cpp with parse_special=false for the content). Without specials in
    /// the content, the text and ids are exactly those of transformers.
    /// `text` is the prompt without escapes.
    public func prompt(
        messages: [JSONValue], tools: [JSONValue]? = nil, addGenerationPrompt: Bool = true,
        kwargs: [(String, JSONValue)] = []
    ) throws -> (text: String, ids: [Int]) {
        func esc(_ j: JSONValue) -> JSONValue {
            switch j {
            case .string(let s): return .string(tokenizer.escapeSpecials(s))
            case .array(let a): return .array(a.map(esc))
            case .object(let kv): return .object(kv.map { ($0.0, esc($0.1)) })
            default: return j
            }
        }
        let rendered = try template.render(
            messages: messages.map(esc), tools: tools?.map(esc),
            addGenerationPrompt: addGenerationPrompt, kwargs: kwargs)
        return (tokenizer.unescapeSpecials(rendered), tokenizer.encodeRendered(rendered))
    }
}
