import Foundation
import MetalLab

/// OpenAI-compatible server over an InferenceEngine (one model per process; one client at a time, requests waiting in
/// a serial queue, unless the engine batches: then they join the running ones in the same rounds, BatchServe.swift).
/// `POST /v1/chat/completions` (with and without stream), `GET /v1/models`, `GET /health`.
public final class ChatServer: @unchecked Sendable {
    public let engine: InferenceEngine
    public let info: ChatModelInfo
    public let modelId: String
    let engineQueue = DispatchQueue(label: "engine")
    let created = Int(Date().timeIntervalSince1970)
    private let lock = NSLock()
    private var counter = 0
    private var waiting = 0
    /// Batching: requests for the engine loop and whether it is running.
    let requests = EngineQueue()
    private var loopRunning = false
    public var log: @Sendable (String) -> Void = { print($0) }
    /// If non-nil, saves the body and prompt of each request in this folder (NNNN-body.json, NNNN-prompt.txt).
    /// They contain whole conversations: for debugging only.
    public var dumpDirectory: URL?
    /// Agent guards (AgentGuard.swift): stop a response that has fallen into a token loop, with
    /// `finish_reason: "length"`, and drop the third identical tool call in a row with the same result, ending the turn
    /// with `"stop"` and a short notice in `content`. They never change tokens. Off by default (a neutral server); on
    /// (`lab serve --agent-guards on`) for agent sessions.
    public var agentGuards = false

    public init(engine: InferenceEngine, info: ChatModelInfo, modelId: String) {
        self.engine = engine
        self.info = info
        self.modelId = modelId
    }

    public func handle(_ req: HTTPRequest, _ conn: HTTPConnection) {
        switch (req.method, req.path) {
        case ("GET", "/health"), ("GET", "/v1/health"):
            conn.respond(200, json: #"{"status":"ok"}"#)
        case ("GET", "/v1/models"), ("GET", "/models"):
            let m = JSONValue.object([
                ("id", .string(modelId)), ("object", .string("model")), ("created", .int(created)),
                ("owned_by", .string("metal-lab")), ("max_model_len", .int(engine.model.maxCtx)),
            ])
            conn.respond(200, json: JSONValue.object([("object", .string("list")), ("data", .array([m]))]).compact())
        case ("POST", "/v1/chat/completions"), ("POST", "/chat/completions"):
            do {
                let job = try prepare(req.body)
                lock.lock(); waiting += 1; let ahead = waiting - 1; lock.unlock()
                if ahead > 0 { log("[\(job.id)] queued (\(ahead) ahead)") }
                // Stream: headers now and SSE comments while it waits, so a client with an idle timeout (pi: 300 s
                // without headers) does not drop a request stuck behind long ones.
                if job.stream { conn.startStream() }
                let keepAlive = job.stream ? StreamKeepAlive(conn) : nil
                if engine.batchSlots > 1 {
                    enqueueBatched(job, conn, keepAlive: keepAlive)
                } else {
                    engineQueue.async { [self] in
                        lock.lock(); waiting -= 1; lock.unlock()
                        run(job, conn, keepAlive: keepAlive)
                    }
                }
            } catch let e as APIError {
                log("400: \(e.message)")
                conn.respond(e.code, json: e.json)
            } catch {
                conn.respond(400, json: APIError(400, "\(error)").json)
            }
        case (_, "/v1/chat/completions"), (_, "/v1/models"), (_, "/health"):
            conn.respond(405, json: APIError(405, "method not allowed").json)
        default:
            conn.respond(404, json: APIError(404, "unknown route: \(req.path)").json)
        }
    }

    // MARK: - Request

    struct Job {
        var id: String
        var prompt: [UInt32]
        var promptText: String
        var thinking: Bool
        var tools: [JSONValue]?
        var maxTokens: Int
        var sampling: SamplingParams
        var stops: [String]
        var stream: Bool
        var includeUsage: Bool
        /// Debugging (not part of OpenAI): returns the prompt and output ids (`lab_tokens`).
        var returnTokens: Bool
        var messages = 0
        /// Agent guards: the call whose third repeat ends the turn (nil: not armed).
        var repeatedCall: AgentGuard.Call?
        /// `--dump-requests`: the file prefix of this request (`NNNN`), for its `NNNN-output.json`.
        var dumpName: String?
    }

    /// Runs `f` on the engine queue (behind the request in progress).
    public func withEngine<T>(_ f: () throws -> T) rethrows -> T { try engineQueue.sync(execute: f) }

    /// Prompt ids that a request would generate (tests: lab-dev prefixcache).
    public func promptIds(body: Data) throws -> [UInt32] { try prepare(body).prompt }

    func prepare(_ body: Data) throws -> Job {
        let b: JSONValue
        do { b = try JSONValue.parse(body) } catch { throw APIError(400, "invalid JSON body: \(error)") }
        guard b.object != nil else { throw APIError(400, "the body must be a JSON object") }
        guard b["messages"]?.array?.isEmpty == false else {
            throw APIError(400, "messages must be a non-empty list")
        }
        try Self.rejectUnsupported(b)
        let (text, promptIds, msgs, tools) = try Self.prompt(body: b, info: info)
        let ids = promptIds.map { UInt32($0) }
        // Debugging (lab serve --dump-requests DIR): the body and prompt of each request (and its output when it ends).
        var dumpName: String?
        if let dir = dumpDirectory {
            let n = String(format: "%04d", counter + 1)
            dumpName = n
            try? body.write(to: dir.appendingPathComponent("\(n)-body.json"))
            try? text.write(to: dir.appendingPathComponent("\(n)-prompt.txt"), atomically: true, encoding: .utf8)
        }
        // Qwen leaves `<think>\n` at the end of the prompt; LFM2.5, `<think>` (always reasons).
        let thinking = info.dialect.promptOpensReasoning(text)
        // Sampling: what the request asks for or the model's generation_config (as splash).
        func num(_ k: String) -> Double? { b[k]?.number }
        var temp = Float(num("temperature") ?? Double(info.defaultTemperature ?? 1))
        guard temp >= 0 && temp <= 2 else { throw APIError(400, "temperature must be in [0, 2]") }
        if temp > 0 && temp < 0.01 { temp = 0.01 }
        let topP = Float(num("top_p") ?? Double(info.defaultTopP ?? 1))
        guard topP > 0 && topP <= 1 else { throw APIError(400, "top_p must be in (0, 1]") }
        var topK = b["top_k"]?.intValue ?? info.defaultTopK ?? 0
        if topK < 0 { topK = 0 }
        // min-p and penalties (vLLM/OpenAI semantics, no window): request, else generation_config.json.
        let minP = Float(num("min_p") ?? Double(info.defaultMinP ?? 0))
        guard minP >= 0 && minP <= 1 else { throw APIError(400, "min_p must be in [0, 1]") }
        let rep = Float(num("repetition_penalty") ?? Double(info.defaultRepetitionPenalty ?? 1))
        guard rep > 0 && rep.isFinite else { throw APIError(400, "repetition_penalty must be > 0") }
        let pres = Float(num("presence_penalty") ?? Double(info.defaultPresencePenalty ?? 0))
        guard pres >= -2 && pres <= 2 else { throw APIError(400, "presence_penalty must be in [-2, 2]") }
        let freq = Float(num("frequency_penalty") ?? Double(info.defaultFrequencyPenalty ?? 0))
        guard freq >= -2 && freq <= 2 else { throw APIError(400, "frequency_penalty must be in [-2, 2]") }
        let seed = b["seed"]?.intValue.map { UInt64(bitPattern: Int64($0)) }
        var stops: [String] = []
        if let s = b["stop"]?.string { stops = [s] } else if let a = b["stop"]?.array { stops = a.compactMap(\.string) }
        let room = engine.model.maxCtx - ids.count - 2 * 8
        guard room > 0 else {
            throw APIError(
                400,
                "the prompt has \(ids.count) tokens and does not fit in the server max context (\(engine.model.maxCtx)); "
                    + "start lab serve with a larger --max-ctx or trim the conversation",
                type: "context_length_exceeded")
        }
        var maxTokens = room
        if let m = (b["max_completion_tokens"] ?? b["max_tokens"])?.intValue {
            guard m > 0 else { throw APIError(400, "max_tokens must be positive") }
            maxTokens = min(m, room)
        }
        lock.lock(); counter += 1; let n = counter; lock.unlock()
        let id = String(format: "%@%04d", String(UInt32(created) & 0xffff, radix: 16), n)
        return Job(
            id: id, prompt: ids, promptText: text, thinking: thinking, tools: tools, maxTokens: maxTokens,
            sampling: SamplingParams(
                temperature: temp, topP: topP, topK: topK, seed: seed, minP: minP, repetitionPenalty: rep,
                presencePenalty: pres, frequencyPenalty: freq), stops: stops,
            stream: b["stream"]?.bool ?? false, includeUsage: b["stream_options"]?["include_usage"]?.bool ?? false,
            returnTokens: b["lab_return_tokens"]?.bool ?? false, messages: msgs.count,
            repeatedCall: agentGuards && tools != nil ? AgentGuard.suspect(messages: msgs) : nil, dumpName: dumpName)
    }

    /// `--dump-requests`: what a request generated (`NNNN-output.json`), to compare the ids the model wrote with those
    /// of the next prompt (ref/live_diff.py). `outputIds` are all the emitted tokens, the end token included.
    static func dumpOutput(
        promptIds: [UInt32], outputIds: [UInt32], finish: String, cached: Int, cacheSource: String,
        rolledBackFrom: Int?, liveMiss: String?
    ) -> JSONValue {
        .object([
            ("prompt_ids", .array(promptIds.map { .int(Int($0)) })),
            ("output_ids", .array(outputIds.map { .int(Int($0)) })),
            ("finish", .string(finish)), ("cached", .int(cached)), ("cache_source", .string(cacheSource)),
            ("rolled_back_from", rolledBackFrom.map { .int($0) } ?? .null),
            ("live_miss", liveMiss.map { .string($0) } ?? .null),
        ])
    }

    /// Sampling parameters the engine does not implement: a non-neutral value is a 400 (like `n > 1`) instead of
    /// being silently ignored. `*_context_size` (mlx-lm windows) are rejected whenever present: the penalties use
    /// the vLLM semantics, without a window.
    static func rejectUnsupported(_ b: JSONValue) throws {
        func reject(_ k: String) throws { throw APIError(400, "\(k) is not supported by this server") }
        if let n = b["n"]?.intValue, n != 1 { throw APIError(400, "n > 1 is not supported") }
        if let v = b["logit_bias"], !v.isNull, v.object?.isEmpty != true { try reject("logit_bias") }
        if let v = b["logprobs"], !v.isNull, v.bool != false { try reject("logprobs") }
        if let v = b["top_logprobs"], !v.isNull, v.intValue != 0 { try reject("top_logprobs") }
        if let v = b["xtc_probability"], !v.isNull, v.number != 0 { try reject("xtc_probability") }
        for k in ["repetition_context_size", "presence_context_size", "frequency_context_size"] {
            if let v = b[k], !v.isNull { try reject(k + " (penalties apply to the whole text, no window)") }
        }
    }

    /// Prompt of a request: normalized messages, tools (tool_choice "none" removes them) and template parameters
    /// -> rendered text and ids.
    static func prompt(body b: JSONValue, info: ChatModelInfo) throws -> (
        text: String, ids: [Int], messages: [JSONValue], tools: [JSONValue]?
    ) {
        let msgs = try normalize(b["messages"]?.array ?? [], dialect: info.dialect)
        var tools = b["tools"]?.array
        if let tc = b["tool_choice"]?.string, tc == "none" { tools = nil }
        if tools?.isEmpty == true { tools = nil }
        // Template parameters: chat_template_kwargs and a bare enable_thinking.
        var kwargs: [(String, JSONValue)] = []
        if let kw = b["chat_template_kwargs"]?.object { kwargs += kw }
        if let et = b["enable_thinking"], !kwargs.contains(where: { $0.0 == "enable_thinking" }) {
            kwargs.append(("enable_thinking", et))
        }
        // reasoning_effort from the OpenAI API (Flash-Next: the template accepts xhigh, medium and low; "high" is its
        // xhigh and "minimal" its low; gpt-oss takes it as is; Muse's template calls it reasoning_strength).
        let effortKey = info.dialect.reasoningKwarg
        if let re = b["reasoning_effort"]?.string, !kwargs.contains(where: { $0.0 == effortKey }) {
            kwargs.append((effortKey, .string(info.dialect.reasoningEffort(re))))
        }
        let p: (text: String, ids: [Int])
        do { p = try info.prompt(messages: msgs, tools: tools, addGenerationPrompt: true, kwargs: kwargs) } catch {
            throw APIError(400, "chat template failed: \(error)")
        }
        return (p.text, p.ids, msgs, tools)
    }

    /// Messages as the template expects them: system/developer merged at the start, text parts, assistant reasoning
    /// in reasoning_content and tool-call arguments as an object (the template writes each
    /// parameter; vLLM and splash do the same). harmony (gpt-oss) reads the reasoning from `thinking`, and rejects it
    /// next to `content` in a turn with tool calls: there the reasoning is kept (the template would write that
    /// `content` as the analysis anyway).
    static func normalize(_ raw: [JSONValue], dialect: ChatDialect) throws -> [JSONValue] {
        let harmony = dialect.toolFormat == .harmony
        var out: [JSONValue] = []
        func text(_ c: JSONValue?) throws -> JSONValue {
            guard let c else { return .string("") }
            if c.isNull { return .string("") }
            if c.string != nil { return c }
            guard let parts = c.array else { throw APIError(400, "content must be text or a list of parts") }
            var s = ""
            for p in parts {
                if let t = p.string { s += t; continue }
                let type = p["type"]?.string ?? "text"
                guard type == "text" || type == "input_text", let t = p["text"]?.string else {
                    throw APIError(400, "this server only supports text (part of type \(type))")
                }
                s += t
            }
            return .string(s)
        }
        for m in raw {
            guard let role = m["role"]?.string else { throw APIError(400, "each message needs role") }
            switch role {
            case "system", "developer":
                let t = try text(m["content"]).string!
                if out.count == 1, out[0]["role"]?.string == "system", let prev = out[0]["content"]?.string {
                    out[0] = .object([
                        ("role", .string("system")),
                        ("content", .string([prev, t].filter { !$0.isEmpty }.joined(separator: "\n\n"))),
                    ])
                } else if out.isEmpty {
                    out.append(.object([("role", .string("system")), ("content", .string(t))]))
                } else {
                    // A system message mid-conversation: the template rejects it; it goes as user.
                    out.append(.object([("role", .string("user")), ("content", .string(t))]))
                }
            case "user", "tool":
                var o: [(String, JSONValue)] = [("role", .string(role)), ("content", try text(m["content"]))]
                if role == "tool", let id = m["tool_call_id"] { o.append(("tool_call_id", id)) }
                out.append(.object(o))
            case "assistant":
                var o: [(String, JSONValue)] = [("role", .string("assistant")), ("content", try text(m["content"]))]
                let calls = m["tool_calls"]?.array ?? []
                if let r = m["reasoning_content"] ?? m["reasoning"], let rs = try text(r).string, !rs.isEmpty {
                    if harmony && !calls.isEmpty { o[1].1 = .string("") }
                    o.append((harmony ? "thinking" : "reasoning_content", .string(rs)))
                }
                if !calls.isEmpty {
                    var cs: [JSONValue] = []
                    for c in calls {
                        guard let f = c["function"], let name = f["name"]?.string else {
                            throw APIError(400, "tool_call without function.name")
                        }
                        var args = f["arguments"] ?? .object([])
                        if let s = args.string {
                            args = (try? JSONValue.parse(s)).flatMap { $0.object != nil ? $0 : nil } ?? .string(s)
                        }
                        var co: [(String, JSONValue)] = []
                        if let id = c["id"] { co.append(("id", id)) }
                        co += [
                            ("type", .string("function")),
                            ("function", .object([("name", .string(name)), ("arguments", args)])),
                        ]
                        cs.append(.object(co))
                    }
                    o.append(("tool_calls", .array(cs)))
                }
                out.append(.object(o))
            default:
                throw APIError(400, "unsupported role: \(role)")
            }
        }
        return out
    }

    // MARK: - Generation

    func chunk(_ job: Job, delta: [(String, JSONValue)], finish: String? = nil) -> String {
        JSONValue.object([
            ("id", .string("chatcmpl-\(job.id)")), ("object", .string("chat.completion.chunk")),
            ("created", .int(Int(Date().timeIntervalSince1970))), ("model", .string(modelId)),
            (
                "choices",
                .array([
                    .object([
                        ("index", .int(0)), ("delta", .object(delta)),
                        ("finish_reason", finish.map { .string($0) } ?? .null),
                    ])
                ])
            ),
        ]).compact()
    }

    func usage(_ job: Job, _ r: InferenceEngine.Result, completion: Int) -> JSONValue {
        .object([
            ("prompt_tokens", .int(job.prompt.count)), ("completion_tokens", .int(completion)),
            ("total_tokens", .int(job.prompt.count + completion)),
            ("prompt_tokens_details", .object([("cached_tokens", .int(r.cachedTokens))])),
        ])
    }

    static func eventDelta(_ e: OutputEvent) -> [(String, JSONValue)] {
        switch e {
        case .reasoning(let s): return [("reasoning_content", .string(s))]
        case .content(let s): return [("content", .string(s))]
        case .toolStart(let i, let id, let name):
            return [
                (
                    "tool_calls",
                    .array([
                        .object([
                            ("index", .int(i)), ("id", .string(id)), ("type", .string("function")),
                            ("function", .object([("name", .string(name)), ("arguments", .string(""))])),
                        ])
                    ])
                )
            ]
        case .toolArguments(let i, let a):
            return [
                (
                    "tool_calls",
                    .array([.object([("index", .int(i)), ("function", .object([("arguments", .string(a))]))])])
                )
            ]
        }
    }

    /// Full response (non-stream) of /v1/chat/completions: message with content, reasoning and tool calls;
    /// with `returnTokens`, also the prompt and output ids.
    func completionResponse(
        _ job: Job, parser: OutputParser, finish: String, usage u: JSONValue, outIds: [UInt32]
    ) -> JSONValue {
        completionResponse(
            job, content: parser.content, reasoning: parser.reasoningText, calls: parser.calls, finish: finish,
            usage: u,
            outIds: outIds)
    }

    func completionResponse(
        _ job: Job, content: String, reasoning: String, calls: [ToolCall], finish: String, usage u: JSONValue,
        outIds: [UInt32]
    ) -> JSONValue {
        var msg: [(String, JSONValue)] = [
            ("role", .string("assistant")),
            ("content", content.isEmpty && !calls.isEmpty ? .null : .string(content)),
        ]
        if !reasoning.isEmpty { msg.append(("reasoning_content", .string(reasoning))) }
        if !calls.isEmpty {
            msg.append(
                (
                    "tool_calls",
                    .array(
                        calls.map {
                            .object([
                                ("id", .string($0.id)), ("type", .string("function")),
                                (
                                    "function",
                                    .object([("name", .string($0.name)), ("arguments", .string($0.arguments))])
                                ),
                            ])
                        })
                ))
        }
        let resp = JSONValue.object([
            ("id", .string("chatcmpl-\(job.id)")), ("object", .string("chat.completion")),
            ("created", .int(Int(Date().timeIntervalSince1970))), ("model", .string(modelId)),
            (
                "choices",
                .array([
                    .object([
                        ("index", .int(0)), ("message", .object(msg)),
                        ("finish_reason", .string(finish)),
                    ])
                ])
            ),
            ("usage", u),
        ])
        var full = resp
        if job.returnTokens, case .object(var kv) = resp {
            kv.append(("lab_prompt_tokens", .array(job.prompt.map { .int(Int($0)) })))
            kv.append(("lab_tokens", .array(outIds.map { .int(Int($0)) })))
            full = .object(kv)
        }
        return full
    }

    /// Batching: the request goes to the engine loop, which joins it to the conversations running when a
    /// slot is free (`InferenceEngine.serve`).
    func enqueueBatched(_ job: Job, _ conn: HTTPConnection, keepAlive: StreamKeepAlive?) {
        final class Box: @unchecked Sendable { var session: Session? }
        let box = Box()
        let eos = Set(info.eosIds.map { UInt32($0) })
        requests.push(
            EngineRequest(
                prompt: job.prompt, maxTokens: job.maxTokens, sampling: job.sampling, stopTokens: eos,
                onStart: { [self] in
                    lock.lock(); waiting -= 1; lock.unlock()
                    box.session = Session(self, job, conn, keepAlive: keepAlive)
                    return box.session != nil
                },
                onTokens: { box.session?.onTokens($0) ?? false },
                onDone: { r in
                    switch r {
                    case .success(let res): box.session?.finish(res)
                    case .failure(let e): box.session?.fail(e)
                    }
                }))
        // One loop at a time on the engine queue; it ends when no request is left (checked under the lock, so a
        // request pushed meanwhile either is seen by it or starts a new one).
        lock.lock()
        let start = !loopRunning
        loopRunning = true
        lock.unlock()
        guard start else { return }
        engineQueue.async { [self] in
            while true {
                engine.serve(requests)
                lock.lock()
                if requests.isEmpty {
                    loopRunning = false
                    lock.unlock()
                    return
                }
                lock.unlock()
            }
        }
    }

    func run(_ job: Job, _ conn: HTTPConnection, keepAlive: StreamKeepAlive? = nil) {
        guard let session = Session(self, job, conn, keepAlive: keepAlive) else { return }
        do {
            let r = try engine.generate(
                prompt: job.prompt, maxTokens: job.maxTokens, sampling: job.sampling, stopTokens: session.eos,
                onTokens: session.onTokens)
            session.finish(r)
        } catch {
            session.fail(error)
        }
    }
}

extension SamplingParams {
    /// Request log: greedy or T/top-p/top-k, plus min-p and penalties when they are on.
    var logDescription: String {
        var s =
            greedy ? "greedy" : String(format: "T=%.2f top_p=%.2f top_k=%d", temperature, topP, topK)
        if minP > 0 && !greedy { s += String(format: " min_p=%.3g", minP) }
        if repetitionPenalty != 1 { s += String(format: " repetition_penalty=%.3g", repetitionPenalty) }
        if presencePenalty != 0 { s += String(format: " presence_penalty=%.3g", presencePenalty) }
        if frequencyPenalty != 0 { s += String(format: " frequency_penalty=%.3g", frequencyPenalty) }
        return s
    }
}
