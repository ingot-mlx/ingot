import Foundation
import Ingot
import IngotCLI
import IngotServe
import Metal

/// ingot run --model org/repo|dir [--prompt "text" | standard input] [--system "text"] [--max-tokens 1024]
///         [--temp T --top-p P --top-k K --seed S] [--no-think] [engine options]
/// A single prompt with the model's chat template: writes the answer as it is produced (the reasoning, if any,
/// to stderr) and the figures at the end. No prefix cache or snapshots; context = prompt + --max-tokens.
@MainActor func cmdRun() throws {
    let model = try args.required("model", modelHelp)
    var text = args.opts["prompt"] ?? args.positional.joined(separator: " ")
    if text.isEmpty {
        text = String(decoding: FileHandle.standardInput.readDataToEndOfFile(), as: UTF8.self)
    }
    guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
        throw UsageError("missing prompt: --prompt \"text\", ingot run text… or via standard input")
    }
    // What is printed on load (memory plan, lossy options) goes to stderr: stdout carries only the answer.
    fflush(stdout)
    let stdoutFD = dup(STDOUT_FILENO)
    dup2(STDERR_FILENO, STDOUT_FILENO)
    let (dir, cfg, ctx) = try loadModelFiles(model)
    let info = try ChatModelInfo(directory: dir)
    var messages: [JSONValue] = []
    if let sys = args.opts["system"] {
        messages.append(.object([("role", .string("system")), ("content", .string(sys))]))
    }
    messages.append(.object([("role", .string("user")), ("content", .string(text))]))
    let noThink = args.flags.contains("no-think")
    if noThink && !info.template.source.contains("enable_thinking") {
        // gpt-oss always reasons: its template takes reasoning_effort (low, medium, high), not enable_thinking.
        print("warning: --no-think does not apply: the chat template has no enable_thinking")
    }
    let kwargs: [(String, JSONValue)] = noThink ? [("enable_thinking", .bool(false))] : []
    let rendered = try info.prompt(messages: messages, kwargs: kwargs)
    let ids = rendered.ids.map { UInt32($0) }
    let maxTokens = args.int("max-tokens", 1024)
    var mem = memOptions()
    mem.maxContext = mem.maxContext ?? (ids.count + maxTokens + 64)
    mem.prefixCacheBytes = 0
    mem.snapshots = 0
    let plan = try memoryPlan(dir: dir, cfg: cfg, device: ctx.device, options: mem, loading: true)
    let m = try HybridModel(
        ctx: ctx, store: try SafetensorsStore(directory: dir), config: cfg, maxCtx: plan.context,
        chunkRows: plan.chunkRows,
        options: try {
            var o = try engineOptions(cfg, serve: true); applyPlan(&o, plan); return o
        }())
    describeLossy(m)
    let dr = plan.drafter ? try openModelDrafter(ctx, dir, cfg, m) : nil
    let eng = InferenceEngine(model: m, drafter: dr, maxSnapshots: 0, prefixCacheBytes: 0)
    eng.generationMarker = info.dialect.generationMarker.map { info.tokenizer.encode($0).map { UInt32($0) } } ?? []
    fflush(stdout)
    dup2(stdoutFD, STDOUT_FILENO)
    close(stdoutFD)
    let temp = Float(args.opts["temp"].flatMap(Double.init) ?? Double(info.defaultTemperature ?? 0))
    let sampling = SamplingParams(
        temperature: temp, topP: Float(args.opts["top-p"].flatMap(Double.init) ?? Double(info.defaultTopP ?? 1)),
        topK: args.int("top-k", info.defaultTopK ?? 0), seed: args.opts["seed"].flatMap(UInt64.init),
        // min-p and penalties: only the model's generation_config.json (no options of their own).
        minP: info.defaultMinP ?? 0, repetitionPenalty: info.defaultRepetitionPenalty ?? 1,
        presencePenalty: info.defaultPresencePenalty ?? 0, frequencyPenalty: info.defaultFrequencyPenalty ?? 0)
    // Output as it is produced, through ingot serve's OutputParser: the reasoning (up to `</think>`, or the reasoning
    // messages of the header formats: Muse's `to=self`, gpt-oss's `analysis`) goes to stderr so that stdout carries
    // only the answer. Without tools, calls stay in the answer.
    let d = info.dialect
    var detok = StreamingDetokenizer(info.tokenizer)
    var parser = OutputParser(
        thinking: d.promptOpensReasoning(rendered.text), tools: nil, idPrefix: "run", format: d.toolFormat,
        trimReasoningStart: d.reasoningOpen.map { rendered.text.hasSuffix($0) } ?? false,
        reasoningOpen: d.reasoningOpen, reasoningClose: d.reasoningClose ?? "</think>",
        closeSeparator: d.reasoningCloseSeparator)
    let err = FileHandle.standardError, so = FileHandle.standardOutput
    var lastWasReasoning = false
    func write(_ evs: [OutputEvent]) {
        for e in evs {
            switch e {
            case .reasoning(let t): err.write(Data(t.utf8)); lastWasReasoning = true
            case .content(let t):
                // A newline on stderr between the reasoning and the answer (the parser drops the model's own).
                if lastWasReasoning { err.write(Data("\n".utf8)); lastWasReasoning = false }
                so.write(Data(t.utf8))
            case .toolStart, .toolArguments: break
            }
        }
    }
    let eos = Set(info.eosIds.map { UInt32($0) })
    let t0 = Date()
    let r = try eng.generate(
        prompt: ids, maxTokens: maxTokens, sampling: sampling, stopTokens: eos
    ) { toks in
        var text = ""
        for t in toks where !eos.contains(t) { text += detok.push(Int(t)) }
        write(parser.put(text))
        return true
    }
    write(parser.put(detok.flush()))
    write(parser.finish(incomplete: r.stats.finish == .length || r.stats.finish == .contextFull))
    let st = r.stats
    err.write(
        Data(
            String(
                format: "\n[%d prompt tokens in %.2f s; %d decoded in %.2f s, %.1f tok/s; total %.1f s]\n",
                ids.count, r.prefillSeconds, st.tokens.count, st.decodeSeconds,
                Double(max(0, st.tokens.count - 1)) / max(st.decodeSeconds, 1e-9), Date().timeIntervalSince(t0)
            ).utf8))
}
