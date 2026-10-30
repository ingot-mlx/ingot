import Foundation
import MetalLab

extension ChatServer {
    /// The response side of one request: stream or JSON, detokenizing, stop strings, output parsing and the request
    /// log. Shared by the one-at-a-time path (`run`) and the engine loop of batching (`EngineRequest` callbacks).
    final class Session {
        let server: ChatServer
        let job: Job
        let conn: HTTPConnection
        let keepAlive: StreamKeepAlive?
        let eos: Set<UInt32>
        let t0 = Date()
        var detok: StreamingDetokenizer
        var stopper: StopMatcher
        var parser: OutputParser
        var emitted = 0
        var outIds: [UInt32] = []
        /// Agent guards (nil with `--agent-guards off`) and why they stopped the response.
        var agentGuard: AgentGuard?
        var trip: AgentGuard.Trip?
        /// Armed requests (a repeated call may be dropped): what has been sent, which the response keeps if it is.
        var sent = (content: "", reasoning: "", calls: [ToolCall]())
        var firstTokenAt: Date?
        let i8Before: (Int, Int)
        let poolBefore: (Int, Int)?
        let skipBefore: Int

        /// Starts the response (nil if the client already left).
        init?(_ server: ChatServer, _ job: Job, _ conn: HTTPConnection, keepAlive: StreamKeepAlive?) {
            if conn.closed {
                keepAlive?.stop()
                server.log("[\(job.id)] client left before starting")
                return nil
            }
            self.server = server
            self.job = job
            self.conn = conn
            self.keepAlive = keepAlive
            // Out of the queue: the comments go on during the prefill and stop at the first token.
            keepAlive?.setLabel("processing")
            server.log(
                "[\(job.id)] \(job.messages) messages, \(job.tools?.count ?? 0) tools, reasoning \(job.thinking ? "yes" : "no"), "
                    + "max_tokens \(job.maxTokens)\(job.stream ? ", stream" : "")"
            )
            let info = server.info
            detok = StreamingDetokenizer(info.tokenizer)
            stopper = StopMatcher(job.stops)
            let d = info.dialect
            parser = OutputParser(
                thinking: job.thinking, tools: job.tools, idPrefix: job.id, format: d.toolFormat,
                trimReasoningStart: d.reasoningOpen.map { job.promptText.hasSuffix($0) } ?? false,
                reasoningOpen: d.reasoningOpen, reasoningClose: d.reasoningClose ?? "</think>",
                closeSeparator: d.reasoningCloseSeparator)
            eos = Set(info.eosIds.map { UInt32($0) })
            agentGuard = server.agentGuards ? AgentGuard(suspect: job.repeatedCall) : nil
            let m = server.engine.model
            i8Before = (m.prefillI8Chunks, m.prefillQK8Chunks)
            poolBefore = m.kvPool.map { ($0.grows, $0.shrinks) }
            skipBefore = m.expertStream?.stats.skipped ?? 0
            if job.stream {
                conn.startStream()
                conn.event(server.chunk(job, delta: [("role", .string("assistant")), ("content", .string(""))]))
            }
        }

        func send(_ evs: [OutputEvent]) {
            guard job.stream else { return }
            for e in evs { conn.event(server.chunk(job, delta: ChatServer.eventDelta(e))) }
        }

        /// The parser's events through the repeated-call guard (as they are if the request is not armed).
        func emit(_ evs: [OutputEvent]) {
            guard agentGuard?.suspect != nil else { return send(evs) }
            let (out, t) = agentGuard!.filter(evs)
            record(out)
            // A dropped call always wins: it was never sent.
            if let t, trip == nil || !t.isTokenLoop { trip = t }
        }

        /// Sends the events of an armed request and keeps what went out.
        func record(_ evs: [OutputEvent]) {
            for e in evs {
                switch e {
                case .content(let c): sent.content += c
                case .reasoning(let r): sent.reasoning += r
                case .toolStart(_, let id, let name): sent.calls.append(ToolCall(id: id, name: name, arguments: ""))
                case .toolArguments(_, let a):
                    if !sent.calls.isEmpty { sent.calls[sent.calls.count - 1].arguments = a }
                }
            }
            send(evs)
        }

        /// Each round's tokens; false stops the generation.
        func onTokens(_ toks: [UInt32]) -> Bool {
            if trip != nil { return false }
            if firstTokenAt == nil {
                firstTokenAt = Date()
                keepAlive?.stop()
            }
            emitted += toks.count
            if job.returnTokens || job.dumpName != nil { outIds += toks }
            var text = ""
            for t in toks where !eos.contains(t) { text += detok.push(Int(t)) }
            let kept = stopper.put(text)
            emit(parser.put(kept))
            if stopper.stopped || conn.closed || trip != nil { return false }
            // The guard only decides whether to go on, after the round has been sent as it is.
            if let t = agentGuard?.push(toks.filter { !eos.contains($0) }) {
                trip = t
                return false
            }
            return true
        }

        func finish(_ r: InferenceEngine.Result) {
            keepAlive?.stop()
            var tail = stopper.put(detok.flush())
            tail += stopper.finish()
            emit(parser.put(tail))
            // A token-loop trip counts only if it is what stopped the engine (not an end token in the same round).
            let loop = r.stats.finish == .cancelled && trip?.isTokenLoop == true
            let incomplete = r.stats.finish == .length || r.stats.finish == .contextFull || loop
            emit(parser.finish(incomplete: incomplete))
            if agentGuard?.suspect != nil { record(agentGuard!.flush()) }
            // A dropped call always counts: it was never sent, whatever stopped the engine.
            var repeated: String?
            if case .repeatedCall(let name)? = trip { repeated = name }
            var finish: String
            switch r.stats.finish {
            case .stop: finish = "stop"
            case .length, .contextFull: finish = "length"
            // A loop cut short answers as the token limit does (pi asks for a truncated call again), never "cancelled".
            case .cancelled: finish = stopper.stopped || repeated != nil ? "stop" : loop ? "length" : "cancelled"
            }
            var calls = parser.calls
            if repeated != nil {
                // The turn ends with the server's notice, unless calls before the repeat went out (pi runs them).
                finish = "stop"
                calls = sent.calls
                if calls.isEmpty {
                    let notice = (sent.content.isEmpty ? "" : "\n\n") + AgentGuard.repeatedCallText
                    sent.content += notice
                    send([.content(notice)])
                }
            }
            if !calls.isEmpty && finish == "stop" { finish = "tool_calls" }
            let u = server.usage(job, r, completion: emitted)
            if job.stream {
                conn.event(server.chunk(job, delta: [], finish: finish))
                if job.includeUsage {
                    conn.event(
                        JSONValue.object([
                            ("id", .string("chatcmpl-\(job.id)")), ("object", .string("chat.completion.chunk")),
                            ("created", .int(Int(Date().timeIntervalSince1970))), ("model", .string(server.modelId)),
                            ("choices", .array([])), ("usage", u),
                        ]).compact())
                }
                conn.endStream()
            } else {
                let full =
                    repeated != nil
                    ? server.completionResponse(
                        job, content: sent.content, reasoning: sent.reasoning, calls: calls, finish: finish, usage: u,
                        outIds: outIds)
                    : server.completionResponse(job, parser: parser, finish: finish, usage: u, outIds: outIds)
                conn.respond(200, json: full.compact())
            }
            log(r, finish: finish, calls: calls.count)
            if let n = job.dumpName, let dir = server.dumpDirectory {
                let out = ChatServer.dumpOutput(
                    promptIds: job.prompt, outputIds: outIds, finish: finish, cached: r.cachedTokens,
                    cacheSource: r.cacheSource, rolledBackFrom: r.rolledBackFrom, liveMiss: r.liveMiss)
                try? out.compact().write(
                    to: dir.appendingPathComponent("\(n)-output.json"), atomically: true, encoding: .utf8)
            }
            if let name = repeated {
                server.log(
                    "[\(job.id)] guard: repeated call \(name) (3rd time, same result), dropped at \(emitted) tokens")
            } else if loop, case .tokenLoop(let period, let span)? = trip {
                server.log(
                    "[\(job.id)] guard: token loop (period \(period), \(span) tokens repeated), stopped at \(emitted) of max \(job.maxTokens)"
                )
            }
        }

        private func log(_ r: InferenceEngine.Result, finish: String, calls: Int) {
            let log = server.log, m = server.engine.model
            let st = r.stats
            let dec = st.decodeSeconds > 0 ? Double(max(0, st.tokens.count - 1)) / st.decodeSeconds : 0
            let newTok = job.prompt.count - r.cachedTokens
            log(
                String(
                    format:
                        "[%@] prompt %d tok (%d cached: %@; prefilled %d in %.2f s, %.0f tok/s), "
                        + "%d tok in %.2f s (%.1f tok/s, %d rounds, %.2f tok/round, %.2f ms/round), finish %@, %@%@, total %.2f s",
                    job.id, job.prompt.count, r.cachedTokens, r.cacheSource, newTok, r.prefillSeconds,
                    newTok > 0 ? Double(newTok) / max(r.prefillSeconds, 1e-9) : 0, emitted, st.decodeSeconds, dec,
                    st.rounds,
                    st.tokensPerRound, st.rounds > 0 ? st.decodeSeconds * 1e3 / Double(st.rounds) : 0, finish,
                    job.sampling.logDescription,
                    calls == 0 ? "" : ", \(calls) tool calls", Date().timeIntervalSince(t0)))
            if st.copyRounds > 0 || st.deferred > 0 {
                log(String(format: "[%@]   %@; index %.1f ms", job.id, st.copySummary, r.copyIndexMs))
            }
            if st.rounds > 0 && st.rowsHist.reduce(0, +) > 0 {
                log(
                    "[\(job.id)]   " + st.rowsSummary
                        + String(format: "; Metal memory %.2f GB", Double(m.ctx.device.currentAllocatedSize) / 1e9))
            }
            if !st.wallMs.isEmpty { log("[\(job.id)]   " + st.roundSummary) }
            if let p = m.kvPool, let b = poolBefore, p.grows != b.0 || p.shrinks != b.1 {
                log(
                    "[\(job.id)]   " + p.describe()
                        + String(format: "; Metal memory %.2f GB", Double(m.ctx.device.currentAllocatedSize) / 1e9))
            }
            if let es = m.expertStream, es.skipRank != nil, st.rounds > 0 {
                log(
                    String(
                        format: "[%@]   expert skipping: %.1f experts skipped per round", job.id,
                        Double(es.stats.skipped - skipBefore) / Double(st.rounds)))
            }
            let i8n = m.prefillI8Chunks - i8Before.0, qk8n = m.prefillQK8Chunks - i8Before.1
            if i8n > 0 || qk8n > 0 {
                log("[\(job.id)]   lossy prefill: \(i8n) chunks in W8A8, \(qk8n) with Q·Kᵀ in int8")
            }
        }

        func fail(_ error: Error) {
            keepAlive?.stop()
            let (code, msg) = error is EngineError ? (400, "\(error)") : (500, "\(error)")
            server.log("[\(job.id)] \(code == 400 ? "400" : "error"): \(msg)")
            if job.stream {
                conn.event(APIError(code, msg).json); conn.endStream()
            } else {
                conn.respond(code, json: APIError(code, msg).json)
            }
        }
    }
}
