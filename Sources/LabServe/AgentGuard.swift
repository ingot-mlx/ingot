/// Agent guards of `lab serve`: stop a response that has fallen into a token loop, and a turn that repeats for
/// the third time the same tool call with the same result.
///
/// The guard only decides whether to go on: it never changes tokens, text or the order of events. If it does not trip,
/// the response is the same as without it; if it trips, the ids are a prefix of the ones without it (`Session.onTokens`
/// returns false after that round). A token loop answers `finish_reason: "length"`; a repeated call is dropped and the
/// turn ends with `"stop"` and `repeatedCallText` in `content`.
struct AgentGuard {
    /// Why the guard stopped a response.
    enum Trip: Equatable {
        /// The output ends in a block of `period` tokens repeated over its last `span` tokens.
        case tokenLoop(period: Int, span: Int)
        /// The response repeated, for the third time in a row, the call of the two previous turns.
        case repeatedCall(name: String)

        var isTokenLoop: Bool {
            if case .tokenLoop = self { return true }
            return false
        }
    }

    /// A tool call: its name and its arguments as `canonical` JSON.
    struct Call: Equatable {
        var name: String
        var arguments: String
    }

    /// What the client gets in `content` when a repeated call is dropped (the turn ends there).
    static let repeatedCallText =
        "[lab serve] Stopped: the same tool call was repeated three times with the same result."

    /// Thresholds of the loop detector (Locals `_looping`, server.py:2074, its values "without tools"). Calibrated
    /// on 3,703 pi responses retokenized with their family's tokenizer: 0 trips on the 3,676 that pi accepted
    /// (8 / 64 / 256 also gives 0, 6 / 48 / 128 already gives one), and it stops the Muse loop at token 2,994 and four
    /// GLM replays between 1,043 and 2,450.
    static let reps = 12
    static let minSpan = 96
    static let shortSpan = 256
    static let maxPeriod = 256

    private(set) var ids: [UInt32] = []
    /// The call a third repeat would trip on (nil: the request is not armed, `filter` passes everything as it is).
    let suspect: Call?
    /// A `toolStart` with the suspect's name, held until its arguments say whether it is the repeat.
    private var held: OutputEvent?
    private var dropped = false

    init(suspect: Call? = nil) { self.suspect = suspect }

    /// Adds one round's output tokens and checks the end of the output for a loop.
    mutating func push(_ toks: [UInt32]) -> Trip? {
        ids += toks
        guard !toks.isEmpty, let hit = ids.withUnsafeBufferPointer({ Self.looping($0) }) else { return nil }
        return .tokenLoop(period: hit.period, span: hit.span)
    }

    /// Whether `ids` ends in one block of up to `maxPeriod` tokens repeated `reps` times over at least `minSpan`
    /// tokens (`shortSpan` if the block is shorter than 8 tokens: data repeats a few tokens on purpose, "0, 0, 0").
    /// The smallest period decides (a short block is also periodic at its multiples), and a period must bring the
    /// last token back. Same decisions as Locals `_looping`; the comparisons run from the end, where a tail that is
    /// not periodic differs first.
    static func looping(
        _ ids: UnsafeBufferPointer<UInt32>, reps: Int = reps, minSpan: Int = minSpan, shortSpan: Int = shortSpan,
        maxPeriod: Int = maxPeriod
    ) -> (period: Int, span: Int)? {
        let n = ids.count
        guard n >= minSpan, let ids = ids.baseAddress else { return nil }

        let last = ids[n - 1]
        if n >= 32 {
            for p in 1..<8 where ids[n - 1 - p] == last && periodic(ids, n, p, 32) {
                let span = max(reps * p, minSpan, shortSpan)
                return periodic(ids, n, p, span) ? (p, span) : nil
            }
        }
        let hi = min(maxPeriod, n / reps)
        guard hi >= 8 else { return nil }
        // A while loop: a range loop costs ~20 µs per call in a debug build (tests).
        var p = 8
        while p <= hi {
            if ids[n - 1 - p] == last {
                let span = max(reps * p, minSpan)
                if periodic(ids, n, p, span) { return (p, span) }
            }
            p += 1
        }
        return nil
    }

    /// The last `span` tokens of `ids` have period `p` (compared from the end).
    @inline(__always)
    private static func periodic(_ ids: UnsafePointer<UInt32>, _ n: Int, _ p: Int, _ span: Int) -> Bool {
        guard span <= n else { return false }
        var i = n - 1
        let stop = n - span + p
        while i >= stop {
            if ids[i] != ids[i - p] { return false }
            i -= 1
        }
        return true
    }

    // MARK: - Repeated call

    /// The call a third identical one would repeat, read from the history the client sends: the last two assistant
    /// turns have exactly one call each, with the same name and arguments, and each is followed by one tool result,
    /// both with the same text, which ends the history. In the pi corpus (362 sessions, 3,885 responses) a
    /// response repeats the suspect only in the four long streaks (52 `write`, 138 and 56 `bash`, 3 `write`).
    static func suspect(messages: [JSONValue]) -> Call? {
        var i = messages.count - 1
        var turns: [(call: Call, result: String)] = []
        for _ in 0..<2 {
            guard i >= 1, messages[i]["role"]?.string == "tool" else { return nil }
            let result = messages[i]["content"]?.string ?? ""
            i -= 1
            guard messages[i]["role"]?.string == "assistant", let calls = messages[i]["tool_calls"]?.array,
                calls.count == 1, let f = calls[0]["function"], let name = f["name"]?.string
            else { return nil }
            turns.append((Call(name: name, arguments: canonical(arguments: f["arguments"] ?? .object([]))), result))
            i -= 1
        }
        guard turns[0].call == turns[1].call, turns[0].result == turns[1].result else { return nil }
        return turns[0].call
    }

    /// Passes the parser's events on, except that a call with the suspect's name waits for its arguments: if they are
    /// the suspect's, the call is dropped, nothing more is sent and the guard trips. In qwenXML no event comes between
    /// a `toolStart` and its arguments (OutputParser `beginCall`/`finishCall`), so the events are the same, only the
    /// header arrives later; in the other formats both come in the same round.
    mutating func filter(_ events: [OutputEvent]) -> (send: [OutputEvent], trip: Trip?) {
        guard let s = suspect else { return (events, nil) }
        if dropped { return ([], nil) }
        var out: [OutputEvent] = []
        out.reserveCapacity(events.count)
        for e in events {
            switch e {
            case .toolStart(_, _, let name) where name == s.name:
                if let h = held { out.append(h) }
                held = e
            case .toolArguments(let i, let a):
                if let h = held, case .toolStart(let hi, _, _) = h, hi == i {
                    held = nil
                    if Self.canonical(arguments: .string(a)) == s.arguments {
                        dropped = true
                        return (out, .repeatedCall(name: s.name))
                    }
                    out.append(h)
                }
                out.append(e)
            default:
                if let h = held { out.append(h); held = nil }
                out.append(e)
            }
        }
        return (out, nil)
    }

    /// End of the response: a header still held (its arguments never came) goes out as it is.
    mutating func flush() -> [OutputEvent] {
        guard let h = held, !dropped else { return [] }
        held = nil
        return [h]
    }

    /// Arguments as `ChatServer.normalize` keeps them (a JSON object, or the text if it is not one) in canonical form.
    static func canonical(arguments a: JSONValue) -> String {
        if let s = a.string, let v = try? JSONValue.parse(s), v.object != nil { return canonical(v) }
        return canonical(a)
    }

    /// Compact JSON with the keys of every object sorted, to compare arguments written in any key order.
    static func canonical(_ v: JSONValue) -> String { sorted(v).compact() }

    private static func sorted(_ v: JSONValue) -> JSONValue {
        switch v {
        case .object(let kv): return .object(kv.map { ($0.0, sorted($0.1)) }.sorted { $0.0 < $1.0 })
        case .array(let a): return .array(a.map(sorted))
        default: return v
        }
    }
}
