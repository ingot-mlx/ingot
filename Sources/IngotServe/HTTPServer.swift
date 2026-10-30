import Foundation
import Network

/// HTTP/1.1 request, fully read.
public struct HTTPRequest {
    public var method: String
    public var path: String
    public var headers: [String: String]  // keys in lowercase
    public var body: Data
}

/// Connection of a request: full response or streaming (SSE). It closes when done (`Connection: close`).
public final class HTTPConnection: @unchecked Sendable {
    let conn: NWConnection
    private let lock = NSLock()
    private var _closed = false
    /// The client has gone (or a send failed): generation must stop.
    public var closed: Bool { lock.lock(); defer { lock.unlock() }; return _closed }

    init(_ c: NWConnection) { conn = c }

    func markClosed() { lock.lock(); _closed = true; lock.unlock() }

    static func status(_ code: Int) -> String {
        [
            200: "OK", 400: "Bad Request", 404: "Not Found", 405: "Method Not Allowed", 413: "Payload Too Large",
            500: "Internal Server Error", 503: "Service Unavailable",
        ][code] ?? "Error"
    }

    private func send(_ d: Data, then: (@Sendable () -> Void)? = nil) {
        conn.send(
            content: d,
            completion: .contentProcessed { [weak self] err in
                if err != nil { self?.markClosed() }
                then?()
            })
    }

    public func respond(_ code: Int, json: String) {
        let body = Data(json.utf8)
        let head =
            "HTTP/1.1 \(code) \(Self.status(code))\r\nContent-Type: application/json\r\nContent-Length: \(body.count)\r\nConnection: close\r\n\r\n"
        send(Data(head.utf8) + body) { [weak self] in self?.finish() }
    }

    private var streamStarted = false

    /// Sends the SSE response headers (200, `text/event-stream`); later calls do nothing, so a queued request can
    /// get its headers early and the generation can call it again.
    public func startStream() {
        lock.lock()
        let first = !streamStarted
        streamStarted = true
        lock.unlock()
        guard first else { return }
        send(
            Data(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n"
                    .utf8))
    }

    /// One SSE comment (`: text`): clients ignore it, but it keeps idle timeouts from firing.
    func comment(_ text: String) { send(Data(": \(text)\n\n".utf8)) }

    /// One SSE event (`data: …\n\n`).
    public func event(_ json: String) { send(Data("data: \(json)\n\n".utf8)) }

    public func endStream() {
        send(Data("data: [DONE]\n\n".utf8)) { [weak self] in self?.finish() }
    }

    func finish() {
        conn.send(
            content: nil, contentContext: .finalMessage, isComplete: true,
            completion: .contentProcessed { [weak self] _ in
                self?.markClosed()
                self?.conn.cancel()
            })
    }
}

/// Minimal HTTP/1.1 server over Network.framework: reads the request (headers + Content-Length) and passes it to
/// `handler` on its own queue.
public final class HTTPServer: @unchecked Sendable {
    let listener: NWListener
    let queue = DispatchQueue(label: "http", attributes: .concurrent)
    let handler: @Sendable (HTTPRequest, HTTPConnection) -> Void
    public static let maxBody = 64 << 20

    public init(host: String, port: UInt16, handler: @escaping @Sendable (HTTPRequest, HTTPConnection) -> Void) throws {
        let params = NWParameters.tcp
        params.allowLocalEndpointReuse = true
        params.requiredLocalEndpoint = NWEndpoint.hostPort(
            host: NWEndpoint.Host(host), port: NWEndpoint.Port(rawValue: port)!)
        listener = try NWListener(using: params)
        self.handler = handler
    }

    public func start(ready: @escaping @Sendable (Error?) -> Void) {
        listener.stateUpdateHandler = { st in
            switch st {
            case .ready: ready(nil)
            case .failed(let e): ready(e)
            default: break
            }
        }
        listener.newConnectionHandler = { [weak self] c in self?.accept(c) }
        listener.start(queue: queue)
    }

    func accept(_ c: NWConnection) {
        let h = HTTPConnection(c)
        c.stateUpdateHandler = { st in
            switch st {
            case .failed, .cancelled: h.markClosed()
            default: break
            }
        }
        c.start(queue: queue)
        read(c, h, Data())
    }

    func read(_ c: NWConnection, _ h: HTTPConnection, _ buf: Data) {
        c.receive(minimumIncompleteLength: 1, maximumLength: 1 << 20) { [weak self] data, _, done, err in
            guard let self else { return }
            var b = buf
            if let data { b += data }
            if let r = Self.parse(b) {
                switch r {
                case .request(let req): self.handler(req, h)
                case .error(let code, let msg): h.respond(code, json: APIError(code, msg).json)
                }
                // From here on we only watch whether the client closes (to cancel generation).
                self.watch(c, h)
                return
            }
            if done || err != nil { h.markClosed(); c.cancel(); return }
            self.read(c, h, b)
        }
    }

    func watch(_ c: NWConnection, _ h: HTTPConnection) {
        c.receive(minimumIncompleteLength: 1, maximumLength: 65536) { [weak self] _, _, done, err in
            if done || err != nil { h.markClosed(); return }
            self?.watch(c, h)
        }
    }

    enum Parsed { case request(HTTPRequest), error(Int, String) }

    static func parse(_ b: Data) -> Parsed? {
        guard let end = b.range(of: Data("\r\n\r\n".utf8)) else {
            return b.count > 1 << 16 ? .error(400, "headers too long") : nil
        }
        guard let head = String(data: b[..<end.lowerBound], encoding: .utf8) else {
            return .error(400, "invalid headers")
        }
        var lines = head.components(separatedBy: "\r\n")
        let first = lines.removeFirst().split(separator: " ")
        guard first.count >= 2 else { return .error(400, "invalid request line") }
        var headers: [String: String] = [:]
        for l in lines {
            guard let i = l.firstIndex(of: ":") else { continue }
            headers[l[..<i].lowercased()] = l[l.index(after: i)...].trimmingCharacters(in: .whitespaces)
        }
        if headers["transfer-encoding"]?.lowercased().contains("chunked") == true {
            return .error(400, "chunked body not supported: use Content-Length")
        }
        let len = Int(headers["content-length"] ?? "0") ?? 0
        if len > maxBody { return .error(413, "body too large") }
        let bodyStart = end.upperBound
        guard b.count - bodyStart >= len else { return nil }
        let path = String(first[1]).split(separator: "?").first.map(String.init) ?? "/"
        return .request(
            HTTPRequest(method: String(first[0]), path: path, headers: headers, body: b[bodyStart..<(bodyStart + len)]))
    }
}

/// API error in the OpenAI format.
public struct APIError: Error {
    public let code: Int, message: String, type: String
    public init(_ code: Int, _ message: String, type: String? = nil) {
        self.code = code; self.message = message
        self.type = type ?? (code == 400 ? "invalid_request_error" : code == 404 ? "not_found_error" : "server_error")
    }
    public var json: String {
        JSONValue.object([
            ("error", .object([("message", .string(message)), ("type", .string(type)), ("code", .int(code))]))
        ]).compact()
    }
}
