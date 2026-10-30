import Foundation
import Network
import Testing

@testable import IngotServe

/// Early stream headers and SSE keep-alive comments: the headers go out before the first event, the comments
/// stop at `stop()` and a second `startStream()` does not repeat the headers.
struct StreamKeepAliveTests {
    @Test func headersFirstThenCommentsThenEvents() async throws {
        let server = try HTTPServer(host: "127.0.0.1", port: 0) { _, conn in
            conn.startStream()
            let ka = StreamKeepAlive(conn, interval: 0.1)
            DispatchQueue.global().asyncAfter(deadline: .now() + 0.45) {
                ka.setLabel("processing")
                DispatchQueue.global().asyncAfter(deadline: .now() + 0.25) {
                    ka.stop()
                    conn.startStream()
                    conn.event("{}")
                    conn.endStream()
                }
            }
        }
        let ready = AsyncStream<Error?> { c in
            server.start {
                c.yield($0); c.finish()
            }
        }
        for await e in ready { if let e { throw e } }
        let port = try #require(server.listener.port?.rawValue)
        defer { server.listener.cancel() }

        let t0 = Date()
        let url = try #require(URL(string: "http://127.0.0.1:\(port)/x"))
        let (bytes, resp) = try await URLSession.shared.bytes(from: url)
        let headersAfter = Date().timeIntervalSince(t0)
        let http = try #require(resp as? HTTPURLResponse)
        #expect(http.statusCode == 200)
        #expect(http.value(forHTTPHeaderField: "Content-Type") == "text/event-stream")
        #expect(headersAfter < 0.4)  // before the first event (0.7 s)
        var body = ""
        for try await line in bytes.lines { body += line + "\n" }
        let lines = body.split(separator: "\n").map(String.init)
        let queued = lines.filter { $0 == ": queued" }.count
        let processing = lines.filter { $0 == ": processing" }.count
        #expect(queued >= 2 && processing >= 1)
        #expect(lines.suffix(2) == ["data: {}", "data: [DONE]"])
        #expect(!body.contains("HTTP/1.1"))
    }
}
