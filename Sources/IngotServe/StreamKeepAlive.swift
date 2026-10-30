import Foundation

/// SSE keep-alive for a streamed request that has to wait: an SSE comment every `interval` seconds until `stop()`.
/// It covers the wait in the engine queue (`: queued`) and the prefill (`: processing`), when no token goes out yet.
/// Clients such as pi drop a request after some time without data (headers or body); llama.cpp pings with `:` every
/// 30 s and mlx-lm sends `: keepalive done/total` per prefill chunk.
final class StreamKeepAlive: @unchecked Sendable {
    /// Seconds between comments: well below the 300 s of pi and the 30-60 s of common proxies.
    static let interval = 10.0

    private let conn: HTTPConnection
    private let lock = NSLock()
    private var stopped = false
    private var label = "queued"
    private let timer: DispatchSourceTimer

    init(_ conn: HTTPConnection, interval: Double = StreamKeepAlive.interval) {
        self.conn = conn
        timer = DispatchSource.makeTimerSource(queue: DispatchQueue.global(qos: .utility))
        timer.schedule(deadline: .now() + interval, repeating: interval, leeway: .milliseconds(200))
        timer.setEventHandler { [weak self] in self?.tick() }
        timer.resume()
    }

    deinit { stop() }

    /// The comment the following ticks send (e.g. `processing` once the request leaves the queue).
    func setLabel(_ s: String) { lock.lock(); label = s; lock.unlock() }

    private func tick() {
        lock.lock()
        defer { lock.unlock() }
        // Under the lock: once stop() returns no comment can follow the real events.
        if stopped || conn.closed { return }
        conn.comment(label)
    }

    /// Stops the comments; after it returns no more are sent. Idempotent.
    func stop() {
        lock.lock()
        let first = !stopped
        stopped = true
        lock.unlock()
        if first { timer.cancel() }
    }
}
