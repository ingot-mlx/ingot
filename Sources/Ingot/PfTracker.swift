import Foundation
import Metal

// Helper piece of ExpertStream (SSD expert streaming): tracking of prefill jobs ahead of the GPU.

/// In-flight prefill jobs ahead of the GPU, in value order. Signals `ready` with the value of the last
/// finished job that has no earlier job pending (the GPU waits on each layer by its value).
final class PfTracker: @unchecked Sendable {
    private let cond = NSCondition()
    private var pending: [(v: UInt64, done: Bool)] = []
    private let event: MTLSharedEvent
    init(event: MTLSharedEvent) { self.event = event }

    func register(_ v: UInt64) {
        cond.lock()
        precondition(pending.last.map { $0.v < v } ?? true)
        pending.append((v, false))
        cond.unlock()
    }

    func complete(_ v: UInt64) {
        cond.lock()
        if let i = pending.firstIndex(where: { $0.v == v }) { pending[i].done = true }
        var top: UInt64?
        while let f = pending.first, f.done {
            top = f.v
            pending.removeFirst()
        }
        if let top { event.signaledValue = top }
        cond.broadcast()
        cond.unlock()
    }

    /// No jobs in flight.
    var idle: Bool {
        cond.lock()
        defer { cond.unlock() }
        return pending.isEmpty
    }

    /// Waits until at most `n` jobs remain in flight.
    func waitPending(atMost n: Int) {
        cond.lock()
        while pending.count > n { cond.wait() }
        cond.unlock()
    }
}
