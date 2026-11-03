import Foundation
import Metal

/// Residency of the weight buffers and of the conversation state, kept while the model is in use.
///
/// Weights: each buffer is added to one `MTLResidencySet` and its residency is requested as soon as it is written, on
/// a serial queue of its own so the loading thread does not wait. At the end of the load the set is attached to the
/// command queue, and detached when the model goes away. Without it the first command buffer waits for the system to
/// wire every weight buffer, which by then it has often compressed or swapped out (~13 s with the Qwen3.6 MoE).
///
/// State: the attention KV, the recurrent state, the snapshots and the prefix cache entries go in a second set
/// (`stateSet`), kept in step with the engine by `setStateBuffers` and never attached to the queue: its members change
/// between requests (a commit on a set the queue holds would touch every command buffer after it), and every command
/// buffer already makes resident the state it binds. It is there only so the heartbeat keeps it wired while idle.
///
/// Heartbeat: a standing request does not last. Once the GPU goes idle the system drops the residency within seconds
/// and, after minutes, compresses the weights: the first command buffer of the next request then waits for the whole
/// set to be wired again (TTFT of a 512-token turn on the Qwen3.6 MoE: 363 ms right after a request, 502 ms after 3 s,
/// 1.6 s after 5 min idle). So, while there was a request (or the load) in the last `idleRelease` seconds, both
/// sets' residency is requested again every `heartbeat` seconds; after that the residency is ended, which gives the
/// memory back to the system, and the next request requests it again before it starts (paying the rewiring once).
final class WeightResidency: @unchecked Sendable {
    let set: MTLResidencySet
    /// Conversation state (KV, recurrent state, snapshots, prefix cache entries); not attached to the queue.
    let stateSet: MTLResidencySet
    /// Seconds between residency requests while in use, and seconds without a request before the residency ends.
    let heartbeat: Double, idleRelease: Double
    /// Residency events (released, requested again); `InferenceEngine.log`.
    var log: (@Sendable (String) -> Void)? {
        get { lock.withLock { logSink } }
        set { lock.withLock { logSink = newValue } }
    }

    private let queue = DispatchQueue(label: "ingot.weight-residency")
    private var attached: MTLCommandQueue?
    /// Guards the heartbeat state below.
    private let lock = NSLock()
    private var timer: DispatchSourceTimer?
    private var lastUse: UInt64 = 0
    private var logSink: (@Sendable (String) -> Void)?
    private var beatCount = 0
    /// Guards `stateSet` (its members change on the engine's thread, the heartbeat requests it on `queue`).
    private let stateLock = NSLock()
    private var stateMembers: [ObjectIdentifier: MTLBuffer] = [:]

    /// `loading`: the store whose `deviceTensor` buffers join the set until it clears `residency`.
    init(
        device: MTLDevice, loading store: SafetensorsStore? = nil, heartbeat: Double = 0.5, idleRelease: Double = 600
    ) throws {
        set = try device.makeResidencySet(descriptor: MTLResidencySetDescriptor())
        stateSet = try device.makeResidencySet(descriptor: MTLResidencySetDescriptor())
        self.heartbeat = heartbeat
        self.idleRelease = idleRelease
        store?.residency = self
    }

    /// Whether the heartbeat is running (the residency is held).
    var isHeld: Bool { lock.withLock { timer != nil } }
    /// Heartbeats so far (tests).
    var beats: Int { lock.withLock { beatCount } }

    /// Adds a buffer that is already written and asks for its residency (asynchronous).
    func add(_ buffer: MTLBuffer) {
        nonisolated(unsafe) let buffer = buffer
        queue.async { [self] in
            set.addAllocation(buffer)
            set.commit()
            set.requestResidency()
        }
    }

    /// Waits for every pending `add`.
    func drain() { queue.sync {} }

    /// End of the load: waits for the pending requests, attaches the set to the queue and starts the heartbeat (the
    /// load counts as a use: the first request usually follows).
    func finishLoad(queue commandQueue: MTLCommandQueue) {
        drain()
        attach(to: commandQueue)
        _ = use()
    }

    /// Attaches the set to the command queue (once).
    func attach(to commandQueue: MTLCommandQueue) {
        guard attached == nil else { return }
        commandQueue.addResidencySet(set)
        attached = commandQueue
    }

    /// Detaches the set from the queue it was attached to, stops the heartbeat and ends the residency of both sets.
    func detach() {
        lock.withLock {
            timer?.cancel()
            timer = nil
        }
        drain()
        attached?.removeResidencySet(set)
        attached = nil
        set.endResidency()
        stateLock.withLock { stateSet.endResidency() }
    }

    /// Start of a request: keeps the heartbeat going for `idleRelease` seconds more. If the residency had been
    /// released, it is requested again here, before the request's first command buffer, and the milliseconds that
    /// took are returned (nil otherwise). Once per request: a lock and a clock read while held.
    func use() -> Double? {
        let wasHeld: Bool = lock.withLock {
            lastUse = DispatchTime.now().uptimeNanoseconds
            if timer != nil { return true }
            let t = DispatchSource.makeTimerSource(queue: queue)
            let ns = Int(heartbeat * 1e9)
            t.schedule(deadline: .now() + .nanoseconds(ns), repeating: .nanoseconds(ns), leeway: .milliseconds(50))
            t.setEventHandler { [weak self] in self?.beat() }
            t.resume()
            timer = t
            return false
        }
        if wasHeld { return nil }
        let t0 = DispatchTime.now().uptimeNanoseconds
        requestBoth()
        return Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6
    }

    private func requestBoth() {
        set.requestResidency()
        stateLock.withLock { stateSet.requestResidency() }
    }

    /// One heartbeat (on `queue`): requests the residency again, or ends it after `idleRelease` seconds without use.
    private func beat() {
        let released: Bool = lock.withLock {
            guard timer != nil else { return false }
            let idle = Double(DispatchTime.now().uptimeNanoseconds &- lastUse) / 1e9
            guard idle >= idleRelease else { return false }
            timer?.cancel()
            timer = nil
            // Inside the lock: a request arriving now waits, and requests the residency again after this.
            set.endResidency()
            stateLock.withLock { stateSet.endResidency() }
            return true
        }
        if released {
            log?(String(format: "residency: released after %.0f s without requests", idleRelease))
            return
        }
        requestBoth()
        lock.withLock { beatCount += 1 }
    }

    /// Makes `buffers` the members of `stateSet` (by identity; one commit, only when something changed). The set holds
    /// its members alive, so a buffer the engine dropped is freed once it leaves here: called on the engine's thread
    /// after the state changes (once or twice per request, never per token).
    func setStateBuffers(_ buffers: [MTLBuffer]) {
        var next: [ObjectIdentifier: MTLBuffer] = [:]
        next.reserveCapacity(buffers.count)
        for b in buffers { next[ObjectIdentifier(b)] = b }
        stateLock.withLock {
            var changed = false
            for (k, b) in stateMembers where next[k] == nil {
                stateSet.removeAllocation(b)
                changed = true
            }
            for (k, b) in next where stateMembers[k] == nil {
                stateSet.addAllocation(b)
                changed = true
            }
            guard changed else { return }
            stateMembers = next
            stateSet.commit()
        }
    }
}

extension StateSnapshot {
    /// Every buffer of the snapshot.
    var buffers: [MTLBuffer] { gdn.flatMap { [$0.state, $0.hist] } + draft.flatMap { [$0.kc, $0.vt] } + conv }
}

extension InferenceEngine {
    /// Puts the live attention KV and recurrent state, the snapshots (also those of parked slots) and the prefix cache
    /// entries in the state residency set (`WeightResidency.stateSet`), and only them.
    func syncStateResidency() {
        var b: [MTLBuffer] = []
        for a in attnLayers { b += [a.kc, a.vt, a.ks, a.vs] }
        for l in model.layers {
            if case .gdn(let x) = l.mixer { b += x.state + x.hist }
        }
        for s in snaps + spare + parkedCache.flatMap(\.snaps) { b += s.buffers }
        for e in entries { b += e.snap.buffers + [e.kv] }
        model.weightResidency.setStateBuffers(b)
    }
}
