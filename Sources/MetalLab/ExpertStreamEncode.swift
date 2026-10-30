import Foundation
import Metal

// SSD expert streaming: encoding side of ExpertStream (ExpertStream.swift): encoder cuts at every MoE layer and
// jobs queued for the CPU thread.

extension ExpertStream {

    /// Prefill, before the MoE of layer `layer` in the first chunk of the group. With `bulk`, the CPU fetches now (ahead
    /// of the GPU) all the layer's experts that are not in the cache and the GPU only waits for it to finish; otherwise
    /// the GPU stops after the router (which must already be encoded in `e`) and the CPU fetches those `pmoe` asks for.
    public func prefillBegin(
        _ cb: MTLCommandBuffer, _ e: MTLComputeCommandEncoder, layer: Int, bulk: Bool, pmoe: PrefillMoE, rows: Int
    ) -> MTLComputeCommandEncoder {
        pfSeq += 1
        if !bulk {
            e.endEncoding()
            nextValue += 2
            let v = nextValue - 1
            cb.encodeSignalEvent(event, value: v)
            cb.encodeWaitForEvent(event, value: v + 1)
            enqueue(Job(value: v, layer: layer, kind: .prefill(pmoe: pmoe, rows: rows), pf: pfSeq))
            return cb.makeComputeCommandEncoder()!
        }
        e.endEncoding()
        enqueue(Job(value: pfSeq, layer: layer, kind: .bulk, pf: pfSeq))
        cb.encodeWaitForEvent(readyEvent, value: pfSeq)
        return cb.makeComputeCommandEncoder()!
    }

    /// Flash-Next: like `prefillBegin` with `bulk` (the CPU fetches ahead everything the layer is missing).
    public func prefillBeginBulk(_ cb: MTLCommandBuffer, _ e: MTLComputeCommandEncoder, layer: Int)
        -> MTLComputeCommandEncoder
    {
        pfSeq += 1
        e.endEncoding()
        enqueue(Job(value: pfSeq, layer: layer, kind: .bulk, pf: pfSeq))
        cb.encodeWaitForEvent(readyEvent, value: pfSeq)
        return cb.makeComputeCommandEncoder()!
    }

    /// Flash-Next: like `prefillBegin` without `bulk`: the GPU stops after the routing (already encoded in `e`) and the
    /// CPU fetches the top-k of the first `rows` rows of `inds`.
    public func prefillBeginInds(
        _ cb: MTLCommandBuffer, _ e: MTLComputeCommandEncoder, layer: Int, inds: MTLBuffer, rows: Int
    ) -> MTLComputeCommandEncoder {
        pfSeq += 1
        e.endEncoding()
        nextValue += 2
        let v = nextValue - 1
        cb.encodeSignalEvent(event, value: v)
        cb.encodeWaitForEvent(event, value: v + 1)
        enqueue(Job(value: v, layer: layer, kind: .prefillInds(inds: inds, rows: rows), pf: pfSeq))
        return cb.makeComputeCommandEncoder()!
    }

    /// CPU work ahead of the GPU (the PLE rows of a prefill group): queued now and the GPU waits for its value with
    /// `waitReady` where it needs it.
    public func enqueueAhead(_ work: @escaping () -> Void) -> UInt64 {
        pfSeq += 1
        enqueue(Job(value: pfSeq, layer: 0, kind: .custom(work: work, ahead: true), pf: pfSeq))
        return pfSeq
    }

    public func waitReady(_ cb: MTLCommandBuffer, _ e: MTLComputeCommandEncoder, value v: UInt64)
        -> MTLComputeCommandEncoder
    {
        e.endEncoding()
        cb.encodeWaitForEvent(readyEvent, value: v)
        return cb.makeComputeCommandEncoder()!
    }

    /// Prefill, after the MoE of the layer in the last chunk of the group: its ring slots become free.
    public func prefillEnd(_ cb: MTLCommandBuffer, _ e: MTLComputeCommandEncoder) -> MTLComputeCommandEncoder {
        e.endEncoding()
        cb.encodeSignalEvent(doneEvent, value: pfSeq)
        return cb.makeComputeCommandEncoder()!
    }

    func enqueue(_ j: Job) {
        jobLock.lock()
        jobs.append(j)
        jobLock.unlock()
        jobSem.signal()
    }

    /// Like `breakpoint`, but the GPU only signals (it carries on with the residents); `waitAnswer` sets the wait.
    public func signalOnly(_ cb: MTLCommandBuffer, _ e: MTLComputeCommandEncoder, layer: Int, kind: JobKind)
        -> (MTLComputeCommandEncoder, UInt64)
    {
        e.endEncoding()
        nextValue += 2
        let v = nextValue - 1
        cb.encodeSignalEvent(event, value: v)
        enqueue(Job(value: v, layer: layer, kind: kind))
        if traceOn, let t = roundTrace {
            let l: String
            if case .custom = kind { l = "after.ple.signal" } else { l = "moe.res" }
            if let ne = t.make(cb, l, layer: layer, edge: .signal, value: v) { return (ne, v) }
        }
        return (cb.makeComputeCommandEncoder()!, v)
    }

    /// `label` and `layer` only name the new encoder in the `--round-trace` timeline.
    public func waitAnswer(
        _ cb: MTLCommandBuffer, _ e: MTLComputeCommandEncoder, value v: UInt64, label: String = "moe.miss",
        layer: Int = -1
    ) -> MTLComputeCommandEncoder {
        e.endEncoding()
        cb.encodeWaitForEvent(event, value: v + 1)
        if traceOn, let t = roundTrace, let ne = t.make(cb, label, layer: layer, edge: .wait, value: v) { return ne }
        return cb.makeComputeCommandEncoder()!
    }
}
