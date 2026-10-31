import Foundation
import Metal

/// Residency of the weight buffers, requested while they load. Without it the first command buffer waits for the
/// system to wire every weight buffer, which by then it has often compressed or swapped out (~13 s with the
/// Qwen3.6 MoE). Each buffer is added to one `MTLResidencySet` and its residency is requested as soon as it is
/// written, on a serial queue of its own so the loading thread does not wait. At the end of the load the set is
/// attached to the command queue, and detached when the model goes away.
final class WeightResidency: @unchecked Sendable {
    let set: MTLResidencySet
    private let queue = DispatchQueue(label: "ingot.weight-residency")
    private var attached: MTLCommandQueue?

    /// `loading`: the store whose `deviceTensor` buffers join the set until it clears `residency`.
    init(device: MTLDevice, loading store: SafetensorsStore? = nil) throws {
        set = try device.makeResidencySet(descriptor: MTLResidencySetDescriptor())
        store?.residency = self
    }

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

    /// End of the load: waits for the pending requests and attaches the set to the queue.
    func finishLoad(queue commandQueue: MTLCommandQueue) {
        drain()
        attach(to: commandQueue)
    }

    /// Attaches the set to the command queue (once).
    func attach(to commandQueue: MTLCommandQueue) {
        guard attached == nil else { return }
        commandQueue.addResidencySet(set)
        attached = commandQueue
    }

    /// Detaches the set from the queue it was attached to and ends the residency of its allocations.
    func detach() {
        drain()
        attached?.removeResidencySet(set)
        attached = nil
        set.endResidency()
    }
}
