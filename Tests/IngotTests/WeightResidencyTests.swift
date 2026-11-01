import Foundation
import Metal
import Testing

@testable import Ingot

/// The buffers `deviceTensor` writes while a residency set is attached to the store end up in that set; attaching
/// to and detaching from a command queue works. Needs a Metal device (no kernels).
struct WeightResidencyTests {
    @Test func deviceTensorBuffersJoinTheSet() throws {
        let fmt = FormatTests()
        let dir = try fmt.tmpDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        try fmt.writeSafetensors(
            dir,
            tensors: [
                ("a", "U8", [5], Data([1, 2, 3, 4, 5])),
                ("b", "BF16", [3], Data(count: 6)),
                ("c", "F32", [7], Data(count: 28)),
            ])
        let dev = try #require(MTLCreateSystemDefaultDevice())
        let queue = try #require(dev.makeCommandQueue())
        let store = try SafetensorsStore(directory: dir)
        let residency = try WeightResidency(device: dev, loading: store)
        for name in ["a", "b", "c"] { _ = try store.deviceTensor(name, device: dev) }
        store.residency = nil
        _ = try store.deviceTensor("a", device: dev)  // after the load: not added
        residency.finishLoad(queue: queue)
        #expect(residency.set.allocationCount == 3)
        residency.attach(to: queue)  // idempotent
        residency.detach()
        residency.detach()  // idempotent
    }
}
