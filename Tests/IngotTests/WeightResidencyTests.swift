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

    /// The heartbeat runs while used, ends the residency after `idleRelease` without use, and the next use requests
    /// it again (and says how long that took).
    @Test func heartbeatHoldsThenReleases() async throws {
        let dev = try #require(MTLCreateSystemDefaultDevice())
        let r = try WeightResidency(device: dev, heartbeat: 0.02, idleRelease: 0.3)
        #expect(!r.isHeld)
        #expect(r.use() != nil)  // first use: requested now
        #expect(r.isHeld)
        try await Task.sleep(nanoseconds: 150_000_000)
        #expect(r.use() == nil)  // still held
        #expect(r.beats > 0)
        try await Task.sleep(nanoseconds: 900_000_000)
        #expect(!r.isHeld)
        let beats = r.beats
        try await Task.sleep(nanoseconds: 100_000_000)
        #expect(r.beats == beats)  // stopped
        #expect(r.use() != nil)  // requested again
        #expect(r.isHeld)
        r.detach()
        #expect(!r.isHeld)
    }

    /// The state set follows the buffers it is given, and a buffer that leaves it is freed.
    @Test func stateSetFollowsTheEngine() throws {
        let dev = try #require(MTLCreateSystemDefaultDevice())
        let r = try WeightResidency(device: dev)
        let a = try #require(dev.makeBuffer(length: 1024, options: .storageModeShared))
        weak var gone: MTLBuffer?
        do {
            let b = try #require(dev.makeBuffer(length: 1024, options: .storageModeShared))
            gone = b
            r.setStateBuffers([a, b, a])
            #expect(r.stateSet.allocationCount == 2)
        }
        #expect(gone != nil)  // the set holds it
        r.setStateBuffers([a])
        #expect(r.stateSet.allocationCount == 1)
        #expect(gone == nil)
        r.detach()
    }
}
