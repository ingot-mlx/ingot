import Foundation
import Metal
import Testing

@testable import MetalLab

/// The library cache of `MetalContext`: one lock per key, so the same key requested from several
/// threads compiles once and every caller gets that library. It lives in `KernelLibraryTests` because it uses the
/// device (the CI runs it with the GPU tests).
extension KernelLibraryTests {

    @Test func sameKeyFromManyThreadsCompilesOnce() throws {
        // A key no other test uses (a unique comment as extra source), so the count is only this test's.
        let extra = ["// KernelLibraryCacheTests \(UUID().uuidString)"]
        let dims = ModelDims()
        #expect(MetalContext.libraryCompiles(dims: dims, extraSources: extra) == 0)
        let threads = 8
        let results = Results()
        DispatchQueue.concurrentPerform(iterations: threads) { _ in
            do {
                let ctx = try MetalContext(dims: dims, extraSources: extra)
                results.add(library: ObjectIdentifier(ctx.library as AnyObject))
            } catch {
                results.add(error: "\(error)")
            }
        }
        let (libraries, errors) = results.snapshot()
        #expect(errors.isEmpty, "\(errors)")
        #expect(libraries.count == threads)
        #expect(Set(libraries).count == 1, "every context must share the one compiled library")
        #expect(MetalContext.libraryCompiles(dims: dims, extraSources: extra) == 1)
    }

    /// What the threads report, behind a lock.
    private final class Results: @unchecked Sendable {
        private let lock = NSLock()
        private var libraries: [ObjectIdentifier] = [], errors: [String] = []
        func add(library: ObjectIdentifier) { lock.withLock { libraries.append(library) } }
        func add(error: String) { lock.withLock { errors.append(error) } }
        func snapshot() -> ([ObjectIdentifier], [String]) { lock.withLock { (libraries, errors) } }
    }
}
