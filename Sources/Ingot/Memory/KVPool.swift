import Foundation
import Metal

/// KV that grows on demand (Flash-Next in `ingot serve`: 256K of context without taking expert memory away from
/// normal conversations).
///
/// Each attention layer's KV (kc, vt, ks, vs) and the QSA block keys are reserved for `capacity`
/// positions (one `extent` chunk at load) and, when a conversation needs more, they are reallocated bigger:
/// new zeroed buffers, copy of the old contents with the new stride (blit) and the old ones dropped, layer by layer
/// (at most one extra layer at a time). When a shorter conversation starts they are trimmed the same way. Exact: the
/// same data and the same computations (only the buffers' `ldkv` stride changes). Always with the GPU stopped.
///
/// The planner counts the KV for `allowance` positions (like splash, 64K); whatever grows beyond that is paid by the
/// prefix cache and the snapshots (`InferenceEngine`) and, if more is needed, by the budget slack.
///
/// (It was previously tried to have the KV yield slots from the expert cache with a shared placement heap and sparse
/// Metal 4 buffers: exact only with a residency set on the queue, which made the round 4–5 times slower;
/// without it, long prefill came out wrong.)
public final class KVPool: @unchecked Sendable {
    public static let extent = 16_384

    public let ctx: MetalContext
    /// Positions the buffers currently hold and the maximum (the model's context).
    public private(set) var capacity: Int
    public let maxPositions: Int
    /// Reallocation of each region (an attention layer, the block keys of a QSA layer) to a given capacity.
    var resizers: [(Int) -> Void] = []
    /// Bytes per position across all regions.
    public internal(set) var bytesPerPosition = 0
    public private(set) var grows = 0, shrinks = 0
    /// Positions the plan counts; whatever exceeds that is yielded by the prefix cache and the snapshots.
    public let allowance: Int

    public init(ctx: MetalContext, maxPositions: Int, allowance: Int = 65_536) {
        self.ctx = ctx
        self.maxPositions = maxPositions
        self.allowance = allowance
        capacity = min(maxPositions, Self.extent)
    }

    public var bytes: Int { bytesPerPosition * capacity }
    /// KV bytes above the plan's allocation.
    public var overAllowanceBytes: Int { max(0, bytesPerPosition * (capacity - allowance)) }

    /// Row-wise copy: `rows` rows of `n` bytes with strides `ss` (source) and `ds` (destination).
    static func copyRows(
        _ b: MTLBlitCommandEncoder, _ src: MTLBuffer, _ dst: MTLBuffer, rows: Int, n: Int, ss: Int, ds: Int
    ) {
        for r in 0..<rows { b.copy(from: src, sourceOffset: r * ss, to: dst, destinationOffset: r * ds, size: n) }
    }

    /// A blit command buffer running `body`, awaited.
    func blit(_ body: (MTLBlitCommandEncoder) -> Void) {
        let cb = ctx.queue.makeCommandBuffer()!
        let b = cb.makeBlitCommandEncoder()!
        body(b)
        b.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
    }

    /// Capacity for `positions` positions (in `extent` chunks, up to the maximum), with the GPU stopped.
    @discardableResult
    public func ensure(positions: Int) -> Bool {
        guard positions <= maxPositions else { return false }
        let want = min(maxPositions, (positions + Self.extent - 1) / Self.extent * Self.extent)
        guard want > capacity else { return true }
        resize(want)
        grows += 1
        return true
    }

    /// Trims to `positions` (in chunks, at least one) if two or more chunks are spare; with the GPU stopped and only if
    /// nothing live uses the positions beyond.
    public func trim(positions: Int) {
        let want = max(Self.extent, (positions + Self.extent - 1) / Self.extent * Self.extent)
        guard capacity >= want + 2 * Self.extent else { return }
        resize(want)
        shrinks += 1
    }

    private func resize(_ cap: Int) {
        // Wait for whatever is queued (prefix cache copies, rounds) before touching the buffers.
        let cb = ctx.queue.makeCommandBuffer()!
        cb.commit()
        cb.waitUntilCompleted()
        for r in resizers { r(cap) }
        capacity = cap
    }

    public func describe() -> String {
        String(
            format: "KV on demand: %d positions (%.2f GB, %.1f KB per position; maximum %d, the plan counts %d)",
            capacity, Double(bytes) / 1e9, Double(bytesPerPosition) / 1e3, maxPositions, allowance)
    }
}
