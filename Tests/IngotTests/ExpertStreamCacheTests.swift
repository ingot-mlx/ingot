import Foundation
import Metal
import Testing

@testable import Ingot

/// The cache side of SSD expert streaming (ExpertStreamCache.swift, ExpertStreamIO.swift) without running a kernel:
/// an ExpertStream over a synthetic experts file (each chunk filled with its own key, chunk and word index), driven
/// like the decode path of `handle` does (protect the layer's union, touch the hits, `allocate` the misses, read them
/// and publish them). Checks that the slot tables (`slotOf`, `ownerOf`, `smap`, `smapR`) agree with ARC, that the
/// pass-through ring is never handed to the cache, that an evicted expert is unpublished, and that every expert the
/// GPU would read holds its own bytes after evictions and slot reuse (single and coalesced reads, and chunks that are
/// not multiples of 16 KiB, like the 4 KiB scales of toy-qwen36-moe).
@Suite struct ExpertStreamCacheTests {
    /// Bytes of each chunk of an expert: a 16 KiB one and two 4 KiB ones (offsets not aligned to 16 KiB).
    static let chunks = [16384, 4096, 4096]

    /// Fixture: `layers` × `experts` keys (as in the safetensors layout: per layer and chunk, the E experts in a row),
    /// and room for `slots` experts (the first E are the prefill ring).
    struct Fixture {
        let stream: ExpertStream
        let url: URL
        let layers: Int, experts: Int

        init(layers: Int, experts: Int, slots: Int) throws {
            self.layers = layers
            self.experts = experts
            url = FileManager.default.temporaryDirectory.appendingPathComponent("expertstream-\(UUID().uuidString).bin")
            let chunks = ExpertStreamCacheTests.chunks
            var data = Data(count: layers * experts * chunks.reduce(0, +))
            data.withUnsafeMutableBytes { raw in
                let w = raw.bindMemory(to: UInt32.self)
                for l in 0..<layers {
                    for i in chunks.indices {
                        for e in 0..<experts {
                            let base = Self.offset(l, i, e, experts) / 4
                            for j in 0..<(chunks[i] / 4) {
                                w[base + j] = Self.word(key: l * experts + e, piece: i, index: j)
                            }
                        }
                    }
                }
            }
            try data.write(to: url)
            let fd = open(url.path, O_RDONLY)
            precondition(fd >= 0)
            let src = (0..<layers).map { l in chunks.indices.map { i in (fd: fd, off: Self.offset(l, i, 0, experts)) } }
            let ctx = try MetalContext()
            do {
                stream = try ExpertStream(
                    ctx: ctx, experts: experts, topk: 4, layers: layers, src: src, pieceBytes: chunks, fds: [fd],
                    bytes: slots * chunks.reduce(0, +), prefetchK: 0, predStride: experts)
            } catch {
                close(fd)
                try? FileManager.default.removeItem(at: url)
                throw error
            }
            // The serving thread holds the stream unowned and blocks on its semaphore forever: keep it alive.
            _ = Unmanaged.passRetained(stream)
        }

        /// File offset of chunk i of expert e of layer l.
        static func offset(_ l: Int, _ i: Int, _ e: Int, _ E: Int) -> Int {
            let c = ExpertStreamCacheTests.chunks
            return l * E * c.reduce(0, +) + E * c[..<i].reduce(0, +) + e * c[i]
        }

        static func word(key: Int, piece: Int, index: Int) -> UInt32 {
            UInt32(key) << 16 | UInt32(piece) << 13 | UInt32(index)
        }

        func smap(_ k: Int) -> UInt32 { stream.smap.contents().load(fromByteOffset: k * 4, as: UInt32.self) }
        func smapR(_ k: Int) -> UInt32 { stream.smapR.contents().load(fromByteOffset: k * 4, as: UInt32.self) }

        /// The decode path of `handle` for one layer: the union `needed` is protected, hits are touched, misses get a
        /// slot from `allocate`, are read and published. Returns the number of misses.
        @discardableResult func demand(layer l: Int, _ needed: [Int]) -> Int {
            let s = stream
            let protect = Set(needed.map { l * experts + $0 })
            var miss: [(key: Int, slot: Int)] = []
            for e in needed {
                let k = l * experts + e
                if s.slotOf[k] >= 0 {
                    s.arc.touch(k)
                } else {
                    miss.append((k, s.allocate(k, protect: protect)))
                }
            }
            if !miss.isEmpty { s.readNow(miss) }
            for (k, slot) in miss { s.setResident(k, slot) }
            return miss.count
        }

        /// The expert in `slot` holds key k's bytes, every word of every chunk.
        func holds(_ k: Int, slot: Int) -> Bool {
            ExpertStreamCacheTests.chunks.indices.allSatisfy { i in
                let len = ExpertStreamCacheTests.chunks[i]
                let p = (stream.arenas[i].contents() + slot * len).bindMemory(to: UInt32.self, capacity: len / 4)
                return (0..<(len / 4)).allSatisfy { j in p[j] == Self.word(key: k, piece: i, index: j) }
            }
        }

        /// Tables against ARC: each resident key has a slot outside the ring, owned by it, in `smap`; the free slots
        /// and the resident ones add up to the cache; a key that is not resident has no slot and is unpublished.
        func checkTables() {
            let s = stream
            let Z = experts
            var used = Set<Int>()
            for k in 0..<(layers * experts) {
                let slot = Int(s.slotOf[k])
                let inARC = s.arc.which[k] == 1 || s.arc.which[k] == 2
                #expect((slot >= 0) == inARC, "key \(k): slot \(slot), ARC list \(s.arc.which[k])")
                if slot >= 0 {
                    #expect(slot >= Z && slot < s.slots, "key \(k) in slot \(slot) (ring 0..<\(Z))")
                    #expect(Int(s.ownerOf[slot]) == k)
                    #expect(smap(k) == UInt32(slot))
                    #expect(!used.contains(slot))
                    used.insert(slot)
                } else {
                    #expect(smapR(k) == 0xffff_ffff, "key \(k) evicted but still published")
                }
            }
            #expect(used.count == s.arc.resident)
            #expect(used.count + s.freeSlots.count == s.cacheSlots)
            #expect(s.freeSlots.allSatisfy { Int($0) >= Z && s.ownerOf[Int($0)] < 0 })
        }
    }

    @Test func slotsTablesAndBytesFollowARC() throws {
        // 3 layers × 8 experts (24 keys), 20 slots: 8 for the ring and 12 for the cache, so it evicts all the time.
        let f = try Fixture(layers: 3, experts: 8, slots: 20)
        defer { try? FileManager.default.removeItem(at: f.url) }
        #expect(f.stream.cacheSlots == 12)
        var g = SplitMix64(seed: 3)
        var misses = 0
        for round in 0..<200 {
            for l in 0..<f.layers {
                // The union of a round: 1-6 experts (consecutive ones go through the coalesced read).
                var needed = Set<Int>()
                let n = 1 + Int(g.next() % 6)
                while needed.count < n { needed.insert(Int(g.next() % 8)) }
                misses += f.demand(layer: l, needed.sorted())
                for e in needed {
                    let k = l * f.experts + e
                    let slot = Int(f.stream.slotOf[k])
                    #expect(slot >= 0)
                    #expect(f.smapR(k) == UInt32(slot))
                    #expect(f.holds(k, slot: slot), "round \(round), key \(k): wrong bytes in slot \(slot)")
                }
            }
            f.checkTables()
        }
        // Evictions really happened (more misses than keys) and residents are still the right bytes.
        #expect(misses > 3 * 24)
        for k in 0..<24 where f.stream.slotOf[k] >= 0 {
            #expect(f.holds(k, slot: Int(f.stream.slotOf[k])))
        }
    }

    @Test func evictedReadAheadIsNotPublished() throws {
        let f = try Fixture(layers: 3, experts: 8, slots: 18)  // 10 cache slots
        defer { try? FileManager.default.removeItem(at: f.url) }
        let s = f.stream
        // Read-ahead of layer 2 (keys 16, 17), as runPrefetch leaves it: slot taken, read, waiting in pendingKeys.
        let ahead = [16, 17].map { k in (key: k, slot: s.allocate(k, protect: [16, 17])) }
        s.readNow(ahead)
        s.pendingKeys[2] = ahead
        #expect(f.smapR(16) == 0xffff_ffff)
        // Layer 0 fills the cache (10 residents, all in T1); one expert of layer 1 then drops T1's LRU, which is 16.
        f.demand(layer: 0, Array(0..<8))
        #expect(s.slotOf[16] >= 0)
        f.demand(layer: 1, [0])
        #expect(s.slotOf[16] < 0)
        #expect(Int(s.slotOf[17]) == ahead[1].slot)
        // Reaching layer 2: 17 kept its slot and is published with its bytes; 16 stays unpublished (its slot now
        // belongs to layer 1's expert 0).
        s.publishPending(2)
        #expect(f.smapR(16) == 0xffff_ffff)
        #expect(Int(s.ownerOf[ahead[0].slot]) == 8)
        #expect(f.smapR(17) == UInt32(ahead[1].slot))
        #expect(f.holds(17, slot: ahead[1].slot))
        #expect(f.holds(8, slot: ahead[0].slot))
        #expect(s.pendingKeys[2].isEmpty)
        f.checkTables()
    }

    @Test func tooSmallCacheIsRejected() throws {
        // Fewer than two layers' worth of slots (ring + one layer): the error --expert-cache-gb reports.
        #expect(throws: SafetensorsError.self) { try Fixture(layers: 3, experts: 8, slots: 15) }
    }
}
