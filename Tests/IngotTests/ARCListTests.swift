import Foundation
import Testing

@testable import Ingot

/// ARCList (the replacement policy of the SSD expert cache, ExpertStream): insertion, hits, eviction into the ghost
/// lists B1/B2, reinsertion of an evicted key (ghost hit, adapting p), the capacity limits and what can never be
/// evicted (`protect`, `exclude`, `excludePred`). CPU only.
@Suite struct ARCListTests {
    /// Keys of list w from MRU (head) to LRU (tail), checking the links both ways and the count.
    static func keys(_ a: ARCList, _ w: Int) -> [Int] {
        var out: [Int] = []
        var k = a.head[w]
        var prev: Int32 = -1
        while k >= 0 {
            #expect(a.which[Int(k)] == UInt8(w))
            #expect(a.prev[Int(k)] == prev)
            out.append(Int(k))
            prev = k
            k = a.next[Int(k)]
        }
        #expect(a.tail[w] == prev)
        #expect(a.count[w] == out.count)
        return out
    }

    /// The ARC invariants (Megiddo and Modha): |T1| + |T2| ≤ c, |T1| + |B1| ≤ c, |T1| + |T2| + |B1| + |B2| ≤ 2c,
    /// 0 ≤ p ≤ c; and every key is in at most one list.
    static func checkInvariants(_ a: ARCList) {
        let l = (1...4).map { keys(a, $0) }
        #expect(a.resident == l[0].count + l[1].count)
        #expect(a.resident <= a.c)
        #expect(l[0].count + l[2].count <= a.c)
        #expect(l.map(\.count).reduce(0, +) <= 2 * a.c)
        #expect(a.p >= 0 && a.p <= Double(a.c))
        let all = l.flatMap { $0 }
        #expect(Set(all).count == all.count)
        #expect(a.which.filter { $0 != 0 }.count == all.count)
    }

    @Test func fillsWithoutEvictingUntilFull() {
        var a = ARCList(keys: 16, capacity: 4)
        for k in 0..<4 { #expect(a.insert(k, protect: []) == nil) }
        #expect(a.resident == 4)
        // New keys go to T1 at the MRU end.
        #expect(Self.keys(a, 1) == [3, 2, 1, 0])
        Self.checkInvariants(a)
    }

    @Test func hitMovesToFrequentList() {
        var a = ARCList(keys: 16, capacity: 4)
        for k in 0..<4 { _ = a.insert(k, protect: []) }
        a.touch(1)
        a.touch(0)
        #expect(Self.keys(a, 1) == [3, 2])
        #expect(Self.keys(a, 2) == [0, 1])
        #expect(a.resident == 4)
        Self.checkInvariants(a)
    }

    @Test func dropsLRUOfRecentListWhenItHoldsEverything() {
        var a = ARCList(keys: 16, capacity: 4)
        for k in 0..<4 { _ = a.insert(k, protect: []) }
        // p = 0 and T1 is the whole cache: its LRU (0) leaves.
        #expect(a.insert(4, protect: []) == 0)
        #expect(a.resident == 4)
        Self.checkInvariants(a)
        // |T1| = c: ARC drops T1's LRU without a ghost (case |L1| = c, |T1| = c), never above capacity.
        #expect(a.which[0] == 0)
        #expect(Self.keys(a, 1) == [4, 3, 2, 1])
    }

    @Test func evictsIntoB1WhenT2IsNotEmpty() {
        var a = ARCList(keys: 16, capacity: 4)
        for k in 0..<4 { _ = a.insert(k, protect: []) }
        a.touch(3)
        // T1 = [2, 1, 0], T2 = [3]; T1 > p = 0: T1's LRU (0) goes to B1.
        #expect(a.insert(4, protect: []) == 0)
        #expect(a.which[0] == 3)
        #expect(Self.keys(a, 3) == [0])
        #expect(Self.keys(a, 2) == [3])
        Self.checkInvariants(a)
    }

    @Test func reinsertionFromB1GrowsPAndGoesToT2() {
        var a = ARCList(keys: 16, capacity: 4)
        for k in 0..<4 { _ = a.insert(k, protect: []) }
        a.touch(3)
        #expect(a.insert(4, protect: []) == 0)  // 0 → B1
        #expect(a.p == 0)
        // 0 comes back (ghost hit in B1): p grows by max(|B2| / |B1|, 1) = 1, 0 goes to T2 and someone is evicted.
        let out = a.insert(0, protect: [])
        #expect(a.p == 1)
        #expect(a.which[0] == 2)
        #expect(out == 1)  // T1 = [4, 2, 1] > p = 1: its LRU
        #expect(a.which[1] == 3)
        #expect(a.resident == 4)
        Self.checkInvariants(a)
    }

    @Test func reinsertionFromB2ShrinksP() {
        var a = ARCList(keys: 16, capacity: 2)
        _ = a.insert(0, protect: [])
        _ = a.insert(1, protect: [])
        a.touch(0)
        a.touch(1)  // T2 = [1, 0], T1 empty
        // A new key with T1 empty: T2's LRU (0) goes to B2.
        #expect(a.insert(2, protect: []) == 0)
        #expect(a.which[0] == 4)
        // 2 → B1 by a hit on 3: T1 = [2] > p = 0.
        #expect(a.insert(3, protect: []) == 2)
        #expect(a.which[2] == 3)
        // B1 ghost hit: p = 1.
        #expect(a.insert(2, protect: []) != nil)
        #expect(a.p == 1)
        // B2 ghost hit on 0: p shrinks by max(|B1| / |B2|, 1), clamped at 0.
        let b1 = Double(a.count[3]), b2 = Double(a.count[4])
        let expected = max(0, 1 - max(b1 / max(b2, 1), 1))
        #expect(a.insert(0, protect: []) != nil)
        #expect(a.p == expected)
        #expect(a.which[0] == 2)
        Self.checkInvariants(a)
    }

    @Test func protectedKeysAreNeverEvicted() {
        var a = ARCList(keys: 16, capacity: 3)
        for k in 0..<3 { _ = a.insert(k, protect: []) }
        // T1 = [2, 1, 0]: with 0 and 1 protected, only 2 can leave.
        #expect(a.insert(3, protect: [0, 1]) == 2)
        #expect(a.which[0] == 1 && a.which[1] == 1)
        Self.checkInvariants(a)
    }

    @Test func excludeRangeAndPredicateAreNeverEvicted() {
        var a = ARCList(keys: 32, capacity: 4)
        for k in [0, 1, 8, 9] { _ = a.insert(k, protect: []) }
        a.touch(8)
        a.touch(9)
        // Layer 0 (keys 0..<8) excluded, as runPrefetch does with the layer the GPU is heading to: T1 holds only
        // layer 0, so the victim comes from T2.
        a.exclude = 0..<8
        #expect(a.insert(16, protect: []) == 8)
        a.exclude = 0..<0
        // excludePred (layers with read-ahead in flight): layer 0 (0..<8) busy; T1 = [16, 1, 0], T2 = [9].
        a.excludePred = { $0 / 8 == 0 }
        #expect(a.insert(17, protect: []) == 16)
        a.excludePred = nil
        Self.checkInvariants(a)
    }

    /// A long random trace (reuse like that of experts: a hot set and a cold tail) with random protected keys: the
    /// invariants hold at every step, the cache never goes above capacity, an eviction only happens when full, and the
    /// evicted key was resident and is no longer.
    @Test func randomTraceKeepsInvariants() {
        var g = SplitMix64(seed: 7)
        let keys = 64, c = 12
        var a = ARCList(keys: keys, capacity: c)
        var resident = Set<Int>()
        var hits = 0, ghostHits = 0
        for step in 0..<4000 {
            let k = g.uniform() < 0.7 ? Int(g.next() % 10) : 10 + Int(g.next() % UInt64(keys - 10))
            if resident.contains(k) {
                hits += 1
                a.touch(k)
            } else {
                if a.which[k] == 3 || a.which[k] == 4 { ghostHits += 1 }
                var protect = Set<Int>([k])
                for r in resident where g.uniform() < 0.3 && protect.count < c - 1 { protect.insert(r) }
                let full = a.resident >= c
                let out = a.insert(k, protect: protect)
                #expect((out != nil) == full)
                if let v = out {
                    #expect(resident.contains(v))
                    #expect(!protect.contains(v))
                    #expect(a.which[v] != 1 && a.which[v] != 2)
                    resident.remove(v)
                }
                resident.insert(k)
            }
            #expect(a.resident == resident.count)
            if step % 100 == 0 { Self.checkInvariants(a) }
        }
        Self.checkInvariants(a)
        // The trace exercises both hits and ghost hits (otherwise B1/B2 are not being tested).
        #expect(hits > 1000)
        #expect(ghostHits > 50)
    }
}
