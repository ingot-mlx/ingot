import Foundation

/// ARC (Megiddo and Modha) over keys 0..<keys with intrusive lists: T1 (recent), T2 (frequent) and their
/// ghosts B1, B2. `insert` returns the resident key that gets evicted (if room is needed).
struct ARCList {
    var c: Int
    /// Keys that cannot be evicted right now (predicate; slots lent to the ring).
    var excludePred: ((Int) -> Bool)?
    var p = 0.0
    var which: [UInt8]  // 0 none, 1 T1, 2 T2, 3 B1, 4 B2
    var prev: [Int32], next: [Int32]
    var head = [Int32](repeating: -1, count: 5), tail = [Int32](repeating: -1, count: 5)
    var count = [Int](repeating: 0, count: 5)

    init(keys: Int, capacity: Int) {
        c = capacity
        which = [UInt8](repeating: 0, count: keys)
        prev = [Int32](repeating: -1, count: keys)
        next = [Int32](repeating: -1, count: keys)
    }

    var resident: Int { count[1] + count[2] }

    mutating func unlink(_ k: Int) {
        let w = Int(which[k])
        guard w != 0 else { return }
        let p = prev[k], n = next[k]
        if p >= 0 { next[Int(p)] = n } else { head[w] = n }
        if n >= 0 { prev[Int(n)] = p } else { tail[w] = p }
        prev[k] = -1; next[k] = -1
        which[k] = 0
        count[w] -= 1
    }

    /// To the MRU end (head) of list w.
    mutating func pushMRU(_ k: Int, _ w: Int) {
        unlink(k)
        which[k] = UInt8(w)
        next[k] = head[w]
        prev[k] = -1
        if head[w] >= 0 { prev[Int(head[w])] = Int32(k) } else { tail[w] = Int32(k) }
        head[w] = Int32(k)
        count[w] += 1
    }

    /// The oldest entry of list w that is not protected.
    /// Keys that cannot be evicted right now (in addition to `protect`).
    var exclude = 0..<0

    func lru(_ w: Int, protect: Set<Int>) -> Int? {
        var k = tail[w]
        while k >= 0 {
            if !protect.contains(Int(k)) && !exclude.contains(Int(k)) && !(excludePred?(Int(k)) ?? false) {
                return Int(k)
            }
            k = prev[Int(k)]
        }
        return nil
    }

    mutating func touch(_ k: Int) { pushMRU(k, 2) }

    /// Evicts a resident (to its ghost list) and returns it.
    mutating func replace(_ k: Int, protect: Set<Int>) -> Int? {
        let t1 = Double(count[1])
        if count[1] > 0 && (t1 > p || (which[k] == 4 && t1 == p)), let v = lru(1, protect: protect) {
            pushMRU(v, 3)
            return v
        }
        if let v = lru(2, protect: protect) {
            pushMRU(v, 4)
            return v
        }
        if let v = lru(1, protect: protect) {
            pushMRU(v, 3)
            return v
        }
        return nil
    }

    /// Inserts k (not resident). Returns the resident key that was evicted, if any.
    mutating func insert(_ k: Int, protect: Set<Int>) -> Int? {
        let full = resident >= c
        var out: Int?
        switch which[k] {
        case 3:
            p = min(Double(c), p + max(Double(count[4]) / Double(max(count[3], 1)), 1))
            if full { out = replace(k, protect: protect) }
            pushMRU(k, 2)
        case 4:
            p = max(0, p - max(Double(count[3]) / Double(max(count[4], 1)), 1))
            if full { out = replace(k, protect: protect) }
            pushMRU(k, 2)
        default:
            let l1 = count[1] + count[3]
            if l1 >= c {
                if count[1] < c {
                    if tail[3] >= 0 { unlink(Int(tail[3])) }
                    if full { out = replace(k, protect: protect) }
                } else if full, let v = lru(1, protect: protect) {
                    unlink(v)
                    out = v
                }
            } else {
                let tot = l1 + count[2] + count[4]
                if tot >= c {
                    if tot >= 2 * c && tail[4] >= 0 { unlink(Int(tail[4])) }
                    if full { out = replace(k, protect: protect) }
                }
            }
            pushMRU(k, 1)
        }
        precondition(!full || out != nil, "cache full and everything protected")
        return out
    }
}
