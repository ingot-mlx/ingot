import Foundation
import Metal

@testable import Ingot

// MARK: Block lists

extension KernelCase {
    static let fnBlockList = KernelCase(
        id: "fn-blocklist", model: "qwen3.8-flash-next",
        catches:
            "blocklist_from_bitmap (per-row bitmap → ascending lists with the open block) against the CPU, "
            + "and dattn_i8l_1_4 (round attention over the lists) against dattn_i8s (the bit mask) and the "
            + "dense one with ≤ 512 blocks, with random and clustered selections, 1–16 rows; a row gives bit-exact the "
            + "same with B = 1 as with B = 16"
    ) { ctx, g in
        let E = 64, stride = FlashQSA.topk + 2
        var checks: [Check] = []
        // (P, rows, selection: 0 random, 1 clustered in runs)
        for (P, R, kind) in [(1500, 16, 0), (2200, 4, 0), (9001, 16, 0), (9001, 16, 1), (6142, 7, 1)] {
            let f = try AttnFixture(ctx, positions: P, &g, maxCtx: P + 64)
            let a = f.a
            let x = Syn.roundHalf(Syn.normal(&g, R * f.hidden))
            let nb = (P + R) / 4 + 1, nw = (nb + 31) / 32
            var bm = [UInt32](repeating: 0, count: R * nw)
            var refLists: [[UInt32]] = []
            for r in 0..<R {
                let pos = P + r, vis = (pos + 1) / 4
                var sel: [Int]
                if vis <= 512 {
                    sel = Array(0..<vis)
                } else if kind == 0 {
                    var all = Array(0..<vis)
                    for i in 0..<512 { all.swapAt(i, i + Int(g.next() % UInt64(vis - i))) }
                    sel = Array(all.prefix(512)).sorted()
                } else {
                    var s = Set<Int>()
                    while s.count < 512 {
                        let st = Int(g.next() % UInt64(vis)), len = 8 + Int(g.next() % 57)
                        for b in st..<min(vis, st + len) where s.count < 512 { s.insert(b) }
                    }
                    sel = s.sorted()
                }
                // Bits beyond the complete blocks (the mask and the list ignore them).
                for b in sel { bm[r * nw + b / 32] |= 1 << UInt32(b % 32) }
                if vis + 1 < nb { bm[r * nw + (vis + 1) / 32] |= 1 << UInt32((vis + 1) % 32) }
                var l = sel.map { UInt32($0) << 1 | 1 }
                if (pos + 1) % 4 != 0 { l.append(UInt32(vis) << 1 | 1) }
                refLists.append(l)
            }
            let bmb = ctx.buffer(bm), st = ctx.buffer([UInt32(P), 0, 0, 0, 0, 0, 0, 0])
            let lists = BitmapBlockLists(ctx: ctx, stride: stride, perSplit: E)
            let xb = Syn.halfBuffer(ctx, x), out = ctx.buffer(length: R * f.hidden * 4)
            func halfs(_ rows: Int) -> [Float] {
                let p = a.attnOut.contents().bindMemory(to: UInt16.self, capacity: rows * f.outDim)
                return (0..<(rows * f.outDim)).map { Float(p[$0]) }
            }
            func attnOut(_ rows: Int) -> [Float] {
                let p = a.attnOut.contents().bindMemory(to: Float16.self, capacity: rows * f.outDim)
                return (0..<(rows * f.outDim)).map { Float(p[$0]) }
            }
            // Bit mask (dattn_i8s; with ≤ 512 visible blocks, the dense dattn_i8).
            a.pos = P
            a.qsaMask = P + R - 1 >= FlashQSA.firstSparsePos ? (bmb, nw) : nil
            try ctx.run { e in try a.encode(e, x: xb, out: out, rows: R, state: st) }
            let refO = attnOut(R)
            a.qsaMask = nil
            // Lists.
            try ctx.run { e in
                a.blockList = try lists.encode(e, bitmap: bmb, words: nw, rows: R, pos: P, state: st)
                try a.encode(e, x: xb, out: out, rows: R, state: st)
            }
            let lo = attnOut(R), lbits = halfs(R)
            let tag = "P \(P), \(R) rows, \(kind == 0 ? "random" : "grouped")"
            let cnt = Syn.uints(lists.counts, R), lv = Syn.uints(lists.lists, R * stride)
            var gotL: [Float] = [], wantL: [Float] = []
            for r in 0..<R {
                wantL += [Float(refLists[r].count)] + refLists[r].map { Float($0) }
                gotL += [Float(cnt[r])] + lv[(r * stride)..<(r * stride + Int(cnt[r]))].map { Float($0) }
            }
            checks.append(Check(label: "lists (\(tag))", gpu: gotL, ref: wantL, tol: .bitExact))
            checks.append(
                Check(label: "dattn_i8l against the mask (\(tag))", gpu: lo, ref: refO, tol: .maxRel(2e-3)))
            // B = 1 against B = R: each row alone at its position (the block's KV is already written).
            var b1: [Float] = [], bR: [Float] = []
            for r in Set([0, R / 2, R - 1]).sorted() {
                let st1 = ctx.buffer([UInt32(P + r), 0, 0, 0, 0, 0, 0, 0])
                let x1 = Syn.halfBuffer(ctx, Array(x[(r * f.hidden)..<((r + 1) * f.hidden)]))
                a.pos = P + r
                try ctx.run { e in
                    a.blockList = try lists.encode(
                        e, bitmap: bmb, offset: r * nw * 4, words: nw, rows: 1, pos: P + r, state: st1)
                    try a.encode(e, x: x1, out: out, rows: 1, state: st1)
                }
                b1 += halfs(1)
                bR += Array(lbits[(r * f.outDim)..<((r + 1) * f.outDim)])
            }
            a.blockList = nil
            checks.append(Check(label: "B = 1 against B = \(R) (\(tag))", gpu: b1, ref: bR, tol: .bitExact))
        }
        return checks
    }
}
