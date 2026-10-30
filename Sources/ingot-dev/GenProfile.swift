import Foundation
import Ingot
import IngotCLI
import Metal

// ingot-dev gen --profile: the decode round split by piece (with and without a drafter), the batching ceiling and the
// prefill timeline.

/// --profile without a drafter: the B = 1 round split by piece at the end of this prompt. Each piece over
/// all its layers in one command buffer, GPU median of 20 after 5 warm-ups; bytes = weights the piece reads. Works on
/// models with branch epilogues (timelineVerify refuses them); attention, short-conv and dense-FFN layers only.
@MainActor func genPieceProfile(
    _ m: HybridModel, _ dr: DFlash2Drafter?, _ ctx: MetalContext, label: String
) throws {
    guard args.flags.contains("profile"), dr == nil else { return }
    let supported = m.layers.allSatisfy { l in
        switch l.mixer {
        case .attn, .conv: return l.dense != nil
        default: return false
        }
    }
    guard supported, m.fn == nil else {
        print("  pieces: only attention, short-conv and dense-FFN layers")
        return
    }
    func gpu(_ f: (MTLComputeCommandEncoder) throws -> Void) throws -> Double {
        var t: [Double] = []
        for _ in 0..<25 {
            let cb = ctx.queue.makeCommandBuffer()!, e = cb.makeComputeCommandEncoder()!
            try f(e); e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
            t.append((cb.gpuEndTime - cb.gpuStartTime) * 1e3)
        }
        return median(Array(t.dropFirst(5)))
    }
    let P = m.P
    m.nacc = 1
    func line(_ name: String, _ t: Double, _ bytes: Int) {
        print(
            String(
                format: "  pieces %@ P=%d %-22@ %7.3f ms  %7.1f MB  %6.1f GB/s", label, P, name as NSString, t,
                Double(bytes) / 1e6, bytes > 0 ? Double(bytes) / (t * 1e6) : 0))
    }
    let full = try gpu { _ = try m.encodeVerify($0, rows: 1, drafter: nil) }
    line("round(B=1)", full, m.weightBytes)
    var sum = 0.0
    let attn = m.layers.compactMap { l -> AttentionLayer? in if case .attn(let a) = l.mixer { a } else { nil } }
    for piece in AttentionLayer.Piece.allCases where !attn.isEmpty {
        let t = try gpu { e in
            for a in attn { a.pos = P; try a.encode(e, x: m.xn, out: m.h, rows: 1, accumulate: true, only: piece) }
        }
        let b =
            piece == .proj
            ? attn.reduce(0) { $0 + $1.proj.bytes } : piece == .oproj ? attn.reduce(0) { $0 + $1.oProj.bytes } : 0
        line("attn.\(piece.rawValue) x\(attn.count)", t, b)
        sum += t
    }
    let convs = m.layers.compactMap { l -> ShortConvLayer? in if case .conv(let c) = l.mixer { c } else { nil } }
    if !convs.isEmpty {
        let t = try gpu { e in for c in convs { try c.encode(e, x: m.xn, out: m.h, rows: 1, nacc: 1); c.advance() } }
        line("conv x\(convs.count)", t, convs.reduce(0) { $0 + $1.weightBytes })
        sum += t
    }
    let dense = m.layers.compactMap(\.dense)
    for piece in DenseMLP.Piece.allCases where !dense.isEmpty {
        let t = try gpu { e in for d in dense { try d.encode(e, x: m.xn, out: m.h, rows: 1, only: piece) } }
        let b =
            piece == .gateup
            ? dense.reduce(0) { $0 + $1.gu.bytes } : piece == .down ? dense.reduce(0) { $0 + $1.down.bytes } : 0
        line("ffn.\(piece.rawValue) x\(dense.count)", t, b)
        sum += t
    }
    let head = try gpu { try m.vqHead.encode($0, x: m.xn, m.lmHead, y: m.logits, rows: 1) }
    line("head", head, m.lmHead.bytes)
    sum += head
    print(
        String(
            format: "  pieces %@ P=%d sum %.3f ms; round %.3f ms; rest (norms, epilogues, embed, argmax, gaps) %.3f ms",
            label, P, sum, full, full - sum))
}

/// --profile: three prefill chunks of this prompt (the second, the middle one and the last, with the same balanced
/// split as `prefill`) re-run at their positions by the real path and split by layer kind and piece on the GPU
/// timeline. Median of 3 after a warm-up; `1 encoder` is the same chunk without the timeline's encoders.
@MainActor func genPrefillTimeline(_ m: HybridModel, promptTokens T: Int, label: String) throws {
    guard args.flags.contains("profile"), m.fn == nil, m.expertStream == nil else { return }
    let n = (T + m.chunkRows - 1) / m.chunkRows
    let per = min(m.chunkRows, ((T + n - 1) / n + 63) / 64 * 64)
    let chunks = Array(stride(from: 0, to: T, by: per))
    for c in Set([min(1, chunks.count - 1), chunks.count / 2, chunks.count - 1]).sorted() {
        let p0 = chunks[c], rows = min(per, T - p0)
        _ = try m.timelinePrefill(rows: rows, at: p0)
        let runs = try (0..<3).map { _ in try m.timelinePrefill(rows: rows, at: p0) }
        let total = median(runs.map(\.total)), plain = median(runs.map(\.plain))
        print(
            String(
                format: "  prefill-timeline %@ p0=%d rows=%d: 1 encoder %.3f ms; timeline %.3f ms (gaps %.3f ms)",
                label, p0, rows, plain, total, median(runs.map(\.gaps))))
        for (i, (name, _)) in runs[0].pieces.enumerated() {
            let t = median(runs.map { $0.pieces[i].1 })
            print(
                String(
                    format: "  prefill-timeline %@ p0=%d %-22@ %8.3f ms  %5.1f%%", label, p0, name as NSString, t,
                    100 * t / total))
        }
    }
}

/// --profile: the pieces of the round separately (each in its own command buffer, GPU median), with the final state.
@MainActor func genProfile(_ m: HybridModel, _ dr: DFlash2Drafter, _ ctx: MetalContext) throws {
    func gpu(_ f: (MTLComputeCommandEncoder) throws -> Void) throws -> Double {
        var t: [Double] = []
        for _ in 0..<25 {
            let cb = ctx.queue.makeCommandBuffer()!, e = cb.makeComputeCommandEncoder()!
            try f(e); e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
            t.append((cb.gpuEndTime - cb.gpuStartTime) * 1e3)
        }
        return median(Array(t.dropFirst(5)))
    }
    let P = m.P
    let a = try gpu { try dr.encodeContext($0, rows: 8, nvalid: 4, pos0: P - 4) }
    let b = try gpu { try dr.encodeBlock($0, P: P, embed: m.embed, lmHead: m.lmHead, vtok: m.tok) }
    let c = try gpu { try m.encodeVerify($0, rows: 8, drafter: dr) }
    let d = try gpu { try m.encodeVerify($0, rows: 1, drafter: nil) }
    let all = try gpu { e in
        try dr.encodeContext(e, rows: 8, nvalid: 4, pos0: P - 4)
        try dr.encodeBlock(e, P: P, embed: m.embed, lmHead: m.lmHead, vtok: m.tok)
        try m.encodeVerify(e, rows: 8, drafter: dr)
    }
    let tp = m.tok.contents().bindMemory(to: UInt32.self, capacity: 8)
    // Fixed tokens (frequent in text) to compare configurations with roughly the same union of experts.
    let fixedToks: [UInt32] = [279, 3681, 315, 264, 1532, 11, 323, 13]
    let rnd = try gpu { e in
        for i in 0..<8 { tp[i] = fixedToks[i] }
        try m.encodeVerify(e, rows: 8, drafter: dr)
    }
    let nl = m.layers.count,
        na = m.layers.filter { if case .attn = $0.mixer { return true } else { return false } }.count
    for B in [1, 8] {
        for piece in AttentionLayer.Piece.allCases {
            let t = try gpu { e in
                for l in m.layers {
                    if case .attn(let at) = l.mixer {
                        at.pos = P; try at.encode(e, x: m.xn, out: m.h, rows: B, accumulate: true, only: piece)
                    }
                }
            }
            print(String(format: "profile: attention B=%d, %d layers, %@: %.3f ms", B, na, piece.rawValue, t))
        }
    }
    if m.isMoE {
        for piece in MoELayer.Piece.allCases {
            let t = try gpu { e in
                for l in m.layers { try l.moe.encode(e, x: m.xn, out: m.h, rows: 8, accumulate: true, only: piece) }
            }
            print(String(format: "profile: MoE B=8, %d layers, %@: %.3f ms", nl, piece.rawValue, t))
        }
    } else {
        for B in [1, 8] {
            for piece in DenseMLP.Piece.allCases {
                let t = try gpu { e in
                    for l in m.layers { try l.dense!.encode(e, x: m.xn, out: m.h, rows: B, only: piece) }
                }
                print(String(format: "profile: dense FFN B=%d, %d layers, %@: %.3f ms", B, nl, piece.rawValue, t))
            }
        }
        let fb = m.layers.reduce(0) { $0 + $1.dense!.bytes }
        print(
            String(
                format: "  FFN: %.2f GB per round → floor %.2f ms", Double(fb) / 1e9,
                Double(fb) / (measuredReadGBps * 1e6)))
    }
    for piece in GDNLayer.Piece.allCases {
        let t = try gpu { e in
            for l in m.layers {
                if case .gdn(let g) = l.mixer {
                    try g.encode(e, x: m.xn, out: m.h, rows: 8, nacc: 4, accumulate: true, only: piece)
                }
            }
        }
        print(String(format: "profile: GDN B=8, %d layers, %@: %.3f ms", nl - na, piece.rawValue, t))
    }
    for _ in 0..<2 {
        let dt = try GPUTimeline(ctx: ctx)
        try dr.encodeBlock(nil, P: P, embed: m.embed, lmHead: m.lmHead, vtok: m.tok, timeline: dt)
        let dtl = try dt.run()
        print(
            String(format: "drafter timeline: GPU %.2f ms, gaps %.2f ms; ", dtl.total, dtl.gaps)
                + dtl.pieces.map { String(format: "%@ %.2f", $0.0, $0.1) }.joined(separator: ", "))
        let tl = try m.timelineVerify(rows: 8, drafter: dr)
        print(
            String(format: "timeline B=8: GPU %.2f ms, gaps between encoders %.2f ms; ", tl.total, tl.gaps)
                + tl.pieces.map { String(format: "%@ %.2f", $0.0, $0.1) }.joined(separator: ", "))
    }
    print(
        String(
            format:
                "profile: full round in one command buffer %.2f ms; B=8 verification with fixed tokens %.2f ms",
            all, rnd))
    for na in [1, 8] {
        m.nacc = na
        let t = try gpu { e in
            for i in 0..<8 { tp[i] = UInt32.random(in: 0..<200000) }; try m.encodeVerify(e, rows: 8, drafter: dr)
        }
        print(String(format: "profile: verification with nacc = %d: %.2f ms", na, t))
    }
    print(
        String(
            format:
                "profile (P = %d): drafter context %.2f ms, drafter block %.2f ms, B=8 verification %.2f ms, B=1 decode %.2f ms",
            P, a, b, c, d))
}

/// --profile without a drafter (normal mode, or a family without one): the ceiling of batching N conversations in
/// one decode round, at the position where the prompt's generation ended. Prints, GPU and wall medians:
/// - `verify(B)`: the B = 1, 2, 4, 8 round (weights read once for all rows, the KV of this conversation only);
/// - `state(1)`: what each extra conversation adds, its state pieces with B = 1 (attention prework+attn+reduce over
///   its KV, GDN prework+recur), measured as the difference against verify(1) in the same command buffer, so its
///   KV comes from memory and not from the cache;
/// - `sim(N)`: verify(N) plus N − 1 passes of state pieces in one command buffer (each pass sweeps all the layers);
/// - the ceiling N · verify(1) / round(N), and the state bytes of one conversation.
@MainActor func genProfileBatch(
    _ m: HybridModel, _ dr: DFlash2Drafter?, _ ctx: MetalContext, label: String
) throws {
    guard args.flags.contains("profile"), dr == nil else { return }
    func time(_ f: (MTLComputeCommandEncoder) throws -> Void) throws -> (gpu: Double, wall: Double) {
        var g: [Double] = [], w: [Double] = []
        for _ in 0..<25 {
            let t0 = DispatchTime.now().uptimeNanoseconds
            let cb = ctx.queue.makeCommandBuffer()!, e = cb.makeComputeCommandEncoder()!
            try f(e); e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
            g.append((cb.gpuEndTime - cb.gpuStartTime) * 1e3)
            w.append(Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6)
        }
        return (median(Array(g.dropFirst(5))), median(Array(w.dropFirst(5))))
    }
    let P = m.P
    let tp = m.tok.contents().bindMemory(to: UInt32.self, capacity: 8)
    // Distinct frequent tokens: with an MoE the rows route to different experts, as different conversations would.
    let fixedToks: [UInt32] = [279, 3681, 315, 264, 1532, 11, 323, 13]
    for i in 0..<8 { tp[i] = fixedToks[i] }
    m.nacc = 1
    // State pieces of one conversation (B = 1): attention over its KV and the GDN recurrence over its state.
    func statePass(_ e: MTLComputeCommandEncoder) throws {
        for l in m.layers {
            switch l.mixer {
            case .attn(let at):
                at.pos = P
                for p in [AttentionLayer.Piece.prework, .attn, .reduce] {
                    try at.encode(e, x: m.xn, out: m.h, rows: 1, accumulate: true, only: p)
                }
            case .gdn(let g):
                for p in [GDNLayer.Piece.prework, .recur] {
                    try g.encode(e, x: m.xn, out: m.h, rows: 1, nacc: 1, accumulate: true, only: p)
                }
            case .conv, .none: break
            }
        }
    }
    var verify: [Int: (gpu: Double, wall: Double)] = [:]
    for B in [1, 2, 4, 8] { verify[B] = try time { try m.encodeVerify($0, rows: B, drafter: nil) } }
    let withState = try time { e in
        try m.encodeVerify(e, rows: 1, drafter: nil); try statePass(e)
    }
    let alone = try time(statePass)
    let st = (gpu: withState.gpu - verify[1]!.gpu, wall: withState.wall - verify[1]!.wall)
    var sim: [Int: (gpu: Double, wall: Double)] = [1: verify[1]!]
    for N in [2, 4, 8] {
        sim[N] = try time { e in
            try m.encodeVerify(e, rows: N, drafter: nil)
            for _ in 1..<N { try statePass(e) }
        }
    }
    // State bytes of one conversation at context c: KV per token (ring layers capped at their ring) and GDN buffers.
    var kvPerTok = 0.0, ringBytes: [(Double, Int)] = [], gdnBytes = 0
    for l in m.layers {
        switch l.mixer {
        case .attn(let at):
            let b = Double(at.kc.length + at.vt.length + at.ks.length + at.vs.length) / Double(at.ldkv)
            if at.ring > 0 { ringBytes.append((b, at.ring)) } else { kvPerTok += b }
        case .gdn(let g):
            gdnBytes += (g.state + g.hist + g.tk + g.tv + g.tg + g.tb).reduce(0) { $0 + $1.length }
        case .conv, .none: break
        }
    }
    func stateMB(_ c: Int) -> Double {
        (kvPerTok * Double(c) + ringBytes.reduce(0) { $0 + $1.0 * Double(min(c, $1.1)) } + Double(gdnBytes)) / 1e6
    }
    let floor = Double(m.weightBytes) / (measuredReadGBps * 1e6)
    print(
        String(
            format: "batch profile %@ (P = %d): weights %.2f GB, floor %.2f ms; state of one conversation: "
                + "%.1f MB at 512, %.1f MB at 4K, %.1f MB at P (GDN %.1f MB)",
            label, P, Double(m.weightBytes) / 1e9, floor, stateMB(512), stateMB(4096), stateMB(P),
            Double(gdnBytes) / 1e6))
    print(
        String(
            format: "batch profile %@: state(1) GPU %.3f ms, wall %.3f ms (alone: GPU %.3f, wall %.3f)", label,
            st.gpu, st.wall, alone.gpu, alone.wall))
    let v1 = verify[1]!
    for N in [1, 2, 4, 8] {
        let v = verify[N]!, s = sim[N]!
        let est = (gpu: v.gpu + Double(N - 1) * st.gpu, wall: v.wall + Double(N - 1) * st.wall)
        print(
            String(
                format: "batch profile %@ N=%d: verify GPU %.3f wall %.3f | est GPU %.3f wall %.3f (x%.2f / x%.2f)"
                    + " | sim GPU %.3f wall %.3f (x%.2f / x%.2f)",
                label, N, v.gpu, v.wall, est.gpu, est.wall, Double(N) * v1.gpu / est.gpu,
                Double(N) * v1.wall / est.wall, s.gpu, s.wall, Double(N) * v1.gpu / s.gpu, Double(N) * v1.wall / s.wall)
        )
    }
}
