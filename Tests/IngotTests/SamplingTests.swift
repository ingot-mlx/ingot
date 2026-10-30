import Foundation
import Metal
import Testing

@testable import Ingot

/// GPU sampling (zz_sample.metal): candidates against the CPU and distribution of the rejection verification.
@Suite struct SamplingTests {
    static let V = 3000, R = 8, K = SamplingParams.candidates

    /// The 64 candidates of `rows` rows of `x` (TopK64, the path of the engine).
    static func topK(
        _ ctx: MetalContext, _ e: MTLComputeCommandEncoder, _ x: MTLBuffer, rows: Int, _ kp: SampleKernelParams,
        _ idx: MTLBuffer, _ val: MTLBuffer, _ z: MTLBuffer
    ) throws {
        try TopK64.encode(
            e, ctx: ctx, x: x, xOffset: 0, rows: rows, params: [kp], out: (idx, val, z),
            scratch: ctx.buffer(length: TopK64.scratchLength(rows: rows)))
    }

    /// Synthetic logits: a few dominant tokens per row and a tail.
    static func logits(seed: UInt64) -> [Float] {
        var g = SplitMix64(seed: seed)
        var x = [Float](repeating: 0, count: R * V)
        for r in 0..<R {
            for i in 0..<V { x[r * V + i] = g.uniform() * 4 - 2 }
            for j in 0..<6 { x[r * V + (r * 37 + j * 101) % V] = 6 - Float(j) * 0.6 }
        }
        return x
    }

    /// Exact filtered distribution on the CPU (temperature → top-k → min-p → top-p) over the whole vocabulary.
    static func dist(_ row: ArraySlice<Float>, _ p: SamplingParams) -> [Int: Double] {
        let it = 1 / Double(p.temperature)
        let order = row.indices.sorted { row[$0] > row[$1] || (row[$0] == row[$1] && $0 < $1) }.map {
            $0 - row.startIndex
        }
        let mx = Double(row[order[0] + row.startIndex])
        var pr = order.map { exp((Double(row[$0 + row.startIndex]) - mx) * it) }
        var n = order.count
        if p.topK > 0 { n = min(n, p.topK) }
        let tot = pr.prefix(n).reduce(0, +)
        pr = pr.map { $0 / tot }
        if p.minP > 0 {
            let thr = Double(p.minP) * pr[0]
            n = max(pr.prefix(n).firstIndex { $0 < thr } ?? n, 1)
            let s = pr.prefix(n).reduce(0, +)
            pr = pr.map { $0 / s }
        }
        if p.topP < 1 {
            var c = 0.0
            for j in 0..<n { c += pr[j]; if c >= Double(p.topP) { n = j + 1; break } }
        }
        let s = pr.prefix(n).reduce(0, +)
        var d: [Int: Double] = [:]
        for j in 0..<n { d[order[j]] = pr[j] / s }
        return d
    }

    /// With top-k and top-p; with min-p and no top-k (the min-p set fits in the 64 candidates: exact).
    @Test(arguments: [
        SamplingParams(temperature: 0.8, topP: 0.9, topK: 20),
        SamplingParams(temperature: 0.8, topP: 0.98, topK: 0, minP: 0.03),
    ])
    func topkAndAcceptanceMatchTargetDistribution(params: SamplingParams) throws {
        let ctx = try MetalContext()
        let x = Self.logits(seed: 3)
        var kp = SampleKernelParams(params, vocab: Self.V)
        let lb = ctx.buffer(x), idx = ctx.buffer(length: Self.R * Self.K * 4),
            val = ctx.buffer(length: Self.R * Self.K * 4)
        let z = ctx.buffer(length: Self.R * 8)
        try ctx.run { e in
            try Self.topK(ctx, e, lb, rows: Self.R, kp, idx, val, z)
        }
        let ip = idx.contents().bindMemory(to: UInt32.self, capacity: Self.R * Self.K)
        let zp = z.contents().bindMemory(to: Float.self, capacity: Self.R * 2)
        for r in 0..<Self.R {
            let row = x[(r * Self.V)..<((r + 1) * Self.V)]
            let order = row.indices.sorted { row[$0] > row[$1] || (row[$0] == row[$1] && $0 < $1) }.map {
                UInt32($0 - r * Self.V)
            }
            #expect((0..<Self.K).map { ip[r * Self.K + $0] } == Array(order.prefix(Self.K)))
            let mx = row.max()!
            let Z = row.reduce(0.0) { $0 + exp(Double($1 - mx) / 0.8) }
            #expect(abs(Double(zp[2 * r]) - Double(mx)) < 1e-6)
            #expect(abs(Double(zp[2 * r + 1]) / Z - 1) < 1e-4)
        }
        // Rejection verification: fixed drafts (the 1st likely, the 2nd unlikely, the rest random).
        let n = 20000
        let d0 = Self.dist(x[0..<Self.V], params), d1 = Self.dist(x[Self.V..<(2 * Self.V)], params)
        let draft1 = UInt32(d0.max { $0.value < $1.value }!.key)
        let draft2 = UInt32(d1.sorted { $0.value > $1.value }[3].key)
        var vt: [UInt32] = [0, draft1, draft2, 5, 6, 7, 8, 9]
        let vtok = ctx.buffer(vt), dtok = ctx.buffer(length: 32), st = ctx.buffer(length: 32)
        memset(st.contents(), 0, 32)
        var g = SplitMix64(seed: 11)
        let rnd = ctx.buffer((0..<(n * 32)).map { _ in g.uniform() })
        let rec = ctx.buffer(length: n * 96)
        var ring = UInt32(n)
        try ctx.run { e in
            e.setComputePipelineState(try ctx.pipeline("spec_accept_sample"))
            for (i, b) in [st, vtok, idx, val, z, dtok, rec, rnd].enumerated() { e.setBuffer(b, offset: 0, index: i) }
            e.setBytes(&kp, length: MemoryLayout<SampleKernelParams>.size, index: 8)
            e.setBytes(&ring, length: 4, index: 9)
            var rows = UInt32(8)
            e.setBytes(&rows, length: 4, index: 10)
            for _ in 0..<n {
                // vtok[0] changes (anchor): drafts 1..7 stay fixed.
                e.dispatchThreads(
                    MTLSize(width: 1, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
            }
        }
        _ = vt.removeFirst()
        let rp = rec.contents().bindMemory(to: UInt32.self, capacity: n * 24)
        var c0: [Int: Int] = [:], c1: [Int: Int] = [:], n1 = 0
        for t in 0..<n {
            let r = rp + t * 24
            c0[Int(r[1]), default: 0] += 1
            if r[1] == draft1 { c1[Int(r[2]), default: 0] += 1; n1 += 1 }
        }
        // First token ~ p0; second token, conditioned on the first being the draft, ~ p1.
        let e0 = Set(c0.keys).union(d0.keys).map { abs(Double(c0[$0] ?? 0) / Double(n) - (d0[$0] ?? 0)) }.max()!
        let e1 = Set(c1.keys).union(d1.keys).map { abs(Double(c1[$0] ?? 0) / Double(n1) - (d1[$0] ?? 0)) }.max()!
        #expect(Set(c0.keys).isSubset(of: Set(d0.keys)))
        #expect(e0 < 0.015, "first token: max error \(e0)")
        #expect(e1 < 0.02, "second token: max error \(e1) with \(n1) samples")
        // CPU (first token after the prefill) with the same rule.
        let ids = (0..<Self.K).map { ip[$0] }
        let vals = (0..<Self.K).map { val.contents().bindMemory(to: Float.self, capacity: Self.R * Self.K)[$0] }
        var cc: [Int: Int] = [:]
        var h = SplitMix64(seed: 5)
        for _ in 0..<n {
            cc[
                Int(sampleCandidates(ids: ids, logits: vals, max: zp[0], z: zp[1], params: params, u: h.uniform())),
                default: 0] += 1
        }
        let ec = Set(cc.keys).union(d0.keys).map { abs(Double(cc[$0] ?? 0) / Double(n) - (d0[$0] ?? 0)) }.max()!
        #expect(ec < 0.015, "CPU: max error \(ec)")
    }

    /// min-p with top_k = 0 and a flat row: the min-p set (all the vocabulary) does not fit in the 64 candidates and is
    /// truncated to them, like the top-p nucleus (documented limit of zz_sample.metal). GPU and CPU only pick among the
    /// 64 candidates, roughly uniformly.
    @Test func minPFlatRowTruncatesTo64Candidates() throws {
        let ctx = try MetalContext()
        let V = Self.V, K = Self.K
        var g0 = SplitMix64(seed: 31)
        let x = (0..<V).map { _ in g0.uniform() * 0.01 }
        let params = SamplingParams(temperature: 1, topP: 1, topK: 0, minP: 0.5)
        var kp = SampleKernelParams(params, vocab: V)
        let lb = ctx.buffer(x), idx = ctx.buffer(length: K * 4), val = ctx.buffer(length: K * 4),
            z = ctx.buffer(length: 8)
        let n = 4000
        let vtok = ctx.buffer([UInt32](repeating: 0, count: 16)), dtok = ctx.buffer(length: 64),
            st = ctx.buffer(length: 32)
        memset(st.contents(), 0, 32)
        var g = SplitMix64(seed: 32)
        let rnd = ctx.buffer((0..<(n * 32)).map { _ in g.uniform() })
        let rec = ctx.buffer(length: n * 96)
        var ring = UInt32(n), rows = UInt32(1)
        try ctx.run { e in
            try Self.topK(ctx, e, lb, rows: 1, kp, idx, val, z)
            e.setComputePipelineState(try ctx.pipeline("spec_accept_sample"))
            for (i, b) in [st, vtok, idx, val, z, dtok, rec, rnd].enumerated() { e.setBuffer(b, offset: 0, index: i) }
            e.setBytes(&kp, length: MemoryLayout<SampleKernelParams>.size, index: 8)
            e.setBytes(&ring, length: 4, index: 9)
            e.setBytes(&rows, length: 4, index: 10)
            for _ in 0..<n {
                e.dispatchThreads(
                    MTLSize(width: 1, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
            }
        }
        let ids = (0..<K).map { idx.contents().bindMemory(to: UInt32.self, capacity: K)[$0] }
        let vals = (0..<K).map { val.contents().bindMemory(to: Float.self, capacity: K)[$0] }
        let zp = z.contents().bindMemory(to: Float.self, capacity: 2)
        let rp = rec.contents().bindMemory(to: UInt32.self, capacity: n * 24)
        var gpu: [UInt32: Int] = [:], cpu: [UInt32: Int] = [:]
        var h = SplitMix64(seed: 33)
        for t in 0..<n {
            gpu[rp[t * 24 + 1], default: 0] += 1
            cpu[
                sampleCandidates(ids: ids, logits: vals, max: zp[0], z: zp[1], params: params, u: h.uniform()),
                default: 0] += 1
        }
        let cand = Set(ids)
        #expect(Set(gpu.keys).isSubset(of: cand) && Set(cpu.keys).isSubset(of: cand))
        #expect(gpu.count > 56 && cpu.count > 56, "distinct tokens: GPU \(gpu.count), CPU \(cpu.count)")
        #expect(gpu.values.max()! < 3 * n / K && cpu.values.max()! < 3 * n / K)
    }

    /// Round of 16 rows (those of the context copy): uniforms 2·i and 2·i + 1 up to row 15. Rows 0–11 and 13–14
    /// almost deterministic with the draft on its dominant token; row 12 spread out with an unlikely draft; row
    /// 15 spread out (extra token if all 15 accepted). The position-12 token (conditioned on reaching it) and the extra
    /// one must follow the exact distribution.
    @Test func sixteenRowRoundMatchesTargetDistribution() throws {
        let ctx = try MetalContext()
        let R = 16, V = Self.V, K = Self.K
        let x = Self.roundLogits(seed: 21, rows: R)
        var drafts = [UInt32](repeating: 0, count: R)
        let params = SamplingParams(temperature: 0.8, topP: 0.9, topK: 20)
        var kp = SampleKernelParams(params, vocab: V)
        let lb = ctx.buffer(x), idx = ctx.buffer(length: R * K * 4), val = ctx.buffer(length: R * K * 4),
            z = ctx.buffer(length: R * 8)
        try ctx.run { e in
            try Self.topK(ctx, e, lb, rows: R, kp, idx, val, z)
        }
        let d = (0..<R).map { Self.dist(x[($0 * V)..<(($0 + 1) * V)], params) }
        // Draft of row r (vtok[r + 1]): the dominant one, except in 12, the 4th most likely.
        for r in 0..<(R - 1) {
            let order = d[r].sorted { $0.value > $1.value }
            drafts[r] = UInt32(r == 12 ? order[3].key : order[0].key)
        }
        let vtok = ctx.buffer([0] + Array(drafts.prefix(R - 1)))
        try Self.expectRoundDistribution(ctx, vtok: vtok, candidates: (idx, val, z), &kp, rows: R, seed: 12, d)
    }

    /// Logits of the 16-row round tests: uniform in [−2, 2), with six spread-out large values in rows 12 and 15 and one
    /// dominant token (~all the weight) in the others.
    static func roundLogits(seed: UInt64, rows R: Int) -> [Float] {
        var g = SplitMix64(seed: seed)
        var x = [Float](repeating: 0, count: R * V)
        for r in 0..<R {
            for i in 0..<V { x[r * V + i] = g.uniform() * 4 - 2 }
            if r == 12 || r == 15 {
                for j in 0..<6 { x[r * V + (r * 37 + j * 101) % V] = 6 - Float(j) * 0.6 }
            } else {
                x[r * V + (r * 53 + 7) % V] = 14
            }
        }
        return x
    }

    /// Runs spec_accept_sample 20,000 times over a round of R rows (drafts in `vtok`, candidates of TopK64,
    /// uniforms drawn with `seed`) and checks that the position-12 token (conditioned on reaching it) and the extra one
    /// (all 15 accepted) follow the distributions `d`.
    static func expectRoundDistribution(
        _ ctx: MetalContext, vtok: MTLBuffer, candidates c: (idx: MTLBuffer, val: MTLBuffer, z: MTLBuffer),
        _ kp: inout SampleKernelParams, rows R: Int, seed: UInt64, _ d: [[Int: Double]]
    ) throws {
        let n = 20000
        let dtok = ctx.buffer(length: 64), st = ctx.buffer(length: 32)
        memset(st.contents(), 0, 32)
        var g = SplitMix64(seed: seed)
        let rnd = ctx.buffer((0..<(n * 32)).map { _ in g.uniform() })
        let rec = ctx.buffer(length: n * 96)
        var ring = UInt32(n), rows = UInt32(R)
        try ctx.run { e in
            e.setComputePipelineState(try ctx.pipeline("spec_accept_sample"))
            for (i, b) in [st, vtok, c.idx, c.val, c.z, dtok, rec, rnd].enumerated() {
                e.setBuffer(b, offset: 0, index: i)
            }
            e.setBytes(&kp, length: MemoryLayout<SampleKernelParams>.size, index: 8)
            e.setBytes(&ring, length: 4, index: 9)
            e.setBytes(&rows, length: 4, index: 10)
            for _ in 0..<n {
                e.dispatchThreads(
                    MTLSize(width: 1, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
            }
        }
        let rp = rec.contents().bindMemory(to: UInt32.self, capacity: n * 24)
        var c12: [Int: Int] = [:], c15: [Int: Int] = [:], n12 = 0, n15 = 0
        for t in 0..<n {
            let r = rp + t * 24
            let m = Int(r[0])
            if m >= 12 { c12[Int(r[13]), default: 0] += 1; n12 += 1 }
            if m == 15 { c15[Int(r[16]), default: 0] += 1; n15 += 1 }
        }
        func err(_ c: [Int: Int], _ k: Int, _ p: [Int: Double]) -> Double {
            Set(c.keys).union(p.keys).map { abs(Double(c[$0] ?? 0) / Double(k) - (p[$0] ?? 0)) }.max()!
        }
        #expect(n12 > 15000 && n15 > 1000, "reach row 12: \(n12), accept all 15: \(n15)")
        let e12 = err(c12, n12, d[12]), e15 = err(c15, n15, d[15])
        #expect(e12 < 0.015, "row 12: max error \(e12) with \(n12) samples")
        #expect(e15 < 0.04, "row 15 (extra): max error \(e15) with \(n15) samples")
    }

    // MARK: - Penalties

    static let penParams = SamplingParams(
        temperature: 0.8, topP: 0.9, topK: 20, repetitionPenalty: 1.3, presencePenalty: 1.5, frequencyPenalty: 0.5)

    /// CPU reference of penalize_rows (vLLM order: repetition, frequency, presence): row r of `x` follows the
    /// generated tokens `gen` plus `vtok[1...r]`; repetition also over `prompt`.
    static func penalizeRef(
        _ x: [Float], rows: Int, V: Int, prompt: Set<UInt32>, gen: [UInt32], vtok: [UInt32], _ p: SamplingParams
    ) -> [Float] {
        var y = x
        for r in 0..<rows {
            var c: [UInt32: Int] = [:]
            for t in gen { c[t, default: 0] += 1 }
            for k in stride(from: 1, through: r, by: 1) { c[vtok[k], default: 0] += 1 }
            for t in Set(c.keys).union(prompt) {
                let n = c[t] ?? 0
                var v = y[r * V + Int(t)]
                if n > 0 || prompt.contains(t) {
                    v = v > 0 ? v / p.repetitionPenalty : v * p.repetitionPenalty
                }
                if n > 0 {
                    v = v.addingProduct(-p.frequencyPenalty, Float(n))
                    v -= p.presencePenalty
                }
                y[r * V + Int(t)] = v
            }
        }
        return y
    }

    static func penalize(
        _ ctx: MetalContext, _ lb: MTLBuffer, rows: Int, _ sb: HybridModel.SampleBuffers, vtok: MTLBuffer,
        _ kp: inout SampleKernelParams, _ e: MTLComputeCommandEncoder
    ) throws {
        e.setComputePipelineState(try ctx.pipeline("penalize_rows"))
        for (i, b) in [lb, sb.cnt!, sb.touched!, vtok].enumerated() { e.setBuffer(b, offset: 0, index: i) }
        e.setBytes(&kp, length: MemoryLayout<SampleKernelParams>.size, index: 4)
        e.dispatchThreadgroups(
            MTLSize(width: rows, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    /// penalize_rows bit-identical to the CPU, with prompt, generated history and per-row drafts that repeat (also
    /// tokens already counted), negative logits included; count_emitted and the CPU count() leave the same counters.
    @Test func penalizeRowsMatchesCPUBitExact() throws {
        let ctx = try MetalContext()
        let R = 16, V = Self.V
        var g = SplitMix64(seed: 41)
        let x = (0..<(R * V)).map { _ in g.uniform() * 8 - 4 }
        let prompt: [UInt32] = [5, 9, 9, 100, 2999, 17, 5, 300]
        let gen: [UInt32] = [9, 42, 42, 1000, 42, 7]
        let vt: [UInt32] = [7, 42, 55, 55, 9, 2000, 5, 55, 3, 42, 3, 2999, 1234, 55, 9, 8]
        for p in [
            Self.penParams, SamplingParams(temperature: 0, presencePenalty: -0.7),
            SamplingParams(temperature: 1, repetitionPenalty: 0.8, frequencyPenalty: 1.1),
        ] {
            let sb = HybridModel.SampleBuffers(ctx)
            sb.reset(prompt: prompt, vocab: V, repetition: p.repetitionPenalty != 1)
            for t in gen { sb.count(t, vocab: V) }
            var kp = SampleKernelParams(p, vocab: V)
            let lb = ctx.buffer(x), vtok = ctx.buffer(vt)
            try ctx.run { e in try Self.penalize(ctx, lb, rows: R, sb, vtok: vtok, &kp, e) }
            let got = Array(
                UnsafeBufferPointer(start: lb.contents().bindMemory(to: Float.self, capacity: R * V), count: R * V))
            let ref = Self.penalizeRef(
                x, rows: R, V: V, prompt: p.repetitionPenalty != 1 ? Set(prompt) : [], gen: gen, vtok: vt, p)
            let diff = (0..<(R * V)).filter { got[$0].bitPattern != ref[$0].bitPattern }
            #expect(diff.isEmpty, "\(p): \(diff.count) logits differ, first \(diff.prefix(4))")
            #expect(got != x)
        }
        // count_emitted: the record of the last round (m = 3: drafts 42, 55, 55 and the token 2000) on the GPU, the
        // same tokens with count() on the CPU.
        let sb = HybridModel.SampleBuffers(ctx), cpu = HybridModel.SampleBuffers(ctx)
        for b in [sb, cpu] {
            b.reset(prompt: prompt, vocab: V, repetition: true)
            for t in gen { b.count(t, vocab: V) }
        }
        let ring = 4, RS = HybridModel.recStride
        var recv = [UInt32](repeating: 0, count: ring * RS)
        let emitted: [UInt32] = [42, 55, 55, 2000]
        recv[2 * RS] = 3
        for (i, t) in emitted.enumerated() { recv[2 * RS + 1 + i] = t }
        var stv = [UInt32](repeating: 0, count: 8)
        stv[5] = 7  // round 6 just accepted: record 6 % 4 = 2
        let st = ctx.buffer(stv), rec = ctx.buffer(recv)
        var rg = UInt32(ring), VV = UInt32(V)
        try ctx.run { e in
            e.setComputePipelineState(try ctx.pipeline("count_emitted"))
            for (i, b) in [st, rec, sb.cnt!, sb.touched!].enumerated() { e.setBuffer(b, offset: 0, index: i) }
            e.setBytes(&rg, length: 4, index: 4)
            e.setBytes(&VV, length: 4, index: 5)
            e.dispatchThreads(
                MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        }
        for t in emitted { cpu.count(t, vocab: V) }
        func words(_ b: MTLBuffer, _ n: Int) -> [UInt32] {
            Array(UnsafeBufferPointer(start: b.contents().bindMemory(to: UInt32.self, capacity: n), count: n))
        }
        #expect(words(sb.cnt!, V) == words(cpu.cnt!, V))
        #expect(words(sb.touched!, V + 1) == words(cpu.touched!, V + 1))
        let c = words(cpu.cnt!, V)
        #expect(c[42] == 4 && c[55] == 2 && c[9] == (1 << 31) + 1 && c[2000] == 1)
    }

    /// The 16-row round with penalties that depend on the row (each row's drafts count for the later ones): the
    /// penalized logits on the GPU are bit-identical to the CPU, and the position-12 token (conditioned on reaching
    /// it) and the extra one follow the exact distribution of the penalized rows.
    @Test func sixteenRowRoundWithPenaltiesMatchesTargetDistribution() throws {
        let ctx = try MetalContext()
        let R = 16, V = Self.V, K = Self.K
        let params = Self.penParams
        let x = Self.roundLogits(seed: 51, rows: R)
        // Row 12's best token was generated twice and its second is in the prompt; row 15's second was generated
        // once; row 3's dominant token is in the prompt.
        let prompt: [UInt32] = [545, 166, 11, 12]
        let gen: [UInt32] = [444, 444, 656, 30]
        // Drafts chosen row by row from the penalized distributions (row r's draft counts for rows > r).
        var vt = [UInt32](repeating: 0, count: R)
        var d: [[Int: Double]] = []
        for r in 0..<R {
            let y = Self.penalizeRef(x, rows: r + 1, V: V, prompt: Set(prompt), gen: gen, vtok: vt, params)
            d.append(Self.dist(y[(r * V)..<((r + 1) * V)], params))
            if r < R - 1 {
                let order = d[r].sorted { $0.value > $1.value }
                vt[r + 1] = UInt32(r == 12 ? order[3].key : order[0].key)
            }
        }
        let ref = Self.penalizeRef(x, rows: R, V: V, prompt: Set(prompt), gen: gen, vtok: vt, params)
        let sb = HybridModel.SampleBuffers(ctx)
        sb.reset(prompt: prompt, vocab: V, repetition: true)
        for t in gen { sb.count(t, vocab: V) }
        var kp = SampleKernelParams(params, vocab: V)
        let lb = ctx.buffer(x), idx = ctx.buffer(length: R * K * 4), val = ctx.buffer(length: R * K * 4),
            z = ctx.buffer(length: R * 8)
        let vtok = ctx.buffer(vt)
        try ctx.run { e in
            try Self.penalize(ctx, lb, rows: R, sb, vtok: vtok, &kp, e)
            try Self.topK(ctx, e, lb, rows: R, kp, idx, val, z)
        }
        let got = Array(
            UnsafeBufferPointer(start: lb.contents().bindMemory(to: Float.self, capacity: R * V), count: R * V))
        #expect(got == ref, "penalized logits differ from the CPU")
        // The penalties change row 12's ranking (the generated best token drops).
        #expect(d[12][444] ?? 0 < d[12][646] ?? 0)
        try Self.expectRoundDistribution(ctx, vtok: vtok, candidates: (idx, val, z), &kp, rows: R, seed: 52, d)
    }
}
