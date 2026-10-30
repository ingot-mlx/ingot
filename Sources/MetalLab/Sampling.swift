import Foundation
import Metal

/// Sampling parameters. `temperature == 0`: greedy (argmax, exact and with the same tokens as without a drafter).
/// Penalties (vLLM/OpenAI semantics, no window) go first: repetition over prompt + generated tokens (logit > 0 ⇒
/// ÷r, else ×r), then frequency (−f per occurrence) and presence (−p once) over the generated tokens. With
/// temperature 0 they also apply (argmax of the penalized logits).
/// With temperature: temperature → top-k → min-p → top-p (equivalent to vLLM's temperature → min-p → top-k → top-p:
/// min-p and top-k commute) over the 64 best candidates of each row (zz_sample.metal), with speculative verification
/// by rejection.
public struct SamplingParams: Sendable, Equatable {
    public var temperature: Float
    public var topP: Float
    /// 0 = no top-k (the 64 candidates).
    public var topK: Int
    /// nil = random seed.
    public var seed: UInt64?
    /// 0 = off. Keeps the candidates with p ≥ minP · p_max (after temperature and top-k).
    public var minP: Float
    /// 1 = off.
    public var repetitionPenalty: Float
    /// 0 = off.
    public var presencePenalty: Float
    /// 0 = off.
    public var frequencyPenalty: Float
    public init(
        temperature: Float = 0, topP: Float = 1, topK: Int = 0, seed: UInt64? = nil, minP: Float = 0,
        repetitionPenalty: Float = 1, presencePenalty: Float = 0, frequencyPenalty: Float = 0
    ) {
        self.temperature = temperature; self.topP = topP; self.topK = topK; self.seed = seed; self.minP = minP
        self.repetitionPenalty = repetitionPenalty; self.presencePenalty = presencePenalty
        self.frequencyPenalty = frequencyPenalty
    }
    public var greedy: Bool { temperature <= 0 }
    var hasPenalties: Bool { repetitionPenalty != 1 || presencePenalty != 0 || frequencyPenalty != 0 }
    /// What the chained round samples with: nil = greedy by today's path (top1, spec_accept, tree). Greedy with
    /// penalties goes through the sampling path with top-k 1 (argmax of the penalized logits, ties by lower id like
    /// tk_better: the draft is accepted iff it is candidate 0, otherwise candidate 0 comes out).
    var chainSampling: SamplingParams? {
        if !greedy { return self }
        if !hasPenalties { return nil }
        return SamplingParams(
            temperature: 1, topP: 1, topK: 1, seed: seed, repetitionPenalty: repetitionPenalty,
            presencePenalty: presencePenalty, frequencyPenalty: frequencyPenalty)
    }
    public static let candidates = 64
}

/// SplitMix64: reproducible uniforms from the seed.
public struct SplitMix64: RandomNumberGenerator {
    var s: UInt64
    public init(seed: UInt64) { s = seed }
    public mutating func next() -> UInt64 {
        s &+= 0x9E37_79B9_7F4A_7C15
        var z = s
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
    /// Uniform in [0, 1) with 24 bits (what fits exactly in a float).
    public mutating func uniform() -> Float { Float(next() >> 40) / Float(1 << 24) }
}

/// Kernel parameters (SampleParams of zz_sample.metal).
struct SampleKernelParams {
    var temp: Float, topP: Float, topK: UInt32, V: UInt32
    var minP: Float, rep: Float, pres: Float, freq: Float
    init(_ s: SamplingParams, vocab: Int) {
        temp = s.temperature; topP = s.topP; topK = UInt32(max(0, s.topK)); V = UInt32(vocab)
        minP = s.minP; rep = s.repetitionPenalty; pres = s.presencePenalty; freq = s.frequencyPenalty
    }
}

/// The same as `sample_dist` + `sample_pick` of zz_sample.metal, on the CPU (first token, after prefill).
public func sampleCandidates(
    ids: [UInt32], logits: [Float], max M: Float, z Z: Float, params p: SamplingParams,
    u: Float
) -> UInt32 {
    let it = 1 / p.temperature
    var n = ids.firstIndex(of: 0xffff_ffff) ?? ids.count
    var tot = Z
    if p.topK > 0 && p.topK < n { n = p.topK }
    if p.topK > 0 { tot = (0..<n).reduce(0) { $0 + exp((logits[$1] - M) * it) } }
    var pr = (0..<n).map { exp((logits[$0] - M) * it) / tot }
    if p.minP > 0 {
        let thr = p.minP * pr[0]
        n = max(pr.firstIndex { $0 < thr } ?? n, 1)
        let s = pr.prefix(n).reduce(0, +)
        pr = pr.prefix(n).map { $0 / s }
    }
    if p.topP < 1 {
        var c: Float = 0, k = n
        for j in 0..<n { c += pr[j]; if c >= p.topP { k = j + 1; break } }
        n = max(k, 1)
        pr = Array(pr.prefix(n))
    }
    let s = pr.reduce(0, +)
    var acc: Float = 0
    for j in 0..<n {
        acc += pr[j] / s
        if acc > u { return ids[j] }
    }
    return ids[n - 1]
}

/// The 64 candidates of each row (sorted, ties by lower id), its max and Σ exp((x − max)/T) over the whole row, in two
/// dispatches (zz_sample.metal): `topk64_part` (one threadgroup per chunk of the vocabulary) and `topk64_merge` (one
/// simdgroup per row).
enum TopK64 {
    /// Chunks per row (SPG in zz_sample.metal).
    static let chunks = 32
    /// Bytes of the partial lists for `rows` rows.
    static func scratchLength(rows: Int) -> Int { rows * chunks * (SamplingParams.candidates * 8 + 8) }

    /// `params`: one for all rows or one per row (`topk64_*_each`); the candidates go to `out`.
    static func encode(
        _ e: MTLComputeCommandEncoder, ctx: MetalContext, x: MTLBuffer, xOffset: Int, rows: Int,
        params: [SampleKernelParams], out: (idx: MTLBuffer, val: MTLBuffer, z: MTLBuffer), scratch: MTLBuffer
    ) throws {
        let each = params.count > 1
        precondition(rows >= 1 && (params.count == 1 || params.count == rows))
        precondition(scratch.length >= scratchLength(rows: rows))
        let K = SamplingParams.candidates, list = rows * chunks * K * 4
        var kp = params
        let plen = kp.count * MemoryLayout<SampleKernelParams>.stride
        e.setComputePipelineState(try ctx.pipeline(each ? "topk64_part_each" : "topk64_part"))
        e.setBuffer(x, offset: xOffset, index: 0)
        e.setBuffer(scratch, offset: 0, index: 1)
        e.setBuffer(scratch, offset: list, index: 2)
        e.setBuffer(scratch, offset: 2 * list, index: 3)
        e.setBytes(&kp, length: plen, index: 4)
        e.dispatchThreadgroups(
            MTLSize(width: chunks, height: rows, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        e.setComputePipelineState(try ctx.pipeline(each ? "topk64_merge_each" : "topk64_merge"))
        e.setBuffer(scratch, offset: 0, index: 0)
        e.setBuffer(scratch, offset: list, index: 1)
        e.setBuffer(scratch, offset: 2 * list, index: 2)
        e.setBuffer(out.idx, offset: 0, index: 3)
        e.setBuffer(out.val, offset: 0, index: 4)
        e.setBuffer(out.z, offset: 0, index: 5)
        e.setBytes(&kp, length: plen, index: 6)
        e.dispatchThreadgroups(
            MTLSize(width: rows, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    }
}

extension HybridModel {
    /// Sampling buffers (candidates for up to 16 rows, and the partial lists of `TopK64`) and, only once a request
    /// uses penalties, their counters.
    final class SampleBuffers {
        let idx, val, z, part: MTLBuffer
        let ctx: MetalContext
        /// [V]: occurrences of each token among the generated ones; bit 31 = in the prompt (only with repetition).
        private(set) var cnt: MTLBuffer?
        /// [1 + V]: [0] = n, [1...n] = the tokens with cnt ≠ 0 (no duplicates).
        private(set) var touched: MTLBuffer?
        /// The counters hold the prompt of the current request and the first token is not counted yet
        /// (`preparePenalties` → `decode`).
        var armed = false
        init(_ ctx: MetalContext) {
            self.ctx = ctx
            idx = ctx.buffer(length: 16 * SamplingParams.candidates * 4)
            val = ctx.buffer(length: 16 * SamplingParams.candidates * 4)
            z = ctx.buffer(length: 16 * 2 * 4)
            part = ctx.buffer(length: TopK64.scratchLength(rows: 16))
        }

        /// Clears the counters (allocated the first time) and marks the distinct prompt tokens if `repetition`.
        func reset(prompt: [UInt32], vocab: Int, repetition: Bool) {
            if cnt == nil {
                cnt = ctx.buffer(length: vocab * 4)
                touched = ctx.buffer(length: (vocab + 1) * 4)
            }
            let c = cnt!.contents().bindMemory(to: UInt32.self, capacity: vocab)
            let t = touched!.contents().bindMemory(to: UInt32.self, capacity: vocab + 1)
            memset(c, 0, vocab * 4)
            t[0] = 0
            if repetition {
                for id in prompt where Int(id) < vocab && c[Int(id)] == 0 {
                    c[Int(id)] = 1 << 31
                    t[0] += 1
                    t[Int(t[0])] = id
                }
            }
            armed = true
        }

        /// Counts a generated token on the CPU (the first one, chosen after the prefill); the same as count_emitted.
        func count(_ id: UInt32, vocab: Int) {
            guard let cnt, let touched, Int(id) < vocab else { return }
            let c = cnt.contents().bindMemory(to: UInt32.self, capacity: vocab)
            let t = touched.contents().bindMemory(to: UInt32.self, capacity: vocab + 1)
            if c[Int(id)] == 0 {
                t[0] += 1
                t[Int(t[0])] = id
            }
            c[Int(id)] += 1
        }
    }

    /// Prepares the penalty counters for a request (the whole prompt, also its cached part); without penalties, only
    /// disarms them.
    public func preparePenalties(prompt: [UInt32], params: SamplingParams) {
        guard params.hasPenalties else { sampleBufs.armed = false; return }
        sampleBufs.reset(prompt: prompt, vocab: vocab, repetition: params.repetitionPenalty != 1)
    }

    /// Penalizes in place the first `rows` rows of `logits` from `firstRow` (penalize_rows): row i with the counters
    /// plus the drafts `tok[1...i]`.
    func encodePenalties(_ e: MTLComputeCommandEncoder, rows: Int, params: SamplingParams, firstRow: Int = 0) throws {
        var kp = SampleKernelParams(params, vocab: vocab)
        e.setComputePipelineState(try ctx.pipeline("penalize_rows"))
        e.setBuffer(logits, offset: firstRow * vocab * 4, index: 0)
        e.setBuffer(sampleBufs.cnt, offset: 0, index: 1)
        e.setBuffer(sampleBufs.touched, offset: 0, index: 2)
        e.setBuffer(tok, offset: 0, index: 3)
        e.setBytes(&kp, length: MemoryLayout<SampleKernelParams>.size, index: 4)
        e.dispatchThreadgroups(
            MTLSize(width: rows, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    /// After the accept kernel: adds the tokens emitted by the round (its record) to the counters (count_emitted).
    func encodeCountEmitted(_ e: MTLComputeCommandEncoder, state: MTLBuffer, rec: MTLBuffer, ring: Int) throws {
        var rg = UInt32(ring), V = UInt32(vocab)
        e.setComputePipelineState(try ctx.pipeline("count_emitted"))
        e.setBuffer(state, offset: 0, index: 0)
        e.setBuffer(rec, offset: 0, index: 1)
        e.setBuffer(sampleBufs.cnt, offset: 0, index: 2)
        e.setBuffer(sampleBufs.touched, offset: 0, index: 3)
        e.setBytes(&rg, length: 4, index: 4)
        e.setBytes(&V, length: 4, index: 5)
        e.dispatchThreads(
            MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
    }

    /// Start of a chained decode with penalties: counts the first token (the counters must come from
    /// `preparePenalties` for this request). Without penalties, nothing.
    func startPenalties(first: UInt32, params: SamplingParams) {
        guard params.hasPenalties else { return }
        precondition(sampleBufs.armed, "penalties without preparePenalties(prompt:params:) for this request")
        sampleBufs.armed = false
        sampleBufs.count(first, vocab: vocab)
    }

    /// Candidates of the first `rows` rows of `logits` (top-64, max and normalizer with the temperature).
    func encodeTopK(_ e: MTLComputeCommandEncoder, rows: Int, params: SamplingParams, firstRow: Int = 0) throws {
        try TopK64.encode(
            e, ctx: ctx, x: logits, xOffset: firstRow * vocab * 4, rows: rows,
            params: [SampleKernelParams(params, vocab: vocab)], out: (sampleBufs.idx, sampleBufs.val, sampleBufs.z),
            scratch: sampleBufs.part)
    }

    /// Next token from row `row` of `logits` (the last of the prompt): argmax with temperature 0 (already in
    /// top1) or sampled. With penalties (counters from `preparePenalties`), the row is penalized in place first.
    public func sampleFromLogits(argmax: UInt32, row: Int = 0, params: SamplingParams, rng: inout SplitMix64) throws
        -> UInt32
    {
        guard let params = params.chainSampling else { return argmax }
        if params.hasPenalties { precondition(sampleBufs.armed) }
        try ctx.run { e in
            if params.hasPenalties { try encodePenalties(e, rows: 1, params: params, firstRow: row) }
            try encodeTopK(e, rows: 1, params: params, firstRow: row)
        }
        let K = SamplingParams.candidates
        let ids = Array(
            UnsafeBufferPointer(start: sampleBufs.idx.contents().bindMemory(to: UInt32.self, capacity: K), count: K))
        let vals = Array(
            UnsafeBufferPointer(start: sampleBufs.val.contents().bindMemory(to: Float.self, capacity: K), count: K))
        let z = sampleBufs.z.contents().bindMemory(to: Float.self, capacity: 2)
        return sampleCandidates(ids: ids, logits: vals, max: z[0], z: z[1], params: params, u: rng.uniform())
    }
}
