import Foundation
import Metal

/// How many rows each DFlash2 round verifies, decided **on the GPU** after the drafter (kernel
/// `spec_rows_choose`), from each draft's confidence and the measured cost per number of rows. The chained round
/// enqueues verification with the maximum rows (8) and the pieces whose cost depends on rows (routing and expert union
/// of the MoE, GDN recurrence) and acceptance read the chosen number from the round state (st[6]).
///
/// Policy ("chain-cost", simulated offline with drafter dumps that are no longer generated):
///   conf_t = softmax(s/1.5) of the draft chosen by the selector over its top-16 (written by `dselect`),
///   q_t    = sigmoid(a_g · logit(conf_t) + b_g), g = depth 0, 1, 2, ≥ 3 (Platt, calibrated on Heretic),
///   E(k)   = 1 + Σ_{i<k−1} Π_{t≤i} q_t, and the k in [2, 8] with the highest E(k) / cost(k) is chosen.
/// Exact: the decision depends only on the drafter, never on the big model (identical greedy; with sampling, the same
/// rejection verification over the k rows).
///
/// The calibration (Platt and cost per rows) is per model family and per machine: it lives in a file
/// (`SpecRowsConfig.load`), not in code. A family without a file always verifies all 8 rows.
public struct SpecRowsConfig: Sendable {
    /// Platt per depth group (a, b): q = sigmoid(a · logit(conf) + b).
    public var platt: [(Float, Float)]
    /// ms per round with k = 2…8 rows per context (tokens in the cache), sorted by context.
    public var cost: [(ctx: Int, ms: [Float])]
    /// Minimum and maximum rows (anchor included). With 2 the simulation gains one more point than with 3.
    public var kmin = 2, kmax = 8

    public init(platt: [(Float, Float)], cost: [(ctx: Int, ms: [Float])], kmin: Int = 2, kmax: Int = 8) {
        self.platt = platt
        self.cost = cost.sorted { $0.ctx < $1.ctx }
        self.kmin = kmin
        self.kmax = kmax
    }

    /// File format: {"platt": [[a0, b0], …, [a3, b3]], "cost": [{"ctx": n, "ms": [ms2, …, ms8]}, …],
    /// "kmin": 2, "kmax": 8}; "family" and "about" are informational.
    struct FileCost: Decodable { let ctx: Int; let ms: [Float] }
    struct File: Decodable {
        let platt: [[Float]]
        let cost: [FileCost]
        let kmin: Int?, kmax: Int?
    }

    public init(json data: Data) throws {
        let f = try JSONDecoder().decode(File.self, from: data)
        guard f.platt.count == 4, f.platt.allSatisfy({ $0.count == 2 }), !f.cost.isEmpty,
            f.cost.allSatisfy({ $0.ms.count == 7 })
        else {
            throw SafetensorsError.format("row calibration: 4 Platt pairs and costs for 2…8 rows are required")
        }
        let kmin = f.kmin ?? 2, kmax = f.kmax ?? 8
        guard 2 <= kmin, kmin <= kmax, kmax <= 8 else {
            throw SafetensorsError.format("row calibration: kmin/kmax")
        }
        self.init(platt: f.platt.map { ($0[0], $0[1]) }, cost: f.cost.map { ($0.ctx, $0.ms) }, kmin: kmin, kmax: kmax)
    }

    /// Calibration family (ModelFamily.calibrationName: qwen3_5_moe for Qwen3.6 Heretic and Ornith; qwen3_5
    /// for the dense 27B, which has no file); with an unknown family, the `model_type`.
    public static func family(_ cfg: ModelConfig) -> String {
        cfg.family?.calibrationName ?? cfg.modelType
    }

    /// The calibration of a model: `metal-lab-rows.json` in its folder if it exists (to recalibrate a specific model)
    /// otherwise the one for its family that ships with the engine (`Calibration/rows-<family>.json`). nil if there is none.
    public static func load(modelDirectory dir: URL, config cfg: ModelConfig) throws -> SpecRowsConfig? {
        let own = dir.appendingPathComponent("metal-lab-rows.json")
        if FileManager.default.fileExists(atPath: own.path) { return try SpecRowsConfig(json: Data(contentsOf: own)) }
        return try bundled(family: family(cfg))
    }

    /// The calibration of a family that ships with the engine, or nil.
    public static func bundled(family: String) throws -> SpecRowsConfig? {
        guard
            let url = Bundle.module.url(
                forResource: "rows-\(family)", withExtension: "json", subdirectory: "Calibration")
        else { return nil }
        return try SpecRowsConfig(json: Data(contentsOf: url))
    }

    /// Cost table (k = 2…8) for `ctx` tokens: interpolated in log(ctx) between the measurements (outside, the nearest).
    public func costTable(ctx: Int) -> [Float] {
        let x = log(Double(max(ctx, 1)))
        guard let first = cost.first, let last = cost.last else { return [Float](repeating: 1, count: 7) }
        if Double(ctx) <= Double(first.ctx) { return first.ms }
        if Double(ctx) >= Double(last.ctx) { return last.ms }
        for i in 0..<(cost.count - 1) where ctx <= cost[i + 1].ctx {
            let x0 = log(Double(cost[i].ctx)), x1 = log(Double(cost[i + 1].ctx)), w = Float((x - x0) / (x1 - x0))
            return (0..<7).map { cost[i].ms[$0] * (1 - w) + cost[i + 1].ms[$0] * w }
        }
        return last.ms
    }

    /// CPU reference of `spec_rows_choose` (used by the tests; same arithmetic in Float).
    public func chooseCPU(conf: [Float], ctx: Int) -> Int {
        let c = costTable(ctx: ctx)
        var acc: Float = 1, lp: Float = 1, bv: Float = -1, best = kmax
        for t in 0..<(kmax - 1) {
            let (a, b) = platt[min(t, 3)]
            let x = min(max(conf[t], 1e-6), 1 - 1e-6)
            let q = 1 / (1 + exp(-(a * log(x / (1 - x)) + b)))
            lp *= q; acc += lp
            let k = t + 2
            if k >= kmin { let v = acc / c[k - 2]; if v > bv { bv = v; best = k } }
        }
        return best
    }

    /// RowsParams words for setBytes (the fixed cost table for the context).
    public func words(ctx: Int) -> [UInt32] {
        var w = [UInt32](repeating: 0, count: 27)
        for g in 0..<4 { w[g] = platt[g].0.bitPattern; w[4 + g] = platt[g].1.bitPattern }
        let c = costTable(ctx: ctx)
        for k in 0...16 { w[8 + k] = Float(k >= 2 && k <= 8 ? c[k - 2] : 1e9).bitPattern }
        w[25] = UInt32(kmin); w[26] = UInt32(kmax)
        return w
    }
}

extension HybridModel {
    /// Enqueues `spec_rows_choose`: writes this round's rows into st[6] (after `dselect`, before verification).
    func encodeRowsChoose(
        _ e: MTLComputeCommandEncoder, drafter dr: DFlash2Drafter, state: MTLBuffer, ctx n: Int,
        config: SpecRowsConfig
    ) throws {
        var w = config.words(ctx: n)
        e.setComputePipelineState(try ctx.pipeline("spec_rows_choose"))
        e.setBuffer(state, offset: 0, index: 0)
        e.setBuffer(dr.conf, offset: 0, index: 1)
        e.setBytes(&w, length: w.count * 4, index: 2)
        e.dispatchThreads(
            MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
    }
}

extension HybridModel {
    /// Exactness test (lab-dev rowscheck): the 8-row verification with tokens `toks` on the current state, full and
    /// trimmed to k rows on the GPU (st[6] = k), and another full one with rows ≥ k replaced by
    /// `alt`. Returns whether the logits of rows < k come out bit-exact in all three cases. It does not change the state (the
    /// GDN goes back to its ping-pong after each pass; the KV of positions P… is rewritten in the next round).
    public func rowLimitBitCheck(toks: [UInt32], alt: [UInt32], k: Int) throws -> (
        masked: Bool, laterRows: Bool, maxDiff: Float
    ) {
        precondition(toks.count == 8 && alt.count == 8 && (1...8).contains(k))
        let sb = ctx.buffer(length: 32)
        let sp = sb.contents().bindMemory(to: UInt32.self, capacity: 8)
        for i in 0..<8 { sp[i] = 0 }
        sp[0] = UInt32(P); sp[1] = UInt32(nacc); sp[6] = UInt32(k)
        let tp = tok.contents().bindMemory(to: UInt32.self, capacity: 16)
        let n = k * vocab
        func pass(_ t: [UInt32], limit: Bool) throws -> [Float] {
            for i in 0..<8 { tp[i] = t[i] }
            // With SSD experts (Flash-Next), verification needs the command buffer and cuts the encoder.
            guard let cb = ctx.queue.makeCommandBuffer(), let e = cb.makeComputeCommandEncoder() else {
                throw SafetensorsError.io("cb")
            }
            try encodeVerify(e, rows: 8, drafter: nil, state: sb, posBound: P, rowLimit: limit, cb: cb).endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            if let err = cb.error { throw err }
            flipGDN()
            return Array(
                UnsafeBufferPointer(start: logits.contents().bindMemory(to: Float.self, capacity: n), count: n))
        }
        let full = try pass(toks, limit: false)
        let masked = try pass(toks, limit: true)
        var mixed = toks
        for i in k..<8 { mixed[i] = alt[i] }
        let later = try pass(mixed, limit: false)
        var md: Float = 0
        for i in 0..<n { md = max(md, abs(full[i] - masked[i]), abs(full[i] - later[i])) }
        let bits = { (a: [Float], b: [Float]) in zip(a, b).allSatisfy { $0.bitPattern == $1.bitPattern } }
        return (bits(full, masked), bits(full, later), md)
    }
}
