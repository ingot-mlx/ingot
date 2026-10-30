import Foundation
import LabCLI
import Metal
import MetalLab

/// `lab-dev rowscal`: generates the calibration file for the verification rows (`SpecRowsConfig`) of a
/// model on this Mac.
/// - Platt per depth: greedy generation with 8 fixed rows over the prompts in `--pdir`; for each round, the
///   confidence of each draft (`dr.conf`, written by `dselect`) and whether it was accepted, up to the first miss (the
///   conditional probability the policy uses); q = sigmoid(a·logit(conf) + b) per group 0, 1, 2, ≥ 3 (Newton).
/// - Cost: GPU ms of the queued round with 8 rows and trimmed to k = 2…8 on the GPU after
///   prefill + 128 tokens of each prompt in `--cost-prompts`, with isotonic regression over k.
/// Writes the JSON to `--out` (default metal-lab-rows.json): it goes in the model folder (takes precedence over the
/// family one) or in Sources/MetalLab/Calibration/rows-<family>.json.
func cmdRowsCal() throws {
    let model = try args.required("model", modelHelp)
    let n = args.int("n", 256)
    let (dir, cfg, ctx) = try openModel(model)
    // Prompts from real conversations (es/en/zh/ca, code, agent; with and without reasoning): the calibration set.
    // Qwen3.5/3.6 have a calibration set of their own; other families use their default set (`--prompts` also takes
    // .json paths with `ids`, e.g. real agent turns).
    let qwen = cfg.family == .qwen35 || cfg.family == .qwen35Moe
    if args.opts["pdir"] == nil, qwen { promptSet = cfg.isMoE ? "prompts-es" : "prompts-q27-es" }
    let pdir = promptSet
    let names: [String]
    if let p = args.opts["prompts"] {
        names = p.split(separator: ",").map(String.init)
    } else {
        names = try FileManager.default.contentsOfDirectory(atPath: repoRoot.path + "/ref/out/\(pdir)")
            .filter { $0.hasSuffix(".json") }.map { String($0.dropLast(5)) }.sorted()
    }
    let costPdir = qwen ? (cfg.isMoE ? "prompts" : "prompts-q27") : defaultPromptSet(cfg)
    let costNames = args.str("cost-prompts", "512,4K,16K,32K").split(separator: ",").map(String.init)
    let prompts = try names.map { ($0, try promptIds($0)) }
    let costPrompts = try costNames.map { ($0, try promptIds(repoRoot.path + "/ref/out/\(costPdir)/\($0).json")) }
    let maxCtx = ((prompts + costPrompts).map { $0.1.count }.max() ?? 0) + max(n, 128) + 64
    var opts = try engineOptions(cfg)
    opts.bench.specRowsFixed = 8
    let m = try HybridModel(
        ctx: ctx, store: try SafetensorsStore(directory: dir), config: cfg, maxCtx: maxCtx, options: opts)
    let dr = try requiredModelDrafter(ctx, dir, cfg, m)
    let family = SpecRowsConfig.family(cfg)

    // Samples (depth, conf, hit).
    var depth: [Int] = [], conf: [Double] = [], hit: [Double] = []
    var rounds = 0
    for (name, ids) in prompts {
        let gs = genStart(ids)
        let st = try m.generateSpec(
            prompt: ids, drafter: dr, maxTokens: n, stopAtEOS: false, genStart: gs,
            onRound: { acc in
                let c = dr.conf.contents().bindMemory(to: Float.self, capacity: 8)
                for t in 0..<7 {
                    depth.append(t); conf.append(Double(c[t])); hit.append(t < acc ? 1 : 0)
                    if t >= acc { break }
                }
            })
        rounds += st.rounds
        print(String(format: "%@: %d rounds, %.2f tok/round", name, st.rounds, st.tokensPerRound))
    }
    let platt = (0..<4).map { g -> (Double, Double) in
        let idx = depth.indices.filter { min(depth[$0], 3) == g }
        return fitPlatt(idx.map { conf[$0] }, idx.map { hit[$0] })
    }
    for (g, p) in platt.enumerated() {
        let k = depth.filter { min($0, 3) == g }.count
        print(
            String(format: "Platt depth %@: a = %.3f, b = %.3f (%d samples)", g < 3 ? "\(g)" : "≥3", p.0, p.1, k)
        )
    }

    // Cost by number of rows.
    var cost: [(ctx: Int, ms: [Double])] = []
    let tp = m.tok.contents().bindMemory(to: UInt32.self, capacity: 16)
    for (name, ids) in costPrompts {
        let st = try m.generatePipelined(
            prompt: ids, drafter: dr, maxTokens: 128, stopAtEOS: false, genStart: genStart(ids))
        let P = m.P, gen = Array(st.tokens.suffix(8))
        var ms: [Double] = []
        for k in 2...8 {
            let sb = ctx.buffer(length: 8 * 4)
            let sp = sb.contents().bindMemory(to: UInt32.self, capacity: 8)
            for i in 0..<8 { sp[i] = 0 }
            sp[0] = UInt32(P); sp[1] = UInt32(m.nacc); sp[6] = UInt32(k)
            var t: [Double] = []
            for _ in 0..<25 {
                let cb = ctx.queue.makeCommandBuffer()!, e = cb.makeComputeCommandEncoder()!
                try dr.encodeContext(e, rows: 8, nvalid: 8, pos0: P - 8)
                try dr.encodeBlock(e, P: P, embed: m.embed, lmHead: m.lmHead, vtok: m.tok)
                for i in 0..<8 { tp[i] = gen[i] }
                // With SSD-streamed experts (Flash-Next) verification cuts the encoder: continue with the one it
                // returns.
                try m.encodeVerify(e, rows: 8, drafter: dr, state: sb, posBound: P, rowLimit: true, cb: cb)
                    .endEncoding()
                cb.commit(); cb.waitUntilCompleted()
                if let err = cb.error { throw err }
                t.append((cb.gpuEndTime - cb.gpuStartTime) * 1e3)
            }
            ms.append(median(Array(t.dropFirst(5))))
        }
        let iso = isotonic(ms)
        print(
            "cost \(name) (ctx \(P)): "
                + zip(ms, iso).map { String(format: "%.2f→%.2f", $0, $1) }.joined(separator: " ") + " ms (k = 2…8)")
        cost.append((P, iso))
    }

    // Hand-written JSON: same as Calibration/rows-qwen3_5_moe.json (JSONSerialization writes 20.199999999999999).
    let date = ISO8601DateFormatter().string(from: Date()).prefix(10)
    let about =
        "lab-dev rowscal (\(date)) with \(dir.lastPathComponent): Platt per depth 0, 1, 2, >= 3 with "
        + "\(prompts.count) prompts from \(pdir) × \(n) tokens (\(rounds) rounds, \(depth.count) samples); cost with "
        + "\(costPdir)/\(costNames.joined(separator: ",")): ms per round with k = 2..8 rows trimmed on the GPU, with "
        + "isotonic regression."
    let plattText = platt.map { String(format: "[%.3f, %.3f]", $0.0, $0.1) }.joined(separator: ", ")
    let costText = cost.map { c in
        "    {\"ctx\": \(c.ctx), \"ms\": [" + c.ms.map { String(format: "%.2f", $0) }.joined(separator: ", ") + "]}"
    }.joined(separator: ",\n")
    let text = """
        {
          "family": "\(family)",
          "about": "\(about)",
          "platt": [\(plattText)],
          "cost": [
        \(costText)
          ],
          "kmin": 2,
          "kmax": 8
        }

        """
    let data = Data(text.utf8)
    _ = try SpecRowsConfig(json: data)  // the engine must be able to read it
    let out = args.str("out", "metal-lab-rows.json")
    try data.write(to: URL(fileURLWithPath: out))
    print(
        "written \(out) (family \(family)): to the model folder or to Sources/MetalLab/Calibration/rows-\(family).json"
    )
}

/// One-variable logistic regression (q = sigmoid(a·logit(c) + b)) by Newton, as in h11_rows.py, with step halving
/// while the log-loss does not drop: the v1 drafter's confidences reach 1 − 1e-6 (logit 13.8) and plain Newton from
/// (1, 0) diverges on them (a, b ~ 1e8).
func fitPlatt(_ c: [Double], _ y: [Double], iters: Int = 300) -> (Double, Double) {
    let x = c.map { v -> Double in
        let p = min(max(v, 1e-6), 1 - 1e-6); return log(p / (1 - p))
    }
    func loss(_ a: Double, _ b: Double) -> Double {
        x.indices.reduce(0) { s, i in
            let z = a * x[i] + b; return s + log1p(exp(-abs(z))) + max(z, 0) - y[i] * z
        }
    }
    var a = 1.0, b = 0.0
    for _ in 0..<iters {
        var ga = 0.0, gb = 0.0, haa = 1e-6, hab = 0.0, hbb = 1e-6
        for i in x.indices {
            let p = 1 / (1 + exp(-(a * x[i] + b))), h = p * (1 - p)
            ga += (p - y[i]) * x[i]; gb += p - y[i]
            haa += h * x[i] * x[i]; hab += h * x[i]; hbb += h
        }
        let det = haa * hbb - hab * hab
        guard det > 0 else { break }
        let da = (hbb * ga - hab * gb) / det, db = (haa * gb - hab * ga) / det
        let f0 = loss(a, b)
        var t = 1.0
        while t > 1e-8, loss(a - t * da, b - t * db) > f0 { t /= 2 }
        a -= t * da
        b -= t * db
    }
    return (a, b)
}

/// Increasing isotonic regression (PAV) with equal weights.
func isotonic(_ v: [Double]) -> [Double] {
    var blocks: [(sum: Double, n: Int)] = []
    for x in v {
        blocks.append((x, 1))
        while blocks.count > 1,
            blocks[blocks.count - 2].sum / Double(blocks[blocks.count - 2].n) > blocks.last!.sum
                / Double(blocks.last!.n)
        {
            let l = blocks.removeLast()
            blocks[blocks.count - 1].sum += l.sum
            blocks[blocks.count - 1].n += l.n
        }
    }
    return blocks.flatMap { [Double](repeating: $0.sum / Double($0.n), count: $0.n) }
}
