import Foundation
import LabBench
import LabCLI
import Metal
import MetalLab

// lab-dev kl: KL of the lossy prefill against the exact one (--ref exact), against mlx-lm (--ref mlx, mlx-f32,
// mlx-prefix).

/// lab-dev kl --prompts 512,4K,16K,32K --n 64 --int8 1: KL(exact path || W8A8 prefill MoE) at the last prompt position
/// and over n decode steps forced with the tokens of the exact path (decoding is exact in both; only what the prefill
/// left in the KV cache and in the GDN state differs). Plus the top-1 agreement.
@MainActor func cmdKL() throws {
    let model = try args.required("model", modelHelp)
    let names = args.str("prompts", "512,4K").split(separator: ",").map(String.init)
    let n = args.int("n", 64), mode = args.int("int8", 1), qk8 = args.int("qk8", 0) > 0
    let (dir, cfg, ctx) = try openModel(model)
    let prompts = try names.map { ($0, try promptIds($0)) }
    let maxCtx = (prompts.map { $0.1.count }.max() ?? 0) + n + 64
    let m = try HybridModel(
        ctx: ctx, store: try SafetensorsStore(directory: dir), config: cfg, maxCtx: maxCtx,
        options: try engineOptions(cfg))
    let V = m.vocab
    func logp() -> [Double] { logProbs(m) }
    func run(_ ids: [UInt32], int8: Int, force: [UInt32]?) throws -> (lp: [[Double]], toks: [UInt32], prefill: Double) {
        let lossy = int8 < 0
        m.options.prefillInt8 = lossy ? mode > 0 : int8 > 0
        m.options.prefillQK8 = lossy ? qk8 : false
        m.reset()
        let (first, sec) = try m.prefill(ids, drafter: nil)
        var lps = [logp()], toks = [first]
        for i in 0..<n {
            let t = force?[i] ?? toks[i]
            toks.append(try m.stepLogits(t))
            lps.append(logp())
        }
        return (lps, toks, sec)
    }
    func compare(_ ref: [[Double]], _ q: [[Double]]) -> (kls: [Double], agree: Int) {
        var kls: [Double] = [], agree = 0
        for (a, b) in zip(ref, q) {
            var kl = 0.0
            for i in 0..<V where a[i] > -30 { kl += exp(a[i]) * (a[i] - b[i]) }
            kls.append(kl)
            if argmaxIndex(a) == argmaxIndex(b) { agree += 1 }
        }
        return (kls, agree)
    }
    // --cfgs: several configurations with the exact references computed once (see klConfigs).
    if let cfgs = args.opts["cfgs"] {
        try klConfigs(cfgs, m: m, prompts: prompts, n: n, mode: mode, qk8: qk8) {
            _ = try run(Array(prompts[0].1.prefix(64)), int8: -1, force: nil)
        }
        return
    }
    print(
        "KL(exact || prefill \(mode > 0 ? "rotated W8A8" : "exact")\(qk8 ? " + Q·Kᵀ int8" : "")) "
            + "\(m.isMoE ? "of Qwen3.6 (dense GEMMs; exact experts)" : "of the dense GEMMs"); \(n) forced steps"
    )
    for (name, ids) in prompts {
        // int8: −1 = the lossy configuration (--int8 and --qk8).
        _ = try run(Array(ids.prefix(64)), int8: -1, force: nil)  // warm-up (per-row scales, pipelines)
        let ref = try run(ids, int8: 0, force: nil)
        let q = try run(ids, int8: -1, force: ref.toks)
        let (kls, agree) = compare(ref.lp, q.lp)
        let mean = kls.reduce(0, +) / Double(kls.count)
        print(
            String(
                format:
                    "%@ (%d tok): KL last position %.5f, mean over %d positions %.5f, median %.5f, "
                    + "max %.4f, top-1 equal %d/%d; exact read %.0f tok/s, W8A8 %.0f tok/s",
                name, ids.count, kls[0], kls.count, mean, kls.sorted()[kls.count / 2], kls.max()!, agree, kls.count,
                Double(ids.count) / ref.prefill, Double(ids.count) / q.prefill))
    }
}

/// lab-dev kl --ref mlx|mlx-f32 --model ~/Models/LFM2.5-24B-A2B-MLX-4bit-calib --prompts chat,512,4K --n 64:
/// KL(mlx-lm ‖ lab) at the last prompt position and over n steps forced with mlx-lm's greedy (logits from
/// ref/logits_ref.py in ref/out/<--gen>/<prompt>_logits.npy, default the model's genRefSet; mlx-lm's bf16 output),
/// plus the top-1 agreement. It serves models without a drafter, where `--check` cannot compare two exact lab paths.
/// `--ref mlx-f32`: against mlx-lm with fp32 activations (logits_ref.py --f32), the precision reference.
@MainActor func cmdKLMLX() throws {
    let model = try args.required("model", modelHelp)
    let names = args.str("prompts", "chat,512").split(separator: ",").map(String.init)
    let n = args.int("n", 64)
    let (dir, cfg, ctx) = try openModel(model)
    let gen = args.str("gen", genRefSet)
    let prompts = try names.map { ($0, try promptIds($0)) }
    let maxCtx = (prompts.map { $0.1.count }.max() ?? 0) + n + 64
    let m = try HybridModel(
        ctx: ctx, store: try SafetensorsStore(directory: dir), config: cfg, maxCtx: maxCtx,
        options: try engineOptions(cfg))
    let V = m.vocab
    func logp(_ p: UnsafePointer<Float>) -> [Double] {
        var mx = -Double.infinity
        for i in 0..<V { mx = max(mx, Double(p[i])) }
        var s = 0.0
        for i in 0..<V { s += exp(Double(p[i]) - mx) }
        let l = log(s) + mx
        return (0..<V).map { Double(p[$0]) - l }
    }
    for (name, ids) in prompts {
        let base = repoRoot.path + "/ref/out/\(gen)/\(name)"
        let ref = try NPYArray(
            contentsOf: URL(fileURLWithPath: base + "_logits\(args.opts["ref"] == "mlx-f32" ? "_f32" : "").npy"))
        let force = try jsonUInt32s(at: URL(fileURLWithPath: base + "_mlx.json"), key: "ids")
        let steps = min(n, force.count, ref.shape[0] - 1)
        precondition(ref.kind == .f4 && ref.shape[1] == V)
        m.reset()
        _ = try m.prefill(ids, drafter: nil)
        var labLP = [logp(m.logits.contents().bindMemory(to: Float.self, capacity: V))]
        for i in 0..<steps {
            _ = try m.stepLogits(force[i])
            labLP.append(logp(m.logits.contents().bindMemory(to: Float.self, capacity: V)))
        }
        var kls: [Double] = [], agree = 0, sig = 0.0, err = 0.0
        var misses: [String] = []  // positions with a different top-1 and mlx-lm's top1 − top2 margin there
        ref.data.withUnsafeBytes { raw in
            let rp = raw.bindMemory(to: Float.self)
            for (j, b) in labLP.enumerated() {
                let a = logp(rp.baseAddress! + j * V)
                // SNR of the logits (log-probabilities centred on their mean) in dB.
                let ma = a.reduce(0, +) / Double(V), mb = b.reduce(0, +) / Double(V)
                for i in 0..<V { let x = a[i] - ma, d = (b[i] - mb) - x; sig += x * x; err += d * d }
                var kl = 0.0
                for i in 0..<V where a[i] > -30 { kl += exp(a[i]) * (a[i] - b[i]) }
                kls.append(kl)
                let ta = argmaxIndex(a)
                if ta == argmaxIndex(b) {
                    agree += 1
                } else {
                    let second = a.indices.filter { $0 != ta }.map { a[$0] }.max() ?? a[ta]
                    misses.append(String(format: "%d (%.3f)", j, a[ta] - second))
                }
            }
        }
        let mean = kls.reduce(0, +) / Double(kls.count)
        print(
            String(
                format:
                    "%@ (%d tok): KL(mlx-lm ‖ lab) last prompt position %.5f, mean over %d positions %.5f, "
                    + "median %.5f, max %.4f, top-1 equal %d/%d, logits SNR %.1f dB",
                name, ids.count, kls[0], kls.count, mean, kls.sorted()[kls.count / 2], kls.max()!, agree, kls.count,
                10 * log10(sig / max(err, 1e-300))))
        if !misses.isEmpty {
            print("  top-1 differs at (mlx-lm top1 - top2 margin): " + misses.joined(separator: ", "))
        }
    }
}

/// lab-dev kl --ref mlx-prefix --prompts 4K: KL(mlx-lm fp32 ‖ lab prefill) at the last position of the prefixes of
/// ref/prefix_ref.py, next to that of mlx-lm bf16 (the reference for how mlx-lm runs), with the same fp32 reference.
@MainActor func cmdPFKL() throws {
    let model = try args.required("model", modelHelp)
    let name = args.str("prompts", "4K")
    let (dir, cfg, ctx) = try openModel(model)
    let gen = args.str("gen", genRefSet)
    let ids = try promptIds(name)
    let base = repoRoot.path + "/ref/out/\(gen)/\(name)"
    let lens = try jsonUInt32s(at: URL(fileURLWithPath: base + "_prefixes.json"), key: "lens").map(Int.init)
    let f32 = try NPYArray(contentsOf: URL(fileURLWithPath: base + "_prefix_f32.npy"))
    let b16 = try NPYArray(contentsOf: URL(fileURLWithPath: base + "_prefix_bf16.npy"))
    let m = try HybridModel(
        ctx: ctx, store: try SafetensorsStore(directory: dir), config: cfg, maxCtx: lens.max()! + 64,
        options: try engineOptions(cfg))
    let V = m.vocab
    func logp(_ p: UnsafePointer<Float>) -> [Double] {
        var mx = -Double.infinity
        for i in 0..<V { mx = max(mx, Double(p[i])) }
        var s = 0.0
        for i in 0..<V { s += exp(Double(p[i]) - mx) }
        let l = log(s) + mx
        return (0..<V).map { Double(p[$0]) - l }
    }
    func kl(_ a: [Double], _ b: [Double]) -> Double {
        var k = 0.0; for i in 0..<V where a[i] > -30 { k += exp(a[i]) * (a[i] - b[i]) }; return k
    }
    var kLab: [Double] = [], kB16: [Double] = [], tLab = 0, tB16 = 0
    try f32.data.withUnsafeBytes { rf in
        try b16.data.withUnsafeBytes { rb in
            for (j, L) in lens.enumerated() {
                m.reset()
                _ = try m.prefill(Array(ids.prefix(L)), drafter: nil)
                let lab = logp(m.logits.contents().bindMemory(to: Float.self, capacity: V))
                let t = logp(rf.bindMemory(to: Float.self).baseAddress! + j * V),
                    b = logp(rb.bindMemory(to: Float.self).baseAddress! + j * V)
                kLab.append(kl(t, lab)); kB16.append(kl(t, b))
                if argmaxIndex(t) == argmaxIndex(lab) { tLab += 1 }
                if argmaxIndex(t) == argmaxIndex(b) { tB16 += 1 }
            }
        }
    }
    func st(_ k: [Double]) -> String {
        String(
            format: "mean %.4f, median %.5f, max %.3f", k.reduce(0, +) / Double(k.count), k.sorted()[k.count / 2],
            k.max()!)
    }
    print("\(name), \(lens.count) prefixes, KL against mlx-lm fp32 at the last position:")
    print("  lab prefill:             \(st(kLab)); top-1 equal \(tLab)/\(lens.count)")
    print("  mlx-lm bf16:             \(st(kB16)); top-1 equal \(tB16)/\(lens.count)")
}

/// Log-probabilities of the model's last logits row.
private func logProbs(_ m: HybridModel) -> [Double] {
    let V = m.vocab
    let p = m.logits.contents().bindMemory(to: Float.self, capacity: V)
    var mx = -Double.infinity
    for i in 0..<V { mx = max(mx, Double(p[i])) }
    var s = 0.0
    for i in 0..<V { s += exp(Double(p[i]) - mx) }
    let l = log(s) + mx
    return (0..<V).map { Double(p[$0]) - l }
}

/// Sparse references of klConfigs (log-prob > −30), so the whole vocabulary is not stored per position.
private struct SparseRef: Codable { let idx: [Int32]; let val: [Float]; let top: Int; let margin: Double }

/// lab-dev kl --cfgs "a;b;...": several configurations with the exact references computed once. Each one is
/// tokens separated by '/': exact, noise (exact with the fp32 recurrence), L=0-9+12 (layers in int8), P=gp+go+ap+ao,
/// R=1024 (position from which it runs in int8; without R=, 0), Q=0|1 (Q·Kᵀ int8), QM=8192.
/// scanL / dropL: one per layer (only that layer in int8 / all but that one); base: --int8 and --qk8 as given. Each
/// token changes m.options (PrefillInt8Options), which the engine reads on every chunk. `warm` warms the model up.
@MainActor private func klConfigs(
    _ cfgs: String, m: HybridModel, prompts: [(String, [UInt32])], n: Int, mode: Int, qk8: Bool,
    warm: () throws -> Void
) throws {
    let V = m.vocab
    let nl = m.layers.count
    var list: [String] = []
    for c in cfgs.split(separator: ";").map(String.init) {
        if c.hasPrefix("scanL") {
            let rest = c.dropFirst(5); list += (0..<nl).map { "L=\($0)" + rest }
        } else if c.hasPrefix("dropL") {
            let rest = c.dropFirst(5)
            list += (0..<nl).map { i in
                "L=" + (0..<nl).filter { $0 != i }.map(String.init).joined(separator: "+") + rest
            }
        } else {
            list.append(c)
        }
    }
    func sparse(_ a: [Double]) -> SparseRef {
        var idx: [Int32] = [], val: [Float] = []
        for i in 0..<V where a[i] > -30 { idx.append(Int32(i)); val.append(Float(a[i])) }
        let top = argmaxIndex(a)
        let second = a.indices.filter { $0 != top }.max { a[$0] < a[$1] }!
        return SparseRef(idx: idx, val: val, top: top, margin: a[top] - a[second])
    }
    var badMargin = 0.0  // largest reference margin on a differing top-1 (criterion: < 0.5)
    func kl(_ r: SparseRef, _ b: [Double]) -> (Double, Bool) {
        var k = 0.0
        for (i, v) in zip(r.idx, r.val) { let a = Double(v); k += exp(a) * (a - b[Int(i)]) }
        let same = argmaxIndex(b) == r.top
        if !same { badMargin = max(badMargin, r.margin) }
        return (k, same)
    }
    let def = m.options.int8
    // Applies the tokens of a configuration (starts from the defaults, all layers, no minimum of rows).
    func applyCfg(_ c: String) -> (Int, Bool, Bool) {
        m.options.int8 = def
        m.options.int8.layers = nil
        m.options.int8.minPos = 0
        var int8 = mode, q8 = qk8, nz = false
        for t in c.split(separator: "/").map(String.init) {
            let kv = t.split(separator: "=", maxSplits: 1).map(String.init), v = kv.count > 1 ? kv[1] : ""
            switch kv[0] {
            case "exact": int8 = 0
            case "noise": int8 = 0; nz = true
            case "L": m.options.int8.layers = parseLayerSet(v.replacingOccurrences(of: "+", with: ","))
            case "P": m.options.int8.parts = Set(v.split(separator: "+").map(String.init))
            case "R": m.options.int8.minPos = Int(v) ?? 0
            case "Q": q8 = v == "1"
            case "QM": m.options.int8.qk8MinCtx = Int(v) ?? def.qk8MinCtx
            case "base": break
            default: print("unknown token: \(t)")
            }
        }
        return (int8, q8, nz)
    }
    var refs: [[SparseRef]] = [], toks: [[UInt32]] = [], tref: [Double] = []
    try warm()
    for (_, ids) in prompts {
        m.options.prefillInt8 = false; m.options.prefillQK8 = false; m.options.bench.gdnChunk = true
        m.reset()
        let (_, sec) = try m.prefill(ids, drafter: nil)
        var lps = [sparse(logProbs(m))], tk: [UInt32] = [m.top1.contents().load(as: UInt32.self)]
        for i in 0..<n { tk.append(try m.stepLogits(tk[i])); lps.append(sparse(logProbs(m))) }
        refs.append(lps); toks.append(tk); tref.append(sec)
    }
    print(
        "KL(exact || configuration), \(n) forced steps; prompts: "
            + zip(prompts, tref).map {
                String(
                    format: "%@ (%d tok, exact %.0f tok/s)", ($0.0.0 as NSString).lastPathComponent, $0.0.1.count,
                    Double($0.0.1.count) / $0.1)
            }.joined(separator: ", "))
    for c in list {
        let (int8, q8, nz) = applyCfg(c)
        m.reconfigureI8()
        var per: [String] = [], all: [Double] = [], agreeT = 0, posT = 0, rate: [String] = []
        for (j, (name, ids)) in prompts.enumerated() {
            m.options.prefillInt8 = int8 > 0; m.options.prefillQK8 = q8; m.options.bench.gdnChunk = !nz
            m.reset()
            let (_, sec) = try m.prefill(ids, drafter: nil)
            var kls: [Double] = [], agree = 0
            let (k0, a0) = kl(refs[j][0], logProbs(m)); kls.append(k0); if a0 { agree += 1 }
            for i in 0..<n {
                _ = try m.stepLogits(toks[j][i]); let (k, a) = kl(refs[j][i + 1], logProbs(m)); kls.append(k)
                if a { agree += 1 }
            }
            all += kls; agreeT += agree; posT += kls.count
            per.append(
                String(
                    format: "%@ %.2e (med %.0e, max %.3f, %d/%d)",
                    (name as NSString).lastPathComponent.replacingOccurrences(of: ".json", with: ""),
                    kls.reduce(0, +) / Double(kls.count), kls.sorted()[kls.count / 2], kls.max()!, agree, kls.count))
            rate.append(String(format: "%+.0f%%", (tref[j] / sec - 1) * 100))
        }
        m.options.bench.gdnChunk = true
        print(
            String(
                format: "[%@] mean %.2e, median %.1e, top-1 %d/%d", c, all.reduce(0, +) / Double(all.count),
                all.sorted()[all.count / 2], agreeT, posT) + " | " + per.joined(separator: "; ")
                + " | read " + rate.joined(separator: " ")
                + String(format: " | max reference margin on a differing top-1 %.3f", badMargin))
        badMargin = 0
    }
}
