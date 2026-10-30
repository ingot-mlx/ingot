import Foundation
import Ingot
import IngotBench
import IngotCLI

// ingot-dev fncheck: Flash-Next (HybridModel, the engine path) against the reference harness (ref/flashnext/harness.py,
// mlx-vlm layer by layer over the same weights). Compares the streams after each prompt layer (a single chunk; prefill
// truncated with `debugStopAfter`), the harness row logits (top-1 and KL from the reference to ours; the
// first row comes from prefill and the rest from one-row rounds with the prompt forced) and the greedy tokens.
//   ingot-dev fncheck --model DIR --ref HARNESS_DIR [--layers N] [--gen N] [--chunk 512]
// The separate FlashNextModel, which it was first validated against, was removed; the
// invariants it checked (split prefill, 8-row round vs 8 one-row ones) are covered by splitcheck and gen --check.

func cmdFNCheck() throws {
    if args.opts["emit"] != nil { return try fnEmit() }
    let (dir, cfg, ctx) = try loadModelFiles(args.required("model", modelHelp))
    let ref = URL(fileURLWithPath: (args.str("ref", "") as NSString).expandingTildeInPath)
    let ids = try NPYArray(contentsOf: ref.appendingPathComponent("ids.npy")).data.withUnsafeBytes {
        $0.bindMemory(to: Int32.self).map { UInt32($0) }
    }
    let meta = try jsonObject(at: ref.appendingPathComponent("meta.json"))
    let nl = args.int("layers", (meta["layers"] as? NSNumber)?.intValue ?? cfg.numLayers)
    let chunk = args.int("chunk", max(64, (ids.count + 63) / 64 * 64))
    let gen = args.int("gen", 0)
    let m = try HybridModel(
        ctx: ctx, store: try SafetensorsStore(directory: dir), config: cfg, maxCtx: ids.count + gen + 256,
        chunkRows: chunk, options: try engineOptions(cfg))
    guard m.fn != nil else { throw SafetensorsError.format("fncheck: \(dir.lastPathComponent) is not Flash-Next") }
    if nl < cfg.numLayers { m.debugStopAfter = nl - 1 }
    print("Flash-Next: \(nl) layers, prompt \(ids.count) tokens, chunk \(chunk)")
    let fm = FileManager.default
    // 1) streams after each layer (only if the prompt fits in one chunk): one truncated prefill per layer.
    let dumpLayers =
        ids.count <= chunk
        ? (0..<nl).filter { fm.fileExists(atPath: ref.appendingPathComponent("streams_L\($0).npy").path) } : []
    for i in dumpLayers {
        m.debugStopAfter = i
        _ = try m.prefill(ids, drafter: nil)
        let o = m.flashStreams(rows: ids.count)
        let r = try NPYArray(contentsOf: ref.appendingPathComponent("streams_L\(i).npy")).floats()
        guard r.count == o.count else { print("layer \(i): \(o.count) vs \(r.count) values"); continue }
        print(streamReport(layer: i, ours: o, ref: r, rows: ids.count))
    }
    m.debugStopAfter = nl < cfg.numLayers ? nl - 1 : nil
    // 2) logits of the harness rows.
    let rows = ((meta["logits_rows"] as? [NSNumber])?.map(\.intValue) ?? []).filter { $0 < ids.count }.sorted()
    if let first = rows.first, fm.fileExists(atPath: ref.appendingPathComponent("logits.npy").path) {
        let rl = try NPYArray(contentsOf: ref.appendingPathComponent("logits.npy"))
        let allRows = (meta["logits_rows"] as? [NSNumber])?.map(\.intValue) ?? []
        let rowIndex = Dictionary(allRows.enumerated().map { ($1, $0) }, uniquingKeysWith: { a, _ in a })
        let V = m.vocab
        var top1 = 0, n = 0, klSum = 0.0, klMax = 0.0
        let t0 = Date()
        _ = try m.prefill(Array(ids[...first]), drafter: nil)
        var pos = first
        while true {
            if let j = rowIndex[pos] {
                let refRow = rl.data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)[(j * V)..<((j + 1) * V)]) }
                let lg = logitsRow0(m)
                let kl = klDivergence(ref: refRow, ours: lg)
                klSum += kl; klMax = max(klMax, kl); n += 1
                if argmaxIndex(refRow) == argmaxIndex(lg) { top1 += 1 }
            }
            guard pos < rows.last! else { break }
            pos += 1
            _ = try m.stepLogits(ids[pos])
        }
        print(
            String(
                format: "logits (%d rows, %.1f s): top-1 equal %.1f%%, mean KL %.4g, max %.4g", n,
                Date().timeIntervalSince(t0), 100 * Double(top1) / Double(max(n, 1)), klSum / Double(max(n, 1)),
                klMax))
    }
    // 3) greedy
    if gen > 0,
        let want = try? JSONSerialization.jsonObject(with: Data(contentsOf: ref.appendingPathComponent("greedy.json")))
            as? [NSNumber]
    {
        let t1 = Date()
        // Greedy with the gap between the top two logits at each step (to see whether a divergence is a near tie).
        var got = [try m.prefill(ids, drafter: nil).token], gaps = [topGap(logitsRow0(m))]
        while got.count < min(gen, want.count) {
            got.append(try m.stepLogits(got.last!))
            gaps.append(topGap(logitsRow0(m)))
        }
        let w = want.map { UInt32($0.intValue) }
        let common = zip(got, w).prefix { $0 == $1 }.count
        print(
            String(
                format: "greedy: %d of %d tokens equal to the harness before the first difference (%.1f s)", common,
                got.count, Date().timeIntervalSince(t1)))
        if common < got.count {
            print(String(format: "  at the first difference, our logit 1st − 2nd = %.4f", gaps[common]))
        }
        print("  ours:    \(got.map(String.init).joined(separator: " "))")
        print("  harness: \(w.prefix(got.count).map(String.init).joined(separator: " "))")
    }
}

/// Error of a layer's streams against the harness: relative RMS, maximum, and the worst row (an expert that flips from
/// a near tie in the router shows up in a single row).
private func streamReport(layer i: Int, ours o: [Float], ref r: [Float], rows: Int) -> String {
    var mxd: Float = 0, mxr: Float = 0, se = 0.0, sr = 0.0
    for k in 0..<r.count {
        let dd = abs(o[k] - r[k])
        mxd = max(mxd, dd); mxr = max(mxr, abs(r[k]))
        se += Double(dd) * Double(dd); sr += Double(r[k]) * Double(r[k])
    }
    let W = o.count / rows
    var worst = (row: 0, err: 0.0), rowsOver = 0
    for row in 0..<rows {
        var e2 = 0.0, r2 = 0.0
        for k in (row * W)..<((row + 1) * W) {
            e2 += Double(o[k] - r[k]) * Double(o[k] - r[k]); r2 += Double(r[k]) * Double(r[k])
        }
        let er = (e2 / max(r2, 1e-30)).squareRoot()
        if er > worst.err { worst = (row, er) }
        if er > 1e-2 { rowsOver += 1 }
    }
    return String(
        format:
            "layer %2d: streams relative RMS error %.2e, max |diff| %.3g (max |ref| %.3g); worst row %d (%.2e), "
            + "%d rows > 1e-2", i, (se / sr).squareRoot(), mxd, mxr, worst.row, worst.err, rowsOver)
}

/// Logits of row 0 of the last round or of the prefill.
private func logitsRow0(_ m: HybridModel) -> [Float] {
    Array(UnsafeBufferPointer(start: m.logits.contents().bindMemory(to: Float.self, capacity: m.vocab), count: m.vocab))
}

/// KL(ref ‖ ours) in fp64.
private func klDivergence(ref: [Float], ours: [Float]) -> Double {
    let lr = logSumExp(ref), lo = logSumExp(ours)
    var kl = 0.0
    for k in ref.indices {
        let pr = Double(ref[k]) - lr
        kl += exp(pr) * (pr - (Double(ours[k]) - lo))
    }
    return kl
}

/// Difference between the two largest logits.
func topGap(_ l: [Float]) -> Float {
    var a = -Float.infinity, b = -Float.infinity
    for v in l { if v > a { b = a; a = v } else if v > b { b = v } }
    return a - b
}

/// ingot-dev fncheck --emit DIR --ids PROMPT.npy --gen N [--chunk 512]: engine greedy (prefill and one-row rounds) and
/// its logits, to compare with the harness by forcing the same tokens (ref/flashnext/tfcompare.py). Writes to DIR
/// ids_tf.npy (prompt + the first N − 1 generated tokens: what the harness has to read), ours_logits.npy (fp32 [N,
/// vocab]: the last prompt row and the N − 1 rounds) and ours_tokens.json.
func fnEmit() throws {
    let (dir, cfg, ctx) = try loadModelFiles(args.required("model", modelHelp))
    let out = URL(fileURLWithPath: (args.str("emit", "") as NSString).expandingTildeInPath)
    try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
    let ids = try NPYArray(contentsOf: URL(fileURLWithPath: (args.str("ids", "") as NSString).expandingTildeInPath))
        .data.withUnsafeBytes { $0.bindMemory(to: Int32.self).map { UInt32($0) } }
    let gen = args.int("gen", 64)
    let chunk = args.int("chunk", 512)
    let t0 = Date()
    // The HybridModel path: whole prefill with head and one-row rounds (stepLogits). `--hybrid` (formerly, to select it
    // over the separate model) is accepted and does nothing.
    let m = try HybridModel(
        ctx: ctx, store: try SafetensorsStore(directory: dir), config: cfg,
        maxCtx: max(ids.count + gen, 8192) + 64,
        chunkRows: chunk, options: try engineOptions(cfg))
    if let nl = args.opts["layers"].flatMap({ Int($0) }) { m.debugStopAfter = nl - 1 }
    let V = m.vocab
    func row0() -> [Float] {
        Array(UnsafeBufferPointer(start: m.logits.contents().bindMemory(to: Float.self, capacity: V), count: V))
    }
    // --force DIR: the tokens from another emit (ours_tokens.json) instead of our own greedy (engine-vs-engine KL, same
    // context, e.g. lossy options against the exact path).
    let forced: [UInt32]? = try args.opts["force"].map {
        let u = URL(fileURLWithPath: ($0 as NSString).expandingTildeInPath).appendingPathComponent(
            "ours_tokens.json")
        return try jsonUInt32s(at: u, key: "tokens")
    }
    let first = try m.prefill(ids, drafter: nil).token
    var got = [forced?[0] ?? first]
    var lg = row0()
    while got.count < gen {
        let t = try m.stepLogits(got.last!)
        got.append(forced.map { $0[got.count] } ?? t)
        lg += row0()
    }
    if let es = m.expertStream { print("  " + es.summary(rounds: gen)) }
    if m.options.prefillInt8 || m.options.prefillQK8 {
        print("  lossy prefill: \(m.prefillI8Chunks) chunks in W8A8, \(m.prefillQK8Chunks) with Q·Kᵀ in int8")
    }
    if args.flags.contains("mincore"), let fp = m.fn {
        // SSD reads bypass the file cache (F_NOCACHE, aligned): resident pages of
        // the PLE rows of this prompt and of the expert records of layers 0, 24 and 47.
        let idx = fp.fmt.pleIndices(context: [fp.fmt.pleEos, fp.fmt.pleEos], tokens: ids)
        let r = fp.pleRows.residentPages(rows: idx)
        let x = mincoreFile(
            dir.appendingPathComponent("experts.bin"),
            ranges: [0, 24, 47].map {
                ($0 * fp.E * fp.fmt.recordBytes, fp.E * fp.fmt.recordBytes)
            })
        print(
            "  mincore: ple.bin \(r.resident) of \(r.checked) pages resident; experts.bin \(x.resident) of \(x.checked)"
        )
    }
    if let fp = m.fn {
        let s = fp.pleRows.stats
        print(
            String(
                format: "  PLE: %d rows requested, hits %.1f%%, %d read from the SSD (%.1f ms reading)", s.lookups,
                s.hitRate * 100, s.reads, Double(s.readNs) / 1e6))
    }
    let tf = (ids + got.dropLast()).map { Int32(bitPattern: $0) }
    try NPYArray(kind: .i4, shape: [tf.count], data: tf.withUnsafeBytes { Data($0) })
        .write(to: out.appendingPathComponent("ids_tf.npy"))
    try NPYArray(kind: .f4, shape: [gen, V], data: lg.withUnsafeBytes { Data($0) })
        .write(to: out.appendingPathComponent("ours_logits.npy"))
    try JSONSerialization.data(withJSONObject: ["prompt": ids.count, "tokens": got.map(Int.init), "chunk": chunk])
        .write(to: out.appendingPathComponent("ours_tokens.json"))
    print(
        String(
            format: "emit: prompt %d, %d tokens, chunk %d, %.1f s → %@", ids.count, gen, chunk,
            Date().timeIntervalSince(t0), out.path))
}

/// Resident pages in the file cache (mincore) of the (offset, bytes) ranges of a file.
func mincoreFile(_ url: URL, ranges: [(Int, Int)]) -> (resident: Int, checked: Int) {
    let fd = open(url.path, O_RDONLY)
    guard fd >= 0 else { return (0, 0) }
    defer { close(fd) }
    var st = stat()
    guard fstat(fd, &st) == 0 else { return (0, 0) }
    let size = Int(st.st_size)
    guard let base = mmap(nil, size, PROT_READ, MAP_SHARED, fd, 0), base != MAP_FAILED else { return (0, 0) }
    defer { munmap(base, size) }
    let pg = Int(getpagesize())
    var res = 0, tot = 0
    for (o, n) in ranges {
        let a = o & ~(pg - 1), pages = (n + o - a + pg - 1) / pg
        var vec = [CChar](repeating: 0, count: pages)
        guard mincore(base + a, pages * pg, &vec) == 0 else { continue }
        res += vec.filter { $0 & 1 != 0 }.count
        tot += pages
    }
    return (res, tot)
}
