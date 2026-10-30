import Foundation
import LabCLI
import MetalLab

// lab-dev fnprof: per-piece profile of Flash-Next (HybridModel.flashTimeline): one round of --rows rows and one prefill
// chunk of --prefill rows over a history of --pos tokens, kernels only (no SSD, no waits).
//   lab-dev fnprof --model DIR [--rows 8] [--pos 4096] [--prefill 2048] [--reps 3]
func cmdFNProf() throws {
    let (dir, cfg, ctx) = try loadModelFiles(args.required("model", modelHelp))
    let rows = args.int("rows", 8), pos = args.int("pos", 4096), pfRows = args.int("prefill", 2048)
    var o = try engineOptions(cfg)
    if o.expertCacheBytes == nil { o.expertCacheBytes = 4_000_000_000 }
    let m = try HybridModel(
        ctx: ctx, store: try SafetensorsStore(directory: dir), config: cfg, maxCtx: pos + max(pfRows, 64) + 64,
        chunkRows: max(pfRows, 64), options: o)
    for (label, pf) in [("round of \(rows) rows", nil as Int?), ("prefill of \(pfRows) rows", pfRows)]
    where pf == nil || pfRows > 0 {
        var best: (pieces: [(String, Double)], total: Double, gaps: Double)?
        for _ in 0..<args.int("reps", 3) {
            m.reset()
            let r = try m.flashTimeline(rows: rows, pos: pos, prefill: pf)
            if best == nil || r.total < best!.total { best = r }
        }
        let b = best!
        print(
            String(
                format: "\n## %@ (history %d): %.2f ms GPU time (gaps between pieces %.2f ms)", label, pos, b.total,
                b.gaps))
        for (n, ms) in b.pieces.sorted(by: { $0.1 > $1.1 }) {
            print(String(format: "  %-18@ %8.3f ms  %5.1f %%", n, ms, ms / b.total * 100))
        }
    }
}
