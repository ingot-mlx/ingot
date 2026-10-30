import Foundation

// Qwen3.8-Flash-Next format in the lab (scripts/flashnext/fnformat.py): Q2 expert records from
// experts.bin and PLE table from ple.bin, described in config.json → "metal_lab". Used by FlashParts, ExpertStream,
// FlashPLERows and the tests.

/// Formato de Flash-Next (config.json → "metal_lab").
public struct FlashNextFormat: Sendable {
    public let recordBytes: Int
    public let gateCodes: Int, gateScales: Int, upCodes: Int, upScales: Int, downCodes: Int, downScales: Int
    public let experts: Int, layers: Int
    public let pleRows: Int, pleRowBytes: Int, pleRowsPerPage: Int, plePage: Int, pleLayer: Int
    public let pleMult: [UInt64], pleOff: [UInt64], pleSize: [UInt64], pleEos: UInt32
    /// PLE heads (all of them, at most 16, and the 2-gram ones: the first ones).
    public let pleHeads: Int, pleHeadsPerNgram: Int

    public init(_ cfg: ModelConfig) throws {
        guard let lab = cfg.raw["metal_lab"] as? [String: Any], let ex = lab["experts"] as? [String: Any],
            let ple = lab["ple"] as? [String: Any], let off = ex["offsets"] as? [String: [String: NSNumber]]
        else { throw SafetensorsError.format("config.json without metal_lab (not our Flash-Next format)") }
        func i(_ d: [String: Any], _ k: String) throws -> Int {
            guard let v = d[k] as? NSNumber else { throw SafetensorsError.format("metal_lab: missing \(k)") }
            return v.intValue
        }
        recordBytes = try i(ex, "record_bytes")
        func o(_ m: String, _ k: String) throws -> Int {
            guard let v = off[m]?[k] else { throw SafetensorsError.format("metal_lab: missing \(m).\(k)") }
            return v.intValue
        }
        gateCodes = try o("gate", "codes"); gateScales = try o("gate", "scales")
        upCodes = try o("up", "codes"); upScales = try o("up", "scales")
        downCodes = try o("down", "codes"); downScales = try o("down", "scales")
        experts = try i(ex, "experts"); layers = try i(ex, "layers")
        pleRows = try i(ple, "rows"); pleRowBytes = try i(ple, "row_bytes")
        pleRowsPerPage = try i(ple, "rows_per_page"); plePage = try i(ple, "page_bytes"); pleLayer = try i(ple, "layer")
        func u64(_ k: String) -> [UInt64] { (ple[k] as? [NSNumber] ?? []).map { $0.uint64Value } }
        pleMult = u64("multipliers"); pleOff = u64("head_offsets"); pleSize = u64("head_vocab_sizes")
        pleEos = UInt32(try i(ple, "eos_token_id"))
        pleHeads = pleSize.count
        pleHeadsPerNgram = (cfg.text["heads_per_ngram"] as? NSNumber)?.intValue ?? 8
        // MLX 4-bit g32 rows: codes | bf16 scales | bf16 biases, of the per-head width (dim).
        let dim = (ple["dim"] as? NSNumber)?.intValue ?? 160
        guard recordBytes % 16384 == 0, pleMult.count == 3, pleOff.count == pleHeads, (1...16).contains(pleHeads),
            pleHeadsPerNgram <= pleHeads, dim % 32 == 0, pleRowBytes == dim / 2 + dim / 32 * 4
        else { throw SafetensorsError.format("metal_lab: unsupported Flash-Next format") }
    }

    /// Indices into the PLE table (`pleHeads` per position) of `tokens` with 2 context tokens in front (`ctx`), like mlx-vlm:
    /// a previous token that crosses an EOS (within the context + tokens window) is EOS.
    public func pleIndices(context ctx: [UInt32], tokens: [UInt32]) -> [UInt32] {
        precondition(ctx.count == 2)
        let h = ctx + tokens
        let H = pleHeads, HN = pleHeadsPerNgram
        var out = [UInt32](repeating: 0, count: tokens.count * H)
        var lastEos = -1  // last EOS strictly before the position
        for p in 0..<h.count {
            if p >= 2 {
                let inSeg = p - (lastEos + 1)
                let t0 = UInt64(h[p]), t1 = UInt64(inSeg >= 1 ? h[p - 1] : pleEos)
                let t2 = UInt64(inSeg >= 2 ? h[p - 2] : pleEos)
                let m2 = (t0 &* pleMult[0]) ^ (t1 &* pleMult[1]), m3 = m2 ^ (t2 &* pleMult[2])
                for k in 0..<H {
                    out[(p - 2) * H + k] = UInt32((k < HN ? m2 : m3) % pleSize[k] + pleOff[k])
                }
            }
            if h[p] == pleEos { lastEos = p }
        }
        return out
    }
}
