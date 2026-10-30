import Foundation
import IngotBench

// Toy Flash-Next (qwen4_exp in our format, scripts/flashnext/fnformat.py): 4-stream hyper-connections at 8 bits,
// PLE in layer 1 with its table in ple.bin, QSA in the attention layer, 512-expert Q2_0 top-10 MoE in experts.bin
// (read from the SSD by the expert cache) plus the 4-bit shared expert and the 6-bit lm_head.

// Hidden (2560), streams, PLE and QSA as in the real model (PLE uses ple_embed_dim = hidden and 100 B rows out of 160
// per head); the experts and the shared expert use the toy intermediate size (ToyCut.moeI versus 640): that is what
// weighed (2.8 of the 3.2 GB). The ci.sh expert cache is tuned to the 17-page records.
// The dimensions come from the real model (ref/toy/real/qwen3.8-flash-next.json, ToyDims.flashNext).
private let page = 16384, pleRow = 100, pleRowsPerPage = 163

/// Bytes of codes and of scales of an [n, k] matrix in planar Q2_0 (fnformat.plane_sizes).
private func q2Planes(_ n: Int, _ k: Int) -> (codes: Int, scales: Int) { (n * k / 4, n * (k / 64) * 2) }

func writeToyFlash(_ dir: URL, seed: UInt64, _ d: ToyDims) throws {
    let t = d.real.text
    let fnH = d.hidden, fnE = d.experts, fnI = d.moeI
    guard let hcS = t["hc_count"] as? Int, let fnR = t["hc_lowrank"] as? Int, let ixH = t["indexer_n_heads"] as? Int,
        let ixD = t["indexer_head_dim"] as? Int, let ixKV = t["indexer_kv_heads"] as? Int
    else { throw CocoaError(.fileReadCorruptFile) }
    let fnW = hcS * fnH
    // Multipliers of the PLE n-gram hash: those of the real model (ingot.ple in fnformat.py).
    guard let realPLE = (d.real.json["ingot"] as? [String: Any])?["ple"] as? [String: Any],
        let pleMul = realPLE["multipliers"] as? [Int]
    else { throw CocoaError(.fileReadCorruptFile) }
    let st = ToySafetensors(seed: seed ^ 0xF1A5)
    let root = "language_model.model."
    var over: [String: Any] = [:]
    func q8(_ name: String, rows: Int, cols: Int, gain: Float = 1) {
        st.quant(name, rows: rows, cols: cols, bits: 8, gain: gain)
        over[name] = ["group_size": 64, "bits": 8, "mode": "affine"]
    }
    func f32(_ name: String, _ shape: [Int], _ mean: Float, _ sd: Float) {
        st.float(name, .f32, shape) { mean + sd * $0.normal() }
    }
    /// Hyper-connection: zero-centred norms (the engine adds 1), down [R, 4H], up [4H, R] and injection [4, 4H].
    func hc(_ p: String, inject: Bool, normMean: Float = 0) {
        f32(p + "hc_norm.weight", [fnW], normMean, 0.5)
        q8(p + "input_mix_weight_down", rows: fnR, cols: fnW)
        q8(p + "input_mix_weight_up", rows: fnW, cols: fnR)
        if inject { q8(p + "block_inject_weight", rows: hcS, cols: fnW) }
    }
    for i in 0..<ToySpec.layers {
        let p = root + "layers.\(i)."
        hc(p + "attn_hyper_connection.", inject: true)
        hc(p + "mlp_hyper_connection.", inject: true)
        if d.layerTypes[i] == "full_attention" {
            let a = p + "self_attn."
            st.quant(a + "q_proj", rows: d.qh * d.hd * 2, cols: fnH)
            st.quant(a + "k_proj", rows: d.kvh * d.hd, cols: fnH)
            st.quant(a + "v_proj", rows: d.kvh * d.hd, cols: fnH)
            st.quant(a + "o_proj", rows: fnH, cols: d.qh * d.hd)
            f32(a + "q_norm.weight", [d.hd], 0.28, 0.1)
            f32(a + "k_norm.weight", [d.hd], 0.28, 0.1)
            let sd = 1 / Float(fnH).squareRoot()
            st.float(a + "indexer.index_qk_proj.weight", .bf16, [(ixH + ixKV) * ixD, fnH]) { sd * $0.normal() }
            f32(a + "indexer.q_layernorm.weight", [ixD], -0.04, 0.07)
            f32(a + "indexer.k_layernorm.weight", [ixD], -0.04, 0.07)
        } else {
            let g = p + "linear_attn."
            f32(g + "A_log", [d.gdnHV], 1, 2)
            f32(g + "dt_bias", [d.gdnHV], -3.5, 2.5)
            f32(g + "conv1d.weight", [d.gdnConv, 4, 1], 0, 0.2)
            f32(g + "norm.weight", [d.gdnDV], 0.97, 0.03)
            st.quant(g + "in_proj_qkv", rows: d.gdnConv, cols: fnH)
            st.quant(g + "in_proj_z", rows: d.gdnHV * d.gdnDV, cols: fnH)
            st.quant(g + "in_proj_a", rows: d.gdnHV, cols: fnH)
            st.quant(g + "in_proj_b", rows: d.gdnHV, cols: fnH)
            st.quant(g + "out_proj", rows: fnH, cols: d.gdnHV * d.gdnDV)
        }
        let m = p + "mlp."
        q8(m + "gate", rows: fnE, cols: fnH, gain: 2)
        q8(m + "shared_expert_gate", rows: 1, cols: fnH)
        st.quant(m + "shared_expert.gate_proj", rows: d.sharedI, cols: fnH)
        st.quant(m + "shared_expert.up_proj", rows: d.sharedI, cols: fnH)
        st.quant(m + "shared_expert.down_proj", rows: fnH, cols: d.sharedI)
        if i == ToySpec.pleLayer {
            let e = p + "ple."
            f32(e + "conv1d.weight", [fnW, 4, 1], 0, 0.2)
            st.quant(e + "key_proj", rows: fnW, cols: fnH)
            for n in ["norm_conv", "norm_key", "norm_query"] { f32(e + n + ".weight", [fnW], -0.1, 0.1) }
            q8(e + "value_proj", rows: fnH, cols: fnH)
        }
    }
    hc(root + "hyper_connection_mixer.", inject: false, normMean: 1)
    st.quant(root + "embed_tokens", rows: ToySpec.vocab, cols: fnH)
    st.quant("language_model.lm_head", rows: ToySpec.vocab, cols: fnH, bits: 6, gain: 2, boost: ToySpec.lmBoost)
    over["language_model.lm_head"] = ["group_size": 64, "bits": 6, "mode": "affine"]
    try st.write(to: dir.appendingPathComponent("model.safetensors"))

    // experts.bin: [layer][expert] 17-page records (85 at intermediate 640); planes gate | up | down (u32 + f16).
    let gate = q2Planes(fnI, fnH), down = q2Planes(fnH, fnI)
    let offs = [
        ("gate", 0, gate), ("up", gate.codes + gate.scales, gate),
        ("down", 2 * (gate.codes + gate.scales), down),
    ]
    let payload = 2 * (gate.codes + gate.scales) + down.codes + down.scales
    let record = (payload + page - 1) / page * page
    try writeRecords(dir.appendingPathComponent("experts.bin"), count: ToySpec.layers * fnE, size: record) { r, buf in
        var g = ToyRNG(seed: seed ^ 0xE4E4, name: "expert.\(r)")
        for (name, o, pl) in offs {
            g.fill(UnsafeMutableRawBufferPointer(rebasing: buf[o..<(o + pl.codes)]))
            let k = name == "down" ? fnI : fnH
            let base = (name == "down" ? 0.5 : 1) / Float(k).squareRoot() / 1.118
            let s = (buf.baseAddress! + o + pl.codes).bindMemory(to: UInt16.self, capacity: pl.scales / 2)
            for j in 0..<(pl.scales / 2) {
                s[j] = Float16(g.sign() * base * max(0.25, 1 + 0.25 * g.normal())).bitPattern
            }
        }
    }

    // ple.bin: 163 rows of 100 B per page (codes 80 B | 5 bf16 scales | 5 bf16 biases).
    let rows = (ToySpec.pleSizes.reduce(0, +) + 127) / 128 * 128
    let pages = (rows + pleRowsPerPage - 1) / pleRowsPerPage
    try writeRecords(dir.appendingPathComponent("ple.bin"), count: pages, size: page) { pg, buf in
        var g = ToyRNG(seed: seed ^ 0x9AE, name: "ple.\(pg)")
        for r in 0..<pleRowsPerPage where pg * pleRowsPerPage + r < rows {
            let o = r * pleRow
            g.fill(UnsafeMutableRawBufferPointer(rebasing: buf[o..<(o + 80)]))
            let sb = (buf.baseAddress! + o + 80).bindMemory(to: UInt16.self, capacity: 10)
            for j in 0..<5 {
                let s = g.sign() * 0.004 * max(0.25, 1 + 0.25 * g.normal())
                sb[j] = toyBF16(s); sb[5 + j] = toyBF16(-s * (7.5 + 0.3 * g.normal()))
            }
        }
    }

    var offsets: [Int] = [], acc = 0
    for s in ToySpec.pleSizes { offsets.append(acc); acc += s }
    var quant: [String: Any] = ["group_size": 64, "bits": 4, "mode": "affine"]
    quant.merge(over) { a, _ in a }
    let expOffsets = Dictionary(
        uniqueKeysWithValues: offs.map { ($0.0, ["codes": $0.1, "scales": $0.1 + $0.2.codes]) })
    let block: [String: Any] = [
        "format": 1,
        "experts": [
            "file": "experts.bin", "kind": "q2_0_planar", "record_bytes": record, "payload_bytes": payload,
            "layers": ToySpec.layers, "experts": fnE, "hidden": fnH, "intermediate": fnI, "group": 64,
            "offsets": expOffsets, "source": "ingot-dev toymodels (aleatorio)",
        ] as [String: Any],
        "ple": [
            "file": "ple.bin", "rows": rows, "dim": 160, "bits": 4, "group": 32, "row_bytes": pleRow,
            "rows_per_page": pleRowsPerPage, "page_bytes": page, "layer": ToySpec.pleLayer,
            "multipliers": pleMul, "head_offsets": offsets,
            "head_vocab_sizes": ToySpec.pleSizes, "eos_token_id": ToySpec.tok(248_044),
            "source": "ingot-dev toymodels (aleatorio)",
        ] as [String: Any],
        "dense": ["source": "ingot-dev toymodels (aleatorio)", "method": "aleatorio", "hc_bits": 8] as [String: Any],
    ]
    let cfg = ToyConfig.flash(d, pleLayer: ToySpec.pleLayer, quant: quant, ingotBlock: block, seed: Int(seed))
    try JSONSerialization.data(withJSONObject: cfg, options: [.sortedKeys, .prettyPrinted])
        .write(to: dir.appendingPathComponent("config.json"))
}

/// Writes `count` records of `size` bytes filled by `fill` (zero wherever it does not write).
private func writeRecords(
    _ url: URL, count: Int, size: Int, _ fill: (Int, UnsafeMutableRawBufferPointer) -> Void
) throws {
    FileManager.default.createFile(atPath: url.path, contents: nil)
    let fh = try FileHandle(forWritingTo: url)
    defer { try? fh.close() }
    let per = max(1, (64 << 20) / size)  // ~64 MB per write
    let buf = UnsafeMutableRawBufferPointer.allocate(byteCount: per * size, alignment: page)
    defer { buf.deallocate() }
    var r = 0
    while r < count {
        let n = min(per, count - r)
        memset(buf.baseAddress!, 0, n * size)
        for j in 0..<n { fill(r + j, UnsafeMutableRawBufferPointer(rebasing: buf[(j * size)..<((j + 1) * size)])) }
        try fh.write(contentsOf: Data(bytesNoCopy: buf.baseAddress!, count: n * size, deallocator: .none))
        r += n
    }
}
