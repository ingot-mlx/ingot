import Foundation
import LabBench
import LabCLI
import LabServe

// lab-dev toymodels: toy models for `scripts/ci.sh --toy` (sustainable regression). One tiny model per architecture
// with deterministic random weights (seed) in the same format the real loaders read: MLX safetensors with their
// config.json, real quantizations (affine 4-bit g64, 8-bit for the router, the shared-expert gate and the
// hyper-connections, 6-bit for Flash-Next's lm_head, Q2_0 for the experts of experts.bin) and each one's DFlash2
// drafter. What the structure and the code paths depend on is kept (layer types and order, experts and top-k, shared
// expert, heads, head size and RoPE, QK-norm, group and bits, dense layers, drafter, Flash-Next's HC/PLE/QSA) and
// these are cut down: the number of layers (4: 3 GDN + 1 attention), the vocabulary (32768: the tokenizer and
// ref/out/prompts-toy do not change), the drafter (1 layer with 4 context layers) and the widths:
// ToyCut.qwenHidden and the intermediate sizes.

/// Toy architectures: directory name and golden name (ref/golden/<name>.json). One toy per tier-1 family, each from
/// its real model's config (ref/toy/real/) with only the ToyCut cuts.
enum ToyArch: String, CaseIterable {
    case moe = "toy-qwen36-moe", dense = "toy-qwen38-dense", flash = "toy-flashnext", lfm = "toy-lfm2"
    case granite = "toy-granite", gemma4 = "toy-gemma4"
    case gptOss = "toy-gptoss", nemotron = "toy-nemotron", glm = "toy-glm", muse = "toy-muse", lfm3 = "toy-lfm3"

    /// Short name for `--arch`.
    var short: String {
        switch self {
        case .moe: "moe"
        case .dense: "dense"
        case .flash: "flash"
        case .lfm: "lfm"
        case .granite: "granite"
        case .gemma4: "gemma4"
        case .gptOss: "gptoss"
        case .nemotron: "nemotron"
        case .glm: "glm"
        case .muse: "muse"
        case .lfm3: "lfm3"
        }
    }
    /// Without a drafter: LFM2.5, toy-granite, toy-gptoss, toy-nemotron, toy-glm and toy-lfm3 (their families have none
    /// in the engine). toy-gemma4's is a DFlash v1 like z-lab's for the 26B-A4B (writeToyDrafterV1). The
    /// drafter is written after the weights with its own seed, so the toy's weights do not depend on it.
    var hasDrafter: Bool { ![.lfm, .granite, .gptOss, .nemotron, .glm, .lfm3].contains(self) }
}

/// Generator constants; the cuts against the real model are in LabBench (`ToyCut`, Sources/LabBench/ToyConfig.swift).
struct ToySpec {
    /// Gain of the router scale of toy-qwen36-moe (and of the former toy-qwen36-g32; sweep
    /// {2, 3, 4, 6} × 2 seeds): with 2 the router logits end up
    /// crowded together and a bf16 rounding changes the top-8 (fid-g32 4K: median 0.00108); with 4 the mean KL against
    /// mlx-lm fp32 stays < 0.003 and the median ≤ 0.00011 in both seeds, with all 256 experts used.
    static let routerGain: Float = 4
    static let vocab = ToyCut.vocab
    static let layers = ToyCut.layers
    /// PLE layer in Flash-Next (layer 1, as in the real model: `ple_layer_ids` [2] counts from 1).
    static let pleLayer = 1
    /// Sizes of the 16 heads of the PLE table (small primes: 64.8 K rows, 6.6 MB instead of 32 GB).
    static let pleSizes: [Int] = [
        4001, 4003, 4007, 4013, 4019, 4021, 4027, 4049, 4051, 4057, 4073, 4079, 4091, 4093, 4099, 4111,
    ]
    /// lm_head rows with the scale multiplied: two tokens that often win, so the (random) drafter guesses right
    /// sometimes and verification accepts drafts (with a uniform lm_head it accepts none and the
    /// folding of several GDN rows and the commit of several KV rows stay untested).
    static let lmBoost: (rows: Set<Int>, factor: Float) = ([1000, 2000], 16)
    static func tok(_ qwenId: Int) -> Int { ToyTokenizer.id(qwen: qwenId, vocab: vocab) }
}

/// lab-dev toymodels --out DIR [--arch moe,dense,flash,lfm,granite,gemma4,gptoss,nemotron,glm,muse,lfm3]
/// [--seed 1] [--force]: writes (or reuses, if the generator stamp matches) the toy models in
/// DIR/<architecture> and copies their prompts to ref/out/prompts-toy.
@MainActor func cmdToyModels() throws {
    let out = URL(fileURLWithPath: (args.str("out", "~/.cache/metal-lab/toy") as NSString).expandingTildeInPath)
    let seed = UInt64(args.int("seed", 1))
    let all = ToyArch.allCases.map(\.short).joined(separator: ",")
    let which = try args.str("arch", all).split(separator: ",").map { s -> ToyArch in
        guard let a = ToyArch.allCases.first(where: { $0.short == s || $0.rawValue == s }) else {
            throw UsageError("--arch: unknown architecture \(s) (\(all))")
        }
        return a
    }
    let stamp = try toyStamp(seed: seed)
    let fm = FileManager.default
    try fm.createDirectory(at: out, withIntermediateDirectories: true)
    var tokenizer: Data?
    func tok() throws -> Data {
        if let t = tokenizer { return t }
        let corpus = try ["0001", "0008", "0015"].map {
            try String(contentsOf: repoRoot.appendingPathComponent("ref/agent_dump/\($0)-body.json"), encoding: .utf8)
        }.joined(separator: "\n")
        let t = try ToyTokenizer.json(corpus: corpus, merges: 6000, vocab: ToySpec.vocab)
        tokenizer = t
        return t
    }
    for arch in which {
        let dir = out.appendingPathComponent(arch.rawValue)
        let stampURL = dir.appendingPathComponent(".toy-stamp")
        if !args.flags.contains("force"), (try? String(contentsOf: stampURL, encoding: .utf8)) == stamp {
            print("\(arch.rawValue): up to date (\(stamp))")
            continue
        }
        let t0 = Date()
        try? fm.removeItem(at: dir)
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        try writeToyCommon(dir, tokenizer: try tok())
        let d = try toyDims(arch)
        switch arch {
        case .moe, .dense: try writeToyQwen(dir, seed: seed, d, routerGain: ToySpec.routerGain)
        case .flash: try writeToyFlash(dir, seed: seed, d)
        case .lfm: try writeToyLFM(dir, seed: seed, d)
        case .granite: try writeToyGranite(dir, seed: seed, d)
        case .gemma4: try writeToyGemma4(dir, seed: seed, d)
        case .gptOss: try writeToyGptOss(dir, seed: seed, d)
        case .nemotron: try writeToyNemotron(dir, seed: seed, d)
        case .glm: try writeToyGLM(dir, seed: seed, d)
        case .muse: try writeToyMuse(dir, seed: seed, d)
        case .lfm3: try writeToyLFM3(dir, seed: seed, d)
        }
        if arch.hasDrafter {
            try fm.createDirectory(at: dir.appendingPathComponent("drafter"), withIntermediateDirectories: true)
            // toy-muse's like the real one: block 16 and the model's logits post-processing in dflash_config.
            let museKeys: [String: Any] =
                arch == .muse
                ? [
                    "block_size": 16, "output_multiplier": d.real.text["output_multiplier"] as Any,
                    "final_logit_softcapping": ToyConfig.museSoftcap,
                ] : [:]
            let dr = dir.appendingPathComponent("drafter")
            if arch == .gemma4 {
                try writeToyDrafterV1(dr, hidden: d.hidden, seed: seed)
            } else {
                try writeToyDrafter(dr, arch: arch, hidden: d.hidden, seed: seed, dflash: museKeys)
            }
        }
        try stamp.write(to: stampURL, atomically: true, encoding: .utf8)
        print(String(format: "%@: generated in %.1f s (%@)", arch.rawValue, Date().timeIntervalSince(t0), stamp))
    }
    // Prompts (with the toy tokenizer, the same in all of them): ref/out/prompts-toy.
    let pdir = repoRoot.appendingPathComponent("ref/out/prompts-toy")
    try fm.createDirectory(at: pdir, withIntermediateDirectories: true)
    let anyDir = out.appendingPathComponent(which[0].rawValue)
    let bpe = try BPETokenizer(directory: anyDir)
    for (name, ids) in try toyPrompts(bpe) {
        try JSONSerialization.data(withJSONObject: ["ids": ids]).write(to: pdir.appendingPathComponent("\(name).json"))
    }
    print("toy prompts in \(pdir.path)")
}

/// Real model each toy comes from (ref/toy/real/<name>.json) and its dimensions with the cuts.
func toyRealName(_ arch: ToyArch) -> String {
    switch arch {
    case .moe: return "qwen3.6-35b-a3b"
    case .dense: return "qwen3.8-27b"
    case .flash: return "qwen3.8-flash-next"
    case .lfm: return "lfm2.5-24b-a2b"
    case .granite: return "granite-4.1-3b"
    case .gemma4: return "gemma4-26b-a4b"
    case .gptOss: return "gpt-oss-20b"
    case .nemotron: return "nemotron-3.5-lightning-30b-a3b"
    case .glm: return "glm-4.7-flash"
    case .muse: return "muse-glimmer-30b"
    case .lfm3: return "lfm3-1.7b"
    }
}

func toyDims(_ arch: ToyArch) throws -> ToyDims {
    let r = try ToyReal.load(toyRealName(arch), root: repoRoot)
    switch arch {
    case .moe, .dense: return ToyDims.qwen(r)
    case .flash: return ToyDims.flashNext(r)
    case .lfm: return ToyDims.lfm(r)
    case .granite: return ToyDims.granite(r)
    case .gemma4: return ToyDims.gemma4(r)
    case .gptOss: return ToyDims.gptOss(r)
    case .nemotron: return ToyDims.nemotronH(r)
    case .glm: return ToyDims.glm(r)
    case .muse: return ToyDims.muse(r)
    case .lfm3: return ToyDims.lfm3(r)
    }
}

/// Generator stamp: hash of its sources (ToyConfig.swift and real configs too), template, agent dumps and seed
/// (change: regenerate).
func toyStamp(seed: UInt64) throws -> String {
    let dir = repoRoot.appendingPathComponent("Sources/lab-dev/Toy")
    var files = try FileManager.default.contentsOfDirectory(atPath: dir.path).filter { $0.hasSuffix(".swift") }.sorted()
        .map { dir.appendingPathComponent($0) }
    files.append(repoRoot.appendingPathComponent("Sources/LabBench/ToyConfig.swift"))
    files.append(repoRoot.appendingPathComponent("ref/toy/chat_template.jinja"))
    // The agent dumps are the tokenizer corpus and the text of the prompts.
    files += ["0001", "0008", "0015"].map { repoRoot.appendingPathComponent("ref/agent_dump/\($0)-body.json") }
    let real = repoRoot.appendingPathComponent("ref/toy/real")
    files += try FileManager.default.contentsOfDirectory(atPath: real.path).filter { $0.hasSuffix(".json") }.sorted()
        .map { real.appendingPathComponent($0) }
    var h: UInt64 = 0xCBF2_9CE4_8422_2325 ^ seed
    for f in files { for b in try Data(contentsOf: f) { h = (h ^ UInt64(b)) &* 0x100_0000_01B3 } }
    return String(format: "%016llx", h)
}

/// Prompts: `chat` (short), `512` and `4K` (509 and 3989 tokens: not multiples of 64, the tail of each chunk counts),
/// with Qwen's template by hand and reasoning enabled, over the text of ref/agent_dump/0008-body.json.
func toyPrompts(_ bpe: BPETokenizer) throws -> [(String, [Int])] {
    let head = bpe.encode("<|im_start|>user\n"), tail = bpe.encode("<|im_end|>\n<|im_start|>assistant\n<think>\n")
    let text = try String(contentsOf: repoRoot.appendingPathComponent("ref/agent_dump/0008-body.json"), encoding: .utf8)
    let body = bpe.encode(text)
    func prompt(_ n: Int) -> [Int] {
        precondition(body.count >= n, "text too short for the \(n)-token prompt")
        return head + Array(body.prefix(n - head.count - tail.count)) + tail
    }
    let chat = head + bpe.encode("Hola, ¿qué tal? Explícame en dos frases qué hace un kernel de Metal.") + tail
    return [("chat", chat), ("512", prompt(509)), ("4K", prompt(3989))]
}

private func writeJSON(_ obj: Any, _ url: URL) throws {
    try JSONSerialization.data(withJSONObject: obj, options: [.sortedKeys, .prettyPrinted]).write(to: url)
}

/// tokenizer.json, tokenizer_config.json, chat_template.jinja and generation_config.json (the same in all of them).
private func writeToyCommon(_ dir: URL, tokenizer: Data) throws {
    try tokenizer.write(to: dir.appendingPathComponent("tokenizer.json"))
    try writeJSON(
        [
            "eos_token": "<|im_end|>", "pad_token": "<|endoftext|>", "add_prefix_space": false,
            "model_max_length": 262144,
        ],
        dir.appendingPathComponent("tokenizer_config.json"))
    try FileManager.default.copyItem(
        at: repoRoot.appendingPathComponent("ref/toy/chat_template.jinja"),
        to: dir.appendingPathComponent("chat_template.jinja"))
    try writeJSON(
        [
            "bos_token_id": ToySpec.tok(248_044), "eos_token_id": [ToySpec.tok(248_046), ToySpec.tok(248_044)],
            "pad_token_id": ToySpec.tok(248_044), "do_sample": true, "temperature": 1.0, "top_k": 20, "top_p": 0.95,
        ], dir.appendingPathComponent("generation_config.json"))
}

/// MLX norms of Qwen3.5/3.6 (stored as 1 + w).
private func mlxNorm(_ mean: Float, _ sd: Float) -> (inout ToyRNG) -> Float { { mean + sd * $0.normal() } }

/// Qwen3.6-35B-A3B (the Heretic: 256-expert top-8 MoE + gated shared expert) or Qwen3.8-27B (GDN with 48 v heads,
/// 24 × 4 attention and dense FFN), with the dimensions of `d` (the real model's with the ToyCut cuts); `routerGain`:
/// gain of the router scale.
private func writeToyQwen(_ dir: URL, seed: UInt64, _ d: ToyDims, routerGain: Float) throws {
    let moe = d.experts > 0
    let H = d.hidden, HV = d.gdnHV, QH = d.qh, KVH = d.kvh, hd = d.hd
    let E = d.experts, eI = d.moeI, sI = d.sharedI, denseI = d.denseI
    let st = ToySafetensors(seed: seed)
    let root = "language_model.model."
    var q8: [String: Any] = [:]
    for i in 0..<ToySpec.layers {
        let p = root + "layers.\(i)."
        st.float(p + "input_layernorm.weight", .bf16, [H], mlxNorm(1, 0.05))
        st.float(p + "post_attention_layernorm.weight", .bf16, [H], mlxNorm(1, 0.05))
        if d.layerTypes[i] == "full_attention" {
            let a = p + "self_attn."
            st.quant(p + "self_attn.q_proj", rows: QH * hd * 2, cols: H)
            st.quant(p + "self_attn.k_proj", rows: KVH * hd, cols: H)
            st.quant(p + "self_attn.v_proj", rows: KVH * hd, cols: H)
            st.quant(p + "self_attn.o_proj", rows: H, cols: QH * hd)
            st.float(a + "q_norm.weight", .bf16, [hd], mlxNorm(1.2, 0.1))
            st.float(a + "k_norm.weight", .bf16, [hd], mlxNorm(1.2, 0.1))
        } else {
            let g = p + "linear_attn.", conv = d.gdnConv
            st.float(g + "A_log", .bf16, [HV]) { min(4, max(-5, -1.5 + 1.5 * $0.normal())) }
            st.float(g + "dt_bias", .bf16, [HV]) { 3 * $0.normal() }
            st.float(g + "conv1d.weight", .bf16, [conv, 4, 1]) { 0.2 * $0.normal() }
            st.float(g + "norm.weight", .bf16, [d.gdnDV], mlxNorm(0.9, 0.05))
            st.quant(p + "linear_attn.in_proj_qkv", rows: conv, cols: H)
            st.quant(p + "linear_attn.in_proj_z", rows: HV * d.gdnDV, cols: H)
            st.quant(p + "linear_attn.in_proj_a", rows: HV, cols: H)
            st.quant(p + "linear_attn.in_proj_b", rows: HV, cols: H)
            st.quant(p + "linear_attn.out_proj", rows: H, cols: HV * d.gdnDV)
        }
        let m = p + "mlp."
        if moe {
            st.quant(m + "gate", rows: E, cols: H, bits: 8, gain: routerGain)
            st.quant(m + "shared_expert_gate", rows: 1, cols: H, bits: 8)
            st.quant(m + "shared_expert.gate_proj", rows: sI, cols: H)
            st.quant(m + "shared_expert.up_proj", rows: sI, cols: H)
            st.quant(m + "shared_expert.down_proj", rows: H, cols: sI)
            st.quant(m + "switch_mlp.gate_proj", rows: eI, cols: H, experts: E)
            st.quant(m + "switch_mlp.up_proj", rows: eI, cols: H, experts: E)
            st.quant(m + "switch_mlp.down_proj", rows: H, cols: eI, experts: E)
            for k in ["gate", "shared_expert_gate"] { q8[m + k] = ["group_size": 64, "bits": 8] }
        } else {
            st.quant(m + "gate_proj", rows: denseI, cols: H)
            st.quant(m + "up_proj", rows: denseI, cols: H)
            st.quant(m + "down_proj", rows: H, cols: denseI)
        }
    }
    st.quant(root + "embed_tokens", rows: ToySpec.vocab, cols: H)
    st.float(root + "norm.weight", .bf16, [H], mlxNorm(1, 0.1))
    st.quant("language_model.lm_head", rows: ToySpec.vocab, cols: H, gain: 2, boost: ToySpec.lmBoost)
    try st.write(to: dir.appendingPathComponent("model.safetensors"))

    var quant: [String: Any] = ["group_size": st.group4, "bits": 4, "mode": "affine"]
    quant.merge(q8) { a, _ in a }
    try writeJSON(ToyConfig.qwen(d, quant: quant, seed: Int(seed)), dir.appendingPathComponent("config.json"))
}

/// DFlash2 drafter (bf16; the engine quantizes it to 4 bits on load): 1 layer, context of the model's 4 layers, with
/// the model's hidden (Drafter requires it) and intermediate of 3 × hidden (that of the real Qwen3.6 and Flash-Next).
/// `dflash`: keys that replace or extend the dflash_config (toy-muse's block 16 and logits post-processing).
private func writeToyDrafter(
    _ dir: URL, arch: ToyArch, hidden H: Int, seed: UInt64, dflash: [String: Any] = [:]
) throws {
    let inter = 3 * H
    let targets = Array(0..<ToySpec.layers)
    let st = ToySafetensors(seed: seed ^ 0xD4AF)
    func w(_ n: String, _ rows: Int, _ cols: Int, gain: Float = 2) {
        let sd = gain / Float(cols).squareRoot()
        st.float(n, .bf16, [rows, cols]) { sd * $0.normal() }
    }
    w("candidate_selector.hidden_projection.weight", 256, H, gain: 1)
    st.float("candidate_selector.predecessor_codebook", .bf16, [ToySpec.vocab, 256]) { 1.2 * $0.normal() }
    st.float("candidate_selector.successor_codebook", .bf16, [ToySpec.vocab, 256]) { 0.06 * $0.normal() }
    w("fc.weight", H, targets.count * H)
    st.float("hidden_norm.weight", .bf16, [H], mlxNorm(1, 0.02))
    st.float("norm.weight", .bf16, [H], mlxNorm(1.5, 0.1))
    let p = "layers.0."
    for c in ["attention_conv", "mlp_conv"] {
        st.float(p + c + ".base_kernel", .bf16, [2, 2, H]) { 0.5 + 0.5 * $0.normal() }
        st.float(p + c + ".kernel_projection.weight", .bf16, [H / 4, H]) { 0.015 * $0.normal() }
    }
    st.float(p + "input_layernorm.weight", .bf16, [H], mlxNorm(0.95, 0.03))
    st.float(p + "post_attention_layernorm.weight", .bf16, [H], mlxNorm(1, 0.02))
    w(p + "mlp.gate_proj.weight", inter, H)
    w(p + "mlp.up_proj.weight", inter, H)
    w(p + "mlp.down_proj.weight", H, inter)
    st.float(p + "self_attn.q_norm.weight", .bf16, [128], mlxNorm(1.1, 0.07))
    st.float(p + "self_attn.k_norm.weight", .bf16, [128], mlxNorm(1.1, 0.07))
    w(p + "self_attn.q_proj.weight", 4096, H)
    w(p + "self_attn.k_proj.weight", 1024, H)
    w(p + "self_attn.v_proj.weight", 1024, H)
    w(p + "self_attn.o_proj.weight", H, 4096)
    try st.write(to: dir.appendingPathComponent("model.safetensors"))
    let dflashConfig: [String: Any] = [
        "block_size": 8, "conv_group_size": 16, "conv_kernel_size": 2,
        "mask_token_id": ToySpec.tok(arch == .dense ? 248_070 : 248_077), "selector_rank": 256,
        "selector_top_k": 16, "target_layer_ids": targets,
    ]
    try writeJSON(
        [
            "architectures": ["DFlash2DraftModel"], "model_type": "qwen3", "attention_bias": false,
            "dflash_config": dflashConfig.merging(dflash) { _, new in new },
            "eos_token_id": ToySpec.tok(248_044), "head_dim": 128, "hidden_act": "silu", "hidden_size": H,
            "intermediate_size": inter, "is_causal": false, "layer_types": ["sliding_attention"],
            "max_position_embeddings": 262144, "max_window_layers": 1, "num_attention_heads": 32,
            "num_hidden_layers": 1, "num_key_value_heads": 8, "num_target_layers": ToySpec.layers,
            "rms_norm_eps": 1e-6,
            "rope_parameters": ["rope_theta": arch == .flash ? 10000 : 10_000_000, "rope_type": "default"]
                as [String: Any],
            "sliding_window": 2048, "tie_word_embeddings": false, "use_sliding_window": true,
            "vocab_size": ToySpec.vocab,
        ], dir.appendingPathComponent("config.json"))
}

/// DFlash v1 drafter as z-lab's `gemma-4-26B-A4B-it-DFlash` (bf16; quantized to 4 bits on load): the qwen3 layers with
/// the big model's context but no dynamic conv nor selector, so the engine runs it with block 8 (of the 16 its root
/// `block_size` says), its 2048 window and the block embedding × √hidden (Drafter.swift). 2 layers (sliding then
/// full, as the real one's last two), context of the toy's 4 layers, intermediate 2 × hidden as the real one, and the
/// real config's keys at the root (`block_size`, `final_logit_softcapping`; no `is_causal`).
private func writeToyDrafterV1(_ dir: URL, hidden H: Int, seed: UInt64) throws {
    let inter = 2 * H, layerTypes = ["sliding_attention", "full_attention"]
    let targets = Array(0..<ToySpec.layers)
    let st = ToySafetensors(seed: seed ^ 0xD4F1)
    func w(_ n: String, _ rows: Int, _ cols: Int, gain: Float = 2) {
        let sd = gain / Float(cols).squareRoot()
        st.float(n, .bf16, [rows, cols]) { sd * $0.normal() }
    }
    w("fc.weight", H, targets.count * H)
    st.float("hidden_norm.weight", .bf16, [H], mlxNorm(1, 0.02))
    st.float("norm.weight", .bf16, [H], mlxNorm(1.5, 0.1))
    for i in layerTypes.indices {
        let p = "layers.\(i)."
        st.float(p + "input_layernorm.weight", .bf16, [H], mlxNorm(0.95, 0.03))
        st.float(p + "post_attention_layernorm.weight", .bf16, [H], mlxNorm(1, 0.02))
        w(p + "mlp.gate_proj.weight", inter, H)
        w(p + "mlp.up_proj.weight", inter, H)
        w(p + "mlp.down_proj.weight", H, inter)
        st.float(p + "self_attn.q_norm.weight", .bf16, [128], mlxNorm(1.1, 0.07))
        st.float(p + "self_attn.k_norm.weight", .bf16, [128], mlxNorm(1.1, 0.07))
        w(p + "self_attn.q_proj.weight", 4096, H)
        w(p + "self_attn.k_proj.weight", 1024, H)
        w(p + "self_attn.v_proj.weight", 1024, H)
        w(p + "self_attn.o_proj.weight", H, 4096)
    }
    try st.write(to: dir.appendingPathComponent("model.safetensors"))
    try writeJSON(
        [
            "architectures": ["DFlashDraftModel"], "model_type": "qwen3", "attention_bias": false, "block_size": 16,
            "dflash_config": ["mask_token_id": ToySpec.tok(248_077), "target_layer_ids": targets] as [String: Any],
            "eos_token_id": ToySpec.tok(248_044), "final_logit_softcapping": 30.0, "head_dim": 128,
            "hidden_act": "silu", "hidden_size": H, "intermediate_size": inter, "layer_types": layerTypes,
            "max_position_embeddings": 262144, "max_window_layers": layerTypes.count, "num_attention_heads": 32,
            "num_hidden_layers": layerTypes.count, "num_key_value_heads": 8, "num_target_layers": ToySpec.layers,
            "rms_norm_eps": 1e-6, "rope_theta": 1_000_000, "sliding_window": 2048, "tie_word_embeddings": false,
            "use_sliding_window": true, "vocab_size": ToySpec.vocab,
        ], dir.appendingPathComponent("config.json"))
}
