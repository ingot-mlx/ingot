import Foundation

// Configs of the toy models of ingot-dev toymodels: each toy comes from the config.json
// of the real model (trimmed copy in ref/toy/real/<model>.json) applying only the cuts of `ToyCut`. It lives here and not in ingot-dev so the GPU-free test
// `ModelSpecTests.toyMatchesReal` compares the real ModelSpec with the toy's. ingot-dev writes the weights
// (Sources/ingot-dev/Toy) with these same dimensions (`ToyDims`).

/// Cuts of the toys against the real model:
/// - layers: 4, the pattern of one period of the real (Qwen and Flash-Next: 3 GDN + 1 attention; LFM2.5: conv, conv,
///   attention, conv);
/// - vocabulary 32768 (the toy tokenizer; the special ids are Qwen's shifted by `tok`);
/// - hidden of the Qwens 1024: the minimum that keeps the real paths (2048 and 5120): multiple of 128 (MoE guard,
///   moe_down2 in tiles of 128 columns) and of 1024 (the 256 Hadamard rotation of the --int8 W8A8,
///   `DenseI8Scratch.lb`). LFM2.5's and Flash-Next's is untouched (in LFM2 the head is hidden / heads; in Flash-Next
///   the HC, the PLE and the QSA fix it);
/// - expert intermediate 128 (multiple of 64 by MoEKernelSet and FN_I, with two K blocks of 64 in the down) and
///   the shared expert's in the same proportion as in the real one;
/// - dense FFN of the Qwens 9216 (> 8192 and multiple of 1024 like the 27B's 17408: tile of the int8 down GEMM,
///   encodeQmmI8, and rotation of 256) and of LFM2.5 2048.
public enum ToyCut {
    public static let vocab = 32768
    public static let layers = 4
    public static let qwenHidden = 1024
    public static let moeI = 128
    public static let qwenDenseI = 9216
    public static let lfmDenseI = 2048
    /// Id in the toy tokenizer for a Qwen3.6 id (the added ones 248044… at the end of the toy vocabulary).
    public static func tok(_ qwen: Int) -> Int { qwen - 248_320 + vocab }
}

/// Copy of the config.json of a real model (ref/toy/real/<name>.json).
public struct ToyReal {
    public let name: String
    public let json: [String: Any]
    /// text_config if it has one (Qwen3.5/3.6/3.8 and Flash-Next), else the whole config (LFM2.5, Granite, Gemma 3).
    public var text: [String: Any] { json["text_config"] as? [String: Any] ?? json }

    public init(name: String, json: [String: Any]) {
        self.name = name; self.json = json
    }

    /// Loads ref/toy/real/<name>.json under `root` (the repo root).
    public static func load(_ name: String, root: URL) throws -> ToyReal {
        let url = root.appendingPathComponent("ref/toy/real/\(name).json")
        guard let j = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any] else {
            throw CocoaError(.fileReadCorruptFile, userInfo: [NSFilePathErrorKey: url.path])
        }
        return ToyReal(name: name, json: j)
    }

    func int(_ k: String) -> Int {
        guard let v = text[k] as? Int else { preconditionFailure("\(name): \(k) missing in the real config") }
        return v
    }
}

/// Dimensions of a toy: the real one's with the `ToyCut` cuts (and the declared differences of the shape
/// toys). The ingot-dev weight writers and the configs below read them.
public struct ToyDims {
    public var real: ToyReal
    public var hidden: Int
    public var layerTypes: [String]
    // Attention.
    public var qh: Int, kvh: Int, hd: Int
    /// Partial RoPE (Qwen); nil if the real one has none.
    public var partialRotary: Double?
    // GDN (Qwen and Flash-Next).
    public var gdnHV = 0, gdnHK = 0, gdnDK = 0, gdnDV = 0
    // MoE (0 experts: dense).
    public var experts = 0, topK = 0, moeI = 0, sharedI = 0
    /// Dense FFN (the Qwen dense one and that of the first layers of LFM2.5).
    public var denseI = 0
    /// Dense layers at the start (LFM2.5 `num_dense_layers`).
    public var denseLayers = 0
    /// Gemma 4: head and kv heads of the full attention layers (`global_head_dim`, `num_global_key_value_heads`; 0 =
    /// those of the rest).
    public var globalHD = 0, globalKVH = 0

    /// Qwen3.5/3.6/3.8 (MoE or dense).
    public static func qwen(_ r: ToyReal) -> ToyDims {
        var d = ToyDims(
            real: r, hidden: ToyCut.qwenHidden, layerTypes: period(r), qh: r.int("num_attention_heads"),
            kvh: r.int("num_key_value_heads"), hd: r.int("head_dim"),
            partialRotary: r.text["partial_rotary_factor"] as? Double)
        d.gdn(r)
        if r.text["num_experts"] != nil {
            d.moe(r)
        } else {
            d.denseI = ToyCut.qwenDenseI
        }
        return d
    }

    /// Flash-Next (qwen4_exp): the real one's hidden.
    public static func flashNext(_ r: ToyReal) -> ToyDims {
        var d = ToyDims(
            real: r, hidden: r.int("hidden_size"), layerTypes: period(r), qh: r.int("num_attention_heads"),
            kvh: r.int("num_key_value_heads"), hd: r.int("head_dim"),
            partialRotary: r.text["partial_rotary_factor"] as? Double)
        d.gdn(r)
        d.moe(r)
        return d
    }

    /// LFM2.5 (lfm2_moe): the real one's hidden; the head is hidden / heads.
    public static func lfm(_ r: ToyReal) -> ToyDims {
        let h = r.int("hidden_size"), qh = r.int("num_attention_heads")
        var d = ToyDims(
            real: r, hidden: h, layerTypes: period(r), qh: qh, kvh: r.int("num_key_value_heads"), hd: h / qh,
            partialRotary: nil)
        d.moe(r)
        d.denseI = ToyCut.lfmDenseI
        d.denseLayers = r.int("num_dense_layers")
        return d
    }

    /// The first `ToyCut.layers` layers of the real one (one period of the pattern).
    static func period(_ r: ToyReal) -> [String] {
        guard let t = r.text["layer_types"] as? [String], t.count >= ToyCut.layers else {
            preconditionFailure("\(r.name): no layer_types")
        }
        return Array(t.prefix(ToyCut.layers))
    }

    mutating func gdn(_ r: ToyReal) {
        gdnHV = r.int("linear_num_value_heads"); gdnHK = r.int("linear_num_key_heads")
        gdnDK = r.int("linear_key_head_dim"); gdnDV = r.int("linear_value_head_dim")
    }

    mutating func moe(_ r: ToyReal) {
        experts = r.int("num_experts"); topK = r.int("num_experts_per_tok"); moeI = ToyCut.moeI
        if let s = r.text["shared_expert_intermediate_size"] as? Int {
            sharedI = s * ToyCut.moeI / r.int("moe_intermediate_size")
        }
    }

    /// Width of the GDN conv (q, k and v).
    public var gdnConv: Int { 2 * gdnHK * gdnDK + gdnHV * gdnDV }
}

/// Configs (config.json) of the toys: keys of the real one that do not change, plus those of `ToyDims`.
public enum ToyConfig {
    static func pick(_ src: [String: Any], _ keys: [String]) -> [String: Any] {
        var o: [String: Any] = [:]
        for k in keys {
            guard let v = src[k] else { preconditionFailure("\(k) missing in the real config") }
            o[k] = v
        }
        return o
    }

    /// text_config of Qwen3.5/3.6/3.8 (MoE or dense).
    public static func qwenText(_ d: ToyDims) -> [String: Any] {
        var t = pick(
            d.real.text,
            [
                "attention_bias", "attn_output_gate", "full_attention_interval", "hidden_act", "linear_conv_kernel_dim",
                "linear_key_head_dim", "linear_num_key_heads", "linear_value_head_dim", "mamba_ssm_dtype",
                "max_position_embeddings", "model_type", "rms_norm_eps", "rope_parameters", "tie_word_embeddings",
                "use_cache",
            ])
        t["bos_token_id"] = ToyCut.tok(248_044); t["eos_token_id"] = ToyCut.tok(248_044)
        t["hidden_size"] = d.hidden; t["layer_types"] = d.layerTypes; t["num_hidden_layers"] = d.layerTypes.count
        t["vocab_size"] = ToyCut.vocab
        t["head_dim"] = d.hd; t["num_attention_heads"] = d.qh; t["num_key_value_heads"] = d.kvh
        t["linear_num_value_heads"] = d.gdnHV
        if let p = d.partialRotary {
            t["partial_rotary_factor"] = p
            if var rp = t["rope_parameters"] as? [String: Any] {
                rp["partial_rotary_factor"] = p
                t["rope_parameters"] = rp
            }
        }
        if d.experts > 0 {
            t["moe_intermediate_size"] = d.moeI; t["num_experts"] = d.experts; t["num_experts_per_tok"] = d.topK
            t["shared_expert_intermediate_size"] = d.sharedI
        } else {
            t["intermediate_size"] = d.denseI
            t.merge(pick(d.real.text, ["output_gate_type"])) { a, _ in a }
        }
        return t
    }

    /// config.json of a Qwen; `quant`: the quantization (global and per tensor) that ingot-dev wrote.
    public static func qwen(_ d: ToyDims, quant: [String: Any], seed: Int) -> [String: Any] {
        var c = pick(d.real.json, ["architectures", "model_type", "tie_word_embeddings"])
        let moe = d.experts > 0
        c["text_config"] = qwenText(d)
        c["eos_token_id"] = [ToyCut.tok(248_046), ToyCut.tok(248_044)]
        c["quantization"] = quant; c["quantization_config"] = quant
        c["ingot_toy"] = ["arch": moe ? "qwen3_5_moe" : "qwen3_5", "seed": seed] as [String: Any]
        return c
    }

    /// text_config of Flash-Next (the fields the engine fixes at compile time are the real one's).
    public static func flashText(_ d: ToyDims, pleLayer: Int) -> [String: Any] {
        var t = pick(
            d.real.text,
            [
                "attention_bias", "full_attention_interval", "hc_count", "hc_lowrank", "heads_per_ngram",
                "hidden_act", "indexer_budget", "indexer_compress_ratio", "indexer_head_dim", "indexer_kv_heads",
                "indexer_n_heads", "linear_conv_kernel_dim", "linear_key_head_dim", "linear_num_key_heads",
                "linear_value_head_dim", "mamba_ssm_dtype", "max_position_embeddings", "model_type", "ngram_size",
                "output_gate_type", "partial_rotary_factor", "ple_conv_kernel_size", "rms_norm_eps",
                "rope_parameters", "tie_word_embeddings", "use_cache",
            ])
        t["bos_token_id"] = ToyCut.tok(248_044); t["eos_token_id"] = ToyCut.tok(248_044)
        t["hidden_size"] = d.hidden; t["layer_types"] = d.layerTypes; t["num_hidden_layers"] = d.layerTypes.count
        t["head_dim"] = d.hd; t["num_attention_heads"] = d.qh; t["num_key_value_heads"] = d.kvh
        t["linear_num_value_heads"] = d.gdnHV
        t["moe_intermediate_size"] = d.moeI; t["shared_expert_intermediate_size"] = d.sharedI
        t["num_experts"] = d.experts; t["num_experts_per_tok"] = d.topK
        // The PLE uses ple_embed_dim = hidden (the real one's); the layer counts from 1.
        t["ple_embed_dim"] = d.hidden; t["ple_layer_ids"] = [pleLayer + 1]
        return t
    }

    /// config.json of Flash-Next; `ingotBlock`: the ingot block (experts, PLE) that ingot-dev wrote.
    public static func flash(
        _ d: ToyDims, pleLayer: Int, quant: [String: Any], ingotBlock: [String: Any], seed: Int
    ) -> [String: Any] {
        var c = pick(d.real.json, ["architectures", "model_type", "tie_word_embeddings"])
        var t = flashText(d, pleLayer: pleLayer)
        t["vocab_size"] = ToyCut.vocab
        c["text_config"] = t
        c["eos_token_id"] = [ToyCut.tok(248_046), ToyCut.tok(248_044)]
        c["quantization"] = quant; c["quantization_config"] = quant; c["ingot"] = ingotBlock
        c["ingot_toy"] = ["arch": "qwen4_exp", "seed": seed] as [String: Any]
        return c
    }

    /// LFM2.5 config.json (toy-lfm2): the real config with the ToyCut cuts and no declared deviations.
    public static func lfm(_ d: ToyDims, quant: [String: Any], seed: Int) -> [String: Any] {
        var c = pick(
            d.real.json,
            [
                "architectures", "model_type", "norm_topk_prob", "use_expert_bias", "conv_L_cache", "conv_bias",
                "norm_eps", "tie_word_embeddings",
            ])
        c["hidden_size"] = d.hidden; c["num_hidden_layers"] = d.layerTypes.count; c["layer_types"] = d.layerTypes
        c["num_dense_layers"] = d.denseLayers
        c["num_attention_heads"] = d.qh; c["num_key_value_heads"] = d.kvh; c["intermediate_size"] = d.denseI
        c["moe_intermediate_size"] = d.moeI; c["num_experts"] = d.experts; c["num_experts_per_tok"] = d.topK
        for k in ["rope_parameters", "rope_theta", "max_position_embeddings"] { c[k] = d.real.json[k] }
        c["vocab_size"] = ToyCut.vocab; c["eos_token_id"] = ToyCut.tok(248_046)
        c["quantization"] = quant; c["quantization_config"] = quant
        c["ingot_toy"] = ["arch": "lfm2_moe", "seed": seed] as [String: Any]
        return c
    }
}

extension ToyDims {
    /// Bits of a tensor per layer per the real quantization (`suffix` after `layers.*.`; if not listed, the global).
    public func realBits(_ suffix: String) -> Int {
        guard let q = real.json["quantization"] as? [String: Any] else { return 4 }
        for (k, v) in q where k.hasSuffix("layers.*." + suffix) {
            if let b = (v as? [String: Any])?["bits"] as? Int { return b }
        }
        return q["bits"] as? Int ?? 4
    }

    /// Group size of the 4-bit matrices per the real global quantization (64 if it has none).
    public var realGroup4: Int { (real.json["quantization"] as? [String: Any])?["group_size"] as? Int ?? 64 }
}

// MARK: - Granite

extension ToyDims {
    /// Granite 4.1 (granite, granite-4.1-3b): attention only with a dense SwiGLU FFN; the real hidden (2560: the head
    /// is hidden / heads in mlx-lm, so cutting it would change the head), the real heads (40 over 8 KV heads, G = 5) and
    /// the real intermediate (8192, a multiple of 1024 like the Qwen cut).
    public static func granite(_ r: ToyReal) -> ToyDims {
        var d = ToyDims(
            real: r, hidden: r.int("hidden_size"), layerTypes: Array(repeating: "full_attention", count: ToyCut.layers),
            qh: r.int("num_attention_heads"), kvh: r.int("num_key_value_heads"),
            hd: r.int("hidden_size") / r.int("num_attention_heads"), partialRotary: nil)
        d.denseI = r.int("intermediate_size")
        return d
    }
}

extension ToyConfig {
    /// toy-granite differences with granite-4.1-3b that are not cuts: `attention_bias` true (the real one has
    /// none; the toy carries q/k/v/o biases so the q/k/v bias prework and the o_proj bias of the branch epilogue run).
    public static let graniteAttentionBias = true

    /// config.json of Granite (no text_config, like the real one): the real μP multipliers; `quant`: the
    /// quantization ingot-dev wrote (the real one's group size, ToyDims.realGroup4).
    public static func granite(_ d: ToyDims, quant: [String: Any], seed: Int) -> [String: Any] {
        var c = pick(
            d.real.json,
            [
                "architectures", "model_type", "attention_multiplier", "embedding_multiplier", "hidden_act",
                "logits_scaling", "max_position_embeddings", "mlp_bias", "residual_multiplier", "rms_norm_eps",
                "rope_scaling", "rope_theta", "tie_word_embeddings", "use_cache",
            ])
        c["attention_bias"] = graniteAttentionBias
        c["bos_token_id"] = ToyCut.tok(248_044); c["eos_token_id"] = ToyCut.tok(248_046)
        c["hidden_size"] = d.hidden; c["num_hidden_layers"] = d.layerTypes.count; c["vocab_size"] = ToyCut.vocab
        c["num_attention_heads"] = d.qh; c["num_key_value_heads"] = d.kvh; c["intermediate_size"] = d.denseI
        c["quantization"] = quant; c["quantization_config"] = quant
        c["ingot_toy"] = ["arch": "granite", "seed": seed] as [String: Any]
        return c
    }
}

// MARK: - Gemma 4

extension ToyDims {
    /// Gemma 4 (gemma4_text, gemma-4-26B-A4B-it), as toy-gemma4 v3: attention layers with the dense GeGLU MLP
    /// and the MoE in parallel, Qwen hidden (ToyCut.qwenHidden; the real 2816 is not a multiple of 1024: the --int8
    /// rotation), the real heads (16 q; sliding 8 kv × 256, full 2 kv × 512 with K = V), dense intermediate (2112),
    /// experts and top-k (128, top-8) with the expert intermediate cut (ToyCut.moeI). The layer pattern and the rest of
    /// the deviations are ToyConfig.gemma4Deviations.
    public static func gemma4(_ r: ToyReal) -> ToyDims {
        let p = ToyConfig.gemma4Deviations.pattern
        var d = ToyDims(
            real: r, hidden: ToyCut.qwenHidden,
            layerTypes: (0..<ToyCut.layers).map { ($0 + 1) % p == 0 ? "full_attention" : "sliding_attention" },
            qh: r.int("num_attention_heads"), kvh: r.int("num_key_value_heads"), hd: r.int("head_dim"),
            partialRotary: nil)
        d.globalHD = r.int("global_head_dim")
        d.globalKVH = r.int("num_global_key_value_heads")
        d.denseI = r.int("intermediate_size")
        d.experts = r.int("num_experts"); d.topK = r.int("top_k_experts"); d.moeI = ToyCut.moeI
        return d
    }

    /// Head and kv heads of layer `i` (the full layers' own ones in Gemma 4).
    public func attention(layer i: Int) -> (hd: Int, kvh: Int) {
        layerTypes[i] == "full_attention" && globalHD > 0 ? (globalHD, globalKVH) : (hd, kvh)
    }
}

extension ToyConfig {
    /// toy-gemma4 differences with gemma-4-26B-A4B-it that are not cuts: a full layer every 2 (so the 4 layers are
    /// sliding, full, sliding, full: two full layers, the real one has one every 6), a window of 300 (the real one,
    /// 1024; not a multiple of 128 and shorter than the 512 prompt), `final_logit_softcapping` 30.5 (the real one, 30:
    /// JSONSerialization writes 30.0 as 30 and transformers refuses an int there) and the E2B/E4B pieces the 26B lacks:
    /// per-layer inputs of the E2B's width (`hidden_size_per_layer_input` 256), the last 2 layers reading the KV of the
    /// first two (`num_kv_shared_layers` 2: layer 2 reads layer 0, sliding; layer 3 reads layer 1, full) and the
    /// double-wide MLP in those two (`use_double_wide_mlp`, as the E2B). The MoE (v3) is the real one with
    /// only the expert intermediate cut.
    public static let gemma4Deviations:
        (pattern: Int, window: Int, softcap: Double, ple: Int, kvShared: Int, doubleMLP: Bool) =
            (2, 300, 30.5, 256, 2, true)

    /// The keys of a Gemma 4 text config that carry the deviations (also applied to the real config by the toy test).
    public static func gemma4Deviate(_ t: inout [String: Any], layers: Int) {
        let d = gemma4Deviations
        t["layer_types"] = (0..<layers).map { ($0 + 1) % d.pattern == 0 ? "full_attention" : "sliding_attention" }
        t["sliding_window"] = d.window
        t["final_logit_softcapping"] = d.softcap
        t["hidden_size_per_layer_input"] = d.ple
        t["num_kv_shared_layers"] = d.kvShared
        t["use_double_wide_mlp"] = d.doubleMLP
    }

    /// config.json of toy-gemma4: flat gemma4_text (no text_config; mlx-lm loads gemma4_text directly) with the real
    /// text keys, the cuts and the deviations.
    public static func gemma4(_ d: ToyDims, quant: [String: Any], seed: Int) -> [String: Any] {
        var c = pick(
            d.real.text,
            [
                "model_type", "attention_bias", "attention_k_eq_v", "enable_moe_block", "hidden_activation",
                "hidden_size_per_layer_input",
                "max_position_embeddings", "num_kv_shared_layers", "rms_norm_eps", "rope_parameters",
                "tie_word_embeddings", "use_bidirectional_attention", "use_double_wide_mlp", "use_cache",
            ])
        c["architectures"] = ["Gemma4ForCausalLM"]
        c["bos_token_id"] = ToyCut.tok(248_044); c["eos_token_id"] = ToyCut.tok(248_046)
        c["hidden_size"] = d.hidden; c["num_hidden_layers"] = d.layerTypes.count; c["vocab_size"] = ToyCut.vocab
        c["vocab_size_per_layer_input"] = ToyCut.vocab
        c["num_attention_heads"] = d.qh; c["num_key_value_heads"] = d.kvh; c["head_dim"] = d.hd
        c["global_head_dim"] = d.globalHD; c["num_global_key_value_heads"] = d.globalKVH
        c["intermediate_size"] = d.denseI
        c["num_experts"] = d.experts; c["top_k_experts"] = d.topK; c["moe_intermediate_size"] = d.moeI
        gemma4Deviate(&c, layers: d.layerTypes.count)
        c["quantization"] = quant; c["quantization_config"] = quant
        c["ingot_toy"] = ["arch": "gemma4_text", "seed": seed] as [String: Any]
        return c
    }
}

// MARK: - gpt-oss

extension ToyCut {
    /// toy-gptoss hidden: 9 × 64, a multiple of 64 and not of 128 like the real 2880 (45 × 64), so the MoE down's shifted
    /// last tile (LAB_MOE_H64) and every other path with an odd number of 64-blocks run.
    public static let gptOssHidden = 576
    /// toy-gptoss expert intermediate: 3 × 64 (an odd number of 64-column tiles and K blocks, like the real 2880).
    public static let gptOssMoeI = 192
}

extension ToyDims {
    /// gpt-oss (gpt_oss, gpt-oss-20b): one period of the real layers (sliding, full, sliding, full), the real heads (64 q
    /// over 8 kv of 64), experts and top-k (32, top-4), hidden and expert intermediate cut (ToyCut.gptOssHidden,
    /// gptOssMoeI).
    public static func gptOss(_ r: ToyReal) -> ToyDims {
        var d = ToyDims(
            real: r, hidden: ToyCut.gptOssHidden, layerTypes: period(r), qh: r.int("num_attention_heads"),
            kvh: r.int("num_key_value_heads"), hd: r.int("head_dim"), partialRotary: nil)
        d.experts = r.int("num_local_experts"); d.topK = r.int("num_experts_per_tok"); d.moeI = ToyCut.gptOssMoeI
        return d
    }
}

extension ToyConfig {
    /// toy-gptoss's quantization default, the experts', as in the real MXFP4-Q8: mxfp4 (4 bits g32). Since v2
    /// the toy has no declared difference with gpt-oss-20b besides the cuts: window 128, flat YaRN with `truncate:
    /// false` in every layer, sinks, q/k/v/o biases, router and expert biases, swiglu_limit 7, MXFP4 experts, router,
    /// attention, embedding and lm_head at 8 bits g64, untied embedding (v1 had affine 4-bit g64 experts).
    public static var gptOssExpertQuant: [String: Any] { ["group_size": 32, "bits": 4, "mode": "mxfp4"] }

    /// config.json of toy-gptoss (flat, like the real one); `quant`: the quantization ingot-dev wrote.
    public static func gptOss(_ d: ToyDims, quant: [String: Any], seed: Int) -> [String: Any] {
        var c = pick(
            d.real.json,
            [
                "architectures", "model_type", "attention_bias", "head_dim", "hidden_act", "initial_context_length",
                "max_position_embeddings", "rms_norm_eps", "rope_scaling", "rope_theta", "sliding_window",
                "swiglu_limit", "tie_word_embeddings", "use_cache",
            ])
        c["bos_token_id"] = ToyCut.tok(248_044); c["eos_token_id"] = ToyCut.tok(248_046)
        c["pad_token_id"] = ToyCut.tok(248_044)
        c["hidden_size"] = d.hidden; c["intermediate_size"] = d.moeI
        c["num_hidden_layers"] = d.layerTypes.count; c["layer_types"] = d.layerTypes; c["vocab_size"] = ToyCut.vocab
        c["num_attention_heads"] = d.qh; c["num_key_value_heads"] = d.kvh
        c["num_local_experts"] = d.experts; c["num_experts_per_tok"] = d.topK; c["experts_per_token"] = d.topK
        c["quantization"] = quant; c["quantization_config"] = quant
        c["ingot_toy"] = ["arch": "gpt_oss", "seed": seed] as [String: Any]
        return c
    }
}

// MARK: - Nemotron-H

extension ToyCut {
    /// toy-nemotron hidden: 5 × 128, a multiple of 128 like the Lightning's 2688 (21 × 128).
    public static let nemotronHidden = 640
    /// toy-nemotron intermediates: experts 192 = 3 × 64 (a multiple of 64 but not of 128, like the Lightning's 1856 =
    /// 29 × 64), the shared expert twice that (as in the real one, 3712 = 2 × 1856), and the dense MLP (the Nano 4B's
    /// kind of layer: the Lightning has none) 1024.
    public static let nemotronMoeI = 192
    public static let nemotronDenseI = 1024
}

extension ToyDims {
    /// toy-nemotron v2 from Nemotron 3.5 Lightning: layers Mamba2, MoE, Mamba2, Mamba2, attention, MoE, MLP
    /// (`M E M M * E -`: Mamba2 followed by MoE, the `MM` pair, attention followed by MoE, and a dense relu² MLP at the
    /// end), each one block. The real Mamba2 (64 heads × 64, 8 groups, state 128, conv 4 with bias), attention (32 q / 2
    /// kv of 128, NoPE) and router (128 experts, top-6, sigmoid with selection bias, ×2.5); hidden and intermediates cut
    /// (ToyCut.nemotronHidden, nemotronMoeI, nemotronDenseI). Layer types: "mamba", "moe", "full_attention", "mlp".
    public static func nemotronH(_ r: ToyReal) -> ToyDims {
        var d = ToyDims(
            real: r, hidden: ToyCut.nemotronHidden,
            layerTypes: ["mamba", "moe", "mamba", "mamba", "full_attention", "moe", "mlp"],
            qh: r.int("num_attention_heads"), kvh: r.int("num_key_value_heads"), hd: r.int("head_dim"),
            partialRotary: nil)
        d.gdnHV = r.int("mamba_num_heads"); d.gdnHK = r.int("n_groups")
        d.gdnDK = r.int("ssm_state_size"); d.gdnDV = r.int("mamba_head_dim")
        d.experts = r.int("n_routed_experts"); d.topK = r.int("num_experts_per_tok")
        d.moeI = ToyCut.nemotronMoeI
        d.sharedI = ToyCut.nemotronMoeI * r.int("moe_shared_expert_intermediate_size") / r.int("moe_intermediate_size")
        d.denseI = ToyCut.nemotronDenseI
        return d
    }
}

extension ToyConfig {
    /// config.json of toy-nemotron v2 (flat, like the real one, with its `layers_block_type` spelling). Declared
    /// differences with the Lightning besides the cuts: a dense MLP layer (`-`, the Nano 4B's: the Lightning has none),
    /// so `intermediate_size` is the MLP's (the Lightning's equals its experts'); no MTP keys (`num_nextn_predict_layers`,
    /// `mtp_layers_block_type`) and no `moe_latent_size` (null in the real one).
    public static func nemotronH(_ d: ToyDims, quant: [String: Any], seed: Int) -> [String: Any] {
        var c = pick(
            d.real.json,
            [
                "architectures", "model_type", "attention_bias", "chunk_size", "conv_kernel", "expand", "head_dim",
                "layer_norm_epsilon", "mamba_head_dim", "mamba_hidden_act", "mamba_num_heads", "mamba_proj_bias",
                "mamba_ssm_cache_dtype", "max_position_embeddings", "mlp_bias", "mlp_hidden_act", "n_group", "n_groups",
                "n_routed_experts", "n_shared_experts", "norm_topk_prob", "num_attention_heads", "num_experts_per_tok",
                "num_key_value_heads", "residual_in_fp32", "routed_scaling_factor", "ssm_state_size",
                "tie_word_embeddings", "time_step_floor", "time_step_max", "time_step_min", "topk_group", "use_bias",
                "use_cache", "use_conv_bias",
            ])
        let names = ["mamba": "mamba", "full_attention": "attention", "moe": "moe", "mlp": "mlp"]
        c["layers_block_type"] = d.layerTypes.map { names[$0]! }
        c["intermediate_size"] = d.denseI; c["moe_intermediate_size"] = d.moeI
        c["moe_shared_expert_intermediate_size"] = d.sharedI
        c["bos_token_id"] = ToyCut.tok(248_044); c["eos_token_id"] = ToyCut.tok(248_046)
        c["pad_token_id"] = ToyCut.tok(248_044)
        c["hidden_size"] = d.hidden; c["num_hidden_layers"] = d.layerTypes.count; c["vocab_size"] = ToyCut.vocab
        c["quantization"] = quant; c["quantization_config"] = quant
        c["ingot_toy"] = ["arch": "nemotron_h", "seed": seed] as [String: Any]
        return c
    }
}

// MARK: - GLM-4.7-Flash

extension ToyDims {
    /// toy-glm from GLM-4.7-Flash (glm4_moe_lite): 4 layers, all MLA attention with the real shape (20 heads,
    /// q_lora_rank 768, kv_lora_rank 512, heads 192 + 64, v 256, interleaved RoPE θ 1e6), layer 0 dense SwiGLU and 3
    /// MoE of the real 64 experts top-4 (sigmoid router with selection bias, ×1.8) with the ungated shared expert. Hidden
    /// that of the Qwens (ToyCut.qwenHidden), expert and shared intermediate ToyCut.moeI, dense FFN ToyCut.qwenDenseI.
    public static func glm(_ r: ToyReal) -> ToyDims {
        let qh = r.int("num_attention_heads")
        var d = ToyDims(
            real: r, hidden: ToyCut.qwenHidden, layerTypes: Array(repeating: "full_attention", count: ToyCut.layers),
            qh: qh, kvh: qh, hd: r.int("qk_nope_head_dim") + r.int("qk_rope_head_dim"), partialRotary: nil)
        d.experts = r.int("n_routed_experts"); d.topK = r.int("num_experts_per_tok"); d.moeI = ToyCut.moeI
        d.sharedI = ToyCut.moeI * r.int("n_shared_experts")
        d.denseI = ToyCut.qwenDenseI
        d.denseLayers = r.int("first_k_dense_replace")
        return d
    }
}

extension ToyConfig {
    /// config.json of toy-glm (flat, like the real one). Declared difference with GLM-4.7-Flash besides the cuts: no
    /// `num_nextn_predict_layers` (the MTP layer, which the MLX conversions do not carry either).
    public static func glm(_ d: ToyDims, quant: [String: Any], seed: Int) -> [String: Any] {
        var c = pick(
            d.real.json,
            [
                "architectures", "model_type", "attention_bias", "attention_dropout", "first_k_dense_replace",
                "hidden_act", "kv_lora_rank", "max_position_embeddings", "n_group", "n_routed_experts",
                "n_shared_experts", "norm_topk_prob", "num_attention_heads", "num_experts_per_tok",
                "num_key_value_heads", "partial_rotary_factor", "q_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim",
                "rms_norm_eps", "rope_scaling", "rope_theta", "routed_scaling_factor", "tie_word_embeddings",
                "topk_group", "topk_method", "v_head_dim",
            ])
        c["intermediate_size"] = d.denseI; c["moe_intermediate_size"] = d.moeI
        c["bos_token_id"] = ToyCut.tok(248_044); c["eos_token_id"] = ToyCut.tok(248_046)
        c["pad_token_id"] = ToyCut.tok(248_044)
        c["hidden_size"] = d.hidden; c["num_hidden_layers"] = d.layerTypes.count; c["vocab_size"] = ToyCut.vocab
        c["quantization"] = quant; c["quantization_config"] = quant
        c["ingot_toy"] = ["arch": "glm4_moe_lite", "seed": seed] as [String: Any]
        return c
    }
}

// MARK: - Muse Glimmer

extension ToyDims {
    /// toy-muse from Muse Glimmer 30B (muse_glimmer): one period of the real layers (sliding, sliding,
    /// sliding, full), the real attention (32 q / 2 kv of 128, the separate gate_proj, weightless QK-norm with
    /// qk_scale_factor, RoPE θ 5e5 on the whole head in the sliding layers and NoPE in the full one from
    /// `layer_rope_theta`, the real window of 2048), dense SwiGLU; hidden and intermediate cut like the Qwens'
    /// (ToyCut.qwenHidden, ToyCut.qwenDenseI: the real 6656 and 19968 are not multiples of 1024).
    public static func muse(_ r: ToyReal) -> ToyDims {
        var d = ToyDims(
            real: r, hidden: ToyCut.qwenHidden, layerTypes: period(r), qh: r.int("num_attention_heads"),
            kvh: r.int("num_key_value_heads"), hd: r.int("head_dim"), partialRotary: nil)
        d.denseI = ToyCut.qwenDenseI
        return d
    }
}

extension ToyConfig {
    /// toy-muse's only difference with Muse Glimmer 30B besides the cuts: `final_logit_softcapping` 20.5 (the real one,
    /// 20: JSONSerialization writes 20.0 as 20 and transformers refuses an int there).
    public static let museSoftcap = 20.5

    /// config.json of toy-muse, nested like the real one (`muse_glimmer` with a `muse_glimmer_text` text_config: the
    /// engine and mlx-lm read both forms): the real text keys, the cuts, the per-layer θ of the kept layers and the
    /// softcap above.
    public static func muse(_ d: ToyDims, quant: [String: Any], seed: Int) -> [String: Any] {
        var t = pick(
            d.real.text,
            [
                "model_type", "attention_bias", "attention_dropout", "head_dim", "hidden_activation",
                "max_position_embeddings", "num_attention_heads", "num_key_value_heads", "output_multiplier",
                "post_norm_eps", "qk_scale_factor", "rms_norm_eps", "rope_parameters", "sliding_window",
                "tie_word_embeddings", "use_cache",
            ])
        guard let thetas = d.real.text["layer_rope_theta"] as? [Any] else {
            preconditionFailure("\(d.real.name): no layer_rope_theta")
        }
        t["layer_rope_theta"] = Array(thetas.prefix(d.layerTypes.count))
        t["layer_types"] = d.layerTypes; t["num_hidden_layers"] = d.layerTypes.count
        t["hidden_size"] = d.hidden; t["intermediate_size"] = d.denseI; t["vocab_size"] = ToyCut.vocab
        t["final_logit_softcapping"] = museSoftcap
        t["bos_token_id"] = ToyCut.tok(248_044); t["eos_token_id"] = ToyCut.tok(248_046)
        t["pad_token_id"] = ToyCut.tok(248_044)
        var c = pick(d.real.json, ["architectures", "model_type"])
        c["text_config"] = t
        c["eos_token_id"] = [ToyCut.tok(248_046), ToyCut.tok(248_044)]
        c["quantization"] = quant; c["quantization_config"] = quant
        c["ingot_toy"] = ["arch": "muse_glimmer", "seed": seed] as [String: Any]
        return c
    }
}

// MARK: - LFM3

extension ToyCut {
    /// toy-lfm3's first layer in the real LFM3-1.7B: its layers 1-4 (conv, full, conv, sliding) are one period with all
    /// three layer kinds; the real one starts conv, conv, full.
    public static let lfm3FirstLayer = 1
}

extension ToyDims {
    /// toy-lfm3 from LFM3-1.7B (lfm3): the real layers 1-4 (conv, full, conv, sliding: ToyCut.lfm3FirstLayer),
    /// the real hidden (2048), attention (16 q / 4 kv of 128 with the scalar gate per head, QK-norm, window 256),
    /// the 7-tap short convolution, the RoPE per layer kind (zero-tail partial 0.5 at θ 1e6 in the full layers, θ 1e4 on
    /// the whole head in the sliding ones) and the sandwich post-norms; the dense SwiGLU intermediate cut
    /// (ToyCut.lfmDenseI; the real 7680).
    public static func lfm3(_ r: ToyReal) -> ToyDims {
        guard let t = r.text["layer_types"] as? [String], t.count >= ToyCut.lfm3FirstLayer + ToyCut.layers else {
            preconditionFailure("\(r.name): no layer_types")
        }
        let first = ToyCut.lfm3FirstLayer
        var d = ToyDims(
            real: r, hidden: r.int("hidden_size"), layerTypes: Array(t[first..<first + ToyCut.layers]),
            qh: r.int("num_attention_heads"), kvh: r.int("num_key_value_heads"), hd: r.int("head_dim"),
            partialRotary: nil)
        d.denseI = ToyCut.lfmDenseI
        return d
    }
}

extension ToyConfig {
    /// Bits of every toy-lfm3 matrix (affine, g64), as in the real reference LiquidAI/LFM3-1.7B-MLX-8bit (its config in
    /// ref/toy/real/ is stored without the `quantization` key).
    public static let lfm3Bits = 8

    /// config.json of toy-lfm3 (flat, like the real one): the real keys, the cuts and no declared deviations.
    public static func lfm3(_ d: ToyDims, quant: [String: Any], seed: Int) -> [String: Any] {
        var c = pick(
            d.real.json,
            [
                "architectures", "model_type", "conv_L_cache", "conv_bias", "head_dim", "max_position_embeddings",
                "norm_eps", "rope_parameters", "sliding_window", "tie_word_embeddings", "use_sandwich_norm",
            ])
        c["hidden_size"] = d.hidden; c["num_hidden_layers"] = d.layerTypes.count; c["layer_types"] = d.layerTypes
        c["num_attention_heads"] = d.qh; c["num_key_value_heads"] = d.kvh; c["intermediate_size"] = d.denseI
        c["vocab_size"] = ToyCut.vocab
        c["bos_token_id"] = ToyCut.tok(248_044); c["eos_token_id"] = [ToyCut.tok(248_046)]
        c["pad_token_id"] = ToyCut.tok(248_044)
        c["quantization"] = quant; c["quantization_config"] = quant
        c["ingot_toy"] = ["arch": "lfm3", "seed": seed] as [String: Any]
        return c
    }
}
