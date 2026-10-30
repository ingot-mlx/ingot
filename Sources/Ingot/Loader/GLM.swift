import Foundation

// GLM-4.7-Flash (glm4_moe_lite) in the loader: the MLA shape, its weight names, the row order that moves its
// interleaved RoPE to the engine's convention, and the named load errors of what the engine does not run. Kept apart
// from ModelSpec.swift, GPU.swift and Config.swift, which only call into it.

/// Multi-head latent attention as mlx-lm's glm4_moe_lite builds it (`Glm4MoeLiteAttention`):
/// - the query is `q_b_proj(q_a_layernorm(q_a_proj(x)))`, `heads` heads of `qk_nope_head_dim + qk_rope_head_dim`, each
///   head laid out as `[q_nope | q_pe]`;
/// - `kv_a_proj_with_mqa(x)` gives the latent `c` (`kv_lora_rank`, through `kv_a_layernorm`) and one `k_pe`
///   (`qk_rope_head_dim`) shared by every head;
/// - the conversions split the original `kv_b_proj` into two per-head `MultiLinear`: `embed_q` [heads, kv_lora_rank,
///   qk_nope_head_dim] (k_nope = c·embed_qᵀ per head, or q_lat = embed_q·q_nope) and `unembed_out` [heads, v_head_dim,
///   kv_lora_rank] (v = unembed_out·c);
/// - the RoPE rotates only `q_pe` and `k_pe`, interleaved (`traditional=True`: pairs (2i, 2i + 1)), with θ `rope_theta`.
///
/// The engine's attention rotates the first `AT_ROT` dimensions of a head by halves (pairs (i, i + AT_ROT / 2)), so the
/// loader reorders output rows (`qBRowOrder`, `kvARowOrder`): exact, as whole rows move with their scales and biases, and
/// q·k does not change when q and k are permuted alike. The frequencies match: pair i has θ^(−2i / qk_rope_head_dim) in
/// both conventions.
public struct MLASpec: Sendable, Equatable {
    public let heads: Int
    /// `q_lora_rank`: nil is a direct `q_proj` (DeepSeek-V2-Lite), which the engine does not run (`ModelSpec.glmGaps`).
    public let qLoraRank: Int?
    public let kvLoraRank: Int
    public let qkNopeHeadDim: Int, qkRopeHeadDim: Int, vHeadDim: Int

    /// The MLA keys of a config (`text`); nil without `kv_lora_rank`, `qk_nope_head_dim`, `qk_rope_head_dim`,
    /// `v_head_dim` or `num_attention_heads`.
    public init?(_ t: [String: Any]) {
        func int(_ k: String) -> Int? { (t[k] as? NSNumber)?.intValue }
        guard let h = int("num_attention_heads"), let kv = int("kv_lora_rank"), let nope = int("qk_nope_head_dim"),
            let rope = int("qk_rope_head_dim"), let v = int("v_head_dim")
        else { return nil }
        heads = h; qLoraRank = int("q_lora_rank"); kvLoraRank = kv
        qkNopeHeadDim = nope; qkRopeHeadDim = rope; vHeadDim = v
    }

    /// Width of a q (and k) head: `qk_nope_head_dim + qk_rope_head_dim` (256 in GLM-4.7-Flash; attention scale 1/16).
    public var qHeadDim: Int { qkNopeHeadDim + qkRopeHeadDim }
    /// Head of the latent cache (absorbed form, Layers/MLA.swift): `[k_pe | c]`, qk_rope_head_dim + kv_lora_rank (576 in
    /// GLM-4.7-Flash). It is the attention head of the kernels (one KV head read by every query head).
    public var latentHeadDim: Int { qkRopeHeadDim + kvLoraRank }
    /// Output rows of `kv_a_proj_with_mqa`: the latent `c` and then `k_pe`.
    public var kvAWidth: Int { kvLoraRank + qkRopeHeadDim }

    /// Source row of each output row when an interleaved block of `n` rotary dimensions becomes one rotated by halves:
    /// the even dimensions first, then the odd ones (row j < n/2 takes 2j; row n/2 + j takes 2j + 1).
    public static func deinterleave(_ n: Int) -> [Int] {
        precondition(n % 2 == 0)
        return (0..<n).map { $0 < n / 2 ? 2 * $0 : 2 * ($0 - n / 2) + 1 }
    }

    /// Row order of `q_b_proj` for the engine (new row r takes row `qBRowOrder[r]` of the checkpoint): each head as
    /// `[q_pe de-interleaved | q_nope]`, so the prework rotates its first `qk_rope_head_dim` dimensions by halves.
    public var qBRowOrder: [Int] {
        let d = qHeadDim, pe = Self.deinterleave(qkRopeHeadDim).map { qkNopeHeadDim + $0 }
        return (0..<heads).flatMap { h in (pe + Array(0..<qkNopeHeadDim)).map { h * d + $0 } }
    }

    /// Row order of `kv_a_proj_with_mqa`: the latent `c` as is and `k_pe` de-interleaved (the k head is assembled as
    /// `[k_pe | k_nope]` from it, like the q head).
    public var kvARowOrder: [Int] {
        Array(0..<kvLoraRank) + Self.deinterleave(qkRopeHeadDim).map { kvLoraRank + $0 }
    }
}

/// Weight names of the MLA projections inside the attention block (`self_attn.`); `o_proj` is WeightNames.oProj.
public struct MLANames: Sendable, Equatable {
    public var qA = "q_a_proj", qANorm = "q_a_layernorm.weight", qB = "q_b_proj"
    public var kvA = "kv_a_proj_with_mqa", kvANorm = "kv_a_layernorm.weight"
    /// The per-head MultiLinear halves of `kv_b_proj` (3D quantized tensors, `QuantizedLinear(multiLinear:)`).
    public var embedQ = "embed_q", unembedOut = "unembed_out"
}

extension WeightNames {
    /// GLM-4.7-Flash: Qwen's names (norms, `mlp.`, `mlp.gate` with its `e_score_correction_bias`, `switch_mlp.`, dense
    /// `mlp.{gate,up,down}_proj` in the first layers), the ungated `shared_experts.` and the MLA projections.
    static let glm: WeightNames = {
        var n = WeightNames.qwen
        n.sharedExpert = "shared_experts."
        n.mla = MLANames()
        return n
    }()
}

extension ModelSpec {
    /// GLM-4.7-Flash pieces the engine does not run: what is not supported (a direct
    /// `q_proj`, DeepSeek's YaRN attention scale, MoE layers every n-th layer, a latent head other than 64 + 512, biases)
    /// with its name. The router (`n_group` > 1 is already an error in `unsupported`), the dense first layers and the
    /// shared expert are those of the other families. The MTP layer (`num_nextn_predict_layers`, `layers.47.` in the
    /// original) is not an error: the MLX conversions do not carry it (mlx-lm's sanitize drops it).
    static func glmGaps(_ cfg: ModelConfig) -> [String] {
        guard cfg.family == .glm4MoeLite else { return [] }
        let t = cfg.text
        guard let m = MLASpec(t) else {
            return [
                "glm4_moe_lite without its MLA keys (kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim, v_head_dim)"
            ]
        }
        var out: [String] = []
        if m.qLoraRank == nil {
            out.append("MLA without q_lora_rank (a direct q_proj, DeepSeek-V2-Lite: not supported)")
        }
        if let rs = t["rope_scaling"] as? [String: Any], ((rs["mscale_all_dim"] as? NSNumber)?.doubleValue ?? 0) != 0 {
            out.append("MLA RoPE scaling with mscale_all_dim (DeepSeek's YaRN attention scale: not supported)")
        }
        if let f = (t["moe_layer_freq"] as? NSNumber)?.intValue, f != 1 {
            out.append("moe_layer_freq \(f) (dense layers between MoE ones)")
        }
        // The latent cache is one attention head of [k_pe | c]: the kernels have that head only as 64 ⊕ 512 (the
        // block Hadamard of the int8 K, at_fwht). embed_q and unembed_out run as per-head matmuls (K = nope and
        // kv_lora_rank in groups of 64; N = kv_lora_rank and v_head_dim in tiles of 64).
        if m.qkRopeHeadDim != 64 || m.kvLoraRank != 512 {
            out.append(
                "MLA latent head of \(m.qkRopeHeadDim) + \(m.kvLoraRank) (the attention kernels have 64 + 512 only)")
        }
        if m.qkNopeHeadDim <= 0 || m.qkNopeHeadDim % 64 != 0 || m.vHeadDim <= 0 || m.vHeadDim % 64 != 0 {
            out.append("MLA heads of nope \(m.qkNopeHeadDim) and v \(m.vHeadDim) (multiples of 64)")
        }
        if m.kvLoraRank % 64 != 0 || (m.qLoraRank ?? 0) % 64 != 0 {
            out.append("MLA ranks q \(m.qLoraRank ?? 0) / kv \(m.kvLoraRank) (multiples of 64)")
        }
        if (t["attention_bias"] as? Bool) == true { out.append("MLA projection biases (attention_bias)") }
        return out
    }
}
