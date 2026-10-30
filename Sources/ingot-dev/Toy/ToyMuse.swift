import Foundation
import IngotBench

// toy-muse (muse_glimmer, Muse Glimmer 30B): the real attention-only block with the ToyCut cuts (ToyDims.muse)
// and a toy DFlash2 drafter like the real one (block 16, draft logits × output_multiplier and softcap).

/// Rows of q_proj (`low`) or k_proj scaled up by 5 so the scores stay as calm as the other toys': q in the RoPE pairs
/// (i, i + hd/2) with i < hd/4 of each head, k in the rest. Muse's weightless QK-norm with qk_scale_factor 3.87 gives
/// random q and k scores of std 3.87 (3.87/√128 times a dot product of std √128), and that toy was chaotic: mlx-lm bf16
/// against fp32 already 0.037 of mean KL over the 512 prompt (0.0017 with the factor at 1). With q and k mostly in
/// disjoint pairs (the RoPE rotates within a pair, so it holds in every layer and position) the dot product shrinks by
/// 2r/(1 + r²) with r = 1/5, scores of std ~1.5, while the norms, the scale and the RoPE stay the real ones.
private func museQKPairs(rows: Int, hd: Int, low: Bool) -> (rows: Set<Int>, factor: Float) {
    (Set((0..<rows).filter { ($0 % hd) % (hd / 2) < hd / 4 ? low : !low }), 5)
}

/// toy-muse: 4 layers (sliding, sliding, sliding, full), named like the mlx-community 4bit (`language_model.model.*`
/// and `language_model.lm_head`, which mlx-lm's sanitize and the engine's layer prefix both read), all affine 4-bit
/// g64 like the real conversion, norms in bf16. Attention 32 q / 2 kv of 128 with q_proj, gate_proj, k_proj and v_proj
/// apart (the engine interleaves q and gate at load), q and k mostly in disjoint RoPE pairs (museQKPairs) and no
/// q_norm/k_norm tensors (weightless QK-norm). The four centered norms of each layer, applied as x·(1 + w), have weights
/// away from zero with a different mean each, so x·w in place of x·(1 + w), or one norm taken for another, changes the
/// branches; the final norm is plain x·w around 1. The
/// gate at gain 2 (σ of std-2 values: a wrong q|gate interleave shows), the embedding without gain (its weightless
/// RMSNorm undoes any), and the lm_head at gain 12 with ToySpec.lmBoost: logits of std ~2.4 after
/// output_multiplier 0.196, and the two boosted rows reach the softcap (20.5).
func writeToyMuse(_ dir: URL, seed: UInt64, _ d: ToyDims) throws {
    let st = ToySafetensors(seed: seed ^ 0x4D55)
    let root = "language_model.model.", H = d.hidden, qd = d.qh * d.hd, kvd = d.kvh * d.hd
    func norm(_ name: String, _ mean: Float, _ sd: Float) { st.float(name, .bf16, [H]) { mean + sd * $0.normal() } }
    for i in 0..<d.layerTypes.count {
        let p = root + "layers.\(i).", a = p + "self_attn.", m = p + "mlp."
        norm(p + "input_layernorm.weight", 0.1, 0.1)
        norm(p + "post_attention_layernorm.weight", 0.4, 0.1)
        norm(p + "pre_feedforward_layernorm.weight", -0.2, 0.1)
        norm(p + "post_feedforward_layernorm.weight", 0.25, 0.1)
        st.quant(a + "q_proj", rows: qd, cols: H, boost: museQKPairs(rows: qd, hd: d.hd, low: true))
        st.quant(a + "gate_proj", rows: qd, cols: H, gain: 2)
        st.quant(a + "k_proj", rows: kvd, cols: H, boost: museQKPairs(rows: kvd, hd: d.hd, low: false))
        st.quant(a + "v_proj", rows: kvd, cols: H)
        st.quant(a + "o_proj", rows: H, cols: qd)
        st.quant(m + "gate_proj", rows: d.denseI, cols: H)
        st.quant(m + "up_proj", rows: d.denseI, cols: H)
        st.quant(m + "down_proj", rows: H, cols: d.denseI)
    }
    st.quant(root + "embed_tokens", rows: ToySpec.vocab, cols: H)
    norm(root + "norm.weight", 1, 0.1)
    st.quant("language_model.lm_head", rows: ToySpec.vocab, cols: H, gain: 12, boost: ToySpec.lmBoost)
    try st.write(to: dir.appendingPathComponent("model.safetensors"))
    let quant: [String: Any] = ["group_size": 64, "bits": 4, "mode": "affine"]
    try JSONSerialization.data(
        withJSONObject: ToyConfig.muse(d, quant: quant, seed: Int(seed)), options: [.sortedKeys, .prettyPrinted]
    ).write(to: dir.appendingPathComponent("config.json"))
}
