import Foundation
import IngotBench

// toy-lfm3 (lfm3, LiquidAI/LFM3-1.7B): the real layers 1-4 (conv, full, conv, sliding) with the ToyCut cuts
// (ToyDims.lfm3), named like the official MLX conversion.

/// toy-lfm3: named like LiquidAI/LFM3-1.7B-MLX-8bit (`language_model.model.*`, which mlx-lm's sanitize and the engine's
/// layer prefix both read; tied embedding, no lm_head), every matrix affine 8-bit g64 like the real reference
/// (ToyConfig.lfm3Bits), norms and the conv weight in bf16. Each layer has its four norms (operator_norm, ffn_norm and
/// the sandwich post_operator_norm and post_ffn_norm, all x·w) with weights of a different mean each, so a norm taken
/// for another or a missing post-norm changes the branches. Conv layers: in_proj H → 3H (B|C|x), the 7-tap conv weight
/// [H, 7, 1] (MLX layout, tap 0 the oldest) and out_proj. Attention layers: q/k/v/o, QK-norm weights around 1.2 and the
/// scalar gate per head `gate_proj` [heads, H] at gain 2, so σ of each head's gate spreads over (0, 1) instead of
/// sitting at ~0.5 (a gate repeated into the wrong rows, or ignored, shows). Dense SwiGLU `feed_forward.w1/w3/w2`.
/// The embedding at gain 2 (logits of std ~2 through the tied head) and the final embedding_norm around 1.
func writeToyLFM3(_ dir: URL, seed: UInt64, _ d: ToyDims) throws {
    let st = ToySafetensors(seed: seed ^ 0x1F33)
    let root = "language_model.model.", H = d.hidden, bits = ToyConfig.lfm3Bits
    let convL = d.real.text["conv_L_cache"] as? Int ?? 3
    func norm(_ name: String, _ n: Int, _ mean: Float, _ sd: Float) {
        st.float(name, .bf16, [n]) { mean + sd * $0.normal() }
    }
    for i in 0..<d.layerTypes.count {
        let p = root + "layers.\(i)."
        norm(p + "operator_norm.weight", H, 1, 0.05)
        norm(p + "post_operator_norm.weight", H, 0.6, 0.1)
        norm(p + "ffn_norm.weight", H, 0.9, 0.05)
        norm(p + "post_ffn_norm.weight", H, 0.4, 0.1)
        if d.layerTypes[i] == "conv" {
            let c = p + "conv."
            st.quant(c + "in_proj", rows: 3 * H, cols: H, bits: bits)
            st.quant(c + "out_proj", rows: H, cols: H, bits: bits)
            st.float(c + "conv.weight", .bf16, [H, convL, 1]) { 0.4 * $0.normal() }
        } else {
            let a = p + "self_attn."
            st.quant(a + "q_proj", rows: d.qh * d.hd, cols: H, bits: bits)
            st.quant(a + "k_proj", rows: d.kvh * d.hd, cols: H, bits: bits)
            st.quant(a + "v_proj", rows: d.kvh * d.hd, cols: H, bits: bits)
            st.quant(a + "o_proj", rows: H, cols: d.qh * d.hd, bits: bits)
            st.quant(a + "gate_proj", rows: d.qh, cols: H, bits: bits, gain: 2)
            norm(a + "q_norm.weight", d.hd, 1.2, 0.1)
            norm(a + "k_norm.weight", d.hd, 1.2, 0.1)
        }
        let f = p + "feed_forward."
        st.quant(f + "w1", rows: d.denseI, cols: H, bits: bits)
        st.quant(f + "w3", rows: d.denseI, cols: H, bits: bits)
        st.quant(f + "w2", rows: H, cols: d.denseI, bits: bits)
    }
    st.quant(root + "embed_tokens", rows: ToySpec.vocab, cols: H, bits: bits, gain: 2)
    norm(root + "embedding_norm.weight", H, 1, 0.1)
    try st.write(to: dir.appendingPathComponent("model.safetensors"))
    let quant: [String: Any] = ["group_size": 64, "bits": bits, "mode": "affine"]
    try JSONSerialization.data(
        withJSONObject: ToyConfig.lfm3(d, quant: quant, seed: Int(seed)), options: [.sortedKeys, .prettyPrinted]
    ).write(to: dir.appendingPathComponent("config.json"))
}
