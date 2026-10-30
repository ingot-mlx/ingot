import Foundation
import IngotBench

// Toy LFM2.5 (lfm2_moe, like LiquidAI/LFM2.5-24B-A2B converted to MLX): hidden 2048, short-convolution blocks
// (in_proj D → 3D, 3-tap conv) and attention blocks with ungated 64-wide heads (32 q × 8 KV, q_layernorm and
// k_layernorm, out_proj); layers 0 and 1 with a dense FFN (num_dense_layers) and the rest 64-expert top-4 MoE with an
// 8-bit sigmoid router and fp32 expert_bias, no shared expert; 6-bit tied embedding that doubles as lm_head. No
// drafter (nor the lm_head row boost: with the tied embedding, the model repeats those tokens). Layers: conv,
// conv, attention, conv (the real model's pattern).

// Dimensions of the real model (ref/toy/real/lfm2.5-24b-a2b.json) with the ToyCut cuts (ToyDims.lfm): hidden 2048 as
// in the real model (in LFM2 the head size is hidden / heads, ModelDims), ToyCut.moeI experts, ToyCut.lfmDenseI FFN.

func writeToyLFM(_ dir: URL, seed: UInt64, _ d: ToyDims) throws {
    let lfmH = d.hidden, lfmQH = d.qh, lfmKVH = d.kvh, lfmHD = d.hd, lfmE = d.experts, lfmI = d.moeI
    let lfmDenseI = d.denseI
    let st = ToySafetensors(seed: seed ^ 0x1F32)
    let root = "model."
    var over: [String: Any] = [:]
    func norm(_ name: String, _ n: Int, _ mean: Float = 1, _ sd: Float = 0.05) {
        st.float(name, .bf16, [n]) { mean + sd * $0.normal() }
    }
    for i in 0..<ToySpec.layers {
        let p = root + "layers.\(i)."
        norm(p + "operator_norm.weight", lfmH)
        norm(p + "ffn_norm.weight", lfmH)
        if d.layerTypes[i] == "full_attention" {
            let a = p + "self_attn."
            st.quant(a + "q_proj", rows: lfmQH * lfmHD, cols: lfmH)
            st.quant(a + "k_proj", rows: lfmKVH * lfmHD, cols: lfmH)
            st.quant(a + "v_proj", rows: lfmKVH * lfmHD, cols: lfmH)
            st.quant(a + "out_proj", rows: lfmH, cols: lfmQH * lfmHD)
            norm(a + "q_layernorm.weight", lfmHD, 1.2, 0.1)
            norm(a + "k_layernorm.weight", lfmHD, 1.2, 0.1)
        } else {
            let c = p + "conv."
            st.quant(c + "in_proj", rows: 3 * lfmH, cols: lfmH)
            st.quant(c + "out_proj", rows: lfmH, cols: lfmH)
            st.float(c + "conv.weight", .bf16, [lfmH, 3, 1]) { 0.4 * $0.normal() }
        }
        let f = p + "feed_forward."
        if i < d.denseLayers {
            st.quant(f + "gate_proj", rows: lfmDenseI, cols: lfmH)
            st.quant(f + "up_proj", rows: lfmDenseI, cols: lfmH)
            st.quant(f + "down_proj", rows: lfmH, cols: lfmDenseI)
        } else {
            // Router with the real model's bits (8 in the 24B, 4 in the 8B).
            let rb = d.realBits("feed_forward.gate")
            st.quant(f + "gate", rows: lfmE, cols: lfmH, bits: rb, gain: 2)
            if rb != 4 { over[f + "gate"] = ["group_size": 64, "bits": rb] }
            st.float(f + "expert_bias", .f32, [lfmE]) { 0.1 * $0.normal() }
            st.quant(f + "switch_mlp.gate_proj", rows: lfmI, cols: lfmH, experts: lfmE)
            st.quant(f + "switch_mlp.up_proj", rows: lfmI, cols: lfmH, experts: lfmE)
            st.quant(f + "switch_mlp.down_proj", rows: lfmH, cols: lfmI, experts: lfmE)
        }
    }
    st.quant(root + "embed_tokens", rows: ToySpec.vocab, cols: lfmH, bits: 6, gain: 2)
    over[root + "embed_tokens"] = ["group_size": 64, "bits": 6]
    norm(root + "embedding_norm.weight", lfmH, 1, 0.1)
    try st.write(to: dir.appendingPathComponent("model.safetensors"))

    var quant: [String: Any] = ["group_size": 64, "bits": 4, "mode": "affine"]
    quant.merge(over) { a, _ in a }
    try JSONSerialization.data(
        withJSONObject: ToyConfig.lfm(d, quant: quant, seed: Int(seed)),
        options: [.sortedKeys, .prettyPrinted]
    ).write(to: dir.appendingPathComponent("config.json"))
}
