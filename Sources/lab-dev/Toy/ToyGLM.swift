import Foundation
import LabBench

// toy-glm (glm4_moe_lite, GLM-4.7-Flash): the real MLA shape with the ToyCut cuts (ToyDims.glm). No drafter.

/// toy-glm: 4 MLA attention layers, layer 0 with the dense SwiGLU and layers 1-3 with the MoE (64 experts top-4,
/// ungated shared expert), all affine 4-bit g64 like the mlx-community conversion, `embed_q` [20, 512, 192] and
/// `unembed_out` [20, 256, 512] included (mlx-lm's MultiLinear layout, which it loads without sanitize), except the
/// router, in bf16 without quantizing as in the real 4bit (the engine applies it in bf16 as well), and
/// e_score_correction_bias in fp32. The router at gain 2 with a selection bias of std 0.1 (which changes the choice),
/// the norms (plain x·w) around 1.
func writeToyGLM(_ dir: URL, seed: UInt64, _ d: ToyDims) throws {
    let st = ToySafetensors(seed: seed ^ 0x474C)
    let m = d.real.text
    func int(_ k: String) -> Int { (m[k] as? NSNumber)!.intValue }
    let root = "model.", H = d.hidden, QH = d.qh, E = d.experts
    let qRank = int("q_lora_rank"), kvRank = int("kv_lora_rank"), rope = int("qk_rope_head_dim")
    let nope = int("qk_nope_head_dim"), vHD = int("v_head_dim")
    func vec(_ name: String, _ shape: [Int], _ gen: @escaping (inout ToyRNG) -> Float) {
        st.float(name, .bf16, shape, gen)
    }
    for i in 0..<d.layerTypes.count {
        let p = root + "layers.\(i).", a = p + "self_attn.", f = p + "mlp."
        vec(p + "input_layernorm.weight", [H]) { 1 + 0.05 * $0.normal() }
        vec(p + "post_attention_layernorm.weight", [H]) { 1 + 0.05 * $0.normal() }
        st.quant(a + "q_a_proj", rows: qRank, cols: H)
        vec(a + "q_a_layernorm.weight", [qRank]) { 1 + 0.05 * $0.normal() }
        st.quant(a + "q_b_proj", rows: QH * (nope + rope), cols: qRank)
        st.quant(a + "kv_a_proj_with_mqa", rows: kvRank + rope, cols: H)
        vec(a + "kv_a_layernorm.weight", [kvRank]) { 1 + 0.05 * $0.normal() }
        st.quant(a + "embed_q", rows: kvRank, cols: nope, experts: QH)
        st.quant(a + "unembed_out", rows: vHD, cols: kvRank, experts: QH)
        st.quant(a + "o_proj", rows: H, cols: QH * vHD)
        if i < d.denseLayers {
            st.quant(f + "gate_proj", rows: d.denseI, cols: H)
            st.quant(f + "up_proj", rows: d.denseI, cols: H)
            st.quant(f + "down_proj", rows: H, cols: d.denseI, gain: 0.5)
        } else {
            vec(f + "gate.weight", [E, H]) { 2 / Float(H).squareRoot() * $0.normal() }
            st.float(f + "gate.e_score_correction_bias", .f32, [E]) { 0.1 * $0.normal() }
            st.quant(f + "switch_mlp.gate_proj", rows: d.moeI, cols: H, experts: E)
            st.quant(f + "switch_mlp.up_proj", rows: d.moeI, cols: H, experts: E)
            st.quant(f + "switch_mlp.down_proj", rows: H, cols: d.moeI, experts: E, gain: 0.5)
            st.quant(f + "shared_experts.gate_proj", rows: d.sharedI, cols: H)
            st.quant(f + "shared_experts.up_proj", rows: d.sharedI, cols: H)
            st.quant(f + "shared_experts.down_proj", rows: H, cols: d.sharedI, gain: 0.5)
        }
    }
    st.quant(root + "embed_tokens", rows: ToySpec.vocab, cols: H, gain: 16)
    vec(root + "norm.weight", [H]) { 1 + 0.1 * $0.normal() }
    st.quant("lm_head", rows: ToySpec.vocab, cols: H, gain: 2)
    try st.write(to: dir.appendingPathComponent("model.safetensors"))
    let quant: [String: Any] = ["group_size": 64, "bits": 4, "mode": "affine"]
    try JSONSerialization.data(
        withJSONObject: ToyConfig.glm(d, quant: quant, seed: Int(seed)), options: [.sortedKeys, .prettyPrinted]
    ).write(to: dir.appendingPathComponent("config.json"))
}
