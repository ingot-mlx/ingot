import Foundation
import LabBench

// toy-gptoss (gpt_oss, gpt-oss-20b): the real layer pattern and heads with the ToyCut cuts, MXFP4 experts like the real
// MXFP4-Q8 (v2; v1 had affine 4-bit experts). No drafter (gpt-oss has none in the engine).

/// toy-gptoss v2: sliding (window 128), full, sliding, full layers with flat YaRN (`truncate: false`), learned sinks,
/// q/k/v/o biases, a 32-expert top-4 MoE with router and expert biases and swiglu_oai (limit 7), x·w norms around 1,
/// MXFP4 experts (the quantization default, ToyConfig.gptOssExpertQuant; codes and e8m0 exponents written at random,
/// ToySafetensors.mxfp4) and router, attention, embedding and lm_head at 8 bits g64 (the router override without
/// `mode`), like the real MXFP4-Q8. Magnitudes, chosen by the mean KL of mlx-lm bf16 against fp32 on the 512 prompt (a sweep:
/// 0.026 with q/k at 1.2, gate/up at 2.5, down at 0.5 and the embedding at 16; 0.0003 with these): q and k at gain
/// 0.7 with biases of 0.5 (scale 1/8, YaRN mscale 1.35 on q and k); sinks around 2 so they take a real share; the v bias
/// small (0.05, the toy-granite lesson: a large one makes the attention branch almost constant); gate and up at gain
/// 1.25 with per-channel biases of 2, so a few channels cross swiglu_oai's clip at ±7; down at 0.25; the router at gain
/// 4 with a bias of 0.5 (ToySpec.routerGain); the embedding at gain 32 keeps the token against the branches.
func writeToyGptOss(_ dir: URL, seed: UInt64, _ d: ToyDims) throws {
    let st = ToySafetensors(seed: seed ^ 0x6055)
    let root = "model.", H = d.hidden, qd = d.qh * d.hd, kvd = d.kvh * d.hd, E = d.experts, I = d.moeI
    let q8: [String: Any] = ["group_size": 64, "bits": 8, "mode": "affine"]
    var quant: [String: Any] = ToyConfig.gptOssExpertQuant
    func norm(_ name: String, _ n: Int) { st.float(name, .bf16, [n]) { 1 + 0.05 * $0.normal() } }
    func vec(_ name: String, _ shape: [Int], _ sd: Float, _ mean: Float = 0) {
        st.float(name, .bf16, shape) { mean + sd * $0.normal() }
    }
    func lin8(_ name: String, rows: Int, cols: Int, gain: Float = 1, mode: Bool = true) {
        st.quant(name, rows: rows, cols: cols, bits: 8, gain: gain)
        quant[name] = mode ? q8 : ["group_size": 64, "bits": 8]
    }
    for i in 0..<ToySpec.layers {
        let p = root + "layers.\(i).", a = p + "self_attn.", m = p + "mlp."
        norm(p + "input_layernorm.weight", H)
        norm(p + "post_attention_layernorm.weight", H)
        lin8(a + "q_proj", rows: qd, cols: H, gain: 0.7)
        lin8(a + "k_proj", rows: kvd, cols: H, gain: 0.7)
        lin8(a + "v_proj", rows: kvd, cols: H)
        lin8(a + "o_proj", rows: H, cols: qd)
        vec(a + "q_proj.bias", [qd], 0.5)
        vec(a + "k_proj.bias", [kvd], 0.5)
        vec(a + "v_proj.bias", [kvd], 0.05)
        vec(a + "o_proj.bias", [H], 0.5)
        vec(a + "sinks", [d.qh], 1, 2)
        lin8(m + "router", rows: E, cols: H, gain: ToySpec.routerGain, mode: false)
        vec(m + "router.bias", [E], 0.5)
        st.mxfp4(m + "experts.gate_proj", rows: I, cols: H, experts: E, gain: 1.25)
        st.mxfp4(m + "experts.up_proj", rows: I, cols: H, experts: E, gain: 1.25)
        st.mxfp4(m + "experts.down_proj", rows: H, cols: I, experts: E, gain: 0.25)
        vec(m + "experts.gate_proj.bias", [E, I], 2)
        vec(m + "experts.up_proj.bias", [E, I], 2)
        vec(m + "experts.down_proj.bias", [E, H], 0.2)
    }
    lin8(root + "embed_tokens", rows: ToySpec.vocab, cols: H, gain: 32)
    norm(root + "norm.weight", H)
    lin8("lm_head", rows: ToySpec.vocab, cols: H, gain: 2)
    try st.write(to: dir.appendingPathComponent("model.safetensors"))
    let json = try JSONSerialization.data(
        withJSONObject: ToyConfig.gptOss(d, quant: quant, seed: Int(seed)), options: [.sortedKeys, .prettyPrinted])
    try floatKeys(String(decoding: json, as: UTF8.self)).write(
        to: dir.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
}

/// JSONSerialization writes 7.0 as 7, and transformers' strict GptOssConfig (which mlx-lm's tokenizer loader builds)
/// refuses an int in its float fields: the whole-number values of these keys get their `.0` back.
private func floatKeys(_ json: String) -> String {
    var s = json
    for k in ["swiglu_limit", "factor", "beta_fast", "beta_slow"] {
        s = s.replacingOccurrences(
            of: "(\"\(k)\" : )(-?[0-9]+)(,?\n)", with: "$1$2.0$3", options: .regularExpression)
    }
    return s
}
