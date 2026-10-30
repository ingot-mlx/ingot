import Foundation
import IngotBench

// Dense attention-only toys: the benches of the block pieces (BlockSpec) that mlx-lm only runs in its own
// model types. Dimensions from ref/toy/real/<model>.json with the cuts of ToyCut and their declared differences
// (ToyConfig). No drafter (the real ones have none).

private func writeConfig(_ obj: [String: Any], _ dir: URL) throws {
    try JSONSerialization.data(withJSONObject: obj, options: [.sortedKeys, .prettyPrinted])
        .write(to: dir.appendingPathComponent("config.json"))
}

/// toy-granite (granite, granite-4.1-3b): 4 attention layers without QK-norm (40 q heads over 8 KV heads, G = 5),
/// q/k/v/o biases (ToyConfig.graniteAttentionBias), dense SwiGLU FFN, tied embedding of 4 bits, every 4-bit matrix in
/// the real one's groups (32, like mlx-community's conversions) and the real μP multipliers. With a
/// tied embedding the residual must move away from the embedding row, or the head picks the input token again (an
/// own logit ~√hidden times the others): o_proj and down_proj at gain 256 make the branches dominate after
/// `residual_multiplier` 0.22; q and k at 4 give scores of std ~2 with `attention_multiplier` 1/64; the embedding at
/// 20 gives logits of std ~2 after `logits_scaling` 10. The v bias stays small (0.05): it survives the attention
/// average, and at 0.5 the o_proj gain turned it into a branch almost equal for every token, so the residual fell into
/// a fixed point and the greedy repeated one token (reuse never reached N+1; with 0.05, 73 and 19 distinct
/// tokens in the 128 of the 512 and 4K goldens).
func writeToyGranite(_ dir: URL, seed: UInt64, _ d: ToyDims) throws {
    let st = ToySafetensors(seed: seed ^ 0x6A41)
    st.group4 = d.realGroup4
    let root = "model.", H = d.hidden, qd = d.qh * d.hd, kvd = d.kvh * d.hd
    func norm(_ name: String, _ n: Int, _ mean: Float = 1, _ sd: Float = 0.05) {
        st.float(name, .bf16, [n]) { mean + sd * $0.normal() }
    }
    func bias(_ name: String, _ n: Int, _ sd: Float) { st.float(name, .bf16, [n]) { sd * $0.normal() } }
    for i in 0..<ToySpec.layers {
        let p = root + "layers.\(i)."
        norm(p + "input_layernorm.weight", H)
        norm(p + "post_attention_layernorm.weight", H)
        let a = p + "self_attn."
        st.quant(a + "q_proj", rows: qd, cols: H, gain: 4)
        st.quant(a + "k_proj", rows: kvd, cols: H, gain: 4)
        st.quant(a + "v_proj", rows: kvd, cols: H)
        st.quant(a + "o_proj", rows: H, cols: qd, gain: 256)
        if ToyConfig.graniteAttentionBias {
            bias(a + "q_proj.bias", qd, 1)
            bias(a + "k_proj.bias", kvd, 1)
            bias(a + "v_proj.bias", kvd, 0.05)
            bias(a + "o_proj.bias", H, 1)
        }
        let m = p + "mlp."
        st.quant(m + "gate_proj", rows: d.denseI, cols: H)
        st.quant(m + "up_proj", rows: d.denseI, cols: H)
        st.quant(m + "down_proj", rows: H, cols: d.denseI, gain: 256)
    }
    st.quant(root + "embed_tokens", rows: ToySpec.vocab, cols: H, gain: 20)
    norm(root + "norm.weight", H, 1, 0.1)
    try st.write(to: dir.appendingPathComponent("model.safetensors"))
    try writeConfig(
        ToyConfig.granite(d, quant: ["group_size": st.group4, "bits": 4], seed: Int(seed)), dir)
}

/// toy-gemma4 v3 (gemma4_text, gemma-4-26B-A4B-it plus the E2B pieces): sliding, full,
/// sliding, full layers with the two attention shapes (sliding 8 kv × 256 with v_proj; full 2 kv × 512 with K = V, no
/// v_proj), x·w norms (the layer and final ones around 1, the sandwich post-norms around 30 so the branches weigh
/// against the embedding × √hidden), q and k norms around 0.3 (the attention scale is 1: scores of std ~2 with heads of
/// 256 and 512), a `layer_scalar` per layer in [0.75, 1.25), GeGLU FFN and a tied embedding of 4 bits with gain 8.
/// Since v2 the last two layers read the KV of the first two (no k_proj, v_proj nor k_norm) with a double-wide MLP, and
/// every layer ends with the PLE branch (table and projection of 4 × 256 per row; its post-norm around 10). Since v3
/// every layer runs the 26B's MoE in parallel with the dense MLP: 8-bit router whose own norm weight (`router.scale`
/// around √hidden, so scale · hidden^-0.5 is about 1) and gain (ToySpec.routerGain) give logits of std ~4 as in
/// toy-qwen36-moe, `per_expert_scale` in [0.5, 1.5] (the real one stays within 1 ± 0.03: wider here so a missing
/// factor shows), 128 experts of ToyCut.moeI, and the pre-norm of the MoE and the post-norms of both halves around 1.
func writeToyGemma4(_ dir: URL, seed: UInt64, _ d: ToyDims) throws {
    let st = ToySafetensors(seed: seed ^ 0x6E34)
    let root = "model.", H = d.hidden, dev = ToyConfig.gemma4Deviations, L = ToySpec.layers, P = dev.ple
    func norm(_ name: String, _ n: Int, _ mean: Float, _ sd: Float) {
        st.float(name, .bf16, [n]) { mean + sd * $0.normal() }
    }
    var q8: [String: Any] = [:]
    for i in 0..<L {
        let p = root + "layers.\(i).", (hd, kvh) = d.attention(layer: i)
        let keqv = d.layerTypes[i] == "full_attention", ownKV = i < L - dev.kvShared
        let inter = d.denseI * (dev.doubleMLP && !ownKV ? 2 : 1)
        norm(p + "input_layernorm.weight", H, 1, 0.05)
        norm(p + "post_attention_layernorm.weight", H, 30, 3)
        norm(p + "pre_feedforward_layernorm.weight", H, 1, 0.05)
        norm(p + "post_feedforward_layernorm.weight", H, 30, 3)
        st.float(p + "layer_scalar", .bf16, [1]) { 0.75 + 0.5 * $0.uniform() }
        let a = p + "self_attn."
        st.quant(a + "q_proj", rows: d.qh * hd, cols: H)
        if ownKV {
            st.quant(a + "k_proj", rows: kvh * hd, cols: H)
            if !keqv { st.quant(a + "v_proj", rows: kvh * hd, cols: H) }
        }
        st.quant(a + "o_proj", rows: H, cols: d.qh * hd)
        norm(a + "q_norm.weight", hd, 0.3, 0.05)
        if ownKV { norm(a + "k_norm.weight", hd, 0.3, 0.05) }
        let m = p + "mlp."
        st.quant(m + "gate_proj", rows: inter, cols: H)
        st.quant(m + "up_proj", rows: inter, cols: H)
        st.quant(m + "down_proj", rows: H, cols: inter)
        norm(p + "pre_feedforward_layernorm_2.weight", H, 1, 0.05)
        norm(p + "post_feedforward_layernorm_1.weight", H, 1, 0.05)
        norm(p + "post_feedforward_layernorm_2.weight", H, 1, 0.05)
        st.quant(p + "router.proj", rows: d.experts, cols: H, bits: 8, gain: ToySpec.routerGain)
        q8[p + "router.proj"] = ["group_size": 64, "bits": 8]
        norm(p + "router.scale", H, Float(H).squareRoot(), 1)
        st.float(p + "router.per_expert_scale", .bf16, [d.experts]) { 0.5 + $0.uniform() }
        let x = p + "experts.switch_glu."
        st.quant(x + "gate_proj", rows: d.moeI, cols: H, experts: d.experts)
        st.quant(x + "up_proj", rows: d.moeI, cols: H, experts: d.experts)
        st.quant(x + "down_proj", rows: H, cols: d.moeI, experts: d.experts)
        st.quant(p + "per_layer_input_gate", rows: P, cols: H)
        st.quant(p + "per_layer_projection", rows: H, cols: P)
        norm(p + "post_per_layer_input_norm.weight", H, 30, 3)
    }
    st.quant(root + "embed_tokens", rows: ToySpec.vocab, cols: H, gain: 8)
    norm(root + "norm.weight", H, 1, 0.1)
    st.quant(root + "embed_tokens_per_layer", rows: ToySpec.vocab, cols: L * P)
    st.quant(root + "per_layer_model_projection", rows: L * P, cols: H)
    norm(root + "per_layer_projection_norm.weight", P, 1, 0.05)
    try st.write(to: dir.appendingPathComponent("model.safetensors"))
    var quant: [String: Any] = ["group_size": 64, "bits": 4]
    quant.merge(q8) { a, _ in a }
    try writeConfig(ToyConfig.gemma4(d, quant: quant, seed: Int(seed)), dir)
}
