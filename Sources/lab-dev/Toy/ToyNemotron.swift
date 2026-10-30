import Foundation
import LabBench

// toy-nemotron (nemotron_h, Nemotron 3.5 Lightning): the real Mamba2 and attention shapes with the ToyCut cuts and the
// declared differences of ToyConfig.nemotronH. No drafter (Nemotron-H has none in the engine).

/// toy-nemotron v2: layers M E M M * E - (ToyDims.nemotronH), each one block (norm, block, residual),
/// everything affine 4-bit g64 like the mlx-community conversion except the router, in bf16 without quantizing as in
/// the Lightning's 4bit conversion (the engine applies it in bf16 as well); norms, conv, A_log, dt_bias and D in
/// bf16, e_score_correction_bias in fp32.
/// Mamba2 with magnitudes close to a trained one: A_log = log(1…16) and dt_bias around −3 (dt = softplus of about
/// 0.05, decays g of 0.45–0.95 per step), D around 1, conv weights of 0.3 with biases of 0.1. The router at gain 2
/// (logits of std ~2) with a selection bias of std 0.1, which changes the choice; the relu² FFNs (experts `fc1`/`fc2`,
/// shared and dense `up_proj`/`down_proj`) with the down projections at gain 0.5. The embedding at gain 16 keeps the
/// token against the branches.
func writeToyNemotron(_ dir: URL, seed: UInt64, _ d: ToyDims) throws {
    let st = ToySafetensors(seed: seed ^ 0x4E48)
    let root = "backbone.", H = d.hidden, HV = d.gdnHV, VD = d.gdnHV * d.gdnDV, conv = d.gdnConv
    func vec(_ name: String, _ shape: [Int], _ gen: @escaping (inout ToyRNG) -> Float) {
        st.float(name, .bf16, shape, gen)
    }
    for i in 0..<d.layerTypes.count {
        let p = root + "layers.\(i).", m = p + "mixer."
        vec(p + "norm.weight", [H]) { 1 + 0.05 * $0.normal() }
        switch d.layerTypes[i] {
        case "moe":
            let E = d.experts
            vec(m + "gate.weight", [E, H]) { 2 / Float(H).squareRoot() * $0.normal() }
            st.float(m + "gate.e_score_correction_bias", .f32, [E]) { 0.1 * $0.normal() }
            st.quant(m + "switch_mlp.fc1", rows: d.moeI, cols: H, experts: E)
            st.quant(m + "switch_mlp.fc2", rows: H, cols: d.moeI, experts: E, gain: 0.5)
            st.quant(m + "shared_experts.up_proj", rows: d.sharedI, cols: H)
            st.quant(m + "shared_experts.down_proj", rows: H, cols: d.sharedI, gain: 0.5)
        case "mlp":
            st.quant(m + "up_proj", rows: d.denseI, cols: H)
            st.quant(m + "down_proj", rows: H, cols: d.denseI, gain: 0.5)
        case "full_attention":
            st.quant(m + "q_proj", rows: d.qh * d.hd, cols: H)
            st.quant(m + "k_proj", rows: d.kvh * d.hd, cols: H)
            st.quant(m + "v_proj", rows: d.kvh * d.hd, cols: H)
            st.quant(m + "o_proj", rows: H, cols: d.qh * d.hd)
        default:
            st.quant(m + "in_proj", rows: VD + conv + HV, cols: H)
            st.quant(m + "out_proj", rows: H, cols: VD)
            vec(m + "conv1d.weight", [conv, 4, 1]) { 0.3 * $0.normal() }
            vec(m + "conv1d.bias", [conv]) { 0.1 * $0.normal() }
            vec(m + "A_log", [HV]) { log(1 + 15 * $0.uniform()) }
            vec(m + "dt_bias", [HV]) { -3 + 0.5 * $0.normal() }
            vec(m + "D", [HV]) { 1 + 0.1 * $0.normal() }
            vec(m + "norm.weight", [VD]) { 1 + 0.05 * $0.normal() }
        }
    }
    st.quant(root + "embeddings", rows: ToySpec.vocab, cols: H, gain: 16)
    vec(root + "norm_f.weight", [H]) { 1 + 0.1 * $0.normal() }
    st.quant("lm_head", rows: ToySpec.vocab, cols: H, gain: 2)
    try st.write(to: dir.appendingPathComponent("model.safetensors"))
    let quant: [String: Any] = ["group_size": 64, "bits": 4, "mode": "affine"]
    try JSONSerialization.data(
        withJSONObject: ToyConfig.nemotronH(d, quant: quant, seed: Int(seed)), options: [.sortedKeys, .prettyPrinted]
    ).write(to: dir.appendingPathComponent("config.json"))
}
