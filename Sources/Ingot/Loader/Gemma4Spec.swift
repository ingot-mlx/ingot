import Foundation

// Gemma 4 pieces of ModelSpec: the KV source of each layer and the load errors for what the engine does not run.

extension ModelSpec {
    /// Source of the KV of each layer (`kvSource`): with `num_kv_shared_layers` = s, layers ≥ n − s read the last layer
    /// below n − s of their type (nil if there is none, a load error in `gemma4Gaps`); the others, their own.
    static func kvSources(_ cfg: ModelConfig) -> [Int?] {
        let n = cfg.layerTypes.count, shared = (cfg.text["num_kv_shared_layers"] as? NSNumber)?.intValue ?? 0
        var last: [String: Int] = [:]
        return cfg.layerTypes.enumerated().map { i, ty in
            if i < n - shared {
                last[ty] = i
                return nil
            }
            return last[ty]
        }
    }

    /// Gemma 4 pieces the engine does not run yet, each with its name. Read by
    /// key, whatever the family, so no config that asks for them loads silently; empty for the other families' configs.
    /// The two attention shapes, the 512 head, K = V, the v_norm and the proportional RoPE run; PLE, KV
    /// shared across layers and the double-wide MLP, and the MoE in parallel with the dense MLP (Gemma 4 only).
    static func gemma4Gaps(_ cfg: ModelConfig) -> [String] {
        let t = cfg.text
        func int(_ k: String) -> Int? { (t[k] as? NSNumber)?.intValue }
        func flag(_ k: String) -> Bool { (t[k] as? Bool) ?? false }
        var out: [String] = []
        // Full layers with their own head (`global_head_dim`) and, with K = V, kv heads: a second shape
        // (ModelDims.fullAttention), whose RoPE rotates the whole head, so it needs the proportional table.
        if let full = ModelDims.fullAttention(cfg) {
            if ![64, 128, 256, 512].contains(full.attnHD) {
                out.append("attention with head of \(full.attnHD) in the full layers (global_head_dim)")
            }
            if full.attnQH % full.attnKVH != 0 { out.append("full layers with \(full.attnKVH) kv heads") }
            let p = RopeSpec.parameters(t, layerType: "full_attention")
            if p.map(RopeScaling.type) != "proportional" {
                out.append("global_head_dim without a proportional RoPE in the full layers")
            }
        }
        if flag("attention_k_eq_v") && cfg.family != .gemma4 {
            out.append("K = V in the full layers outside Gemma 4 (attention_k_eq_v)")
        }
        // PLE, KV shared across layers and the double-wide MLP of the KV-sharing layers run in Gemma 4.
        let g4 = cfg.family == .gemma4
        if let n = int("hidden_size_per_layer_input"), n > 0 {
            if !g4 { out.append("per-layer inputs (PLE, hidden_size_per_layer_input \(n)) outside Gemma 4") }
            // ple_inputs runs one thread per value of a layer's slice; the gate's prefill GEMM needs N % 64 = 0.
            if g4 && (n % 64 != 0 || n > 1024) {
                out.append("per-layer inputs (PLE) of width \(n) (a multiple of 64 up to 1024 is needed)")
            }
        }
        let shared = int("num_kv_shared_layers") ?? 0
        if shared > 0 && !g4 { out.append("KV shared across layers (num_kv_shared_layers \(shared)) outside Gemma 4") }
        if shared > 0 && g4 {
            let src = kvSources(cfg)
            let orphans = (max(0, src.count - shared)..<src.count).filter { src[$0] == nil }
            if !orphans.isEmpty {
                out.append(
                    "KV shared across layers: layers \(orphans.map(String.init).joined(separator: ",")) have no earlier "
                        + "layer of their type with its own KV")
            }
        }
        if flag("use_double_wide_mlp") && shared > 0 && !g4 {
            out.append("double-wide MLP in the KV-sharing layers (use_double_wide_mlp) outside Gemma 4")
        }
        // The 26B-A4B's MoE in parallel with the dense MLP (router with its own input norm and per_expert_scale) runs
        // in Gemma 4 (FFNKind.denseAndMoE).
        if flag("enable_moe_block") && !g4 { out.append("MoE in parallel with the dense MLP outside Gemma 4") }
        // `use_bidirectional_attention: "vision"` only touches image tokens (text runs causal); `true` (EmbeddingGemma)
        // is refused.
        switch t["use_bidirectional_attention"] {
        case let s as String where s != "vision": out.append("use_bidirectional_attention «\(s)»")
        case let b as Bool where b: out.append("use_bidirectional_attention")
        default: break
        }
        return out
    }
}
