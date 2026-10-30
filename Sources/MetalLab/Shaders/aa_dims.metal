// Model dimensions as compile-time constants (MetalContext passes them as macros; default: Qwen3.6-35B-A3B).
// This file goes first (the .metal files are concatenated in alphabetical order).
#ifndef LAB_HIDDEN
#define LAB_HIDDEN 2048     // hidden size of the model and of the DFlash2 drafter
#endif
#ifndef LAB_GDN_HV
#define LAB_GDN_HV 32       // GDN v heads (the k heads are always 16 × 128)
#endif
#ifndef LAB_AT_QH
#define LAB_AT_QH 16        // query heads of the full attention (× 256)
#endif
#ifndef LAB_AT_KVH
#define LAB_AT_KVH 2        // KV heads
#endif
#ifndef LAB_AT_HD
#define LAB_AT_HD 256       // attention head dimension: 64, 128, 256 or 512 (64 in LFM2; 512 in Gemma 4's full layers)
#endif
#ifndef LAB_AT_ROT
#define LAB_AT_ROT 64       // dimensions the attention RoPE rotates (head_dim · partial_rotary_factor; LFM2: 64)
#endif
#ifndef LAB_SC_L
#define LAB_SC_L 3          // taps of the short convolution (conv_L_cache: 3 in LFM2, 7 in LFM3)
#endif
#ifndef LAB_GDN_HK
#define LAB_GDN_HK 16       // GDN k heads and their dimension: the kernels assume 16 × 128 (only checked)
#endif
#ifndef LAB_GDN_DK
#define LAB_GDN_DK 128
#endif
#ifndef LAB_GDN_DV
#define LAB_GDN_DV 128      // GDN v dimension per head (Mamba2's head_dim: 64 in Nemotron 3.5 Lightning, 80 in the Nano 4B)
#endif
#ifndef LAB_SSM
#define LAB_SSM 0           // 1: the recurrent layers are Nemotron-H's Mamba2 (no delta term, D·x skip, gated group norm)
#endif
#ifndef LAB_AT_QKN
#define LAB_AT_QKN 1        // 1: per-head RMSNorm of q and k before the RoPE (no weight = buffer of ones); 0: none
#endif
#ifndef LAB_AT_VN
#define LAB_AT_VN 0         // 1: weightless per-head RMSNorm of V before it is stored (Gemma 4's v_norm)
#endif
#ifndef LAB_AT_QKVB
#define LAB_AT_QKVB 0       // 1: q|k|v projections with bias (attention_bias / qkv_bias), added in the prework
#endif
#ifndef LAB_AT_GATE
#define LAB_AT_GATE 1       // 1: q_proj with a per-head gate [q | gate] (Qwen3.5/3.6); 0: no gate (LFM2)
#endif
#ifndef LAB_GDN_SIG
#define LAB_GDN_SIG 0       // 1: sigmoid gate in the GDN norm (Flash-Next, output_gate_type sigmoid); 0: SiLU
#endif
#ifndef LAB_MOE_E
#define LAB_MOE_E 256       // routed experts, top-k and intermediate size of the MoE (the generic kernels of zf_flashnext.metal)
#endif
#ifndef LAB_MOE_K
#define LAB_MOE_K 8
#endif
#ifndef LAB_MOE_I
#define LAB_MOE_I 512
#endif
#ifndef LAB_MOE_SHARED
#define LAB_MOE_SHARED 1    // MoE shared expert (vmoe_route, moe_route_topk_cfg): 0 none, 1 with sigmoid gate, 2 without gate
#endif
#ifndef LAB_MOE_SCORE
#define LAB_MOE_SCORE 0     // router scoring of vmoe_route and moe_route_topk_cfg: 0 softmax, 1 sigmoid
#endif
#ifndef LAB_MOE_BIAS
#define LAB_MOE_BIAS 0      // 1: selection-only bias added to the sigmoid scores before the top-k
#endif
#ifndef LAB_MOE_PES
#define LAB_MOE_PES 0       // 1: per-expert scale [E] multiplied into the chosen experts' weights (Gemma 4 per_expert_scale)
#endif
#ifndef LAB_MOE_EPS
#define LAB_MOE_EPS 0.0f    // epsilon of the renormalization of the sigmoid top-k weights (LFM2: 1e-6)
#endif
// LAB_MOE_RSCALE: routed_scaling_factor, defined only when it is not 1 (the routed weights are multiplied by it).
#ifndef LAB_HC_S
#define LAB_HC_S 4          // hyper-connection streams (Flash-Next: hc_count)
#endif
#ifndef LAB_PLE_H
#define LAB_PLE_H 16        // heads of the PLE table (heads_per_ngram · (ngram_size − 1)); at most 16
#endif
#ifndef LAB_PLE_HPN
#define LAB_PLE_HPN 8       // heads per n-gram (the first ones, 2-grams; the rest, 3-grams)
#endif
#ifndef LAB_PLE_DIM
#define LAB_PLE_DIM 2560    // width of the PLE embedding (ple_embed_dim): LAB_PLE_DIM / LAB_PLE_H per head
#endif
#ifndef LAB_NWT
#define LAB_NWT bfloat      // type of the small weights (norms, conv, dt_bias): float in Bonsai 2 and Flash-Next (A_log: always float)
#endif
#ifndef LAB_FFN_GELU
#define LAB_FFN_GELU 0      // 1 if the FFN uses GELU-tanh: compiles the `_gelu` instances of the gated FFN kernels
#endif
#ifndef LAB_FFN_OAI
#define LAB_FFN_OAI 0       // 1 if the MoE uses gpt-oss's swiglu_oai: compiles the `_oai` instances of the MoE gate+up kernels
#endif
#ifndef LAB_FFN_RELU2
#define LAB_FFN_RELU2 0     // 1 if the FFN is Nemotron-H's relu² without gate: compiles the `_relu2` instances (gate unread)
#endif
#ifndef LAB_FFN_LIMIT
#define LAB_FFN_LIMIT 7.0f  // swiglu_limit of swiglu_oai (the clip of the gate from above and of the linear part to ±limit)
#endif
#ifndef LAB_MOE_LBIAS
#define LAB_MOE_LBIAS 0     // 1: router linear with bias (gpt-oss), added to the logits before top-k and weights (buffer of LAB_MOE_BIAS)
#endif
#ifndef LAB_MOE_EBIAS
#define LAB_MOE_EBIAS 0     // 1: experts with biases (gpt-oss): gate and up before the activation, down before the weight
#endif
#ifndef LAB_MOE_H64
#define LAB_MOE_H64 0       // 1: MoE hidden a multiple of 64 but not of 128: the last moe_down2 column tile is shifted back
#endif
#ifndef LAB_G32
#define LAB_G32 0           // 1 if the model has 4-bit affine layers with groups of 32: instantiates their kernels
#endif
#ifndef LAB_AT_SINKS
#define LAB_AT_SINKS 0      // 1 if the attention has learned sinks per query head: pattn_i8 prefill and vattn_reduce
#endif
#ifndef LAB_AT_SWA
#define LAB_AT_SWA 0        // 1 if the model has sliding-window attention layers: instantiates the windowed kernels
#endif
#ifndef LAB_AT_ROPETAB
#define LAB_AT_ROPETAB 0    // 1 if some attention layer has a scaled RoPE: the prework kernels read a per-layer table
#endif
