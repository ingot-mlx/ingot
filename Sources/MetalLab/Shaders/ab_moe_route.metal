// Common pieces of the MoE router kernels: vmoe_route (verify.metal) and moe_route_topk_cfg
// (qmm_nax_moe.metal). Concatenated right after aa_dims.metal, so it brings its own includes.
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

// Row stride of the router logits: E rounded up to 64 (the router's output rows are zero-padded to 64 when loading,
// MoELayer.loadRouter, for the 64-column tiles of the prefill GEMM). With E a multiple of 64
// it is E, the stride used before the padding.
constant constexpr uint MOE_LD = (LAB_MOE_E + 63) / 64 * 64;

#define MR_UNROLL _Pragma("clang loop unroll(full)")

// Sigmoid router of one row in one simdgroup (LAB_MOE_SCORE = 1): p = sigmoid(logit) in fp32, top-k by p (+ the
// selection bias with LAB_MOE_BIAS), ties to the lower index; the weights are the chosen p without the bias, divided by
// their sum (in order of k) + LAB_MOE_EPS (norm_topk_prob, always on), then multiplied by
// routed_scaling_factor (LAB_MOE_RSCALE). With LFM2's settings it is LFM2's router.
// Lane k < K gets the k-th expert (descending selection score) and its weight; NaN logits are never selected (if all
// are -inf, expert k). Semantics of mlx-lm deepseek_v3/glm4_moe group_expert_select (n_group 1) and lfm2_moe.
template <int VPL, int K>
inline void moe_route_sigmoid(const device float* lr, const device float* bias, ushort lane, thread uint& myi,
                              thread float& myw) {
    float pr[VPL], v[VPL];
    MR_UNROLL for (int i = 0; i < VPL; i++) {
        const uint e = lane + 32 * i;
        const float l = lr[e];
        pr[i] = 1.0f / (1.0f + exp(-l));
#if LAB_MOE_BIAS
        v[i] = isnan(l) ? -INFINITY : pr[i] + bias[e];
#else
        v[i] = isnan(l) ? -INFINITY : pr[i];
#endif
    }
    float sum = 0;
    myw = 0;
    myi = 0;
    MR_UNROLL for (int k = 0; k < K; k++) {
        float bv = -INFINITY; uint bi = 0xffffu;
        MR_UNROLL for (int i = 0; i < VPL; i++) if (v[i] > bv) { bv = v[i]; bi = lane + 32 * i; }
        const float m = simd_max(bv);
        uint sel = simd_min(bv == m ? bi : 0xffffu);
        if (sel >= uint(32 * VPL)) sel = k;
        float ps = 0;
        MR_UNROLL for (int i = 0; i < VPL; i++) if (lane + 32 * i == sel) { v[i] = -INFINITY; ps = pr[i]; }
        const float wv = simd_sum(ps);
        sum += wv;
        if (lane == uint(k)) { myw = wv; myi = sel; }
    }
    myw = myw / (sum + LAB_MOE_EPS);
#ifdef LAB_MOE_RSCALE
    myw *= LAB_MOE_RSCALE;
#endif
}
