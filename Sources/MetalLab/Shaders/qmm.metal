// Matmul with MLX-style affine quantized weights (4 and 8 bits, groups of 64): common pieces.
//
// MLX format (MIT, Copyright © 2023 Apple Inc.): mlx/backend/metal/kernels/quantized.h. This file goes first
// after aa_dims.metal: includes, UNROLL and the common matmul parameters.

#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

// Gated FFN activation `act(g) · u`, on float or floatN: ACT 0 = SiLU (the expression the kernels had
// before, unchanged), 1 = GELU with the tanh approximation (mlx-lm `nn.gelu_approx`; `_gelu` instances, LAB_FFN_GELU),
// written as g·σ(2z) with z = √(2/π)·(g + 0.044715·g³): the same function as 0.5·g·(1 + tanh z), without the
// cancellation of 1 + tanh z in the negative tail. ACT 2 = gpt-oss's swiglu_oai (mlx-lm activations.swiglu_oai;
// `_oai` instances, LAB_FFN_OAI): the gate clipped from above and the linear part to ±limit (LAB_FFN_LIMIT, the
// config's swiglu_limit), α = 1.702, `(clip(u) + 1) · g·σ(α·g)`. ACT 3 = Nemotron-H's relu² without gate (mlx-lm
// `nn.relu2`; `_relu2` instances, LAB_FFN_RELU2): `max(u, 0)²`, g is ignored (the kernels do not load the gate).
template <int ACT, typename T>
inline T lab_glu(T g, T u) {
    if constexpr (ACT == 0) {
        return g / (1.0f + exp(-g)) * u;
    } else if constexpr (ACT == 1) {
        const T z2 = 1.5957691216057308f * (g + 0.044715f * g * g * g);
        return g / (1.0f + exp(-z2)) * u;
    } else if constexpr (ACT == 3) {
        const T r = max(u, T(0));
        return r * r;
    } else {
        const T gc = min(g, T(LAB_FFN_LIMIT)), uc = clamp(u, T(-LAB_FFN_LIMIT), T(LAB_FFN_LIMIT));
        return gc / (1.0f + exp(-1.702f * gc)) * (uc + 1.0f);
    }
}

// Unrolling is mandatory in loops over fragment arrays (see qmm_nax.metal).
#define UNROLL _Pragma("clang loop unroll(full)")

struct QmmParams {
    uint M;   // rows of X (tokens)
    uint N;   // outputs (rows of W)
    uint K;   // inputs (columns of W, unpacked)
};
