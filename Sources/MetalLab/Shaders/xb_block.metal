// Block pieces that only run when the config asks for them (BlockSpec): the branch epilogue (sandwich
// post-norm, residual multiplier, o_proj bias, layer scalar), the embedding scale and norm, and the logits
// post-processing (multiplier, divisor, softcap). The current families never dispatch them. Goes after verify.metal
// (NormParams, LAB_NWT).

// Mean of squares of a row of D values (one threadgroup per row; `v(i)` gives value i). All threads get the result.
template <typename F>
inline float b4_mean_sq(F v, uint D, uint tid, ushort sg, ushort lane, uint ntg, threadgroup float* part) {
    float ss = 0;
    for (uint i = tid; i < D; i += ntg) { const float x = v(i); ss += x * x; }
    ss = simd_sum(ss);
    if (lane == 0) part[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0;
    for (uint i = 0; i < ntg / 32; i++) tot += part[i];
    return tot / float(D);
}

struct BranchParams {
    uint D;       // hidden
    float eps;    // epsilon of the post-norm
    float mul;    // residual multiplier
    float scale;  // layer scalar (1 except after the last branch of a layer that has one)
    uint flags;   // 1: RMSNorm with weight w; 2: bias b
};

// Branch epilogue, per row: h = (h + mul · N(t + b)) · scale, with N the post-norm (RMSNorm with weight w and eps) or
// the identity. t is the branch output (the mixer or the FFN accumulated into it from zero); it is left at zero for
// the next branch.
[[kernel]] void branch_add_rows(device float*         h [[buffer(0)]],
                                device float*         t [[buffer(1)]],
                                const device LAB_NWT* w [[buffer(2)]],
                                const device float*   b [[buffer(3)]],
                                constant BranchParams& p [[buffer(4)]],
                                uint row [[threadgroup_position_in_grid]],
                                uint tid [[thread_index_in_threadgroup]],
                                ushort sg [[simdgroup_index_in_threadgroup]],
                                ushort lane [[thread_index_in_simdgroup]],
                                uint ntg [[threads_per_threadgroup]]) {
    threadgroup float part[32];
    device float* hr = h + size_t(row) * p.D;
    device float* tr = t + size_t(row) * p.D;
    const bool norm = p.flags & 1, bias = p.flags & 2;
    float r = 1;
    if (norm) {
        r = rsqrt(b4_mean_sq([&](uint i) { return tr[i] + (bias ? b[i] : 0.0f); }, p.D, tid, sg, lane, ntg, part)
                  + p.eps);
    }
    for (uint i = tid; i < p.D; i += ntg) {
        float v = tr[i] + (bias ? b[i] : 0.0f);
        if (norm) v = v * r * float(w[i]);
        hr[i] = (hr[i] + p.mul * v) * p.scale;
        tr[i] = 0;
    }
}

struct ScaleParams { float s; uint n; };

// x[i] *= s over n values (the embedding multiplier on the fp32 residual stream).
[[kernel]] void scale_rows(device float* x [[buffer(0)]], constant ScaleParams& p [[buffer(1)]],
                           uint i [[thread_position_in_grid]]) {
    if (i < p.n) x[i] *= p.s;
}

// RMSNorm of fp32 rows in place (the embedding norm): x = x · rsqrt(mean(x²) + eps) · w.
[[kernel]] void rmsnorm_rows_f(device float*         x [[buffer(0)]],
                               const device LAB_NWT* w [[buffer(1)]],
                               constant NormParams&  p [[buffer(2)]],
                               uint row [[threadgroup_position_in_grid]],
                               uint tid [[thread_index_in_threadgroup]],
                               ushort sg [[simdgroup_index_in_threadgroup]],
                               ushort lane [[thread_index_in_simdgroup]],
                               uint ntg [[threads_per_threadgroup]]) {
    threadgroup float part[32];
    device float* xr = x + size_t(row) * p.ldx;
    const float r = rsqrt(b4_mean_sq([&](uint i) { return xr[i]; }, p.D, tid, sg, lane, ntg, part) + p.eps);
    for (uint i = tid; i < p.D; i += ntg) xr[i] = xr[i] * r * float(w[i]);
}

struct LogitsParams {
    uint n;        // values (rows · vocab)
    float mul;     // multiplier (Muse output_multiplier)
    float div;     // divisor (Granite logits_scaling)
    float softcap; // 0 = none
};

// Logits post-processing in mlx-lm order: y = x · mul / div, then softcap · tanh(y / softcap).
[[kernel]] void logits_post(device float* x [[buffer(0)]], constant LogitsParams& p [[buffer(1)]],
                            uint i [[thread_position_in_grid]]) {
    if (i >= p.n) return;
    float y = x[i] * p.mul / p.div;
    if (p.softcap > 0) y = p.softcap * precise::tanh(y / p.softcap);
    x[i] = y;
}

// Per-layer inputs of Gemma 4 E2B/E4B (PLE; GemmaPLE.swift, mlx-lm gemma4_text). n values per layer.
struct PLEParams {
    uint n;        // values per layer (hidden_size_per_layer_input)
    uint layers;   // layers (a table row holds layers · n)
    uint layer;    // ple_gate_rows: the layer
    float eps;     // ple_inputs: epsilon of per_layer_projection_norm
    float pscale;  // hidden^-0.5 on the projection
    float tscale;  // √n on the table row
    float oscale;  // 2^-0.5 on the sum
};

// fp32 → half (the residual stream as the input of the PLE matrices).
[[kernel]] void cast_rows_h(const device float* x [[buffer(0)]], device half* y [[buffer(1)]],
                            constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < n) y[i] = half(x[i]);
}

// PLE inputs in place over the table rows t [rows, layers · n] (dequantized embed_tokens_per_layer):
// t = (N(proj · pscale) + t · tscale) · oscale, with N the RMSNorm (weight w) of a layer's n values. Grid (layers,
// rows), n threads.
[[kernel]] void ple_inputs(device float*         t    [[buffer(0)]],
                           const device float*   proj [[buffer(1)]],
                           const device LAB_NWT* w    [[buffer(2)]],
                           constant PLEParams&   p    [[buffer(3)]],
                           uint2 tg [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]],
                           ushort sg [[simdgroup_index_in_threadgroup]],
                           ushort lane [[thread_index_in_simdgroup]],
                           uint2 ntg [[threads_per_threadgroup]]) {
    threadgroup float part[32];
    const size_t o = (size_t(tg.y) * p.layers + tg.x) * p.n;
    const float r =
        rsqrt(b4_mean_sq([&](uint i) { return proj[o + i] * p.pscale; }, p.n, tid, sg, lane, ntg.x, part) + p.eps);
    const float x = proj[o + tid] * p.pscale * r * float(w[tid]);
    t[o + tid] = (x + t[o + tid] * p.tscale) * p.oscale;
}

// PLE gate of layer p.layer: y[r, j] = half(gelu_tanh(g[r, j]) · ple[r, layer, j]), one thread per value.
[[kernel]] void ple_gate_rows(const device float* g   [[buffer(0)]],
                              const device float* ple [[buffer(1)]],
                              device half*        y   [[buffer(2)]],
                              constant PLEParams& p   [[buffer(3)]],
                              uint i [[thread_position_in_grid]]) {
    const uint r = i / p.n, j = i % p.n;
    y[i] = half(lab_glu<1>(g[i], ple[(size_t(r) * p.layers + p.layer) * p.n + j]));
}
