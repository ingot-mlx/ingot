// Full verification attention (B <= 16 rows over the KV). Qwen3.6-35B-A3B: 16 query heads x 256, 2 KV heads
// (GQA 8), q_proj with a per-head gate [q 256 | gate 256], q_norm/k_norm (RMSNorm with weight), partial RoPE (64
// dimensions, non-interleaved halves, base 1e7), scale 1/16, output * sigmoid(gate) before o_proj. All comes from
// the macros (aa_dims.metal): head of 64, 128 or 256 (AT_HD), with or without gate, RoPE of AT_ROT dims, per-layer
// theta (AttnParams.theta) or none (bit AT_DYN_NOPE), QK-norm or none (LAB_AT_QKN).
//
// Here: parameters, shared helpers and the reduction of the key chunks (flash-decoding style: each chunk leaves its
// (max, sum)). The int8 K/V write lives in verify_attn_nax.metal and the products in verify_attn_pf.metal; the fp16
// KV path (its prework and NAX kernel) was removed.
// Concatenated after verify.metal.

struct AttnParams {
    uint B;      // rows of the block
    uint ctx;    // positions in the cache before the block
    uint C;      // keys per chunk (multiple of 8)
    uint nsplit; // chunks
    uint ldkv;   // rows reserved per KV head in the cache
    float scale; // 1/sqrt(AT_HD)
    float eps;   // rms_norm_eps
    float theta; // RoPE base of the layer
    uint dyn;    // bit 0 (AT_DYN_ST, _i8 kernels): ctx is read from the GPU round state (st[0]); bit 1 (AT_DYN_NOPE):
                 // layer without RoPE; bits 8-15 (dattn_i8t): per-row threadgroups of the tree
    uint window; // sliding window, counting the row's own key (0 = full); read only by the SWA kernels
    uint ring;   // ring KV of a sliding-window layer: slot = position % ring (0 = linear; LAB_AT_SWA kernels)
};
constant constexpr uint AT_DYN_ST = 1u, AT_DYN_NOPE = 2u;

// Head dimension; AT_QH query heads and AT_KVH KV heads (GQA AT_G); projection row: q|gate (AT_QK),
// k (AT_KVH*AT_HD), v (AT_KVH*AT_HD); output AT_O = AT_QH*AT_HD. Qwen3.6: 16 / 2 (9216); 27B: 24 / 4 (14336).
// The queries of a GQA group are ordered m = row*AT_G + h; every kernel (prework, pattn/dattn_i8, reduce) takes any GQA.
// LFM2: heads of 64 without gate (LAB_AT_HD, LAB_AT_GATE); AT_QS = stride of a query head in the row.
// prework and reduce: one thread per dimension (AT_HD threads, AT_NSG simdgroups).
constant constexpr uint AT_HD = LAB_AT_HD, AT_QH = LAB_AT_QH, AT_KVH = LAB_AT_KVH, AT_G = AT_QH / AT_KVH;
// 576: MLA's latent head (GLM-4.7-Flash, Layers/MLA.swift): one KV head [k_pe (64) | c (512)] read by every query head.
static_assert(AT_HD == 64 || AT_HD == 128 || AT_HD == 256 || AT_HD == 512 || AT_HD == 576,
              "attention: head of 64, 128, 256 or 512 (576: MLA's latent head)");
constant constexpr bool AT_GATED = LAB_AT_GATE != 0, AT_QKN = LAB_AT_QKN != 0;
// Weightless RMSNorm of each v head before it is stored (Gemma 4's v_norm), with the eps of q and k.
constant constexpr bool AT_VN = LAB_AT_VN != 0;
// Bias of q|k|v (fp32 [AT_QP], the layout of the projection row; buffer 12 of the prework kernels), added before
// the QK-norm and the RoPE. Without LAB_AT_QKVB the buffer is not read.
constant constexpr bool AT_QKVB = LAB_AT_QKVB != 0;
static_assert(!(AT_QKVB && AT_GATED), "q|k|v biases with a gated q_proj are not supported");
// Learned attention sinks (gpt-oss `sinks`, Flint `softmax_offset`): one logit per query head that joins the
// softmax denominator and carries no value. Applied where the softmax closes: the pattn_i8 prefill epilogue and
// vattn_reduce (decode and verification). The extra buffer only exists with LAB_AT_SINKS.
constant constexpr bool AT_SINKS = LAB_AT_SINKS != 0;
#if LAB_AT_SINKS
#define AT_SINK_ARG(I) , const device float* sinks [[buffer(I)]]
#define AT_SINK_T , const device float*
#else
#define AT_SINK_ARG(I)
#define AT_SINK_T
#endif
constant constexpr uint AT_QS = AT_GATED ? 2 * AT_HD : AT_HD, AT_NSG = AT_HD / 32;
// Sum of x² over the head held one value per thread (AT_HD threads, AT_NSG simdgroups), in fixed order. Every thread
// of the threadgroup must call it (barrier); the caller must be done reading `part`.
inline float at_head_ss(float x, threadgroup float* part, ushort sg, ushort lane) {
    const float ss = simd_sum(x * x);
    if (lane == 0) part[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0;
    VUNROLL for (uint i = 0; i < AT_NSG; i++) tot += part[i];
    return tot;
}
// RoPE: rotates the first AT_ROT dimensions of the head (halves of AT_RH).
constant constexpr uint AT_ROT = LAB_AT_ROT, AT_RH = AT_ROT / 2;
// With LAB_AT_ROPETAB the prework kernels take the angle of pair i from a per-layer table (fp32
// [AT_RH + 1], buffer 11: inv_freq[i], then mscale, which multiplies the rotated dimensions; RopeTable.swift) instead
// of pow(theta, -i/AT_RH): YaRN and linear. Without it the buffer does not exist and the code is the previous.
#if LAB_AT_ROPETAB
#define AT_RTAB_ARG(I) , const device float* rtab [[buffer(I)]]
#define AT_RTAB_T , const device float*
#else
#define AT_RTAB_ARG(I)
#define AT_RTAB_T
#endif
// Angle of pair i at position pos, and the rotated value times the table's mscale.
#if LAB_AT_ROPETAB
#define AT_ROPE_ANG(pos, i) (float(pos) * rtab[i])
#define AT_ROPE_SCALE(x) ((x) * rtab[AT_RH])
#else
#define AT_ROPE_ANG(pos, i) (float(pos) * pow(p.theta, -float(i) / float(AT_RH)))
#define AT_ROPE_SCALE(x) (x)
#endif
static_assert(AT_ROT % 2 == 0 && AT_ROT <= AT_HD, "RoPE: rotated dimensions must be even and within the head");
constant constexpr uint AT_QK = AT_QH * AT_QS, AT_QP = AT_QK + 2 * AT_KVH * AT_HD, AT_O = AT_QH * AT_HD;
// Column of the projection row that thread d of slot `slot` reads (q heads, then k, then v).
METAL_FUNC uint at_col(uint slot, uint d) {
    if (slot < AT_QH) return slot * AT_QS + d;
    if (slot < AT_QH + AT_KVH) return AT_QK + (slot - AT_QH) * AT_HD + d;
    return AT_QK + AT_KVH * AT_HD + (slot - AT_QH - AT_KVH) * AT_HD + d;
}
// The gate: sigmoid(g) with a gate, 1 without.
METAL_FUNC float at_gate(const device float* proj, uint t, uint h, uint d) {
    if (!AT_GATED) return 1.0f;
    const float g = proj[size_t(t) * AT_QP + h * AT_QS + AT_HD + d];
    return 1.0f / (1.0f + exp(-g));
}

// Orthonormal Walsh-Hadamard transform (Sylvester order, H/sqrt(AT_HD)) of one head held one value per thread (thread
// d holds index d): steps 1..16 between lanes with simd_shuffle_xor and 32..AT_HD/2 through xs, like tern_fwht1024.
// Every thread of the threadgroup must call it (barriers); the caller must be done reading xs.
// MLA's latent head (AT_HD 576 = 64 ⊕ 512): one orthonormal transform per block, H₆₄ on [k_pe] (dims 0…63) and H₅₁₂ on
// [c] (64…575), so each block keeps its own norm and (H q)·(H k) = q·k still holds. 64 is a multiple of 32, so the
// lane steps are the same and the threadgroup steps 32…256 stay inside each block (the k_pe threads skip those of 64
// and up, but take every barrier).
#if LAB_AT_HD == 576
inline float at_fwht(float v, threadgroup float* xs, uint d, ushort lane) {
    VUNROLL for (ushort s = 1; s < 32; s <<= 1) {
        const float o = simd_shuffle_xor(v, s);
        v = (lane & s) ? o - v : v + o;
    }
    const bool pe = d < 64;
    const uint i = pe ? d : d - 64;
    VUNROLL for (uint s = 32; s < 512; s <<= 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        xs[d] = v;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (!pe || s < 64) {
            const float o = xs[(i ^ s) + (pe ? 0 : 64)];
            v = (i & s) ? o - v : v + o;
        }
    }
    // 1/√64 and 1/√512 as literals.
    return v * (pe ? 0.125f : 0.044194173824159216f);
}
#else
inline float at_fwht(float v, threadgroup float* xs, uint d, ushort lane) {
    VUNROLL for (ushort s = 1; s < 32; s <<= 1) {
        const float o = simd_shuffle_xor(v, s);
        v = (lane & s) ? o - v : v + o;
    }
    VUNROLL for (uint s = 32; s < AT_HD; s <<= 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        xs[d] = v;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float o = xs[d ^ s];
        v = (d & s) ? o - v : v + o;
    }
    // 1/sqrt(AT_HD) as a literal (64, 128, 256 or 512), not through the fast-math sqrt.
    return v * (AT_HD == 512   ? 0.044194173824159216f
                : AT_HD == 256 ? 0.0625f
                : AT_HD == 128 ? 0.088388347648318441f
                               : 0.125f);
}
#endif

// Combines the chunks and applies the gate: out[t, h*AT_HD + d] = (sum e^{m_s-M} O_s / sum e^{m_s-M} l_s)*sigmoid(g).
// Grid (AT_G*B, AT_KVH), AT_HD threads.
// MLA's latent head (576): V is the latent part of K, which the cache holds rotated (H₅₁₂ c), so the chunks sum
// P·(H₅₁₂ c) and the reduction rotates the result back (H₅₁₂ is orthonormal and symmetric: its own inverse). The
// attention kernels leave dims 0…63 of opart unwritten (V has no RoPE part); they come out as zero (H₆₄ of zero).
[[kernel]] void vattn_reduce(const device float*  opart [[buffer(0)]],
                             const device float2* ml    [[buffer(1)]],
                             const device float*  proj  [[buffer(2)]],
                             device half*         out   [[buffer(3)]],   // [B, AT_O]
                             constant AttnParams& p     [[buffer(4)]]
                             AT_SINK_ARG(5),
                             uint2 tg [[threadgroup_position_in_grid]],
                             uint d [[thread_index_in_threadgroup]]) {
    const uint m = tg.x, kvh = tg.y, Q = AT_G * p.B, t = m / AT_G, h = kvh * AT_G + m % AT_G;
#if LAB_AT_HD == 576
    static_assert(!AT_GATED && !AT_SINKS, "MLA's latent reduction has no gate nor sinks");
    threadgroup float xs[AT_HD];
    const bool live = d >= 64;
#else
    const bool live = true;
#endif
    const device float2* mlh = ml + size_t(kvh) * p.nsplit * Q;
    float M = -INFINITY;
    for (uint s = 0; s < p.nsplit; s++) M = max(M, mlh[s * Q + m].x);
    float o = 0, L = 0;
#if LAB_AT_SINKS
    // Sink: M = max(M, s) and L += e^(s − M) (no value).
    M = max(M, sinks[h]);
    L = exp(sinks[h] - M);
#endif
    for (uint s = 0; s < p.nsplit; s++) {
        const float2 v = mlh[s * Q + m];
        if (v.x == -INFINITY) continue;
        const float w = exp(v.x - M);
        L += w * v.y;
        if (live) o += w * opart[((size_t(kvh) * p.nsplit + s) * Q + m) * AT_HD + d];
    }
#if LAB_AT_HD == 576
    const float r = at_fwht(o / L, xs, d, ushort(d % 32));
    out[size_t(t) * AT_O + h * AT_HD + d] = half(r);
#else
    out[size_t(t) * AT_O + h * AT_HD + d] = half(o / L * at_gate(proj, t, h, d));
#endif
}
