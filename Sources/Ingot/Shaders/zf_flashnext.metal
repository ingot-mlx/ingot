// Qwen3.8-Flash-Next (qwen4_exp_text): 4-stream hyper-connections, n-gram PLE table and 512-expert top-10 MoE
// with the Q2_0 experts of our format (scripts/flashnext/fnformat.py). Concatenated after verify*.metal
// (VUNROLL) and yy_model.metal. Reference: mlx-vlm qwen4_exp (language.py) and splash 1.4.0 (hyper_connection.metal,
// ngram_embedding.metal); the ref/flashnext/harness.py harness gives the comparison values.
//
// Layout: the residual stream is 4 fp32 streams [rows, 4·D] (D = LAB_HIDDEN, stream s at [s·D, (s+1)·D)).
// The norms already store 1 + w in fp32 (the model adds it at load). Each row is always computed with the same order
// of sums (independent of how many rows there are): per-row invariance for speculative verification.

constant constexpr uint FN_D = LAB_HIDDEN, FN_S = LAB_HC_S, FN_W = FN_S * LAB_HIDDEN;
// PLE: heads, 2-gram heads, embedding width and per-head width; rows of ple.bin, 4-bit g32 MLX
// (codes | bf16 scales | bf16 biases: 80 | 10 | 10 B, 160 per head).
constant constexpr uint FN_PH = LAB_PLE_H, FN_PHN = LAB_PLE_HPN, FN_PD = LAB_PLE_DIM, FN_PHD = FN_PD / FN_PH,
                        FN_PSC = FN_PHD / 2, FN_PBI = FN_PSC + FN_PHD / 32 * 2, FN_PRB = FN_PBI + FN_PHD / 32 * 2;
static_assert(FN_D % 64 == 0, "Flash-Next: hidden must be a multiple of 64");
static_assert(FN_PD % FN_PH == 0 && FN_PHD % 32 == 0, "PLE: per-head width must be whole and a multiple of the group (32)");
static_assert(FN_PH <= 16 && FN_PHN <= FN_PH, "PLE: at most 16 heads (the parameters carry 16 slots)");
// Tiles of the per-row reduction kernels (fn_gnorm, fn_ple_gate, fn_bf16_mm): FN_TG threads, FN_NSG simdgroups.
constant constexpr uint FN_TG = 256, FN_NSG = FN_TG / 32;

inline float fn_sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }

// ---------------------------------------------------------------------------------------------
// RMSNorm by groups of D (the 4 streams, or the PLE keys): o[r, g·D + i] = x·rsqrt(mean(x²) + eps)·w[g·D + i].
// Grid (groups, rows), FN_TG threads. x and o fp32 with strides ldx / ldo.
struct FnNormParams { uint D; float eps; uint ldx; uint ldo; };

[[kernel]] void fn_gnorm(const device float*    x [[buffer(0)]],
                         const device float*    w [[buffer(1)]],
                         device float*          o [[buffer(2)]],
                         constant FnNormParams& p [[buffer(3)]],
                         device half*           oh [[buffer(4)]],   // half copy (input of the 8-bit HC)
                         uint2 tg [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]],
                         ushort sg [[simdgroup_index_in_threadgroup]],
                         ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[FN_NSG];
    const device float* xr = x + size_t(tg.y) * p.ldx + tg.x * p.D;
    float ss = 0;
    for (uint i = tid; i < p.D; i += FN_TG) ss += xr[i] * xr[i];
    ss = simd_sum(ss);
    if (lane == 0) part[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0;
    VUNROLL for (uint i = 0; i < FN_NSG; i++) tot += part[i];
    const float r = rsqrt(tot / float(p.D) + p.eps);
    device float* orow = o + size_t(tg.y) * p.ldo + tg.x * p.D;
    device half* hrow = oh + size_t(tg.y) * p.ldo + tg.x * p.D;
    for (uint i = tid; i < p.D; i += FN_TG) {
        const float v = xr[i] * r * w[tg.x * p.D + i];
        orow[i] = v;
        hrow[i] = half(v);
    }
}

// ---------------------------------------------------------------------------------------------
// Y[M, N] (+)= X[M, K] · Wᵀ with W bf16 [N, K] and X fp32: hyper-connection projections (down|inject 10240 → 324,
// up 320 → 10240). One simdgroup per column and up to 16 rows per threadgroup (tg.y = block of 16 rows); the threads
// split K 4 at a time (each row's sum follows the same order for any M).
struct FnMMParams { uint M; uint N; uint K; uint ldx; uint ldy; uint flags; };   // flags: bit 0 accumulate

[[kernel]] void fn_bf16_mm(const device float*  x [[buffer(0)]],
                           const device bfloat* w [[buffer(1)]],
                           device float*        y [[buffer(2)]],
                           constant FnMMParams& p [[buffer(3)]],
                           uint2 tg [[threadgroup_position_in_grid]],
                           ushort sg [[simdgroup_index_in_threadgroup]],
                           ushort lane [[thread_index_in_simdgroup]]) {
    const uint n = tg.x * FN_NSG + sg;
    if (n >= p.N) return;
    const uint m0 = tg.y * 16, mr = min(16u, p.M - m0);
    float acc[16];
    VUNROLL for (uint r = 0; r < 16; r++) acc[r] = 0;
    const device bfloat4* wr = (const device bfloat4*)(w + size_t(n) * p.K);
    for (uint k4 = lane; k4 < p.K / 4; k4 += 32) {
        const float4 wv = float4(wr[k4]);
        VUNROLL for (uint r = 0; r < 16; r++) {
            if (r < mr) acc[r] += dot(wv, ((const device float4*)(x + size_t(m0 + r) * p.ldx))[k4]);
        }
    }
    VUNROLL for (uint r = 0; r < 16; r++) {
        if (r >= mr) break;
        const float v = simd_sum(acc[r]);
        if (lane == 0) {
            device float* o = y + size_t(m0 + r) * p.ldy + n;
            *o = (p.flags & 1) ? *o + v : v;
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Hyper-connections (BF16 weights with fn_bf16_mm, or 8-bit with the engine's qmv/qmm on the half copies), after the
// down|inject projection (dp [rows, ldd]: 320 of down and, if there is injection, 4 more):
// low = SiLU(down / 4) [rows, 320] and inj = 2·σ(inject / 4) [rows, 4]. One thread per value.
struct FnHCParams { uint rows; uint R; uint ldd; uint inject; };

[[kernel]] void fn_hc_act(const device float*  dp  [[buffer(0)]],
                          device float*        low [[buffer(1)]],
                          device float*        inj [[buffer(2)]],
                          device half*         lowh [[buffer(3)]],   // half copy (input of up in 8-bit)
                          constant FnHCParams& p   [[buffer(4)]],
                          uint gid [[thread_position_in_grid]]) {
    const uint per = p.R + (p.inject ? FN_S : 0), r = gid / per, j = gid % per;
    if (r >= p.rows) return;
    const float v = dp[size_t(r) * p.ldd + j] / float(FN_S);
    if (j < p.R) {
        const float a = v * fn_sigmoid(v);
        low[size_t(r) * p.R + j] = a;
        lowh[size_t(r) * p.R + j] = half(a);
    }
    else inj[r * FN_S + (j - p.R)] = 2.0f * fn_sigmoid(v);
}

// Input mix: mixed[r, d] = mean_s n[r, s, d]·σ(g[r, s, d]) → half (block input) and optional fp32.
[[kernel]] void fn_hc_mix(const device float* n  [[buffer(0)]],
                          const device float* g  [[buffer(1)]],
                          device half*        oh [[buffer(2)]],
                          device float*       of [[buffer(3)]],
                          constant uint2&     p  [[buffer(4)]],   // (rows, flags: bit 0 writes of; bit 1 g in d·4 + s order)
                          uint gid [[thread_position_in_grid]]) {
    const uint r = gid / FN_D, d = gid % FN_D;
    if (r >= p.x) return;
    const device float* nr = n + size_t(r) * FN_W;
    const device float* gr = g + size_t(r) * FN_W;
    float acc = 0;
    VUNROLL for (uint s = 0; s < FN_S; s++) acc += nr[s * FN_D + d] * fn_sigmoid(gr[(p.y & 2) ? d * FN_S + s : s * FN_D + d]);
    acc = acc / float(FN_S);
    oh[size_t(r) * FN_D + d] = half(acc);
    if (p.y & 1) of[size_t(r) * FN_D + d] = acc;
}

// Injection after the block: x[r, s, d] += br[r, d]·inj[r, s].
[[kernel]] void fn_hc_inject(device float*       x   [[buffer(0)]],
                             const device float* br  [[buffer(1)]],
                             const device float* inj [[buffer(2)]],
                             constant uint&      rows [[buffer(3)]],
                             uint gid [[thread_position_in_grid]]) {
    const uint r = gid / FN_W, c = gid % FN_W;
    if (r >= rows) return;
    x[size_t(r) * FN_W + c] += br[size_t(r) * FN_D + c % FN_D] * inj[r * FN_S + c / FN_D];
}

// Drafter capture: mean of the 4 streams in half, in the columns of its layer (stride ldo).
[[kernel]] void fn_stream_mean(const device float* x [[buffer(0)]],
                               device half*        o [[buffer(1)]],
                               constant uint2&     p [[buffer(2)]],   // (rows, ldo)
                               uint gid [[thread_position_in_grid]]) {
    const uint r = gid / FN_D, d = gid % FN_D;
    if (r >= p.x) return;
    const device float* xr = x + size_t(r) * FN_W;
    float acc = 0;
    VUNROLL for (uint s = 0; s < FN_S; s++) acc += xr[s * FN_D + d];
    o[size_t(r) * p.y + d] = half(acc / float(FN_S));
}

// ---------------------------------------------------------------------------------------------
// PLE (n-gram table of layer 1). Hash (fn_ple_round, below): for each position, the 2-grams (heads 0–7) and 3-grams
// (8–15) with the previous tokens of the same segment (EOS cuts: a previous token that crosses an EOS is EOS),
// t0·m0 ^ t1·m1 [^ t2·m2], modulo the head size plus its offset.

// Gather: table rows (MLX 4-bit g32: 20 u32 of codes | 5 bf16 scales | 5 bf16 biases, 100 B) already in
// `rows` (the row cache or the staging buffer; slot[t·16 + h] = row of that position and head) →
// emb half [rows, FN_PD] with head h at [FN_PHD·h, FN_PHD·(h + 1)) (2560 and 160 in Flash-Next).
[[kernel]] void fn_ple_gather(const device uchar* rows [[buffer(0)]],
                              const device uint*  slot [[buffer(1)]],
                              device half*        emb  [[buffer(2)]],
                              constant uint&      n    [[buffer(3)]],
                              uint gid [[thread_position_in_grid]]) {
    const uint t = gid / FN_PD, c = gid % FN_PD, h = c / FN_PHD, j = c % FN_PHD;
    if (t >= n) return;
    const device uchar* r = rows + size_t(slot[t * FN_PH + h]) * FN_PRB;
    const uint word = ((const device uint*)r)[j / 8];
    const float q = float((word >> (4 * (j % 8))) & 15u);
    const float s = float(((const device bfloat*)(r + FN_PSC))[j / 32]);
    const float b = float(((const device bfloat*)(r + FN_PBI))[j / 32]);
    emb[size_t(t) * FN_PD + c] = half(fma(q, s, b));
}

// PLE gate per (row, stream): k = norm_key(key[r, s]), q = norm_query(x[r, s]); g = Σ k·q / √D,
// g = sign(g)·√max(|g|, 1e-6); gated[r, s, :] = σ(g)·value[r, :]; normed = norm_conv(gated). Grid (4, rows), FN_TG threads.
[[kernel]] void fn_ple_gate(const device float* key   [[buffer(0)]],   // [rows, 4·D]
                            const device float* val   [[buffer(1)]],   // [rows, D]
                            const device float* x     [[buffer(2)]],   // [rows, 4·D] (streams)
                            const device float* wk    [[buffer(3)]],   // 1 + w of norm_key [4·D]
                            const device float* wq    [[buffer(4)]],   // norm_query
                            const device float* wc    [[buffer(5)]],   // norm_conv
                            device float*       gated [[buffer(6)]],   // [rows, 4·D]
                            device float*       normd [[buffer(7)]],   // [rows, 4·D]
                            constant float&     eps   [[buffer(8)]],
                            uint2 tg [[threadgroup_position_in_grid]],
                            uint tid [[thread_index_in_threadgroup]],
                            ushort sg [[simdgroup_index_in_threadgroup]],
                            ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[3][FN_NSG];
    const uint s = tg.x, r = tg.y;
    const device float* kr = key + size_t(r) * FN_W + s * FN_D;
    const device float* xr = x + size_t(r) * FN_W + s * FN_D;
    const device float* vr = val + size_t(r) * FN_D;
    float sk = 0, sq = 0, dk = 0;
    for (uint i = tid; i < FN_D; i += FN_TG) {
        const float a = kr[i], b = xr[i];
        sk += a * a; sq += b * b;
        dk += (a * wk[s * FN_D + i]) * (b * wq[s * FN_D + i]);
    }
    sk = simd_sum(sk); sq = simd_sum(sq); dk = simd_sum(dk);
    if (lane == 0) { part[0][sg] = sk; part[1][sg] = sq; part[2][sg] = dk; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tk = 0, tq = 0, td = 0, tv = 0;
    VUNROLL for (uint i = 0; i < FN_NSG; i++) { tk += part[0][i]; tq += part[1][i]; td += part[2][i]; }
    const float rk = rsqrt(tk / float(FN_D) + eps), rq = rsqrt(tq / float(FN_D) + eps);
    float g = td * rk * rq / sqrt(float(FN_D));
    g = sign(g) * sqrt(max(abs(g), 1e-6f));
    const float sgm = fn_sigmoid(g);
    // norm_conv of the gated value: needs Σ (σ·v)².
    float sv = 0;
    for (uint i = tid; i < FN_D; i += FN_TG) { const float a = sgm * vr[i]; sv += a * a; }
    sv = simd_sum(sv);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) part[0][sg] = sv;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    VUNROLL for (uint i = 0; i < FN_NSG; i++) tv += part[0][i];
    const float rc = rsqrt(tv / float(FN_D) + eps);
    device float* gr = gated + size_t(r) * FN_W + s * FN_D;
    device float* nr = normd + size_t(r) * FN_W + s * FN_D;
    for (uint i = tid; i < FN_D; i += FN_TG) {
        const float a = sgm * vr[i];
        gr[i] = a;
        nr[i] = a * rc * wc[s * FN_D + i];
    }
}

// Dilated depthwise conv (4 taps, dilation 3: 9 rows of history) + SiLU, and add to the streams:
// x[t] += gated[t] + SiLU(Σ_j w[c, j]·in[t − 3·(3 − j)]), with in = [history (9 rows, from row `nacc` of
// hprev) | normd]. Writes the new history into hcur: the 9 of the window and, if `tape`, the B rows of the block
// (the next round takes its window from row nacc). One thread per (row, channel).
struct FnPleConvParams { uint rows; uint nacc; uint tape; uint dyn; };   // dyn bit 0: nacc from the round state (st[1])

[[kernel]] void fn_ple_conv(const device float*  normd [[buffer(0)]],
                            const device float*  gated [[buffer(1)]],
                            const device float*  cw    [[buffer(2)]],   // [4·D, 4]
                            const device float*  hprev [[buffer(3)]],   // [9 + B_prev, 4·D]
                            device float*        hcur  [[buffer(4)]],   // [9 + B, 4·D]
                            device float*        x     [[buffer(5)]],   // streams
                            constant FnPleConvParams& p [[buffer(6)]],
                            const device uint*   rst   [[buffer(7)]],
                            uint gid [[thread_position_in_grid]]) {
    const uint t = gid / FN_W, c = gid % FN_W;
    if (t >= p.rows) return;
    const uint nacc = (p.dyn & 1) ? rst[1] : p.nacc;
    float acc = 0;
    VUNROLL for (uint j = 0; j < 4; j++) {
        const int i = int(t) - int(3 * (3 - j));   // input row (negative: history)
        const float in = i < 0 ? hprev[size_t(nacc + 9 + i) * FN_W + c] : normd[size_t(i) * FN_W + c];
        acc = fma(cw[c * 4 + j], in, acc);
    }
    const float o = acc / (1.0f + exp(-acc));
    x[size_t(t) * FN_W + c] += gated[size_t(t) * FN_W + c] + o;
    if (p.tape) {
        if (t == 0) VUNROLL for (uint q = 0; q < 9; q++) hcur[size_t(q) * FN_W + c] = hprev[size_t(nacc + q) * FN_W + c];
        hcur[size_t(9 + t) * FN_W + c] = normd[size_t(t) * FN_W + c];
    } else if (t + 9 >= p.rows) {
        // Prefill: only the final window (the last 9 inputs: of the chunk or, if it is short, of the history).
        hcur[size_t(t + 9 - p.rows) * FN_W + c] = normd[size_t(t) * FN_W + c];
    }
    if (!p.tape && t == 0 && p.rows < 9) {
        // Chunk of fewer than 9 rows: the new window carries over 9 − rows from the previous history.
        for (uint q = 0; q < 9 - p.rows; q++) hcur[size_t(q) * FN_W + c] = hprev[size_t(nacc + p.rows + q) * FN_W + c];
    }
}

// ---------------------------------------------------------------------------------------------
// Flash-Next MoE: router (fp32 logits from the 8-bit qmv/qmm, with the shared-expert gate in column E),
// top-10 of the softmax with normalized weights (moe_route_topk_fn) and grouping by expert (moe_group), both in
// qmm_nax_moe.metal; gate|up and down Q2 in tiles of ≤ 16 rows of the same expert, and combination with the shared one.
// E, top-k and intermediate size come from the config (LAB_MOE_*): E a multiple of 32 and ≤ 512, K ≤ 16, I a multiple of 64.
constant constexpr uint FN_E = LAB_MOE_E, FN_K = LAB_MOE_K, FN_I = LAB_MOE_I, FN_TILE = 16;

// Q2 expert in its record (fnformat.py): gate planes (codes [I, D/16] u32 | scales [I, D/64] f16), up, down
// (codes [D, I/16] | scales [D, I/64]). The record of expert e is at arena + smap[e]·rec.
struct FnQ2Params { uint rec; uint gc; uint gs; uint uc; uint us; uint dc; uint ds; uint T; };

// SwiGLU of the shared expert: gu fp32 [T, 2·I] (gate | up) → half [T, I].
[[kernel]] void fn_swiglu(const device float* gu [[buffer(0)]],
                          device half*        o  [[buffer(1)]],
                          constant uint2&     p  [[buffer(2)]],   // (T, I)
                          uint gid [[thread_position_in_grid]]) {
    const uint r = gid / p.y, i = gid % p.y;
    if (r >= p.x) return;
    const float g = gu[size_t(r) * 2 * p.y + i], u = gu[size_t(r) * 2 * p.y + p.y + i];
    o[size_t(r) * p.y + i] = half(g * fn_sigmoid(g) * u);
}

// Zeroes n float4.
[[kernel]] void fn_zero(device float4* x [[buffer(0)]], constant uint& n [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    if (gid < n) x[gid] = 0;
}

// ---------------------------------------------------------------------------------------------
// Rounds (chained or not) inside HybridModel: writes the round's tokens into the history by position
// (hist[P + t] = tok[t]; P from st[0] with dyn, the fixed position otherwise) and computes the PLE indices of each row with
// the two previous tokens: t1 = h[p − 1], t2 = h[p − 1] == EOS ? EOS : h[p − 2], and EOS before position 0
// (same as the mlx-vlm rule: a previous token that crosses an EOS is EOS).
struct FnPleRoundParams {
    ulong mult[3];
    ulong size[16];
    ulong off[16];
    uint rows;
    uint eos;
    uint pos;
    uint dyn;
};

inline uint fn_tok_at(const device uint* tok, const device uint* hist, int P, int q, uint eos) {
    if (q < 0) return eos;
    return q >= P ? tok[q - P] : hist[q];
}

[[kernel]] void fn_ple_round(const device uint*         tok  [[buffer(0)]],
                             device uint*               hist [[buffer(1)]],
                             device uint*               idx  [[buffer(2)]],   // [rows, FN_PH]
                             constant FnPleRoundParams& p    [[buffer(3)]],
                             const device uint*         st   [[buffer(4)]],
                             uint gid [[thread_position_in_grid]]) {
    const uint t = gid / FN_PH, h = gid % FN_PH;
    if (t >= p.rows) return;
    const uint P = p.dyn ? st[0] : p.pos;
    const int pos = int(P + t);
    const ulong t0 = tok[t];
    const uint a1 = fn_tok_at(tok, hist, int(P), pos - 1, p.eos);
    const ulong t1 = a1, t2 = a1 == p.eos ? ulong(p.eos) : ulong(fn_tok_at(tok, hist, int(P), pos - 2, p.eos));
    ulong mixed = (t0 * p.mult[0]) ^ (t1 * p.mult[1]);
    if (h >= FN_PHN) mixed ^= t2 * p.mult[2];
    idx[t * FN_PH + h] = uint(mixed % p.size[h] + p.off[h]);
    if (h == 0) hist[P + t] = tok[t];
}

// In Flash-Next: splits the moe_group tiles into those of already resident experts (smapR[e] valid: the GPU
// computes them before waiting for the CPU) and the missing ones, with their indirect grids (like args). One thread.
[[kernel]] void fn_moe_split(const device uint4* tiles  [[buffer(0)]],
                             const device uint*  args   [[buffer(1)]],
                             const device uint*  smapR  [[buffer(2)]],   // of the layer
                             device uint4*       tilesR [[buffer(3)]],
                             device uint4*       tilesM [[buffer(4)]],
                             device uint*        argsR  [[buffer(5)]],
                             device uint*        argsM  [[buffer(6)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid != 0) return;
    const uint n = args[1];
    uint nr = 0, nm = 0;
    for (uint t = 0; t < n; t++) {
        const uint4 tl = tiles[t];
        if (smapR[tl.x] != 0xffffffffu) tilesR[nr++] = tl; else tilesM[nm++] = tl;
    }
    argsR[0] = args[0]; argsR[1] = nr; argsR[2] = 1; argsR[4] = args[4]; argsR[5] = nr; argsR[6] = 1;
    argsM[0] = args[0]; argsM[1] = nm; argsM[2] = 1; argsM[4] = args[4]; argsM[5] = nm; argsM[6] = 1;
}

// ---------------------------------------------------------------------------------------------
// Q2 experts of the prefill on NAX (matmul2d), with the bodies of moe_gateup_swiglu2 / moe_down2
// (moe_gateup_body and moe_down_body of qmm_nax_moe.metal, with FnQ2Src, fixed K and N and gate|up in fp32): tiles of one expert × BM rows (moe_group with BM tiles), the X rows by `sorted`
// into threadgroup, the 2-bit weights (fp16 scale per group of 64, w = d·(q − 1)) dequantized into threadgroup with the
// 0x6400 trick (q − 1 ∈ {−1, 0, 1, 2}: d·(q − 1) is exact in half) and prefetch of the next step into registers.
// gate|up leaves h = SiLU(g)·u in half [pairs, I] (grouped order); down leaves y fp32 [pairs, D] for fn_moe_combine_inj.
// What each row computes does not depend on the other rows of the tile (same K loop and the same operations).

template <int rows, int BK>
struct FnQ2Stage {
    static constant constexpr uint upr = BK / 16, NI = rows * upr / NX_THREADS;
    static_assert(rows * upr % NX_THREADS == 0, "FnQ2Stage: rows × words must be a multiple of 128");
    uint v[NI];
    half sc[NI];
    METAL_FUNC void load(const device uint* codes, const device half* scales, uint K, uint n0, uint k0, uint tid) {
        NAX_UNROLL for (uint i = 0; i < NI; i++) {
            const uint idx = tid + i * NX_THREADS;
            const uint r = idx / upr, u = idx % upr, k = k0 + u * 16;
            v[i] = codes[size_t(n0 + r) * (K / 16) + k / 16];
            sc[i] = scales[size_t(n0 + r) * (K / 64) + k / 64];
        }
    }
    template <class S>
    METAL_FUNC void load(const thread S& src, uint K, uint n0, uint k0, uint tid) {
        load(src.codes, src.scales, K, n0, k0, tid);
    }
    METAL_FUNC void store(threadgroup half* ws, uint tid) const {
        constexpr int LD = BK + 8;
        NAX_UNROLL for (uint i = 0; i < NI; i++) {
            const uint idx = tid + i * NX_THREADS;
            const uint r = idx / upr, u = idx % upr;
            threadgroup half* o = ws + r * LD + u * 16;
            const half2 s2 = half2(sc[i]);
            const uint x = v[i];
            half lo[8], hi[8];
            NAX_UNROLL for (int j = 0; j < 8; j++) {
                const half2 q = (as_type<half2>(((x >> (2 * j)) & 0x00030003u) | 0x64006400u) - half2(1025.0h)) * s2;
                lo[j] = q.x; hi[j] = q.y;
            }
            *(threadgroup half4*)o = half4(lo[0], lo[1], lo[2], lo[3]);
            *(threadgroup half4*)(o + 4) = half4(lo[4], lo[5], lo[6], lo[7]);
            *(threadgroup half4*)(o + 8) = half4(hi[0], hi[1], hi[2], hi[3]);
            *(threadgroup half4*)(o + 12) = half4(hi[4], hi[5], hi[6], hi[7]);
        }
    }
};

// Q2 weight source (one code plane and its scale) for the NAX bodies of qmm_nax_moe.metal.
struct FnQ2Src {
    const device uint* codes;
    const device half* scales;
    template <int rows, int BK> using Stage = FnQ2Stage<rows, BK>;
};

// BM × 64 columns of I tiles (gate and up at once): the 4 simdgroups share the BM rows and split 16
// columns each. Indirect grid (I / 64, tiles, 1).
template <int BM, int BK>
[[kernel, max_total_threads_per_threadgroup(NX_THREADS)]] void fn_moe_gateup_nax(
    const device half*   x      [[buffer(0)]],
    const device uchar*  arena  [[buffer(1)]],
    const device uint*   smap   [[buffer(2)]],
    const device uint4*  tiles  [[buffer(3)]],
    const device uint*   sorted [[buffer(4)]],
    device half*         h      [[buffer(5)]],
    constant FnQ2Params& p      [[buffer(6)]],
    uint3  tgid [[threadgroup_position_in_grid]],
    uint   tid  [[thread_index_in_threadgroup]],
    ushort sgid [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]])
{
    constexpr int FM = BM / 16, LDW = BK + 8, LDA = BK + 8;
    using AS = NaxAStage<BM, BK>;
    threadgroup half wgs[64 * LDW];
    threadgroup half wus[64 * LDW];
    threadgroup half xs[BM * LDA];
    const uint4 td = tiles[tgid.y];
    const uint r0 = td.y, rows = td.z;
    const uint n0 = tgid.x * 64;
    const int tn = sgid * 16;
    const short2 c = nax_coord(lane);
    array<const device half*, AS::NI> rp;
    NAX_UNROLL for (int i = 0; i < AS::NI; i++) {
        const uint r = (tid + i * NX_THREADS) / AS::CPR;
        rp[i] = r < rows ? x + size_t(sorted[r0 + r] / FN_K) * FN_D : nullptr;
    }
    const device uchar* rec = arena + size_t(smap[td.x]) * p.rec;
    const FnQ2Src g{(const device uint*)(rec + p.gc), (const device half*)(rec + p.gs)};
    const FnQ2Src u{(const device uint*)(rec + p.uc), (const device half*)(rec + p.us)};
    const int live = min(int(rows + 15) / 16, FM);
#define FGU(L) moe_gateup_body<FnQ2Src, false, BM, 1, BK, 1, L, FN_D, FN_I>(g, u, h, rp, wgs, wus, xs, FN_D, FN_I, n0, r0, rows, tn, c, tid)
    if (FM >= 4 && live == 4) { if constexpr (FM >= 4) FGU((4 > FM ? FM : 4)); }
    else if (FM >= 3 && live == 3) { if constexpr (FM >= 3) FGU((3 > FM ? FM : 3)); }
    else if (FM >= 2 && live == 2) { if constexpr (FM >= 2) FGU((2 > FM ? FM : 2)); }
    else FGU(1);
#undef FGU
}

// BM × 128 columns of D tiles: the 4 simdgroups split 32 columns each. Indirect grid (D / 128,
// tiles, 1).
template <int BM, int BK>
[[kernel, max_total_threads_per_threadgroup(NX_THREADS)]] void fn_moe_down_nax(
    const device half*   hin    [[buffer(0)]],
    const device uchar*  arena  [[buffer(1)]],
    const device uint*   smap   [[buffer(2)]],
    const device uint4*  tiles  [[buffer(3)]],
    device float*        y      [[buffer(4)]],
    constant FnQ2Params& p      [[buffer(5)]],
    uint3  tgid [[threadgroup_position_in_grid]],
    uint   tid  [[thread_index_in_threadgroup]],
    ushort sgid [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]])
{
    constexpr int FM = BM / 16, LDW = BK + 8, LDA = BK + 8;
    using AS = NaxAStage<BM, BK>;
    threadgroup half ws[128 * LDW];
    threadgroup half xs[BM * LDA];
    const uint4 td = tiles[tgid.y];
    const uint r0 = td.y, rows = td.z;
    const uint n0 = tgid.x * 128;
    const int tn = sgid * 32;
    const short2 c = nax_coord(lane);
    array<const device half*, AS::NI> rp;
    NAX_UNROLL for (int i = 0; i < AS::NI; i++) {
        const uint r = (tid + i * NX_THREADS) / AS::CPR;
        rp[i] = r < rows ? hin + size_t(r0 + r) * FN_I : nullptr;
    }
    const device uchar* rec = arena + size_t(smap[td.x]) * p.rec;
    const FnQ2Src w{(const device uint*)(rec + p.dc), (const device half*)(rec + p.ds)};
    const int live = min(int(rows + 15) / 16, FM);
#define FDN(L) moe_down_body<FnQ2Src, float, BM, 1, BK, 1, L, FN_I, FN_D>(w, y, rp, ws, xs, FN_I, FN_D, n0, r0, rows, tn, c, tid)
    if (FM >= 4 && live == 4) { if constexpr (FM >= 4) FDN((4 > FM ? FM : 4)); }
    else if (FM >= 3 && live == 3) { if constexpr (FM >= 3) FDN((3 > FM ? FM : 3)); }
    else if (FM >= 2 && live == 2) { if constexpr (FM >= 2) FDN((2 > FM ? FM : 2)); }
    else FDN(1);
#undef FDN
}

#define FN_NAX_INST(BM, BK) \
  template [[host_name("fn_moe_gateup_nax_bm" #BM "_bk" #BK)]] [[kernel]] void fn_moe_gateup_nax<BM, BK>( \
      const device half*, const device uchar*, const device uint*, const device uint4*, const device uint*, \
      device half*, constant FnQ2Params&, uint3, uint, ushort, ushort); \
  template [[host_name("fn_moe_down_nax_bm" #BM "_bk" #BK)]] [[kernel]] void fn_moe_down_nax<BM, BK>( \
      const device half*, const device uchar*, const device uint*, const device uint4*, device float*, \
      constant FnQ2Params&, uint3, uint, ushort, ushort);
FN_NAX_INST(16, 64)
FN_NAX_INST(32, 64)
FN_NAX_INST(48, 64)
FN_NAX_INST(64, 64)

// In Flash-Next: window of a conv with history (the PLE one, 9 rows, or the QSA raw-key one, 3) after
// row r of a prefill chunk: the W entries before position r + 1 of the chunk, from `src` (chunk rows)
// or, if they fall before the chunk, from the previous window `hprev`. p = (r, W, width, 0). One thread per value.
[[kernel]] void fn_win_capture(const device float* src   [[buffer(0)]],
                               const device float* hprev [[buffer(1)]],
                               device float*       out   [[buffer(2)]],
                               constant uint4&     p     [[buffer(3)]],
                               uint gid [[thread_position_in_grid]]) {
    const uint i = gid / p.z, c = gid % p.z;
    if (i >= p.y) return;
    const int q = int(p.x) + 1 - int(p.y) + int(i);
    out[gid] = q >= 0 ? src[size_t(q) * p.z + c] : hprev[size_t(int(p.y) + q) * p.z + c];
}

// ---------------------------------------------------------------------------------------------
// Q2 experts of the round on NAX (v3, like vnax_qmv of verify_nax.metal): each weight is dequantized once and the
// products go to matmul2d, so the cost no longer grows with the tile rows (v2 did one fp32 FMA per weight and
// row: with 8 rows, compute-bound). 16-row fragments of the tile (≤ 16, repeated) × 16 of K with the
// vnax permutation: the thread with fragment column fn carries the 16 contiguous values [16·(fn/4), +16) of the block
// of 64 (one 2-bit word) and in chunk c uses {c, c+4, 8+c, 12+c}; x with the same permutation. Split-K across SGS
// simdgroups with the sum in threadgroup memory in fixed order: each row gives the same with any number of rows.

inline nax_hfrag fn_q2_frag(uint w0, half d0, uint w1, half d1, int c) {
    nax_hfrag f;
    VUNROLL for (int j = 0; j < 4; j++) {
        f[j] = d0 * (half((w0 >> (2 * (c + 4 * j))) & 3u) - 1.0h);
        f[4 + j] = d1 * (half((w1 >> (2 * (c + 4 * j))) & 3u) - 1.0h);
    }
    return f;
}

// gate|up + SiLU: 16 columns of I per threadgroup, SGS simdgroups split the 40 K blocks. h fp32 [pairs, I].
template <int SGS>
[[kernel]] void fn_moe_gateup_vnax(const device half*   x      [[buffer(0)]],
                                   const device uchar*  arena  [[buffer(1)]],
                                   const device uint*   smap   [[buffer(2)]],
                                   const device uint4*  tiles  [[buffer(3)]],
                                   const device uint*   sorted [[buffer(4)]],
                                   device float*        h      [[buffer(5)]],
                                   constant FnQ2Params& p      [[buffer(6)]],
                                   uint2 tg [[threadgroup_position_in_grid]],
                                   uint tid [[thread_index_in_threadgroup]],
                                   ushort sg [[simdgroup_index_in_threadgroup]],
                                   ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[2][SGS * 16 * 16];
    const uint4 tl = tiles[tg.y];
    const uint R = tl.z, n0 = tg.x * 16;
    const short2 co = nax_coord(lane);
    const uint fm = co.x, fn = co.y;
    const uint r0 = sorted[tl.y + min(fm, R - 1)] / FN_K, r1 = sorted[tl.y + min(fm + 8, R - 1)] / FN_K;
    const device uchar* rec = arena + size_t(smap[tl.x]) * p.rec;
    const device uint* wg = (const device uint*)(rec + p.gc);
    const device uint* wu = (const device uint*)(rec + p.uc);
    const device half* sgp = (const device half*)(rec + p.gs);
    const device half* sup = (const device half*)(rec + p.us);
    nax_ffrag ag = 0, au = 0;
    for (uint kb = sg; kb < FN_D / 64; kb += SGS) {
        const uint kk = kb * 64 + (fn / 4) * 16, wi = kk / 16;
        half x0[16], x1[16];
        vnax_load16(x0, x + size_t(r0) * FN_D + kk);
        vnax_load16(x1, x + size_t(r1) * FN_D + kk);
        const uint na = n0 + fm, nb = n0 + fm + 8;
        const uint g0 = wg[size_t(na) * (FN_D / 16) + wi], g1 = wg[size_t(nb) * (FN_D / 16) + wi];
        const uint u0 = wu[size_t(na) * (FN_D / 16) + wi], u1 = wu[size_t(nb) * (FN_D / 16) + wi];
        const half dg0 = sgp[size_t(na) * (FN_D / 64) + kb], dg1 = sgp[size_t(nb) * (FN_D / 64) + kb];
        const half du0 = sup[size_t(na) * (FN_D / 64) + kb], du1 = sup[size_t(nb) * (FN_D / 64) + kb];
        VUNROLL for (int c = 0; c < 4; c++) {
            const nax_hfrag a = vnax_afrag(x0, x1, c);
            nax_mma2(ag, au, a, fn_q2_frag(g0, dg0, g1, dg1, c), fn_q2_frag(u0, du0, u1, du1, c));
        }
    }
    VUNROLL for (int e = 0; e < 8; e++) {
        const uint i = (sg * 16 + fm + 8 * (e / 4)) * 16 + fn + e % 4;
        red[0][i] = ag[e]; red[1][i] = au[e];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < R * 16; i += SGS * 32) {
        const uint m = i / 16, n = i % 16;
        float gv = 0, uv = 0;
        VUNROLL for (int q = 0; q < SGS; q++) { gv += red[0][(q * 16 + m) * 16 + n]; uv += red[1][(q * 16 + m) * 16 + n]; }
        h[size_t(tl.y + m) * FN_I + n0 + n] = gv * fn_sigmoid(gv) * uv;
    }
}

// down: 32 columns of D per threadgroup (two fragments), SGS simdgroups split the 10 K blocks (I = 640);
// h fp32 → half at load (like in the prefill). y fp32 [pairs, D].
template <int SGS>
[[kernel]] void fn_moe_down_vnax(const device float*  h      [[buffer(0)]],
                                 const device uchar*  arena  [[buffer(1)]],
                                 const device uint*   smap   [[buffer(2)]],
                                 const device uint4*  tiles  [[buffer(3)]],
                                 device float*        y      [[buffer(4)]],
                                 constant FnQ2Params& p      [[buffer(5)]],
                                 uint2 tg [[threadgroup_position_in_grid]],
                                 uint tid [[thread_index_in_threadgroup]],
                                 ushort sg [[simdgroup_index_in_threadgroup]],
                                 ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[SGS * 16 * 32];
    const uint4 tl = tiles[tg.y];
    const uint R = tl.z, n0 = tg.x * 32;
    const short2 co = nax_coord(lane);
    const uint fm = co.x, fn = co.y;
    const uint r0 = tl.y + min(fm, R - 1), r1 = tl.y + min(fm + 8, R - 1);
    const device uchar* rec = arena + size_t(smap[tl.x]) * p.rec;
    const device uint* wd = (const device uint*)(rec + p.dc);
    const device half* sd = (const device half*)(rec + p.ds);
    nax_ffrag acc0 = 0, acc1 = 0;
    for (uint kb = sg; kb < FN_I / 64; kb += SGS) {
        const uint kk = kb * 64 + (fn / 4) * 16, wi = kk / 16;
        half x0[16], x1[16];
        VUNROLL for (int i = 0; i < 4; i++) {
            const float4 a = ((const device float4*)(h + size_t(r0) * FN_I + kk))[i];
            const float4 b = ((const device float4*)(h + size_t(r1) * FN_I + kk))[i];
            x0[4 * i] = half(a.x); x0[4 * i + 1] = half(a.y); x0[4 * i + 2] = half(a.z); x0[4 * i + 3] = half(a.w);
            x1[4 * i] = half(b.x); x1[4 * i + 1] = half(b.y); x1[4 * i + 2] = half(b.z); x1[4 * i + 3] = half(b.w);
        }
        uint w[4];
        half d[4];
        // W rows: fragment 0 = columns n0 + fm and n0 + fm + 8; fragment 1 = n0 + 16 + fm and n0 + 24 + fm.
        const uint na = n0 + fm, nb = n0 + fm + 8, nc = n0 + 16 + fm, nd = n0 + 24 + fm;
        w[0] = wd[size_t(na) * (FN_I / 16) + wi]; w[1] = wd[size_t(nb) * (FN_I / 16) + wi];
        w[2] = wd[size_t(nc) * (FN_I / 16) + wi]; w[3] = wd[size_t(nd) * (FN_I / 16) + wi];
        d[0] = sd[size_t(na) * (FN_I / 64) + kb]; d[1] = sd[size_t(nb) * (FN_I / 64) + kb];
        d[2] = sd[size_t(nc) * (FN_I / 64) + kb]; d[3] = sd[size_t(nd) * (FN_I / 64) + kb];
        VUNROLL for (int c = 0; c < 4; c++) {
            const nax_hfrag a = vnax_afrag(x0, x1, c);
            nax_mma2(acc0, acc1, a, fn_q2_frag(w[0], d[0], w[1], d[1], c), fn_q2_frag(w[2], d[2], w[3], d[3], c));
        }
    }
    VUNROLL for (int e = 0; e < 8; e++) {
        const uint rr = sg * 16 + fm + 8 * (e / 4), cc = fn + e % 4;
        red[rr * 32 + cc] = acc0[e]; red[rr * 32 + 16 + cc] = acc1[e];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < R * 32; i += SGS * 32) {
        const uint m = i / 32, n = i % 32;
        float v = 0;
        VUNROLL for (int q = 0; q < SGS; q++) v += red[(q * 16 + m) * 32 + n];
        y[size_t(tl.y + m) * FN_D + n0 + n] = v;
    }
}
template [[host_name("fn_moe_gateup_vnax_sg4")]] [[kernel]] void fn_moe_gateup_vnax<4>(const device half*,
    const device uchar*, const device uint*, const device uint4*, const device uint*, device float*,
    constant FnQ2Params&, uint2, uint, ushort, ushort);
template [[host_name("fn_moe_gateup_vnax_sg8")]] [[kernel]] void fn_moe_gateup_vnax<8>(const device half*,
    const device uchar*, const device uint*, const device uint4*, const device uint*, device float*,
    constant FnQ2Params&, uint2, uint, ushort, ushort);
template [[host_name("fn_moe_down_vnax_sg2")]] [[kernel]] void fn_moe_down_vnax<2>(const device float*,
    const device uchar*, const device uint*, const device uint4*, device float*, constant FnQ2Params&, uint2, uint,
    ushort, ushort);
template [[host_name("fn_moe_down_vnax_sg5")]] [[kernel]] void fn_moe_down_vnax<5>(const device float*,
    const device uchar*, const device uint*, const device uint4*, device float*, constant FnQ2Params&, uint2, uint,
    ushort, ushort);

// ---------------------------------------------------------------------------------------------
// Prefill HC fused (8-bit): the `up` rows are reordered at load (row d·4 + s = the original s·D + d),
// so a 64-column tile carries 16 values of d with their 4 streams and the mix comes out in the GEMM epilogue:
// xh[r, d] = mean_s n[r, s, d]·σ(g[r, s, d]) with n = (x·rr)·w recomputed from the streams (fn_gnorm_rr leaves rr). Neither g
// nor n goes through memory. Same sums in the same order per element as fn_gnorm + qmm_t_nax + fn_hc_mix.

// Like fn_gnorm, but without the fp32 output: only the half copy and rr[row, group] = rsqrt(mean(x²) + eps).
[[kernel]] void fn_gnorm_rr(const device float*    x [[buffer(0)]],
                            const device float*    w [[buffer(1)]],
                            device float*          rr [[buffer(2)]],
                            constant FnNormParams& p [[buffer(3)]],
                            device half*           oh [[buffer(4)]],
                            uint2 tg [[threadgroup_position_in_grid]],
                            uint tid [[thread_index_in_threadgroup]],
                            ushort sg [[simdgroup_index_in_threadgroup]],
                            ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[8];
    const device float* xr = x + size_t(tg.y) * p.ldx + tg.x * p.D;
    float ss = 0;
    for (uint i = tid; i < p.D; i += 256) ss += xr[i] * xr[i];
    ss = simd_sum(ss);
    if (lane == 0) part[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0;
    VUNROLL for (uint i = 0; i < 8; i++) tot += part[i];
    const float r = rsqrt(tot / float(p.D) + p.eps);
    if (tid == 0) rr[tg.y * 4 + tg.x] = r;
    device half* hrow = oh + size_t(tg.y) * p.ldo + tg.x * p.D;
    for (uint i = tid; i < p.D; i += 256) hrow[i] = half(xr[i] * r * w[tg.x * p.D + i]);
}


// 8 bits into threadgroup 4 at a time (half4 with fma; nax_load_w<8> goes value by value). For the HC GEMMs.
template <int rows>
METAL_FUNC void fn_load_w8(const device uint* w, const device bfloat* s, const device bfloat* b,
                           threadgroup half* ws, uint K, uint n0, uint k0, uint tid) {
    constexpr uint upr = NX_BK / 4;
    for (uint i = 0; i < (rows * upr) / NX_THREADS; i++) {
        const uint idx = tid + i * NX_THREADS;
        const uint r = idx / upr, u = idx % upr;
        const uint k = k0 + u * 4;
        const uint g = (n0 + r) * (K / 64) + k / 64;
        const half sc = half(float(s[g])), bi = half(float(b[g]));
        const uint v = w[size_t(n0 + r) * (K / 4) + k / 4];
        const half4 q = half4(as_type<uchar4>(v));
        *(threadgroup half4*)(ws + r * NX_LD + u * 4) = fma(q, half4(sc), half4(bi));
    }
}

// Y fp32 [M, N] = X · Wᵀ with 8-bit W (like qmm_t_nax_f32_b8_g64, with fn_load_w8): the HC down projection.
[[kernel, max_total_threads_per_threadgroup(NX_THREADS)]] void fn_qmm8_f32_nax(
    const device half*   x   [[buffer(0)]],
    const device uint*   w   [[buffer(1)]],
    const device bfloat* s   [[buffer(2)]],
    const device bfloat* b   [[buffer(3)]],
    device float*        y   [[buffer(4)]],
    constant QmmParams&  p   [[buffer(5)]],
    uint3  tgid [[threadgroup_position_in_grid]],
    uint   tid  [[thread_index_in_threadgroup]],
    ushort sgid [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]])
{
    constexpr int BN = 64;
    threadgroup half ws[BN * NX_LD];
    const uint m0 = tgid.y * NX_BM, n0 = tgid.x * BN;
    const uint tm = (sgid / 2) * 32, tn = (sgid % 2) * 32;
    const short2 c = nax_coord(lane);
    const device half* xs = x + size_t(m0 + tm) * p.K;
    nax_ffrag acc[2][2];
    NAX_UNROLL for (int i = 0; i < 2; i++) NAX_UNROLL for (int j = 0; j < 2; j++) acc[i][j] = 0;
    for (uint k0 = 0; k0 < p.K; k0 += NX_BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        fn_load_w8<BN>(w, s, b, ws, p.K, n0, k0, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        nax_hfrag a[2][4], bt[2][4];
        NAX_UNROLL for (int kk = 0; kk < 4; kk++) {
            NAX_UNROLL for (int i = 0; i < 2; i++) nax_load(a[i][kk], xs + size_t(i * 16) * p.K + k0 + kk * 16, p.K, c);
            NAX_UNROLL for (int j = 0; j < 2; j++) nax_load(bt[j][kk], ws + (tn + j * 16) * NX_LD + kk * 16, NX_LD, c);
        }
        NAX_UNROLL for (int i = 0; i < 2; i++)
            NAX_UNROLL for (int kk = 0; kk < 4; kk++) nax_mma2(acc[i][0], acc[i][1], a[i][kk], bt[0][kk], bt[1][kk]);
    }
    NAX_UNROLL for (int i = 0; i < 2; i++) NAX_UNROLL for (int j = 0; j < 2; j++) NAX_UNROLL for (int e = 0; e < 2; e++) {
        const uint r = m0 + tm + i * 16 + c.x + e * 8, col = n0 + tn + j * 16 + c.y;
        *(device float4*)(y + size_t(r) * p.N + col) =
            float4(acc[i][j][e * 4], acc[i][j][e * 4 + 1], acc[i][j][e * 4 + 2], acc[i][j][e * 4 + 3]);
    }
}

struct FnUpMixParams { uint M; uint rows; uint K; uint flags; };   // flags: bit 0 writes of (fp32)

// Up + HC mix of the prefill, with the mix from registers: each thread holds in a fragment the columns c.y..c.y+3
// (c.y a multiple of 4) = the 4 streams of the same d, so gt is not needed (17 KB of threadgroup memory that left
// one threadgroup per core). Same sums and same order: bit-identical.
[[kernel, max_total_threads_per_threadgroup(NX_THREADS)]] void fn_hc_upmix_nax_r(
    const device half*   x    [[buffer(0)]],
    const device uint*   w    [[buffer(1)]],
    const device bfloat* s    [[buffer(2)]],
    const device bfloat* b    [[buffer(3)]],
    const device float*  S    [[buffer(4)]],
    const device float*  rr   [[buffer(5)]],
    const device float*  nw   [[buffer(6)]],
    device half*         oh   [[buffer(7)]],
    device float*        of   [[buffer(8)]],
    constant FnUpMixParams& p [[buffer(9)]],
    uint3  tgid [[threadgroup_position_in_grid]],
    uint   tid  [[thread_index_in_threadgroup]],
    ushort sgid [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]])
{
    constexpr int BN = 64;
    threadgroup half ws[BN * NX_LD];
    const uint m0 = tgid.y * NX_BM, n0 = tgid.x * BN;
    const uint tm = (sgid / 2) * 32, tn = (sgid % 2) * 32;
    const short2 c = nax_coord(lane);
    const device half* xs = x + size_t(m0 + tm) * p.K;
    nax_ffrag acc[2][2];
    NAX_UNROLL for (int i = 0; i < 2; i++) NAX_UNROLL for (int j = 0; j < 2; j++) acc[i][j] = 0;
    for (uint k0 = 0; k0 < p.K; k0 += NX_BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        fn_load_w8<BN>(w, s, b, ws, p.K, n0, k0, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        nax_hfrag a[2][4], bt[2][4];
        NAX_UNROLL for (int kk = 0; kk < 4; kk++) {
            NAX_UNROLL for (int i = 0; i < 2; i++) nax_load(a[i][kk], xs + size_t(i * 16) * p.K + k0 + kk * 16, p.K, c);
            NAX_UNROLL for (int j = 0; j < 2; j++) nax_load(bt[j][kk], ws + (tn + j * 16) * NX_LD + kk * 16, NX_LD, c);
        }
        NAX_UNROLL for (int i = 0; i < 2; i++)
            NAX_UNROLL for (int kk = 0; kk < 4; kk++) nax_mma2(acc[i][0], acc[i][1], a[i][kk], bt[0][kk], bt[1][kk]);
    }
    NAX_UNROLL for (int i = 0; i < 2; i++) NAX_UNROLL for (int e = 0; e < 2; e++) {
        const uint r = m0 + tm + i * 16 + c.x + e * 8;
        if (r >= p.rows) continue;
        const float4 r4 = *(const device float4*)(rr + r * 4);
        NAX_UNROLL for (int j = 0; j < 2; j++) {
            const uint d = (n0 + tn + j * 16 + c.y) / 4;
            float a = 0;
            NAX_UNROLL for (uint q = 0; q < FN_S; q++) {
                const float n = S[size_t(r) * FN_W + q * FN_D + d] * r4[q] * nw[q * FN_D + d];
                a += n * fn_sigmoid(acc[i][j][e * 4 + q]);
            }
            a = a / float(FN_S);
            oh[size_t(r) * FN_D + d] = half(a);
            if (p.flags & 1) of[size_t(r) * FN_D + d] = a;
            else if (p.flags & 2) of[size_t(r) * FN_D + d] = 0;
        }
    }
}
