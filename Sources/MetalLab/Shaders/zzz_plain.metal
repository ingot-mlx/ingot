// Chained decode without a drafter (B = 1; LFM2): at the end of each round the GPU picks the token (argmax/sampled) and
// leaves it as the input of the next round, so the CPU enqueues round N+1 before waiting for N. Same state and
// record as spec_accept (yy_model.metal): st = [P, nacc, -, -, last token, round]; rec[round % ring] =
// (0, token, …, P before in [REC_P]). Goes after zz_sample.metal (sample_dist, sample_pick).

[[kernel]] void plain_accept(device uint*       st   [[buffer(0)]],
                             device uint*       vtok [[buffer(1)]],
                             const device uint* top1 [[buffer(2)]],
                             device uint*       rec  [[buffer(4)]],
                             constant uint&     ring [[buffer(5)]],
                             uint tid [[thread_position_in_grid]]) {
    if (tid != 0) return;
    const uint t = top1[0];
    device uint* r = rec + (st[5] % ring) * REC_STRIDE;
    r[0] = 0; r[1] = t; r[REC_P] = st[0];
    st[0] += 1; st[1] = 1; st[4] = t; st[5] += 1;
    vtok[0] = t;
}

// With temperature: rnd has RND_PER_ROUND uniforms per round (the second is used, as in spec_accept_sample's resample).
[[kernel]] void plain_accept_sample(device uint*        st   [[buffer(0)]],
                                    device uint*        vtok [[buffer(1)]],
                                    const device uint*  cidx [[buffer(2)]],
                                    const device float* cval [[buffer(3)]],
                                    const device float* z    [[buffer(4)]],
                                    device uint*        rec  [[buffer(6)]],
                                    const device float* rnd  [[buffer(7)]],
                                    constant SampleParams& p [[buffer(8)]],
                                    constant uint&      ring [[buffer(9)]],
                                    uint tid [[thread_position_in_grid]]) {
    if (tid != 0) return;
    const device float* u = rnd + size_t(st[5]) * RND_PER_ROUND;
    float pr[SK];
    const uint n = sample_dist(cidx, cval, z[0], z[1], p, pr);
    const uint t = sample_pick(cidx, pr, n, 0xffffffffu, 0, u[1]);
    device uint* r = rec + (st[5] % ring) * REC_STRIDE;
    r[0] = 0; r[1] = t; r[REC_P] = st[0];
    st[0] += 1; st[1] = 1; st[4] = t; st[5] += 1;
    vtok[0] = t;
}
