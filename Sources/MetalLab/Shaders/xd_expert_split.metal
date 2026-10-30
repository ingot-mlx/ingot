// Splits the expert union of vmoe_route into the resident experts (slot in the resident table, `smapR`, which the
// CPU only changes at sync points) and the missing ones, with their indirect grids (same format as the
// `args` of vmoe_route). The GPU computes the residents while the CPU fetches the missing ones; each pair writes its own
// h and its own `part` slot, and vmoe_combine9 sums in the same order: bit-identical to the unsplit version.
[[kernel]] void expert_split(const device uint4* entries  [[buffer(0)]],   // [U + 1] (the shared one at the end)
                          const device uint*  args     [[buffer(1)]],   // [tiles, U+1, 1, U+1, blocks, U+1, 1]
                          const device uint*  smapR    [[buffer(2)]],   // [E] slot or 0xffffffff
                          device uint4*       entriesR [[buffer(3)]],
                          device uint*        argsR    [[buffer(4)]],
                          device uint4*       entriesM [[buffer(5)]],
                          device uint*        argsM    [[buffer(6)]],
                          constant uint&      E        [[buffer(7)]],
                          uint tid [[thread_index_in_threadgroup]],
                          ushort sg [[simdgroup_index_in_threadgroup]],
                          ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup uint t_tot[8];
    const uint n = args[1];
    uint4 en = uint4(0);
    uint res = 0;
    if (tid < n) {
        en = entries[tid];
        res = (en.x >= E || smapR[en.x] != 0xffffffffu) ? 1u : 0u;
    }
    const uint pr = simd_prefix_exclusive_sum(res);
    if (lane == 31) t_tot[sg] = pr + res;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint base = 0, nR = 0;
    for (uint s = 0; s < 8; s++) { if (s < sg) base += t_tot[s]; nR += t_tot[s]; }
    if (tid < n) {
        const uint ir = base + pr;
        if (res) entriesR[ir] = en; else entriesM[tid - ir] = en;
    }
    if (tid == 0) {
        const uint nM = n - nR;
        argsR[0] = args[0]; argsR[1] = nR; argsR[2] = 1; argsR[3] = nR;
        argsR[4] = args[4]; argsR[5] = nR; argsR[6] = 1;
        argsM[0] = args[0]; argsM[1] = nM; argsM[2] = 1; argsM[3] = nM;
        argsM[4] = args[4]; argsM[5] = nM; argsM[6] = 1;
    }
}
