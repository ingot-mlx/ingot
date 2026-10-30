"""Checks that experts.bin is the lossless Q2_0 of the GGUF.

1) All experts: the code and scale bytes of each record are those of the GGUF block and the padding is zero.
2) Sample of N experts (2000 by default): the GGUF reference dequantization (byte by byte, d·(q−1)) and that of
   our format with the MLX reading give the same float32, bit-exact.
Usage: python -I verify_experts.py <GGUF part 1> <experts.bin> [N]
"""
import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fnformat as F

src, binp = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
meta, ts = F.read_gguf(src)
mm = np.memmap(src, dtype=np.uint8, mode='r')
eb = np.memmap(binp, dtype=np.uint8, mode='r')
assert eb.size == F.LAYERS * F.EXPERTS * F.RECORD, eb.size
mats = (("gate", "ffn_gate_exps", F.INTER, F.HIDDEN), ("up", "ffn_up_exps", F.INTER, F.HIDDEN),
        ("down", "ffn_down_exps", F.HIDDEN, F.INTER))
bad = 0
for L in range(F.LAYERS):
    recs = eb[L * F.EXPERTS * F.RECORD:(L + 1) * F.EXPERTS * F.RECORD].reshape(F.EXPERTS, F.RECORD)
    for name, g, n, k in mats:
        dims, t, off = ts[f'blk.{L}.{g}.weight']
        raw = mm[off:off + n * (k // 64) * 18 * F.EXPERTS].reshape(F.EXPERTS, n, k // 64, 18)
        c0, s0 = F.OFFSETS[name]
        cb, sb = F.plane_sizes(n, k)
        ok = np.array_equal(recs[:, c0:c0 + cb], raw[:, :, :, 2:].reshape(F.EXPERTS, cb)) and \
            np.array_equal(recs[:, s0:s0 + sb], raw[:, :, :, :2].reshape(F.EXPERTS, sb))
        bad += not ok
    bad += int(np.any(recs[:, F.PAYLOAD:]))
print('bytes: layers x different matrices =', bad)
rng = np.random.default_rng(20261021)
picks = rng.choice(F.LAYERS * F.EXPERTS, size=N, replace=False)
maxdiff = 0.0
nexact = 0
for p in picks:
    L, e = divmod(int(p), F.EXPERTS)
    rec = eb[(L * F.EXPERTS + e) * F.RECORD:(L * F.EXPERTS + e + 1) * F.RECORD]
    planes = F.record_planes(rec)
    ok = True
    for name, g, n, k in mats:
        dims, t, off = ts[f'blk.{L}.{g}.weight']
        per = n * (k // 64) * 18
        ref = F.dequant_gguf_q2(mm[off + e * per:off + (e + 1) * per], n, k)
        c, s, n2, k2 = planes[name]
        ours = F.dequant_planar_q2(c, s, n, k)
        ok &= ref.tobytes() == ours.tobytes()
        maxdiff = max(maxdiff, float(np.max(np.abs(ref - ours))))
    nexact += ok
print(f'sample: {nexact}/{N} experts bit-exact (max |diff| {maxdiff})')
sys.exit(0 if bad == 0 and nexact == N else 1)
