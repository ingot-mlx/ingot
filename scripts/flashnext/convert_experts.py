"""Rewrites the Q2_0 GSQ experts of the ISTA GGUF into experts.bin (fnformat.py), lossless.

Usage: python -I convert_experts.py <GGUF part 1> <output folder>
Resumable by layer (experts.bin.prog). Only reads the three expert tensors of each layer (mmap).
"""
import os
import sys
import json
import time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fnformat as F

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
meta, ts = F.read_gguf(src)
assert meta['general.architecture'] == 'qwen4exp' and meta['qwen4exp.expert_count'] == F.EXPERTS
mm = np.memmap(src, dtype=np.uint8, mode='r')
dst = os.path.join(out, 'experts.bin')
prog = dst + '.prog'
done = set(json.load(open(prog))) if os.path.exists(prog) else set()
total = F.LAYERS * F.EXPERTS * F.RECORD
fd = os.open(dst, os.O_RDWR | os.O_CREAT, 0o644)
if os.fstat(fd).st_size != total:
    os.ftruncate(fd, total)
t0 = time.time()
for L in range(F.LAYERS):
    if L in done:
        continue
    recs = np.zeros((F.EXPERTS, F.RECORD), dtype=np.uint8)
    for name, gname, (n, k) in (("gate", "ffn_gate_exps", (F.INTER, F.HIDDEN)), ("up", "ffn_up_exps", (F.INTER, F.HIDDEN)),
                                ("down", "ffn_down_exps", (F.HIDDEN, F.INTER))):
        dims, t, off = ts[f'blk.{L}.{gname}.weight']
        assert t == F.GGML_Q2_0 and dims == [k, n, F.EXPERTS], (L, gname, dims, t)
        per = n * (k // F.GROUP) * 18
        raw = mm[off:off + per * F.EXPERTS].reshape(F.EXPERTS, n, k // F.GROUP, 18)
        c0, s0 = F.OFFSETS[name]
        cb, sb = F.plane_sizes(n, k)
        recs[:, c0:c0 + cb] = raw[:, :, :, 2:].reshape(F.EXPERTS, cb)
        recs[:, s0:s0 + sb] = raw[:, :, :, :2].reshape(F.EXPERTS, sb)
    os.pwrite(fd, recs.tobytes(), L * F.EXPERTS * F.RECORD)
    os.fsync(fd)
    done.add(L)
    json.dump(sorted(done), open(prog, 'w'))
    print(f'layer {L}: {time.time() - t0:.0f} s', flush=True)
os.close(fd)
if len(done) == F.LAYERS and os.path.exists(prog):
    os.remove(prog)
print('DONE', total, 'bytes', flush=True)
