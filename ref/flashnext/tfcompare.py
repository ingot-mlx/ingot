"""Compares the engine greedy with the harness by forcing the same tokens (Flash-Next phase 0b).

The engine (`ingot-dev fncheck --emit DIR`) leaves its greedy and its logits (the last row of the prompt and each one-row round);
the harness reads prompt + those tokens (`ids_tf.npy`) and gives the logits of the same positions. So, after a difference, the
two still see the same context and can be compared position by position.

Usage: python -I tfcompare.py ENGINE_DIR HARNESS_DIR [--tau 0.5] [--other ENGINE_DIR2]
Per position: engine token, harness argmax, harness margin (logit 1st − 2nd) and engine margin, KL(harness ‖ engine).
Criterion: every position with a different argmax has a harness margin < tau.
"""
import argparse
import json
import os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('eng')
ap.add_argument('ref', nargs='?')
ap.add_argument('--tau', type=float, default=0.5)
ap.add_argument('--other', help='another engine emit (e.g. another chunk) to compare engine against engine')
a = ap.parse_args()


def lsm(x):
    x = x.astype(np.float64)
    m = x.max(-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))


def margin(x):
    t = np.partition(x, -2, axis=-1)
    return t[..., -1] - t[..., -2]


ours = np.load(os.path.join(a.eng, 'ours_logits.npy'))
meta = json.load(open(os.path.join(a.eng, 'ours_tokens.json')))
P, toks = meta['prompt'], meta['tokens']
N = ours.shape[0]
if a.other:
    oth = np.load(os.path.join(a.other, 'ours_logits.npy'))
    t2 = json.load(open(os.path.join(a.other, 'ours_tokens.json')))['tokens']
    same = next((i for i in range(N) if toks[i] != t2[i]), N)
    n = min(N, same + 1)
    kl = (np.exp(lsm(oth[:n])) * (lsm(oth[:n]) - lsm(ours[:n]))).sum(-1)
    print(f'engine vs engine: greedy equal in {same} of {N}; KL mean {kl.mean():.3g} max {kl.max():.3g} '
          f'(up to the 1st difference); margin at the 1st difference {margin(ours[min(same, N - 1)]):.3f}')
if not a.ref:
    raise SystemExit
rm = json.load(open(os.path.join(a.ref, 'meta.json')))
rows = rm['logits_rows']
rl = np.load(os.path.join(a.ref, 'logits.npy'))
idx = [rows.index(P - 1 + i) for i in range(N)]
ref = rl[idx]
lr, lo = lsm(ref), lsm(ours)
kl = (np.exp(lr) * (lr - lo)).sum(-1)
ra, oa = ref.argmax(-1), ours.argmax(-1)
mr, mo = margin(ref), margin(ours)
agree = (ra == oa)
print(f'{a.eng}: prompt {P}, {N} forced positions; top-1 equal {agree.mean() * 100:.1f}%; '
      f'KL mean {kl.mean():.4g}, p95 {np.percentile(kl, 95):.4g}, max {kl.max():.4g}')
bad = np.nonzero(~agree)[0]
for i in bad:
    print(f'  pos {i}: engine {oa[i]} (margin {mo[i]:.3f}), harness {ra[i]} (margin {mr[i]:.3f}), '
          f'harness logit for the engine one {ref[i, oa[i]] - ref[i, ra[i]]:+.3f}, KL {kl[i]:.4g}')
worst = float(mr[bad].max()) if len(bad) else 0.0
print(f'  harness margin at the differences: max {worst:.3f}; criterion margin < {a.tau}: '
      f'{"MET" if worst < a.tau else "NOT MET"}')
print(f'  harness margin: median {np.median(mr):.2f}, positions with margin < {a.tau}: {(mr < a.tau).sum()}')
