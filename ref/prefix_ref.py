"""Last-position logits of several prefixes of a prompt with mlx-lm, in bf16 (as it runs) and with fp32 activations
(precision reference), to measure the exactness of the lab engine prefill (lab-dev kl --ref mlx-prefix). Writes
out/<gen>/<prompt>_prefixes.json (lengths), _prefix_bf16.npy and _prefix_f32.npy (fp32 [n, V]), and the bf16 KL.

Usage:
  .venv/bin/python prefix_ref.py --model <model-dir> --prompt 4K --n 40
"""
import argparse
import json
import os

import numpy as np
import mlx.core as mx
from mlx_lm.utils import load

from moe_ref import resolve, HERE


def ls(x):
    m = x.max(1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(1, keepdims=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pdir", default="prompts-lfm")
    ap.add_argument("--gen", default="gen-lfm")
    ap.add_argument("--prompt", default="4K")
    ap.add_argument("--n", type=int, default=40)
    a = ap.parse_args()
    ids = json.load(open(os.path.join(HERE, "out", a.pdir, f"{a.prompt}.json")))["ids"]
    lens = sorted(set(int(x) for x in np.linspace(24, len(ids), a.n)))
    res = {}
    for dt in ["bf16", "f32"]:
        model, _ = load(resolve(a.model))
        if dt == "f32":
            model.set_dtype(mx.float32)
        rows = []
        for L in lens:
            lg = model(mx.array(ids[:L])[None])[0, -1].astype(mx.float32)
            rows.append(np.array(lg))
        res[dt] = np.stack(rows).astype(np.float32)
        np.save(os.path.join(HERE, "out", a.gen, f"{a.prompt}_prefix_{dt}.npy"), res[dt])
        del model
    json.dump({"lens": lens}, open(os.path.join(HERE, "out", a.gen, f"{a.prompt}_prefixes.json"), "w"))
    la, lb = ls(res["f32"].astype(np.float64)), ls(res["bf16"].astype(np.float64))
    kl = (np.exp(la) * (la - lb)).sum(1)
    print(f"{a.prompt}: {len(lens)} prefixes; KL(fp32 ‖ mlx-lm bf16) mean {kl.mean():.5f} median {np.median(kl):.5f} "
          f"max {kl.max():.4f}; top-1 equal {(la.argmax(1) == lb.argmax(1)).sum()}/{len(kl)}")


if __name__ == "__main__":
    main()
