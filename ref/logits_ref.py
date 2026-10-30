"""mlx-lm reference logits to measure the lab's KL (lab-dev kl --ref mlx): last prompt position and n steps
forced with mlx-lm's greedy (out/<gen>/<prompt>_mlx.json from gen_ref.py). Writes out/<gen>/<prompt>_logits.npy
(fp32 [n + 1, V], mlx-lm's bf16 output converted). With --f32, the model with every activation in fp32 (the
quantized weights do not change): precision reference to compare mlx-lm bf16 and the lab
(out/<gen>/<prompt>_logits_f32.npy).

Usage:
  .venv/bin/python logits_ref.py --model ~/Models/LFM2.5-24B-A2B-MLX-4bit-calib --pdir prompts-lfm --gen gen-lfm --prompt 512 --n 64
"""
import argparse
import json
import os

import numpy as np
import mlx.core as mx
from mlx_lm.utils import load
from mlx_lm.models.cache import make_prompt_cache

from moe_ref import resolve, HERE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pdir", default="prompts-lfm")
    ap.add_argument("--gen", default="gen-lfm")
    ap.add_argument("--prompt", default="512")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--f32", action="store_true", help="fp32 activations (precision reference): _logits_f32.npy")
    a = ap.parse_args()
    ids = json.load(open(os.path.join(HERE, "out", a.pdir, f"{a.prompt}.json")))["ids"]
    force = json.load(open(os.path.join(HERE, "out", a.gen, f"{a.prompt}_mlx.json")))["ids"][: a.n]
    model, _ = load(resolve(a.model))
    if a.f32:
        model.set_dtype(mx.float32)
    cache = make_prompt_cache(model)
    for s in range(0, len(ids), a.chunk):
        logits = model(mx.array(ids[s: s + a.chunk])[None], cache=cache)
        mx.eval(logits)
    rows = [np.array(logits[0, -1].astype(mx.float32))]
    for t in force:
        logits = model(mx.array([[t]]), cache=cache)
        rows.append(np.array(logits[0, -1].astype(mx.float32)))
    out = np.stack(rows).astype(np.float32)
    np.save(os.path.join(HERE, "out", a.gen, f"{a.prompt}_logits{'_f32' if a.f32 else ''}.npy"), out)
    print(f"{a.prompt}: {out.shape}")


if __name__ == "__main__":
    main()
