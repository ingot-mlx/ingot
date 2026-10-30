"""Reference greedy decoding with mlx-lm (bf16) to compare against the lab engine.

Reads the ids from `out/prompts/<name>.json`, prefills in chunks and generates N tokens with argmax. Writes
`out/gen-qwen36/<name>_mlx.json` (--out) with the generated ids (and, with --logits, the top1-top2 margin of each step, useful to
explain where the lab engine diverges).

Usage:
  .venv/bin/python gen_ref.py --prompt chat --n 128
"""
import argparse
import json
import os
import time

import mlx.core as mx
from mlx_lm.utils import load
from mlx_lm.models.cache import make_prompt_cache

from moe_ref import resolve, HERE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="~/Models/Qwen3.6-35B-A3B-Heretic-MLX-4bit-calib")
    ap.add_argument("--prompt", default="chat")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--pdir", default="prompts", help="subdirectory of out/ with the prompts (prompts-lfm, prompts-q27)")
    ap.add_argument(
        "--out", default="gen-qwen36", help="subdirectory of out/ for the result (gen is lab-dev gen's output)"
    )
    a = ap.parse_args()
    ids = json.load(open(os.path.join(HERE, "out", a.pdir, f"{a.prompt}.json")))["ids"]
    model, tok = load(resolve(a.model))
    # EOS from generation_config.json or, if missing (mlx-community Qwen3-30B-A3B), from config.json.
    gc = os.path.join(resolve(a.model), "generation_config.json")
    eos = json.load(open(gc if os.path.exists(gc) else os.path.join(resolve(a.model), "config.json"))).get(
        "eos_token_id", [248044, 248046])
    eos = set(eos if isinstance(eos, list) else [eos])
    cache = make_prompt_cache(model)
    t0 = time.time()
    logits = None
    for s in range(0, len(ids), a.chunk):
        logits = model(mx.array(ids[s: s + a.chunk])[None], cache=cache)
        mx.eval(logits)
    t1 = time.time()
    out, margins = [], []
    last = logits[0, -1].astype(mx.float32)
    for _ in range(a.n):
        top2 = mx.topk(last, 2)
        y = int(mx.argmax(last).item())
        srt = sorted(top2.tolist(), reverse=True)
        margins.append(round(srt[0] - srt[1], 4))
        out.append(y)
        if y in eos:
            break
        logits = model(mx.array([[y]]), cache=cache)
        last = logits[0, -1].astype(mx.float32)
    t2 = time.time()
    os.makedirs(os.path.join(HERE, "out", a.out), exist_ok=True)
    with open(os.path.join(HERE, "out", a.out, f"{a.prompt}_mlx.json"), "w") as f:
        json.dump({"prompt": a.prompt, "ids": out, "margins": margins}, f)
    print(f"prefill {len(ids)} tokens {t1 - t0:.2f} s; {len(out)} tokens {t2 - t1:.2f} s")
    print(repr(tok.decode(out)))


if __name__ == "__main__":
    main()
