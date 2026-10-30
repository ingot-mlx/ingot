"""Prompts of the benchmark tool (`bench.py`) tokenized with the Qwen3.6 chat template.

Reproduces the bench.py generator: corpus = asyncio sources of the Python the bench is launched with (mise python3,
3.14), trimmed to 1600 / 13000 / 52000 / 104000 characters, with a `[session <uuid>]` prefix (with
BENCH_SEED=1 the uuid is deterministic) and the final question. The chat template is applied with reasoning
enabled (the splash default mode), and the ids are written to `out/prompts/<name>.json`, along with a short
conversation prompt (`chat`) for the correctness tests.

Usage:
  .venv/bin/python prompts.py [--asyncio-dir DIR] [--seed 1] [--model DIR --out prompts-<x>] [--fixed-now]
  (gpt-oss: --model ~/Models/gpt-oss-20b-MXFP4-Q8 --out prompts-gptoss --fixed-now)
"""
import argparse
import glob
import json
import os
import random
import uuid

from transformers import AutoTokenizer

from moe_ref import resolve, HERE

TARGETS = {"512": 1600, "4K": 13000, "16K": 52000, "32K": 104000}
QUESTION = ("\n\nBased on the code above, explain how the event loop schedules callbacks, "
            "then write a small example using asyncio.gather with error handling.")
CHAT = "Explica en pocas frases qué es la decodificación especulativa y por qué el resultado es idéntico al normal."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="~/Models/Qwen3.6-35B-A3B-Heretic-MLX-4bit-calib")
    ap.add_argument("--asyncio-dir", default=os.path.expanduser(
        "~/.local/share/mise/installs/python/3.14/lib/python3.14/asyncio"))
    ap.add_argument("--seed", default="1")
    ap.add_argument("--out", default="prompts", help="subdirectory of out/ (prompts-q27 for the 27B)")
    ap.add_argument("--fixed-now", action="store_true",
                    help="pin the template clock (strftime_now) to tok_ref.FIXED_NOW, for templates that print the "
                         "date (gpt-oss), so the prompts can be regenerated identical any day")
    a = ap.parse_args()
    if a.fixed_now:
        from tok_ref import pin_clock
        pin_clock()
    tok = AutoTokenizer.from_pretrained(resolve(a.model))
    corpus = "".join(open(f).read() for f in sorted(glob.glob(a.asyncio_dir + "/*.py")))
    out = os.path.join(HERE, "out", a.out)
    os.makedirs(out, exist_ok=True)
    n = 1  # bench.py: request 1 is the warm-up; 512 uses 2, 4K uses 3...
    prompts = {"chat": CHAT}
    for name, chars in TARGETS.items():
        n += 1
        sid = uuid.UUID(int=random.Random(f"{a.seed}-{n}").getrandbits(128), version=4)
        prompts[name] = f"[session {sid}]\n" + corpus[:chars] + QUESTION
    for name, text in prompts.items():
        ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True)
        if not isinstance(ids, list):
            ids = ids["input_ids"]
        with open(os.path.join(out, f"{name}.json"), "w") as f:
            json.dump({"name": name, "chars": len(text), "ids": list(map(int, ids))}, f)
        print(f"{name}: {len(ids)} tokens; final: {tok.decode(ids[-8:])!r}")


if __name__ == "__main__":
    main()
