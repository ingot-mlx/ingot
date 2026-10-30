"""Checks that lab serve at temperature 0 gives the same tokens as lab-dev gen (same prompt, same model).

It checks what lab serve serves by default (what pi gets), so both sides run lab serve's engine options
(EngineOptions.serve): on MoE models the lossy prefill (W8A8 from the family's position and Q·Kᵀ in int8 from 8K,
--int8 1 --qk8 1), on dense models the exact one. lab serve starts without --int8/--qk8 and lab-dev gen gets the same
options by hand (exact gen against lossy serve diverged at 4K on the Gemma 4 26B, gpt-oss passed by luck).
lab-dev gen writes the options it ran with and lab serve's defaults for the model in each gen JSON; this script
fails if they differ (it cannot see the server's options: start lab serve without engine options).

First lab-dev gen, then `lab serve --model M` (never both at once):
  MoE:   lab-dev gen --model M --prompts chat,512,4K --n 256 --mode spec|plain --int8 1 --qk8 1
  dense: lab-dev gen --model M --prompts chat,512,4K --n 256 --mode spec|plain
(spec with a drafter, plain without one; leaves ref/out/gen/<name>_lab_<mode>.json). Experts read from the SSD
(Flash-Next): lab serve also skips missing experts by default, lossy and timing-dependent, so it starts with
--expert-skip-rank 0 (as lab-dev prefixcache). The text of each prompt is rebuilt as in prompts.py and sent as
a user message; the server applies its own template and tokenizer (the prompt ids must match
ref/out/<pdir>/<name>.json).

Usage:
  .venv/bin/python serve_check.py [--url http://127.0.0.1:8090] [--prompts chat,512,4K] [--pdir prompts]
                                 [--gen-mode spec|plain|normal]
"""
import argparse
import glob
import json
import os
import random
import urllib.request
import uuid

from prompts import TARGETS, QUESTION, CHAT
from moe_ref import HERE


def texts(seed="1", asyncio_dir=os.path.expanduser("~/.local/share/mise/installs/python/3.14/lib/python3.14/asyncio")):
    corpus = "".join(open(f).read() for f in sorted(glob.glob(asyncio_dir + "/*.py")))
    out, n = {"chat": CHAT}, 1
    for name, chars in TARGETS.items():
        n += 1
        sid = uuid.UUID(int=random.Random(f"{seed}-{n}").getrandbits(128), version=4)
        out[name] = f"[session {sid}]\n" + corpus[:chars] + QUESTION
    return out


def options_match(name, o):
    """lab-dev gen ran with lab serve's default prefill options (the gen JSON's `options`)?"""
    if o is None:
        print(f"{name}: the gen JSON has no engine options (written before the engine options were stored in it): rerun lab-dev gen")
        return False
    want = f"--int8 {int(o['serve_int8'])} --qk8 {int(o['serve_qk8'])}"
    if (o["int8"], o["qk8"]) != (o["serve_int8"], o["serve_qk8"]):
        print(f"{name}: OPTIONS DIFFER: lab-dev gen ran with --int8 {int(o['int8'])} --qk8 {int(o['qk8'])} and "
              f"lab serve defaults to {want} on this model: rerun lab-dev gen with {want}")
        return False
    print(f"{name}: lab-dev gen with lab serve's defaults ({want}); lab serve must run without --int8/--qk8"
          + (" and with --expert-skip-rank 0 (experts from the SSD)" if o.get("ssd_experts") else ""))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8090")
    ap.add_argument("--prompts", default="chat,512,4K")
    ap.add_argument("--pdir", default="prompts")
    ap.add_argument("--gen-mode", default="spec", help="lab-dev gen mode to compare with (plain for models without a drafter)")
    a = ap.parse_args()
    ok = True
    t = texts()
    for name in a.prompts.split(","):
        ref_prompt = json.load(open(os.path.join(HERE, "out", a.pdir, f"{name}.json")))["ids"]
        gen = json.load(open(os.path.join(HERE, "out", "gen", f"{name}_lab_{a.gen_mode}.json")))
        ref = gen["ids"]
        ok = options_match(name, gen.get("options")) and ok
        body = {"model": "x", "messages": [{"role": "user", "content": t[name]}], "max_tokens": len(ref),
                "temperature": 0, "lab_return_tokens": True,
                # Neutral explicitly: lab serve takes min_p and the penalties from generation_config.json (LFM2.5
                # has repetition_penalty 1.05) and lab-dev gen does not apply them.
                "min_p": 0, "repetition_penalty": 1, "presence_penalty": 0, "frequency_penalty": 0}
        req = urllib.request.Request(a.url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
        r = json.load(urllib.request.urlopen(req, timeout=600))
        got, gp = r["lab_tokens"], r["lab_prompt_tokens"]
        same_p = gp == ref_prompt
        k = min(len(got), len(ref))
        div = next((i for i in range(k) if got[i] != ref[i]), None)
        same = div is None and len(got) == len(ref)
        ok = ok and same and same_p
        print(f"{name}: prompt {len(gp)} tok ({'same' if same_p else 'DIFFERENT'} as prompts.py), "
              f"{len(got)} tokens: {'identical to lab-dev gen' if same else f'DIVERGE at token {div}'}")
    print("serve_check:", "OK" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
