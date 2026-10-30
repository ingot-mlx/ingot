"""Reference greedy decoding with mlx-lm and penalties with vLLM/OpenAI semantics (no window), to compare against
`ingot serve`.

Same as gen_ref.py (ids from `out/<pdir>/<name>.json`, prefill in chunks, argmax), but each step's logits (fp32) are
penalized before the argmax, in the engine's order:
  repetition over prompt + generated tokens (logit > 0 => / r, else * r),
  frequency over the generated tokens (- f per occurrence),
  presence over the generated tokens (- p once).
mlx-lm's own processors (make_logits_processors) use a window of the last 20 tokens including the prompt; vLLM and
ingot serve use the whole text, so they are reimplemented here.

With --forced, runs the reference over the tokens ingot serve gave (teacher forcing, see forced()).
With --url, also asks `ingot serve` (temperature 0, same penalties, the prompt text rebuilt as serve_check.py does)
and prints the first divergence and the reference margin there. Never run both at once: --url only reads
`out/<out>/<name>_mlx_pen.json` written by an earlier run.

Usage:
  .venv/bin/python penalty_ref.py --prompt chat --n 256 --rep 1.3 --pres 1.5 --freq 0.5
  .venv/bin/python penalty_ref.py --prompt chat --n 256 --rep 1.3 --pres 1.5 --freq 0.5 --url http://127.0.0.1:8090
  .venv/bin/python penalty_ref.py --prompt chat --rep 1.3 --pres 1.5 --freq 0.5 --forced
"""
import argparse
import json
import os
import time
import urllib.request

from moe_ref import resolve, HERE


def tag(a):
    return f"{a.prompt}_r{a.rep:g}_p{a.pres:g}_f{a.freq:g}"


def penalize(logits, prompt_set, counts, a, mx):
    """logits: fp32 [V] (mlx array). prompt_set: ids in the prompt; counts: {id: occurrences among generated}."""
    touched = sorted(set(counts) | (prompt_set if a.rep != 1 else set()))
    if not touched:
        return logits
    idx = mx.array(touched)
    x = logits[idx]
    if a.rep != 1:
        x = mx.where(x > 0, x / a.rep, x * a.rep)
    c = mx.array([counts.get(t, 0) for t in touched], dtype=mx.float32)
    x = x - a.freq * c - a.pres * (c > 0).astype(mx.float32)
    logits[idx] = x
    return logits


def reference(a):
    import mlx.core as mx
    from mlx_lm.utils import load
    from mlx_lm.models.cache import make_prompt_cache

    ids = json.load(open(os.path.join(HERE, "out", a.pdir, f"{a.prompt}.json")))["ids"]
    model, tok = load(resolve(a.model))
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
    prompt_set, counts, out, margins = set(ids), {}, [], []
    last = logits[0, -1].astype(mx.float32)
    for _ in range(a.n):
        last = penalize(last, prompt_set, counts, a, mx)
        top2 = sorted(mx.topk(last, 2).tolist(), reverse=True)
        y = int(mx.argmax(last).item())
        margins.append(round(top2[0] - top2[1], 4))
        out.append(y)
        counts[y] = counts.get(y, 0) + 1
        if y in eos:
            break
        logits = model(mx.array([[y]]), cache=cache)
        last = logits[0, -1].astype(mx.float32)
    t2 = time.time()
    os.makedirs(os.path.join(HERE, "out", a.out), exist_ok=True)
    with open(os.path.join(HERE, "out", a.out, f"{tag(a)}_mlx_pen.json"), "w") as f:
        json.dump({"prompt": a.prompt, "rep": a.rep, "pres": a.pres, "freq": a.freq, "ids": out,
                   "margins": margins}, f)
    print(f"prefill {len(ids)} tokens {t1 - t0:.2f} s; {len(out)} tokens {t2 - t1:.2f} s")
    print(repr(tok.decode(out)))


def forced(a):
    """Teacher forcing over the ids that ingot serve returned (`<tag>_ingot_serve.json`): at each position, the penalized
    reference logits with ingot's own history; reports where ingot's token is not the reference argmax and by how much
    (reference logit of its argmax minus that of ingot's token)."""
    import mlx.core as mx
    from mlx_lm.utils import load
    from mlx_lm.models.cache import make_prompt_cache

    ids = json.load(open(os.path.join(HERE, "out", a.pdir, f"{a.prompt}.json")))["ids"]
    got = json.load(open(os.path.join(HERE, "out", a.out, f"{tag(a)}_ingot_serve.json")))["ids"]
    model, _ = load(resolve(a.model))
    cache = make_prompt_cache(model)
    logits = None
    for s in range(0, len(ids), a.chunk):
        logits = model(mx.array(ids[s: s + a.chunk])[None], cache=cache)
        mx.eval(logits)
    rows = [logits[0, -1]]
    if len(got) > 1:
        rows += list(model(mx.array(got[:-1])[None], cache=cache)[0])
    prompt_set, counts, diffs = set(ids), {}, []
    for i, y in enumerate(got):
        x = penalize(rows[i].astype(mx.float32), prompt_set, counts, a, mx)
        best = int(mx.argmax(x).item())
        if best != y:
            diffs.append((i, round(float((x[best] - x[y]).item()), 4)))
        counts[y] = counts.get(y, 0) + 1
    worst = max((d for _, d in diffs), default=0.0)
    print(f"{tag(a)} forced over {len(got)} ingot tokens: {len(diffs)} not the reference argmax, worst gap {worst:.4f}"
          + (f" at {[i for i, _ in diffs]}" if diffs else ""))


def compare(a):
    from serve_check import texts

    ref = json.load(open(os.path.join(HERE, "out", a.out, f"{tag(a)}_mlx_pen.json")))
    ref_prompt = json.load(open(os.path.join(HERE, "out", a.pdir, f"{a.prompt}.json")))["ids"]
    body = {"model": "x", "messages": [{"role": "user", "content": texts()[a.prompt]}],
            "max_tokens": len(ref["ids"]), "temperature": 0, "ingot_return_tokens": True, "min_p": 0,
            "repetition_penalty": a.rep, "presence_penalty": a.pres, "frequency_penalty": a.freq}
    req = urllib.request.Request(a.url + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=900))
    got, gp, want = r["ingot_tokens"], r["ingot_prompt_tokens"], ref["ids"]
    k = min(len(got), len(want))
    div = next((i for i in range(k) if got[i] != want[i]), None)
    same_p = gp == ref_prompt
    if div is None:
        res = f"identical over {k} tokens" + ("" if len(got) == len(want) else f" (lengths {len(got)}/{len(want)})")
    else:
        res = f"same {div} tokens, diverge at {div} (reference margin there {ref['margins'][div]:.4f})"
    print(f"{tag(a)}: prompt {len(gp)} tok ({'same' if same_p else 'DIFFERENT'} as prompts.py); {res}")
    with open(os.path.join(HERE, "out", a.out, f"{tag(a)}_ingot_serve.json"), "w") as f:
        json.dump({"ids": got, "first_divergence": div}, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="~/Models/Qwen3.6-35B-A3B-Heretic-MLX-4bit-calib")
    ap.add_argument("--prompt", default="chat")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--rep", type=float, default=1.0, help="repetition_penalty (1 = off)")
    ap.add_argument("--pres", type=float, default=0.0, help="presence_penalty (0 = off)")
    ap.add_argument("--freq", type=float, default=0.0, help="frequency_penalty (0 = off)")
    ap.add_argument("--pdir", default="prompts", help="subdirectory of out/ with the prompts")
    ap.add_argument("--out", default="gen", help="subdirectory of out/ for the result")
    ap.add_argument("--url", help="compare a running ingot serve against the saved reference (no MLX)")
    ap.add_argument("--forced", action="store_true", help="teacher forcing over the ids saved by --url")
    a = ap.parse_args()
    if a.url:
        compare(a)
    elif a.forced:
        forced(a)
    else:
        reference(a)


if __name__ == "__main__":
    main()
