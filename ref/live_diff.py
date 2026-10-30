"""Where the next prompt stops matching the live state, from a `ingot serve --dump-requests DIR` folder (no GPU).

For each request n that has a NNNN-output.json, the live state it leaves is prompt_ids + output_ids (the server keeps
all but the last emitted token). The next request's prompt ids (its own NNNN-output.json) are compared with it:
  c       = longest common prefix of the live state and the next prompt;
  where   = "live" (the whole live state is a prefix), "answer k" (token k of the answer) or "prompt" (before the
            answer: another conversation, or the template rewrote earlier history);
  kind    = (a) same text, other ids: decode(model ids) == decode(prompt ids) up to the point where both resync,
                and, with --tokenizer, whether the prompt side is the canonical encoding of that text;
            (b) different text (template, output parser, tojson or the client);
            (c) divergence before the answer.
Besides the first divergence, every later (a) spot inside the answer is counted (resyncing after each one), for the
rate of non-canonical tokens per 1K generated tokens.

Usage:
  ref/.venv/bin/python -I ref/live_diff.py DIR --tokenizer MODEL_DIR [--json OUT] [--ctx 8]
"""
import argparse
import json
import os
import sys

RESYNC = 4  # ids that must match again after a divergence for it to count as resynced
WINDOW = 96  # how far (in ids on each side) a resync is looked for


def load(d):
    out = {}
    for f in sorted(os.listdir(d)):
        if f.endswith("-output.json"):
            with open(os.path.join(d, f)) as fh:
                out[int(f[:4])] = json.load(fh)
    return out


def lcp(a, b, i=0, j=0):
    k = 0
    while i + k < len(a) and j + k < len(b) and a[i + k] == b[j + k]:
        k += 1
    return k


def resync(a, b, i, j, dec):
    """Smallest (di, dj) after a divergence at a[i] != b[j] with equal text and RESYNC equal ids after it."""
    best = None
    for di in range(1, min(WINDOW, len(a) - i) + 1):
        ta = dec(a[i : i + di])
        for dj in range(1, min(WINDOW, len(b) - j) + 1):
            if best and di + dj >= best[0] + best[1]:
                break
            tb = dec(b[j : j + dj])
            if len(tb) > len(ta) + 8:
                break
            if ta == tb and (i + di == len(a) or lcp(a, b, i + di, j + dj) >= min(RESYNC, len(a) - i - di)):
                best = (di, dj)
                break
    return best


def classify(live, ans0, nxt, tok, ctx):
    dec = (lambda ids: tok.decode(ids, skip_special_tokens=False)) if tok else (lambda ids: repr(ids))
    c = lcp(live, nxt)
    r = {"c": c, "live_len": len(live), "answer_start": ans0, "answer_len": len(live) - ans0}
    if c == len(live):
        r.update(where="live", kind="-", spots=0)
        return r
    if c < ans0:
        r.update(where="prompt", kind="c", spots=0, at_prompt=c)
        r["model"] = dec(live[max(0, c - ctx) : c + ctx])
        r["prompt"] = dec(nxt[max(0, c - ctx) : c + ctx])
        return r
    r.update(where="answer %d" % (c - ans0))
    rs = resync(live, nxt, c, c, dec) if tok else None
    r["model_ids"] = live[c : c + (rs[0] if rs else ctx)]
    r["prompt_ids"] = nxt[c : c + (rs[1] if rs else ctx)]
    r["before"] = dec(live[max(ans0, c - ctx) : c])
    r["model"] = dec(r["model_ids"])
    r["prompt"] = dec(r["prompt_ids"])
    if not rs:
        r.update(kind="b", spots=1)
        r["model"] = dec(live[c : c + ctx])
        r["prompt"] = dec(nxt[c : c + ctx])
        return r
    seg = dec(r["model_ids"])
    r["prompt_is_canonical"] = tok.encode(seg, add_special_tokens=False).ids == r["prompt_ids"]
    r["model_is_canonical"] = tok.encode(seg, add_special_tokens=False).ids == r["model_ids"]
    r["kind"] = "a"
    # Later spots inside the answer, resyncing after each one (until the end of the live state or a text divergence).
    spots, i, j = 1, c + rs[0], c + rs[1]
    while True:
        k = lcp(live, nxt, i, j)
        i, j = i + k, j + k
        if i >= len(live):
            break
        s = resync(live, nxt, i, j, dec)
        if not s:
            r["later_text_diff_at_answer"] = i - ans0
            break
        spots += 1
        i, j = i + s[0], j + s[1]
    r["spots"] = spots
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dir")
    ap.add_argument("--tokenizer", help="model folder with tokenizer.json (decode and canonical check)")
    ap.add_argument("--json", help="write the rows here")
    ap.add_argument("--ctx", type=int, default=8)
    a = ap.parse_args()
    tok = None
    if a.tokenizer:
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(os.path.join(a.tokenizer, "tokenizer.json"))
    outs = load(a.dir)
    keys = sorted(outs)
    rows, gen = [], 0
    for n, m in zip(keys, keys[1:]):
        o, nx = outs[n], outs[m]
        ids = o["output_ids"]
        gen += len(ids)
        live = o["prompt_ids"] + ids[:-1]
        r = classify(live, len(o["prompt_ids"]), nx["prompt_ids"], tok, a.ctx)
        r.update(req=n, next=m, generated=len(ids), finish=o["finish"], next_source=nx["cache_source"],
                 next_cached=nx["cached"], next_rolled_back_from=nx.get("rolled_back_from"),
                 next_live_miss=nx.get("live_miss"), reread=max(0, min(len(live), len(nx["prompt_ids"])) - nx["cached"])
                 if r["where"] != "prompt" else 0)
        rows.append(r)
        extra = ""
        if r["kind"] == "a":
            extra = " model %s %r | prompt %s %r (prompt canonical %s, model canonical %s) spots %d" % (
                r["model_ids"], r["model"], r["prompt_ids"], r["prompt"], r["prompt_is_canonical"],
                r["model_is_canonical"], r["spots"])
        elif r["kind"] in "bc":
            extra = " model %r | prompt %r" % (r.get("model"), r.get("prompt"))
        print("%04d->%04d gen %5d %-14s kind %s next %s %d (reread %d)%s" % (
            n, m, len(ids), r["where"], r["kind"], r["next_source"], r["next_cached"], r["reread"], extra))
    if keys:
        gen += len(outs[keys[-1]]["output_ids"])
    miss = [r for r in rows if r["where"] != "live"]
    spots = sum(r.get("spots", 0) for r in rows if r["kind"] == "a")
    print("pairs %d, live %d, not live %d (a %d, b %d, c %d); non-canonical spots %d in %d generated (%.2f per 1K)" % (
        len(rows), len(rows) - len(miss), len(miss), sum(r["kind"] == "a" for r in miss),
        sum(r["kind"] == "b" for r in miss), sum(r["kind"] == "c" for r in miss), spots, gen,
        1000 * spots / max(gen, 1)))
    if a.json:
        with open(a.json, "w") as fh:
            json.dump({"pairs": rows, "generated": gen, "spots": spots}, fh, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    sys.exit(main())
