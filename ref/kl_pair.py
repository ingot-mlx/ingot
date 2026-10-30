"""KL(A || B) between two logits files of the same shape ([rows, V], fp32; for example two _logits_f32.npy of
logits_ref.py, [n + 1, V], whose row 0 is the last prompt position): mean, median, row 0, last row, max and equal
top-1, with the same formula as lab-dev kl (natural log, terms with log p_A > -30). From a scratch script.

Usage:
  .venv/bin/python kl_pair.py out/gen-a/4K_logits_f32.npy out/gen-b/4K_logits_f32.npy "label" [--json out.jsonl]
Importable: kl_rows(a, b) -> per-row KL, summary(a, b) -> dict.
"""
import json
import sys

import numpy as np


def logp(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.float64)
    m = v.max(axis=-1, keepdims=True)
    return v - (np.log(np.exp(v - m).sum(axis=-1, keepdims=True)) + m)


def kl_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-row KL(softmax(a) || softmax(b)) of two logits arrays [rows, V]."""
    assert a.shape == b.shape, f"shapes differ: {a.shape} vs {b.shape}"
    la, lb = logp(a), logp(b)
    return np.where(la > -30, np.exp(la) * (la - lb), 0).sum(axis=-1)


def summary(a: np.ndarray, b: np.ndarray) -> dict:
    kls = kl_rows(a, b)
    top = int((a.argmax(-1) == b.argmax(-1)).sum())
    return {"n": int(len(kls)), "mean": float(kls.mean()), "median": float(np.median(kls)), "row0": float(kls[0]),
            "last": float(kls[-1]), "max": float(kls.max()), "top1": top}


def main() -> None:
    args = sys.argv[1:]
    out = None
    if "--json" in args:
        i = args.index("--json")
        out = args[i + 1]
        del args[i: i + 2]
    s = summary(np.load(args[0]), np.load(args[1]))
    label = args[2] if len(args) > 2 else ""
    print(f"{label}: KL row 0 {s['row0']:.5f}, mean over {s['n']} rows {s['mean']:.5f}, median {s['median']:.5f}, "
          f"last row {s['last']:.5f}, max {s['max']:.4f}, top-1 equal {s['top1']}/{s['n']}")
    if out:
        with open(out, "a") as f:
            f.write(json.dumps({"label": label, **s}) + "\n")


if __name__ == "__main__":
    main()
