"""RoPE frequency tables from mlx-lm for RopeTableTests: `inv_freq` per rotated pair (what mx.fast.rope
uses: the reciprocal of `freqs`) and `mscale`, for the scaled RoPEs the engine runs: YaRN with
the rounded correction range (yarn-truncate), gpt-oss (YaRN with truncate=false, from its config.json), a linear one and the default
one.

mlx-lm has no `truncate`: its YarnRoPE always rounds the correction range. For gpt-oss (truncate=false) the table is a
float32 numpy port of transformers' `_compute_yarn_parameters` (torch is not installed), marked `source: transformers`.

Usage: .venv/bin/python rope_ref.py [out.json]  (default: rope/rope_ref.json next to this script)
"""
import json
import math
import os
import sys

import mlx.core as mx
import numpy as np
from mlx_lm.models.rope_utils import YarnRoPE, initialize_rope

HERE = os.path.dirname(os.path.abspath(__file__))

CASES = [
    # name, theta, rotated dims, rope parameters (as in config.json)
    # YaRN with the rounded correction range (truncate, mlx-lm's only one) over θ 1e6 and head 256 (the global layer
    # of the former toy-gemma3-yarn, since removed; ModelSpecRopeTests.yarnTruncate).
    ("yarn-truncate", 1000000.0, 256, {"rope_type": "yarn", "factor": 16.0, "original_max_position_embeddings": 8192,
                                       "beta_fast": 32.0, "beta_slow": 1.0, "attention_factor": 1.2772588722239782}),
    ("gpt-oss", 150000.0, 64, {"rope_type": "yarn", "factor": 32.0, "original_max_position_embeddings": 4096,
                               "beta_fast": 32.0, "beta_slow": 1.0, "truncate": False}),
    ("yarn-mscale", 10000.0, 64, {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 2048,
                                  "mscale": 0.707}),
    ("linear", 10000.0, 128, {"rope_type": "linear", "factor": 4.0}),
    ("default", 10000000.0, 64, {"rope_type": "default"}),
]


def f32(a):
    return [float(x) for x in np.asarray(a, dtype=np.float32)]


def yarn_transformers(theta, dims, p, dt=np.float32):
    """transformers _compute_yarn_parameters in float32 (torch.arange(..., dtype=int64).float() / dim, etc.), or in
    float64 with dt (the same formula as mlx-lm's YarnRoPE: mask / extra + (1 - mask) / inter)."""
    factor, orig = p["factor"], p["original_max_position_embeddings"]
    beta_fast, beta_slow = p.get("beta_fast", 32), p.get("beta_slow", 1)
    truncate = p.get("truncate", True)

    def corr_dim(n):
        return (dims * math.log(orig / (n * 2 * math.pi))) / (2 * math.log(theta))

    low, high = corr_dim(beta_fast), corr_dim(beta_slow)
    if truncate:
        low, high = math.floor(low), math.ceil(high)
    low, high = max(low, 0), min(high, dims - 1)
    if low == high:
        high += 0.001
    pos_freqs = dt(theta) ** (np.arange(0, dims, 2, dtype=np.int64).astype(dt) / dt(dims))
    inv_extra = dt(1.0) / pos_freqs
    inv_inter = dt(1.0) / (dt(factor) * pos_freqs)
    ramp = np.clip((np.arange(dims // 2, dtype=dt) - dt(low)) / dt(high - low), 0, 1)
    extra = dt(1) - ramp
    inv = inv_inter * (1 - extra) + inv_extra * extra
    mscale = p.get("attention_factor") or (0.1 * p.get("mscale", 1) * math.log(factor) + 1.0 if factor > 1 else 1.0)
    return inv, mscale


def table_f64(theta, dims, p):
    """The same formulas in float64, rounded to float32 once: separates mlx-lm's float32 rounding (pow) from a
    formula error in RopeTable."""
    ty = p.get("rope_type", "default")
    if ty == "yarn":
        return f32(yarn_transformers(theta, dims, p, np.float64)[0])
    period = np.float64(theta) ** (np.arange(0, dims, 2, dtype=np.float64) / dims)
    scale = 1 / p["factor"] if ty == "linear" else 1.0
    return f32(scale / period)


def table(theta, dims, p):
    ty = p.get("rope_type", "default")
    if ty == "yarn" and p.get("truncate", True) is False:
        inv, ms = yarn_transformers(theta, dims, p)
        return f32(inv), ms, "transformers"
    if ty == "yarn":
        kw = {k: p[k] for k in ["original_max_position_embeddings", "beta_fast", "beta_slow", "mscale"] if k in p}
        r = YarnRoPE(dims, base=theta, scaling_factor=p["factor"], **kw)
        # transformers' attention_factor overrides the computed mscale (mlx-lm ignores the key; equal in these configs).
        ms = p.get("attention_factor", r.mscale)
        return f32(mx.reciprocal(r._freqs)), ms, "mlx-lm"
    # default / linear: nn.RoPE(dims, base, scale) rotates by (pos · scale) · base^(-2i/dims).
    r = initialize_rope(dims, theta, False, p if ty != "default" else None)
    inv = mx.array(r.scale, dtype=mx.float32) / (mx.array(theta, dtype=mx.float32) ** (
        mx.arange(0, dims, 2, dtype=mx.float32) / dims))
    return f32(inv), 1.0, "mlx-lm"


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "rope", "rope_ref.json")
    rows = []
    for name, theta, dims, p in CASES:
        inv, ms, src = table(theta, dims, p)
        rows.append({"name": name, "theta": theta, "dims": dims, "params": p, "inv_freq": inv,
                     "inv_freq_f64": table_f64(theta, dims, p),
                     "mscale": float(np.float32(ms)), "source": src})
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(rows, f, indent=1)
        f.write("\n")
    print(f"{len(rows)} cases -> {out}")


if __name__ == "__main__":
    main()
