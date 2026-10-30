"""Mean KL between two logits files from logits_ref.py (by default, mlx-lm fp32 ‖ mlx-lm bf16 of one prompt): the
own noise of mlx-lm in bf16, which the fidelity of the toys (ci.sh --toy) used as the fidelity ceiling when the routing of the toy MoE
is unstable (see the fidelity function in scripts/ci/fidelity.sh).

Usage:
  ref/.venv/bin/python ref/kl_npy.py --gen ci-fid-moe --prompt 512
"""
import argparse
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def logp(v):
    v = v.astype(np.float64)
    m = v.max(axis=-1, keepdims=True)
    return v - (np.log(np.exp(v - m).sum(axis=-1, keepdims=True)) + m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True)
    ap.add_argument("--prompt", default="512")
    a = ap.parse_args()
    base = os.path.join(HERE, "out", a.gen, a.prompt)
    p, q = logp(np.load(base + "_logits_f32.npy")), logp(np.load(base + "_logits.npy"))
    kls = np.where(p > -30, np.exp(p) * (p - q), 0).sum(axis=-1)
    print(f"{a.prompt}: KL(mlx-lm fp32 ‖ mlx-lm bf16) mean over {len(kls)} positions {kls.mean():.5f}, "
          f"median {np.median(kls):.5f}, max {kls.max():.4f}")


if __name__ == "__main__":
    main()
