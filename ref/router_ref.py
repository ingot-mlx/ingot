"""Reference vectors of the MoE router: expert indices and weights computed by mlx-lm's own router code on
random logits, written to ref/router_vectors.json for KernelCaseTests (router-mlx cases). One case per router of a
tier-1 family (LFM2.5, GLM-4.7-Flash, gpt-oss, Gemma 4 26B-A4B, Nemotron-H).

Each family's real module runs unchanged: its router linear is set to the identity (so x = logits) and, where the
module also runs the experts, they are replaced by a one-hot of the chosen indices, so the module's output is the dense
weight vector over experts (zero where not chosen). That captures selection, renormalization (with each family's own
epsilon) and routed_scaling_factor exactly as mlx-lm applies them. Rows 0-1 of each case carry exact ties at the k-th
place (the engine keeps the lower index; mlx-lm's argpartition does not promise any order, so the test only compares
rows without ties against mlx-lm and checks ties against the engine's rule).

Usage (from ref/, MLX needs the GPU lock):
  .venv/bin/python router_ref.py            writes router_vectors.json
"""
import json
import os
import types

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models import gemma4_text, glm4_moe_lite, gpt_oss, lfm2_moe, nemotron_h

HERE = os.path.dirname(os.path.abspath(__file__))
B = 8


def onehot_experts(E):
    def f(x, inds):
        return (mx.arange(E) == inds[..., None]).astype(mx.float32)

    return f


def identity_linear(E, bias=False):
    lin = nn.Linear(E, E, bias=bias)
    lin.weight = mx.eye(E, dtype=mx.float32)
    if bias:
        lin.bias = mx.zeros((E,), dtype=mx.float32)
    return lin


def dense(inds, w, E):
    """Dense [B, E] weights from (inds, weights) [B, k]."""
    out = np.zeros((inds.shape[0], E), dtype=np.float32)
    for r in range(inds.shape[0]):
        for i, v in zip(inds[r], w[r]):
            out[r, int(i)] += v
    return out


def logits_with_ties(rng, E, K):
    lg = rng.uniform(-4, 4, size=(B, E)).astype(np.float32)
    # Rows 0 and 1: K - 1 clear winners and an exact tie for the K-th place (lower index first in row 0, reversed
    # roles in row 1 so the bias, if any, cannot break it).
    for r, (a, b) in enumerate([(E // 4 + 1, 3 * E // 4 - 1), (E // 2 + 3, E // 2 - 5)]):
        lg[r, :] = -3
        for e in range(1, K):
            lg[r, (7 * e) % E] = 5
        lg[r, a] = lg[r, b] = 2
    return lg


def case(name, family, E, K, score, logits, bias, norm, scale, eps, w, extra=None):
    c = {
        "name": name, "family": family, "experts": E, "topK": K, "score": score, "norm": norm, "scale": scale,
        "eps": eps, "rows": B, "logits": [float(v) for v in logits.reshape(-1)],
        "bias": None if bias is None else [float(v) for v in bias], "weights": [float(v) for v in w.reshape(-1)],
    }
    if extra:
        c.update(extra)
    return c


def run_lfm2(rng, E, K):
    args = types.SimpleNamespace(
        hidden_size=E, moe_intermediate_size=8, num_experts=E, num_experts_per_tok=K, norm_topk_prob=True,
        use_expert_bias=True, routed_scaling_factor=1.0)
    m = lfm2_moe.Lfm2MoeSparseMoeBlock(args)
    m.gate = identity_linear(E)
    m.switch_mlp = onehot_experts(E)
    bias = rng.uniform(-0.3, 0.3, size=E).astype(np.float32)
    m.expert_bias = mx.array(bias)
    lg = logits_with_ties(rng, E, K)
    for r in range(2):  # no bias on the tied pair or the winners
        bias[lg[r] > 0] = 0
    m.expert_bias = mx.array(bias)
    y = np.array(m(mx.array(lg)[None]).astype(mx.float32))[0]
    return case(f"lfm2-e{E}", "lfm2_moe", E, K, "sigmoid", lg, bias, True, 1.0, 1e-6, y)


def run_group_select(mod, rng, name, E, K, with_bias, norm, scale, eps):
    lg = logits_with_ties(rng, E, K)
    bias = rng.uniform(-0.3, 0.3, size=E).astype(np.float32) if with_bias else np.zeros(E, np.float32)
    for r in range(2):
        bias[lg[r] > 0] = 0
    inds, w = mod.group_expert_select(mx.array(lg), mx.array(bias), K, 1, 1, scale, norm)
    y = dense(np.array(inds), np.array(w.astype(mx.float32)), E)
    return case(name, mod.__name__.split(".")[-1], E, K, "sigmoid", lg, bias if with_bias else None, norm, scale,
                eps, y)


def run_gpt_oss(rng, E, K, with_bias=False):
    """gpt-oss's MLPBlock; `with_bias`: its router linear bias (`mlp.router.bias`) non-zero, added to the logits before
    the top-k and the softmax (written as `logitBias`; zero on the tied pair and the winners of rows 0-1)."""
    cfg = types.SimpleNamespace(
        hidden_size=E, num_local_experts=E, num_experts_per_tok=K, intermediate_size=8, swiglu_limit=7.0)
    m = gpt_oss.MLPBlock(cfg)
    m.router = identity_linear(E, bias=True)
    m.experts = onehot_experts(E)
    lg = logits_with_ties(rng, E, K)
    extra = None
    if with_bias:
        bias = rng.uniform(-1, 1, size=E).astype(np.float32)
        for r in range(2):
            bias[lg[r] > 0] = 0
        m.router.bias = mx.array(bias)
        extra = {"logitBias": [float(v) for v in bias]}
    y = np.array(m(mx.array(lg)[None]).astype(mx.float32))[0]
    return case(f"gpt-oss-e{E}" + ("-bias" if with_bias else ""), "gpt_oss", E, K, "softmax", lg, None, True, 1.0,
                0.0, y, extra)


class FixedLogits(nn.Module):
    """Stands in for a router linear that comes after the router's own input norm (Gemma 4): returns the given logits
    whatever its input, so the rest of the module (top-k, softmax, per_expert_scale) runs unchanged on them."""

    def __init__(self, logits):
        super().__init__()
        self.lg = mx.array(logits)

    def __call__(self, x):
        return self.lg


def run_gemma4(rng, E, K):
    """Gemma 4's Router (26B-A4B: 128 experts, top-8): softmax over the top-k logits, then x per_expert_scale of each
    chosen expert (written as `perExpertScale`, here in [0.5, 1.5); the real one stays within 1 +- 0.03). Its input norm
    (router.scale x hidden^-0.5) runs before the logits, in the model; here the linear returns the logits directly."""
    cfg = types.SimpleNamespace(hidden_size=E, num_experts=E, top_k_experts=K, rms_norm_eps=1e-6)
    m = gemma4_text.Router(cfg)
    lg = logits_with_ties(rng, E, K)
    pes = rng.uniform(0.5, 1.5, size=E).astype(np.float32)
    m.proj = FixedLogits(lg)
    m.per_expert_scale = mx.array(pes)
    inds, w = m(mx.zeros((B, E)))
    y = dense(np.array(inds), np.array(w.astype(mx.float32)), E)
    return case(f"gemma4-e{E}-k{K}", "gemma4_text", E, K, "softmax", lg, None, True, 1.0, 0.0, y,
                {"perExpertScale": [float(v) for v in pes]})


def skip_draws(rng, E, K, with_bias):
    """Consumes the random numbers of a removed case (logits, and the bias if it had one), so the cases after it keep
    their vectors."""
    logits_with_ties(rng, E, K)
    if with_bias:
        rng.uniform(-0.3, 0.3, size=E)


def main():
    mx.set_default_device(mx.cpu)
    rng = np.random.default_rng(20261024)
    cases = [
        # LFM2.5 (lfm2_moe): sigmoid + bias + renormalized with epsilon 1e-6.
        run_lfm2(rng, 64, 4),
        run_lfm2(rng, 32, 4),
        # GLM-4.7-Flash (glm4_moe_lite): sigmoid + noaux_tc bias + renormalized (epsilon 1e-20) x 1.8, its
        # routed_scaling_factor.
        run_group_select(glm4_moe_lite, rng, "glm4lite-bias-e64-x1.8", 64, 4, True, True, 1.8, 1e-20),
    ]
    # toy-moe-sigmoid's router (sigmoid without bias, top-6 of 64) left with the toy (Nemotron-H's router,
    # below, has the sigmoid with bias and the scale), and the two routers without renormalization (norm_topk_prob =
    # false, sigmoid with bias and softmax) left with the piece: their draws stay so the later vectors do not
    # change.
    skip_draws(rng, 64, 6, False)
    skip_draws(rng, 128, 8, True)
    skip_draws(rng, 128, 8, False)
    cases += [
        # gpt-oss: softmax over the top-k with 32 experts.
        run_gpt_oss(rng, 32, 4),
        # gpt-oss with its router linear bias: it counts for the choice and for the weights.
        run_gpt_oss(rng, 32, 4, with_bias=True),
        # Gemma 4 26B-A4B: softmax top-8 of 128 with per_expert_scale (LAB_MOE_PES).
        run_gemma4(rng, 128, 8),
        # Nemotron 3.5 Lightning (nemotron_h): sigmoid + e_score_correction_bias + renormalized (its 1e-20 is
        # below the float32 resolution of the sum: eps 0) x 2.5, top-6 of 128.
        run_group_select(nemotron_h, rng, "nemotron-e128-k6", 128, 6, True, True, 2.5, 0.0),
    ]
    out = os.path.join(HERE, "router_vectors.json")
    with open(out, "w") as f:
        json.dump({"source": "ref/router_ref.py (mlx-lm " + __import__("mlx_lm").__version__ + ")", "cases": cases}, f)
    for c in cases:
        w = np.array(c["weights"]).reshape(B, c["experts"])
        print(f"{c['name']:32s} E={c['experts']:3d} k={c['topK']:2d} chosen/row={sorted(set((w != 0).sum(1)))} "
              f"sum/row={w.sum(1)[2]:.6f}")
    print("wrote", out)


if __name__ == "__main__":
    main()
