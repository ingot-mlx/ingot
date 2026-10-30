"""Reference vectors of Nemotron-H's Mamba2 mixer: mlx-lm's own NemotronHMamba2Mixer (conv with bias, the
SSM in its two forms and the gated group norm) in fp32, written to ref/ssm_vectors.json for KernelCaseTests (case
`mamba2`, the GDN kernels with LAB_SSM against it).

The in_proj is left out: the mixer runs from its projected rows (z | x B C | dt), which the engine's kernels read from
the projection output. Inputs and weights are not stored: both sides build them from the integer hash `hval` below
(uint32 arithmetic, exact in fp32; the bf16 weights are exact in bf16), so the file only carries outputs, sampled with a
stride. Scenario, over a zero state, as the engine runs it:
  1. a prefill of 300 rows (mlx-lm's chunked SSD form; the engine's sequential recurrence);
  2. one decode row (mlx-lm's Metal ssm kernel);
  3. a block of 16 rows, one at a time (the engine's round of 16 with nacc 1), recording the state after 4 and 16;
  4. back to the state after 4 of those 16 (a rejection: the engine's round with nacc 4) and a block of 2 new rows,
     with the state after them (the engine's fold).
Shapes: the Lightning's 64 heads x 64 and the Nano 4B's 96 x 80, both with 8 groups and a state of 128.
The outputs y (before the norm) and o (after it) are mlx-lm's. The states are a float64 sequential recurrence of the same
formulas (`Seq`): mlx-lm's own fp32 state is ~1e-3 (max error over max value) from it after the 300-row prefill (its SSD
form takes exp of cumulative sums), more than the engine's sequential fp32, so the states are checked against the
float64 one; the script checks that mlx-lm's states stay within MLX_STATE_TOL of it.

Usage (from ref/, MLX needs the GPU lock):
  .venv/bin/python ssm_ref.py            writes ssm_vectors.json
"""
import json
import os

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import ArraysCache
from mlx_lm.models.nemotron_h import ModelArgs, NemotronHMamba2Mixer

HERE = os.path.dirname(os.path.abspath(__file__))
M32 = 0xFFFFFFFF
PREFILL, ROUND, KEEP, NEW = 300, 16, 4, 2
ROWS = PREFILL + 1 + ROUND + NEW
# Sampling strides of the stored outputs (every Y-th channel of a row, every S-th element of a state).
Y_STRIDE, S_STRIDE = 31, 509


def hval(n, seed):
    """n integer hashes (murmur3 finalizer of i·0x9E3779B1 + seed), uint32."""
    h = (np.arange(n, dtype=np.uint64) * 0x9E3779B1 + seed) & M32
    h ^= h >> 15
    h = (h * 0x85EBCA77) & M32
    h ^= h >> 13
    h = (h * 0xC2B2AE3D) & M32
    h ^= h >> 16
    return h


def unit(n, seed):
    """[-1, 1) with 24 bits: exact in fp32."""
    return ((hval(n, seed) >> 8).astype(np.float64) - 8388608.0) / 8388608.0


def byte(n, seed):
    """Integers 0…255 (the top byte of the hash)."""
    return (hval(n, seed) >> 24).astype(np.float64)


def inputs(H, P, G, N):
    """Projected rows and weights, as KernelCase.mamba2Inputs builds them."""
    d_inner, conv = H * P, H * P + 2 * G * N
    W = d_inner + conv + H
    return {
        "proj": unit(ROWS * W, 11).reshape(ROWS, W),
        # bf16-exact weights: k/256, k/1024, −k/64 (k in 128…255), (k + 64)/128 and fp32 A_log = k/64, D = 1 + k/512.
        "conv_w": (byte(conv * 4, 12) - 128) / 256,
        "conv_b": (byte(conv, 13) - 128) / 1024,
        "dt_bias": -(128 + byte(H, 14) // 2) / 64,
        "A_log": (byte(H, 15) % 181) / 64,
        "D": 1 + (byte(H, 16) - 128) / 512,
        "norm_w": (byte(d_inner, 17) // 2 + 64) / 128,
    }


def mixer(H, P, G, N, w):
    args = ModelArgs(
        model_type="nemotron_h", vocab_size=64, hidden_size=64, intermediate_size=64, max_position_embeddings=4096,
        num_attention_heads=1, num_key_value_heads=1, attention_bias=False, mamba_num_heads=H, mamba_head_dim=P,
        mamba_proj_bias=False, ssm_state_size=N, conv_kernel=4, n_groups=G, mlp_bias=False, layer_norm_epsilon=1e-5,
        use_bias=False, use_conv_bias=True, hybrid_override_pattern=["M"],
    )
    m = NemotronHMamba2Mixer(args)
    conv = H * P + 2 * G * N
    f32 = lambda a: mx.array(np.asarray(a, dtype=np.float32))
    m.conv1d.weight = f32(w["conv_w"]).reshape(conv, 4, 1)
    m.conv1d.bias = f32(w["conv_b"])
    m.dt_bias = f32(w["dt_bias"])
    m.A_log = f32(w["A_log"])
    m.D = f32(w["D"])
    m.norm.weight = f32(w["norm_w"])
    return m


def run(m, cache, rows):
    """The mixer without in_proj (NemotronHMamba2Mixer.__call__ from `projected`): pre-norm y and normed o."""
    d_inner, conv = m.intermediate_size, m.conv_dim
    gate, conv_in, dt = mx.split(rows[None], [d_inner, d_inner + conv], axis=-1)
    co = m._conv(conv_in, cache, None)
    x, B, C = mx.split(co, [d_inner, d_inner + m.n_groups * m.ssm_state_size], axis=-1)
    y = m._ssm(x, B, C, dt, cache, None)
    o = m.norm(y, gate)
    mx.eval(y, o, cache[0], cache[1])
    return np.array(y[0]), np.array(o[0])


MLX_STATE_TOL = 3e-3


class Seq:
    """Float64 sequential Mamba2 (mlx-lm's formulas: conv with bias + SiLU, dt = softplus(dt + dt_bias),
    S = exp(−exp(A_log)·dt)·S + dt·x⊗B, y = S·C + D·x) over the projected rows."""

    def __init__(self, H, P, G, N, w):
        self.H, self.P, self.G, self.N, self.w = H, P, G, N, w
        self.d_inner, self.conv = H * P, H * P + 2 * G * N
        self.S = np.zeros((H, P, N))
        self.win = np.zeros((3, self.conv))

    def rows(self, proj):
        di, cv, H, P, G, N, w = self.d_inner, self.conv, self.H, self.P, self.G, self.N, self.w
        cw = w["conv_w"].reshape(cv, 4)
        for r in proj:
            xin = np.concatenate([self.win, r[None, di:di + cv]])
            self.win = xin[1:]
            c = (xin * cw.T).sum(0) + w["conv_b"]
            c = c / (1 + np.exp(-c))
            x, B, C = c[:di].reshape(H, P), c[di:di + G * N].reshape(G, N), c[di + G * N:].reshape(G, N)
            dt = np.logaddexp(r[di + cv:] + w["dt_bias"], 0)
            g = np.exp(-np.exp(w["A_log"]) * dt)
            Bh, _Ch = np.repeat(B, H // G, axis=0), np.repeat(C, H // G, axis=0)
            self.S = g[:, None, None] * self.S + (dt[:, None] * x)[:, :, None] * Bh[:, None, :]

    def copy(self):
        o = Seq(self.H, self.P, self.G, self.N, self.w)
        o.S, o.win = self.S.copy(), self.win.copy()
        return o


def rel(a, b):
    return float(np.abs(np.asarray(a, np.float64) - b).max() / np.abs(b).max())


def sample(a, stride):
    """Every `stride`-th value, with 8 significant digits (enough for fp32 and half the file)."""
    return [float(f"{v:.8g}") for v in np.asarray(a, dtype=np.float32).reshape(-1)[::stride]]


def case(H, P, G, N):
    w = inputs(H, P, G, N)
    m = mixer(H, P, G, N, w)
    proj = mx.array(w["proj"].astype(np.float32))
    cache = ArraysCache(size=2)
    seq, pr = Seq(H, P, G, N, w), w["proj"]
    errs = []

    def state(key):
        errs.append((key, rel(np.array(cache[1][0]), seq.S)))
        out[key] = sample(seq.S, S_STRIDE)

    out = {"name": f"mamba2-{H}x{P}", "H": H, "P": P, "G": G, "N": N, "rows": ROWS,
           "y_stride": Y_STRIDE, "s_stride": S_STRIDE}
    y, o = run(m, cache, proj[:PREFILL])
    pre_rows = [0, 1, 63, 64, PREFILL - 1]
    out["prefill_rows"] = pre_rows
    out["prefill_y"] = [sample(y[r], Y_STRIDE) for r in pre_rows]
    out["prefill_o"] = [sample(o[r], Y_STRIDE) for r in pre_rows]
    seq.rows(pr[:PREFILL])
    state("prefill_state")
    y, o = run(m, cache, proj[PREFILL:PREFILL + 1])
    seq.rows(pr[PREFILL:PREFILL + 1])
    out["step_y"], out["step_o"] = sample(y[0], Y_STRIDE), sample(o[0], Y_STRIDE)
    b0 = PREFILL + 1
    ys, os_, saved = [], [], None
    for r in range(ROUND):
        y, o = run(m, cache, proj[b0 + r:b0 + r + 1])
        seq.rows(pr[b0 + r:b0 + r + 1])
        ys.append(y[0]); os_.append(o[0])
        if r + 1 == KEEP:
            state("state_keep")
            saved = (cache[0], cache[1], seq.copy())
    state("state_round")
    round_rows = [0, 3, 4, ROUND - 1]
    out["round_rows"] = round_rows
    out["round_y"] = [sample(ys[r], Y_STRIDE) for r in round_rows]
    out["round_o"] = [sample(os_[r], Y_STRIDE) for r in round_rows]
    cache[0], cache[1], seq = saved
    y, o = run(m, cache, proj[b0 + ROUND:b0 + ROUND + NEW])
    seq.rows(pr[b0 + ROUND:b0 + ROUND + NEW])
    out["new_y"] = [sample(y[r], Y_STRIDE) for r in range(NEW)]
    out["new_o"] = [sample(o[r], Y_STRIDE) for r in range(NEW)]
    state("state_new")
    print(out["name"], "mlx-lm fp32 state against float64:", ", ".join(f"{k} {e:.2e}" for k, e in errs))
    assert all(e < MLX_STATE_TOL for _, e in errs), "mlx-lm and the float64 recurrence disagree"
    return out


def main():
    mx.set_default_device(mx.gpu)
    cases = [case(64, 64, 8, 128), case(96, 80, 8, 128)]
    path = os.path.join(HERE, "ssm_vectors.json")
    with open(path, "w") as f:
        json.dump({"source": "mlx-lm nemotron_h NemotronHMamba2Mixer (fp32; states: float64 sequential), ref/ssm_ref.py",
                   "cases": cases}, f,
                  separators=(",", ":"))
    print(f"{path}: {len(cases)} cases, {os.path.getsize(path) // 1024} KB")


if __name__ == "__main__":
    main()
