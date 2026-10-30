"""Reference with YaRN `truncate: false` for gpt-oss: mlx-lm's YarnRoPE always rounds the correction range
(math.floor/math.ceil in rope_utils.py) and does not read `truncate`, while transformers and OpenAI's gpt-oss leave it
unrounded when the config says `truncate: false` (gpt-oss-20b's does). This wraps gpt_oss.py's `initialize_rope` so
that, with `truncate: false`, the YarnRoPE it builds gets its frequencies recomputed with the unrounded range (the same
expression as mlx-lm's YarnRoPE otherwise, in float32), and then runs gen_ref.py or logits_ref.py unchanged. The
engine's table (RopeTable, checked against transformers' formula by RopeTableTests) follows the config.

Usage (same arguments as the wrapped script; chains with the other wrappers, e.g. kv8_ref.py):
  .venv/bin/python yarn_ref.py gen_ref.py --model ~/.cache/metal-lab/toy/toy-gptoss ...
  .venv/bin/python yarn_ref.py logits_ref.py --f32 --model ... --gen ...
"""
import math
import os
import runpy
import sys

import mlx.core as mx
from mlx_lm.models import gpt_oss
from mlx_lm.models.rope_utils import YarnRoPE

HERE = os.path.dirname(os.path.abspath(__file__))
_initialize_rope = gpt_oss.initialize_rope
patched = []


def unrounded_freqs(dims, base, cfg):
    """YarnRoPE._freqs with the correction range left unrounded (transformers' `truncate=False`)."""
    factor, orig = cfg["factor"], cfg.get("original_max_position_embeddings", 4096)
    beta_fast, beta_slow = cfg.get("beta_fast", 32), cfg.get("beta_slow", 1)

    def corr_dim(n):
        return (dims * math.log(orig / (n * 2 * math.pi))) / (2 * math.log(base))

    low, high = max(corr_dim(beta_fast), 0), min(corr_dim(beta_slow), dims - 1)
    if low == high:
        high += 0.001
    freq_extra = base ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims)
    freq_inter = factor * freq_extra
    mask = 1.0 - mx.clip((mx.arange(dims // 2, dtype=mx.float32) - low) / (high - low), 0, 1)
    return (freq_inter * freq_extra) / (freq_inter * mask + freq_extra * (1 - mask))


def initialize_rope(dims, base, traditional, scaling_config=None, max_position_embeddings=None):
    rope = _initialize_rope(
        dims, base, traditional, scaling_config=scaling_config, max_position_embeddings=max_position_embeddings)
    if isinstance(rope, YarnRoPE) and scaling_config and scaling_config.get("truncate", True) is False:
        rope._freqs = unrounded_freqs(dims, base, scaling_config)
        patched.append(dims)
    return rope


gpt_oss.initialize_rope = initialize_rope

if __name__ == "__main__":
    script = os.path.join(HERE, sys.argv[1])
    sys.argv = [script] + sys.argv[2:]
    sys.path.insert(0, HERE)
    try:
        runpy.run_path(script, run_name="__main__")
    finally:
        print(f"yarn_ref: YaRN with truncate=false in {len(patched)} attention layers", file=sys.stderr)
