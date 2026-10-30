"""Reference for the lfm3 family (LFM3-1.7B, toy-lfm3): mlx-lm 0.32.2 has no lfm3, so this registers
ref/lfm3_mlx.py (lfm3.py of mlx-lm PR #2016 at commit 9027ac5, copied verbatim) as the module mlx_lm.models.lfm3, which
mlx_lm.utils._get_classes imports by model_type, and then runs the wrapped script unchanged: gen_ref.py,
logits_ref.py or another wrapper (kv8_ref.py), which can be chained. If the installed mlx-lm already ships an lfm3
module, the PR's copy still replaces it (the reference stays pinned to 9027ac5; comparing both is left for later).

Usage (same arguments as the wrapped script):
  .venv/bin/python lfm3_ref.py gen_ref.py --model ~/.cache/ingot/toy/toy-lfm3 --pdir prompts-toy ...
  .venv/bin/python lfm3_ref.py logits_ref.py --f32 --model ... --gen ...
  .venv/bin/python lfm3_ref.py kv8_ref.py logits_ref.py --f32 --model ...
"""
import importlib.util
import os
import runpy
import sys

import mlx_lm.models

HERE = os.path.dirname(os.path.abspath(__file__))


def install():
    """Registers ref/lfm3_mlx.py as mlx_lm.models.lfm3 (idempotent) and returns the module."""
    name = "mlx_lm.models.lfm3"
    mod = sys.modules.get(name)
    if mod is not None and getattr(mod, "__file__", None) == os.path.join(HERE, "lfm3_mlx.py"):
        return mod
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, "lfm3_mlx.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    mlx_lm.models.lfm3 = mod
    return mod


install()

if __name__ == "__main__":
    script = os.path.join(HERE, sys.argv[1])
    sys.argv = [script] + sys.argv[2:]
    sys.path.insert(0, HERE)
    runpy.run_path(script, run_name="__main__")
