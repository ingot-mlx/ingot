#!/usr/bin/env python3
"""Reference vectors of the gated FFN activations of mlx-lm in fp32, for KernelCaseActivations.

    ref/.venv/bin/python ref/act_ref.py      prints x and nn.gelu_approx(x) (mlx-lm Gemma 2/3/4 GeGLU) as Swift arrays,
                                             then gpt-oss's swiglu_oai(gate, up) on a grid and Nemotron-H's
                                             nn.relu2(x)

The values are copied into Tests/MetalLabTests/KernelCaseActivations.swift (GeluReferenceTests, Relu2ReferenceTests) and
KernelCaseGptOss.swift (SwigluOAIReferenceTests);
the test checks that the
CPU formula of the kernel cases matches them, so the kernels are compared against what mlx-lm computes.
"""
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.activations import swiglu_oai

xs = [-12.0, -6.5, -4.0, -3.0, -2.25, -1.5, -1.0, -0.75, -0.5, -0.25, -0.0625, 0.0, 0.03125, 0.125, 0.4, 0.7, 1.0,
      1.3, 2.0, 2.7, 3.5, 5.0, 8.0, 15.0]
x = mx.array(xs, dtype=mx.float32)
y = nn.gelu_approx(x)
mx.eval(y)
print("x:", "[" + ", ".join(f"{v:.9g}" for v in x.tolist()) + "]")
print("y:", "[" + ", ".join(f"{v:.9g}" for v in y.tolist()) + "]")

# swiglu_oai (mlx-lm activations.py, gpt_oss.py's experts): gate × up on a grid that crosses the clip at 7 (gate from
# above only, up at ±7) and the sigmoid's tails.
gs = [-9.0, -7.5, -3.0, -1.0, -0.2, 0.0, 0.3, 1.5, 4.0, 6.9, 7.0, 7.4, 12.0]
us = [-11.0, -7.0, -2.5, -1.0, 0.0, 0.6, 3.0, 6.99, 8.0]
g = mx.array([a for a in gs for _ in us], dtype=mx.float32)
u = mx.array([b for _ in gs for b in us], dtype=mx.float32)
y = swiglu_oai(g, u)
mx.eval(y)
print("g:", "[" + ", ".join(f"{v:.9g}" for v in gs) + "]")
print("u:", "[" + ", ".join(f"{v:.9g}" for v in us) + "]")
print("oai:", "[" + ", ".join(f"{v:.9g}" for v in y.tolist()) + "]")

# relu2 (Nemotron-H's MLP, experts and shared expert: mlx-lm nemotron_h.py, nn.relu2 = square(relu(x))).
rs = [-12.0, -1.0, -0.0625, 0.0, 0.03125, 0.4, 1.0, 2.7, 15.0, 255.0]
r = mx.array(rs, dtype=mx.float32)
y = nn.relu2(r)
mx.eval(y)
print("relu2 x:", "[" + ", ".join(f"{v:.9g}" for v in r.tolist()) + "]")
print("relu2 y:", "[" + ", ".join(f"{v:.9g}" for v in y.tolist()) + "]")
