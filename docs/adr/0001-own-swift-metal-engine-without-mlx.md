# 1. Own Swift + Metal engine, without MLX at runtime

Status: Accepted

## Context

The goal is to get the most out of Apple-silicon Macs with 16 to 48 GB for a short list of popular model families,
faster than the MLX-based servers people already use. The speed comes from things a general array framework does not
let us control: kernels fused across what would be several ops, decode rounds chained so the CPU encodes the next one
before waiting for the current one, decisions taken on the GPU in the middle of a round (verification rows, the expert
union of a MoE layer), and experts read from the SSD between two layers. The reference implementations (mlx-lm,
transformers) are Python.

## Decision

- The engine is a Swift package: the `MetalLab` library (loader, layers, kernels, drafter, generation, memory plan),
  `LabServe` (the server), and two executables, `lab` (use) and `lab-dev` (exactness checks and benches).
- Every op is a hand-written Metal kernel in `Sources/MetalLab/Shaders`. At runtime the only dependencies are Apple's
  frameworks and `swift-jinja` (chat templates). No MLX, no Python.
- Weights are the community's MLX safetensors, mapped from disk (ADR 6). The tokenizer and the chat template run in
  Swift.
- Only M5 or later (Metal family `Apple10`, matrix accelerators through `mpp::tensor_ops`) and macOS 27.
  `MachineRequirements` checks this when the device is created and says what is missing.
- MLX and mlx-lm stay outside the binary, as the reference: the Python scripts in `ref/` and the fidelity steps of the
  CI (ADR 2).

## Alternatives considered

- Build on MLX (or mlx-swift): its lazy graph does not fuse across ops or decide on the GPU in the middle of a round.
- Extend an existing MLX-based server: its large test suite and rebases slow down every experiment.
- Read GGUF and follow llama.cpp's kernels: other quantization formats than the MLX conversions in use (ADR 6).
- Support older chips: a second set of kernels without the matrix accelerators for a shrinking share of Macs.

## Consequences

- Every new piece of a model (an attention variant, an activation, a quantization) costs kernel work. That is why the
  set of families is closed (ADR 4).
- Each kernel needs its own test against a CPU reference (`KernelCaseTests`), and each model needs a fidelity check
  against mlx-lm, because nothing is inherited from a framework.
- The engine has to follow mlx-lm when a family changes upstream; Python is a development tool, never a runtime
  requirement.
- No portability: other GPUs, Linux or older macOS are out of scope.
