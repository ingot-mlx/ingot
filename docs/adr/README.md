# Architecture decision records

The decisions that shape the rest of the engine. Each record says what was decided, what else was considered and what
it costs. A record is not edited when the decision changes: a new one supersedes it.

| ADR | Decision |
|---|---|
| [0001](0001-own-swift-metal-engine-without-mlx.md) | Own engine in Swift with hand-written Metal kernels, no MLX or Python at runtime, M5 and macOS 27 only. |
| [0002](0002-exactness-bit-exact-paths-and-fidelity-against-mlx-lm.md) | Bit-exact between paths within a version (goldens), mean KL < 0.01 against mlx-lm fp32 against the outside. |
| [0003](0003-exact-by-default-lossy-options-and-experiments.md) | Exact by default; lossy options only where measured, announced and switchable; experiments behind `--exp`, never in main. |
| [0004](0004-family-tiers-and-two-mode-ci-on-toys-from-real-configs.md) | Tier-1 families get a toy and a real model in the CI; a `--toy` mode on every integration and `--real` only when needed. |
| [0005](0005-exact-speculative-decoding-with-dflash2-and-gpu-chosen-rows.md) | Exact DFlash2 speculation, with verification rows chosen on the GPU from the drafter only and row-invariant kernels. |
| [0006](0006-weights-mlx-safetensors-no-gguf-flash-next-format-and-ssd-experts.md) | MLX safetensors only, never GGUF; Flash-Next in its own lossless format; experts can be read from the SSD. |
| [0007](0007-lab-serve-openai-compatible-with-exact-prefix-cache.md) | `lab serve` speaks the OpenAI chat API and resumes from an exact prefix cache. |
