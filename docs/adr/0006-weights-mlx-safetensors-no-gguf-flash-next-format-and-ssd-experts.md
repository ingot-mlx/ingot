# 6. Weights: MLX safetensors only; Flash-Next in its own lossless format, with experts read from the SSD

Status: Accepted

## Context

What people download for Apple silicon are the community's MLX conversions. Qwen3.8-Flash-Next is the exception: its
size only makes sense with the 2-bit (Q2_0) experts published as a third-party GGUF, and even then the experts do not
fit in the memory of the Macs this engine is for. Large MoE models in general fit smaller Macs only if not all experts
stay in memory.

## Decision

- The engine reads only MLX safetensors, mapped from disk (`SafetensorsStore`): affine quantization at the bit widths
  in use, MXFP4, mixed bits per matrix. **The engine never reads GGUF.**
- For Flash-Next, a one-off offline conversion (`scripts/flashnext/`) rewrites the GGUF's Q2_0 experts, without loss,
  into fixed-size records in `experts.bin`. `verify_experts.py` checks the bytes and the dequantization against the
  GGUF. The PLE table goes to `ple.bin`, and the layout is described under `metal_lab` in `config.json`
  (`FlashNextFormat`). The dense part comes from the bf16 weights.
- Experts can be read from the SSD (`ExpertStream*.swift`): after each MoE layer's router the GPU signals, a CPU thread
  reads the missing experts with aligned uncached `pread`s into a slot cache (ARC for decode, a separate pass-through
  area for prefill so a long prompt does not evict decode's experts), and the GPU goes on. The math and the weights are
  the same, so it is exact. On Flash-Next this is the only path; on the Qwen3.6 MoE the memory plan
  turns it on when the model does not fit in memory (the expert cache takes what is left), or `--expert-cache-gb`
  sets it.

## Alternatives considered

- A GGUF loader in the engine: a second quantization zoo and loader to keep exact.
- Requantize the experts to our own 2-bit format: a new quality loss to validate, for no gain over the lossless
  rewrite.
- Keep all experts in memory: the largest models would not fit the target Macs.

## Consequences

- Flash-Next needs its own model pack built by the conversion scripts; a change of the format means converting again.
- The SSD path adds a CPU synchronization per MoE layer and its own options; benches that encode verification
  themselves refuse to run with it.
- The memory planner counts the expert cache instead of the experts' bytes, so the same model can be planned for
  smaller Macs.
- The CI covers the SSD path and Flash-Next only with the real model.
