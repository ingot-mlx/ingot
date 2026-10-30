# ref/golden

Reference outputs that `lab-dev gen --golden <file>` compares against (generated tokens and, with `--golden-logits`,
logit fingerprints). One JSON file per model, named after the model directory.

## Toy models (`toy-*.json`)

Not tied to local checkpoints. `lab-dev toymodels` generates the toy weights deterministically (under
`~/.cache/metal-lab/toy`) and writes their prompts to `ref/out/prompts-toy`; `scripts/ci.sh --toy` runs it when needed.
They store the 128 tokens of `--prompts 512,4K --n 128` and their logit fingerprints; with another `--n` they do not
apply. The fingerprints are bit-exact, so another GPU or OS version may still differ. The toys with a drafter
(`drafter/`: DFlash2, or toy-gemma4's DFlash v1) store one fingerprint per prompt and mode, normal and speculative;
their tokens are the same in both.

Rewrite them only when tokens change on purpose: `TOY_WRITE_GOLDEN=1 scripts/ci.sh --toy`.

## Real models (everything else)

Machine-local. They were written on the maintainer's machine from specific local checkpoints
(`Qwen3.6-35B-A3B-Heretic-MLX-4bit-calib`, `Qwen3.5-0.8B-4bit` (`mlx-community`, revision `da28692b5f13`, prompts
`ref/out/prompts-q35-08b` with the default template, without reasoning), `Qwen3.8-Flash-Next-lab-Q2`,
`LFM2.5-8B-A1B-MLX-4bit` (from `LiquidAI`), `granite-4.1-3b-4bit` (from `mlx-community`; Granite 4.1 with the `tokenizer.json` and `tokenizer_config.json` of `ibm-granite/granite-4.1-3b`,
because mlx-community's pre-tokenizer is GPT-2's), and
`gemma-4-e2b-it-4bit` (`mlx-community`, revision `238767527555`, prompts `ref/out/prompts-gemma4` with the default
template, without reasoning; the CI's Gemma 4), the bench golden `gemma-4-26b-a4b-it-4bit`
(`mlx-community`, revision `0d77464eeb23`, prompts `ref/out/prompts-gemma4-26b` with the default template, without
reasoning; written without a drafter (the greedy does not depend on it; z-lab's DFlash v1 is its default drafter);
the weights are kept outside the repo), and `gpt-oss-20b-MXFP4-Q8` (`mlx-community`, revision `773a7da77e56`, prompts
`ref/out/prompts-gptoss` from `ref/prompts.py --fixed-now` (its template prints the date) with the default template,
`Reasoning: medium`; mlx-lm reference through `ref/yarn_ref.py`; the weights are kept outside the repo), and
`NVIDIA-Nemotron-3-Nano-4B-4bit` (`mlx-community`, revision `c4d79ba1901d`, prompts `ref/out/prompts-nemotron` with
the default template, reasoning on (the prompt ends in `<think>\n`); Nemotron-H's CI model, kept locally), `GLM-4.7-Flash-4bit-h8` (our conversion: `mlx-community/GLM-4.7-Flash-4bit` revision `1454cffb1a21` with `lm_head` re-quantized at 8 bits (affine, g64)
from `zai-org/GLM-4.7-Flash` bf16 revision `7dd20894a642` by a short script (load the bf16 model, quantize only `lm_head` with `mlx.core.quantize` at 8 bits, group size 64, and write it over the 4-bit checkpoint's `lm_head` tensors), because the 4-bit
head glued stray digits to words; prompts `ref/out/prompts-glm` with the default template, reasoning on (the prompt ends in
`<|assistant|><think>`); the CI's GLM (`glm4_moe_lite`, MLA over the latent int8 KV); the weights are kept outside the repo), and `Muse-Glimmer-30B-4bit` (`mlx-community`, revision `3e7677d7a40d`, prompts `ref/out/prompts-muse` from
`ref/prompts.py --fixed-now` (its template prints the date) with the default template, `Reasoning strength: high`;
written without a drafter (the greedy does not depend on it); the weights, with the incoai DFlash2 in `drafter/`, are kept outside the repo), and `LFM3-1.7B-MLX-8bit` (`LiquidAI`, revision `ee7667f3cfcc`, 8 bits g64, prompts
`ref/out/prompts-lfm3` with the default template, reasoning on (the prompt ends in `<|im_start|>assistant\n<think>`);
mlx-lm reference through `ref/lfm3_ref.py` (mlx-lm PR #2016's `lfm3.py`); the CI's LFM3; the weights are kept outside the repo))
and from tokenized prompts in `ref/out/prompts*` (written by `ref/prompts.py`, not tracked in
git). Elsewhere they only match with the same weights, the same prompt files and the same hardware; otherwise write
your own with `lab-dev gen ... --golden ref/golden/<model>.json --write-golden`.

`scripts/ci.sh --real TAG,...` reads the model paths from the variables of the `REAL_MODELS` table in
`scripts/ci/real.sh` (one real model per tier-1 family: `CI_MODEL_MOE`, `CI_MODEL_DENSE`, `CI_MODEL_FLASH`,
`CI_MODEL_LFM`, `CI_MODEL_GRANITE`, `CI_MODEL_GEMMA4`, `CI_MODEL_GPTOSS`, `CI_MODEL_NEMOTRON`, `CI_MODEL_GLM`,
`CI_MODEL_MUSE` and `CI_MODEL_LFM3`) and looks up
`ref/golden/<basename of the model path>.json`. The other real goldens here are bench goldens, not loaded by the CI.
Gemma 3 (tier 2) has no golden here any more: its 270M left the CI and its references left `ref/out`.
