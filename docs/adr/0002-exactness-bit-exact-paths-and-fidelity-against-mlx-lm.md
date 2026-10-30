# 2. Exactness: bit-exact paths and fidelity against mlx-lm fp32

Status: Accepted

## Context

Speed work changes kernels every day. A check with a tolerance lets small errors pile up and hides the bugs that
matter most here: two paths that should compute the same state and do not (with and without the prefix cache, a
prompt split in chunks or read whole, a round of several rows or several rounds of one). The external reference can
also be wrong, since it is another reimplementation of the model.

## Decision

Two levels, both enforced by the CI (ADR 4):

- **Within a version, bit-exact.** Every path that should compute the same thing gives the same bits: cached or cold,
  prefill split at 64-aligned points or whole, a round of k rows or k rounds of one (`rowscheck`), joint rounds or each
  conversation alone, with or without a drafter (greedy tokens), and the same seed gives the same text. Goldens in
  `ref/golden/` keep the tokens and logit fingerprints of each toy and real model and are compared bit for bit. The
  checks run with negative controls (`--perturb`) that must fail, so a check that stops looking is caught.
- **Against the outside, fidelity.** Against mlx-lm in fp32 on the same quantized weights, with 64 forced greedy
  positions: mean KL < 0.01, and top-1 differences only where the reference itself has a small margin. The mlx-lm bf16
  floor is reported next to it. When a family is added, mlx-lm is cross-checked once against transformers with bf16
  weights. Where mlx-lm lacks a piece or deviates from the config, a wrapper `ref/<name>_ref.py` fixes the reference.
- **Between versions,** tokens change only on purpose: fidelity is measured again and the goldens are regenerated.
- **One exception, as a class: algebraically different paths of one family.** When a family's prefill has two
  formulations of the same layer that cannot round alike, and the cheaper one depends on the chunk's shape, a prompt
  split elsewhere may cross from one to the other. Today the only case is GLM's latent MLA: absorbed attention, or
  decompressed attention for chunks of at least 1,024 rows over at least 8K keys (`MLAProjection.decompressed`). There,
  `splitcheck` compares split and whole with a tolerance instead of bit for bit: the same top-1 and a bounded KL at
  every step (`latentMaxKL`, with the measured distance far below it), on a prompt long enough to cross the condition
  (with the exact prefill: under W8A8 the int8 roundings amplify the paths' difference, so that run stays on a prompt
  that never crosses it and bit-exact); that family's prefill has no lossy Q·Kᵀ, which would make a row lossy or exact
  by its chunk; its negative control (`--perturb` scales the latent KV) has to fail. Within each path the rest stays bit-exact
  (`reuse`, `prefixcache`, goldens, `rowscheck`). No other family gets a tolerance without its own decision.

## Alternatives considered

- A tolerance against the previous binary: errors drift and path bugs stay below the threshold.
- Compare against transformers directly: it cannot read the MLX 4-bit weights, so quantization and engine errors mix.
- Compare tokens only: a divergence can start at the last prompt row and show in the text hundreds of tokens later.
- mlx-lm in bf16 as the reference: its own rounding noise is as large as the errors to detect.

## Consequences

- The summation order of a kernel cannot depend on the chunk size or the number of rows; faster kernels that round
  differently by shape are rejected or rewritten.
- A change that alters the summation order on purpose regenerates goldens, with its fidelity measured first.
- Toys whose loss comes from a known source (a chaotic random router, the int8 KV format) get their own bound, set
  relative to their bf16 floor and justified in `scripts/ci/fidelity.sh`.
- Paths that only real weights exercise (near ties, drafter copies, SSD experts) need the real-model mode of the CI.
