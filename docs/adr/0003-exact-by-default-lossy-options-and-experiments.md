# 3. Exact by default; lossy options measured and switchable; experiments behind `--exp`

Status: Accepted

## Context

Many speed ideas trade accuracy: int8 activations, int8 attention scores, skipping experts that are not in memory,
fewer bits. Each one is tempting alone, and together they make comparisons with other engines and the bit-exact checks
of ADR 2 meaningless. Experiments also tend to leave options, fields and environment variables behind.

## Decision

- **The exact path is the default.** `ingot-dev` and the engine's defaults are exact. A lossy option becomes a default
  only where it was measured to pay, with its fidelity measured on that family, and only in that scope. Today, in
  `ingot serve` on MoE models: W8A8 dense GEMMs in the prefill from a per-family position (`ModelSpec.int8MinPos`) and
  int8 Q·Kᵀ in the prefill from long contexts (`EngineOptions.serve`); on Flash-Next, the int8 activations of its Q2
  experts and not waiting for low-ranked missing experts.
- Every lossy default is announced at startup (`describeLossy`) and turned off with a flag (`--int8 0 --qk8 0`).
- A lossy option that does not pay for its KL is removed, not left switched off.
- The KV cache is int8 (K rotated by a Hadamard), as the storage format, not an option; its loss counts inside the
  fidelity bound of ADR 2.
- **Experiments** go through `--exp key=value` into `EngineOptions.exp`. The static step of the CI fails if any source
  reads it (`scripts/api_surface.py --exp-check`), so no experiment reaches main.
- **The API is a file.** The options of `ingot` and `ingot-dev`, the fields of `EngineOptions` and the few allowed
  environment variables are listed in `ref/api-surface.txt`, generated and checked by the CI. A new option enters with
  the condition under which it will be removed or made the default.

## Alternatives considered

- Always exact: leaves measured long-prompt read speed on the table for no visible loss.
- Lossy by default everywhere: breaks the comparisons with mlx-lm and the bit-exact goldens.
- Environment variables for experiments: invisible in the API and easy to forget in main.
- Keep finished experiments as options switched off: dead code that every later change has to carry.

## Consequences

- Speed comparisons with other engines run with the lossy options off.
- The end-to-end check of the server (`ref/serve_check.py`) runs `ingot-dev gen` with the options of `ingot serve` and
  refuses to compare runs with different options.
- Changing a lossy default is a decision with its own fidelity measurement, not a side effect of a speed change.
- Reviewing an integration includes the diff of `ref/api-surface.txt`.
