# 5. Exact speculative decoding with DFlash2 and verification rows chosen on the GPU

Status: Accepted

## Context

Decoding is bound by memory: verifying several rows in the big model costs little more than computing one. Trained
block drafters (DFlash2) exist for the main families. Speculation is only acceptable here if it is exact (ADR 2): the
text must not depend on whether a drafter was used or on how many rows a round verified.

## Decision

- Decode runs in DFlash2 rounds (`Drafter.swift`, `Generate.swift`). The drafter (a few dense layers, fed with hidden
  states of the big model, a non-causal block of 8 or 16, the big model's `lm_head` and a candidate selector) proposes
  a block. The big model verifies it in one pass, and acceptance runs on the GPU. Rounds are chained: the CPU encodes
  round N+1 before waiting for round N.
- **How many rows** a round verifies is decided on the GPU after the drafter (`SpecRows.swift`), from the drafter's
  calibrated confidence and a measured cost per row count. Both live in a per-family calibration file, not in code; a
  family without one verifies the whole block. The choice depends only on the drafter, never on the big model, so
  greedy output is identical with and without the drafter, and sampling uses rejection verification over the chosen
  rows.
- **Row invariance:** the logits of a row do not depend on how many rows the round has. `rowscheck` checks it bit for
  bit, also without a drafter.
- The same verification serves the other proposal sources: copies from the context (`CopySpec.swift`), the main
  speculation of families without a drafter, and a draft tree on dense models in greedy (`TreeVerify.swift`).
- A family's default drafter is only used if it matches the model (hidden size, target layers, heads);
  `--drafter none` works everywhere. Gemma 4's (z-lab's DFlash v1 for the 26B-A4B) is the default only for that
  model's hidden size, and a missing one is a load error, as with Qwen3.6's.
- DFlash v1 drafters (no candidate selector: argmax drafts, block 8) take as confidence that of the argmax over
  their top-16; with a family calibration (Gemma 4 26B-A4B, gpt-oss 20b) their rounds choose rows like DFlash2's,
  otherwise they verify the whole block.

## Alternatives considered

- Plain one-token decode: leaves most of the memory-bound headroom unused.
- Other drafter families (EAGLE-style, other block drafters): they do not fit these architectures or vocabularies.
- Training our own drafter: out of scope for an engine.
- A fixed number of rows: verifies rows that are almost never accepted.
- Choosing rows on the CPU: breaks the chaining of rounds.
- Choosing rows from the big model's outputs: the text would depend on the round structure.

## Consequences

- Every kernel used in verification must be row-invariant (ADR 2); a kernel that switches code path with the row
  count is a bug.
- The drafter keeps its own context ring, which the prefix cache and the server's slots have to save and restore.
- Conversations with a drafter are served one at a time (ADR 7); joint rounds are only for models without one.
- Families without a trained drafter get only copy-from-context speculation.
