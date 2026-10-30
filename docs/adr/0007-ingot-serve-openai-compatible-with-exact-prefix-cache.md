# 7. `ingot serve`: an OpenAI-compatible server with an exact prefix cache

Status: Accepted

## Context

The engine is used from coding-agent clients that speak the OpenAI chat completions API. Agent sessions resend long
prompts that are mostly identical (system prompt and tool definitions), turn after turn and from one session to the
next, and subagents run several conversations against the same server. The runtime has no Python (ADR 1).

## Decision

- `ingot serve` (`Sources/IngotTool/Serve.swift`, `IngotServe`) serves `POST /v1/chat/completions` (with and without streaming),
  `GET /v1/models` and `GET /health` over Network.framework. Tokenizer, Jinja chat template (`swift-jinja`), reasoning
  split into `reasoning_content` and per-dialect tool-call parsers are in Swift.
- Request fields it does not implement (`logit_bias`, `logprobs`…) get a 400; they are never ignored. Sampling defaults
  come from the model's `generation_config`.
- One model per process. Context, prefix cache, snapshots, prefill chunk and drafter are sized from the machine's
  memory by `MemoryPlanner`, which trims in that order and never switches model.
- **Prefix cache** (`PrefixCache.swift`): an entry is the exact engine state at a cut (recurrent state plus a copy of
  the int8 KV), taken at 64-aligned points chosen from the template's dialect: the end of the system block, the
  boundary shared with recent prompts, and an abandoned conversation. Entries are kept in an LRU with a byte limit and
  can be saved to disk. A request resumes from the longest entry that prefixes its prompt, and the rest of the prompt
  always goes through the prefill, so a resumed request gives the same bits as a cold one (ADR 2), also with the same
  seed.
- **Several conversations:** models without a drafter join queued conversations into joint rounds; models with a
  drafter (ADR 5) or with layers the joint round does not cover keep one slot per conversation, served one at a time,
  so no conversation re-reads its history. Headers are sent when a request is queued, with keep-alive comments until
  the first token.

## Alternatives considered

- Put the engine behind a Python server: brings back the Python runtime.
- Ignore unknown request fields: the client would get silently different behavior.
- Paged KV with a prefix tree shared across requests: more machinery than a single-user server needs, and hybrid
  models still need snapshots of their recurrent state.
- No prefix cache: every agent turn would re-read the whole system prompt and tools.

## Consequences

- Templates and tool-call parsers must match transformers and mlx-lm; `IngotServeTests` keeps vectors per tier-1
  family, and a new family needs its dialect's prefix-cache markers.
- `ref/serve_check.py` checks end to end that the server gives the same prompt ids and tokens as `ingot-dev gen` with
  the same options.
- The default memory plan can leave no room for extra slots; serving several conversations of a large model needs a
  smaller context and cache.
- Batching is limited to what the joint round covers; with a drafter, concurrency only avoids re-reading, it does not
  add throughput.
