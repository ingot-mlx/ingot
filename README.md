# ingot

An LLM inference engine for Apple silicon written in Swift with hand-written Metal kernels, with no MLX or Python at
run time. It loads MLX safetensors checkpoints and serves an OpenAI-compatible chat API (`ingot serve`).
It runs exact speculative decoding with DFlash2 drafters and copy-from-context, and keeps an exact prefix cache.
Exactness is the rule: the same tokens with and without speculation, bit-exact goldens per model and fidelity against
mlx-lm in fp32; a lossy option is never on unless it is measured and announced (see [Main options](#main-options)).

## Requirements

- A Mac with an M5 chip or later: a GPU with matrix accelerators (Metal family `Apple10`; the kernels use
  `mpp::tensor_ops::matmul2d`) and Metal 4.
- macOS 27 or later and a Swift 6.2 toolchain.

The check runs when the Metal device is created (`MachineRequirements`): if something is missing, the error names the
chip, the OS version and what is missing. `INGOT_SKIP_CHECK=1` skips it (diagnostics only).

## Quick start

Install with Homebrew (a bottle; without one it builds with Xcode, about 30 s) or build from source (the binaries
end up in `.build/release/`: use `.build/release/ingot` below):

```sh
brew install ingot-mlx/tap/ingot            # ingot and ingot-dev on your PATH; brew upgrade ingot to update
git clone https://github.com/ingot-mlx/ingot && cd ingot        # build from source instead of brew
swift build -c release                      # .build/release/ingot (and ingot-dev); ingot --version: 0.1.0
uv tool install huggingface_hub             # or: pip install -U huggingface_hub; gives the `hf` command
hf download LiquidAI/LFM2.5-8B-A1B-MLX-4bit # 4.8 GB; goes to the Hugging Face cache
ingot serve --model LiquidAI/LFM2.5-8B-A1B-MLX-4bit    # 127.0.0.1:8090
```

`ingot` never downloads anything: `--model` is a directory or an `org/repo` already in the Hugging Face cache
(`HF_HUB_CACHE`, default `~/.cache/huggingface/hub`), and so is a family's default drafter (table of families): if it
is missing, `ingot serve` and `ingot run` stop with the `hf download` line that gets it (or run with `--drafter none`).
When the server prints its memory plan and starts listening, from another terminal:

```sh
curl -s http://127.0.0.1:8090/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "LiquidAI/LFM2.5-8B-A1B-MLX-4bit",
  "messages": [{"role": "user", "content": "Explain speculative decoding in two sentences."}]
}'
```

`GET /health` and `GET /v1/models` are there too; Ctrl-C stops the server. More models: next section and
[Supported models](#supported-models).

## Which model for my Mac

`ingot serve` chooses the context, the prefix cache, the snapshots, the prefill chunk and the drafter from the Mac's
memory and the model's size, and refuses to load a model that does not fit even at 8K context and without a drafter.
The table is that plan (context · drafter) for models in 4-bit MLX form, as of 2026-10-30. **Only the 32 GB column was
run on a real Mac**; the 16, 24 and 48 GB columns are `ingot plan --mem-limit-gb N` simulations (the GPU memory limit
scaled with the Mac's memory, which a real Mac may not match), and nothing here is a speed measurement. «no» = does not
fit.

| Model (weights) | 16 GB | 24 GB | 32 GB | 48 GB |
|---|---|---|---|---|
| Qwen3.6-35B-A3B MoE and its fine-tunes (19.5 GB) | 64K · drafter, experts from the SSD (6.4 GB cache) | 64K · drafter, experts from the SSD (13.1 GB cache) | 128K · drafter | 256K · drafter |
| Qwen3.8-27B (15.1 GB) | no | 16K · no drafter | 64K · drafter | 256K · drafter |
| LFM2.5-24B-A2B (13.6 GB)\* | no | 128K | 128K | 128K |
| Gemma 4 26B-A4B (14.2 GB) | no | 64K · drafter | 256K · drafter | 256K · drafter |
| gpt-oss-20b MXFP4-Q8 (12.1 GB) | no | 128K · drafter | 128K · drafter | 128K · drafter |
| GLM-4.7-Flash (17.0 GB) | no | no | 99K | 198K |
| LFM2.5-8B-A1B (4.8 GB)\* | 125K | 125K | 125K | 125K |

\* The plan warns that the calibration for this family is estimated, not measured.

When the Qwen3.6 MoE does not fit in memory with 32K of context and its drafter, the plan reads its experts from the SSD
and gives the expert cache all the memory left after 64K of context, a 0.25 GB prefix cache, 4 snapshots and the drafter
(exact, but slower than in memory: the more cache, the faster). Only the Qwen3.6 MoE family can do it (Flash-Next
always does). `--expert-cache-gb X` sets the cache by hand, and then the plan fits the rest around it. If a Mac reports
less GPU memory than assumed, fewer rows fit and the expert cache is smaller: with 10.7 GB of GPU memory on a 16 GB
Mac, 3.7 GB (`ingot plan --mem-limit-gb 16 --rws-gb 10.7`); with 16 GB on a 24 GB Mac, 9.0 GB, and only gpt-oss-20b (64K,
without its drafter) and LFM2.5-8B-A1B fit without SSD experts (`ingot plan --mem-limit-gb 24 --rws-gb 16`).

### Plan a model before downloading it

`ingot plan` reads `config.json` and the headers of the `.safetensors` files, not the weights, so a folder with the
headers only gives the same plan as the full model (a model with a single file has no index: its file is
`model.safetensors`). The drafter is looked up in the Hugging Face cache; without one, the plan says no drafter:

```sh
R=LiquidAI/LFM2.5-8B-A1B-MLX-4bit; U=https://huggingface.co/$R/resolve/main; mkdir stub && cd stub
for f in config.json model.safetensors.index.json; do curl -sfL -O $U/$f; done
for f in $(python3 -c 'import json;print(*sorted(set(json.load(open("model.safetensors.index.json"))["weight_map"].values())))'); do
  n=$(curl -sL -r 0-7 $U/$f | od -An -tu8 | tr -d ' ')       # the first 8 bytes: the header length
  curl -sL -r 0-$((8+n-1)) -o $f $U/$f                        # the header
done
cd .. && .build/release/ingot plan --model stub --mem-limit-gb 16   # --mem-limit-gb and --rws-gb simulate another Mac
```

## Commands

Two executables: **`ingot`** to use the engine and **`ingot-dev`** for exactness checks and benches (see
[docs/development.md](docs/development.md)). `--model` is required and is a directory (`~` is expanded) or an `org/repo`
in the Hugging Face cache.

| `ingot` | What it does |
|---|---|
| `serve` | OpenAI-compatible server (below) |
| `run` | one prompt with the model's chat template: `ingot run --model M --prompt "text"` (or standard input); the answer on stdout, reasoning and figures on stderr; `--system`, `--max-tokens`, `--no-think`, `--temp`, `--top-p`, `--top-k`, `--seed` |
| `plan` | the memory plan without loading weights; without `--model`, every model under `$INGOT_MODELS`; `--mem-limit-gb N` and `--rws-gb X` simulate another Mac |
| `info` | model files, config and quantization of one layer (`--layer N`) and the GPU |

## ingot serve

```sh
.build/release/ingot serve --model ~/models/Qwen3.6-35B-A3B-MLX-4bit --name local/qwen3.6-35b-a3b --port 8090
```

It listens on `127.0.0.1:8090` by default (`--host`, `--port`); `--name` is the model id the API reports. One model per
process, with its DFlash2 or DFlash v1 drafter when the family has one (`--drafter none` turns it off).

- **Memory plan:** without options, the context, prefix cache, snapshots, prefill chunk and drafter are chosen from the
  Mac's memory (`recommendedMaxWorkingSetSize` and physical memory) and the model's footprint (read from `config.json`
  and the safetensors headers). Below a 32K context it trims, in order, the prefix cache, the snapshots, the chunk, the
  context and the drafter; if even that does not fit, it refuses to load and says why. `--max-ctx N`,
  `--prefix-cache-gb X`, `--snapshots N`, `--chunk N` and `--drafter none` override the plan (with a warning if they do
  not fit). `ingot plan` shows the plan for the same options.
- **Concurrency:** requests queue. Dense attention and GDN models without a drafter decode up to 4 conversations in
  joint rounds; models with a drafter or MoE layers keep one slot per conversation and serve them one at a time, each
  resuming from its live state.
- **API:** `POST /v1/chat/completions` (with and without `stream`, `stream_options.include_usage`), `GET /v1/models`,
  `GET /health`. Reasoning comes back in `reasoning_content`; `chat_template_kwargs` (e.g. `enable_thinking`),
  `reasoning_effort`, `tools`/`tool_choice` and `tool` messages reach the chat template. Sampling: `temperature`,
  `top_p`, `top_k`, `min_p`, `seed`, `stop`, the repetition, presence and frequency penalties and `max_tokens`
  (`max_completion_tokens`); the defaults come from `generation_config.json`. Temperature 0 gives the same tokens as
  `ingot-dev gen`; with temperature, speculation verifies by rejection sampling, so the output distribution does not
  change. Fields the engine does not implement (`n`, `logprobs`, `logit_bias`, …) return HTTP 400 unless neutral, and
  so does a prompt that does not fit. `ingot_return_tokens: true` (an extension) returns the token ids. The full list is
  in `ref/api-surface.txt`.
- **Tool calls:** the syntax is detected from the chat template (Qwen XML, LFM pythonic, GLM, Hermes JSON, Gemma 4,
  Muse ATEM, gpt-oss harmony) and returned as OpenAI `tool_calls`, also when streaming.
- **Tokenizer and template** in Swift: BPE over `tokenizer.json` and `chat_template.jinja` through swift-jinja.
- **Prefix cache:** after each reply the state stays reusable, with snapshots at turn boundaries, so the next turn
  starts from the longest cached prefix. Entries across sessions live in an LRU (`--prefix-cache-gb`, 0 = off) and are
  saved to `~/.cache/ingot/prefix/` at shutdown: SIGINT, SIGTERM or SIGHUP (closing the terminal) (`--prefix-disk 0`
  turns that off). Resuming gives the same bits as reading the whole prompt (`ingot-dev prefixcache`).
- Other options: `--agent-guards on` (off by default; for agent sessions: a reply stuck in a token loop is cut with
  `finish_reason: "length"`),
  `--dump-requests DIR` (saves every request, whole conversations included: debugging only), `--expert-preload 0`,
  `--route-trace FILE` and the [engine options](#main-options).

Any OpenAI client works; for example, in Python:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8090/v1", api_key="local")
stream = client.chat.completions.create(
    model="local/qwen3.6-35b-a3b", stream=True,
    messages=[{"role": "user", "content": "Write a haiku about Metal."}])
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="")
```

## Supported models

Tier-1 families: each has a toy model and a real model in the CI, fidelity against mlx-lm and performance work
([ADR 0004](docs/adr/0004-family-tiers-and-two-mode-ci-on-toys-from-real-configs.md)). The family comes from
`model_type` in `config.json` (`Loader/ModelSpec.swift`); a family or shape the engine cannot run is a load error that
names what is missing, never a silent fallback.

| Family (`model_type`) | CI real model (`REAL_MODELS` tag) | Weights | Drafter |
|---|---|---|---|
| Qwen3.5/3.6 MoE (`qwen3_5_moe`) | Qwen3.6-35B-A3B Heretic, 4-bit (`moe`) | 4-bit MLX conversions such as `froggeric/Qwen3.6-35B-A3B-Uncensored-Heretic-MLX-4bit`; the CI uses a calibrated 4-bit conversion | `incoai/Qwen3.6-35B-A3B-DFlash2` |
| Qwen3.5/3.8 dense (`qwen3_5`; also Bonsai 2) | Qwen3.5-0.8B (`dense`) | `mlx-community/Qwen3.5-0.8B-4bit`; larger ones such as Qwen3.8-27B | the model's `drafter/` folder (27B); none on the small ones |
| Qwen3.8-Flash-Next (`qwen4_exp`) | Flash-Next Q2 (`flash`) | own lossless format, converted with `scripts/flashnext/` from `Qwen/Qwen3.8-Flash-Next` and the Q2 GGUF experts, layout under `ingot` in `config.json` ([ADR 0006](docs/adr/0006-weights-mlx-safetensors-no-gguf-flash-next-format-and-ssd-experts.md)) | `incoai/Qwen3.8-Flash-Next-DFlash2` |
| LFM2 / LFM2.5 (`lfm2_moe`) | LFM2.5-8B-A1B (`lfm`) | `LiquidAI/LFM2.5-8B-A1B-MLX-4bit` | none |
| LFM3 (`lfm3`) | LFM3-1.7B (`lfm3`) | `LiquidAI/LFM3-1.7B-MLX-8bit` | none |
| Granite (`granite`) | Granite 4.1 3B (`granite`) | `mlx-community/granite-4.1-3b-4bit` with the tokenizer files of `ibm-granite/granite-4.1-3b` | none |
| Gemma 4 (`gemma4`) | Gemma 4 E2B (`gemma4`) | `mlx-community/gemma-4-e2b-it-4bit`; also the 26B-A4B and the 31B | `z-lab/gemma-4-26B-A4B-it-DFlash` (DFlash v1) on the 26B-A4B; none on the others |
| gpt-oss (`gpt_oss`) | gpt-oss-20b (`gptoss`) | `mlx-community/gpt-oss-20b-MXFP4-Q8` | `z-lab/gpt-oss-20b-DFlash` (DFlash v1, 0.77 GB loaded) on the 20b; none on the 120b |
| Nemotron-H (`nemotron_h`) | Nemotron 3 Nano 4B (`nemotron`) | `mlx-community/NVIDIA-Nemotron-3-Nano-4B-4bit` | none |
| GLM-4.7-Flash (`glm4_moe_lite`) | GLM-4.7-Flash (`glm`) | `mlx-community/GLM-4.7-Flash-4bit` with `lm_head` re-quantized to 8 bits from `zai-org/GLM-4.7-Flash` | none |
| Muse Glimmer (`muse_glimmer`) | Muse Glimmer 30B (`muse`) | `mlx-community/Muse-Glimmer-30B-4bit` (text tower only) | `incoai/Muse-Glimmer-30B-DFlash2` |

No tier-2 families: Qwen3 MoE (`qwen3_moe`) and Gemma 3 text (`gemma3_text`) are load errors with their
name. The checkpoints and revisions the real goldens were written from are in `ref/golden/README.md`.

## Main options

Engine options are the same in `ingot` and `ingot-dev` and are set on the command line (`EngineOptions`); the engine reads
no environment variables other than the ones listed in `ref/api-surface.txt`. Defaults depend on the architecture and
the command, not on the model's name.

| Option | What it does | Default |
|---|---|---|
| `--int8 0\|1` | prefill dense GEMMs in rotated W8A8 from the family's position (lossy) | `ingot serve` and `prefixcache`: 1 on MoE models, 0 on dense ones; elsewhere 0 |
| `--qk8 0\|1` | prefill Q·Kᵀ in int8 from 8K (lossy) | as `--int8` |
| `--copy auto\|on\|off` | copy-from-context speculation (exact) | `auto`: on for MoE models and models without a drafter, off for dense models with one |
| `--tree N` | draft tree of N nodes on dense Qwen models with DFlash2, greedy, prompts under 16K (exact) | 12 (`0` = the chain) |
| `--drafter org/repo\|dir\|none` | drafter | the model's `drafter/` or the family's |
| `--max-ctx`, `--prefix-cache-gb`, `--snapshots`, `--chunk`, `--mem-limit-gb`, `--rws-gb`, `--kv-pool` | memory plan overrides | from the plan |
| `--expert-cache-gb X` | keep X GB of experts in memory and read the rest from the SSD (exact) | all in memory if the model fits; if not, the Qwen3.6 MoE streams with the cache the plan leaves; Flash-Next always streams |
| `--expert-skip-rank N` | Flash-Next: do not wait for missing experts whose best rank among the rows is N + 1 or worse (5 = ranks 6–10; lossy; 0 = off) | 5 in `ingot serve`, off elsewhere |
| `--spec-rows N`, `--prep-trace` | benches: fixed verification rows; prompt preparation timings | off |
| `--exp key=value` | experiments on branches only; nothing reads them on main | none |

Every lossy option is announced at load. The other `--expert-*` options and `--ple-cache-mb` tune the SSD expert path
(`Sources/IngotCLI/EngineCLI.swift`); the tuning options kept on purpose, with when to use each, are listed in
`ref/api-surface.txt` («Permanent tuning options», `PERMANENT` in `scripts/api_surface.py`). The environment
variables (`HF_HUB_CACHE`, `INGOT_MODELS`, …) are in [docs/development.md](docs/development.md#environment-variables).

## Code layout

```
Package.swift
Sources/Ingot/        the engine
  GPU.swift              MetalContext (device, queue, library, pipelines), ModelDims (the LAB_* kernel macros)
  Loader/                safetensors by mmap, config.json, quantization, Hugging Face cache paths; ModelSpec.swift is
                         the family table (the only place that reads model_type), with a spec file per family
  Layers/                quantized linear (DenseQ, NAX GEMMs, W8A8), MoE, GDN, short convolution, attention, MLA
  Model.swift            HybridModel: loading, chunked prefill, verification of B rows (with ModelPrefill.swift,
                         ModelRound.swift, ModelCapture.swift)
  Generate.swift         normal, speculative and chained generation; TreeVerify.swift, CopySpec.swift, SpecRows.swift
  Drafter.swift          DFlash2 drafter (context ring, block, selector)
  Engine.swift           InferenceEngine: live state and snapshots; PrefixCache.swift, BatchServe.swift
  Flash*.swift           Qwen3.8-Flash-Next (hyper-connections, PLE, QSA, int8 Q2 experts)
  Expert*.swift          experts read from the SSD (cache, read-ahead, prefill streaming)
  Sampling.swift         sampling parameters and GPU candidates
  EngineOptions.swift    engine options (set by the CLI)
  Memory/                MachineRequirements, MachineInfo, ModelFootprint, MemoryPlanner, KVPool
  Calibration/           per-family calibration of the verification rows (rows-<family>.json)
  Shaders/               Metal kernels, concatenated in alphabetical order and compiled at startup with the model's
                         dimensions as macros
Sources/IngotServe/        server: tokenizer, chat template, chat dialects and tool-call parsers, HTTP, ChatServer
Sources/IngotCLI/          what both CLIs share: arguments, engine options, memory plan
Sources/IngotBench/        ingot-dev only: .npy reading and the toy models' configs
Sources/IngotTool/       ingot: serve, run, plan, info
Sources/ingot-dev/         ingot-dev: gen, kl, rowscal, Flash-Next tools, Checks/ (reuse, prefixcache, splitcheck,
                         rowscheck, batchcheck, shapes), Toy/ (toy model generator)
Tests/                   IngotTests (CPU, GPU and per-kernel cases), IngotServeTests (tokenizer, templates, parsers)
ref/                     Python references with mlx/mlx-lm, prompts, goldens (golden/) and toy configs (toy/)
scripts/                 ci.sh and ci/, api_surface.py, flashnext/ (the Flash-Next conversion)
docs/                    architecture decision records (adr/) and development notes
```

## Development and design decisions

Build, tests, the CI, goldens, references and the `ingot-dev` commands (including `ingot-dev gen`, the exactness regression
and the in-process benchmark): [docs/development.md](docs/development.md). The decisions that shape the engine (own
Swift + Metal engine, exactness, lossy options, family tiers and the CI, DFlash2, weight formats, `ingot serve`) are
recorded in [docs/adr/](docs/adr/README.md).

GitHub Actions only runs lint (SwiftLint, swift-format, ruff, shellcheck). The real CI (`scripts/ci.sh --toy`) needs a Mac
with an M5 and macOS 27, so it runs locally before every merge.

## Credits

Quantization format and fragment layout taken from MLX (MIT, Apple). Code and ideas from MLX, mlx-lm, Strata,
splash, oMLX, TensorFold, mlx-serve and others are credited, with their licences, in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and in the comments where they are used. Public Apple API only.

## License

MIT, see [LICENSE](LICENSE).