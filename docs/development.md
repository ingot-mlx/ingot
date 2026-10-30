# Development

Notes for working on the engine: what the checks prove, goldens, references and the `lab-dev` commands. The why of
each rule is in the [ADRs](adr/README.md); the help of every script is authoritative (`scripts/ci.sh --help`).
To build and use the engine, see the [README](../README.md).

## Build, test and CI

```sh
swift build -c release            # .build/release/lab (use the engine) and .build/release/lab-dev (exactness, benches)
swift test                        # unit tests; the GPU ones need the Metal device
scripts/ci.sh --toy [--deadcode]  # the CI of every change, toy models only
scripts/ci.sh --real dense,lfm    # only the named real models (tags of REAL_MODELS in scripts/ci/real.sh; all = every one)
             [--validate]         #   with Metal's debug layer and shader validation
```

- **`--toy`**: static checks (release build without warnings, SwiftLint `--strict`, swift-format,
  `ref/api-surface.txt`), `swift test`, the exactness regression of every toy model (token goldens and logit
  fingerprints, `reuse`, `prefixcache`, `splitcheck`, and negative controls that must fail) and the toys' fidelity
  against mlx-lm fp32. `lab-dev toymodels` generates the toys from real configs into `~/.cache/metal-lab/toy`. It needs
  a Python with mlx-lm: `ref/.venv` (below) or `CI_REF_PY`; `--deadcode` adds Periphery.
- **`--real`**: for each named model, `lab-dev gen --check` against its golden (and mlx-lm's greedy for models without
  a drafter), `reuse`, `prefixcache` with `lab serve`'s defaults and the model's own extra steps. The model folders come
  from the `CI_MODEL_*` variables and a missing folder fails the run. Real goldens are machine-local.
- **GitHub Actions** (`.github/workflows/lint.yml`) only runs lint on Linux, without building: SwiftLint `--strict`, swift-format
  `lint --strict`, ruff on `ref/` and `scripts/` (`ruff.toml`) and shellcheck on `scripts/**/*.sh` (`.shellcheckrc`). The real
  CI above needs a Mac with an M5 and macOS 27: run `scripts/ci.sh --toy` locally before every merge.
- Tools: `brew install swiftlint` (swift-format ships with the toolchain; `brew install --build-from-source periphery`
  for `--deadcode`). Style lives in `.swift-format`, `.swiftlint.yml` and `.periphery.yml`.

## What each check proves

- **`swift test`** covers what the end-to-end checks cannot see: tokenizer and chat-template parity with transformers
  (vectors from `ref/tok_ref.py`, `ref/serve_ref.py`, `ref/tool_ref.py`), special tokens inside content, the streaming
  output and tool-call parsers, GPU sampling, GPU row choice against the CPU, the copy-from-context index, the memory
  planner, option defaults, the family specs read from real configs, MLX safetensors reading and that every kernel
  instance compiles.
- **Kernel cases** (`KernelCase*.swift`, with the GPU) compare kernels, through the engine's launchers, against a CPU
  reference on synthetic tensors, only where end-to-end exactness does not reach: families or shapes without a model on
  the machine, the lossy paths against their emulation, row invariance. Each case says in one line which bug it
  catches. `scripts/ci.sh --toy` runs them when the branch touches what they cover (`CI_KCASE=1` always).
- **Toy models** (`lab-dev toymodels`): 4 layers with random weights built from the real `config.json` files in
  `ref/toy/real/`, one per tier-1 family, each checked against its real config (`toyMatchesReal`). Their goldens
  (`ref/golden/toy-*.json`) store the tokens of `--prompts 512,4K --n 128` and logit fingerprints, bit-exact.
- **Path checks** (`lab-dev reuse`, `prefixcache`, `splitcheck`, `rowscheck`, `batchcheck`): different paths that must
  give the same bits (reusable state and snapshots against the step-by-step path, the prefix cache against reading the
  whole prompt, a prefill split at 64-aligned cuts against one pass, trimmed or k-row rounds against full ones, joint
  rounds against one conversation alone). When a bit-exact test fails it also prints by how much (max |Δlogit| and
  NMSE).
- **Negative controls:** `reuse`, `prefixcache` and `splitcheck` with the hidden `--perturb` flag must report FAILED;
  if they pass, the test is not looking. `CI_NEG_SELFTEST=1` runs them unperturbed and the CI has to fail.
- **Fidelity:** forced greedy of 64 positions against mlx-lm with fp32 activations, mean KL < 0.01 (toys in `--toy`,
  real models without a drafter in `--real` through mlx-lm's greedy).
- **Real goldens** (`ref/golden/<model folder>.json`): catch what changes every path at once (a loader or prefill bug).
  They are machine-local; `ref/golden/README.md` lists the checkpoints they were written from.

Every CI step leaves its duration in `.build/ci/timings.txt`.

## Goldens

A change that should keep tokens must keep every golden. If an exact change alters rounding on purpose (another
summation order), rewrite the goldens and say so in the commit:

```sh
TOY_WRITE_GOLDEN=1 scripts/ci.sh --toy                                               # toys
.build/release/lab-dev gen --model M --prompts 512,4K --n 128 --pdir prompts-<x> \
    --golden ref/golden/<model folder>.json --write-golden                          # one real model
```

On another Mac, write your own real goldens: they only match with the same weights, prompt files and hardware.

## References (`ref/`)

Tokenized prompts and mlx-lm references come from the Python scripts in `ref/` (their output, `ref/out/`, is not
tracked):

```sh
cd ref && uv venv -p 3.13 .venv && uv pip install -p .venv/bin/python mlx mlx-lm numpy pillow
.venv/bin/python prompts.py --model <model dir> --out prompts-<x>   # tokenized prompts (chat, 512, 4K, 16K, 32K)
.venv/bin/python gen_ref.py --prompt 512 --n 200                    # mlx-lm's greedy, to compare against
```

The scripts, in `ref/`:

- `prompts.py`: tokenized prompts (`chat`, `512`, `4K`, `16K`, `32K`) with a model's chat template into
  `out/prompts-<x>`; `--fixed-now` for templates that print the date.
- `gen_ref.py`: mlx-lm's greedy (`out/gen-<x>`), what `gen --check` compares models without a drafter against.
- `logits_ref.py [--f32]`, `prefix_ref.py`: logits for `lab-dev kl --ref mlx|mlx-f32|mlx-prefix`.
- `serve_check.py`: with `lab serve` running, the same tokens as `lab-dev gen` with the server's defaults.
- `tok_ref.py`, `serve_ref.py`, `tool_ref.py`: vectors for the tokenizer, template and tool-call tests.
- `kl_pair.py`, `toy_long_prompt.py`, `ttft_turn2.py`: by-hand tools: the KL between two logits files, a long prompt for the
  toy models, and the time to the first token of a second turn with `lab serve` running.
- Per-family references for pieces mlx-lm does not cover the same way (`lfm3_ref.py`, `yarn_ref.py`, `mxfp4_ref.py`,
  `flashnext/harness.py`, …).

## lab-dev gen

The exactness regression of any change, and the in-process benchmark:

```sh
.build/release/lab-dev gen --model M --prompts 512,4K --n 128 --check                       # regression
.build/release/lab-dev gen --model M --prompts 512,4K --n 128 --check --golden ref/golden/<model>.json
.build/release/lab-dev gen --model M --prompts chat,512,4K,16K,32K --n 512 --jsonl f.jsonl --label run1   # measure
```

`--check` generates each prompt with and without the drafter at temperature 0 (without a drafter: normal against
chained decoding, and against mlx-lm's greedy when its reference exists) and exits with 1 if the tokens differ.
Generated ids go to `ref/out/gen/<prompt>_lab_<mode>.json`.

| Option | What it does |
|---|---|
| `--model org/repo\|dir` | model (required) |
| `--prompts a,b,…` | names in `ref/out/<pdir>/<name>.json`, or paths to `.json` files with `ids` (default `chat`) |
| `--pdir dir` | prompt set under `ref/out/` (default: the family's) |
| `--n N` | tokens to generate (default 128) |
| `--mode both\|spec\|normal` | with and without the drafter, only with it, or only without it |
| `--check` | mode `both`, and fail if the tokens differ |
| `--golden f.json [--write-golden]` | compare with (or write) the model's golden; `--golden-strict` fails if the golden or the mlx-lm reference is missing; `--golden-logits` adds logit fingerprints |
| `--drafter org/repo\|dir\|none` | drafter (default: the model's `drafter/` folder or the family's) |
| `--nostop` | do not stop at the end token |
| `--profile` | the pieces of a round separately and a GPU timeline; without a drafter, the batching ceiling and the B = 1 round by piece (attention, short-conv and dense-FFN layers) |
| `--jsonl path --label label` | append one line per measurement |

## lab-dev commands

`lab-dev` with no command prints its usage. Every command that loads a model needs `--model`.

| Command | What it checks or measures |
|---|---|
| `gen` | generation with and without the drafter; `--check`, `--golden`, `--jsonl`, `--profile` (above) |
| `reuse` | state after a reply and snapshots equal to the step-by-step path; `--n A-B --cover N,N+1` searches the reply lengths that end in each rewind case |
| `prefixcache` | the cross-session prefix cache with `lab serve`'s defaults (`--int8 0 --qk8 0` for the exact path) |
| `splitcheck` | prefill split at 64-aligned cuts (`--cuts`) equal to one pass |
| `rowscheck` | with a drafter, verification trimmed to k rows on the GPU equal to 8 rows; without one, k-row rounds equal to k one-row steps |
| `batchcheck` | joint decoding of N conversations equal to each alone (`--rows 2,4`, `--seed S` with sampling, `--reps R` to time it; `--serve` for `lab serve`'s slot loop) |
| `shapes` | several toy shapes in one process, each against its golden |
| `toymodels` | generates the toy models (`--out DIR`, `--arch`, `--force`) |
| `kl` | KL against the exact path (default) or mlx-lm (`--ref mlx`, `mlx-f32`, `mlx-prefix`) |
| `rowscal` | calibrates the confidence-based verification rows on this Mac and writes `metal-lab-rows.json` |
| `fncheck`, `fnprof` | Flash-Next against its harness; per-piece kernel profile |

## Verification rows

The DFlash2 rounds of a family with a calibration verify between 2 and 8 rows chosen on the GPU from the drafter's
calibrated confidence and the measured cost per row count. The calibration ships per family in
`Sources/MetalLab/Calibration/rows-<family>.json`; a `metal-lab-rows.json` in the model folder takes precedence. A
family without one always verifies 8 rows. `lab-dev rowscal --model M --out f` writes one for the current Mac.

## Environment variables

Nothing in the engine reads the environment except:

- `HF_HUB_CACHE`: the Hugging Face cache (default `~/.cache/huggingface/hub`).
- `METAL_LAB_MODELS`: the models directory `lab plan` walks without `--model`. With it, as in the CI, the tokenizer and
  template tests fail when a family's tokenizer is missing from the Hugging Face cache; without it they skip it.
- `METAL_LAB_SKIP_CHECK=1`: skips the machine requirements check.

The CI's own variables (`CI_MODEL_*`, `CI_REF_PY`, `CI_KCASE`, …) are described in `scripts/ci.sh --help`.

## Weights in the GPU

Safetensors are mapped with `mmap`; each tensor is copied into an aligned buffer at load. Matrices sharing K are concatenated by rows (for example the GDN's
`qkv|z|b|a`, attention's `q|k|v` and the dense FFN's `gate|up`).
