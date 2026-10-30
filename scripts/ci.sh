#!/usr/bin/env bash
# Local continuous integration for ingot. Two modes that do not mix («toy es toy, real es
# real»); the steps live in scripts/ci/{lib,static,toys,real,fidelity}.sh.
#
#   scripts/ci.sh --toy      the CI of every integration, with the toy models only (ingot-dev toymodels, in
#                            ~/.cache/ingot/toy; generated if missing or if the generator changes): the static part
#                            (API: exp and ref/api-surface.txt; build, release, no warnings; SwiftLint --strict, 0
#                            warnings; formatting), swift test, the exactness regression of every toy (token golden and
#                            logits fingerprints, reuse, prefixcache and splitcheck, also with INT8, and the negative
#                            control: the same tests with a deliberate perturbation have to report FAILED) and the
#                            fidelity of the toys against mlx-lm fp32 (forced greedy of 64 positions, mean KL < 0.01;
#                            needs CI_REF_PY). swift test, the toys and the fidelity run in five parallel lanes
#                            (scripts/ci/toys.sh).
#                            TOY_WRITE_GOLDEN=1 rewrites ref/golden/toy-*.json (only for deliberately changed tokens).
#   scripts/ci.sh --real M   only the real models named in M (comma-separated tags of the table REAL_MODELS in
#                            scripts/ci/real.sh: moe, dense, flash, lfm, granite, gemma4, gptoss, nemotron, glm, muse,
#                            lfm3; all = every one), each with all its steps: gen --check (golden, and mlx-lm's greedy
#                            for models without a drafter: the real fidelity), kvgrow on Flash-Next, reuse,
#                            prefixcache (with ingot serve's defaults: INT8 prefill on the MoE models, exact on the dense
#                            ones) and rowscheck on the Heretic. No toys and no swift test; of the static part only
#                            the release build that the steps run. Only when needed: a branch that touches what only
#                            a real model covers, before rewriting ref/golden/, or to validate one model; `all` only
#                            to regenerate every golden.
#   --validate               with --real: the real steps with MTL_DEBUG_LAYER=1 and --n 16, then gen --check of each
#                            model with MTL_SHADER_VALIDATION=1
#   --deadcode               with --toy: additionally, dead code with Periphery
#
# Every step is left with its duration in .build/ci/timings.txt (and the GPU tests, "suite NAME" and "kcase ID").
# Variables: CI_MODEL_MOE, CI_MODEL_DENSE, CI_MODEL_FLASH, CI_MODEL_LFM, CI_MODEL_GRANITE, CI_MODEL_GEMMA4,
# CI_MODEL_GPTOSS, CI_MODEL_NEMOTRON, CI_MODEL_GLM, CI_MODEL_MUSE and CI_MODEL_LFM3 (folders of the real models, one per
# family: Qwen3.6-35B-A3B, Qwen3.5-0.8B-4bit, Flash-Next, LFM2.5-8B-A1B-MLX-4bit, granite-4.1-3b-4bit,
# gemma-4-e2b-it-4bit, gpt-oss-20b-MXFP4-Q8, NVIDIA-Nemotron-3-Nano-4B-4bit, GLM-4.7-Flash-4bit-h8,
# Muse-Glimmer-30B-4bit and LFM3-1.7B-MLX-8bit; the golden is ref/golden/<folder name>.json; --real needs those it
# names, and a missing folder fails the CI with how to get it), CI_REF_PY (Python with mlx-lm for --toy; default ref/.venv/bin/python),
# CI_BASE (comparison branch used to decide whether --toy runs the KernelCase* suites; main), CI_KCASE=1 (always run
# them), CI_GPU_WRAP (command prefix for the `swift build` of the static part, e.g. the one that reserves the GPU),
# CI_NEG_SELFTEST=1 (negative-control self-test: runs them without the perturbation and the CI has to fail),
# CI_FID_REF_CACHE=0 (recompute the toys' mlx-lm references instead of taking them from $TOY_DIR/fidref; see
# scripts/ci/fidelity.sh).
# CI_PHASE=static|gpu (internal, for whoever coordinates the GPU externally, like an external CI wrapper; not a mode):
# only the part before the GPU (static part or build, toy models, Python check) or only the GPU part of the same mode.
# Exits with 1 as soon as a required step fails; the informative ones only warn.
# INGOT_MODELS (optional): directory with the calibrated models. With it, the tokenizer and chat-template tests
# (TokenizerTemplateTests, ChatDialectTests) fail when a family's tokenizer repo is missing from the Hugging Face cache
# (the failure says how to download it, without weights); without it they skip it.

# shellcheck disable=SC2034  # STEP_NAME and STEP_T0 are read by the files it sources (ci/lib.sh)
set -euo pipefail
cd "$(dirname "$0")/.."

MODE=""; REAL_LIST=""; VALIDATE=0; DEADCODE=0
while [ $# -gt 0 ]; do
    case "$1" in
        --toy) [ -z "$MODE" ] || { echo "--toy and --real separately" >&2; exit 2; }; MODE=toy ;;
        --real)
            [ -z "$MODE" ] || { echo "--toy and --real separately" >&2; exit 2; }
            { [ $# -ge 2 ] && [ "${2#-}" = "$2" ]; } || { echo "--real needs the models: --real dense,lfm (or all)" >&2; exit 2; }
            MODE=real; REAL_LIST="$2"; shift ;;
        --validate) VALIDATE=1 ;;
        --deadcode) DEADCODE=1 ;;
        -h|--help) sed -n '2,/^$/p' "$0" | sed '$d'; exit 0 ;;
        *) echo "unknown option: $1 (modes: --toy, --real MODELS; see --help)" >&2; exit 2 ;;
    esac
    shift
done
[ -n "$MODE" ] || { echo "usage: scripts/ci.sh --toy [--deadcode] | --real MODELS [--validate]   (see --help)" >&2; exit 2; }
[ "$VALIDATE" = 0 ] || [ "$MODE" = real ] || { echo "--validate goes with --real" >&2; exit 2; }
[ "$DEADCODE" = 0 ] || [ "$MODE" = toy ] || { echo "--deadcode goes with --toy" >&2; exit 2; }
PHASE="${CI_PHASE:-all}"
case "$PHASE" in all|static|gpu) ;; *) echo "CI_PHASE=$PHASE: expected static or gpu" >&2; exit 2 ;; esac

LOG_DIR=".build/ci"; mkdir -p "$LOG_DIR"
CI_GPU_WRAP="${CI_GPU_WRAP:-}"
# Timings per step: NAME<TAB>seconds in .build/ci/timings.txt (CI_PHASE=gpu appends to what CI_PHASE=static left);
# after swift test also "suite NAME<TAB>s" and "kcase ID<TAB>s" (test_timings in scripts/ci/lib.sh).
TIMES="$LOG_DIR/timings.txt"; STEP_NAME=""; STEP_T0=0
if [ "$PHASE" != gpu ] || [ ! -f "$TIMES" ]; then
    printf '# scripts/ci.sh --%s, commit %s, %s\n' "$MODE${REAL_LIST:+ $REAL_LIST}" "$(git rev-parse --short HEAD)" \
        "$(date '+%Y-%m-%d %H:%M:%S')" > "$TIMES"
fi
SCRIPTS=scripts/ci
. "$SCRIPTS/lib.sh"; . "$SCRIPTS/static.sh"; . "$SCRIPTS/toys.sh"; . "$SCRIPTS/real.sh"; . "$SCRIPTS/fidelity.sh"

TOYDIR="${TOY_DIR:-$HOME/.cache/ingot/toy}"
DEV=.build/release/ingot-dev

REAL_TAGS=""
if [ "$MODE" = real ]; then
    # The tags asked for, checked against the table before anything runs; then their model folders.
    REAL_TAGS=$(real_resolve "$REAL_LIST") || fail "--real $REAL_LIST"
    for tag in $REAL_TAGS; do
        v=$(real_model_var "$tag")
        [ -n "${!v:-}" ] || fail "--real $tag needs $v (folder of the real model; see scripts/ci.sh --help)"
    done
fi

# --- Before the GPU ------------------------------------------------------------------------------------------------
if [ "$PHASE" != gpu ]; then
    if [ "$MODE" = toy ]; then
        ci_static
        # Toy models (CPU and disk; ~1 min the first time) and the fidelity Python.
        step "toy models (ingot-dev toymodels in $TOYDIR; no GPU)"
        $DEV toymodels --out "$TOYDIR" > "$LOG_DIR/toymodels.log" 2>&1 \
            || { tail -n 20 "$LOG_DIR/toymodels.log"; fail "toymodels (see $LOG_DIR/toymodels.log)"; }
        cat "$LOG_DIR/toymodels.log"
        ref_py_check
    else
        ci_build   # only the binary the real steps run
    fi
    step_close
    if [ "$PHASE" = static ]; then printf '\nOK (before the GPU, --%s, %d s)\n' "$MODE" "$SECONDS"; exit 0; fi
elif [ "$MODE" = toy ]; then
    ref_py_check
fi

# --- GPU part ------------------------------------------------------------------------------------------------------
CI_GPU_WRAP=""   # here whoever launches the CI already holds the GPU (or does not coordinate it)
if [ "$MODE" = toy ]; then
    ci_toys
else
    if [ "$VALIDATE" = 1 ]; then
        export MTL_DEBUG_LAYER=1 MTL_DEBUG_LAYER_ERROR_MODE=assert
        echo "Metal debug layer enabled (MTL_DEBUG_LAYER), --n 16"
    fi
    # shellcheck disable=SC2086
    ci_real $REAL_TAGS
    if [ "$VALIDATE" = 1 ]; then
        unset MTL_DEBUG_LAYER MTL_DEBUG_LAYER_ERROR_MODE
        export MTL_SHADER_VALIDATION=1
        # shellcheck disable=SC2086
        ci_real_validate $REAL_TAGS
        unset MTL_SHADER_VALIDATION
    fi
fi
step_close
[ "$PHASE" = gpu ] || printf 'total excluding wait\t%d\n' "$SECONDS" >> "$TIMES"
printf '\nOK (--%s, %d s; timings per step in %s)\n' "$MODE${REAL_LIST:+ $REAL_LIST}" "$SECONDS" "$TIMES"
