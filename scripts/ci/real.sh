# Exactness regression with real models, --real only (loaded with `source` from scripts/ci.sh).

# One real model per tier-1 family: the smallest that has every piece of the family (what it lacks is covered by the
# toy built from the big model's config). One line per family, whitespace-separated:
#   TAG  VARIABLE (folder of the model)  PROMPTS (ref/out/...)  STEPS
# STEPS are comma-separated (gen, kvgrow, reuse, prefix; see exactness). `scripts/ci.sh --real TAG[,TAG...]` runs the
# lines it names (all = every line) and nothing else: the toys are --toy's.
REAL_MODELS=(
    "moe      CI_MODEL_MOE      prompts            gen,reuse,prefix"
    "dense    CI_MODEL_DENSE    prompts-q35-08b    gen,reuse,prefix"
    "flash    CI_MODEL_FLASH    prompts-fn         gen,kvgrow,reuse,prefix"
    "lfm      CI_MODEL_LFM      prompts-lfm8b      gen,reuse,prefix"
    "granite  CI_MODEL_GRANITE  prompts-granite41  gen,reuse,prefix"
    "gemma4   CI_MODEL_GEMMA4   prompts-gemma4     gen,reuse,prefix"
    "gptoss   CI_MODEL_GPTOSS   prompts-gptoss     gen,reuse,prefix"
    "nemotron CI_MODEL_NEMOTRON prompts-nemotron   gen,reuse,prefix"
    "glm      CI_MODEL_GLM      prompts-glm        gen,reuse,prefix"
    "muse     CI_MODEL_MUSE     prompts-muse       gen,reuse,prefix"
    "lfm3     CI_MODEL_LFM3     prompts-lfm3       gen,reuse,prefix"
)
# real_field TAG N: field N (1-4) of TAG's line of the table (empty if TAG is not there).
real_field() {
    local line f
    for line in "${REAL_MODELS[@]}"; do
        read -r -a f <<< "$line"
        if [ "${f[0]}" = "$1" ]; then echo "${f[$(($2 - 1))]}"; return 0; fi
    done
}
real_model_var() { real_field "$1" 2; }
# real_resolve LIST: the tags of LIST (comma-separated; all = every one), one per line, in the order asked and without
# repeats; an unknown tag fails naming the valid ones (status 1, message on stderr). an external CI wrapper also uses
# it (with real_model_var) to know which CI_MODEL_* folders to download, so it needs nothing else of the CI.
real_resolve() {
    local valid="" line t out=" " bad=""
    for line in "${REAL_MODELS[@]}"; do valid="$valid ${line%% *}"; done
    for t in ${1//,/ }; do
        if [ "$t" = all ]; then
            for line in $valid; do case "$out" in *" $line "*) ;; *) out="$out$line " ;; esac; done
        else
            case " $valid " in
                *" $t "*) case "$out" in *" $t "*) ;; *) out="$out$t " ;; esac ;;
                *) bad="$bad $t" ;;
            esac
        fi
    done
    if [ -n "$bad" ]; then echo "unknown real model(s):$bad (valid:$valid; all = every one)" >&2; return 1; fi
    if [ "$out" = " " ]; then echo "no real models given (valid:$valid; all = every one)" >&2; return 1; fi
    for t in $out; do echo "$t"; done
}

# Exactness with a real model: with and without drafter the same tokens (--check; without a drafter, normal against
# chained decoding and the normal one against mlx-lm's greedy in ref/out/gen-<x> for prompts-<x>, gen_ref.py: the real
# model's fidelity), equal to the reference ones of that model on this Mac (ref/golden, --golden), reusable state and
# snapshots (reuse) and prefix cache (prefixcache). --golden-strict: a missing golden or mlx-lm reference fails (it used
# to be skipped silently). --pdir: the PROMPTS column is what gen uses, so it also names the mlx-lm reference
# (prompts-<x> -> gen-<x>). exactness TAG MODEL PROMPTS STEPS VAR (STEPS space-separated).
# A missing model folder fails the CI (a regression skipped with a warning looked green); an external CI wrapper
# downloads the folders it is asked for before the GPU part and deletes them afterwards.
exactness() {
    local tag="$1" var="$5" model="$2" pdir="$3" what=" $4 " n=128 nocold=""
    if [ "$VALIDATE" = 1 ]; then n=16; nocold=--no-cold; fi
    [ -d "$model" ] || fail "$var=$model does not exist (real model of the $tag regression). Download or convert it (see\
 ref/golden/README.md) into ~/Models/$(basename "$model"), or point $var at it."
    # Without the prompts it is not skipped: it fails (a regression skipped silently looks green).
    [ -d "ref/out/$pdir" ] || fail "missing ref/out/$pdir (prompts of $tag)"
    local golden
    golden="ref/golden/$(basename "$model").json"
    case "$what" in *" gen "*) run "gen-check-$tag" "$LAB" gen --model "$model" --prompts 512,4K --n "$n" --check \
        --pdir "$pdir" --golden "$golden" --golden-strict ;; esac
    # The KV that grows on demand (the one of lab serve; KVPool).
    case "$what" in *" kvgrow "*) run "gen-check-$tag-kvgrow" "$LAB" gen --model "$model" --prompts 512,4K --n "$n" \
        --check --pdir "$pdir" --golden "$golden" --golden-strict --kv-pool 1 ;; esac
    case "$what" in *" reuse "*) run "reuse-$tag" "$LAB" reuse --model "$model" ;; esac
    # Step 5 of prefixcache (returning to a conversation) is strict against the reference split at the same cut.
    # With the no-cache turns of step 1 (they only give timings; without them with --validate). With lab serve's
    # defaults, what pi runs: the lossy prefill (W8A8 and Q·Kᵀ int8) on the MoE models, with the family's
    # int8 experts (Flash-Next's Q2 ones; the rest exact); the exact prefill on the dense ones. The exact prefill of
    # the MoE models is covered by the toys.
    # shellcheck disable=SC2086
    case "$what" in *" prefix "*) run "prefixcache-$tag" "$LAB" prefixcache --model "$model" $nocold ;; esac
}
# ci_real TAG...: every step of each tag, in the order given.
ci_real() {
    [ -d ref/out/prompts-es ] || fail "missing ref/out/prompts* (ref/prompts.py and ref/prompts_es.py, or links to those of the main repo)"
    local tag var
    for tag in "$@"; do
        var=$(real_model_var "$tag")
        exactness "$tag" "${!var}" "$(real_field "$tag" 3)" \
            "$(real_field "$tag" 4 | tr , ' ')" "$var"
        # The Heretic also checks the choice of rows of the round (only there: the copy with a drafter on a MoE).
        if [ "$tag" = moe ]; then
            run rowscheck "$LAB" rowscheck --model "${!var}" --pdir prompts-es --prompts es-chat --reps 2
        fi
    done
}
# --validate, after ci_real with the debug layer: gen --check of each tag with MTL_SHADER_VALIDATION=1 (set by ci.sh).
ci_real_validate() {
    local tag var
    for tag in "$@"; do
        var=$(real_model_var "$tag")
        run "sv-gen-check-$tag" "$LAB" gen --model "${!var}" --prompts 512 --n 16 --check \
            --pdir "$(real_field "$tag" 3)"
    done
}
