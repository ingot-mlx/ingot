# Fidelity of the toys against mlx-lm, in --toy's fid lane (loaded with `source` from scripts/ci.sh; it came
# from the old --full).

# CI_REF_PY: Python with mlx-lm (required with --toy; default ref/.venv/bin/python, the one docs/development.md sets up).
ref_py_check() {
    REF_PY="${CI_REF_PY:-$PWD/ref/.venv/bin/python}"
    # shellcheck disable=SC2015  # `A && B || fail`: fail on either
    [ -n "$REF_PY" ] && "$REF_PY" -c 'import mlx_lm' 2>/dev/null \
        || fail "--toy: missing the Python with mlx-lm (CI_REF_PY; see README, ref/.venv)"
}

# Fidelity of the toys against mlx-lm (the project's rule for a new model against its external reference): mlx-lm's
# greedy (ref/gen_ref.py), its logits at those positions with fp32 activations (ref/logits_ref.py --f32) and
# ingot-dev kl --ref mlx-f32 with 64 forced positions. It passes with mean KL < 0.01. Against mlx-lm bf16 the toys do
# not pass (the bf16 reference itself is noisy with random weights). The toy MoEs used to need an exception because of
# their random router (clustered router logits: a bf16 rounding changed the top-8); with the router gain set to 4 in
# toy-qwen36-moe (ToySpec.routerGain) it passes the normal limit and the exception was removed.
# Flash-Next does not: mlx-lm has no qwen4_exp (its reference is ref/flashnext/harness.py, with mlx-vlm, by hand).
# The mlx-lm reference (greedy ids and fp32 logits of 512 and 4K) depends only on its inputs, so it is kept in
# $TOYDIR/fidref/TAG-KEY (its 36 Python runs were ~59 s of the fid lane in the CI, 28 s alone) and only the
# engine side, ingot-dev kl, runs every time. KEY hashes everything the reference reads: the toy (its .toy-stamp,
# config.json and the size of its safetensors), the toy prompts 512 and 4K, every ref/*.py (gen_ref.py,
# logits_ref.py, the wrappers and what they import), the mlx and mlx-lm versions of CI_REF_PY, WRAPS and N. Any change
# recomputes it (36/36 files byte-identical to the cached ones when recomputed); CI_FID_REF_CACHE=0
# recomputes it always and refreshes the cache. Not the mtime of the safetensors: the toys are a function of the
# generator that .toy-stamp hashes (ingot-dev toymodels --force twice gives the same bytes), and when two
# branches with different generators take turns in the CI each one regenerates the shared toys, which with the mtime
# in the key recomputed every reference each time (~40 s more in the fid lane). For the same reason a tag keeps
# its FID_REF_KEEP most recently used entries (3), not only the last one: branches with different ref/*.py also take
# turns.
# fid_ref_key MODEL WRAPS N (in a subshell: FID_PY_VERSIONS is read before, in fidelity).
fid_ref_key() {
    local name="$1" wraps="$2" n="$3" d="$TOYDIR/$1"
    {
        echo "$name|$wraps|$n|$FID_PY_VERSIONS"
        cat "$d/.toy-stamp" "$d/config.json"
        stat -f '%N %z' "$d"/*.safetensors
        cat ref/out/prompts-toy/512.json ref/out/prompts-toy/4K.json ref/*.py
    } | shasum -a 256 | cut -c1-16
}
# fid_ref_store TAG OUT CACHE: the reference just computed into the cache, written aside and renamed (a CI that
# stops halfway leaves no entry that looks complete); of the entries of TAG, the FID_REF_KEEP most recently used stay
# (a cache hit touches its entry).
fid_ref_store() {
    local tag="$1" out="$2" cache="$3" tmp p
    mkdir -p "$TOYDIR/fidref"
    tmp=$(mktemp -d "$TOYDIR/fidref/.tmp-$tag-XXXXXX")
    for p in 512 4K; do cp "ref/out/$out/${p}_mlx.json" "ref/out/$out/${p}_logits_f32.npy" "$tmp/"; done
    touch "$tmp/complete"
    rm -rf "$cache"
    mv "$tmp" "$cache"
    touch "$cache"
    # shellcheck disable=SC2012
    ls -dt "$TOYDIR/fidref/$tag"-* | tail -n +$((FID_REF_KEEP + 1)) | while read -r p; do rm -rf "$p"; done
}
FID_REF_KEEP=3
# fidelity TAG MODEL [WRAPS [LIMIT [N]]]: WRAPS is a space-separated chain of mlx-lm patches, ref/W_ref.py, that the
# reference scripts run through, outermost first (yarn: YaRN with `truncate: false`, which mlx-lm's YarnRoPE does not
# read, for gpt-oss, ref/yarn_ref.py; kv8: the engine's int8 KV cache emulated, ref/kv8_ref.py; lfm3: the lfm3 model
# of mlx-lm PR #2016, which mlx-lm 0.32.2 lacks, ref/lfm3_ref.py). LIMIT is the bound on the mean KL (0.01 by default), N the forced positions (64 by default).
fidelity() {
    local tag="$1" name="$2" wraps="${3:-}" limit="${4:-0.01}" n="${5:-64}" out="ci-fid-$1" p kl w
    local pre=() gen lgt
    for w in $wraps; do
        if [ ${#pre[@]} = 0 ]; then pre=(ref/"$w"_ref.py); else pre+=("$w"_ref.py); fi
    done
    if [ ${#pre[@]} = 0 ]; then gen=(ref/gen_ref.py); lgt=(ref/logits_ref.py)
    else gen=("${pre[@]}" gen_ref.py); lgt=("${pre[@]}" logits_ref.py); fi
    local key cache
    [ -n "${FID_PY_VERSIONS:-}" ] || FID_PY_VERSIONS=$("$REF_PY" -c \
        'import importlib.metadata as m; print(m.version("mlx"), m.version("mlx-lm"))') \
        || fail "fidelity: cannot read the mlx and mlx-lm versions of $REF_PY"
    key=$(fid_ref_key "$name" "$wraps" "$n") || fail "fidelity of $tag: cannot hash the inputs of its mlx-lm reference"
    cache="$TOYDIR/fidref/$tag-$key"
    if [ "${CI_FID_REF_CACHE:-1}" = 1 ] && [ -f "$cache/complete" ]; then
        step "fid-$tag-ref (mlx-lm reference from the cache)"
        touch "$cache"
        mkdir -p "ref/out/$out"
        for p in 512 4K; do cp "$cache/${p}_mlx.json" "$cache/${p}_logits_f32.npy" "ref/out/$out/"; done
        echo "mlx-lm greedy and fp32 logits of 512 and 4K from $cache"
    else
        for p in 512 4K; do
            run "fid-$tag-$p-mlx" "$REF_PY" "${gen[@]}" --model "$TOYDIR/$name" --pdir prompts-toy --out "$out" \
                --prompt "$p" --n "$n"
            run "fid-$tag-$p-logits" "$REF_PY" "${lgt[@]}" --f32 --model "$TOYDIR/$name" --pdir prompts-toy \
                --gen "$out" --prompt "$p" --n "$n"
        done
        fid_ref_store "$tag" "$out" "$cache"
    fi
    run "fid-$tag-kl" "$DEV" kl --ref mlx-f32 --model "$TOYDIR/$name" --prompts 512,4K --n "$n" --gen "$out"
    for p in 512 4K; do
        kl=$(sed -n "s/^$p (.*mean over [0-9]* positions \([0-9.]*\),.*/\1/p" "$LOG_DIR/fid-$tag-kl.log")
        [ -n "$kl" ] || fail "fidelity of $tag: cannot find the mean KL of $p in $LOG_DIR/fid-$tag-kl.log"
        awk -v k="$kl" -v l="$limit" 'BEGIN { exit !(k < l) }' \
            || fail "fidelity of $tag against mlx-lm: mean KL $kl with $p (limit $limit; see $LOG_DIR/fid-$tag-kl.log)"
        echo "fidelity of $tag with $p: mean KL $kl (< $limit)"
    done
}
# toy-gemma4's own limit on the mean KL: its random MoE (128 experts, top-8) with PLE makes mlx-lm itself noisy. A run
# (n 64, the same as here): engine 0.0178 / 0.0090 at 512 / 4K, and mlx-lm bf16 against fp32 already 0.0193 / 0.0110
# (ref/kl_npy.py); the engine stays below that bf16 floor. The limit is ~1.5x the floor at 512: a broken piece shows
# up far above it (toy-gemma4 v1 and v2, without the MoE, gave 0.0005-0.0018).
FID_GEMMA4_LIMIT=0.03
# toy-glm's own limit on the mean KL: the latent int8 KV (one 576 head, K = V = [k_pe | c]) moves this
# chaotic toy by a whole int8-sized step. A run (n 64): engine 0.0090 / 0.0176 at 512 / 4K (n 256: 0.0110 / 0.0180),
# and mlx-lm fp32 with the engine's latent KV emulated (ref/kv8_ref.py) already 0.0080 / 0.0171 against plain fp32,
# while with K and V in fp16 the same emulation falls to 0.0007 / 0.0003: the KV format explains the whole loss.
# The real GLM-4.7-Flash pays ~0.003 for it, under the normal limit. The limit is ~1.7x the worst measured
# mean (4K): it catches a broken piece that adds more than the KV format does, not a small one (the MLA pieces have
# their own exact checks: the mla-* kernel cases and the bit-exact golden). The kv8 line
# (engine against the emulated KV) is not used: its int8 roundings are not the engine's bit for bit and it gives
# 0.021 / 0.025, worse than against plain fp32.
FID_GLM_LIMIT=0.03
# toy-nemotron's own limit on the mean KL. With the 8-bit router (n 64) it gave 0.00313 / 0.00234
# at 512 / 4K, and the whole 512 error was that router (the last prompt position at 0.066). With the router in bf16, as
# the checkpoint ships it, the last prompt position falls to 0.00002 and n 256 drops from 0.0129 / 0.0078 to
# 0.0056 / 0.0003 (4K: top-1 257/257), but the n 64 mean at 512 is 0.0106: one near-tie of the random toy router
# (max 0.39, median 0.00001) moves it, as in the other toy MoEs. mlx-lm bf16 against fp32 gives 0.0170 / 0.0144 on the
# same positions; the limit sits ~1.2x above that floor at 512: a broken piece shows up far above it.
FID_NEMOTRON_LIMIT=0.02
ci_fidelity() {
    fidelity moe toy-qwen36-moe
    fidelity dense toy-qwen38-dense
    fidelity lfm toy-lfm2
    fidelity granite toy-granite
    # toy-gemma4 (no wrapper: mlx-lm's gemma4_text applies the softcap) with its own limit, FID_GEMMA4_LIMIT above.
    fidelity gemma4 toy-gemma4 "" "$FID_GEMMA4_LIMIT"
    # toy-gptoss with yarn_ref.py (mlx-lm rounds YaRN's correction range; gpt-oss's config says truncate false). A run
    # (n 64): 0.00006 / 0.00045 at 512 / 4K, under the normal limit.
    fidelity gptoss toy-gptoss yarn
    # toy-nemotron with its own limit, FID_NEMOTRON_LIMIT above.
    fidelity nemotron toy-nemotron "" "$FID_NEMOTRON_LIMIT"
    # toy-glm (no wrapper: mlx-lm's glm4_moe_lite) with its own limit, FID_GLM_LIMIT above.
    fidelity glm toy-glm "" "$FID_GLM_LIMIT"
    # toy-muse (no wrapper: mlx-lm's muse_glimmer applies the multiplier and the softcap) with the normal
    # limit: a run (n 64) gave 0.00396 / 0.00117 at 512 / 4K, with the toy's attention calmed (museQKPairs) so that
    # mlx-lm bf16 against fp32 stays at ~0.002 instead of 0.037.
    fidelity muse toy-muse
    # toy-lfm3 with lfm3_ref.py (mlx-lm 0.32.2 has no lfm3: lfm3.py of mlx-lm PR #2016 at 9027ac5) and the
    # normal limit. A run (n 64): 0.00026 / 0.00010 at 512 / 4K, top-1 differing only at mlx-lm margins < 0.06;
    # mlx-lm bf16 against fp32 0.0016 / 0.0011.
    fidelity lfm3 toy-lfm3 lfm3
}

# Attention-only toys against mlx-lm fp32 with the engine's int8 KV emulated (ref/kv8_ref.py): the check
# above (against plain mlx-lm fp32) measures fidelity to the real model, KV loss included; this one watches the engine
# without the KV noise. 256 forced positions so that the mean does not hang on 1-3 near-ties.
# NOT CALLED: the rule is a limit of 3x the worst mean KL and never above 0.003, and there is no such margin. With n =
# 256 the worst mean KL against this reference was 0.0034 with the unrotated KV (a tier-2 attention-only toy, 4K) and
# is 0.0029 with the Hadamard KV (another, 512; both removed), while the same toys against plain mlx-lm fp32
# give 0.0029-0.0080: near-ties of the trajectory dominate the mean. FID_KV8_LIMIT is the one value to move; the kv8 wrapper emulates the engine's KV by default (hadk).
FID_KV8_LIMIT=0.003
FID_KV8_N=256
ci_fidelity_kv8() {
    fidelity gptoss-kv8 toy-gptoss "kv8 yarn" "$FID_KV8_LIMIT" "$FID_KV8_N"
    fidelity granite-kv8 toy-granite kv8 "$FID_KV8_LIMIT" "$FID_KV8_N"
    fidelity gemma4-kv8 toy-gemma4 kv8 "$FID_KV8_LIMIT" "$FID_KV8_N"
}
