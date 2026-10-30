# Exactness regression with the toy models (loaded with `source` from scripts/ci.sh).

# The KernelCase* suites (KernelCaseTests and KernelCaseFlashNextI8Tests, ~220 s and ~100 s: kernels of families
# that are not on this Mac, the W8A8 of lab serve against its emulation and per-row invariance) only if the branch
# touches kernels, layers or the tests relative to main (or with CI_KCASE=1; when in doubt, if there is no main, also):
# it is the most expensive part of the filter and does not change unless the kernels are touched.
kcase_needed() {
    [ "${CI_KCASE:-0}" = 1 ] && return 0
    local base
    base=$(git merge-base HEAD "${CI_BASE:-main}" 2>/dev/null) || return 0
    { git diff --name-only "$base" HEAD; git diff --name-only HEAD; } \
        | grep -qE '\.metal$|^Tests/MetalLabTests/|^Sources/MetalLab/(Layers/|Shaders/|Loader/|Flash|GPU\.swift)'
}
# ci_toys computes once into KCASE_RUN whether this run's swift test includes the KernelCase* suites.
lane_test() {
    if [ "${KCASE_RUN:-1}" = 0 ]; then
        run swift-test swift test --skip 'KernelCase[A-Za-z0-9]*Tests'
        echo "(without the KernelCase* suites: the branch touches no kernels, layers or tests relative to main; CI_KCASE=1 to run them)"
    else
        run swift-test swift test
    fi
    # Closed here so that the suite and case timings go after the step.
    step_close
    test_timings "$LOG_DIR/swift-test.log"
    return 0
}
# Exactness with the toy models: with and without drafter the same tokens, and tokens and logits fingerprints equal to
# those of ref/golden/toy-*.json (a missing golden or prompt fails; if a fingerprint does not match, gen prints
# max |Δlogit| and NMSE against the logits of the reference fingerprint, which it stores in $TOYDIR/logits when they
# match), reusable state and snapshots, prefix cache and prefill split at cuts aligned to 64 equal to the whole.
# toy TAG MODEL "GEN-ONLY OPTIONS" [OPTIONS FOR ALL]. TOY_REUSE_N (reply lengths, 32-128 by default) and TOY_COVER
# (in the call's environment): the rewind cases reuse must reach, N,N+1 by default (see "reuse" below for the toys that
# cannot reach N). TOY_GEN_ONLY=1 (in the call's environment) stops after gen, for a toy whose other steps are spread
# over other lanes (toy-lfm3).
toy() {
    local tag="$1" name="$2" genx="$3"; shift 3
    local model="$TOYDIR/$name" golden="ref/golden/$name.json" gw="--golden-strict"
    [ -f "$model/.toy-stamp" ] || fail "missing toy model $model"
    [ -f ref/out/prompts-toy/4K.json ] || fail "missing toy prompts (ref/out/prompts-toy)"
    if [ "${TOY_WRITE_GOLDEN:-0}" = 1 ]; then gw="--write-golden"; else
        [ -f "$golden" ] || fail "missing $golden (TOY_WRITE_GOLDEN=1 scripts/ci.sh --toy, only for deliberately changed tokens)"
    fi
    # shellcheck disable=SC2086  # $genx is a list of flags (or empty): it must split
    run "toy-gen-$tag" "$LAB" gen --model "$model" --prompts 512,4K --n 128 --nostop --check --golden "$golden" \
        --golden-logits $gw --logits-store "$TOYDIR/logits" $genx "$@"
    if [ "$tag" = flash ]; then
        # shellcheck disable=SC2086  # $genx is a list of flags (or empty): it must split
        run "toy-gen-$tag-kvgrow" "$LAB" gen --model "$model" --prompts 512,4K --n 128 --nostop --check \
            --golden "$golden" --golden-logits --golden-strict --logits-store "$TOYDIR/logits" --kv-pool 1 $genx "$@"
    fi
    [ "${TOY_GEN_ONLY:-0}" = 0 ] || return 0
    run "toy-reuse-$tag" "$LAB" reuse --model "$model" --n "${TOY_REUSE_N:-32-128}" --cover "${TOY_COVER:-N,N+1}" "$@"
    # Step 5 (returning to a conversation from a cut not aligned to 64) is informative: with the toy logits the greedy
    # diverges because of the rounding of the chunked GDN (see PrefixCheck.swift); the prefill split at aligned cuts is
    # checked bit by bit by splitcheck. --int8 0 --qk8 0: prefixcache takes lab serve's defaults (lossy prefill on MoE);
    # here the exact prefill, the int8 one is the line below.
    run "toy-prefixcache-$tag" "$LAB" prefixcache --model "$model" --no-cold --n 64 --int8 0 --qk8 0 "$@"
    run "toy-splitcheck-$tag" "$LAB" splitcheck --model "$model" "$@"
    # With the lossy prefill of lab serve (W8A8 and Q·Kᵀ int8): each row goes in int8 according to its position, so
    # the state at a cut does not depend on how the prompt is split. With and without prefix cache, and with or
    # without int8, the whole prompt goes through the prefill (Prompt.swift).
    run "toy-prefixcache-$tag-int8" "$LAB" prefixcache --model "$model" --no-cold --n 64 --int8 1 --qk8 1 "$@"
    run "toy-splitcheck-$tag-int8" "$LAB" splitcheck --model "$model" --int8 1 --qk8 1 "$@"
}
# Negative control (like sglang's test_self_e2e_perturb_*): reuse, prefixcache and splitcheck with --perturb (hidden
# lab-dev option: alters the restored recurrent state or the one between two prefill chunks; in rowscheck without a
# drafter, sends the rounds of k rows through the prefill) have to report FAILED. If they report OK, the test is not
# checking what it claims and the CI fails. With the toy MoE (GDN) and the shortest form of each test (~20 s; with
# LFM2.5's short convolution it also fails, checked by hand). CI_NEG_SELFTEST=1 launches them without perturbing: the CI
# has to fail.
negative() {
    local tag="$1" name="$2" test="$3"; shift 3
    local log="$LOG_DIR/neg-$test-$tag.log" pert="--perturb" rc=0
    [ "${CI_NEG_SELFTEST:-0}" = 1 ] && pert=""
    step "neg-$test-$tag (negative control: has to report FAILED)"
    "$LAB" "$test" --model "$TOYDIR/$name" $pert "$@" > "$log" 2>&1 || rc=$?
    if [ "$rc" = 1 ] && grep -q "^$test: FAILED" "$log"; then
        echo "$(grep -c '\[negative control\]' "$log" || true) perturbations; $(grep "^$test: FAILED" "$log") (as it should)"
    elif [ "$rc" = 0 ]; then
        tail -n 8 "$log"; fail "negative control: $test reports OK with the perturbed state: the test is not looking (see $log)"
    else
        tail -n 15 "$log"; fail "negative control: $test ends with $rc without the expected FAILED (see $log)"
    fi
}
# Why each step has its own --n and prompt (the toys are cheap per token; what is expensive is loading the model and
# the 4K prefills, so the trimming is done by steps and not by extra tokens):
#  - gen: --prompts 512,4K --n 128 are those of the golden (ref/golden/toy-*.json stores those 128 tokens and their
#    logits fingerprints): 512 fits in one prefill chunk and 4K crosses the cuts of 64 and the engine's chunks; with
#    another n the golden is not valid.
#  - reuse: --n 32-128 --cover N,N+1: most lengths end in the rewind N+1 case, so reuse probes the greedy replies
#    (~0.1-0.3 s each) from 32 up and checks the first length of each case; a case that no length reaches fails (a
#    fixed --n list silently loses a case when the toy's numbers change). The first N is at 32-44 in the toys with a
#    drafter. The N case needs rounds of more than one row: the toys without a drafter only reach N+1, and
#    toy-granite, whose copy rounds wait for the previous one, also "N without extra round".
#  - prefixcache: --no-cold drops the informative parts (no-cache turns of step 1, step 3 and the whole-prefill
#    comparison of step 5); --n 64 is the greedy of step 4: the restored state is already compared bit by bit in step 2
#    and the logits of the last prompt position in step 4, so 64 tokens cover the same as 128 with half the decode.
#  - splitcheck: default cuts 64, 1024, 2112, 3008 on the 4K prompt (aligned to 64, one prefill chunk and the split in
#    a single call): they are the engine's thresholds and are not removed.
#  - negative control: the shortest form that still touches the perturbed state (reuse --n 48, prefixcache --n 16,
#    splitcheck with one cut at 1024 and 1 step).
# Lanes: the same steps spread over five processes that share the GPU, evened out by the duration of each block
# (~215 s each in --toy, with the test lane's swift test as measured without the KernelCase* suites and without
# rebuilding):
#  - test: swift test (it also builds the tests while the others run), toy-gptoss and toy-muse;
#  - moe: toy-qwen36-moe (also with the SSD expert cache), toy-lfm2, toy-granite with batchcheck and the serve guard, batchcheck on toy-qwen38-dense,
#    the negative control and toy-lfm3's reuse;
#  - dense: toy-qwen38-dense, toy-gemma4 with its ring cuts, toy-muse's ring cuts and batchcheck --serve,
#    batchcheck --serve on the MoE and toy-lfm3's prefixcache;
#  - flash: toy-shapes, toy-flashnext, the drafter-less rowscheck of the toys that run without one, and toy-lfm3's gen
#    and splitcheck;
#  - fid: toy-glm, toy-nemotron, toy_rows and the fidelity against mlx-lm (scripts/ci/fidelity.sh: KL, not speed;
#    when its references are recomputed, ~40 s more, it is the longest).
# The test lane is not evened out with more toys: its swift test adds the KernelCase* suites in half of the
# integrations (8 of the last 16) and the debug rebuild after merging main.
# A sixth lane does not help: more lanes on one GPU only add contention.
lane_moe() {
    toy moe toy-qwen36-moe ""
    # The SSD expert cache that the README recommends for the Qwen3.6 MoE on 16 and 24 GB Macs (--expert-cache-gb):
    # 0.15 GB is 678 slots of 216 KiB, the first 256 for the prefill ring and 422 for ARC, for the 1024 experts of the
    # toy (4 layers x 256), so decode misses and evicts in every round (55-68 % hits) and the 4K prefill goes through
    # the ring. Its golden is the one without the cache: the same tokens and logits, bit for bit (~4 s alone).
    run toy-gen-moe-expertcache "$LAB" gen --model "$TOYDIR/toy-qwen36-moe" --prompts 512,4K --n 128 --nostop --check \
        --golden ref/golden/toy-qwen36-moe.json --golden-logits --golden-strict --expert-cache-gb 0.15
    # LFM2.5 (lfm2_moe): short convolution, attention with 64-wide heads, MoE of 64 experts top-4 sigmoid; no drafter
    # (gen compares the normal decode with the chained one).
    TOY_COVER=N+1 toy lfm toy-lfm2 ""
    # toy-granite (Granite 4.1 3B's config): granite family, attention only without QK-norm, GQA 40/8 (G = 5), q/k/v/o
    # biases (declared; the real one has none), dense SwiGLU, the μP multipliers (embedding × 12, residual × 0.22,
    # attention 1/64, logits ÷ 10) through the branch epilogue, embedding and logits hooks, and every 4-bit matrix in
    # groups of 32 (embed_rows, qmv, the prefill qmm and the --int8 W8A8 in g32, as mlx-community converts Granite 4.1);
    # no drafter. The six steps.
    TOY_COVER="N without extra round,N+1" toy granite toy-granite ""
    # Batched decoding (one round for N conversations, each in its own KV slot) gives each conversation the tokens and
    # logits it gives alone, bit for bit; N = 2 and 4 over the 512, 4K and chat prompts (~5 s), with G = 5.
    run toy-batchcheck-granite "$LAB" batchcheck --model "$TOYDIR/toy-granite" --rows 2,4 --n 24
    # The only step through lab serve's ChatServer and Session (batched path): with the agent guards on, a toy looping
    # on purpose is cut with finish_reason "length" and ids that are a prefix of the uncut ones (~4.6 s, 3.7 of them
    # starting lab serve).
    run toy-serve-guard python3 scripts/ci/serve_guard.py .build/release/lab "$TOYDIR/toy-granite" \
        "$LOG_DIR/toy-serve-guard-server.log"
    # The same with GDN layers (toy-qwen38-dense, GDN and attention): each conversation's recurrent state, conv history
    # and tape in its slot; with --seed, sampled, each conversation with its own seed, temperature, top-k/top-p/min-p
    # and penalties, the same ids as alone by the lab serve path (decodePlain).
    run toy-batchcheck-qwen38-dense "$LAB" batchcheck --model "$TOYDIR/toy-qwen38-dense" --rows 2,4 --n 24 --seed 1
    # (batchcheck --serve on the MoE is in the dense lane: toy_batch_serve_moe.)
    negative moe toy-qwen36-moe reuse --n 48
    negative moe toy-qwen36-moe prefixcache --n 16 --no-cold --int8 0 --qk8 0
    negative moe toy-qwen36-moe splitcheck --cuts 1024 --n 1
    toy_lfm3_reuse
}
lane_dense() {
    toy dense toy-qwen38-dense ""
    toy_gemma4
    toy_muse_ring
    toy_batch_serve_moe
    toy_lfm3_prefixcache
}
# lab serve's engine loop (InferenceEngine.serve) with slots served one at a time on the MoE with its toy drafter: 2 and
# 4 conversations of two turns queued at once (and 4 with 2 slots), greedy and sampled, each turn the same ids and
# rounds as alone and every turn 2 from its slot's live state (drafter ring per slot). The same on toy-muse, with each
# slot's drafter ring over a ring KV, is in toy_muse_ring.
toy_batch_serve_moe() {
    run toy-batchcheck-serve-moe "$LAB" batchcheck --serve --model "$TOYDIR/toy-qwen36-moe" --rows 2,4 --n 24 --seed 1 \
        --copy off
}
# toy-gemma4: gemma4_text from gemma-4-26B-A4B-it's config with every piece of the family: sliding (head 256, 8 kv, W
# 300: not a multiple of 128, the 512 and 4K prompts go past it; θ 1e4) and full layers (head 512, 2 kv, K = V,
# proportional RoPE θ 1e6), weightless v_norm, x·w norms with sandwich post-norms, GeGLU, embedding × √hidden, final
# softcap 30.5, per-layer inputs (PLE), the last two layers reading the KV of the first two and the double-wide MLP
# there, and the dense MLP in parallel with the real MoE (128 experts, top-8, router with its own norm,
# per_expert_scale), and a DFlash v1 drafter like z-lab's for the 26B-A4B (2 layers, sliding then full; no selector:
# argmax drafts, block 8 of the 16 its config says, block embedding × √hidden), the default of the 26B. gen compares
# normal and speculative decode against the golden; its random drafter guesses nothing (only the copy rounds accept),
# so reuse reaches "N without extra round" and N+1 but not N (toy-muse covers N with a drafter over a ring KV). The six
# steps: prefixcache adds entries at W − 1, W, W + 1 and ring ± 64 (its step 7), and splitcheck also cuts there
# (attention only: no 64 alignment needed).
toy_gemma4() {
    TOY_COVER="N without extra round,N+1" toy gemma4 toy-gemma4 ""
    run toy-splitcheck-gemma4-ring "$LAB" splitcheck --model "$TOYDIR/toy-gemma4" --cuts 299,300,301,2368,2496
}
# toy-muse: muse_glimmer from Muse Glimmer 30B's config, one period of the real layers (three sliding with RoPE θ 5e5 on
# the whole head and the real window of 2048, one global with NoPE), GQA 32/2 with 128-wide heads, the separate
# gate_proj, weightless QK-norm with qk_scale_factor, centred sandwich norms, the weightless norm after the embedding,
# output_multiplier then softcap, and its DFlash2 toy drafter of block 16 with the logits post-processing in the
# selector. It is the toy with a drafter over a ring KV (window 2048, ring 4224): gen compares normal and speculative
# decode against the golden, reuse reaches N and N+1 with the drafter (the N rewind over the ring KV), prefixcache adds
# entries at W − 1, W, W + 1 (its step 7; ring ± 64 does not fit the 4K prompt) and checks the drafter ring of each
# entry (step 6), and its conversations (up to ~4.6K tokens) wrap the ring. The six steps (test lane); toy_muse_ring
# (dense lane, to even out the lanes) adds splitcheck at the window cuts (2112, the ring's first chunk past W, is a
# default cut) and batchcheck --serve: lab serve's slots served one at a time with each slot's drafter ring over a ring
# KV, which toy-qwen36-moe's (full attention only) does not reach. Its rowscheck (drafter form) goes with the others in
# toy_rows.
toy_muse() {
    toy muse toy-muse ""
}
toy_muse_ring() {
    run toy-splitcheck-muse-ring "$LAB" splitcheck --model "$TOYDIR/toy-muse" --cuts 2047,2048,2049
    run toy-batchcheck-serve-muse "$LAB" batchcheck --serve --model "$TOYDIR/toy-muse" --rows 2,4 --n 24 --seed 1 \
        --copy off
}
# toy-gptoss: gpt_oss from gpt-oss-20b's config, with only the cuts: sliding (W 128) and full layers, learned sinks (the
# pattn_i8 prefill epilogue and vattn_reduce), YaRN with truncate false in every layer (per-layer RoPE table,
# LAB_AT_ROPETAB), q|k|v|o biases, no QK-norm, and the MoE with router and expert biases, swiglu_oai and MXFP4 experts
# (LAB_MXFP4). Its toy drafter is a DFlash v1 like z-lab's gpt-oss-20b-DFlash (MP-21 T1b: heads of 64 with q/k/v/o
# biases and YaRN, the `_h64` kernel instances and dbias_add, and the 4-bit copy of the toy's 8-bit lm_head for the
# drafts; 2 layers, sliding with the 2048 ring, which the 4K prompt wraps, then full with the 8192 one). z-lab's drafter
# is gpt-oss 20b's default (MP-21 T1d) and the toy runs with its own drafter/, so gen compares the normal decode with
# the speculative one against the golden (taken with the drafter), and toy-gen-gptoss-nodrafter (--drafter none, the
# other models of the family) the normal decode with the chained one (tokens only). Its random drafter guesses nothing
# and the toy's replies have no copy rounds, so reuse only reaches N+1 (as without a drafter, but with the drafter's
# ring and its rewind). The six steps: prefixcache adds entries at W − 1, W, W + 1 and
# ring ± 64 (its step 7; ring 2304, so 2240 and 2368), and splitcheck cuts there too (attention only: no 64 alignment
# needed) and at the drafter ring's 2047, 2048 and 2049.
toy_gptoss() {
    TOY_COVER=N+1 toy gptoss toy-gptoss ""
    run toy-gen-gptoss-nodrafter "$LAB" gen --model "$TOYDIR/toy-gptoss" --prompts 512,4K --n 128 --nostop --check \
        --golden ref/golden/toy-gptoss.json --golden-strict --drafter none
    run toy-splitcheck-gptoss-ring "$LAB" splitcheck --model "$TOYDIR/toy-gptoss" \
        --cuts 127,128,129,2047,2048,2049,2240,2368
}
# LFM2.5-8B-A1B has no toy (one per family: toy-lfm2): it is the real model "lfm" of scripts/ci/real.sh.
# All the toys (ToyArch, one shape each) loaded at once in one process, each with the tokens and the fingerprints of
# its golden at the 512 prompt (the shape goes through MetalContext: if one overwrote another, they would change).
toy_shapes() {
    [ "${TOY_WRITE_GOLDEN:-0}" != 1 ] || return 0
    run toy-shapes "$LAB" shapes --dir "$TOYDIR" --prompts 512
}
# Flash-Next: in gen, an expert cache of 0.3 GB (565 slots for 2048 experts of 272 KiB: misses in every round); in the
# rest, the default one (all fit after reading them once: faster). The PLE one, 1 MB in all.
lane_flash() {
    toy_shapes
    # Toy corpus: no reply of 32-128 tokens ends in the rewind N case, 8-31 does
    TOY_REUSE_N=8-128 toy flash toy-flashnext "--expert-cache-gb 0.3" --ple-cache-mb 1
    toy_rows_nodrafter
    toy_lfm3
}
# toy-lfm3: lfm3 from LFM3-1.7B's config with only the cuts: the real layers 1-4 (conv, full, conv, sliding), the 7-tap
# short convolution (LAB_SC_L 7: the generic conv kernels; toy-lfm2 keeps the 3-tap ones, which stay apart under #if
# LAB_SC_L == 3), sandwich post-norms around both mixers (the branch epilogue around the conv), attention 16/4 of 128
# with QK-norm and the scalar gate per head (repeated into the gated q rows at load), the zero-tail partial RoPE of the
# full layers (the proportional table) and θ 1e4 in the sliding one (window 256, ring 2432: the 512 and 4K prompts go
# past the window and the 4K one wraps the ring), dense w1/w3/w2 SwiGLU, every matrix 8-bit g64 like the real MLX-8bit,
# tied embedding. No drafter, so gen compares the normal decode with the chained one and reuse only reaches N+1. Four
# steps, no -int8 ones (dense: lab serve runs the exact prefill), spread over three lanes so that flash is not the
# longest: gen and splitcheck here, splitcheck with the default cuts and the window and ring ones in a single
# run (192, 256, 320, 2368, 2432, 2496, aligned to 64 for the conv snapshots); reuse in the moe lane and prefixcache in
# the dense one. Its rowscheck (toy_rows) and fidelity (ci_fidelity) are in the fid lane.
toy_lfm3() {
    TOY_GEN_ONLY=1 toy lfm3 toy-lfm3 ""
    run toy-splitcheck-lfm3 "$LAB" splitcheck --model "$TOYDIR/toy-lfm3" \
        --cuts 64,192,256,320,1024,2112,2368,2432,2496,3008
}
toy_lfm3_reuse() {
    run toy-reuse-lfm3 "$LAB" reuse --model "$TOYDIR/toy-lfm3" --n 32-128 --cover N+1
}
# prefixcache with --n 32 (64 in the other toys): the restored state is already compared bit by bit in its step 2 and
# the last prompt position's logits in step 4, and the 4K prompt crosses the window and wraps the ring before that.
toy_lfm3_prefixcache() {
    run toy-prefixcache-lfm3 "$LAB" prefixcache --model "$TOYDIR/toy-lfm3" --no-cold --n 32 --int8 0 --qk8 0
}
# toy-nemotron: nemotron_h from Nemotron 3.5 Lightning's config, with Mamba2 layers (64 heads × 64, groups of the GDN
# kernels), attention without RoPE (head 128, 32/2), and the MoE with relu² without gate, 128 experts top-6, a sigmoid
# router with selection bias and routed_scaling_factor 2.5, and the ungated shared expert of 2 × the intermediate size.
# No drafter (Nemotron has none in the engine), so gen compares the normal decode with the chained one and reuse only
# reaches N+1. The six steps.
toy_nemotron() {
    TOY_COVER=N+1 toy nemotron toy-nemotron ""
}
# toy-glm: glm4_moe_lite from GLM-4.7-Flash's config, with the latent MLA (q_a/q_b and kv_a projections, absorbed
# embed_q/unembed_out heads, one 576-wide KV head K = V = [k_pe | c] in the int8 cache, G = 20), the dense layer 0 and
# the MoE with a sigmoid router, selection bias, routed scaling 1.8 and an ungated shared expert. No drafter
# (GLM-4.7-Flash has none in the engine), so gen compares the normal decode with the chained one and reuse only reaches
# N+1. The six steps (fid lane).
toy_glm() {
    TOY_COVER=N+1 toy glm toy-glm ""
    # splitcheck of toy-glm runs on 10,240 tokens and crosses between the absorbed and the decompressed prefill paths,
    # which it compares with a tolerance (docs/adr/0002); its negative control (scaled latent KV) has to fail.
    negative glm toy-glm splitcheck --cuts 1024 --n 1
}
# rowscheck without a drafter: in each toy without a drafter, the same tokens in rounds of k = 1…16 rows through
# the verification path (the copy rounds and the short prompt tails) and in steps of one row (decodePlain) from the
# same prefill of the 512 prompt, logits and argmax bit by bit (it catches a kernel whose bits depend on the row's
# place in the round or on the number of rows, like a fused multiply-add or a 6-bit head kernel that changes with
# the rows). The toys with a drafter take rowscheck's drafter form only with the real models (scripts/ci/real.sh),
# except toy-muse (the one toy whose drafter runs over a ring KV; 48 cases: trimmed verification of k = 1…8 rows
# against the full round; it checks k ≤ 8 although Muse's block is 16) and toy-gptoss (its head-64 drafter with
# biases and YaRN over the SWA ring of 128; 48 cases; its drafter-less form is in toy_rows_nodrafter). ~3-4 s each
# alone.
# Negative control: --perturb sends the rounds through the prefill (another summation order), so it has to fail;
# with toy-granite, attention only (no recurrent state to perturb: it is the route that changes).
toy_rows() {
    local t
    for t in lfm2 lfm3 granite gptoss nemotron glm; do
        run "toy-rowscheck-$t" "$LAB" rowscheck --model "$TOYDIR/toy-$t" --prompts 512
    done
    run toy-rowscheck-muse "$LAB" rowscheck --model "$TOYDIR/toy-muse" --prompts 512
    negative granite toy-granite rowscheck --prompts 512
}
# The same drafter-less rowscheck, with --drafter none, for the families whose other models run without a drafter:
# toy-gptoss (gpt-oss other than the 20b, or the 20b with --drafter none),
# toy-gemma4 (Gemma 4 E2B, E4B and 31B), toy-qwen38-dense and toy-qwen36-moe (the small Qwen3.5 models, whose Qwen3.6
# drafter is discarded, and the 27B or the MoE served without one: the copy rounds of decodeSpec). ~3-7 s each alone (gemma4's window of
# 300 < 512 takes it over the ring). In the flash lane, to keep the fid lane from being the longest.
toy_rows_nodrafter() {
    run toy-rowscheck-gptoss-nodrafter "$LAB" rowscheck --model "$TOYDIR/toy-gptoss" --prompts 512 --drafter none
    run toy-rowscheck-gemma4 "$LAB" rowscheck --model "$TOYDIR/toy-gemma4" --prompts 512 --drafter none
    run toy-rowscheck-qwen38-dense "$LAB" rowscheck --model "$TOYDIR/toy-qwen38-dense" --prompts 512 --drafter none
    run toy-rowscheck-qwen36-moe "$LAB" rowscheck --model "$TOYDIR/toy-qwen36-moe" --prompts 512 --drafter none
}
lane_fid() {
    toy_glm
    toy_nemotron
    toy_rows
    ci_fidelity
}
# A swift test failure does not hide the toys of the lane: they still run and the lane fails afterwards.
lane_test_toys() {
    local rc=0
    ( lane_test ) || rc=$?
    toy_gptoss
    toy_muse
    step_close
    return "$rc"
}
ci_toys() {
    KCASE_RUN=0; kcase_needed && KCASE_RUN=1
    lane test lane_test_toys; lane moe lane_moe; lane dense lane_dense; lane flash lane_flash; lane fid lane_fid
    lanes_wait
}
