# Common CI functions (loaded with `source` from scripts/ci.sh; not run on its own).
# Named steps with their duration in $TIMES, `run` (output to .build/ci/NAME.log) and the parallel lanes.

fail() { step_close; echo "FAILED: $*" >&2; printf 'FAILED\t%d\n' "$SECONDS" >> "$TIMES"; exit 1; }
step_close() {
    if [ -n "$STEP_NAME" ]; then printf '%s\t%d\n' "$STEP_NAME" $((SECONDS - STEP_T0)) >> "$TIMES"; fi
    STEP_NAME=""
}
step() { step_close; STEP_NAME="$*"; STEP_T0=$SECONDS; printf '\n== %s (%d s)\n' "$*" "$SECONDS"; }

# --- GPU steps ----------------------------------------------------------------------------------------------
# run NAME command...: full output in .build/ci/NAME.log; shows the tail and fails if the command fails.
run() {
    local name="$1"; shift
    step "$name"
    if "$@" > "$LOG_DIR/$name.log" 2>&1; then grep "Test run with" "$LOG_DIR/$name.log" || tail -n 4 "$LOG_DIR/$name.log"
    else tail -n 25 "$LOG_DIR/$name.log"; fail "$name (see $LOG_DIR/$name.log)"; fi
}

# test_timings LOG: after a test process, its per-suite and per-kernel-case timings into $TIMES, as
# "suite NAME<TAB>s" (from Swift Testing's "Suite NAME passed|failed after N seconds": the moment the suite ends with
# every suite running in parallel, not what it costs alone) and "kcase ID<TAB>s" (the "kcase-time ID s" that the
# KernelCase* harness prints: context creation plus run and checks of that case, which run serialized).
test_timings() {
    awk '$2 == "Suite" && $4 ~ /^(passed|failed)$/ && $5 == "after" && $7 ~ /^seconds/ { printf "suite %s\t%s\n", $3, $6 }
         $1 == "kcase-time" && NF == 3 { printf "kcase %s\t%s\n", $2, $3 }' "$1" >> "$TIMES"
}

# Lanes: in --toy the toys run in parallel lab-dev processes, all under the GPU already reserved by whoever launches
# the CI. It measures exactness (tokens, bit-exact logits), not speed, so sharing the GPU changes no result; the
# timings per step come out longer than when run alone, and what counts is that of each lane. Every lane writes its
# output to .build/ci/lane-NAME.out and its timings to lane-NAME.txt; ci.sh merges them when all have finished.
LANE_PIDS=(); LANE_NAMES=()
lane() {
    local name="$1"; shift
    # shellcheck disable=SC2030  # each lane has its own TIMES file, on purpose
    ( TIMES="$LOG_DIR/lane-$name.txt"; : > "$TIMES"; STEP_NAME=""; "$@"; step_close ) \
        > "$LOG_DIR/lane-$name.out" 2>&1 &
    LANE_PIDS+=("$!"); LANE_NAMES+=("$name")
}
lanes_wait() {
    local i bad=""
    for i in "${!LANE_PIDS[@]}"; do
        wait "${LANE_PIDS[$i]}" || bad="$bad ${LANE_NAMES[$i]}"
        printf '\n== lane %s\n' "${LANE_NAMES[$i]}"; cat "$LOG_DIR/lane-${LANE_NAMES[$i]}.out"
        # shellcheck disable=SC2031  # TIMES is the parent's (the lane subshell only set its own)
        grep -v '^#' "$LOG_DIR/lane-${LANE_NAMES[$i]}.txt" >> "$TIMES" || true
    done
    LANE_PIDS=(); LANE_NAMES=()
    [ -z "$bad" ] || fail "failing lanes:$bad (see $LOG_DIR/lane-*.out)"
}
