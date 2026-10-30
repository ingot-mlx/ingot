# No-GPU CI steps (loaded with `source` from scripts/ci.sh): API, build, SwiftLint, formatting and Periphery (the
# static part of --toy) and the release build alone (all --real needs).
# CI_GPU_WRAP (optional): command prefix for whatever uses the GPU even without GPU compute (`swift build`,
# `swift test`); empty if whoever launches it does not need to coordinate the GPU.

ci_static() {
step "engine API (no GPU: exp outside Sources/ and ref/api-surface.txt up to date)"
scripts/api_surface.py --exp-check || fail "Sources/ uses EngineOptions.exp (only valid on experiment branches)"
scripts/api_surface.py --check || fail "ref/api-surface.txt out of date: regenerate with scripts/api_surface.py and review the diff"

ci_build
if grep -q 'warning:' "$LOG_DIR/build.log"; then fail "the build gives warnings (see $LOG_DIR/build.log)"; fi

step "SwiftLint (--strict: fails on any warning)"
if command -v swiftlint >/dev/null; then
    # First line: the commit checked (and whether there were uncommitted changes), to know what the report is about.
    dirty=$(git diff --quiet HEAD -- Sources Tests .swiftlint.yml || echo ' + uncommitted changes')
    echo "commit $(git rev-parse --short HEAD)$dirty" > "$LOG_DIR/swiftlint.txt"
    set +e; swiftlint lint --strict --quiet >> "$LOG_DIR/swiftlint.txt" 2>&1; rc=$?; set -e
    echo "$(grep -c 'error:' "$LOG_DIR/swiftlint.txt" || true) warnings ($LOG_DIR/swiftlint.txt)"
    [ $rc = 0 ] || { grep 'error:' "$LOG_DIR/swiftlint.txt" | head -20; fail "SwiftLint"; }
else
    echo "swiftlint is not installed (brew install swiftlint): skipped"
fi

step "formatting (swift-format lint with .swift-format; fails on any difference)"
set +e; xcrun swift-format lint -p -r Sources Tests > "$LOG_DIR/swift-format.txt" 2>&1; set -e
nfmt=$(grep -c 'warning:' "$LOG_DIR/swift-format.txt" || true)
echo "$nfmt formatting differences ($LOG_DIR/swift-format.txt)"
if [ "$nfmt" != 0 ]; then head -20 "$LOG_DIR/swift-format.txt"; fail "formatting (xcrun swift-format format -i -r Sources Tests)"; fi

if [ "$DEADCODE" = 1 ]; then
    step "dead code (Periphery)"
    command -v periphery >/dev/null || fail "periphery is not installed (brew install --build-from-source periphery)"
    # Periphery 3.8 calls `swift build --build-tests false`, which SwiftPM 6.4 no longer accepts: it is built
    # separately with the index and passed with --skip-build (configuration in .periphery.yml).
    $CI_GPU_WRAP swift build --build-tests --enable-index-store --scratch-path .build/periphery > "$LOG_DIR/periphery-build.log" 2>&1 \
        || fail "build for Periphery"
    periphery scan --quiet --relative-results > "$LOG_DIR/periphery.txt" 2>&1 || true
    echo "$(grep -c 'warning:' "$LOG_DIR/periphery.txt" || true) warnings ($LOG_DIR/periphery.txt)"
fi
}

# The release build (lab and lab-dev); --toy also fails on its warnings (ci_static).
ci_build() {
    step "build (release)"
    $CI_GPU_WRAP swift build -c release > "$LOG_DIR/build.log" 2>&1 || { grep -E 'error:' "$LOG_DIR/build.log" | head -20; fail "swift build"; }
    tail -n 1 "$LOG_DIR/build.log"
}
