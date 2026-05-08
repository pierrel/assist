#!/usr/bin/env bash
#
# Run the eval suite per-file so one runaway test cannot kill the rest.
#
# Background:
# `pytest --timeout=600 --timeout-method=thread` reliably caps a runaway
# test on Qwen3.6 + llama.cpp (signal method does not).  But the
# KeyboardInterrupt the thread method raises propagates into langgraph's
# `concurrent.futures.wait`, which prevents pytest from running its
# session_finish hook — so no JUnit XML is written for the whole pytest
# session.  Running pytest one file at a time bounds the blast radius:
# we lose at most one file's XML, not the entire night.
#
# Each file gets:
#   - per-test cap: 600s (pytest-timeout, thread method)
#   - per-file cap: 1800s (outer `timeout`)
#   - own JUnit XML at edd/history/<base>-<ts>.xml
#
# A summary line lands in edd/history/eval-summary-<ts>.txt as each file
# completes, so partial progress is visible during long runs.
set -u

PYTEST="${PYTEST:-.venv/bin/pytest}"
HISTORY_DIR="edd/history"
PER_TEST_TIMEOUT="${PER_TEST_TIMEOUT:-600}"
PER_FILE_TIMEOUT="${PER_FILE_TIMEOUT:-1800}"
TS="$(date +%Y%m%d-%H%M)"

# All eval-time tempfile activity (test workspaces, langgraph SqliteSaver
# threads.db, sandbox bind mounts) goes to a dedicated scratch dir that
# is wiped at the START of every run. Reproducible state, bounded disk
# use, isolated from prod's $ASSIST_THREADS_DIR/threads.db (which has
# its own growth issue tracked separately in roadmap).
#
# Default location: $HOME/deploy/assist/tmp/eval/
# Override via EVAL_SCRATCH_DIR for testing the script itself.
EVAL_SCRATCH_DIR="${EVAL_SCRATCH_DIR:-$HOME/deploy/assist/tmp/eval}"
EVAL_SCRATCH_LOCK="$HOME/deploy/assist/tmp/eval.lock"

# Path-validation guard: must resolve to exactly $HOME/deploy/assist/tmp/eval.
# Anchored to $HOME (not just a tail-match) so a hostile EVAL_SCRATCH_DIR
# override pointing at e.g. /some/other/user/deploy/assist/tmp/eval cannot
# bypass the guard. Uses GNU realpath -m (paths needn't exist yet); this
# script targets the Linux cron host.
EVAL_SCRATCH_REAL="$(realpath -m "$EVAL_SCRATCH_DIR")"
EXPECTED_REAL="$(realpath -m "$HOME/deploy/assist/tmp/eval")"
if [ "$EVAL_SCRATCH_REAL" != "$EXPECTED_REAL" ]; then
    echo "ERROR: EVAL_SCRATCH_DIR realpath ($EVAL_SCRATCH_REAL) is not the canonical $EXPECTED_REAL — refusing to wipe" >&2
    exit 2
fi

# Single-writer lock — a second concurrent eval run sharing the wipe-dir
# would corrupt state. Lock auto-releases when this script exits.
mkdir -p "$(dirname "$EVAL_SCRATCH_LOCK")"
exec 9>"$EVAL_SCRATCH_LOCK"
if ! flock -n 9; then
    echo "ERROR: another eval run holds $EVAL_SCRATCH_LOCK — exiting" >&2
    exit 3
fi

# Wipe + recreate. Only runs after the realpath guard above.
rm -rf "$EVAL_SCRATCH_DIR"
mkdir -p "$EVAL_SCRATCH_DIR"
export TMPDIR="$EVAL_SCRATCH_DIR"

mkdir -p "$HISTORY_DIR"
SUMMARY="$HISTORY_DIR/eval-summary-$TS.txt"

echo "=== eval suite starting at $(date -Iseconds) ===" | tee -a "$SUMMARY"
echo "  per-test timeout: ${PER_TEST_TIMEOUT}s, per-file timeout: ${PER_FILE_TIMEOUT}s" | tee -a "$SUMMARY"
echo "  TMPDIR: $TMPDIR (wiped, $(df -h "$TMPDIR" | awk 'NR==2 {print $4 " free"}'))" | tee -a "$SUMMARY"

for f in edd/eval/test_*.py; do
    base="$(basename "$f" .py)"
    xml="$HISTORY_DIR/${base}-${TS}.xml"
    log="$HISTORY_DIR/${base}-${TS}.log"

    echo "===> $base" | tee -a "$SUMMARY"
    start=$(date +%s)
    timeout "$PER_FILE_TIMEOUT" "$PYTEST" \
        --timeout="$PER_TEST_TIMEOUT" \
        --timeout-method=thread \
        --junit-xml="$xml" \
        "$f" \
        > "$log" 2>&1
    rc=$?
    end=$(date +%s)
    wall=$((end - start))

    if [ -s "$xml" ]; then
        xml_status="xml-ok"
    else
        xml_status="NO-XML"
    fi
    echo "<==  $base : ${wall}s rc=$rc $xml_status" | tee -a "$SUMMARY"
done

echo "=== eval suite finished at $(date -Iseconds) ===" | tee -a "$SUMMARY"
