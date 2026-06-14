#!/usr/bin/env bash
#
# Run the eval suite one TEST per process, with an OS-level SIGKILL cap so
# a runaway test cannot pin llama's single inference slot and starve the
# rest of the suite.  See docs/2026-06-14-eval-per-test-sigkill.org.
#
# Why per-test + SIGKILL (not the old per-file + pytest-timeout):
# A timed-out eval leaves agent worker THREADS blocked in C-level socket
# reads to llama.cpp's `--parallel 1` slot.  pytest's in-process timeout
# (and SIGTERM) can't unwind a thread parked in a syscall — Python only
# services signals at bytecode boundaries, which a blocked-in-C thread
# never reaches.  Only SIGKILL frees the slot.  `timeout --kill-after`
# sends SIGTERM at the deadline, then SIGKILL after a grace window; the
# SIGKILL kills the pytest process and all its threads, freeing the slot
# for the next test.  Per-test (not per-file) bounds the blast radius to
# one test — the observed pain is a hung test starving its file-mates.
#
# Each test gets:
#   - cap: 1200s (outer `timeout`, SIGTERM then SIGKILL after KILL_GRACE)
#   - own JUnit XML at edd/history/<sanitized-nodeid>-<ts>.xml
# No `pytest --timeout`: the in-process timeout can't unblock those
# threads and would return rc=1, masking the timeout.  rc>=124 from
# `timeout` (124=TERM honored, 137=SIGKILL fired) is the timeout signal.
#
# A summary line lands in edd/history/eval-summary-<ts>.txt as each test
# completes, so partial progress is visible during long runs.
set -u

PYTEST="${PYTEST:-.venv/bin/pytest}"
HISTORY_DIR="edd/history"
PER_TEST_TIMEOUT="${PER_TEST_TIMEOUT:-1200}"
KILL_GRACE="${KILL_GRACE:-30s}"
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
echo "  per-test timeout: ${PER_TEST_TIMEOUT}s (SIGTERM, then SIGKILL after ${KILL_GRACE})" | tee -a "$SUMMARY"
echo "  TMPDIR: $TMPDIR (wiped, $(df -h "$TMPDIR" | awk 'NR==2 {print $4 " free"}'))" | tee -a "$SUMMARY"

# Collect test nodeids once (one import of the eval modules; cheap).
mapfile -t NODEIDS < <("$PYTEST" --collect-only -q edd/eval/ 2>/dev/null | grep '::')
if [ "${#NODEIDS[@]}" -eq 0 ]; then
    echo "ERROR: collected 0 eval tests — refusing to run" | tee -a "$SUMMARY" >&2
    exit 4
fi
echo "  collected ${#NODEIDS[@]} tests" | tee -a "$SUMMARY"

for nodeid in "${NODEIDS[@]}"; do
    # Sanitize the nodeid into an XML filename whose prefix matches
    # manage/eval_history.py's _RUN_ID_RE = ^(.+?)-(\d{8}-\d{4})\.xml$
    # (the live /evals page parses it).  Same scheme as the cassette
    # conftest: '::' -> '__', '/' -> '_'.  Test ids have no hyphens, so
    # the only '-<date>' is the shared TS.
    safe="${nodeid//::/__}"
    safe="${safe//\//_}"
    xml="$HISTORY_DIR/${safe}-${TS}.xml"
    log="$HISTORY_DIR/${safe}-${TS}.log"

    echo "===> $nodeid" | tee -a "$SUMMARY"
    start=$(date +%s)
    # OS-level cap: SIGTERM at the deadline, SIGKILL ${KILL_GRACE} later.
    # The SIGKILL is load-bearing — a test whose worker threads are
    # blocked in a C socket read to llama's slot can't service SIGTERM, so
    # only SIGKILL frees the slot for the next test.
    timeout --kill-after="$KILL_GRACE" -s TERM "$PER_TEST_TIMEOUT" \
        "$PYTEST" --junit-xml="$xml" "$nodeid" \
        > "$log" 2>&1
    rc=$?
    end=$(date +%s)
    wall=$((end - start))

    if [ "$rc" -ge 124 ]; then
        status="TIMED-OUT(rc=$rc)"      # 124=SIGTERM honored, 137=SIGKILL fired
    elif [ -s "$xml" ]; then
        status="xml-ok"
    else
        status="NO-XML"
    fi
    echo "<==  $nodeid : ${wall}s rc=$rc $status" | tee -a "$SUMMARY"
done

echo "=== eval suite finished at $(date -Iseconds) ===" | tee -a "$SUMMARY"
