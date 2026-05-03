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

mkdir -p "$HISTORY_DIR"
SUMMARY="$HISTORY_DIR/eval-summary-$TS.txt"

echo "=== eval suite starting at $(date -Iseconds) ===" | tee -a "$SUMMARY"
echo "  per-test timeout: ${PER_TEST_TIMEOUT}s, per-file timeout: ${PER_FILE_TIMEOUT}s" | tee -a "$SUMMARY"

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
