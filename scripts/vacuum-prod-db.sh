#!/usr/bin/env bash
# Stop assist-web, run sqlite3 VACUUM on threads.db, restart.
#
# Designed to run from a systemd oneshot service (ExecStart) on a
# daily timer.  Also runnable on-demand via `make vacuum-now`.
#
# Required env (set by the systemd unit or the Makefile target):
#   ASSIST_THREADS_DB  — absolute path to threads.db
#   ASSIST_SERVICE     — systemd unit name to stop/start (assist-web)
#   ASSIST_DB_OWNER    — POSIX user that owns the DB file (e.g. pierre);
#                        VACUUM rewrites the file, so we run sqlite3 as
#                        this user via runuser to preserve ownership.

set -euo pipefail

DB="${ASSIST_THREADS_DB:?ASSIST_THREADS_DB not set}"
SERVICE="${ASSIST_SERVICE:?ASSIST_SERVICE not set}"
OWNER="${ASSIST_DB_OWNER:?ASSIST_DB_OWNER not set}"
# /tmp/assist-eval.lock is the flock the nightly eval cron holds for
# the duration of its 6h budget.  If it's held when we wake up, the
# eval is still running — skip rather than racing it.  Override the
# path via env to disable the check (e.g. ASSIST_EVAL_LOCK=/dev/null
# in unit tests).
EVAL_LOCK="${ASSIST_EVAL_LOCK:-/tmp/assist-eval.lock}"

if [ ! -f "$DB" ]; then
    echo "[vacuum] $DB does not exist; nothing to do" >&2
    exit 0
fi

# Bail if the nightly eval is still holding its flock.  Without -n
# we'd block until eval finished, then run VACUUM at an unpredictable
# hour — better to skip and let next week's scheduled run catch up.
if [ -e "$EVAL_LOCK" ] && ! flock -n -x "$EVAL_LOCK" -c true 2>/dev/null; then
    echo "[vacuum] $(date -Is) skipping: eval cron is still holding $EVAL_LOCK" >&2
    exit 0
fi

size_before=$(stat -c %s "$DB")
echo "[vacuum] $(date -Is) starting; db=$DB size=${size_before}B service=$SERVICE owner=$OWNER"

# Stop the service so VACUUM can take its exclusive lock.  --quiet
# avoids polluting journal with redundant status output.
systemctl stop "$SERVICE"

# Run VACUUM as the DB owner so the rebuilt file keeps its ownership.
# trap restores the service even if VACUUM fails (disk full, lock,
# corruption) — leaving the service down would be worse than a stale
# DB.
trap 'systemctl start "$SERVICE"' EXIT

start=$(date +%s)
runuser -u "$OWNER" -- sqlite3 "$DB" 'VACUUM;'
elapsed=$(( $(date +%s) - start ))

size_after=$(stat -c %s "$DB")
reclaimed=$(( size_before - size_after ))

echo "[vacuum] $(date -Is) done in ${elapsed}s; size=${size_after}B reclaimed=${reclaimed}B"
