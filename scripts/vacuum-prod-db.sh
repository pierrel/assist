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

if [ ! -f "$DB" ]; then
    echo "[vacuum] $DB does not exist; nothing to do" >&2
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
