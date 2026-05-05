#!/usr/bin/env bash
# Stop assist-web, run sqlite3 VACUUM on threads.db, restart.
#
# Designed to run from pierre's user crontab on a weekly schedule —
# add this line to `crontab -e`:
#
#   0 6 * * 0 /usr/bin/flock -n /tmp/assist-eval.lock /home/pierre/deploy/assist/code/scripts/vacuum-prod-db.sh
#
# Sunday 06:00 local clears the 23:00-cron eval window (which has a
# 360-min budget so it can run until ~05:02 worst case).  The cron-
# level flock against /tmp/assist-eval.lock guarantees we never run
# while the eval cron is alive — the eval cron uses the same lock.
#
# Also runnable on-demand via `make vacuum-now`.
#
# Required env (set by `make vacuum-now` or in the cron line):
#   ASSIST_THREADS_DB  — absolute path to threads.db (default below)
#   ASSIST_SERVICE     — systemd unit name to stop/start (default assist-web)
#
# This script runs as the invoking user (pierre).  The threads.db
# file is owned by pierre, so sqlite3 needs no privilege escalation.
# Service stop/start uses sudo; that pair is in the existing
# /etc/sudoers.d/assist-deploy passwordless allowlist.

set -euo pipefail

DB="${ASSIST_THREADS_DB:-/home/pierre/deploy/assist/threads/threads.db}"
SERVICE="${ASSIST_SERVICE:-assist-web}"

if [ ! -f "$DB" ]; then
    echo "[vacuum] $DB does not exist; nothing to do" >&2
    exit 0
fi

size_before=$(stat -c %s "$DB")
echo "[vacuum] $(date -Is) starting; db=$DB size=${size_before}B service=$SERVICE"

# Stop the service so VACUUM can take its exclusive lock.
sudo systemctl stop "$SERVICE"

# Trap restores the service even if VACUUM fails (disk full, lock,
# corruption) — leaving the service down would be worse than a stale
# DB.
trap 'sudo systemctl start "$SERVICE"' EXIT

start=$(date +%s)
# VACUUM writes a temp database equal in size to the original.  By
# default sqlite picks /tmp, which on this host is a 15 GB tmpfs —
# nowhere near enough for a 187 GB DB.  Point it at the same big
# filesystem the DB lives on.  Discovered the hard way with
# "Error: stepping, database or disk is full (13)" at 18:53 PDT
# 2026-05-04.
SQLITE_TMPDIR="$(dirname "$DB")" sqlite3 "$DB" 'VACUUM;'
elapsed=$(( $(date +%s) - start ))

size_after=$(stat -c %s "$DB")
reclaimed=$(( size_before - size_after ))

echo "[vacuum] $(date -Is) done in ${elapsed}s; size=${size_after}B reclaimed=${reclaimed}B"

# Mirror progress to syslog so cron output (which goes to mail/void)
# isn't the only record.
logger -t assist-vacuum \
    "vacuum done in ${elapsed}s; size=${size_after}B reclaimed=${reclaimed}B"
