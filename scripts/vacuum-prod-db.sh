#!/usr/bin/env bash
# Stop assist-web, prune old threads, run sqlite3 VACUUM on threads.db, restart.
#
# Designed to run from the deploy user's crontab on a weekly schedule
# (see scripts/crontab.example for the line to add) and on demand
# via `make vacuum-now`.  Both paths pass paths in via env so this
# script holds no host-specific defaults.
#
# Required env (the Makefile passes these from .deploy.env; the cron
# line sets them inline):
#   ASSIST_THREADS_DIR  — directory containing threads.db
#   SERVICE_NAME        — systemd unit name to stop/start (e.g. assist-web)
#   DEPLOY_PATH         — root of the deploy tree (we use $DEPLOY_PATH/.venv/bin/python)
# Optional:
#   MIN_THREADS         — retention floor; defaults to 100 (set in
#                         scripts/crontab.example).  See
#                         docs/2026-05-04-threads-db-layer-0-thread-retention.org.
#
# This script runs as the invoking user.  threads.db is owned by
# that user, so sqlite3 needs no privilege escalation.  Service
# stop/start uses sudo; that pair is in the existing
# /etc/sudoers.d/assist-deploy passwordless allowlist.

set -euo pipefail

: "${ASSIST_THREADS_DIR:?ASSIST_THREADS_DIR must be set (the directory containing threads.db)}"
: "${SERVICE_NAME:?SERVICE_NAME must be set (the systemd unit, typically assist-web)}"
: "${DEPLOY_PATH:?DEPLOY_PATH must be set (root of the deploy tree)}"

DB="$ASSIST_THREADS_DIR/threads.db"
LOCK="$DB.lock"

if [ ! -f "$DB" ]; then
    echo "[vacuum] $DB does not exist; nothing to do" >&2
    exit 0
fi

# Wrap the whole sweep+VACUUM body in flock — belt to the cron-level
# /tmp/assist-eval.lock 's suspenders.  This lock is specifically for
# threads.db so `make vacuum-now` and the cron can't race each other
# even if the eval lock isn't held.
exec 9>"$LOCK"
if ! flock -n 9; then
    echo "[vacuum] another run holds $LOCK; bailing" >&2
    exit 0
fi

size_before=$(stat -c %s "$DB")
echo "[vacuum] $(date -Is) starting; db=$DB size=${size_before}B service=$SERVICE_NAME min_threads=${MIN_THREADS:-default}"

# Stop the service so the retention sweep + VACUUM can take exclusive
# access.
sudo systemctl stop "$SERVICE_NAME"

# Trap restores the service even if anything fails (sweep crashes,
# disk full, lock, corruption) — leaving the service down would be
# worse than a stale DB.
trap 'sudo systemctl start "$SERVICE_NAME"' EXIT

# Layer 0: prune the threads dir + DB to MIN_THREADS most-recent
# threads before VACUUM.  Without this, VACUUM just rewrites the
# same multi-hundred-GB DB.  See
# docs/2026-05-04-threads-db-layer-0-thread-retention.org.
echo "[vacuum] $(date -Is) running retention sweep"
ASSIST_THREADS_DIR="$ASSIST_THREADS_DIR" \
    MIN_THREADS="${MIN_THREADS:-100}" \
    "$DEPLOY_PATH/.venv/bin/python" -m assist.retention

start=$(date +%s)
# VACUUM writes a temp database equal in size to the original.  By
# default sqlite picks /tmp, which on most hosts is a small tmpfs —
# nowhere near enough for a hundreds-of-GB DB.  Point it at the same
# big filesystem the DB lives on.  Discovered the hard way with
# "Error: stepping, database or disk is full (13)" the first time
# this ran on prod.
SQLITE_TMPDIR="$ASSIST_THREADS_DIR" sqlite3 "$DB" 'VACUUM;'
elapsed=$(( $(date +%s) - start ))

size_after=$(stat -c %s "$DB")
reclaimed=$(( size_before - size_after ))

echo "[vacuum] $(date -Is) done in ${elapsed}s; size=${size_after}B reclaimed=${reclaimed}B"

# Mirror progress to syslog so cron output (which goes to mail/void)
# isn't the only record.
logger -t assist-vacuum \
    "vacuum done in ${elapsed}s; size=${size_after}B reclaimed=${reclaimed}B"
