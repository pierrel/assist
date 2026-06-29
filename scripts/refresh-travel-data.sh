#!/usr/bin/env bash
#
# Refresh the data behind the `travel` tool and rebuild the engines:
#   - the 511 regional GTFS (transit schedules — change often, the main staleness)
#   - the NorCal OSM extract (roads/addresses — change slowly), only when it's stale
# Re-imports MOTIS (routing) every run; re-imports Nominatim (geocoding) ONLY when
# the OSM extract actually changed (its heavy import isn't worth running weekly).
#
# Designed for an unattended weekly cron: idempotent, holds a lock so runs can't
# overlap, and VALIDATES every download before swapping it in so a bad fetch never
# replaces good data. Uses only `docker` (no sudo). The 511 token is read from a
# file outside the repo and never printed.
#
# Config via env (defaults suit the standard single-box deploy):
#   TRAVEL_INFRA_DIR     REQUIRED — dir with input/ + the MOTIS config.yml + data/
#                        graph (no host-specific default; the cron passes it)
#   MOTIS_CONTAINER      default: motis-travel
#   MOTIS_IMAGE          default: ghcr.io/motis-project/motis:latest
#   NOMINATIM_CONTAINER  default: nominatim-geocoder
#   NOMINATIM_VOLUME     default: nominatim-geocoder-data
#   NOMINATIM_IMAGE      default: mediagis/nominatim:4.5
#   NOMINATIM_PORT       default: 8089
#   OSM_URL              default: Geofabrik NorCal extract
#   OSM_FILE             basename under input/ (default: norcal.osm.pbf)
#   OSM_MAX_AGE_DAYS     refresh OSM only if older than this (default: 30)
#   GTFS_FILE            basename under input/ (default: 511-regional-gtfs.zip)
#   GTFS_511_OPERATOR    511 operator id (default: RG = regional combined)
#   TOKEN_FILE           file containing `ASSIST_511_TOKEN=...` (default: $TRAVEL_INFRA_DIR/.511-token)
#
# Flags:
#   --check   download + validate into temp files only; do NOT swap, import, or restart
#
set -euo pipefail

# Required (no host-specific default) — the repo convention for ops scripts; the
# cron passes it explicitly. Holds input/, config.yml, and the MOTIS data/ graph.
: "${TRAVEL_INFRA_DIR:?set TRAVEL_INFRA_DIR to the travel infra dir (input/, config.yml, data/)}"
MOTIS_CONTAINER="${MOTIS_CONTAINER:-motis-travel}"
MOTIS_IMAGE="${MOTIS_IMAGE:-ghcr.io/motis-project/motis:latest}"
NOMINATIM_CONTAINER="${NOMINATIM_CONTAINER:-nominatim-geocoder}"
NOMINATIM_VOLUME="${NOMINATIM_VOLUME:-nominatim-geocoder-data}"
NOMINATIM_IMAGE="${NOMINATIM_IMAGE:-mediagis/nominatim:4.5}"
NOMINATIM_PORT="${NOMINATIM_PORT:-8089}"
OSM_URL="${OSM_URL:-https://download.geofabrik.de/north-america/us/california/norcal-latest.osm.pbf}"
OSM_FILE="${OSM_FILE:-norcal.osm.pbf}"
OSM_MAX_AGE_DAYS="${OSM_MAX_AGE_DAYS:-30}"
GTFS_FILE="${GTFS_FILE:-511-regional-gtfs.zip}"
GTFS_511_OPERATOR="${GTFS_511_OPERATOR:-RG}"
TOKEN_FILE="${TOKEN_FILE:-$TRAVEL_INFRA_DIR/.511-token}"

CHECK_ONLY=0
[ "${1:-}" = "--check" ] && CHECK_ONLY=1

INPUT_DIR="$TRAVEL_INFRA_DIR/input"
log() { echo "[$(date '+%Y-%m-%dT%H:%M:%S%z')] $*"; }
die() { log "ERROR: $*"; exit 1; }

[ -d "$INPUT_DIR" ] || die "input dir not found: $INPUT_DIR"

# Single-run lock so a slow run can't overlap the next cron tick.
exec 9>"$TRAVEL_INFRA_DIR/.refresh.lock"
flock -n 9 || die "another refresh is already running"

# Temp dir on the SAME filesystem as INPUT_DIR so the swaps below are atomic
# renames, not a cross-device copy+truncate that could leave a partial file.
tmpdir="$(mktemp -d "$TRAVEL_INFRA_DIR/.refresh-tmp.XXXXXX")"
trap 'rm -rf "$tmpdir"' EXIT

# --- 511 GTFS (always) ---------------------------------------------------------
refresh_gtfs() {
    [ -f "$TOKEN_FILE" ] || { log "no token file ($TOKEN_FILE) — skipping GTFS refresh"; return 1; }
    # Parse the assignment rather than sourcing the file (don't execute config).
    local token; token="$(sed -n 's/^ASSIST_511_TOKEN=//p' "$TOKEN_FILE")"
    [ -n "$token" ] || { log "token file has no ASSIST_511_TOKEN — skipping GTFS refresh"; return 1; }

    local out="$tmpdir/$GTFS_FILE"
    log "downloading 511 GTFS (operator=$GTFS_511_OPERATOR)..."
    # Pass the URL (with the secret token) via a stdin curl config, NOT argv, so the
    # token isn't exposed in `ps` / /proc to other local users during the download.
    printf 'url = "https://api.511.org/transit/datafeeds?api_key=%s&operator_id=%s"\n' \
        "$token" "$GTFS_511_OPERATOR" \
        | curl -fsS --max-time 300 -o "$out" -K - \
        || { log "511 download failed — keeping current GTFS"; return 1; }

    # Validate: a real GTFS zip with the core files (the 511 zip can be a format
    # `unzip` chokes on, so use Python's zipfile).  Explicit sys.exit, NOT assert,
    # so PYTHONOPTIMIZE can't silently disable the check in a cron.
    python3 - "$out" <<'PY' || { log "downloaded GTFS failed validation — keeping current"; return 1; }
import sys, zipfile
names = set(zipfile.ZipFile(sys.argv[1]).namelist())
if not {"stops.txt", "routes.txt", "trips.txt"} <= names:
    sys.exit("missing core GTFS files")
PY
    log "GTFS valid ($(du -h "$out" | cut -f1))"
    if [ "$CHECK_ONLY" = 1 ]; then log "--check: validated, not swapping GTFS"; return 0; fi
    if [ -f "$INPUT_DIR/$GTFS_FILE" ] && cmp -s "$out" "$INPUT_DIR/$GTFS_FILE"; then
        log "GTFS unchanged since last refresh — skipping swap/re-import"
        return 1
    fi
    mv -f "$out" "$INPUT_DIR/$GTFS_FILE"
    log "GTFS swapped in"
    return 0
}

# --- OSM extract (only when stale) ---------------------------------------------
osm_is_stale() {
    local f="$INPUT_DIR/$OSM_FILE"
    [ -f "$f" ] || return 0  # missing => refresh
    local age_days=$(( ( $(date +%s) - $(stat -c %Y "$f") ) / 86400 ))
    log "OSM extract age: ${age_days}d (threshold ${OSM_MAX_AGE_DAYS}d)"
    [ "$age_days" -ge "$OSM_MAX_AGE_DAYS" ]
}

refresh_osm() {
    local out="$tmpdir/$OSM_FILE"
    log "downloading OSM extract..."
    curl -fsS --max-time 1800 -o "$out" "$OSM_URL" \
        || { log "OSM download failed — keeping current"; return 1; }
    # Validate: a PBF starts with a 4-byte big-endian header length then "OSMHeader",
    # and a NorCal extract is hundreds of MB — guard against a truncated/HTML body.
    local sz; sz=$(stat -c %s "$out")
    [ "$sz" -ge 100000000 ] || { log "OSM download too small (${sz}B) — keeping current"; return 1; }
    grep -qa "OSMHeader" <(head -c 64 "$out") || { log "OSM file not a PBF — keeping current"; return 1; }
    log "OSM valid ($(du -h "$out" | cut -f1))"
    if [ "$CHECK_ONLY" = 1 ]; then log "--check: validated, not swapping OSM"; return 0; fi
    if [ -f "$INPUT_DIR/$OSM_FILE" ] && cmp -s "$out" "$INPUT_DIR/$OSM_FILE"; then
        # Stale by age but byte-identical (Geofabrik didn't change it): don't mark
        # "changed" — that would trigger the heavy Nominatim re-import for nothing.
        # Touch to reset the age clock so we don't re-download it again next week.
        log "OSM re-download is byte-identical — touching mtime, skipping re-import"
        touch "$INPUT_DIR/$OSM_FILE"
        return 1
    fi
    mv -f "$out" "$INPUT_DIR/$OSM_FILE"
    log "OSM swapped in"
    return 0
}

# --- engine rebuilds -----------------------------------------------------------
reimport_motis() {
    # Import IN-PLACE while the running container keeps serving its already-loaded
    # data, then restart ONLY on success.  The container is never stopped, so a
    # failed/killed import (reboot/OOM/timeout) can never leave routing down, and a
    # restart never lands on a half-written graph (it happens only after import
    # exits 0).  MOTIS keys artifacts by content hash, so an OSM-unchanged run only
    # rebuilds the timetable -> ~seconds.
    log "re-importing MOTIS (in-place; restart on success)..."
    docker run --rm --user "$(id -u):$(id -g)" -v "$TRAVEL_INFRA_DIR:/work" -w /work \
        --entrypoint /motis "$MOTIS_IMAGE" import \
        || { log "MOTIS import FAILED — keeping the running engine on its current data"; return 1; }
    docker restart "$MOTIS_CONTAINER" >/dev/null || die "MOTIS restart failed after import"
    log "MOTIS restarted"
}

reimport_nominatim() {
    # The mediagis image only imports into an EMPTY DB, so re-import must wipe the
    # volume — the old instance is gone before the new one finishes building. A
    # rebuild failure therefore degrades geocoding to MOTIS's built-in fallback
    # (travel still ROUTES — not an outage) until the next run re-imports from the
    # on-disk OSM. (A clean container-swap isn't practical here: empty-volume import
    # + Docker can't rename volumes.) Catch the most likely failure — image
    # unavailable — BEFORE destroying the working instance.
    log "re-importing Nominatim (OSM changed)..."
    docker image inspect "$NOMINATIM_IMAGE" >/dev/null 2>&1 || docker pull "$NOMINATIM_IMAGE" >/dev/null \
        || { log "Nominatim image unavailable — keeping the current geocoder"; return 1; }
    docker rm -f "$NOMINATIM_CONTAINER" >/dev/null 2>&1 || true
    docker volume rm "$NOMINATIM_VOLUME" >/dev/null 2>&1 || true
    docker run -d --name "$NOMINATIM_CONTAINER" \
        -e PBF_PATH=/data/"$OSM_FILE" -e NOMINATIM_PASSWORD=nominatim -e IMPORT_WIKIPEDIA=false \
        -v "$INPUT_DIR/$OSM_FILE:/data/$OSM_FILE:ro" \
        -v "$NOMINATIM_VOLUME:/var/lib/postgresql/16/main" \
        -p "127.0.0.1:$NOMINATIM_PORT:8080" --restart unless-stopped --shm-size=1g \
        "$NOMINATIM_IMAGE" >/dev/null \
        || { log "Nominatim re-create FAILED — geocoding stays on the MOTIS fallback"; return 1; }
    log "Nominatim re-import started (serves once /status is 200; geocoding uses MOTIS fallback meanwhile)"
}

# --- run -----------------------------------------------------------------------
log "travel-data refresh starting (infra=$TRAVEL_INFRA_DIR, check_only=$CHECK_ONLY)"
gtfs_changed=0; osm_changed=0
refresh_gtfs && gtfs_changed=1 || true
if osm_is_stale; then refresh_osm && osm_changed=1 || true; else log "OSM still fresh — skipping"; fi

if [ "$CHECK_ONLY" = 1 ]; then
    log "--check complete (gtfs_ok=$gtfs_changed osm_ok=$osm_changed); no import/restart"
    exit 0
fi
if [ "$gtfs_changed" = 0 ] && [ "$osm_changed" = 0 ]; then
    log "nothing changed — skipping rebuilds"; exit 0
fi
reimport_motis
[ "$osm_changed" = 1 ] && reimport_nominatim || true
log "travel-data refresh done (gtfs_changed=$gtfs_changed osm_changed=$osm_changed)"
