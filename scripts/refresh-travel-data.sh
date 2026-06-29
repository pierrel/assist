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
#   TRAVEL_INFRA_DIR     dir with input/ + the MOTIS config.yml + data/ graph
#                        (default: $HOME/motis-travel)
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
#   TOKEN_FILE           file containing `511_TOKEN=...` (default: $TRAVEL_INFRA_DIR/.511-token)
#
# Flags:
#   --check   download + validate into temp files only; do NOT swap, import, or restart
#
set -euo pipefail

TRAVEL_INFRA_DIR="${TRAVEL_INFRA_DIR:-$HOME/motis-travel}"
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

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/travel-refresh.XXXXXX")"
trap 'rm -rf "$tmpdir"' EXIT

# --- 511 GTFS (always) ---------------------------------------------------------
refresh_gtfs() {
    [ -f "$TOKEN_FILE" ] || { log "no token file ($TOKEN_FILE) — skipping GTFS refresh"; return 1; }
    # shellcheck disable=SC1090
    local token; token="$(. "$TOKEN_FILE"; echo "${ASSIST_511_TOKEN:-}")"
    [ -n "$token" ] || { log "token file has no ASSIST_511_TOKEN — skipping GTFS refresh"; return 1; }

    local out="$tmpdir/$GTFS_FILE"
    log "downloading 511 GTFS (operator=$GTFS_511_OPERATOR)..."
    curl -fsS --max-time 300 -o "$out" \
        "https://api.511.org/transit/datafeeds?api_key=${token}&operator_id=${GTFS_511_OPERATOR}" \
        || { log "511 download failed — keeping current GTFS"; return 1; }

    # Validate: a real GTFS zip with stops.txt (the 511 zip can be a format `unzip`
    # chokes on, so use Python's zipfile).
    python3 - "$out" <<'PY' || { log "downloaded GTFS failed validation — keeping current"; return 1; }
import sys, zipfile
z = zipfile.ZipFile(sys.argv[1])
names = z.namelist()
assert "stops.txt" in names and "routes.txt" in names and "trips.txt" in names, "missing core GTFS files"
PY
    log "GTFS valid ($(du -h "$out" | cut -f1))"
    if [ "$CHECK_ONLY" = 1 ]; then log "--check: validated, not swapping GTFS"; return 0; fi
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
    mv -f "$out" "$INPUT_DIR/$OSM_FILE"
    log "OSM swapped in"
    return 0
}

# --- engine rebuilds -----------------------------------------------------------
reimport_motis() {
    log "re-importing MOTIS (stop -> import -> start)..."
    docker stop "$MOTIS_CONTAINER" >/dev/null 2>&1 || true
    docker run --rm --user "$(id -u):$(id -g)" -v "$TRAVEL_INFRA_DIR:/work" -w /work \
        --entrypoint /motis "$MOTIS_IMAGE" import \
        || { log "MOTIS import FAILED — starting old container back up"; docker start "$MOTIS_CONTAINER" >/dev/null || true; return 1; }
    docker start "$MOTIS_CONTAINER" >/dev/null
    log "MOTIS restarted"
}

reimport_nominatim() {
    log "re-importing Nominatim (OSM changed)..."
    docker rm -f "$NOMINATIM_CONTAINER" >/dev/null 2>&1 || true
    docker volume rm "$NOMINATIM_VOLUME" >/dev/null 2>&1 || true
    docker run -d --name "$NOMINATIM_CONTAINER" \
        -e PBF_PATH=/data/"$OSM_FILE" -e NOMINATIM_PASSWORD=nominatim -e IMPORT_WIKIPEDIA=false \
        -v "$INPUT_DIR/$OSM_FILE:/data/$OSM_FILE:ro" \
        -v "$NOMINATIM_VOLUME:/var/lib/postgresql/16/main" \
        -p "127.0.0.1:$NOMINATIM_PORT:8080" --restart unless-stopped --shm-size=1g \
        "$NOMINATIM_IMAGE" >/dev/null
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
