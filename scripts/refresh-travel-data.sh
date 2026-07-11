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

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=travel-lib.sh
source "$HERE/travel-lib.sh"   # TRAVEL_INFRA_DIR check + container/merge/reimport defaults
                               # + shared functions (log/die, take_lock, merge_regions,
                               # set_config_osm, reimport_motis, reimport_nominatim).

# Refresh-specific config (everything else — containers, INPUT_DIR, MERGED_OSM, the lock,
# the reimport/merge functions — is in travel-lib.sh, shared with add/remove/transit):
OSM_URL="${OSM_URL:-https://download.geofabrik.de/north-america/us/california/norcal-latest.osm.pbf}"
OSM_FILE="${OSM_FILE:-norcal.osm.pbf}"
OSM_MAX_AGE_DAYS="${OSM_MAX_AGE_DAYS:-30}"
GTFS_FILE="${GTFS_FILE:-511-regional-gtfs.zip}"
GTFS_511_OPERATOR="${GTFS_511_OPERATOR:-RG}"
TOKEN_FILE="${TOKEN_FILE:-$TRAVEL_INFRA_DIR/.511-token}"

CHECK_ONLY=0
[ "${1:-}" = "--check" ] && CHECK_ONLY=1

[ -d "$INPUT_DIR" ] || die "input dir not found: $INPUT_DIR"
take_lock   # shared .travel-data.lock — mutually exclusive with add/remove/transit (B3)

# Temp dir on the SAME filesystem as INPUT_DIR so the swaps below are atomic
# renames, not a cross-device copy+truncate that could leave a partial file.
tmpdir="$(mktemp -d "$TRAVEL_INFRA_DIR/.refresh-tmp.XXXXXX")"
trap 'rm -rf "$tmpdir"' EXIT

# --- 511 GTFS (always) ---------------------------------------------------------
refresh_gtfs() {
    [ -f "$TOKEN_FILE" ] || { log "no token file ($TOKEN_FILE) — skipping GTFS refresh"; return 1; }
    # Parse the assignment rather than sourcing the file (don't execute config).
    # First line only + strip whitespace, then require the 511 token's charset
    # (alnum/hyphen) so it can't inject directives into the curl -K config below.
    local token; token="$(sed -n 's/^ASSIST_511_TOKEN=//p' "$TOKEN_FILE" | head -n1 | tr -d '[:space:]')"
    [ -n "$token" ] || { log "token file has no ASSIST_511_TOKEN — skipping GTFS refresh"; return 1; }
    case "$token" in *[!0-9A-Za-z-]*) log "511 token has unexpected characters — refusing"; return 1;; esac

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
    # -L: Geofabrik's -latest URLs 302 to a dated file (same host); https-only, bounded.
    curl -fsSL --proto '=https' --proto-redir '=https' --max-redirs 3 --max-time 1800 \
        --max-filesize "$TRAVEL_MAX_DOWNLOAD_BYTES" \
        -o "$out" "$OSM_URL" \
        || { log "OSM download failed — keeping current"; return 1; }
    # Validate: a PBF starts with a 4-byte big-endian header length then "OSMHeader",
    # and a NorCal extract is hundreds of MB — guard against a truncated/HTML body.
    # Upper bound (TRAVEL_MAX_DOWNLOAD_BYTES) backstops --max-filesize for a no-Content-
    # Length response; unlike add-region this keeps current rather than dying (unattended
    # cron — a bad fetch must never replace good data, but also must not wedge the run).
    local sz; sz=$(stat -c %s "$out")
    [ "$sz" -le "$TRAVEL_MAX_DOWNLOAD_BYTES" ] || { log "OSM download over ${TRAVEL_MAX_DOWNLOAD_BYTES}B cap (${sz}B) — keeping current"; return 1; }
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

# (reimport_motis / reimport_nominatim / merge_regions / set_config_osm live in
# travel-lib.sh — shared with add/remove/transit so the load-bearing rebuild recipe
# stays in ONE place. The lib's reimport_nominatim loads the MERGED osm, so a refresh
# must re-merge first when the OSM changed; see below.)

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

# When the OSM changed, re-merge every loaded region (norcal + any added regions) into
# the combined OSM both engines load, point config at it, and rebuild both. When only the
# GTFS changed, a MOTIS timetable reimport suffices (Nominatim ignores GTFS).
if [ "$osm_changed" = 1 ]; then
    tmp_merge="$(mktemp "$TRAVEL_INFRA_DIR/.refresh-merge.XXXXXX")"
    merge_regions "$tmp_merge" && mv -f "$tmp_merge" "$INPUT_DIR/$MERGED_OSM" \
        || die "merge failed — keeping the current combined OSM"
    set_config_osm
    reimport_motis
    reimport_nominatim
else
    reimport_motis   # gtfs-only
fi
log "travel-data refresh done (gtfs_changed=$gtfs_changed osm_changed=$osm_changed)"
