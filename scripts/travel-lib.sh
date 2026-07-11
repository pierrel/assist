#!/usr/bin/env bash
#
# Shared functions for the travel-data scripts (refresh-travel-data.sh, add-region.sh,
# remove-region.sh, add-transit.sh). Sourced, not executed.
#
# The model: MOTIS (routing/geocoding) and Nominatim (geocoding) each load ONE OSM file,
# so multi-region coverage is an osmium MERGE of every loaded region's extract into
# input/combined.osm.pbf, which both engines point at. The per-region source extracts
# live at input/<slug>.osm.pbf (slug '/' flattened to '-'); the merge is derived. Transit
# is separate: per-region GTFS datasets under timetable.datasets in config.yml (MOTIS
# ingests many; Nominatim ignores them).
#
# Every mutation VALIDATES before it swaps, and rebuilds the engines the same way the
# weekly refresh does: MOTIS in-place (never stops serving; restart only on import
# success), Nominatim wipe+reimport (degrades geocoding to MOTIS's built-in fallback —
# NOT an outage — until the import finishes).
#
# Config via env (the caller/cron sets TRAVEL_INFRA_DIR; the rest default to the standard
# single-box deploy):
: "${TRAVEL_INFRA_DIR:?set TRAVEL_INFRA_DIR to the travel infra dir (input/, config.yml, data/)}"
MOTIS_CONTAINER="${MOTIS_CONTAINER:-motis-travel}"
MOTIS_IMAGE="${MOTIS_IMAGE:-ghcr.io/motis-project/motis:latest}"
NOMINATIM_CONTAINER="${NOMINATIM_CONTAINER:-nominatim-geocoder}"
NOMINATIM_VOLUME="${NOMINATIM_VOLUME:-nominatim-geocoder-data}"
NOMINATIM_IMAGE="${NOMINATIM_IMAGE:-mediagis/nominatim:4.5}"
NOMINATIM_PORT="${NOMINATIM_PORT:-8089}"
OSMIUM_IMAGE="${OSMIUM_IMAGE:-stefda/osmium-tool:latest}"
# The merged OSM both engines load. A slug never forms this name (it's fixed), and it is
# EXCLUDED from the region-source glob so it never merges into itself.
MERGED_OSM="${MERGED_OSM:-combined.osm.pbf}"

INPUT_DIR="$TRAVEL_INFRA_DIR/input"
CONFIG="$TRAVEL_INFRA_DIR/config.yml"
# ONE lock shared by every writer of input/ + config.yml + the Nominatim volume — the
# weekly refresh and add/remove/transit must be mutually exclusive or two destructive
# rebuilds interleave (design doc B3).
LOCK_FILE="$TRAVEL_INFRA_DIR/.travel-data.lock"

log() { echo "[$(date '+%Y-%m-%dT%H:%M:%S%z')] $*"; }
die() { log "ERROR: $*"; exit 1; }

# Take the shared lock on fd 9 (the caller keeps it for the whole run). Non-blocking:
# refuse rather than queue, so a stuck run can't pile up.
take_lock() {
    exec 9>"$LOCK_FILE"
    flock -n 9 || die "another travel-data operation is already running (lock held)"
}

# A slug is a Geofabrik index id: lowercase alnum, '-', and '/' (nested, e.g.
# us/washington). No leading/trailing '/', no '..', bounded length. The FLATTENED form
# ('/'->'-') is the on-disk filename, so a slug can never traverse a path.
valid_slug() {
    local s="$1"
    case "$s" in
        ""|/*|*/|*..*|*//*) return 1 ;;
        *[!a-z0-9/-]*) return 1 ;;
    esac
    [ "${#s}" -le 64 ]
}
region_file() {  # slug -> input/<flattened>.osm.pbf
    printf '%s/%s.osm.pbf' "$INPUT_DIR" "$(printf '%s' "$1" | tr '/' '-')"
}

# A real OSM PBF: at least a few MB (guard truncation/HTML error pages) and the OSMHeader
# magic. Explicit checks, not asserts.
validate_pbf() {
    local f="$1"
    local sz; sz=$(stat -c %s "$f" 2>/dev/null || echo 0)
    [ "$sz" -ge 1000000 ] || { log "PBF $f too small (${sz}B)"; return 1; }
    grep -qa "OSMHeader" <(head -c 64 "$f") || { log "PBF $f is not an OSM file"; return 1; }
}

# Merge every region-source extract (input/*.osm.pbf except the merged file) into a new
# combined PBF at $1, via a rootless osmium container. Validates the result before the
# caller swaps it in. A single source is still "merged" (osmium copies it) so the path is
# uniform. Fails (non-zero) rather than producing a partial file.
merge_regions() {
    local out="$1"   # absolute path under $TRAVEL_INFRA_DIR (usually a temp file)
    local sources=()
    local f base
    for f in "$INPUT_DIR"/*.osm.pbf; do
        base="$(basename "$f")"
        [ "$base" = "$MERGED_OSM" ] && continue
        [ -f "$f" ] && sources+=("input/$base")
    done
    [ "${#sources[@]}" -ge 1 ] || { log "no region sources to merge"; return 1; }
    # Mount the whole infra dir so both the sources (input/…) and the output (a temp file
    # under $TRAVEL_INFRA_DIR) are visible; use paths RELATIVE to that mount so osmium's
    # -o lands where the caller validates it, not in input/.
    local out_rel="${out#"$TRAVEL_INFRA_DIR"/}"
    log "merging ${#sources[@]} region source(s): ${sources[*]}"
    docker run --rm --user "$(id -u):$(id -g)" -v "$TRAVEL_INFRA_DIR:/data" -w /data \
        "$OSMIUM_IMAGE" osmium merge --overwrite -o "$out_rel" "${sources[@]}" \
        || { log "osmium merge failed"; return 1; }
    validate_pbf "$out" || { log "merged PBF failed validation"; return 1; }
    log "merged OSM built ($(du -h "$out" | cut -f1))"
}

# Point config.yml's single `osm:` at the merged file (idempotent). Only rewrites the top
# `osm:` line; everything else (the timetable/datasets) is untouched.
set_config_osm() {
    local rel="input/$MERGED_OSM"
    grep -q "^osm: $rel$" "$CONFIG" && return 0
    local tmp; tmp="$(mktemp "$CONFIG.XXXXXX")"
    sed "0,/^osm:.*/s|^osm:.*|osm: $rel|" "$CONFIG" > "$tmp" && mv -f "$tmp" "$CONFIG"
    log "config osm -> $rel"
}

# MOTIS: import in-place (container keeps serving), restart only on success — a
# failed/killed import can never leave routing down or land on a half-written graph.
reimport_motis() {
    log "re-importing MOTIS (in-place; restart on success)..."
    docker run --rm --user "$(id -u):$(id -g)" -v "$TRAVEL_INFRA_DIR:/work" -w /work \
        --entrypoint /motis "$MOTIS_IMAGE" import \
        || { log "MOTIS import FAILED — keeping the running engine on its current data"; return 1; }
    docker restart "$MOTIS_CONTAINER" >/dev/null || die "MOTIS restart failed after import"
    log "MOTIS restarted"
}

# Nominatim: the mediagis image imports into an EMPTY DB, so re-import wipes the volume —
# the old geocoder is gone before the new one finishes. During the rebuild, name/address
# geocoding via Nominatim is UNAVAILABLE (assist's _geocode raises → "unavailable"; it only
# falls back to MOTIS's weaker geocoder if ASSIST_GEOCODER_URL is unset, which it isn't).
# Coordinate/"from here" routing is UNAFFECTED (no geocode). Catch image-unavailable BEFORE
# destroying the working instance. Loads the MERGED OSM.
reimport_nominatim() {
    local osmfile="$INPUT_DIR/$MERGED_OSM"
    [ -f "$osmfile" ] || { log "merged OSM missing ($osmfile) — refusing Nominatim reimport"; return 1; }
    log "re-importing Nominatim from $MERGED_OSM (wipe+rebuild)..."
    docker image inspect "$NOMINATIM_IMAGE" >/dev/null 2>&1 || docker pull "$NOMINATIM_IMAGE" >/dev/null \
        || { log "Nominatim image unavailable — keeping the current geocoder"; return 1; }
    docker rm -f "$NOMINATIM_CONTAINER" >/dev/null 2>&1 || true
    docker volume rm "$NOMINATIM_VOLUME" >/dev/null 2>&1 || true
    docker run -d --name "$NOMINATIM_CONTAINER" \
        -e PBF_PATH=/data/"$MERGED_OSM" -e NOMINATIM_PASSWORD=nominatim -e IMPORT_WIKIPEDIA=false \
        -v "$osmfile:/data/$MERGED_OSM:ro" \
        -v "$NOMINATIM_VOLUME:/var/lib/postgresql/16/main" \
        -p "127.0.0.1:$NOMINATIM_PORT:8080" --restart unless-stopped --shm-size=1g \
        "$NOMINATIM_IMAGE" >/dev/null \
        || { log "Nominatim re-create FAILED — name/address geocoding is DOWN until it re-imports (routing unaffected)"; return 1; }
    log "Nominatim re-import started (serves once /status is 200); name/address geocoding is UNAVAILABLE meanwhile (routing + 'from here' coords unaffected)"
}

# Block until the (detached) Nominatim import finishes and the geocoder answers /status
# 200 — so an add/remove only reports success (and the Provisioner marks the region ready)
# once geocoding is actually live for it (design doc T5). The refresh cron does NOT wait
# (it's fire-and-forget). Fails on timeout OR if the container died mid-import.
wait_nominatim_ready() {
    local timeout_s="${1:-21600}"   # 6h — a large merged extract imports slowly
    local deadline=$(( $(date +%s) + timeout_s ))
    log "waiting for the Nominatim import to finish (/status 200; up to $((timeout_s/3600))h)..."
    while [ "$(date +%s)" -lt "$deadline" ]; do
        docker ps --format '{{.Names}}' | grep -qx "$NOMINATIM_CONTAINER" \
            || { log "Nominatim container exited during import"; return 1; }
        if [ "$(curl -s -o /dev/null -w '%{http_code}' \
                "http://127.0.0.1:$NOMINATIM_PORT/status" 2>/dev/null)" = "200" ]; then
            log "Nominatim import complete (/status 200)"
            return 0
        fi
        sleep 30
    done
    log "Nominatim import did not reach /status 200 within ${timeout_s}s"
    return 1
}
