#!/usr/bin/env bash
#
# Add the transit (GTFS) layer to an already-loaded region: download its configured feed,
# register it as a MOTIS dataset, and reimport MOTIS. CHEAP — MOTIS-only (no OSM merge, no
# Nominatim rebuild); a region's serveability is unchanged (it's READY throughout, which
# is why the Provisioner claims transit in memory, not via state:importing).
#
#   add-transit.sh <slug>
#
# The feed comes from the catalog entry's transit_feed for the slug (hand-curated overlay;
# there is no worldwide GTFS index). Two forms:
#   "511:<operator>"  — the 511.org regional API (needs $TRAVEL_INFRA_DIR/.511-token)
#   "https://…"       — a direct GTFS zip URL
# Exit 0 on success; the Provisioner reads the exit code (never touches regions.json).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=travel-lib.sh
source "$HERE/travel-lib.sh"

CATALOG="${CATALOG:-$TRAVEL_INFRA_DIR/catalog.json}"
TOKEN_FILE="${TOKEN_FILE:-$TRAVEL_INFRA_DIR/.511-token}"
SLUG="${1:-}"

valid_slug "$SLUG" || die "invalid region slug: '${SLUG}'"
[ -f "$CATALOG" ] || die "catalog not found: $CATALOG"
[ -f "$(region_file "$SLUG")" ] || die "region '$SLUG' is not loaded — add it before transit"

FEED="$(python3 - "$CATALOG" "$SLUG" <<'PY'
import json, sys
cat = {e["slug"]: e for e in json.load(open(sys.argv[1])) if isinstance(e, dict)}
e = cat.get(sys.argv[2])
print((e or {}).get("transit_feed") or "")
PY
)"
[ -n "$FEED" ] || die "no transit feed configured for '$SLUG'"

take_lock
tmpdir="$(mktemp -d "$TRAVEL_INFRA_DIR/.transit-tmp.XXXXXX")"
gtfs_name="$(printf '%s' "$SLUG" | tr '/' '-')-gtfs.zip"
out="$tmpdir/$gtfs_name"
# Transactional: a non-zero exit must leave NO change (else config.yml + the GTFS zip
# desync from the registry's has_transit). Stash the config; on any failure before
# commit, restore it and remove the just-added GTFS.
cp -f "$CONFIG" "$tmpdir/config.bak"
committed=0
cleanup() {
    if [ "$committed" = 0 ]; then
        [ -f "$tmpdir/config.bak" ] && mv -f "$tmpdir/config.bak" "$CONFIG"
        rm -f "$INPUT_DIR/$gtfs_name"
    fi
    rm -rf "$tmpdir"
}
trap cleanup EXIT

case "$FEED" in
    511:*)
        local_op="${FEED#511:}"
        [ -f "$TOKEN_FILE" ] || die "511 feed needs $TOKEN_FILE"
        token="$(sed -n 's/^ASSIST_511_TOKEN=//p' "$TOKEN_FILE" | head -n1 | tr -d '[:space:]')"
        case "$token" in ""|*[!0-9A-Za-z-]*) die "511 token missing/invalid";; esac
        case "$local_op" in ""|*[!0-9A-Za-z-]*) die "bad 511 operator '$local_op'";; esac
        log "downloading 511 GTFS (operator=$local_op)..."
        # token via a stdin curl config, never argv (not exposed in ps/proc)
        printf 'url = "https://api.511.org/transit/datafeeds?api_key=%s&operator_id=%s"\n' \
            "$token" "$local_op" | curl -fsS --max-time 300 \
            --max-filesize "$TRAVEL_MAX_DOWNLOAD_BYTES" -o "$out" -K - \
            || die "511 GTFS download failed"
        ;;
    https://*)
        log "downloading GTFS from $FEED ..."
        curl -fsSL --proto '=https' --proto-redir '=https' --max-redirs 3 --max-time 600 \
            --max-filesize "$TRAVEL_MAX_DOWNLOAD_BYTES" \
            -o "$out" "$FEED" || die "GTFS download failed"
        ;;
    *) die "unrecognized transit_feed form: '$FEED'" ;;
esac
check_download_size "$out"

# Validate: a real GTFS zip with the core files (511 can produce a zip unzip chokes on).
python3 - "$out" <<'PY' || die "downloaded GTFS failed validation"
import sys, zipfile
names = set(zipfile.ZipFile(sys.argv[1]).namelist())
if not {"stops.txt", "routes.txt", "trips.txt"} <= names:
    sys.exit("missing core GTFS files")
PY
mv -f "$out" "$INPUT_DIR/$gtfs_name"

# Register the dataset under timetable.datasets (idempotent) via a small YAML edit.
python3 - "$CONFIG" "$SLUG" "$gtfs_name" <<'PY'
import sys, yaml
cfg_path, slug, gtfs = sys.argv[1], sys.argv[2], sys.argv[3]
cfg = yaml.safe_load(open(cfg_path))
ds = cfg.setdefault("timetable", {}).setdefault("datasets", {})
key = slug.replace("/", "-")
ds[key] = {"path": f"input/{gtfs}", "default_bikes_allowed": False,
           "default_cars_allowed": False, "extend_calendar": False}
yaml.safe_dump(cfg, open(cfg_path, "w"), sort_keys=False)
PY
log "registered transit dataset for '$SLUG'; reimporting MOTIS..."
reimport_motis || die "MOTIS reimport failed"
committed=1   # config + GTFS + engine all succeeded — keep them
log "transit added for '$SLUG'."
