#!/usr/bin/env bash
#
# Add a geographic region: download its OSM extract, merge it into the combined OSM, and
# rebuild both engines so travel/directions/geocoding cover it.
#
#   add-region.sh <slug> [--check]
#
# <slug> must be an id in the pinned catalog (catalog.json) — the download URL comes from
# the catalog entry, NEVER the caller. --check downloads + validates + trial-merges
# WITHOUT swapping or reimporting (a dry run). Exit 0 on success; the Provisioner reads
# the exit code and updates the registry (this script never touches regions.json).
#
# Env: TRAVEL_INFRA_DIR (required) + the travel-lib defaults. CATALOG defaults to
# $TRAVEL_INFRA_DIR/catalog.json.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=travel-lib.sh
source "$HERE/travel-lib.sh"

CATALOG="${CATALOG:-$TRAVEL_INFRA_DIR/catalog.json}"
SLUG="${1:-}"
CHECK_ONLY=0
[ "${2:-}" = "--check" ] && CHECK_ONLY=1

valid_slug "$SLUG" || die "invalid region slug: '${SLUG}'"
[ -f "$CATALOG" ] || die "catalog not found: $CATALOG (build it: python -m assist.geo.build_catalog \$TRAVEL_INFRA_DIR)"

# The URL comes from the catalog entry for this slug — never the argv. A slug not in the
# catalog is refused (the download allowlist, by construction).
URL="$(python3 - "$CATALOG" "$SLUG" <<'PY'
import json, sys
cat = {e["slug"]: e for e in json.load(open(sys.argv[1])) if isinstance(e, dict)}
e = cat.get(sys.argv[2])
if not e:
    sys.exit(1)
url = e.get("url", "")
# defence in depth: the builder already https/host-gated, re-check before curl
from urllib.parse import urlparse
p = urlparse(url)
if p.scheme != "https" or p.netloc != "download.geofabrik.de":
    sys.exit(2)
print(url)
PY
)" || die "'$SLUG' is not a downloadable catalog region (or its URL is not https-on-geofabrik)"

take_lock

REGION_FILE="$(region_file "$SLUG")"
tmpdir="$(mktemp -d "$TRAVEL_INFRA_DIR/.add-tmp.XXXXXX")"
cp -f "$CONFIG" "$tmpdir/config.bak"
# If the region is already present (a re-add / a manual --check), stash its source (fast
# mv, same fs) so the rollback restores it instead of deleting it; the download recreates
# REGION_FILE.
[ -f "$REGION_FILE" ] && mv -f "$REGION_FILE" "$tmpdir/region.bak"
committed=0
# Transactional: --check and any FAILURE leave input/ + config exactly as found, so a
# non-zero exit never leaves a region source (seed_registry would else mark it READY), a
# stale config, or a half-swapped combined OSM.
_restore_region() {
    if [ -f "$tmpdir/region.bak" ]; then mv -f "$tmpdir/region.bak" "$REGION_FILE"
    else rm -f "$REGION_FILE"; fi
}
cleanup() {
    if [ "$CHECK_ONLY" = 1 ]; then
        _restore_region
    elif [ "$committed" = 0 ]; then
        _restore_region
        mv -f "$tmpdir/config.bak" "$CONFIG" 2>/dev/null || true
        if [ -f "$tmpdir/combined.bak" ]; then
            mv -f "$tmpdir/combined.bak" "$INPUT_DIR/$MERGED_OSM"
        else
            rm -f "$INPUT_DIR/$MERGED_OSM"
        fi
    fi
    rm -rf "$tmpdir"
}
trap cleanup EXIT

if [ -f "$REGION_FILE" ] && [ "$CHECK_ONLY" = 0 ]; then
    log "region '$SLUG' already present at $REGION_FILE — re-downloading + re-merging"
fi

log "downloading $SLUG from $URL ..."
# -L: Geofabrik's -latest URLs 302 to a dated file (same host). --proto-redir '=https'
# keeps every hop https; --max-redirs bounds the chain; validate_pbf (OSMHeader) is the
# backstop against a wrong target.
curl -fsSL --proto '=https' --proto-redir '=https' --max-redirs 3 --max-time 1800 \
    --max-filesize "$TRAVEL_MAX_DOWNLOAD_BYTES" \
    -o "$tmpdir/region.osm.pbf" "$URL" \
    || die "download failed for $SLUG"
check_download_size "$tmpdir/region.osm.pbf"
validate_pbf "$tmpdir/region.osm.pbf" || die "downloaded PBF failed validation"
log "downloaded $SLUG ($(du -h "$tmpdir/region.osm.pbf" | cut -f1))"

# Place the region source, then merge ALL region sources into a temp combined.
mv -f "$tmpdir/region.osm.pbf" "$REGION_FILE"
merge_regions "$tmpdir/$MERGED_OSM" || die "merge failed"

if [ "$CHECK_ONLY" = 1 ]; then
    log "--check OK: $SLUG downloads, validates, and merges cleanly. Not swapping/reimporting."
    exit 0   # cleanup removes the region file + tmp
fi

# Swap the merged OSM in atomically (same filesystem), stashing the prior combined so a
# later failure rolls back to it; point config at it; rebuild.
[ -f "$INPUT_DIR/$MERGED_OSM" ] && mv -f "$INPUT_DIR/$MERGED_OSM" "$tmpdir/combined.bak"
mv -f "$tmpdir/$MERGED_OSM" "$INPUT_DIR/$MERGED_OSM"
set_config_osm
reimport_motis || die "MOTIS reimport failed"
reimport_nominatim || die "Nominatim reimport failed"
wait_nominatim_ready || die "Nominatim import did not complete"
committed=1   # source + combined + config + both engines all succeeded — keep them
log "region '$SLUG' added and live (MOTIS + Nominatim cover it)."
