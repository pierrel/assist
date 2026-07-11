#!/usr/bin/env bash
#
# Remove a loaded region: drop its OSM source, re-merge the rest, rebuild both engines.
#
#   remove-region.sh <slug>
#
# Refuses to remove the LAST region (that would leave the engines with no OSM). The base
# region (BASE_REGION_SLUG, default norcal) is likewise protected. Exit 0 on success; the
# Provisioner reads the exit code (this script never touches regions.json).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=travel-lib.sh
source "$HERE/travel-lib.sh"

BASE_REGION_SLUG="${BASE_REGION_SLUG:-norcal}"
SLUG="${1:-}"

valid_slug "$SLUG" || die "invalid region slug: '${SLUG}'"
[ "$SLUG" = "$BASE_REGION_SLUG" ] && die "refusing to remove the base region '$SLUG'"

take_lock

REGION_FILE="$(region_file "$SLUG")"
[ -f "$REGION_FILE" ] || die "region '$SLUG' is not loaded ($REGION_FILE not found)"

# Count region sources (exclude the merged file); refuse to remove the last one.
count=0
for f in "$INPUT_DIR"/*.osm.pbf; do
    [ "$(basename "$f")" = "$MERGED_OSM" ] && continue
    [ -f "$f" ] && count=$((count + 1))
done
[ "$count" -ge 2 ] || die "refusing to remove the only loaded region"

tmpdir="$(mktemp -d "$TRAVEL_INFRA_DIR/.rm-tmp.XXXXXX")"
# Stash the removed source so a merge/reimport failure can restore it (the region stays
# served — validate-before-swap).
stash="$tmpdir/$(basename "$REGION_FILE")"
mv -f "$REGION_FILE" "$stash"
restore() { [ -f "$stash" ] && mv -f "$stash" "$REGION_FILE"; rm -rf "$tmpdir"; }
trap restore EXIT

merge_regions "$tmpdir/$MERGED_OSM" || die "merge after removal failed (region restored)"
mv -f "$tmpdir/$MERGED_OSM" "$INPUT_DIR/$MERGED_OSM"
set_config_osm
reimport_motis || die "MOTIS reimport failed (region restored)"
reimport_nominatim || die "Nominatim reimport failed"
rm -f "$stash"   # committed — don't restore on the trap
wait_nominatim_ready || die "Nominatim import did not complete after removal"
log "region '$SLUG' removed and the geocoder is live again."
