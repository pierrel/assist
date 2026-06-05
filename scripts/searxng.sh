#!/usr/bin/env bash
# Manage the self-hosted SearXNG metasearch container for assist.
#
# Runs the official searxng/searxng image as a long-lived, localhost-only
# service (127.0.0.1 only — never exposed off-box) with assist's settings
# mounted.  `--restart unless-stopped` keeps it alive across reboots (as long
# as the docker daemon is enabled).  assist's search_internet talks to it via
# ASSIST_SEARCH_URL — there is NO fallback, so if this is down search fails
# loudly rather than silently degrading.
#
# Usage: scripts/searxng.sh {up|down|logs|status}
set -euo pipefail

NAME="assist-searxng"
# Pin the image for reproducible deploys (manifesto: "freeze a version that
# works").  Override via SEARXNG_IMAGE (e.g. to a digest) when you want to
# move deliberately; default tracks the maintained image.
IMAGE="${SEARXNG_IMAGE:-searxng/searxng:latest}"
PORT="${SEARXNG_PORT:-8890}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONF_SRC="$REPO_DIR/dockerfiles/searxng/settings.yml"
# Gitignored per-host runtime dir holds the generated secret + rendered config.
RUNTIME_DIR="${SEARXNG_RUNTIME_DIR:-$REPO_DIR/.searxng}"
SECRET_FILE="$RUNTIME_DIR/secret"

up() {
  mkdir -p "$RUNTIME_DIR"
  if [ ! -s "$SECRET_FILE" ]; then
    openssl rand -hex 32 > "$SECRET_FILE"
    chmod 600 "$SECRET_FILE"
  fi
  # Render a runtime settings.yml with the per-host secret injected; the
  # committed config keeps only the placeholder, so no secret lands in git.
  # Anchor on the whole secret_key line so only the value is replaced, never
  # an incidental "ultrasecretkey" elsewhere in the file.
  sed "s|secret_key: \"ultrasecretkey\"|secret_key: \"$(cat "$SECRET_FILE")\"|" \
    "$CONF_SRC" > "$RUNTIME_DIR/settings.yml"
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  docker run -d --name "$NAME" --restart unless-stopped \
    -p "127.0.0.1:${PORT}:8080" \
    -v "$RUNTIME_DIR/settings.yml:/etc/searxng/settings.yml:ro" \
    "$IMAGE" >/dev/null
  echo "SearXNG up on http://127.0.0.1:${PORT} (set ASSIST_SEARCH_URL to this)"
}

down() { docker rm -f "$NAME" >/dev/null 2>&1 || true; echo "SearXNG removed"; }
logs() { docker logs -f "$NAME"; }
status() { docker ps --filter "name=$NAME" --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'; }

case "${1:-up}" in
  up) up ;;
  down) down ;;
  logs) logs ;;
  status) status ;;
  *) echo "usage: $0 {up|down|logs|status}" >&2; exit 1 ;;
esac
