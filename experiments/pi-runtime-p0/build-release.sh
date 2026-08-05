#!/bin/sh
set -eu
exec python3 "$(dirname "$0")/python/pi_runtime_p0.py" build "$@"
