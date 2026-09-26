"""Disposable host process for bounded browser sidecar startup."""
from __future__ import annotations

import json
import sys

from assist.browser.manager import _launch_sidecar_direct


def main() -> int:
    request = json.load(sys.stdin)
    if not isinstance(request, dict):
        return 2
    result, _ = _launch_sidecar_direct(request)
    print(json.dumps(result, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
