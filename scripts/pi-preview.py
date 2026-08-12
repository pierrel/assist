#!/usr/bin/env python3
"""Locally control the fail-closed Pi web-preview switch."""
from __future__ import annotations

import os
import sys

from assist.pi_preview import PiPreviewPolicy


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in {"enable", "disable", "status"}:
        raise SystemExit("usage: pi-preview.py {enable|disable|status}")
    policy = PiPreviewPolicy(os.environ.get("ASSIST_THREADS_DIR", "/tmp/assist_threads"))
    if sys.argv[1] == "status":
        print("enabled" if policy.enabled() else "disabled")
        return 0
    policy.set_enabled(sys.argv[1] == "enable")
    print("enabled" if policy.enabled() else "disabled")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
