"""Killable host-side Docker setup worker; never mounted into a sandbox."""
from __future__ import annotations

import sys

from assist.sandbox_manager import SandboxManager


def main() -> int:
    if sys.argv[1:] != ["proxy"]:
        return 2
    client = SandboxManager._get_docker_client()
    name = SandboxManager._ensure_egress_proxy_running_direct(client)
    print(name, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
