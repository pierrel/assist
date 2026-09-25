"""Killable host-side Docker setup worker; never mounted into a sandbox."""
from __future__ import annotations

import sys
import json
import re

from assist.sandbox_manager import SandboxManager


def main() -> int:
    client = SandboxManager._get_docker_client()
    if sys.argv[1:] == ["proxy"]:
        name = SandboxManager._ensure_egress_proxy_running_direct(client)
        print(name, flush=True)
        return 0
    if (len(sys.argv) == 5 and sys.argv[1] == "record"
            and re.fullmatch(r"[0-9a-f]{64}", sys.argv[2])
            and sys.argv[4] in {"sandbox", "pi"}):
        container = client.containers.get(sys.argv[2])
        identity = SandboxManager._record_egress_client_direct(
            client, container, sys.argv[3], kind=sys.argv[4])
        print(json.dumps(identity), flush=True)
        return 0
    if (len(sys.argv) == 13 and sys.argv[1] == "browser-record"
            and re.fullmatch(r"[0-9a-f]{64}", sys.argv[2])
            and all(re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}", name)
                    for name in sys.argv[11:13])):
        from assist.browser.manager import _register_browser_client_direct
        from assist.egress.client_map import ClientRecord
        mode = sys.argv[6]
        host = None if sys.argv[7] == "-" else sys.argv[7]
        port = int(sys.argv[8]) or None
        record = ClientRecord(sys.argv[5], sys.argv[2], "browser", mode,
                              host, port)
        _register_browser_client_direct(
            client, sys.argv[3], sys.argv[4], record, sys.argv[9], sys.argv[10],
            proxy_name=sys.argv[11], network_name=sys.argv[12])
        print("ok", flush=True)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
