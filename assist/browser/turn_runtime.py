"""Bounded host setup for a browser process in one existing turn sandbox."""
from __future__ import annotations

import json
import os
import sys
import time

from assist.browser import authority
from assist.egress import runtime_state
from assist.egress.client_map import ClientRecord, read_client, set_sandbox_browser_mode
from assist.run_service import RunService
from assist.sandbox_manager import (
    BROWSER_PROXY_PORT, EGRESS_NETWORK, EGRESS_PROXY_NAME, _proxy_setup_lock,
    SandboxManager)


def _endpoints(client, request):
    sandbox = client.containers.get(request["generation"])
    proxy = client.containers.get(EGRESS_PROXY_NAME)
    sandbox.reload()
    proxy.reload()
    sandbox_network = sandbox.attrs["NetworkSettings"]["Networks"][EGRESS_NETWORK]
    proxy_network = proxy.attrs["NetworkSettings"]["Networks"][EGRESS_NETWORK]
    mounts = sandbox.attrs.get("Mounts") or []
    proxy_mounts = proxy.attrs.get("Mounts") or []
    if (sandbox.status != "running" or proxy.status != "running"
            or sandbox.attrs["Config"]["User"].split(":", 1)[0] in {"0", "10001"}
            or not any(m.get("Destination") == "/workspace"
                       and os.path.realpath(m.get("Source", "")) ==
                       os.path.realpath(request["work_dir"]) for m in mounts)
            or [m.get("Source") for m in proxy_mounts
                if m.get("Destination") == "/client-map"] != [request["map_dir"]]
            or sandbox_network["NetworkID"] != proxy_network["NetworkID"]
            or sandbox_network["IPAddress"] != request["ip"]
            or not proxy_network["IPAddress"]):
        raise RuntimeError("browser sandbox or proxy identity changed")
    runtime_state.assert_admissible("proxy", proxy.id)
    runtime_state.assert_admissible("network", proxy_network["NetworkID"])
    return sandbox, proxy, proxy_network["IPAddress"]


def _check_lease(request):
    with authority.fence(request["threads_root"], request["thread_id"]) as state:
        lease = state.lease
        if (lease is None or lease["owner_run_id"] != request["run_id"]
                or request["generation"] not in lease["generations"]):
            raise RuntimeError("browser turn lease changed")
        runs = RunService(request["threads_root"]).list(request["thread_id"])
        if any(run.status == "revocation_pending" for run in runs):
            raise RuntimeError("newer user message revoked browser startup")
        latest = max((run.admission_sequence for run in runs
                      if run.user_event_id == run.id), default=0)
        if latest != lease["sequence"]:
            raise RuntimeError("newer user message revoked browser startup")


def launch(request):
    """Fence UID, publish exact mode, then launch; caller kills on any doubt."""
    from assist.browser.manager import _bounded_cli

    client = SandboxManager._get_docker_client()
    with _proxy_setup_lock():
        _, proxy, proxy_ip = _endpoints(client, request)
        if read_client(request["map_dir"], request["ip"]) != ClientRecord(
                request["thread_id"], request["generation"], "sandbox"):
            raise RuntimeError("browser sandbox attribution changed")
        proxy_id = proxy.id
    _check_lease(request)
    _bounded_cli([
        "docker", "exec", "--privileged", "--user", "0", request["generation"],
        "/usr/bin/python", "-I", "/opt/assist/install-browser-fence.py", proxy_ip],
        limit=1024, timeout=15)
    with _proxy_setup_lock():
        sandbox, proxy, current_ip = _endpoints(client, request)
        if proxy.id != proxy_id or current_ip != proxy_ip:
            raise RuntimeError("browser proxy changed during network fence setup")
        _check_lease(request)
        set_sandbox_browser_mode(
            request["map_dir"], request["ip"],
            ClientRecord(request["thread_id"], request["generation"], "sandbox"),
            request["mode"], request.get("internal_host"),
            request.get("internal_port"))
    environment = {
        "BROWSER_SESSION_TOKEN": request["token"],
        "BROWSER_BOOT_ID": request["boot_id"],
        "BROWSER_DEADLINE_NS": str(request["deadline_ns"]),
        "BROWSER_GENERATION": request["generation"],
        "BROWSER_PROXY_HOST": proxy_ip,
        "BROWSER_PROXY_PORT": str(BROWSER_PROXY_PORT),
    }
    sandbox.exec_run([
        "/usr/bin/unshare", "--pid", "--fork", "--kill-child", "--",
        "/usr/bin/setpriv", "--reuid=10001", "--regid=10001",
        "--clear-groups", "--inh-caps=-all", "--ambient-caps=-all",
        "--bounding-set=-all", "--no-new-privs", "--",
        "/usr/bin/python", "-I", "/opt/assist/launch-turn-browser.py"],
        detach=True, privileged=True, user="0:0", environment=environment)
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        result = sandbox.exec_run(
            ["/usr/bin/test", "-S", "/run/assist-browser/browser.sock"],
            user="10001:10001")
        if result.exit_code == 0:
            from assist.browser.manager import BrowserUnavailable, _docker_exec
            try:
                answer = json.loads(_docker_exec(
                    request["generation"], json.dumps({
                        "session": request["token"], "operation": "ready"
                    }).encode(), timeout=1, in_sandbox=True))
                if answer == {"result": {
                    "pid": 1, "uid": ["10001"] * 4, "gid": ["10001"] * 4,
                    "caps": ["0000000000000000"] * 5,
                    "no_new_privs": "1", "seccomp": "2"}}:
                    return {"generation": request["generation"], "ip": request["ip"],
                            "map_dir": request["map_dir"]}
            except (BrowserUnavailable, OSError, ValueError, RuntimeError):
                pass
        time.sleep(0.05)
    raise RuntimeError("browser worker PID 1 did not become ready")


def main():
    request = json.load(sys.stdin)
    print(json.dumps(launch(request), separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
