"""Bounded host setup for a browser process in one existing turn sandbox."""
from __future__ import annotations

import json
import ipaddress
import os
import sys
import time

from assist.browser import authority
from assist.egress import runtime_state
from assist.egress.client_map import ClientRecord, arm_sandbox_browser, read_client
from assist.run_service import RunService
from assist.sandbox_manager import (
    BROWSER_NETWORK, BROWSER_PROXY_PORT, EGRESS_NETWORK, EGRESS_PROXY_NAME,
    _proxy_setup_lock,
    SandboxManager)


def _endpoints(client, request):
    sandbox = client.containers.get(request["generation"])
    proxy = client.containers.get(EGRESS_PROXY_NAME)
    sandbox.reload()
    proxy.reload()
    sandbox_network = sandbox.attrs["NetworkSettings"]["Networks"][EGRESS_NETWORK]
    attachments = proxy.attrs["NetworkSettings"]["Networks"]
    proxy_network = attachments[EGRESS_NETWORK]
    browser_network = attachments[BROWSER_NETWORK]
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
            or not proxy_network["IPAddress"]
            or not browser_network["NetworkID"]):
        raise RuntimeError("browser sandbox or proxy identity changed")
    proxy_ips = set()
    for attached in attachments.values():
        if not isinstance(attached, dict) or not attached.get("NetworkID"):
            raise RuntimeError("browser proxy attachment is incomplete")
        try:
            ip = ipaddress.IPv4Address(attached["IPAddress"])
        except (KeyError, ValueError, ipaddress.AddressValueError) as error:
            raise RuntimeError("browser proxy attachment lacks IPv4") from error
        if not ip.is_private or ip.is_loopback or ip.is_link_local:
            raise RuntimeError("browser proxy attachment is invalid")
        proxy_ips.add(str(ip))
    runtime_state.assert_admissible("proxy", proxy.id)
    runtime_state.assert_admissible("network", proxy_network["NetworkID"])
    runtime_state.assert_admissible("network", browser_network["NetworkID"])
    return sandbox, proxy, proxy_network["IPAddress"], tuple(sorted(proxy_ips))


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
    """Fence UID, publish exact browser provenance, then launch."""
    from assist.browser.manager import _bounded_cli

    client = SandboxManager._get_docker_client()
    with _proxy_setup_lock():
        _, proxy, proxy_ip, proxy_ips = _endpoints(client, request)
        if read_client(request["map_dir"], request["ip"]) != ClientRecord(
                request["thread_id"], request["generation"], "sandbox"):
            raise RuntimeError("browser sandbox attribution changed")
        proxy_id = proxy.id
    _check_lease(request)
    _bounded_cli([
        "docker", "exec", "--privileged", "--user", "0", request["generation"],
        "/usr/bin/python", "-I", "/opt/assist/install-browser-fence.py",
        proxy_ip, *proxy_ips],
        limit=1024, timeout=15)
    with _proxy_setup_lock():
        sandbox, proxy, current_ip, current_ips = _endpoints(client, request)
        if (proxy.id != proxy_id or current_ip != proxy_ip
                or current_ips != proxy_ips):
            raise RuntimeError("browser proxy changed during network fence setup")
        _check_lease(request)
        arm_sandbox_browser(
            request["map_dir"], request["ip"],
            ClientRecord(request["thread_id"], request["generation"], "sandbox"))
    environment = {
        "BROWSER_SESSION_TOKEN": request["token"],
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
