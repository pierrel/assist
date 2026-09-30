"""The browser UID alone may use its dedicated proxy endpoint."""
import importlib.util
from pathlib import Path

import pytest


def _fence():
    source = (Path(__file__).parents[1] / "dockerfiles"
              / "install-browser-fence.py")
    spec = importlib.util.spec_from_file_location("browser_uid_fence", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_uid_fence_denies_every_other_ipv4_and_all_ipv6(monkeypatch):
    fence = _fence()
    calls = []
    monkeypatch.setattr(fence, "run", lambda _deadline, *args: calls.append(args))
    fence.install("172.20.0.3")
    assert ("/usr/bin/iptables", "-w", "3", "-A", "ASSIST_BROWSER", "-d",
            "172.20.0.3", "-p", "tcp", "--dport", "8889",
            "-j", "RETURN") in calls
    assert ("/usr/bin/iptables", "-w", "3", "-A", "ASSIST_BROWSER",
            "-j", "REJECT") in calls
    assert ("/usr/bin/iptables", "-w", "3", "-I", "OUTPUT", "1", "-m",
            "owner", "--uid-owner", "10001", "-j", "ASSIST_BROWSER") in calls
    assert ("/usr/bin/iptables", "-w", "3", "-I", "OUTPUT", "1", "-d",
            "172.20.0.3", "-p", "tcp", "--dport", "8889", "-m", "owner",
            "!", "--uid-owner", "10001", "-j", "REJECT") in calls
    assert ("/usr/bin/ip6tables", "-w", "3", "-I", "OUTPUT", "1", "-m",
            "owner", "--uid-owner", "10001", "-j", "REJECT") in calls
    assert calls.index(("/usr/bin/iptables", "-w", "3", "-A", "ASSIST_BROWSER",
                        "-j", "REJECT")) < calls.index((
                            "/usr/bin/iptables", "-w", "3", "-I", "OUTPUT", "1",
                            "-m", "owner", "--uid-owner", "10001", "-j",
                            "ASSIST_BROWSER"))


@pytest.mark.parametrize("address", ["127.0.0.1", "169.254.1.2", "8.8.8.8",
                                     "::1", "bad-ip"])
def test_uid_fence_rejects_non_bridge_endpoint(address):
    with pytest.raises(ValueError):
        _fence().install(address)
