"""Managed execution has a host deadline outside Docker's frozen cgroup."""
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from assist import sandbox_expiry as expiry

GENERATION = "a" * 64


def test_host_timer_uses_original_creation_and_explicit_bus(monkeypatch):
    created = datetime.now(timezone.utc).isoformat()
    calls = []
    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(stdout=created)
    monkeypatch.setattr(expiry.subprocess, "run", run)
    expiry.arm(GENERATION)
    command, kwargs = calls[1]
    assert command[-1] == GENERATION and command[-2] == str(Path(expiry.__file__).resolve())
    assert "--on-calendar" in command and "--on-active" not in command
    assert command[command.index("--on-calendar") + 1].endswith(" UTC")
    assert kwargs["env"]["DBUS_SESSION_BUS_ADDRESS"] == f"unix:path=/run/user/{os.geteuid()}/bus"
    assert kwargs["timeout"] == 10 and kwargs["check"]


@pytest.mark.parametrize("identity", ["name", "a" * 63, "b;id", "-" + "a" * 63])
def test_timer_rejects_nonimmutable_ids(monkeypatch, identity):
    run = Mock()
    monkeypatch.setattr(expiry.subprocess, "run", run)
    with pytest.raises(ValueError):
        expiry.arm(identity)
    run.assert_not_called()


def test_expiry_requires_owned_labels(monkeypatch):
    run = Mock(return_value=SimpleNamespace(returncode=0, stdout="true foreign", stderr=""))
    monkeypatch.setattr(expiry.subprocess, "run", run)
    with pytest.raises(ValueError, match="host-owned"):
        expiry.expire(GENERATION)
    assert run.call_count == 1


def test_timer_arm_failure_is_not_silently_accepted(monkeypatch):
    run = Mock(side_effect=[SimpleNamespace(stdout=datetime.now(timezone.utc).isoformat()),
                           subprocess.CalledProcessError(1, "systemd-run")])
    monkeypatch.setattr(expiry.subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError):
        expiry.arm(GENERATION)


@pytest.mark.parametrize("daemon_down", [False, True])
def test_expiry_normal_teardown_race_and_daemon_failure(monkeypatch, daemon_down):
    run = Mock(side_effect=[SimpleNamespace(returncode=0, stdout="true " + expiry.MARKER),
        SimpleNamespace(returncode=1, stderr="daemon unavailable" if daemon_down else
                        "Error response from daemon: No such container: " + GENERATION)])
    monkeypatch.setattr(expiry.subprocess, "run", run)
    if daemon_down:
        with pytest.raises(RuntimeError, match="unconfirmed"):
            expiry.expire(GENERATION)
    else:
        expiry.expire(GENERATION)


@pytest.mark.parametrize("during_removal", [False, True])
def test_missing_daemon_socket_is_not_missing_generation(monkeypatch, during_removal):
    error = SimpleNamespace(returncode=1,
        stderr="dial unix /tmp/missing.sock: connect: no such file or directory")
    responses = ([SimpleNamespace(returncode=0, stdout="true " + expiry.MARKER), error]
                 if during_removal else [error])
    monkeypatch.setattr(expiry.subprocess, "run", Mock(side_effect=responses))
    with pytest.raises(RuntimeError):
        expiry.expire(GENERATION)


def test_exact_missing_generation_converges(monkeypatch):
    run = Mock(return_value=SimpleNamespace(returncode=1,
        stderr="error: no such object: " + GENERATION))
    monkeypatch.setattr(expiry.subprocess, "run", run)
    expiry.expire(GENERATION)
    assert run.call_count == 1


@pytest.mark.skipif(os.getenv("ASSIST_BROWSER_DOCKER_TEST") != "1", reason="explicit isolated Docker/systemd test")
def test_host_deadline_removes_paused_container_after_owner_worker_exit():
    image = os.environ["ASSIST_BROWSER_TEST_SANDBOX_IMAGE"]
    cid = subprocess.check_output([
        "docker", "run", "-d", "--rm", "--network", "none",
        "--label", "assist.sandbox=true", "--label", expiry.LABEL + "=" + expiry.MARKER,
        "--entrypoint", "/bin/sleep", image, "60"], text=True, timeout=10).strip()
    try:
        worker = subprocess.run([sys.executable, "-c",
            f"from assist.sandbox_expiry import arm; arm({cid!r}, lifetime=4)"],
            capture_output=True, text=True, timeout=15)
        assert worker.returncode == 0, worker.stderr
        subprocess.run(["docker", "pause", cid], capture_output=True, check=True, timeout=5)
        time.sleep(5)
        probe = subprocess.run(["docker", "inspect", cid], capture_output=True, timeout=5)
        assert probe.returncode != 0, probe.stdout
    finally:
        subprocess.run(["docker", "rm", "-f", cid], capture_output=True, timeout=5)
        expiry.cancel(cid)
