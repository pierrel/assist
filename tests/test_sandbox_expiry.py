"""Managed execution has a host deadline outside Docker's frozen cgroup."""
from datetime import datetime, timezone
import json
import logging
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
THREAD_ID = "synthetic-expiry-thread"


@pytest.mark.parametrize("thread_id", [None, THREAD_ID, "thread%u${HOME}\n\"雪"])
def test_host_timer_uses_original_creation_and_explicit_bus(monkeypatch, thread_id):
    created = datetime.now(timezone.utc).isoformat()
    calls = []
    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(stdout=created)
    monkeypatch.setattr(expiry.subprocess, "run", run)
    expiry.arm(GENERATION, thread_id=thread_id)
    command, kwargs = calls[1]
    script_index = command.index(str(Path(expiry.__file__).resolve()))
    assert command[script_index + 1:] == [GENERATION] + (
        [thread_id.encode("utf-8").hex()] if thread_id is not None else [])
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


@pytest.mark.parametrize("case,outcome,error", [
    ("removed", "removed", None),
    ("missing", "already_absent", None),
    ("remove_race", "already_absent", None),
    ("foreign", "ownership_rejected", ValueError),
    ("inspect_error", "inspect_failed", RuntimeError),
    ("remove_error", "removal_unconfirmed", RuntimeError),
    ("inspect_timeout", "inspect_failed", subprocess.TimeoutExpired),
    ("remove_timeout", "removal_unconfirmed", subprocess.TimeoutExpired),
])
def test_expiry_events_retain_identity_and_report_only_confirmed_outcomes(
        monkeypatch, caplog, case, outcome, error):
    owned = SimpleNamespace(returncode=0, stdout="true " + expiry.MARKER)
    absent = SimpleNamespace(returncode=1, stderr="Error: No such object: " + GENERATION)
    failed = SimpleNamespace(returncode=1, stderr="PRIVATE_DAEMON_RESPONSE")
    timeout = subprocess.TimeoutExpired("docker", 5, stderr="PRIVATE_TIMEOUT_RESPONSE")
    responses = {
        "removed": [owned, SimpleNamespace(returncode=0)],
        "missing": [absent], "remove_race": [owned, absent],
        "foreign": [SimpleNamespace(returncode=0, stdout="true PRIVATE_LABEL")],
        "inspect_error": [failed], "remove_error": [owned, failed],
        "inspect_timeout": [timeout], "remove_timeout": [owned, timeout],
    }
    run = Mock(side_effect=responses[case])
    monkeypatch.setattr(expiry.subprocess, "run", run)
    caplog.set_level(logging.INFO, logger=expiry.__name__)
    before = datetime.now(timezone.utc)
    if error:
        with pytest.raises(error):
            expiry.expire(GENERATION, thread_id=THREAD_ID)
    else:
        expiry.expire(GENERATION, thread_id=THREAD_ID)
    after = datetime.now(timezone.utc)
    records = [json.loads(record.message) for record in caplog.records]
    assert [record["outcome"] for record in records] == ["attempt", outcome]
    for record in records:
        timestamp = datetime.fromisoformat(record.pop("timestamp"))
        assert before <= timestamp <= after
        assert record == {"event": "sandbox_expiry", "generation": GENERATION,
                          "thread_id": THREAD_ID, "outcome": record["outcome"]}
    assert "PRIVATE" not in caplog.text
    assert all(call.args[0][-1] == GENERATION for call in run.call_args_list)
    if case in {"missing", "foreign", "inspect_error", "inspect_timeout"}:
        assert run.call_count == 1


@pytest.mark.parametrize("thread_id", [None, "thread%u${HOME}\n\"雪"])
def test_standalone_expiry_logs_after_container_is_gone(tmp_path, monkeypatch, thread_id):
    """The timer entry point retains identity without Docker or thread files."""
    docker = tmp_path / "docker"
    docker.write_text("#!/bin/sh\nprintf 'Error: No such object: %s\\n' \"$4\" >&2\nexit 1\n")
    docker.chmod(0o700)
    monkeypatch.setenv("PATH", str(tmp_path))
    result = subprocess.run([
        sys.executable, expiry.__file__, GENERATION,
        *([thread_id.encode("utf-8").hex()] if thread_id is not None else []),
    ], capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr
    records = [json.loads(line) for line in result.stderr.splitlines()]
    assert [record["outcome"] for record in records] == ["attempt", "already_absent"]
    assert all(record["thread_id"] == thread_id and record["generation"] == GENERATION
               for record in records)


@pytest.mark.skipif(os.getenv("ASSIST_BROWSER_DOCKER_TEST") != "1", reason="explicit isolated Docker/systemd test")
def test_host_deadline_removes_paused_container_after_owner_worker_exit():
    image = os.environ["ASSIST_BROWSER_TEST_SANDBOX_IMAGE"]
    cid = subprocess.check_output([
        "docker", "run", "-d", "--rm", "--network", "none",
        "--label", "assist.sandbox=true", "--label", expiry.LABEL + "=" + expiry.MARKER,
        "--entrypoint", "/bin/sleep", image, "60"], text=True, timeout=10).strip()
    try:
        worker = subprocess.run([sys.executable, "-c",
            f"from assist.sandbox_expiry import arm; arm({cid!r}, thread_id={THREAD_ID!r}, lifetime=4)"],
            capture_output=True, text=True, timeout=15)
        assert worker.returncode == 0, worker.stderr
        subprocess.run(["docker", "pause", cid], capture_output=True, check=True, timeout=5)
        time.sleep(5)
        probe = subprocess.run(["docker", "inspect", cid], capture_output=True, timeout=5)
        assert probe.returncode != 0, probe.stdout
        journal = subprocess.check_output([
            "journalctl", "--user", "-u", expiry._identity(cid) + ".service",
            "--no-pager", "-o", "cat"], env=expiry._environment(), text=True, timeout=5)
        events = [json.loads(line) for line in journal.splitlines()
                  if line.startswith('{"event": "sandbox_expiry"')]
        assert [event["outcome"] for event in events] == ["attempt", "removed"], journal
        assert all(event["thread_id"] == THREAD_ID and event["generation"] == cid
                   for event in events)
        print(json.dumps(events))
    finally:
        subprocess.run(["docker", "rm", "-f", cid], capture_output=True, timeout=5)
        expiry.cancel(cid)
