"""Host age checks retain owned execution and log exact-generation outcomes."""
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from assist import sandbox_expiry as expiry

GENERATION = "a" * 64
THREAD_ID = "synthetic-expiry-thread"


@pytest.fixture
def scope(tmp_path):
    from assist.browser import authority
    (tmp_path / THREAD_ID).mkdir()
    authority.mark_new_thread(str(tmp_path), THREAD_ID)
    return {"threads_root": str(tmp_path), "thread_id": THREAD_ID}


@pytest.mark.parametrize("thread_id", [None, THREAD_ID, "thread%u${HOME}\n\"雪"])
def test_host_timer_uses_original_creation_and_explicit_bus(monkeypatch, tmp_path, thread_id):
    created = datetime.now(timezone.utc).isoformat()
    calls = []
    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(stdout=created)
    monkeypatch.setattr(expiry.subprocess, "run", run)
    expiry.arm(GENERATION, thread_id=thread_id, threads_root=str(tmp_path))
    command, kwargs = calls[1]
    script_index = command.index(str(Path(expiry.__file__).resolve()))
    assert command[script_index + 1:] == [GENERATION] + (
        [thread_id.encode("utf-8").hex(), str(tmp_path).encode("utf-8").hex()]
        if thread_id is not None else [])
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
def test_expiry_normal_teardown_race_and_daemon_failure(monkeypatch, scope, daemon_down):
    run = Mock(side_effect=[SimpleNamespace(returncode=0, stdout="true " + expiry.MARKER),
        SimpleNamespace(returncode=1, stderr="daemon unavailable" if daemon_down else
                        "Error response from daemon: No such container: " + GENERATION)])
    monkeypatch.setattr(expiry.subprocess, "run", run)
    if daemon_down:
        with pytest.raises(RuntimeError, match="unconfirmed"):
            expiry.expire(GENERATION, **scope)
    else:
        expiry.expire(GENERATION, **scope)


@pytest.mark.parametrize("during_removal", [False, True])
def test_missing_daemon_socket_is_not_missing_generation(monkeypatch, scope, during_removal):
    error = SimpleNamespace(returncode=1,
        stderr="dial unix /tmp/missing.sock: connect: no such file or directory")
    responses = ([SimpleNamespace(returncode=0, stdout="true " + expiry.MARKER), error]
                 if during_removal else [error])
    monkeypatch.setattr(expiry.subprocess, "run", Mock(side_effect=responses))
    with pytest.raises(RuntimeError):
        expiry.expire(GENERATION, **scope)


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
        monkeypatch, caplog, scope, case, outcome, error):
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
            expiry.expire(GENERATION, **scope)
    else:
        expiry.expire(GENERATION, **scope)
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


def test_standalone_retains_scoped_owner_without_import_environment(tmp_path, monkeypatch, scope):
    from assist.browser import authority
    with authority.fence(scope["threads_root"], THREAD_ID) as state:
        state.begin("synthetic-owner", 1)
        state.add_generation("synthetic-owner", GENERATION)
    docker = tmp_path / "docker"
    docker.write_text("#!/bin/sh\n[ \"$1\" = inspect ] || exit 42\nprintf 'true host-3h-v1\\n'\n")
    docker.chmod(0o700)
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.delenv("PYTHONPATH", raising=False)
    result = subprocess.run([
        sys.executable, "-I", expiry.__file__, GENERATION,
        THREAD_ID.encode().hex(), scope["threads_root"].encode().hex(),
    ], cwd="/", capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr
    records = [json.loads(line) for line in result.stderr.splitlines()]
    assert [record["outcome"] for record in records] == ["attempt", "retained_owned"]


@pytest.mark.parametrize("stage", ["running", "pending", "fair_pause", "queued_child", "running_child"])
def test_owned_generation_survives_age_check_through_logical_work(
        monkeypatch, caplog, scope, stage):
    from assist.browser import authority
    from assist.run_service import RunService
    runs = RunService(scope["threads_root"])
    parent = runs.create(THREAD_ID, "general-agent", "Synthetic work", user_origin=True)
    if stage != "pending":
        runs.claim(THREAD_ID, parent.id)
    if stage in {"fair_pause", "queued_child", "running_child"}:
        runs.transition(THREAD_ID, parent.id, "interrupted")
        runs.create(THREAD_ID, "general-agent", None, work_id=parent.work_id,
                    resume=True, user_event_id=parent.id)
    if stage in {"queued_child", "running_child"}:
        child = runs.create("sub-synthetic", "delegate-agent", "Synthetic child", mode="child",
                            parent_thread_id=THREAD_ID, parent_run_id=parent.id,
                            dispatch_key="synthetic-child")
        if stage == "running_child":
            runs.claim(child.thread_id, child.id)
    with authority.fence(scope["threads_root"], scope["thread_id"]) as state:
        state.begin(parent.id, parent.admission_sequence)
        state.add_generation(parent.id, GENERATION)
    run = Mock(return_value=SimpleNamespace(returncode=0, stdout="true " + expiry.MARKER))
    monkeypatch.setattr(expiry.subprocess, "run", run)
    caplog.set_level(logging.INFO, logger=expiry.__name__)
    expiry.expire(GENERATION, **scope)
    assert run.call_count == 1
    event = json.loads(caplog.records[-1].message)
    assert event["outcome"] == "retained_owned"
    assert event["thread_id"] == THREAD_ID and event["generation"] == GENERATION
    with authority.fence(scope["threads_root"], scope["thread_id"]) as state:
        assert state.lease["generations"] == [GENERATION]


@pytest.mark.parametrize("terminal", ["success", "cancelled"])
def test_normal_terminal_cleanup_stops_retained_generation_before_clearing_owner(
        monkeypatch, scope, terminal):
    from assist.browser import authority
    from assist.run_service import RunService
    from assist.sandbox_manager import SandboxManager
    runs = RunService(scope["threads_root"])
    parent = runs.create(THREAD_ID, "general-agent", "Synthetic work", user_origin=True)
    runs.claim(THREAD_ID, parent.id)
    with authority.fence(scope["threads_root"], scope["thread_id"]) as state:
        state.begin(parent.id, parent.admission_sequence)
        state.add_generation(parent.id, GENERATION)
    monkeypatch.setattr(expiry.subprocess, "run", Mock(return_value=SimpleNamespace(
        returncode=0, stdout="true " + expiry.MARKER)))
    expiry.expire(GENERATION, **scope)
    runs.transition(THREAD_ID, parent.id, terminal)
    work = str(Path(scope["threads_root"]) / THREAD_ID / "domain")
    container = SimpleNamespace(id=GENERATION)
    monkeypatch.setattr(SandboxManager, "_containers", {work: container})
    monkeypatch.setattr(SandboxManager, "_generation_owners", {
        GENERATION: (scope["threads_root"], THREAD_ID, parent.id)})
    monkeypatch.setattr(SandboxManager, "_forget_egress_client", lambda *_a: None)
    stopped = []
    def stop(generation):
        with authority.fence(scope["threads_root"], scope["thread_id"]) as state:
            assert generation in state.lease["generations"]
        stopped.append(generation)
    monkeypatch.setattr("assist.sandbox_manager.confirm_generation_stopped", stop)
    SandboxManager.cleanup(work, container)
    assert stopped == [GENERATION] and not SandboxManager._containers
    with authority.fence(scope["threads_root"], scope["thread_id"]) as state:
        assert state.lease is None


@pytest.mark.parametrize("problem", ["legacy", "missing_thread", "uncovered", "corrupt"])
def test_unknown_ownership_is_retained(monkeypatch, caplog, scope, problem):
    from assist.browser import authority
    directory = Path(scope["threads_root"]) / THREAD_ID
    if problem == "missing_thread":
        scope = {**scope, "thread_id": "missing-thread"}
    elif problem == "uncovered":
        (directory / authority.STATE_FILE).unlink()
    elif problem == "corrupt":
        (directory / authority.STATE_FILE).write_text("not json")
    elif problem == "legacy":
        scope.pop("threads_root")
    run = Mock(return_value=SimpleNamespace(returncode=0, stdout="true " + expiry.MARKER))
    monkeypatch.setattr(expiry.subprocess, "run", run)
    caplog.set_level(logging.INFO, logger=expiry.__name__)
    expiry.expire(GENERATION, **scope)
    assert run.call_count == 1
    assert json.loads(caplog.records[-1].message)["outcome"] == "retained_unknown"


def test_age_check_waits_for_inflight_generation_admission(monkeypatch, caplog, scope):
    """Real file locks exclude the timer while creation publishes its ownership."""
    from assist.browser import authority
    inspected = threading.Event()
    errors = []
    def docker(argv, **_kwargs):
        assert argv[1] == "inspect", "expiry removed the newly owned generation"
        inspected.set()
        return SimpleNamespace(returncode=0, stdout="true " + expiry.MARKER)
    monkeypatch.setattr(expiry.subprocess, "run", docker)
    def expire():
        try:
            expiry.expire(GENERATION, **scope)
        except BaseException as error:
            errors.append(error)
    with authority.generation_fence(scope["threads_root"], scope["thread_id"]):
        worker = threading.Thread(target=expire)
        worker.start()
        assert inspected.wait(2)
        with authority.fence(scope["threads_root"], scope["thread_id"]) as state:
            state.begin("synthetic-owner", 1)
            state.add_generation("synthetic-owner", GENERATION)
    worker.join(2)
    assert not worker.is_alive() and errors == []


def test_ownerless_removal_holds_both_existing_fences(monkeypatch, scope):
    from assist.browser import authority
    calls = []
    def docker(argv, **_kwargs):
        calls.append(argv[1])
        if argv[1] == "inspect":
            return SimpleNamespace(returncode=0, stdout="true " + expiry.MARKER)
        for fence in (authority.generation_fence, authority.fence):
            with pytest.raises(TimeoutError):
                with fence(scope["threads_root"], scope["thread_id"], timeout=0):
                    pytest.fail("ownership could change during exact removal")
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(expiry.subprocess, "run", docker)
    expiry.expire(GENERATION, **scope)
    assert calls == ["inspect", "rm"]


@pytest.mark.skipif(os.getenv("ASSIST_BROWSER_DOCKER_TEST") != "1", reason="explicit isolated Docker/systemd test")
def test_managed_pid1_and_host_age_check_preserve_paused_work_then_cleanup(
        tmp_path, monkeypatch, scope):
    """The actual factory overrides the image TTL; normal cleanup owns final stop."""
    import docker
    from assist import sandbox_manager as sandbox_module
    from assist.browser import authority
    from assist.browser.manager import BrowserManager
    from assist.run_service import RunService
    from assist.sandbox_manager import SandboxManager
    root = scope["threads_root"]
    work = str(Path(root) / THREAD_ID / "domain")
    Path(work).mkdir()
    runs = RunService(root)
    owner = runs.create(THREAD_ID, "general-agent", "Synthetic retained work", user_origin=True)
    runs.claim(THREAD_ID, owner.id)
    client = docker.from_env(timeout=5)
    monkeypatch.setattr(SandboxManager, "_containers", {})
    monkeypatch.setattr(SandboxManager, "_generation_owners", {})
    monkeypatch.setattr(SandboxManager, "_get_docker_client", lambda: client)
    monkeypatch.setattr(SandboxManager, "_ensure_egress_proxy_running", lambda *_a: None)
    monkeypatch.setattr(SandboxManager, "_record_egress_client", lambda *_a, **_k: None)
    monkeypatch.setattr(SandboxManager, "_forget_egress_client", lambda *_a, **_k: None)
    monkeypatch.setattr(BrowserManager, "reconcile_startup", lambda *_a: True)
    monkeypatch.setattr(BrowserManager, "confirm_owner_stopped", lambda *_a, **_k: False)
    monkeypatch.setattr(sandbox_module, "SANDBOX_IMAGE", os.environ["ASSIST_BROWSER_TEST_SANDBOX_IMAGE"])
    monkeypatch.setattr(sandbox_module, "EGRESS_NETWORK", "none")
    def arm_in_worker(generation, **kwargs):
        worker = subprocess.run([sys.executable, "-c",
            f"from assist.sandbox_expiry import arm; arm({generation!r}, **{kwargs!r}, lifetime=4)"],
            capture_output=True, text=True, timeout=15)
        assert worker.returncode == 0, worker.stderr
    monkeypatch.setattr(expiry, "arm", arm_in_worker)
    backend = SandboxManager.get_sandbox_backend(
        work, thread_scope=(root, THREAD_ID), owner_run_id=owner.id)
    container = backend.container
    try:
        container.reload()
        assert container.attrs["Config"]["Cmd"] == ["sleep", "infinity"]
        container.pause()
        runs.transition(THREAD_ID, owner.id, "interrupted")
        successor = runs.create(THREAD_ID, "general-agent", None,
            work_id=owner.work_id, resume=True, user_event_id=owner.id)
        time.sleep(5)
        container.reload()
        assert container.status == "paused"
        journal = subprocess.check_output([
            "journalctl", "--user", "-u", expiry._identity(container.id) + ".service",
            "--no-pager", "-o", "cat"], env=expiry._environment(), text=True, timeout=5)
        events = [json.loads(line) for line in journal.splitlines()
                  if line.startswith('{"event": "sandbox_expiry"')]
        assert [event["outcome"] for event in events] == ["attempt", "retained_owned"], journal
        assert all(event["thread_id"] == THREAD_ID and event["generation"] == container.id
                   for event in events)
        runs.cancel(THREAD_ID, successor.id)
        SandboxManager.cleanup(work, container)
        with pytest.raises(docker.errors.NotFound):
            client.containers.get(container.id)
        with authority.fence(root, THREAD_ID) as state:
            assert state.lease is None
        print(json.dumps({"events": events, "pid1": ["sleep", "infinity"],
                          "normal_cancel_cleanup": "removed"}))
    finally:
        SandboxManager.cleanup(work, container)


@pytest.mark.skipif(os.getenv("ASSIST_BROWSER_DOCKER_TEST") != "1", reason="explicit isolated Docker/systemd test")
def test_host_age_check_removes_ownerless_paused_container_after_worker_exit(scope):
    image = os.environ["ASSIST_BROWSER_TEST_SANDBOX_IMAGE"]
    cid = subprocess.check_output([
        "docker", "run", "-d", "--rm", "--network", "none",
        "--label", "assist.sandbox=true", "--label", expiry.LABEL + "=" + expiry.MARKER,
        "--entrypoint", "/bin/sleep", image, "60"], text=True, timeout=10).strip()
    try:
        worker = subprocess.run([sys.executable, "-c",
            f"from assist.sandbox_expiry import arm; arm({cid!r}, **{scope!r}, lifetime=4)"],
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


@pytest.mark.skipif(os.getenv("ASSIST_BROWSER_DOCKER_TEST") != "1", reason="explicit isolated Docker/systemd test")
def test_retained_age_check_then_worker_crash_is_reclaimed_at_startup(tmp_path, monkeypatch):
    """Retention does not prevent ordinary fresh-process orphan reconciliation."""
    import select
    import signal
    from textwrap import dedent
    from assist.browser import authority

    root = tmp_path / "threads"
    work = root / THREAD_ID / "domain"
    work.mkdir(parents=True)
    authority.mark_new_thread(str(root), THREAD_ID)
    monkeypatch.setenv("ASSIST_THREADS_DIR", str(root))
    monkeypatch.setenv("ASSIST_EGRESS_CLIENT_MAP_DIR", str(tmp_path / "client-map"))
    cid = subprocess.check_output([
        "docker", "run", "-d", "--rm", "--network", "none",
        "--mount", f"type=bind,src={work},dst=/workspace",
        "--label", "assist.sandbox=true", "--label", expiry.LABEL + "=" + expiry.MARKER,
        "--entrypoint", "/bin/sleep", os.environ["ASSIST_BROWSER_TEST_SANDBOX_IMAGE"],
        "infinity"], text=True, timeout=10).strip()
    worker = None
    try:
        worker = subprocess.Popen([sys.executable, "-c", dedent("""
            import json, sys
            from assist.browser import authority
            from assist.run_service import RunService
            from assist.sandbox_expiry import arm
            root, thread, generation = sys.argv[1:]
            runs = RunService(root)
            owner = runs.create(thread, 'general-agent', 'Synthetic crash work', user_origin=True)
            runs.claim(thread, owner.id)
            with authority.generation_fence(root, thread), authority.fence(root, thread) as state:
                state.begin(owner.id, owner.admission_sequence)
                state.add_generation(owner.id, generation)
            arm(generation, thread_id=thread, threads_root=root, lifetime=4)
            print(json.dumps({'owner_run_id': owner.id, 'generation': generation}), flush=True)
            sys.stdin.read()
        """), str(root), THREAD_ID, cid], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        assert select.select([worker.stdout], [], [], 15)[0], "owner did not arm its timer"
        owner = json.loads(worker.stdout.readline())
        assert owner["generation"] == cid
        time.sleep(5)
        assert worker.poll() is None
        assert subprocess.check_output([
            "docker", "inspect", "--format", "{{.State.Running}}", cid],
            text=True, timeout=5).strip() == "true"
        journal = subprocess.check_output([
            "journalctl", "--user", "-u", expiry._identity(cid) + ".service",
            "--no-pager", "-o", "cat"], env=expiry._environment(), text=True, timeout=5)
        events = [json.loads(line) for line in journal.splitlines()
                  if line.startswith('{"event": "sandbox_expiry"')]
        assert [event["outcome"] for event in events] == ["attempt", "retained_owned"], journal
        assert all(event["thread_id"] == THREAD_ID and event["generation"] == cid
                   for event in events)
        worker.kill()
        assert worker.wait(timeout=5) == -signal.SIGKILL
        assert subprocess.check_output([
            "docker", "inspect", "--format", "{{.State.Running}}", cid],
            text=True, timeout=5).strip() == "true"
        with authority.fence(str(root), THREAD_ID) as state:
            assert state.lease["owner_run_id"] == owner["owner_run_id"]
            assert state.lease["generations"] == [cid]
        restarted = subprocess.run([sys.executable, "-c", dedent("""
            import json, sys
            from assist.browser import authority
            from assist.browser.manager import BrowserManager
            from assist.sandbox_manager import SandboxManager
            root, thread = sys.argv[1:]
            assert not SandboxManager._containers
            SandboxManager.reap_orphans(root)
            assert BrowserManager.reconcile_startup(root)
            with authority.fence(root, thread) as state:
                assert state.covered and state.lease is None
            print(json.dumps({'startup_cleanup': 'reconciled', 'owner': None}))
        """), str(root), THREAD_ID], capture_output=True, text=True, timeout=30)
        assert restarted.returncode == 0, restarted.stderr
        missing = subprocess.run(["docker", "inspect", cid], capture_output=True, text=True, timeout=5)
        assert missing.returncode != 0 and cid in missing.stderr and "no such" in missing.stderr.lower()
        print(json.dumps({"events": events, "crashed_owner": owner,
                          "restart": json.loads(restarted.stdout), "removed_generation": cid}))
    finally:
        if worker is not None:
            if worker.poll() is None:
                worker.kill()
            worker.communicate(timeout=5)
        subprocess.run(["docker", "rm", "-f", cid], capture_output=True, timeout=5)
        expiry.cancel(cid)
