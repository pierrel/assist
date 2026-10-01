"""Host browser transport and exact Docker teardown stay bounded."""
import fcntl
import json
import os
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from assist.browser import manager as browser
from assist.browser import authority
from assist.browser import storage
from assist.run_service import RunService


def test_nonreading_child_with_small_pipe_cannot_block_stdin_write(monkeypatch):
    popen = browser.subprocess.Popen
    children = []

    def small_pipe(*args, **kwargs):
        child = popen(*args, **kwargs)
        fcntl.fcntl(child.stdin.fileno(), fcntl.F_SETPIPE_SZ, 4096)
        assert fcntl.fcntl(child.stdin.fileno(), fcntl.F_GETPIPE_SZ) == 4096
        children.append(child)
        return child

    monkeypatch.setattr(browser.subprocess, "Popen", small_pipe)
    started = time.monotonic()
    for _ in range(2):
        with pytest.raises(browser.BrowserUnavailable, match="timed out"):
            browser._bounded_cli(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                payload=b"x" * 8192, limit=128, timeout=0.15)
    assert time.monotonic() - started < 2
    assert all(child.poll() is not None for child in children)


def test_repeated_transport_timeouts_reap_children_and_leave_no_threads(tmp_path):
    before = threading.active_count()
    for index in range(6):
        pidfile = tmp_path / f"child-{index}"
        script = ("import os,sys,time; "
                  "open(sys.argv[1],'w').write(str(os.getpid())); "
                  "sys.stdin.buffer.read(); time.sleep(30)")
        with pytest.raises(browser.BrowserUnavailable, match="timed out"):
            browser._bounded_cli(
                [sys.executable, "-c", script, str(pidfile)],
                payload=b"request", limit=128, timeout=0.2)
        assert pidfile.exists()
        with pytest.raises(ProcessLookupError):
            os.kill(int(pidfile.read_text()), 0)
    assert threading.active_count() == before


def test_transport_rejects_oversized_child_output():
    with pytest.raises(browser.BrowserUnavailable, match="exceeds limit"):
        browser._bounded_cli(
            [sys.executable, "-c", "import sys; sys.stdout.write('x'*1000000)"],
            limit=64, timeout=2)


def test_stdout_eof_without_process_exit_obeys_transport_deadline():
    started = time.monotonic()
    with pytest.raises(browser.BrowserUnavailable, match="timed out"):
        browser._bounded_cli(
            [sys.executable, "-c", "import os,time; os.close(1); time.sleep(30)"],
            limit=64, timeout=0.15)
    assert time.monotonic() - started < 2


def test_auto_removed_sidecar_inspect_is_confirmed_with_lowercase_docker_error(
        monkeypatch):
    calls = []

    def docker_run(argv, **_kwargs):
        calls.append(argv[:2])
        if argv[1] == "kill":
            return browser.subprocess.CompletedProcess(argv, 1, b"", b"")
        return browser.subprocess.CompletedProcess(
            argv, 1, b"\n", b"error: no such object: generation\n")

    monkeypatch.setattr(browser.subprocess, "run", docker_run)
    browser._kill_container_confirmed("generation")
    assert calls == [["docker", "kill"], ["docker", "inspect"]]


def test_uncertain_turn_startup_cannot_repeat_in_one_run(
        monkeypatch, tmp_path):
    (tmp_path / "thread").mkdir()
    authority.mark_new_thread(str(tmp_path), "thread")
    run = RunService(str(tmp_path)).create(
        "thread", "general-agent", "Read a public page", user_origin=True)
    session = browser.BrowserSession(
        "thread", run.id, str(tmp_path), str(tmp_path), None,
        sandbox_generation="generation")
    monkeypatch.setattr(browser, "configured_directory", lambda: str(tmp_path))
    monkeypatch.setattr(browser.SandboxManager, "current_container",
                        lambda _work_dir: SimpleNamespace(id="generation"))
    monkeypatch.setattr(browser.SandboxManager, "_egress_client_ips", {
        str(tmp_path): (str(tmp_path), "172.20.0.9", "generation")})
    monkeypatch.setattr(browser, "_kill_container_confirmed", lambda _gen: None)
    monkeypatch.setattr(browser, "forget_client", lambda *_args: None)
    monkeypatch.setattr(browser, "disarm_sandbox_browser", lambda *_args: None)
    with authority.fence(str(tmp_path), "thread") as state:
        state.begin(run.id, run.admission_sequence)
        state.add_generation(run.id, "generation")
    launches = []

    def timed_out(*_args, **_kwargs):
        launches.append(1)
        raise browser.BrowserUnavailable("browser transport timed out")

    monkeypatch.setattr(browser, "_bounded_cli", timed_out)
    with pytest.raises(browser.BrowserUnavailable, match="startup failed"):
        session.command("open", url="https://example.com/")
    with pytest.raises(browser.BrowserUnavailable, match="outcome is unconfirmed"):
        session.command("open", url="https://example.com/")
    assert launches == [1]


def test_turn_session_rejects_replaced_sandbox_generation(monkeypatch, tmp_path):
    session = browser.BrowserSession(
        "thread", "run", str(tmp_path), str(tmp_path), None,
        sandbox_generation="original")
    monkeypatch.setattr(browser.SandboxManager, "current_container",
                        lambda _work_dir: SimpleNamespace(id="replacement"))
    monkeypatch.setattr(browser.SandboxManager, "_egress_client_ips", {
        str(tmp_path): (str(tmp_path), "172.20.0.9", "replacement")})
    monkeypatch.setattr(browser, "configured_directory", lambda: str(tmp_path))
    with pytest.raises(browser.BrowserUnavailable, match="identity changed"):
        session._start()


def test_invalid_private_snapshot_loses_continuity_without_blocking_browser(
        monkeypatch, tmp_path):
    (tmp_path / "thread").mkdir()
    authority.mark_new_thread(str(tmp_path), "thread")
    runs = RunService(str(tmp_path))
    run = runs.create("thread", "general-agent", "Read a public page",
                      user_origin=True)
    session = browser.BrowserSession(
        "thread", run.id, str(tmp_path), str(tmp_path), None,
        run_service=runs, sandbox_generation="generation")
    with authority.fence(str(tmp_path), "thread") as state:
        state.begin(run.id, run.admission_sequence)
        state.add_generation(run.id, "generation")
    monkeypatch.setattr(browser.SandboxManager, "current_container",
                        lambda _work_dir: SimpleNamespace(id="generation"))
    monkeypatch.setattr(browser.SandboxManager, "_egress_client_ips", {
        str(tmp_path): (str(tmp_path), "172.20.0.9", "generation")})
    monkeypatch.setattr(browser, "configured_directory", lambda: str(tmp_path))
    monkeypatch.setattr(browser, "_bounded_cli", lambda *_args, **_kwargs:
                        json.dumps({"generation": "generation", "ip": "172.20.0.9",
                                    "map_dir": str(tmp_path)}).encode())
    monkeypatch.setattr(storage, "load", lambda *_args: (_ for _ in ()).throw(
        ValueError("damaged snapshot")))
    monkeypatch.setattr(browser, "_docker_exec", lambda *_args, **_kwargs:
                        b'{"result":null}')
    monkeypatch.setattr(browser, "_kill_container_confirmed", lambda _id: None)
    monkeypatch.setattr(browser, "forget_client", lambda *_args: None)
    monkeypatch.setattr(browser, "disarm_sandbox_browser", lambda *_args: None)
    session._start()
    assert session.identity.generation == "generation"
    session.close()


def test_failed_stop_keeps_identity_and_disarms_before_kill(monkeypatch, tmp_path):
    session = browser.BrowserSession(
        "thread", "run", str(tmp_path), str(tmp_path), None)
    session.identity = browser._ContainerIdentity(
        str(tmp_path), "172.20.0.9", "generation")
    events = []
    monkeypatch.setattr(browser, "disarm_sandbox_browser",
                        lambda *_args: events.append("disarm"))
    monkeypatch.setattr(browser, "_kill_container_confirmed", lambda _id: (
        events.append("kill") or (_ for _ in ()).throw(
            browser.BrowserUnavailable("stop unconfirmed"))))
    with pytest.raises(browser.BrowserUnavailable, match="stop unconfirmed"):
        session._stop()
    assert events == ["disarm", "kill"]
    assert session.identity.generation == "generation"


def test_failed_command_stops_generation_before_releasing_session(monkeypatch, tmp_path):
    (tmp_path / "thread").mkdir()
    authority.mark_new_thread(str(tmp_path), "thread")
    runs = RunService(str(tmp_path))
    run = runs.create("thread", "general-agent", "Read a page", user_origin=True)
    with authority.fence(str(tmp_path), "thread") as state:
        state.begin(run.id, run.admission_sequence)
        state.add_generation(run.id, "generation")
    session = browser.BrowserSession(
        "thread", run.id, str(tmp_path), str(tmp_path), None,
        run_service=runs, sandbox_generation="generation")
    session.identity = browser._ContainerIdentity(
        str(tmp_path), "172.20.0.9", "generation")
    killed = []
    monkeypatch.setattr(browser, "_docker_exec",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(
                            browser.BrowserUnavailable("browser transport timed out")))
    monkeypatch.setattr(browser, "_kill_container_confirmed", killed.append)
    monkeypatch.setattr(browser, "disarm_sandbox_browser", lambda *_args: None)
    monkeypatch.setattr(browser, "forget_client", lambda *_args: None)
    with pytest.raises(browser.BrowserUnavailable, match="timed out"):
        session.command("observe", page_id="old")
    assert killed == ["generation"]
    assert session.identity is None


def test_failed_attribution_removal_still_kills_and_keeps_retry_identity(
        monkeypatch, tmp_path):
    session = browser.BrowserSession("thread", "run", str(tmp_path), str(tmp_path), None)
    session.identity = browser._ContainerIdentity(
        str(tmp_path), "172.20.0.9", "generation")
    killed = []
    monkeypatch.setattr(browser, "forget_client",
                        lambda *_: (_ for _ in ()).throw(OSError("read-only map")))
    monkeypatch.setattr(browser, "_kill_container_confirmed", killed.append)
    with pytest.raises(browser.BrowserUnavailable, match="attribution teardown"):
        session.close()
    assert killed == ["generation"]
    assert session.identity.generation == "generation"


def test_historical_deadline_fields_are_ignored_on_read(tmp_path):
    (tmp_path / "thread").mkdir()
    runs = RunService(str(tmp_path))
    first = runs.create("thread", "general-agent", "Open a page", user_origin=True)
    raw = first.to_dict()
    raw.update(browser_boot_id="old-boot", browser_deadline_ns=1)
    restored = type(first).from_dict(raw)
    assert "browser_boot_id" not in restored.to_dict()
    assert "browser_deadline_ns" not in restored.to_dict()


def test_command_transport_uses_fixed_operation_bound(
        monkeypatch, tmp_path):
    (tmp_path / "thread").mkdir()
    authority.mark_new_thread(str(tmp_path), "thread")
    runs = RunService(str(tmp_path))
    run = runs.create("thread", "general-agent", "Read a page", user_origin=True)
    with authority.fence(str(tmp_path), "thread") as state:
        state.begin(run.id, run.admission_sequence)
        state.add_generation(run.id, "generation")
    session = browser.BrowserSession(
        "thread", run.id, str(tmp_path), str(tmp_path), None,
        run_service=runs, sandbox_generation="generation")
    session.identity = browser._ContainerIdentity(
        str(tmp_path), "172.20.0.9", "generation")
    observed = []
    monkeypatch.setattr(browser, "_docker_exec",
                        lambda _gen, _cmd, *, timeout, in_sandbox: (
                            observed.append(timeout) or b'{"result": {}}'))
    assert session.command("observe", page_id="p1") == {"result": {}}
    assert observed == [20]
    assert session.identity.generation == "generation"


def test_missing_thread_directory_still_stops_exact_turn_sandbox(
        monkeypatch, tmp_path):
    stopped = []
    monkeypatch.setattr(browser, "configured_directory", lambda: None)
    monkeypatch.setattr(browser, "_sandbox_generations",
                        lambda root, tid: {"sandbox-generation"})
    monkeypatch.setattr(browser, "_bounded_cli", lambda *_a, **_k: b"")
    monkeypatch.setattr(browser, "_kill_container_confirmed", stopped.append)
    monkeypatch.setattr(browser.SandboxManager, "_ensure_egress_proxy_bounded",
                        lambda: None)

    assert browser.BrowserManager.confirm_owner_stopped(
        str(tmp_path), "deleted-thread", None) is True
    assert stopped == ["sandbox-generation"]


def test_sandbox_generation_scan_matches_exact_thread_mount(
        monkeypatch, tmp_path):
    def docker(argv, **_kwargs):
        if argv[1] == "ps":
            return b"target\nother\n"
        assert argv[1] == "inspect"
        return (json.dumps([{"Destination": "/workspace",
                             "Source": str(tmp_path / "target-thread" / "domain")}])
                + "\n" + json.dumps([{"Destination": "/workspace",
                                         "Source": str(tmp_path / "other" / "domain")}])
                + "\n").encode()

    monkeypatch.setattr(browser, "_bounded_cli", docker)
    assert browser._sandbox_generations(
        str(tmp_path), "target-thread") == {"target"}


def test_turn_browser_transport_uses_private_socket_and_download_dir(monkeypatch):
    commands = []
    monkeypatch.setattr(browser, "_bounded_cli",
                        lambda argv, **_kwargs: commands.append(argv) or b"{}")
    browser._docker_exec("generation", b"{}", in_sandbox=True)
    browser._docker_download(
        "generation", "/run/assist-browser/downloads/report.txt",
        in_sandbox=True)
    assert ["-e", "BROWSER_RUNTIME_DIR=/run/assist-browser"] == commands[0][
        commands[0].index("-e"):commands[0].index("-e") + 2]
    assert ["-e", "BROWSER_DOWNLOAD_DIR=/run/assist-browser/downloads"] == (
        commands[1][commands[1].index("-e"):commands[1].index("-e") + 2])
