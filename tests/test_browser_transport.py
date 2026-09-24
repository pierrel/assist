"""Host transport remains bounded under repeated Docker/sidecar stalls."""
import fcntl
import os
import sys
import threading
import time

import pytest

from assist.browser import manager as browser


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


def test_uncertain_startup_cannot_accumulate_sidecars_in_one_run(
        monkeypatch, tmp_path):
    session = browser.BrowserSession("thread", "run", str(tmp_path), str(tmp_path), None)
    monkeypatch.setattr(browser, "configured_directory", lambda: str(tmp_path))
    monkeypatch.setattr(browser, "_load_egress_allowlist", lambda: [])
    launches = []

    def timed_out(*_args, **_kwargs):
        launches.append(1)
        raise browser.BrowserUnavailable("browser transport timed out")

    monkeypatch.setattr(browser, "_launch_sidecar_bounded", timed_out)
    with pytest.raises(browser.BrowserUnavailable, match="timed out"):
        session.command("open", url="https://example.com/")
    with pytest.raises(browser.BrowserUnavailable, match="outcome is unconfirmed"):
        session.command("open", url="https://example.com/")
    assert launches == [1]


def test_failed_command_kills_sidecar_before_releasing_session(monkeypatch, tmp_path):
    session = browser.BrowserSession("thread", "run", str(tmp_path), str(tmp_path), None)
    session.identity = browser._ContainerIdentity(
        None, str(tmp_path), "172.20.0.9", "generation", "public", None)
    killed = []
    monkeypatch.setattr(browser, "_docker_exec",
                        lambda *_: (_ for _ in ()).throw(
                            browser.BrowserUnavailable("browser transport timed out")))
    monkeypatch.setattr(browser, "_kill_container_confirmed", killed.append)
    with pytest.raises(browser.BrowserUnavailable, match="timed out"):
        session.command("observe", page_id="old")
    assert killed == ["generation"]
    assert session.identity is None


def test_failed_attribution_removal_still_kills_and_keeps_retry_identity(
        monkeypatch, tmp_path):
    session = browser.BrowserSession("thread", "run", str(tmp_path), str(tmp_path), None)
    session.identity = browser._ContainerIdentity(
        None, str(tmp_path), "172.20.0.9", "generation", "public", None)
    killed = []
    monkeypatch.setattr(browser, "forget_client",
                        lambda *_: (_ for _ in ()).throw(OSError("read-only map")))
    monkeypatch.setattr(browser, "_kill_container_confirmed", killed.append)
    with pytest.raises(browser.BrowserUnavailable, match="attribution teardown"):
        session.close()
    assert killed == ["generation"]
    assert session.identity.generation == "generation"
