"""Host transport remains bounded under repeated Docker/sidecar stalls."""
import os
import sys
import threading

import pytest

from assist.browser import manager as browser


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


def test_failed_command_kills_sidecar_before_releasing_session(monkeypatch, tmp_path):
    session = browser.BrowserSession("thread", "run", str(tmp_path), str(tmp_path), None)
    session.identity = browser._ContainerIdentity(
        None, str(tmp_path), "172.20.0.9", "generation", "public", None)
    killed = []
    monkeypatch.setattr(browser, "_docker_exec",
                        lambda *_: (_ for _ in ()).throw(
                            browser.BrowserUnavailable("browser transport timed out")))
    monkeypatch.setattr(browser, "_kill_container", killed.append)
    with pytest.raises(browser.BrowserUnavailable, match="timed out"):
        session.command("observe", page_id="old")
    assert killed == ["generation"]
    assert session.identity is None
