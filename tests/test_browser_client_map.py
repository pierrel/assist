"""Real process contention and recycled-address behavior for proxy attribution."""
import json
import multiprocessing
import fcntl

import pytest

from assist.egress.client_map import (
    ClientRecord, _locked, forget_client, prune_absent_browser_clients,
    record_client)


def _writer(directory, ip, generation, entered=None, release=None):
    from assist.egress import client_map

    if entered is not None:
        original = client_map._write

        def wait_before_replace(path, entries):
            entered.set()
            if not release.wait(5):
                raise TimeoutError("test writer was not released")
            original(path, entries)

        client_map._write = wait_before_replace
    client_map.record_client(directory, ip, ClientRecord(
        generation, generation, "sandbox"))


def test_client_map_lock_wait_is_bounded(tmp_path):
    with open(tmp_path / ".client-map.lock", "a+b") as holder:
        fcntl.flock(holder, fcntl.LOCK_EX)
        with pytest.raises(TimeoutError, match="lock timed out"):
            with _locked(str(tmp_path), timeout=0.05):
                pass


def test_two_processes_preserve_both_records(tmp_path):
    ctx = multiprocessing.get_context("fork")
    entered, release = ctx.Event(), ctx.Event()
    first = ctx.Process(target=_writer, args=(
        str(tmp_path), "172.20.0.2", "first", entered, release))
    second = ctx.Process(target=_writer, args=(
        str(tmp_path), "172.20.0.3", "second"))
    first.start()
    try:
        assert entered.wait(5), "first writer did not reach its locked replace"
        second.start()
        # The second process must wait at the OS lock until the first commits.
        second.join(0.1)
        assert second.is_alive()
    finally:
        release.set()
        first.join(5)
        if second.pid:
            second.join(5)
    assert first.exitcode == second.exitcode == 0
    entries = json.loads((tmp_path / "client-map.json").read_text())
    assert set(entries) == {"172.20.0.2", "172.20.0.3"}


def test_recycled_ip_survives_delayed_old_cleanup(tmp_path):
    directory = str(tmp_path)
    ip = "172.20.0.2"
    record_client(directory, ip, ClientRecord("old-thread", "old", "sandbox"))
    record_client(directory, ip, ClientRecord("new-thread", "new", "sandbox"))
    forget_client(directory, ip, "old")
    entry = json.loads((tmp_path / "client-map.json").read_text())[ip]
    assert entry["thread_id"] == "new-thread"
    forget_client(directory, ip, "new")
    assert json.loads((tmp_path / "client-map.json").read_text()) == {}


def test_absent_browser_generations_are_pruned_without_touching_sandbox(tmp_path):
    directory = str(tmp_path)
    record_client(directory, "172.20.0.2", ClientRecord(
        "thread", "gone", "browser", "internal", "host.docker.internal"))
    record_client(directory, "172.20.0.3", ClientRecord(
        "thread", "live", "browser", "public"))
    record_client(directory, "172.20.0.4", ClientRecord(
        "thread", "sandbox", "sandbox"))
    assert prune_absent_browser_clients(directory, lambda: {"live"}) == 1
    entries = json.loads((tmp_path / "client-map.json").read_text())
    assert set(entries) == {"172.20.0.3", "172.20.0.4"}


def test_empty_recovery_queue_still_reaps_browser_attribution(monkeypatch):
    import queue

    from manage.web import threads

    called = []
    monkeypatch.setattr(threads.BrowserManager, "reap_orphans",
                        lambda root: called.append(root))
    monkeypatch.setattr(threads.SandboxManager, "reap_orphans",
                        lambda _root: (_ for _ in ()).throw(
                            AssertionError("empty queue should skip sandbox recovery")))
    threads._recovery_prep(queue.Queue())
    assert called == [threads.MANAGER.root_dir]
