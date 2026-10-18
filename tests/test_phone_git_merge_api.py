"""Authenticated Git facts and the shared web/phone merge operation."""

from __future__ import annotations

import asyncio
import subprocess
from contextlib import nullcontext
from types import SimpleNamespace

from fastapi.testclient import TestClient
from fastapi.responses import RedirectResponse

from assist.git_sync import bind
from manage.web import phone_api, state, threads
from manage.web.app import app


def _git(*args: str, cwd=None) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def _setup(tmp_path, monkeypatch):
    source = tmp_path / "source.git"
    seed = tmp_path / "seed"
    thread = tmp_path / "thread-a"
    thread.mkdir()
    _git("init", "--bare", str(source))
    _git("init", "-b", "main", str(seed))
    _git("-C", str(seed), "config", "user.name", "Test")
    _git("-C", str(seed), "config", "user.email", "test@example.invalid")
    (seed / "file.txt").write_text("one\n")
    _git("-C", str(seed), "add", "file.txt")
    _git("-C", str(seed), "commit", "-m", "initial")
    _git("-C", str(seed), "push", str(source), "main")
    _git("-C", str(seed), "checkout", "-b", "assist/thread-a")
    (seed / "file.txt").write_text("two\n")
    _git("-C", str(seed), "commit", "-am", "thread")
    _git("-C", str(seed), "push", str(source), "assist/thread-a")
    bind(str(thread), str(source))
    from assist.git_sync import read_state, _write_state
    binding = read_state(str(thread))
    binding.update(branch="assist/thread-a", floor=_git("-C", str(seed), "rev-parse", "HEAD"))
    _write_state(str(thread), binding)
    monkeypatch.setattr(state.MANAGER, "thread_dir", lambda tid: str(thread))
    monkeypatch.setattr(state, "DOMAINS", [str(source)])
    monkeypatch.setenv(phone_api.PHONE_API_TOKEN_ENV, "test-token")
    return source, thread, seed


def test_phone_git_status_samples_configured_remote_without_checkout_changes(tmp_path, monkeypatch):
    source, thread, seed = _setup(tmp_path, monkeypatch)
    before = _git("-C", str(seed), "rev-parse", "HEAD")
    client = TestClient(app)
    path = "/api/v1/phone/threads/thread-a/git"
    assert client.get(path).status_code == 401
    response = client.get(path, headers={"Authorization": "Bearer test-token"})
    assert response.status_code == 200
    assert response.json() == {
        "thread_id": "thread-a", "repo_key": phone_api._repo_key(str(source)),
        "branch": "assist/thread-a", "published_revision": before,
        "state": "available", "observed_at": response.json()["observed_at"],
    }
    assert response.json()["observed_at"]
    assert _git("-C", str(seed), "rev-parse", "HEAD") == before
    assert _git("--git-dir", str(source), "rev-parse", "refs/heads/assist/thread-a") == before


def test_phone_git_status_unavailable_is_not_a_cached_ref(tmp_path, monkeypatch):
    source, _thread, _seed = _setup(tmp_path, monkeypatch)
    _git("--git-dir", str(source), "update-ref", "-d", "refs/heads/assist/thread-a")
    client = TestClient(app)
    response = client.get("/api/v1/phone/threads/thread-a/git",
                          headers={"Authorization": "Bearer test-token"})
    assert response.json()["state"] == "unpublished"
    assert response.json()["published_revision"] is None
    assert response.json()["observed_at"]


def test_phone_git_status_remote_failure_never_reuses_a_published_sha(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    from assist.git_sync import GitSyncError
    from assist.local_git import LocalGit
    def unavailable(_self, _branch):
        raise GitSyncError("remote unavailable")
    monkeypatch.setattr(LocalGit, "remote_ref", unavailable)
    response = TestClient(app).get(
        "/api/v1/phone/threads/thread-a/git",
        headers={"Authorization": "Bearer test-token"})
    assert response.status_code == 200
    assert response.json()["state"] == "unavailable"
    assert response.json()["published_revision"] is None
    assert response.json()["observed_at"] is None


def test_phone_merge_calls_same_web_operation_once_and_rejects_caller_input(tmp_path, monkeypatch):
    _source, _thread, seed = _setup(tmp_path, monkeypatch)
    calls = []
    expected = _git("-C", str(seed), "rev-parse", "HEAD")
    monkeypatch.setattr(threads, "_merge_thread_result",
                        lambda tid, oid: calls.append((tid, oid)) or "merged")
    client = TestClient(app)
    path = "/api/v1/phone/threads/thread-a/merge"
    assert client.post(path).status_code == 401
    headers = {"Authorization": "Bearer test-token"}
    assert client.post(path, headers=headers, json={"branch": "main"}).status_code == 422
    assert client.post(path, headers=headers).status_code == 428
    headers["X-Assist-Expected-Commit"] = expected
    response = client.post(path, headers=headers)
    assert response.status_code == 200
    assert response.json() == {"outcome": "merged"}
    assert calls == [("thread-a", expected)]


def test_phone_merge_validates_thread_directory_off_event_loop(tmp_path, monkeypatch):
    _source, _thread, seed = _setup(tmp_path, monkeypatch)
    original = phone_api._thread_dir
    worker_checked = []

    def checked_directory(tid):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            worker_checked.append(True)
        return original(tid)

    monkeypatch.setattr(phone_api, "_thread_dir", checked_directory)
    monkeypatch.setattr(threads, "_merge_thread_result", lambda _tid, _oid: "merged")
    response = TestClient(app).post(
        "/api/v1/phone/threads/thread-a/merge",
        headers={"Authorization": "Bearer test-token",
                 "X-Assist-Expected-Commit": _git("-C", str(seed), "rev-parse", "HEAD")})
    assert response.status_code == 200
    assert worker_checked == [True]


def test_phone_merge_reports_exact_conflict_and_busy(tmp_path, monkeypatch):
    _source, _thread, seed = _setup(tmp_path, monkeypatch)
    client = TestClient(app)
    path = "/api/v1/phone/threads/thread-a/merge"
    headers = {"Authorization": "Bearer test-token",
               "X-Assist-Expected-Commit": _git("-C", str(seed), "rev-parse", "HEAD")}
    monkeypatch.setattr(threads, "_merge_thread_result", lambda tid, oid: "conflict")
    response = client.post(path, headers=headers)
    assert (response.status_code, response.json()) == (409, {"outcome": "conflict"})
    def busy(_tid, _oid):
        raise threads.MergeBusyError("busy")
    monkeypatch.setattr(threads, "_merge_thread_result", busy)
    response = client.post(path, headers=headers)
    assert (response.status_code, response.json()) == (409, {"outcome": "busy"})
    def stale(_tid, _oid):
        raise threads.MergeStaleError("changed")
    monkeypatch.setattr(threads, "_merge_thread_result", stale)
    response = client.post(path, headers=headers)
    assert (response.status_code, response.json()) == (409, {"outcome": "stale"})
    monkeypatch.setattr(threads, "_merge_thread_result", lambda _tid, _oid: "merged")
    validated = []
    original_directory = phone_api._thread_dir
    monkeypatch.setattr(phone_api, "_thread_dir",
                        lambda tid: validated.append(tid) or original_directory(tid))
    assert phone_api._MERGE_SLOTS.acquire(blocking=False)
    try:
        response = client.post(path, headers=headers)
        assert (response.status_code, response.json()) == (409, {"outcome": "busy"})
        assert validated == []
    finally:
        phone_api._MERGE_SLOTS.release()


def test_phone_merge_cannot_bypass_confirmation_on_unbound_legacy_thread(
        tmp_path, monkeypatch):
    _source, _thread, seed = _setup(tmp_path, monkeypatch)
    called = []
    domain = SimpleNamespace(repo=object(), merge_and_push=lambda: called.append(True))
    monkeypatch.setattr(threads, "_require_deep_thread", lambda _tid: None)
    monkeypatch.setattr(threads.MANAGER, "get", lambda _tid: object())
    monkeypatch.setattr(threads, "_get_status", lambda _tid: {})
    monkeypatch.setattr(threads, "_get_domain_manager", lambda _tid: domain)
    monkeypatch.setattr(threads, "_git_merge_has_unfinished_work", lambda _tid: False)
    monkeypatch.setattr(threads.GitLifecycle, "acquire",
                        staticmethod(lambda *_args, **_kwargs:
                                     nullcontext(SimpleNamespace(bound=False))))
    response = TestClient(app).post(
        "/api/v1/phone/threads/thread-a/merge",
        headers={"Authorization": "Bearer test-token",
                 "X-Assist-Expected-Commit": _git("-C", str(seed), "rev-parse", "HEAD")})
    assert (response.status_code, response.json()) == (409, {"outcome": "error"})
    assert called == []


def test_web_merge_route_still_maps_shared_outcomes_to_original_redirect(monkeypatch):
    monkeypatch.setattr(threads, "_merge_thread_result", lambda _tid: "merged")
    merged = threads.merge_thread("thread-a")
    assert isinstance(merged, RedirectResponse)
    assert (merged.status_code, merged.headers["location"]) == (
        303, "/thread/thread-a?merged=1")
    monkeypatch.setattr(threads, "_merge_thread_result", lambda _tid: "conflict")
    conflict = threads.merge_thread("thread-a")
    assert (conflict.status_code, conflict.headers["location"]) == (
        303, "/thread/thread-a?conflict=1")
