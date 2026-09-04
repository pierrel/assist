"""Phone API contract: auth, visible snapshots, pins, and safe worktree data."""
from __future__ import annotations

import io
import tarfile
from types import SimpleNamespace

from fastapi.testclient import TestClient

from manage.web import state
from manage.web.app import app
from manage.web import phone_api


def _client(monkeypatch) -> TestClient:
    monkeypatch.setenv(phone_api.PHONE_API_TOKEN_ENV, "phone-test-token")
    return TestClient(app)


def _auth() -> dict[str, str]:
    return {"Authorization": "Bearer phone-test-token"}


def _thread_environment(tmp_path, monkeypatch, messages):
    thread_dir = tmp_path / "thread-a"
    workspace = thread_dir / "workspace"
    workspace.mkdir(parents=True)
    (workspace / "notes.md").write_text("hello")
    monkeypatch.setattr(state.MANAGER, "thread_dir", lambda tid: str(tmp_path / tid))
    monkeypatch.setattr(state.MANAGER, "thread_default_working_dir",
                        lambda tid: str(tmp_path / tid / "workspace"))
    monkeypatch.setattr(phone_api, "_thread_messages", lambda tid: messages)
    monkeypatch.setattr(phone_api, "read_thread_engine",
                        lambda directory: SimpleNamespace(name="deepagents"))
    monkeypatch.setattr(phone_api, "_thread_workspace", lambda tid: {
        "repo_key": "repo-a", "repo_label": "Repo A", "branch": "assist/demo",
        "revision": "abc", "dirty": True,
    })
    monkeypatch.setattr(state, "_thread_title", lambda tid: "🧪 release check")
    monkeypatch.setattr(state, "_get_status", lambda tid: {"stage": "ready"})
    return thread_dir


def test_phone_api_requires_a_configured_bearer_token(monkeypatch):
    monkeypatch.delenv(phone_api.PHONE_API_TOKEN_ENV, raising=False)
    client = TestClient(app)

    assert client.get("/api/v1/phone/threads").status_code == 503

    monkeypatch.setenv(phone_api.PHONE_API_TOKEN_ENV, "phone-test-token")
    assert client.get("/api/v1/phone/threads").status_code == 401


def test_phone_api_rejects_hidden_thread_directories(tmp_path, monkeypatch):
    thread_dir = _thread_environment(tmp_path, monkeypatch, [])
    (thread_dir / ".deleted").write_text("gone")

    response = _client(monkeypatch).get("/api/v1/phone/threads/thread-a", headers=_auth())

    assert response.status_code == 404


def test_snapshot_exposes_visible_messages_and_safe_file_references(tmp_path, monkeypatch):
    _thread_environment(tmp_path, monkeypatch, [
        {"role": "user", "content": "what changed", "message_id": "u1"},
        {"role": "tools", "content": "private tool output", "message_id": "tool1"},
        {"role": "assistant", "content": "Read notes.md and /etc/passwd", "message_id": "a1"},
    ])
    client = _client(monkeypatch)

    response = client.get("/api/v1/phone/threads/thread-a", headers=_auth())

    assert response.status_code == 200
    payload = response.json()
    assert [message["role"] for message in payload["messages"]] == ["user", "assistant"]
    answer = payload["messages"][-1]
    assert answer["id"].startswith("m-")
    assert answer["file_refs"] == [{"path": "notes.md", "label": "notes.md"}]
    assert "private tool output" not in response.text
    assert response.headers["cache-control"] == "no-store"


def test_snapshot_history_pages_are_bounded_and_nonoverlapping(tmp_path, monkeypatch):
    messages = [{"role": "user", "content": f"message {index}",
                 "message_id": f"u{index}"} for index in range(161)]
    _thread_environment(tmp_path, monkeypatch, messages)
    client = _client(monkeypatch)

    newest = client.get("/api/v1/phone/threads/thread-a", headers=_auth()).json()
    older = client.get("/api/v1/phone/threads/thread-a/history?before=80", headers=_auth()).json()
    oldest = client.get("/api/v1/phone/threads/thread-a/history?before=160", headers=_auth()).json()

    assert len(newest["messages"]) == 80
    assert len(older["messages"]) == 80
    assert len(oldest["messages"]) == 1
    assert newest["messages"][0]["text"] == "message 81"
    assert older["messages"][0]["text"] == "message 1"
    assert oldest["messages"][0]["text"] == "message 0"
    assert newest["next_before"] == 80
    assert older["next_before"] == 160
    assert oldest["next_before"] is None


def test_pin_requires_a_visible_assistant_response_and_is_durable(tmp_path, monkeypatch):
    _thread_environment(tmp_path, monkeypatch, [
        {"role": "assistant", "content": "Read notes.md", "message_id": "a1"},
    ])
    client = _client(monkeypatch)
    snapshot = client.get("/api/v1/phone/threads/thread-a", headers=_auth()).json()
    response_id = snapshot["messages"][0]["id"]

    created = client.post("/api/v1/phone/threads/thread-a/pins", headers=_auth(),
                          json={"response_id": response_id})
    listed = client.get("/api/v1/phone/threads/thread-a/pins", headers=_auth())

    assert created.status_code == 200
    assert listed.json()["pins"][0]["response_id"] == response_id
    assert listed.json()["pins"][0]["text"] == "Read notes.md"
    assert client.post("/api/v1/phone/threads/thread-a/pins", headers=_auth(),
                       json={"response_id": "m-missing"}).status_code == 404


def test_workspace_archive_omits_git_and_symlinked_host_files(tmp_path, monkeypatch):
    thread_dir = _thread_environment(tmp_path, monkeypatch, [])
    workspace = thread_dir / "workspace"
    (workspace / ".git").mkdir()
    (workspace / ".git" / "config").write_text("private")
    (workspace / "host-link").symlink_to("/etc/passwd")
    client = _client(monkeypatch)

    response = client.get("/api/v1/phone/threads/thread-a/workspace", headers=_auth())
    archive = client.get("/api/v1/phone/threads/thread-a/workspace/archive", headers=_auth())

    assert response.status_code == 200
    assert response.json()["files"] == [{"path": "notes.md", "type": "file", "size": 5}]
    assert archive.status_code == 200
    assert b"private" not in archive.content


def test_workspace_archive_rechecks_a_file_that_turns_into_a_symlink(tmp_path, monkeypatch):
    thread_dir = _thread_environment(tmp_path, monkeypatch, [])
    workspace = thread_dir / "workspace"
    listed = [{"path": "notes.md", "type": "file", "size": 5}]
    (workspace / "notes.md").unlink()
    (workspace / "notes.md").symlink_to("/etc/hostname")
    monkeypatch.setattr(phone_api, "_workspace_entries", lambda *args, **kwargs: listed)

    archive = _client(monkeypatch).get(
        "/api/v1/phone/threads/thread-a/workspace/archive", headers=_auth())

    with tarfile.open(fileobj=io.BytesIO(archive.content), mode="r:gz") as result:
        assert result.getnames() == []


def test_workspace_archive_rejects_an_intermediate_symlink(tmp_path, monkeypatch):
    thread_dir = _thread_environment(tmp_path, monkeypatch, [])
    workspace = thread_dir / "workspace"
    nested = workspace / "nested"
    nested.mkdir()
    (nested / "hostname").write_text("not-a-host-file")
    listed = [{"path": "nested/hostname", "type": "file", "size": 15}]
    (nested / "hostname").unlink()
    nested.rmdir()
    nested.symlink_to("/etc")
    monkeypatch.setattr(phone_api, "_workspace_entries", lambda *args, **kwargs: listed)

    archive = _client(monkeypatch).get(
        "/api/v1/phone/threads/thread-a/workspace/archive", headers=_auth())

    with tarfile.open(fileobj=io.BytesIO(archive.content), mode="r:gz") as result:
        assert result.getnames() == []
