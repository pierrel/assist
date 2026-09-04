"""Phone API contract: auth, visible snapshots, and safe worktree data."""
from __future__ import annotations

import io
import os
import sqlite3
import tarfile
import threading
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import MessagesState, START, StateGraph

from assist.run_service import InvalidRunTransition
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
    monkeypatch.setattr(phone_api.threads, "_is_pi_thread", lambda tid: True)
    def fixture_history(tid, before):
        if before is None:
            return messages, False, False
        end = next((ordinal - 1 for ordinal, raw in enumerate(messages, start=1)
                    if phone_api._message_id(tid, raw, ordinal) == before), -1)
        if end < 0:
            raise phone_api.HTTPException(status_code=409, detail="History cursor is no longer available")
        return messages[:end], False, False
    monkeypatch.setattr(phone_api, "_thread_history", fixture_history)
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


def test_phone_api_has_no_server_pin_endpoints(tmp_path, monkeypatch):
    _thread_environment(tmp_path, monkeypatch, [])
    client = _client(monkeypatch)

    assert client.get("/api/v1/phone/threads/thread-a/pins", headers=_auth()).status_code == 404
    assert client.post("/api/v1/phone/threads/thread-a/pins", headers=_auth(),
                       json={"response_id": "m-anything"}).status_code == 404
    assert client.delete("/api/v1/phone/threads/thread-a/pins/m-anything",
                         headers=_auth()).status_code == 404


def test_thread_list_uses_stored_titles_and_supplies_chooser_metadata(tmp_path, monkeypatch):
    _thread_environment(tmp_path, monkeypatch, [])
    monkeypatch.setattr(state.MANAGER, "list", lambda: ["thread-a"])
    monkeypatch.setattr(state.MANAGER, "get",
                        lambda tid: (_ for _ in ()).throw(AssertionError("no model title")))
    monkeypatch.setattr(state, "_get_status", lambda tid: {
        "stage": "ready", "domain": "https://example.com/repo.git",
    })
    monkeypatch.setattr(phone_api, "_thread_workspace",
                        lambda tid: (_ for _ in ()).throw(AssertionError("no Git worktree scan")))

    response = _client(monkeypatch).get("/api/v1/phone/threads", headers=_auth())

    assert response.status_code == 200
    thread = response.json()["threads"][0]
    assert thread["description"] == "thread-a"
    assert thread["search_description"] == "thread-a"
    assert thread["repo_key"] == phone_api._repo_key("https://example.com/repo.git")
    assert thread["repo_label"] == "repo"
    assert isinstance(thread["activity_at"], float)
    assert len(thread["revision"]) == 24


def test_thread_list_normalizes_only_leading_pictographs(tmp_path, monkeypatch):
    thread_dir = _thread_environment(tmp_path, monkeypatch, [])
    (thread_dir / "description.txt").write_text(" 🧪 #release check")
    monkeypatch.setattr(state.MANAGER, "list", lambda: ["thread-a"])

    response = _client(monkeypatch).get("/api/v1/phone/threads", headers=_auth())

    assert response.json()["threads"][0]["search_description"] == "#release check"


def test_create_queues_initialization_on_the_dedicated_scheduler(monkeypatch):
    scheduled = []
    monkeypatch.setattr(phone_api, "_create_and_submit",
                        lambda body, key: ("thread-a", SimpleNamespace(id="run-a"), "repo-a", False))
    monkeypatch.setattr(phone_api.threads._INITIALIZATION_SCHEDULER, "submit",
                        lambda run_id, tid, domain: scheduled.append((run_id, tid, domain)))

    response = _client(monkeypatch).post(
        "/api/v1/phone/threads", headers={**_auth(), "Idempotency-Key": "a" * 16},
        json={"message": "start"})

    assert response.status_code == 200
    assert scheduled == [("run-a", "thread-a", "repo-a")]


def test_message_queues_on_the_dedicated_scheduler(monkeypatch):
    scheduled = []
    monkeypatch.setattr(phone_api, "_submit_existing",
                        lambda tid, text, key: (SimpleNamespace(id="run-a", status="pending"), False, False))
    monkeypatch.setattr(phone_api.threads._RESUME_SCHEDULER, "submit",
                        lambda run_id, tid, **kwargs: scheduled.append((run_id, tid, kwargs)))

    response = _client(monkeypatch).post(
        "/api/v1/phone/threads/thread-a/messages",
        headers={**_auth(), "Idempotency-Key": "a" * 16}, json={"message": "continue"})

    assert response.status_code == 200
    assert scheduled == [("run-a", "thread-a", {"user_priority": True})]


def test_phone_thread_creation_is_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(state.MANAGER, "thread_dir", lambda tid: str(tmp_path / tid))
    monkeypatch.setattr(state.MANAGER, "list",
                        lambda: [f"phone-{index}" for index in range(phone_api.MAX_PHONE_THREADS)])

    with pytest.raises(phone_api.HTTPException) as error:
        phone_api._create_and_submit(phone_api._CreateThread(message="start"), "a" * 16)

    assert error.value.status_code == 429


def test_phone_thread_creation_allows_only_one_waiting_initialization(tmp_path, monkeypatch):
    monkeypatch.setattr(state.MANAGER, "thread_dir", lambda tid: str(tmp_path / tid))
    monkeypatch.setattr(state.MANAGER, "list", lambda: ["phone-existing"])
    monkeypatch.setattr(state, "_get_status", lambda tid: {"stage": "initializing"})

    with pytest.raises(phone_api.HTTPException) as error:
        phone_api._create_and_submit(phone_api._CreateThread(message="start"), "a" * 16)

    assert error.value.status_code == 429
    assert error.value.detail == "Phone initialization is busy"


def test_all_first_thread_routes_are_bounded_before_reserving_a_directory(monkeypatch):
    monkeypatch.setattr(phone_api.threads.MANAGER, "list",
                        lambda: ["thread-a", "thread-b"])
    monkeypatch.setattr(phone_api.threads, "_get_status",
                        lambda tid: {"stage": "initializing"})

    with pytest.raises(phone_api.HTTPException) as error:
        phone_api.threads.create_thread_with_message_core("start", domain=None)

    assert error.value.status_code == 429
    assert error.value.detail == "Thread setup is busy"


def test_phone_message_limit_returns_a_clear_backpressure_response(tmp_path, monkeypatch):
    _thread_environment(tmp_path, monkeypatch, [])
    monkeypatch.setattr(phone_api.threads, "_pi_message_admits", lambda tid: True)
    monkeypatch.setattr(phone_api.threads, "_accept_message_run_locked",
                        lambda *args, **kwargs: (_ for _ in ()).throw(
                            InvalidRunTransition("pending run limit reached")))

    response = _client(monkeypatch).post(
        "/api/v1/phone/threads/thread-a/messages",
        headers={**_auth(), "Idempotency-Key": "a" * 16}, json={"message": "continue"})

    assert response.status_code == 429
    assert response.json()["detail"] == "pending run limit reached"


def test_phone_cannot_cancel_its_first_run_while_setup_is_pending(tmp_path, monkeypatch):
    _thread_environment(tmp_path, monkeypatch, [])
    monkeypatch.setattr(state, "_get_status", lambda tid: {
        "stage": "cloning", "pending_run_id": "run-a"})

    response = _client(monkeypatch).delete(
        "/api/v1/phone/threads/thread-a/runs/run-a", headers=_auth())

    assert response.status_code == 409
    assert response.json()["detail"] == "Thread setup is already in progress"


def test_archive_rejects_a_concurrent_download(monkeypatch):
    monkeypatch.setattr(phone_api, "_ARCHIVE_SLOTS", phone_api.threading.BoundedSemaphore(0))

    response = _client(monkeypatch).get(
        "/api/v1/phone/threads/thread-a/workspace/archive", headers=_auth())

    assert response.status_code == 429


def test_initialization_scheduler_never_uses_a_request_worker(monkeypatch):
    scheduler = phone_api.threads._InitializationScheduler()
    completed = threading.Event()
    seen = []
    monkeypatch.setattr(phone_api.threads, "_initialize_thread",
                        lambda tid, run_id, domain, rider=None: (
                            seen.append((tid, run_id, domain, rider)), completed.set()))

    scheduler.start()
    scheduler.submit("run-a", "thread-a", "repo-a")

    assert completed.wait(1)
    assert seen == [("thread-a", "run-a", "repo-a", None)]


def test_blocked_initialization_cannot_stall_the_run_scheduler(monkeypatch):
    initializer = phone_api.threads._InitializationScheduler()
    scheduler = phone_api.threads._ResumeScheduler()
    cloning, release, ran = threading.Event(), threading.Event(), threading.Event()
    monkeypatch.setattr(phone_api.threads, "_initialize_thread",
                        lambda *args: (cloning.set(), release.wait(1)))
    monkeypatch.setattr(phone_api.threads, "_execute_run",
                        lambda *args, **kwargs: ran.set())

    initializer.start()
    initializer.submit("clone-run", "clone-thread", "repo-a")
    assert cloning.wait(1)
    scheduler.start()
    scheduler.submit("turn-run", "turn-thread")

    assert ran.wait(1)
    release.set()


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


def test_snapshot_never_exposes_a_backend_error(tmp_path, monkeypatch):
    _thread_environment(tmp_path, monkeypatch, [])
    monkeypatch.setattr(state, "_get_status", lambda tid: {
        "stage": "error", "error": "clone https://user:secret@example.invalid/repo.git failed",
    })

    response = _client(monkeypatch).get("/api/v1/phone/threads/thread-a", headers=_auth())

    assert response.json()["thread"]["error"] == "Thread failed; inspect Assist Web for details."
    assert "secret" not in response.text


def test_run_status_never_exposes_a_backend_error(monkeypatch):
    monkeypatch.setattr(phone_api, "_thread_dir", lambda tid: "/thread-a")
    monkeypatch.setattr(phone_api.threads, "_runs", lambda: SimpleNamespace(
        get=lambda tid, run_id: SimpleNamespace(
            id=run_id, status="error",
            error="clone https://user:secret@example.invalid/repo.git failed",
            updated_at="2026-09-04T05:00:00+00:00")))
    monkeypatch.setattr(state, "_get_status", lambda tid: {"stage": "error"})

    status = phone_api._run_status("thread-a", "run-a")

    assert status["error"] == "Run failed; inspect Assist Web for details."


def test_snapshot_marks_the_live_last_user_message_incomplete(tmp_path, monkeypatch):
    _thread_environment(tmp_path, monkeypatch, [
        {"role": "user", "content": "still working", "message_id": "u1",
         "timestamp": "2026-09-04T05:00:00+00:00"},
    ])
    monkeypatch.setattr(state, "_get_status", lambda tid: {"stage": "processing"})

    message = _client(monkeypatch).get(
        "/api/v1/phone/threads/thread-a", headers=_auth()).json()["messages"][0]

    assert message["timestamp"] == "2026-09-04T05:00:00+00:00"
    assert message["state"] == "incomplete"


def test_snapshot_revision_changes_when_completion_state_changes(tmp_path, monkeypatch):
    _thread_environment(tmp_path, monkeypatch, [
        {"role": "user", "content": "still working", "message_id": "u1"},
    ])
    status = {"stage": "processing"}
    monkeypatch.setattr(state, "_get_status", lambda tid: status)
    client = _client(monkeypatch)

    processing = client.get("/api/v1/phone/threads/thread-a", headers=_auth()).json()
    status["stage"] = "ready"
    ready = client.get("/api/v1/phone/threads/thread-a", headers=_auth()).json()

    assert processing["thread"]["revision"] != ready["thread"]["revision"]


def test_snapshot_history_pages_are_bounded_and_nonoverlapping(tmp_path, monkeypatch):
    messages = [{"role": "user", "content": f"message {index}",
                 "message_id": f"u{index}"} for index in range(161)]
    _thread_environment(tmp_path, monkeypatch, messages)
    client = _client(monkeypatch)

    newest = client.get("/api/v1/phone/threads/thread-a", headers=_auth()).json()
    older = client.get(
        f"/api/v1/phone/threads/thread-a/history?before={newest['next_before']}",
        headers=_auth()).json()
    oldest = client.get(
        f"/api/v1/phone/threads/thread-a/history?before={older['next_before']}",
        headers=_auth()).json()

    assert len(newest["messages"]) == 80
    assert len(older["messages"]) == 80
    assert len(oldest["messages"]) == 1
    assert newest["messages"][0]["text"] == "message 81"
    assert older["messages"][0]["text"] == "message 1"
    assert oldest["messages"][0]["text"] == "message 0"
    assert newest["next_before"].startswith("m-")
    assert older["next_before"].startswith("m-")
    assert oldest["next_before"] is None


def test_history_cursor_is_stable_when_a_new_message_arrives(tmp_path, monkeypatch):
    messages = [{"role": "user", "content": f"message {index}",
                 "message_id": f"u{index}"} for index in range(161)]
    _thread_environment(tmp_path, monkeypatch, messages)
    client = _client(monkeypatch)

    newest = client.get("/api/v1/phone/threads/thread-a", headers=_auth()).json()
    messages.append({"role": "assistant", "content": "new reply", "message_id": "a-new"})
    older = client.get(
        f"/api/v1/phone/threads/thread-a/history?before={newest['next_before']}",
        headers=_auth()).json()

    assert [message["text"] for message in older["messages"]] == [
        f"message {index}" for index in range(1, 81)]


def test_history_cursor_survives_a_page_of_hidden_records(tmp_path, monkeypatch):
    messages = [{"role": "user", "content": "visible", "message_id": "u1"}] + [
        {"role": "tools", "content": "hidden", "message_id": f"t{index}"}
        for index in range(80)]
    _thread_environment(tmp_path, monkeypatch, messages)
    client = _client(monkeypatch)

    hidden = client.get("/api/v1/phone/threads/thread-a", headers=_auth()).json()
    older = client.get(
        f"/api/v1/phone/threads/thread-a/history?before={hidden['next_before']}",
        headers=_auth()).json()

    assert hidden["messages"] == [] and hidden["next_before"].startswith("m-")
    assert [message["text"] for message in older["messages"]] == ["visible"]


def test_checkpoint_history_pages_message_writes_without_hydrating_graph_state(tmp_path, monkeypatch):
    """The phone reader pages LangGraph's append writes, never get_state()."""
    thread_dir = tmp_path / "thread-a"
    (thread_dir / "workspace").mkdir(parents=True)
    (thread_dir / "workspace" / "notes.md").write_text("hello")
    connection = sqlite3.connect(":memory:", check_same_thread=False)
    saver = SqliteSaver(connection)
    graph = StateGraph(MessagesState)
    graph.add_node("reply", lambda _state: {"messages": [{"role": "assistant", "content": "ok"}]})
    graph.add_edge(START, "reply")
    compiled = graph.compile(checkpointer=saver)
    for index in range(3):
        compiled.invoke({"messages": [{"role": "user", "content": f"message {index}"}]},
                        {"configurable": {"thread_id": "thread-a"}})
    monkeypatch.setattr(state.MANAGER, "checkpointer", saver)
    monkeypatch.setattr(state.MANAGER, "thread_dir", lambda tid: str(tmp_path / tid))
    monkeypatch.setattr(state.MANAGER, "thread_default_working_dir",
                        lambda tid: str(tmp_path / tid / "workspace"))
    monkeypatch.setattr(phone_api, "read_thread_engine", lambda directory: SimpleNamespace(name="deepagents"))
    monkeypatch.setattr(phone_api, "_thread_workspace", lambda tid: {
        "repo_key": None, "repo_label": "No repository", "branch": None,
        "revision": None, "dirty": False,
    })
    monkeypatch.setattr(state, "_get_status", lambda tid: {"stage": "ready"})
    monkeypatch.setattr(phone_api, "MAX_HISTORY_MESSAGES", 2)
    client = _client(monkeypatch)

    newest = client.get("/api/v1/phone/threads/thread-a", headers=_auth()).json()
    compiled.invoke({"messages": [{"role": "user", "content": "message 3"}]},
                    {"configurable": {"thread_id": "thread-a"}})
    older = client.get(
        f"/api/v1/phone/threads/thread-a/history?before={newest['next_before']}",
        headers=_auth()).json()

    assert [message["text"] for message in newest["messages"]] == ["message 2", "ok"]
    assert [message["text"] for message in older["messages"]] == ["message 1", "ok"]
    assert newest["next_before"].startswith("c-")

    monkeypatch.setattr(phone_api, "_thread_messages",
                        lambda tid: (_ for _ in ()).throw(AssertionError("no full state read")))


def test_pi_message_ids_remain_stable_across_reordering(tmp_path, monkeypatch):
    _thread_environment(tmp_path, monkeypatch, [])
    message = {"role": "assistant", "content": "Pi reply", "message_id": "pi:run-1"}

    assert phone_api._message_id("thread-a", message, 1) == phone_api._message_id(
        "thread-a", message, 99)
    assert phone_api._message_id("thread-a", message, 1) != phone_api._message_id(
        "thread-a", {**message, "role": "user"}, 1)


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


def test_workspace_listing_closes_fwalk_after_a_truncated_walk(tmp_path, monkeypatch):
    thread_dir = _thread_environment(tmp_path, monkeypatch, [])
    workspace = thread_dir / "workspace"
    closed = False

    class Walker:
        def __init__(self):
            self.fd = os.open(workspace, os.O_RDONLY | os.O_DIRECTORY)

        def __iter__(self):
            yield ".", [], ["notes.md"], self.fd

        def close(self):
            nonlocal closed
            closed = True
            os.close(self.fd)

    monkeypatch.setattr(phone_api.os, "fwalk", lambda *args, **kwargs: Walker())
    monkeypatch.setattr(phone_api, "MAX_WORKSPACE_NODES", 0)

    assert phone_api._workspace_entries("thread-a") == []
    assert closed


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


def test_workspace_root_symlink_is_rejected(tmp_path, monkeypatch):
    thread_dir = _thread_environment(tmp_path, monkeypatch, [])
    workspace = thread_dir / "workspace"
    (workspace / "notes.md").unlink()
    workspace.rmdir()
    workspace.symlink_to("/etc")
    client = _client(monkeypatch)

    manifest = client.get("/api/v1/phone/threads/thread-a/workspace", headers=_auth())
    archive = client.get("/api/v1/phone/threads/thread-a/workspace/archive", headers=_auth())

    assert manifest.json()["files"] == []
    assert archive.status_code == 409


def test_workspace_manifest_bounds_empty_directory_traversal(tmp_path, monkeypatch):
    thread_dir = _thread_environment(tmp_path, monkeypatch, [])
    workspace = thread_dir / "workspace"
    for index in range(5):
        (workspace / f"empty-{index}").mkdir()
    monkeypatch.setattr(phone_api, "MAX_WORKSPACE_NODES", 3)

    response = _client(monkeypatch).get(
        "/api/v1/phone/threads/thread-a/workspace", headers=_auth())

    assert response.json()["truncated"] is True
