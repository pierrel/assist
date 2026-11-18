"""Private authenticated entry publishes only ciphertext on a thread branch."""

import subprocess
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from assist import git_sync, thread_git
from assist.git_sync import _initial_binding, _write_state, require_clean

from fastapi.testclient import TestClient

from assist.secret_store import FILE, KEY_FILE, REQUEST_FILE, SecretStore
from assist.run_service import RunService
from manage.web import phone_api
from manage.web.app import app
from manage.web.thread_git_lifecycle import ThreadGitLifecycle


def _git(cwd, *args):
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


@pytest.mark.parametrize("ignore_secrets", [False, True])
def test_authenticated_private_entry_commits_ciphertext_only(
        tmp_path, monkeypatch, ignore_secrets):
    remote = tmp_path / "remote.git"
    repo = tmp_path / "repo"
    thread = tmp_path / "thread"
    thread.mkdir()
    _git(tmp_path, "init", "--bare", str(remote))
    _git(tmp_path, "init", "-b", "main", str(repo))
    _git(repo, "config", "user.name", "Synthetic")
    _git(repo, "config", "user.email", "synthetic@example.invalid")
    (repo / "README").write_text("synthetic\n")
    _git(repo, "add", "README")
    if ignore_secrets:
        (repo / ".gitignore").write_text("*secrets*\n")
        _git(repo, "add", ".gitignore")
    _git(repo, "commit", "-m", "base")
    _git(repo, "remote", "add", "origin", str(remote))
    _git(repo, "push", "origin", "main")
    main_tip = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-b", "assist/synthetic")
    _git(repo, "push", "origin", "assist/synthetic")

    binding = _initial_binding(str(remote))
    binding.update(branch="assist/synthetic", floor=main_tip)
    _write_state(str(thread), binding)
    monkeypatch.setenv("ASSIST_DOMAINS", str(remote))
    monkeypatch.setenv(phone_api.PHONE_API_TOKEN_ENV, "synthetic-token")
    monkeypatch.setattr(phone_api, "_repo_secrets",
                        lambda _tid: SecretStore(repo, thread))
    command = f"git --git-dir={repo / '.git'} --work-tree={repo}"
    monkeypatch.setattr(git_sync, "_WORKTREE_GIT", command)
    monkeypatch.setattr(thread_git, "_WORKTREE_GIT", command)

    class LocalSandbox:
        container = None

        def execute(self, script):
            completed = subprocess.run(script, shell=True, capture_output=True)
            return SimpleNamespace(exit_code=completed.returncode, error=None,
                                   truncated=False)

    sandbox = LocalSandbox()
    monkeypatch.setattr(ThreadGitLifecycle, "_backend", lambda self, _tz: sandbox)
    monkeypatch.setattr(ThreadGitLifecycle, "_verify_clean",
                        lambda self, _tz: require_clean(sandbox))
    monkeypatch.setattr(ThreadGitLifecycle, "_host_fence", lambda self: nullcontext())
    monkeypatch.setattr(ThreadGitLifecycle, "_cleanup", lambda self, _gen: None)
    client = TestClient(app)
    path = "/api/v1/phone/threads/synthetic/secrets"
    headers = {"Authorization": "Bearer synthetic-token"}
    assert client.get(path).status_code == 401
    assert client.get(path, headers=headers).json() == {"secrets": [], "unlocked": False}
    private = {"label": "Synthetic site", "login_url": "https://synthetic.example/login",
               "username": "synthetic-user", "password": "synthetic-river-8642",
               "passphrase": "synthetic unlock phrase only"}
    pending = SecretStore(repo, thread).request_input(
        private["label"], private["login_url"])
    stale = client.post(path, headers=headers,
                        json={**private, "request_id": "0" * 32})
    assert stale.status_code == 409
    assert not (repo / FILE).exists()
    assert (thread / REQUEST_FILE).exists()
    invalid = client.post(path, headers=headers, json={**private, "password": ""})
    assert invalid.status_code == 409
    assert not (repo / FILE).exists()
    assert "synthetic unlock phrase only" not in invalid.text
    response = client.post(path, headers=headers,
                           json={**private, "request_id": pending["id"]})
    assert response.status_code == 200
    assert not (thread / REQUEST_FILE).exists()
    assert _git(repo, "log", "-1", "--format=%s") == "Store encrypted credential"
    assert "synthetic-river-8642" not in response.text
    assert "synthetic unlock phrase only" not in response.text
    assert (thread / KEY_FILE).exists()
    assert b"synthetic-river-8642" not in (repo / FILE).read_bytes()
    assert b"synthetic unlock phrase only" not in (repo / FILE).read_bytes()
    committed = _git(remote, "show", "assist/synthetic:" + FILE)
    assert "synthetic-river-8642" not in committed
    assert _git(remote, "rev-parse", "main") == main_tip
    assert client.get(path, headers=headers).json()["unlocked"] is True
    assert len(client.get(path, headers=headers).json()["secrets"]) == 1

    paused_request = SecretStore(repo, thread).request_input(
        "Paused site", "https://synthetic.example/other")
    (tmp_path / "synthetic").mkdir()
    runs = RunService(str(tmp_path))
    pending_run = runs.create("synthetic", "general-agent", "Finish login",
                              user_origin=True)
    runs.claim("synthetic", pending_run.id)
    runs.transition("synthetic", pending_run.id, "interrupted")
    monkeypatch.setattr(phone_api.threads, "_runs", lambda: runs)
    monkeypatch.setattr(ThreadGitLifecycle, "prepare",
                        lambda *_args: pytest.fail("paused browser must not be retired"))
    paused = client.post(path, headers=headers, json={
        **private, "label": "Paused site", "login_url": "https://synthetic.example/other",
        "request_id": paused_request["id"]})
    assert paused.status_code == 409
    assert len(SecretStore(repo, thread).list_public()) == 1
    assert (thread / REQUEST_FILE).exists()

    other = tmp_path / "other-thread"
    other.mkdir()
    monkeypatch.setattr(phone_api, "_repo_secrets", lambda _tid: SecretStore(repo, other))
    assert client.get(path, headers=headers).json()["unlocked"] is False
    wrong = client.post(path + "/unlock", headers=headers,
                        json={"passphrase": "wrong synthetic phrase"})
    assert wrong.status_code == 409
    assert not (other / KEY_FILE).exists()
    good = client.post(path + "/unlock", headers=headers,
                       json={"passphrase": "synthetic unlock phrase only"})
    assert good.json() == {"unlocked": True}


def test_authenticated_thread_snapshot_offers_agent_requested_private_form(
        tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    thread = tmp_path / "thread"
    repo.mkdir()
    thread.mkdir()
    (tmp_path / "synthetic").mkdir()
    runs = RunService(str(tmp_path))
    request = SecretStore(repo, thread).request_input(
        "Synthetic site", "https://synthetic.example/login")
    monkeypatch.setenv(phone_api.PHONE_API_TOKEN_ENV, "synthetic-token")
    monkeypatch.setattr(phone_api, "_repo_secrets", lambda _tid: SecretStore(repo, thread))
    monkeypatch.setattr(phone_api, "_thread_dir", lambda _tid: str(thread))
    monkeypatch.setattr(phone_api, "_thread_history", lambda *_args: ([], False, False))
    monkeypatch.setattr(phone_api, "_workspace_entries", lambda *_args, **_kw: [])
    monkeypatch.setattr(phone_api, "_approval_summary", lambda *_args: (0, False))
    monkeypatch.setattr(phone_api, "_active_thread_run", lambda _tid: None)
    monkeypatch.setattr(phone_api.threads, "_runs", lambda: runs)
    monkeypatch.setattr(phone_api, "_stored_thread_title", lambda _tid: "Synthetic")
    monkeypatch.setattr(phone_api, "_thread_workspace",
                        lambda _tid: {"repo_label": "Synthetic"})
    monkeypatch.setattr(phone_api, "read_thread_engine",
                        lambda _directory: SimpleNamespace(name="assist"))
    monkeypatch.setattr(phone_api.state, "_notification_revision", lambda _tid: "1")
    monkeypatch.setattr(phone_api.state, "_get_status", lambda _tid: {"stage": "ready"})
    client = TestClient(app)
    path = "/api/v1/phone/threads/synthetic"
    assert client.get(path).status_code == 401
    offered = client.get(path, headers={"Authorization": "Bearer synthetic-token"})
    assert offered.status_code == 200
    assert offered.json()["thread"]["private_entry_request"] == request
    assert "password" not in offered.text
    monkeypatch.setattr(phone_api, "_active_thread_run",
                        lambda _tid: {"id": "run", "work_id": "work",
                                      "status": "running"})
    busy = client.get(path, headers={"Authorization": "Bearer synthetic-token"})
    assert busy.status_code == 200
    assert busy.json()["thread"]["private_entry_request"] is None

    monkeypatch.setattr(phone_api, "_active_thread_run", lambda _tid: None)
    pending_run = runs.create("synthetic", "general-agent", "Finish login",
                              user_origin=True)
    runs.claim("synthetic", pending_run.id)
    runs.transition("synthetic", pending_run.id, "interrupted")
    monkeypatch.setattr(phone_api.state, "_get_status", lambda _tid: {"stage": "paused"})
    paused = client.get(path, headers={"Authorization": "Bearer synthetic-token"})
    assert paused.status_code == 200
    assert paused.json()["thread"]["private_entry_request"] is None
    assert (thread / REQUEST_FILE).exists()
