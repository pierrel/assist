"""Complete local-first Git sequences over real bare and working repositories."""

from dataclasses import dataclass
from contextlib import nullcontext
import asyncio
from pathlib import Path
import subprocess

import pytest

from assist.git_sync import (GitSyncError, _initial_binding, _write_state, authorize_branch,
                             bind, identity, publish_initial_branch)
from assist.domain_manager import git_diff_main
from assist.thread_git import ThreadGit
from manage.web.thread_git_lifecycle import ThreadGitLifecycle
from tests.test_local_git import commit, git, repositories  # noqa: F401


@dataclass
class Response:
    exit_code: int = 0
    error: str | None = None
    truncated: bool = False


class LocalSandbox:
    container = None

    def execute(self, command):
        result = subprocess.run(command, shell=True, check=False, capture_output=True)
        return Response(exit_code=result.returncode)

    def upload_files(self, files):
        for name, content in files:
            Path(name).write_bytes(content)
        return [Response() for _ in files]


@pytest.fixture
def bound_thread(tmp_path, repositories, monkeypatch):
    source, server, phone = repositories
    directory = tmp_path / "thread"
    directory.mkdir()
    branch, revision = identity(str(server))
    state = _initial_binding(str(source))
    state.update(branch=branch, floor=revision)
    _write_state(str(directory), state)
    import assist.thread_git as module
    import assist.git_sync as git_sync
    command = f"git --git-dir={server / '.git'} --work-tree={server}"
    monkeypatch.setattr(module, "_WORKTREE_GIT", command)
    monkeypatch.setattr(git_sync, "_WORKTREE_GIT", command)
    return ThreadGit(str(directory), str(server), (str(source),)), source, server, phone


def test_phone_before_turn_and_during_turn_then_user_merge(bound_thread):
    owner, source, server, phone = bound_thread
    first = commit(phone, "before.txt", "before\n")
    git("push", "origin", "assist/thread", cwd=phone)
    owner.apply_prepare(LocalSandbox(), owner.plan_prepare())
    assert identity(str(server))[1] == first
    assert (server / "before.txt").read_text() == "before\n"

    server_result = commit(server, "server.txt", "server\n")
    commit(phone, "during.txt", "during\n")
    git("push", "origin", "assist/thread", cwd=phone)
    result_branch = owner.publish()
    assert result_branch == "assist-result/" + server_result
    assert git("rev-parse", "refs/heads/assist/thread", cwd=source) != server_result
    git("fetch", "origin", result_branch, cwd=phone)
    git("-c", "user.name=Phone", "-c", "user.email=phone@localhost",
        "merge", "--no-ff", "FETCH_HEAD", cwd=phone)
    combined = git("rev-parse", "HEAD", cwd=phone)
    git("push", "origin", "assist/thread", cwd=phone)
    owner.apply_prepare(LocalSandbox(), owner.plan_prepare())
    assert identity(str(server))[1] == combined
    assert (server / "server.txt").read_text() == "server\n"
    assert (server / "during.txt").read_text() == "during\n"


def test_noop_turn_and_remote_retry_without_model(bound_thread):
    owner, source, server, _ = bound_thread
    owner.apply_prepare(LocalSandbox(), owner.plan_prepare())
    assert owner.publish() is None
    desired = commit(server, "retained.txt", "saved\n")
    # A previous push failed; the next top-level preflight republishes the
    # existing commit without invoking any model or altering its bytes.
    owner.apply_prepare(LocalSandbox(), owner.plan_prepare())
    assert git("rev-parse", "refs/heads/assist/thread", cwd=source) == desired


def test_new_thread_publishes_at_main_before_no_change_turn(
        repositories, tmp_path, monkeypatch):
    source, server, _ = repositories
    git("push", "origin", ":assist/thread", cwd=server)
    directory = tmp_path / "new-thread"
    directory.mkdir()
    from assist.browser.manager import BrowserManager
    monkeypatch.setattr(BrowserManager, "confirm_owner_stopped", lambda *args, **kwargs: False)
    monkeypatch.setenv("ASSIST_DOMAINS", str(source))
    bind(str(directory), str(source))
    authorize_branch(str(directory), str(server))
    publish_initial_branch(str(directory), str(server))
    assert git("rev-parse", "refs/heads/assist/thread", cwd=source) == identity(str(server))[1]
    owner = ThreadGit(str(directory), str(server), (str(source),))
    assert owner.plan_prepare().remote == identity(str(server))[1]


def test_empty_web_thread_publishes_its_configured_branch(
        repositories, tmp_path, monkeypatch):
    source, _, _ = repositories
    from assist.browser.manager import BrowserManager
    from manage.web import threads
    thread_root = tmp_path / "threads"
    thread_root.mkdir()
    monkeypatch.setattr(threads.MANAGER, "root_dir", str(thread_root))
    monkeypatch.setattr(threads, "DOMAINS", [str(source)])
    monkeypatch.setenv("ASSIST_DOMAINS", str(source))
    monkeypatch.setattr(BrowserManager, "confirm_owner_stopped", lambda *args, **kwargs: False)
    response = asyncio.run(threads.create_thread(str(source), "deepagents"))
    tid = response.headers["location"].removeprefix("/thread/")
    worktree = threads.MANAGER.thread_default_working_dir(tid)
    branch, revision = identity(worktree)
    assert branch != "main"
    assert git("rev-parse", "refs/heads/" + branch, cwd=source) == revision


def test_new_thread_does_not_advance_an_existing_remote_branch(
        repositories, tmp_path, monkeypatch):
    source, server, _ = repositories
    old_remote = git("rev-parse", "refs/heads/assist/thread", cwd=source)
    commit(server, "later.txt", "new main tip\n")
    git("push", "origin", "HEAD:main", cwd=server)
    directory = tmp_path / "new-thread"
    directory.mkdir()
    from assist.browser.manager import BrowserManager
    monkeypatch.setattr(BrowserManager, "confirm_owner_stopped", lambda *args, **kwargs: False)
    monkeypatch.setenv("ASSIST_DOMAINS", str(source))
    bind(str(directory), str(source))
    authorize_branch(str(directory), str(server))
    with pytest.raises(GitSyncError, match="already names different work"):
        publish_initial_branch(str(directory), str(server))
    assert git("rev-parse", "refs/heads/assist/thread", cwd=source) == old_remote


def test_unconfirmed_browser_generation_is_a_git_hold(bound_thread, monkeypatch):
    owner, _, server, _ = bound_thread
    from assist.browser.manager import BrowserManager, BrowserUnavailable
    def unavailable(*_args, **_kwargs):
        raise BrowserUnavailable("generation scan is incomplete")
    monkeypatch.setattr(BrowserManager, "confirm_owner_stopped", unavailable)
    lifecycle = ThreadGitLifecycle(owner, str(server), (str(server.parent), "thread"))
    with pytest.raises(GitSyncError, match="generation needs verification"):
        with lifecycle._host_fence():
            pass


def test_server_dirty_work_is_not_overwritten_by_phone_push(bound_thread):
    owner, _, server, phone = bound_thread
    (server / "unsaved.txt").write_text("local bytes\n")
    commit(phone, "published.txt", "published\n")
    git("push", "origin", "assist/thread", cwd=phone)
    before = (server / "unsaved.txt").read_bytes()
    plan = owner.plan_prepare()
    with pytest.raises(Exception, match="uncommitted changes"):
        owner.apply_prepare(LocalSandbox(), plan)
    assert (server / "unsaved.txt").read_bytes() == before


def test_dirty_retained_commit_cannot_publish_before_preflight(bound_thread, monkeypatch):
    owner, source, server, _ = bound_thread
    commit(server, "retained.txt", "committed result\n")
    (server / "unsaved.txt").write_text("user draft\n")
    original_remote = git("rev-parse", "refs/heads/assist/thread", cwd=source)
    lifecycle = ThreadGitLifecycle(owner, str(server))
    import assist.git_sync as git_sync
    monkeypatch.setattr(lifecycle, "_verify_clean",
                        lambda _: git_sync.require_clean(LocalSandbox()))
    with pytest.raises(Exception, match="uncommitted changes"):
        lifecycle.prepare(None)
    assert git("rev-parse", "refs/heads/assist/thread", cwd=source) == original_remote
    assert (server / "unsaved.txt").read_text() == "user draft\n"


def test_child_handoff_keeps_parent_base_and_holds_partial_edits(bound_thread, monkeypatch):
    owner, source, server, phone = bound_thread
    original = identity(str(server))[1]
    later = commit(phone, "phone.txt", "phone while child ran\n")
    git("push", "origin", "assist/thread", cwd=phone)
    lifecycle = ThreadGitLifecycle(owner, str(server))
    import assist.git_sync as git_sync
    monkeypatch.setattr(lifecycle, "_verify_clean",
                        lambda _: git_sync.require_clean(LocalSandbox()))
    lifecycle.handoff(None)
    assert identity(str(server))[1] == original
    assert git("rev-parse", "refs/heads/assist/thread", cwd=source) == later
    (server / "partial.txt").write_text("failed child draft\n")
    with pytest.raises(Exception, match="uncommitted changes"):
        lifecycle.handoff(None)
    assert (server / "partial.txt").read_text() == "failed child draft\n"
    assert identity(str(server))[1] == original


def test_merge_uses_published_history_only_and_preserves_checkout(
        bound_thread, tmp_path, monkeypatch):
    owner, source, server, phone = bound_thread
    thread_tip = commit(server, "published.txt", "thread\n")
    assert owner.publish() is None
    (server / "uncommitted.txt").write_text("server draft\n")
    phone_draft = (phone / "phone-draft.txt")
    phone_draft.write_text("phone draft\n")
    before_index = (server / ".git" / "index").read_bytes()
    main_client = tmp_path / "main-client"
    git("clone", "--no-hardlinks", "-b", "main", source, main_client)
    main_tip = commit(main_client, "main.txt", "main\n")
    git("push", "origin", "main", cwd=main_client)

    lifecycle = ThreadGitLifecycle(owner, str(server), (str(tmp_path), "thread"))
    monkeypatch.setattr(lifecycle, "_host_fence", nullcontext)
    monkeypatch.setattr(lifecycle, "_backend", lambda _: LocalSandbox())
    monkeypatch.setattr(lifecycle, "_cleanup", lambda _: None)
    assert lifecycle.merge_and_push() == "Merged published thread assist/thread"
    merged = git("rev-parse", "refs/heads/main", cwd=source)
    assert merged not in {thread_tip, main_tip}
    assert git("merge-base", "--is-ancestor", thread_tip, merged, cwd=source) == ""
    assert git("merge-base", "--is-ancestor", main_tip, merged, cwd=source) == ""
    assert git("rev-parse", "refs/heads/assist/thread", cwd=source) == thread_tip
    assert identity(str(server))[1] == thread_tip
    assert (server / "uncommitted.txt").read_bytes() == b"server draft\n"
    assert phone_draft.read_bytes() == b"phone draft\n"
    assert (server / ".git" / "index").read_bytes() == before_index
    assert git("cat-file", "-e", merged + "^{commit}", cwd=server) == ""
    assert [change.path for change in git_diff_main(str(server), merged)] == [
        "uncommitted.txt"]


def test_merge_conflict_leaves_main_and_thread_unchanged(bound_thread, tmp_path):
    owner, source, server, _ = bound_thread
    commit(server, "README", "thread\n")
    owner.publish()
    thread_tip = git("rev-parse", "refs/heads/assist/thread", cwd=source)
    main_client = tmp_path / "main-client"
    git("clone", "--no-hardlinks", "-b", "main", source, main_client)
    main_tip = commit(main_client, "README", "main\n")
    git("push", "origin", "main", cwd=main_client)
    with pytest.raises(Exception, match="conflicted"):
        owner.plan_merge_main()
    assert git("rev-parse", "refs/heads/main", cwd=source) == main_tip
    assert git("rev-parse", "refs/heads/assist/thread", cwd=source) == thread_tip


def test_merge_import_failure_does_not_publish_main(bound_thread):
    owner, source, server, _ = bound_thread
    commit(server, "thread.txt", "thread\n")
    owner.publish()
    original = git("rev-parse", "refs/heads/main", cwd=source)
    plan = owner.plan_merge_main()
    with pytest.raises(GitSyncError, match="restricted sandbox"):
        owner.import_merge(None, plan)
    assert git("rev-parse", "refs/heads/main", cwd=source) == original


def test_merge_teardown_failure_does_not_publish_main(
        bound_thread, tmp_path, monkeypatch):
    owner, source, server, _ = bound_thread
    commit(server, "thread.txt", "thread\n")
    owner.publish()
    original = git("rev-parse", "refs/heads/main", cwd=source)
    lifecycle = ThreadGitLifecycle(owner, str(server), (str(tmp_path), "thread"))
    monkeypatch.setattr(lifecycle, "_host_fence", nullcontext)
    monkeypatch.setattr(lifecycle, "_backend", lambda _: LocalSandbox())

    def fail_cleanup(_):
        raise GitSyncError("Git sandbox teardown needs operator verification")

    monkeypatch.setattr(lifecycle, "_cleanup", fail_cleanup)
    with pytest.raises(GitSyncError, match="teardown"):
        lifecycle.merge_and_push()
    assert git("rev-parse", "refs/heads/main", cwd=source) == original


def test_merge_holds_when_published_thread_advances_during_import(
        bound_thread, tmp_path):
    owner, source, server, phone = bound_thread
    commit(server, "thread.txt", "thread\n")
    owner.publish()
    original_main = git("rev-parse", "refs/heads/main", cwd=source)
    plan = owner.plan_merge_main()
    owner.import_merge(LocalSandbox(), plan)
    git("pull", "--ff-only", "origin", "assist/thread", cwd=phone)
    commit(phone, "phone.txt", "later\n")
    git("push", "origin", "assist/thread", cwd=phone)
    with pytest.raises(GitSyncError, match="advanced"):
        owner.publish_merge(plan)
    assert git("rev-parse", "refs/heads/main", cwd=source) == original_main
