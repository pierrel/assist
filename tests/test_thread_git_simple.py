"""Real disposable Git remotes for the stateless thread publication path."""

from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from assist.git_sync import GitSyncError, _write_state, bind, authorize_branch, read_state, workspace
from assist import git_sync as sync
from assist.thread_git import ThreadGit
from manage.web.thread_git_lifecycle import preflight_fence_error


def git(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


class LocalBackend:
    """Translate the fixed container paths for a disposable local Git fixture."""

    def __init__(self, path):
        self.path = path

    def execute(self, command):
        command = command.replace("/tmp/assist-git-", str(self.path.parent / "assist-git-"))
        command = command.replace("/workspace", str(self.path))
        result = subprocess.run(command, cwd=self.path, shell=True, capture_output=True,
                                text=True, timeout=30)
        return SimpleNamespace(exit_code=result.returncode, truncated=False)

    def upload_files(self, files):
        for path, content in files:
            Path(path.replace("/tmp/assist-git-", str(self.path.parent / "assist-git-"))).write_bytes(content)
        return [SimpleNamespace(error=None)]


@pytest.fixture
def repositories(tmp_path):
    remote, seed, server, phone, binding = [tmp_path / name for name in
                                            ("remote", "seed", "server", "phone", "binding")]
    subprocess.run(["git", "init", "--bare", "--initial-branch=main", str(remote)],
                   check=True, capture_output=True)
    subprocess.run(["git", "clone", str(remote), str(seed)], check=True, capture_output=True)
    for name in ("user.email", "user.name"):
        git(seed, "config", name, "test@example.invalid" if name.endswith("email") else "Test")
    (seed / "tracked").write_text("base\n")
    git(seed, "add", "tracked")
    git(seed, "commit", "-m", "base")
    git(seed, "push", "origin", "main")
    for checkout in (server, phone):
        subprocess.run(["git", "clone", "--no-hardlinks", str(remote), str(checkout)],
                       check=True, capture_output=True)
        git(checkout, "config", "user.email", "test@example.invalid")
        git(checkout, "config", "user.name", "Test")
    git(server, "checkout", "-b", "assist/test")
    binding.mkdir()
    bind(str(binding), str(remote))
    authorize_branch(str(binding), str(server))
    return remote, server, phone, binding


def owner(repositories):
    remote, server, _, binding = repositories
    return ThreadGit(str(binding), str(server), (str(remote),))


def test_phone_push_is_fast_forwarded_before_server_turn_and_published(repositories):
    remote, server, phone, binding = repositories
    binding_before = read_state(str(binding))
    initial = owner(repositories)
    initial.prepare(LocalBackend(server))
    initial.publish()
    git(phone, "fetch", "origin", "assist/test")
    git(phone, "checkout", "-b", "assist/test", "origin/assist/test")
    (phone / "phone").write_text("from phone\n")
    git(phone, "add", "phone")
    git(phone, "commit", "-m", "phone edit")
    git(phone, "push", "origin", "assist/test")
    phone_tip = git(phone, "rev-parse", "HEAD")

    next_turn = owner(repositories)
    next_turn.prepare(LocalBackend(server))
    assert git(server, "rev-parse", "HEAD") == phone_tip
    (server / "server").write_text("from server\n")
    next_turn.commit(LocalBackend(server), "server turn")
    next_turn.publish()
    final = git(server, "rev-parse", "HEAD")
    assert git(remote, "rev-parse", "refs/heads/assist/test") == final
    git(phone, "pull", "--ff-only", "origin", "assist/test")
    assert git(phone, "rev-parse", "HEAD") == final
    assert (phone / "server").read_text() == "from server\n"
    assert read_state(str(binding)) == binding_before


def test_remote_advance_during_turn_never_rewrites_user_commit(repositories):
    remote, server, phone, _ = repositories
    turn = owner(repositories)
    turn.prepare(LocalBackend(server))
    turn.publish()
    git(phone, "fetch", "origin", "assist/test")
    git(phone, "checkout", "-b", "assist/test", "origin/assist/test")
    (server / "server").write_text("server work\n")
    turn.commit(LocalBackend(server), "server work")
    local = git(server, "rev-parse", "HEAD")
    (phone / "phone").write_text("phone work\n")
    git(phone, "add", "phone")
    git(phone, "commit", "-m", "phone work")
    git(phone, "push", "origin", "assist/test")
    remote_tip = git(remote, "rev-parse", "refs/heads/assist/test")
    with pytest.raises(GitSyncError, match="advanced"):
        turn.publish()
    assert git(remote, "rev-parse", "refs/heads/assist/test") == remote_tip
    assert git(server, "rev-parse", "HEAD") == local
    assert (server / "server").read_text() == "server work\n"


def test_first_turn_publication_lease_rejects_concurrent_ancestor_ref(
        repositories, monkeypatch):
    remote, server, phone, binding = repositories
    branch = git(server, "symbolic-ref", "--short", "HEAD")
    ancestor = git(server, "rev-parse", "HEAD")
    (server / "server").write_text("first turn\n")
    git(server, "add", "server")
    git(server, "commit", "-m", "first turn")
    desired = git(server, "rev-parse", "HEAD")
    original_git = sync._Store.git

    def create_ref_before_push(store, *arguments, **kwargs):
        if arguments[0] == "push":
            git(phone, "push", "origin", ancestor + ":refs/heads/" + branch)
        return original_git(store, *arguments, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(sync._Store, "git", create_ref_before_push)
        with pytest.raises(GitSyncError, match="pending"):
            owner(repositories).publish()

    assert git(remote, "rev-parse", "refs/heads/" + branch) == ancestor
    assert git(server, "rev-parse", "HEAD") == desired
    assert read_state(str(binding))["local_revision"] == ancestor


def test_established_thread_recreates_absent_remote_ref_without_rewrite(repositories):
    remote, server, _, _ = repositories
    first = owner(repositories)
    first.prepare(LocalBackend(server))
    first.publish()
    branch = git(server, "symbolic-ref", "--short", "HEAD")
    revision = git(server, "rev-parse", "HEAD")
    git(remote, "update-ref", "-d", "refs/heads/" + branch)

    later = owner(repositories)
    assert later.prepare(LocalBackend(server)) is False
    later.publish()

    assert git(remote, "rev-parse", "refs/heads/" + branch) == revision


def test_dirty_equal_tip_preflight_preserves_untracked_work(repositories):
    _, server, _, _ = repositories
    first = owner(repositories)
    first.prepare(LocalBackend(server))
    first.publish()
    (server / "untracked").write_bytes(b"user bytes\x00")
    assert owner(repositories).prepare(LocalBackend(server)) is True
    assert (server / "untracked").read_bytes() == b"user bytes\x00"


def test_legacy_publication_records_do_not_hold_dirty_equal_tip(repositories):
    _, server, _, binding = repositories
    first = owner(repositories)
    first.prepare(LocalBackend(server))
    first.publish()
    branch, revision = git(server, "symbolic-ref", "--short", "HEAD"), git(server, "rev-parse", "HEAD")
    state = read_state(str(binding))
    state["preflights"] = {
        name: {"branch": branch, "base": revision, "expected": revision}
        for name in ("terminal-one", "terminal-two", "terminal-three")
    }
    state.update(intent={"branch": branch, "expected": revision, "desired": revision},
                 sandbox_in_flight=False, quarantine=None)
    _write_state(str(binding), state)
    (server / "untracked").write_bytes(b"retained user work\x00")
    before_index = (server / ".git" / "index").read_bytes()

    assert preflight_fence_error(str(binding), str(server)) is None
    assert owner(repositories).prepare(LocalBackend(server)) is True
    assert (server / "untracked").read_bytes() == b"retained user work\x00"
    assert (server / ".git" / "index").read_bytes() == before_index


def test_phone_metadata_selects_bound_branch_not_old_publication_record(repositories):
    _, server, _, binding = repositories
    branch, revision = git(server, "symbolic-ref", "--short", "HEAD"), git(server, "rev-parse", "HEAD")
    state = read_state(str(binding))
    state.update(published={branch: revision}, published_branch=branch,
                 published_revision=revision, error="old publication error")
    _write_state(str(binding), state)
    ready = workspace(str(binding), str(server))
    assert ready["branch"] == ready["thread_branch"] == branch
    assert ready["revision"] == revision
    assert ready["sync_error"] is None
    assert "published_revision" not in ready

    git(server, "checkout", "main")
    busy = workspace(str(binding), str(server))
    assert busy["branch"] is None and busy["revision"] is None
    assert busy["thread_branch"] == branch
    assert busy["sync_error"] == "Thread checkout branch is unavailable"


def test_remote_advance_does_not_touch_dirty_checkout(repositories):
    _, server, phone, _ = repositories
    first = owner(repositories)
    first.prepare(LocalBackend(server))
    first.publish()
    git(phone, "fetch", "origin", "assist/test")
    git(phone, "checkout", "-b", "assist/test", "origin/assist/test")
    (server / "untracked").write_bytes(b"preserve me\x00")
    (phone / "phone").write_text("new phone commit\n")
    git(phone, "add", "phone")
    git(phone, "commit", "-m", "phone")
    git(phone, "push", "origin", "assist/test")
    before_head = git(server, "rev-parse", "HEAD")
    before_index = (server / ".git" / "index").read_bytes()
    with pytest.raises(GitSyncError, match="differs"):
        owner(repositories).prepare(LocalBackend(server))
    assert git(server, "rev-parse", "HEAD") == before_head
    assert (server / ".git" / "index").read_bytes() == before_index
    assert (server / "untracked").read_bytes() == b"preserve me\x00"


def test_hidden_index_flag_never_enters_dirty_continuation(repositories):
    _, server, _, _ = repositories
    first = owner(repositories)
    first.prepare(LocalBackend(server))
    first.publish()
    git(server, "update-index", "--skip-worktree", "tracked")
    (server / "untracked").write_text("still here\n")
    with pytest.raises(GitSyncError, match="Hidden Git index"):
        owner(repositories).prepare(LocalBackend(server))
    assert (server / "untracked").read_text() == "still here\n"


def test_other_named_branch_cannot_publish(repositories):
    remote, server, _, _ = repositories
    git(server, "checkout", "-b", "assist/other")
    with pytest.raises(GitSyncError, match="branch needs operator verification"):
        owner(repositories)
    assert subprocess.run(["git", "-C", str(remote), "show-ref", "--verify", "--quiet",
                           "refs/heads/assist/other"]).returncode != 0
