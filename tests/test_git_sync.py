"""Real two-clone Git probes; the local backend is a disposable test double only."""
import os
from pathlib import Path
import shlex
import subprocess
import zlib
from types import SimpleNamespace

import pytest

from assist import git_sync as sync


def git(path, *arguments):
    return subprocess.check_output(
        ["git", "-C", str(path), *arguments], stderr=subprocess.PIPE, text=True).strip()


class LocalBackend:
    def __init__(self, path):
        self.path = path

    def execute(self, command):
        # Translate only the fixed transfer path to this disposable backend's directory.
        command = command.replace("/tmp/assist-git-", str(self.path.parent / "assist-git-"))
        command = command.replace("/workspace", str(self.path))
        result = subprocess.run(command, cwd=self.path, shell=True, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)
        return SimpleNamespace(exit_code=result.returncode, output=result.stdout)

    def upload_files(self, files):
        for path, content in files:
            Path(path.replace("/tmp/assist-git-", str(self.path.parent / "assist-git-"))).write_bytes(content)
        return [SimpleNamespace(error=None)]


@pytest.fixture
def repos(tmp_path):
    remote, seed, thread, phone = [tmp_path / name for name in ("remote", "seed", "thread", "phone")]
    subprocess.run(["git", "init", "--bare", "--initial-branch=main", str(remote)], check=True,
                   stdout=subprocess.DEVNULL)
    subprocess.run(["git", "clone", str(remote), str(seed)], check=True, capture_output=True)
    git(seed, "config", "user.email", "test@example.invalid")
    git(seed, "config", "user.name", "Test")
    (seed / "tracked").write_text("base\n")
    git(seed, "add", "tracked")
    git(seed, "commit", "-m", "base")
    git(seed, "push", "origin", "main")
    for path in (thread, phone):
        subprocess.run(["git", "clone", str(remote), str(path)], check=True, capture_output=True)
        git(path, "config", "user.email", "test@example.invalid")
        git(path, "config", "user.name", "Test")
    git(thread, "checkout", "-b", "thread/test")
    binding = tmp_path / "state"
    binding.mkdir()
    sync.bind(str(binding), str(remote))
    sync.authorize_branch(str(binding), str(thread))
    return remote, thread, phone, binding


def turn(repos):
    _, thread, _, binding = repos
    owner = sync.GitSync(str(binding), str(thread))
    backend = LocalBackend(thread)
    owner.prepare(backend, "work-1")
    return owner, backend


def test_first_noop_publishes_only_thread_branch(repos):
    remote, thread, _, binding = repos
    main = git(remote, "rev-parse", "main")
    owner, backend = turn(repos)
    owner.commit(backend, "no changes")
    owner.publish()
    head = git(thread, "rev-parse", "HEAD")
    assert git(remote, "rev-parse", "thread/test") == head == main
    assert git(remote, "for-each-ref", "--format=%(refname)") == "refs/heads/main\nrefs/heads/thread/test"
    assert sync.workspace(str(binding), str(thread))["published_revision"] == head


def test_changed_and_staged_only_turn(repos):
    remote, thread, _, _ = repos
    owner, backend = turn(repos)
    (thread / "new").write_text("staged\n")
    git(thread, "add", "new")
    owner.commit(backend, "new file")
    owner.publish()
    assert git(remote, "show", "thread/test:new") == "staged"
    assert git(remote, "rev-parse", "main") != git(remote, "rev-parse", "thread/test")


def test_phone_commit_fast_forwarded_before_next_turn(repos):
    remote, thread, phone, _ = repos
    owner, backend = turn(repos)
    owner.commit(backend, "noop")
    owner.publish()
    git(phone, "fetch", "origin", "thread/test")
    git(phone, "checkout", "-b", "thread/test", "FETCH_HEAD")
    (phone / "phone").write_text("user change\n")
    git(phone, "add", "phone")
    git(phone, "commit", "-m", "phone")
    git(phone, "push", "origin", "thread/test")
    owner, _ = turn(repos)
    assert git(thread, "rev-parse", "HEAD") == git(remote, "rev-parse", "thread/test")
    assert (thread / "phone").read_text() == "user change\n"
    owner.publish()


def test_remote_movement_during_turn_never_rewrites_phone(repos):
    remote, thread, phone, _ = repos
    owner, backend = turn(repos)
    owner.publish()
    owner, backend = turn(repos)
    git(phone, "fetch", "origin", "thread/test")
    git(phone, "checkout", "-b", "thread/test", "FETCH_HEAD")
    (phone / "phone").write_text("user\n")
    git(phone, "add", ".")
    git(phone, "commit", "-m", "phone")
    git(phone, "push", "origin", "thread/test")
    user_tip = git(remote, "rev-parse", "thread/test")
    (thread / "assistant").write_text("assistant\n")
    owner.commit(backend, "assistant")
    with pytest.raises(sync.GitSyncError, match="advanced"):
        owner.publish()
    assert git(remote, "rev-parse", "thread/test") == user_tip
    with pytest.raises(sync.GitSyncError, match="diverged"):
        turn(repos)


def test_later_noop_retries_prior_failed_push(repos, monkeypatch):
    owner, backend = turn(repos)
    owner.commit(backend, "noop")
    original = sync._Store.git

    def fail_push(store, *args, **kwargs):
        if args[0] == "push":
            return "!\tHEAD:refs/heads/thread/test\t[remote rejected]"
        return original(store, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(sync._Store, "git", fail_push)
        with pytest.raises(sync.GitSyncError):
            owner.publish()
    later, backend = turn(repos)
    later.commit(backend, "noop")
    later.publish()
    assert later.state["published_revision"] == git(repos[1], "rev-parse", "HEAD")


@pytest.mark.parametrize("entry", ["dirty", "deleted", "rewritten"])
def test_reconciliation_holds_preserve_local_and_remote(repos, entry):
    remote, thread, _, _ = repos
    owner, backend = turn(repos)
    (thread / "assistant").write_text("assistant\n")
    owner.commit(backend, "assistant")
    owner.publish()
    local = git(thread, "rev-parse", "HEAD")
    if entry == "dirty":
        (thread / "dirty").write_text("preserve\n")
    elif entry == "deleted":
        git(remote, "update-ref", "-d", "refs/heads/thread/test")
    else:
        git(remote, "update-ref", "refs/heads/thread/test", git(remote, "rev-parse", "main"))
    refs = git(remote, "for-each-ref", "--format=%(refname) %(objectname)")
    with pytest.raises(sync.GitSyncError):
        turn(repos)
    assert git(thread, "rev-parse", "HEAD") == local
    assert git(remote, "for-each-ref", "--format=%(refname) %(objectname)") == refs
    if entry == "dirty":
        assert (thread / "dirty").read_text() == "preserve\n"


def test_host_ignores_workspace_execution_config_and_pushurl(repos, tmp_path):
    remote, thread, _, _ = repos
    owner, backend = turn(repos)
    owner.commit(backend, "noop")
    marker = tmp_path / "host-command-ran"
    malicious = "touch " + shlex.quote(str(marker))
    git(thread, "config", "core.sshCommand", malicious)
    git(thread, "config", "core.hooksPath", str(tmp_path))
    git(thread, "config", "remote.origin.pushurl", "ext::" + malicious)
    git(thread, "config", "remote.origin.url", "ext::" + malicious)
    git(thread, "config", "core.fsmonitor", malicious)
    owner.publish()
    assert not marker.exists()
    assert git(remote, "rev-parse", "thread/test") == owner.state["published_revision"]


@pytest.mark.parametrize("kind", ["symlink", "fifo", "limit"])
def test_object_snapshot_fails_closed(repos, tmp_path, monkeypatch, kind):
    _, thread, _, _ = repos
    if kind == "limit":
        monkeypatch.setattr(sync, "MAX_BYTES", 1)
    else:
        directory = thread / ".git" / "objects" / "aa"
        directory.mkdir(exist_ok=True)
        target = directory / ("a" * 38)
        if kind == "fifo":
            os.mkfifo(target)
        else:
            target.symlink_to(thread / ".git" / "config")
    with pytest.raises((sync.GitSyncError, OSError)):
        with sync.tempfile.TemporaryDirectory() as path:
            sync._Store(path).snapshot(str(thread))


def test_main_detached_missing_binding_and_quarantine(repos):
    _, thread, _, binding = repos
    git(thread, "checkout", "main")
    with pytest.raises(sync.GitSyncError, match="non-main"):
        sync.identity(str(thread))
    git(thread, "checkout", "--detach")
    with pytest.raises(sync.GitSyncError, match="non-main"):
        sync.identity(str(thread))
    git(thread, "checkout", "thread/test")
    owner = sync.GitSync(str(binding), str(thread))
    owner.quarantine()
    with pytest.raises(sync.GitSyncError, match="teardown"):
        sync.GitSync(str(binding), str(thread))
    (binding / "git-sync.json").unlink()
    with pytest.raises(sync.GitSyncError, match="operator"):
        sync.GitSync(str(binding), str(thread))


@pytest.mark.parametrize("source", ["ext::sh -c command", "evil::remote", "https://secret@example.test/repo", "-bad"])
def test_binding_rejects_helpers_and_url_secrets(tmp_path, source):
    with pytest.raises(sync.GitSyncError):
        sync.bind(str(tmp_path), source)


def test_timeout_is_bounded_and_terminates_git(repos, monkeypatch):
    monkeypatch.setattr(sync, "GIT_TIMEOUT", 0.001)
    with sync.tempfile.TemporaryDirectory() as path:
        # init is allowed to finish independently of the operation being tested.
        with monkeypatch.context() as patch:
            patch.setattr(sync, "GIT_TIMEOUT", 30)
            store = sync._Store(path)
        with pytest.raises(sync.GitSyncError, match="timed out"):
            store.git("-c", "alias.slow=!sleep 10", "slow")


def test_crash_after_push_recovers_verified_publication(repos, monkeypatch):
    owner, backend = turn(repos)
    owner.commit(backend, "noop")
    original = sync._write_state

    class Crash(BaseException):
        pass

    def crash_after_push(directory, value):
        if value.get("published_revision"):
            raise Crash
        return original(directory, value)

    with monkeypatch.context() as patch:
        patch.setattr(sync, "_write_state", crash_after_push)
        with pytest.raises(Crash):
            owner.publish()
    later, _ = turn(repos)
    assert later.state["intent"] is None
    assert later.state["published_revision"] == git(repos[0], "rev-parse", "thread/test")


def test_ambiguous_push_and_deleted_remote_hold(repos, monkeypatch):
    owner, _ = turn(repos)
    original = sync._Store.git

    def lost_response(store, *args, **kwargs):
        if args[0] == "push":
            original(store, *args, **kwargs)
            raise sync.GitSyncError("response lost")
        return original(store, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(sync._Store, "git", lost_response)
        with pytest.raises(sync.GitSyncError):
            owner.publish()
    git(repos[0], "update-ref", "-d", "refs/heads/thread/test")
    with pytest.raises(sync.GitSyncError, match="unknown"):
        turn(repos)
    assert git(repos[0], "for-each-ref", "--format=%(refname)") == "refs/heads/main"


def test_workspace_fence_and_branch_authority(repos):
    _, thread, _, binding = repos
    with sync.ownership(str(binding), str(thread)):
        with pytest.raises(sync.GitSyncError, match="owns"):
            with sync.ownership(str(binding), str(thread)):
                pytest.fail("Overlapping owner admitted")
    git(thread, "checkout", "-b", "another-user-branch")
    with pytest.raises(sync.GitSyncError, match="changed"):
        turn(repos)


def test_corrupt_binding_and_no_sandbox_fail_closed(repos):
    owner = sync.GitSync(str(repos[3]), str(repos[1]))
    with pytest.raises(sync.GitSyncError, match="sandbox"):
        owner.prepare(None, "work")
    state = sync.read_state(str(repos[3]))
    state["published"] = {"thread/test": "not an oid"}
    sync._write_state(str(repos[3]), state)
    with pytest.raises(sync.GitSyncError, match="binding"):
        sync.GitSync(str(repos[3]), str(repos[1]))


def test_forged_object_filename_cannot_authenticate_ancestry(repos):
    _, thread, _, _ = repos
    revision = git(thread, "rev-parse", "HEAD")
    path = thread / ".git" / "objects" / revision[:2] / revision[2:]
    original = zlib.decompress(path.read_bytes())
    path.unlink()  # Local clone objects can be read-only hardlinks in this fixture.
    path.write_bytes(zlib.compress(original.replace(b"base", b"fake")))
    with pytest.raises(sync.GitSyncError):
        with sync.tempfile.TemporaryDirectory() as directory:
            sync._Store(directory).snapshot(str(thread))


def test_scp_label_and_legacy_metadata_do_not_expose_source(repos):
    assert sync.source_label("git@secret-host:repo.git") == "repo"
    _, thread, _, binding = repos
    (binding / "git-sync.json").unlink()
    value = sync.workspace(str(binding), str(thread))
    assert value["repo_key"] is None
    assert "operator binding" in value["repo_label"]
    assert value["sync_error"]


def test_crash_before_sandbox_teardown_blocks_new_writer(repos):
    owner = sync.GitSync(str(repos[3]), str(repos[1]))
    owner.sandbox_started()
    with pytest.raises(sync.GitSyncError, match="teardown"):
        sync.GitSync(str(repos[3]), str(repos[1]))
    owner.sandbox_stopped()
    sync.GitSync(str(repos[3]), str(repos[1]))


def test_local_domain_clone_does_not_share_mutable_objects(repos, tmp_path):
    from assist.domain_manager import clone_repo
    target = tmp_path / "isolated"
    clone_repo(str(repos[0]), str(target), branch_suffix="probe", timeout_s=10)
    revision = git(target, "rev-parse", "HEAD")
    source = repos[0] / "objects" / revision[:2] / revision[2:]
    copied = target / ".git" / "objects" / revision[:2] / revision[2:]
    assert source.stat().st_ino != copied.stat().st_ino


def web_turn(repos, monkeypatch, model):
    """Run the real Deep turn workflow; only model and Docker adapters are doubled."""
    from manage.web import threads
    remote, thread, _, binding = repos
    monkeypatch.setattr(threads.MANAGER, "root_dir", str(binding.parent))
    monkeypatch.setattr(threads.MANAGER, "thread_dir", lambda _tid: str(binding))
    monkeypatch.setattr(threads.MANAGER, "thread_default_working_dir", lambda _tid: str(thread))
    monkeypatch.setattr(threads.MANAGER, "touch", lambda _tid: None)
    monkeypatch.setattr(threads, "get_cached_description", lambda _tid: "probe")
    monkeypatch.setattr(threads, "_pending_email", lambda _chat: None)
    monkeypatch.setattr(threads, "_claim_seen_interjections", lambda *_: None)
    monkeypatch.setattr(threads, "_dispatch_pending_after", lambda *_: None)
    outcomes = []
    monkeypatch.setattr(threads, "_notify_turn_observers", lambda *args: outcomes.append(args))
    monkeypatch.setattr(threads, "_get_domain_manager",
                        lambda *_: (_ for _ in ()).throw(AssertionError("No host worktree Git")))

    class Chat:
        def message(self, _text):
            return model()

        def pending_reply(self):
            return None

        def get_messages(self):
            return [{"role": "user", "content": "probe"},
                    {"role": "assistant", "content": "saved answer"}]

    monkeypatch.setattr(threads.MANAGER, "get", lambda *_args, **_kwargs: Chat())
    events = []

    def backend(*_args, **_kwargs):
        value = LocalBackend(thread)
        value.container = object()
        threads.SandboxManager._containers[str(thread)] = value.container
        events.append(("start", value.container))
        return value

    def cleanup(worktree, generation):
        assert threads.SandboxManager._containers[worktree] is generation
        events.append(("exit", generation))
        threads.SandboxManager._containers.pop(worktree)

    monkeypatch.setattr(threads.SandboxManager, "get_pi_sandbox_backend", backend)
    monkeypatch.setattr(threads, "_get_sandbox_backend", backend)
    monkeypatch.setattr(threads.SandboxManager, "cleanup_verified", cleanup)
    return threads, events, outcomes


def test_deep_turn_noop_publishes_before_releasing_workspace(repos, monkeypatch):
    threads, events, outcomes = web_turn(repos, monkeypatch, lambda: "saved answer")
    threads._process_message("state", "probe")
    assert outcomes[-1][1:4] == ("ready", None, "saved answer")
    assert [event[0] for event in events] == ["start", "exit"] * 3
    assert len({event[1] for event in events}) == 3
    assert git(repos[0], "rev-parse", "thread/test") == git(repos[1], "rev-parse", "HEAD")
    assert not sync.read_state(str(repos[3]))["sandbox_in_flight"]


def test_deep_turn_incorporates_phone_commit_before_model(repos, monkeypatch):
    owner, _ = turn(repos)
    owner.publish()
    remote, thread, phone, _ = repos
    git(phone, "fetch", "origin", "thread/test")
    git(phone, "checkout", "-b", "thread/test", "FETCH_HEAD")
    (phone / "phone").write_text("user before model\n")
    git(phone, "add", ".")
    git(phone, "commit", "-m", "user")
    git(phone, "push", "origin", "thread/test")
    expected = git(remote, "rev-parse", "thread/test")

    def model():
        assert git(thread, "rev-parse", "HEAD") == expected
        assert (thread / "phone").read_text() == "user before model\n"
        (thread / "assistant").write_text("agent after user\n")
        return "saved answer"

    threads, _, outcomes = web_turn(repos, monkeypatch, model)
    threads._process_message("state", "probe")
    assert outcomes[-1][1] == "ready"
    assert git(remote, "show", "thread/test:assistant") == "agent after user"
    assert git(remote, "merge-base", expected, "thread/test") == expected


def test_deep_missing_sandbox_never_runs_model_or_host_sync(repos, monkeypatch):
    threads, _, outcomes = web_turn(repos, monkeypatch,
                                   lambda: pytest.fail("Model ran without Git preflight"))
    monkeypatch.setattr(threads.SandboxManager, "get_pi_sandbox_backend", lambda *_a, **_k: None)
    threads._process_message("state", "probe")
    assert outcomes[-1][1] == "error"
    assert sync.read_state(str(repos[3]))["sandbox_in_flight"]
    assert git(repos[0], "for-each-ref", "--format=%(refname)") == "refs/heads/main"


def test_failed_model_teardown_preserves_answer_and_quarantines(repos, monkeypatch):
    threads, events, outcomes = web_turn(repos, monkeypatch, lambda: "saved answer")
    original = threads.SandboxManager.cleanup_verified

    def fail_model_cleanup(worktree, generation):
        if len(events) == 3:
            raise RuntimeError("test daemon failure")
        return original(worktree, generation)

    monkeypatch.setattr(threads.SandboxManager, "cleanup_verified", fail_model_cleanup)
    threads._process_message("state", "probe")
    assert outcomes[-1][1:4] == ("ready", None, "saved answer")
    assert sync.read_state(str(repos[3]))["quarantine"]
    assert len(events) == 3  # No Git-only generation or host publication after failed exit.
    assert git(repos[0], "for-each-ref", "--format=%(refname)") == "refs/heads/main"
    # Test double has no actual container; remove only its private registry entry.
    threads.SandboxManager._containers.pop(str(repos[1]), None)


def test_pi_turn_uses_same_preflight_commit_and_publication(repos, monkeypatch):
    from assist.pi_runtime import PiRuntimeResult
    from assist.pi_conversation import PiConversationStore
    threads, events, _ = web_turn(repos, monkeypatch, lambda: "unused Deep model")
    monkeypatch.setattr(threads, "PI_PREVIEW", SimpleNamespace(admits=lambda _engine: True))
    monkeypatch.setattr(threads, "_pi_system_prompt", lambda: "be useful")
    monkeypatch.setattr(threads, "_PI_CONVERSATIONS", PiConversationStore())
    result = PiRuntimeResult("saved Pi answer", 1)

    def runtime(**kwargs):
        pending = sync.read_state(str(repos[3]))["pending"]
        assert pending["base"] == git(repos[1], "rev-parse", "HEAD")
        model = threads._get_sandbox_backend("state")
        (repos[1] / "pi").write_text("Pi committed change\n")
        kwargs["commit"](result)
        kwargs["sandbox_cleanup"](model.container)
        return result

    monkeypatch.setattr(threads, "_PI_RUNTIME", SimpleNamespace(run=runtime))
    run = threads._create_run("state", "probe")
    threads._execute_pi_run(run, user_priority=False)
    assert threads._runs().get("state", run.id).status == "success"
    assert git(repos[0], "show", "thread/test:pi") == "Pi committed change"
    assert [event[0] for event in events] == ["start", "exit"] * 3
    saved = threads._PI_CONVERSATIONS.completed_reply(str(repos[3]), run.id)
    assert saved.text == "saved Pi answer"


def test_verified_cleanup_keeps_generation_on_failure(monkeypatch):
    from assist.sandbox_manager import SandboxManager

    class Container:
        def kill(self):
            raise RuntimeError("daemon failure")

    container = Container()
    monkeypatch.setattr(SandboxManager, "_containers", {"probe": container})
    with pytest.raises(RuntimeError, match="daemon"):
        SandboxManager.cleanup_verified("probe", container)
    assert SandboxManager.current_container("probe") is container


def test_worktree_redirection_and_hidden_index_cannot_hide_dirty_files(repos, tmp_path):
    _, thread, _, _ = repos
    other = tmp_path / "other"
    other.mkdir()
    (other / "tracked").write_text("base\n")
    git(thread, "config", "core.worktree", str(other))
    (thread / "tracked").write_text("user dirty work\n")
    with pytest.raises(sync.GitSyncError, match="uncommitted"):
        turn(repos)
    git(thread, "config", "--unset", "core.worktree")
    git(thread, "update-index", "--assume-unchanged", "tracked")
    with pytest.raises(sync.GitSyncError, match="hidden"):
        turn(repos)
    assert (thread / "tracked").read_text() == "user dirty work\n"
