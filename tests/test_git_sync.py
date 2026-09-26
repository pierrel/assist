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
        subprocess.run(["git", "clone", "--no-hardlinks", str(remote), str(path)], check=True, capture_output=True)
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
        if _kwargs.get("before_start"):
            _kwargs["before_start"]()
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
    monkeypatch.setattr(threads.SandboxManager, "get_git_verification_backend", backend)
    monkeypatch.setattr(threads, "_get_sandbox_backend", backend)
    monkeypatch.setattr(threads.SandboxManager, "cleanup_verified", cleanup)
    return threads, events, outcomes


def test_deep_turn_noop_publishes_before_releasing_workspace(repos, monkeypatch):
    threads, events, outcomes = web_turn(repos, monkeypatch, lambda: "saved answer")
    threads._process_message("state", "probe")
    assert outcomes[-1][1:4] == ("ready", None, "saved answer")
    assert [event[0] for event in events] == ["start", "exit"] * 5
    assert len({event[1] for event in events}) == 5
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
    assert not sync.read_state(str(repos[3])).get("sandbox_in_flight")
    assert git(repos[0], "for-each-ref", "--format=%(refname)") == "refs/heads/main"


def test_failed_model_teardown_preserves_answer_and_quarantines(repos, monkeypatch):
    threads, events, outcomes = web_turn(repos, monkeypatch, lambda: "saved answer")
    original = threads.SandboxManager.cleanup_verified

    def fail_model_cleanup(worktree, generation):
        if len(events) == 5:
            raise RuntimeError("test daemon failure")
        return original(worktree, generation)

    monkeypatch.setattr(threads.SandboxManager, "cleanup_verified", fail_model_cleanup)
    threads._process_message("state", "probe")
    assert outcomes[-1][1:4] == ("error", None, "saved answer")
    assert sync.read_state(str(repos[3]))["quarantine"]
    assert len(events) == 5  # No Git-only generation or host publication after failed exit.
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
        pending = next(iter(sync.read_state(str(repos[3]))["preflights"].values()))
        assert pending["base"] == git(repos[1], "rev-parse", "HEAD")
        model = threads._get_sandbox_backend("state", before_start=kwargs["sandbox_starting"])
        (repos[1] / "pi").write_text("Pi committed change\n")
        kwargs["commit"](result)
        kwargs["sandbox_cleanup"](model.container)
        return result

    monkeypatch.setattr(threads, "_PI_RUNTIME", SimpleNamespace(run=runtime))
    run = threads._create_run("state", "probe")
    threads._execute_pi_run(run, user_priority=False)
    assert threads._runs().get("state", run.id).status == "success"
    assert git(repos[0], "show", "thread/test:pi") == "Pi committed change"
    assert [event[0] for event in events] == ["start", "exit"] * 5
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


def test_changed_turn_commits_without_clone_identity(repos):
    remote, thread, _, _ = repos
    git(thread, "config", "--unset", "user.name")
    git(thread, "config", "--unset", "user.email")
    owner, backend = turn(repos)
    (thread / "new").write_text("change\n")
    owner.commit(backend, "changed")
    owner.publish()
    assert git(remote, "log", "-1", "--format=%an <%ae>|%cn <%ce>", "thread/test") == (
        "Assist <assist@localhost>|Assist <assist@localhost>")


def test_legacy_hardlinked_clone_cannot_be_authorized(repos, tmp_path):
    legacy = tmp_path / "legacy"
    subprocess.run(["git", "clone", str(repos[0]), str(legacy)], check=True, capture_output=True)
    git(legacy, "checkout", "-b", "legacy")
    with pytest.raises(sync.GitSyncError, match="hardlinks"):
        sync.authorize_branch(str(repos[3]), str(legacy))
    assert sync.read_state(str(repos[3]))["branch"] == "thread/test"


def test_long_index_cannot_hide_flag_after_backend_truncation(repos):
    thread = repos[1]
    for index in range(2000):
        (thread / (f"file-{index:04d}-" + "x" * 55)).touch()
    (thread / "zz-hidden").write_text("original\n")
    git(thread, "add", ".")
    git(thread, "commit", "-m", "large index")
    git(thread, "update-index", "--assume-unchanged", "zz-hidden")
    (thread / "zz-hidden").write_text("preserve dirty\n")

    class TruncatingBackend(LocalBackend):
        def execute(self, command):
            result = super().execute(command)
            if len(result.output) > 100_000:
                result.output = result.output[:100_000] + " [truncated]"
            return result

    assert len(git(thread, "ls-files", "-v", "-z")) > 100_000
    owner = sync.GitSync(str(repos[3]), str(thread))
    with pytest.raises(sync.GitSyncError, match="hidden"):
        owner.prepare(TruncatingBackend(thread), "large")
    assert (thread / "zz-hidden").read_text() == "preserve dirty\n"


def test_post_fast_forward_hook_cannot_hide_dirty_worktree(repos):
    remote, thread, phone, _ = repos
    owner, _ = turn(repos)
    owner.publish()
    git(phone, "fetch", "origin", "thread/test")
    git(phone, "checkout", "-b", "thread/test", "FETCH_HEAD")
    (phone / "tracked").write_text("phone version\n")
    git(phone, "add", ".")
    git(phone, "commit", "-m", "phone")
    git(phone, "push", "origin", "thread/test")
    hook = thread / ".git" / "hooks" / "post-merge"
    hook.write_text("#!/bin/sh\nprintf 'hook dirt\\n' > tracked\ngit update-index --assume-unchanged tracked\n")
    hook.chmod(0o700)
    with pytest.raises(sync.GitSyncError, match="hidden"):
        turn(repos)
    assert git(thread, "rev-parse", "HEAD") == git(remote, "rev-parse", "thread/test")
    assert (thread / "tracked").read_text() == "hook dirt\n"


def test_malformed_published_pair_is_unavailable_not_type_error(repos):
    state = sync.read_state(str(repos[3]))
    state.update(published_branch=[], published_revision="a" * 40)
    sync._write_state(str(repos[3]), state)
    with pytest.raises(sync.GitSyncError, match="binding"):
        sync.read_state(str(repos[3]))
    assert sync.workspace(str(repos[3]), str(repos[1]))["repo_label"] == "Repository unavailable"


def test_metadata_write_failure_removes_private_temporary(repos, monkeypatch):
    original = sync.os.fsync
    state = sync.read_state(str(repos[3]))

    def fail(_fd):
        raise OSError("disk failure")

    monkeypatch.setattr(sync.os, "fsync", fail)
    with pytest.raises(OSError, match="disk"):
        sync._write_state(str(repos[3]), state)
    monkeypatch.setattr(sync.os, "fsync", original)
    assert not list(repos[3].glob(".git-sync-*"))
    assert sync.read_state(str(repos[3])) == state


@pytest.mark.parametrize("fault", ["missing-git", "lost-sandbox", "dirty"])
def test_pi_preflight_fault_terminalizes_and_exposes_reason(repos, monkeypatch, fault):
    from assist.sandbox import SandboxContainerLostError
    threads, _, _ = web_turn(repos, monkeypatch, lambda: pytest.fail("Deep model ran"))
    monkeypatch.setattr(threads, "PI_PREVIEW", SimpleNamespace(admits=lambda _engine: True))
    monkeypatch.setattr(threads, "_PI_RUNTIME", SimpleNamespace(
        run=lambda **_kwargs: pytest.fail("Pi model ran after failed preflight")))
    if fault == "missing-git":
        (repos[1] / ".git").rename(repos[1] / "preserved-git")
    elif fault == "lost-sandbox":
        monkeypatch.setattr(sync, "require_clean", lambda _backend: (
            _ for _ in ()).throw(SandboxContainerLostError("lost sandbox")))
    else:
        (repos[1] / "dirty").write_text("preserve\n")
    run = threads._create_run("state", "probe")
    threads._execute_pi_run(run, user_priority=False)
    assert threads._runs().get("state", run.id).status == "error"
    assert sync.workspace(str(repos[3]), str(repos[1]))["sync_error"]
    # An incomplete preflight cannot let suspended work bypass clean proof.
    assert sync.read_state(str(repos[3])).get("sandbox_in_flight")
    with pytest.raises(sync.GitSyncError, match="verification"):
        sync.GitSync(str(repos[3]), str(repos[1]))


def test_hidden_child_commits_before_parent_wake_preflight(repos, monkeypatch, tmp_path):
    def child_model():
        (repos[1] / "child-file").write_text("child output\n")
        return "child result"

    threads, events, _ = web_turn(repos, monkeypatch, child_model)
    child_dir = tmp_path / "child"
    child_dir.mkdir()
    monkeypatch.setattr(threads.MANAGER, "thread_dir",
                        lambda tid: str(child_dir if tid == "sub-child" else repos[3]))
    monkeypatch.setattr(threads, "_child_waits_for_egress", lambda _run: False)
    wakes = []

    def wake(_run):
        owner = sync.GitSync(str(repos[3]), str(repos[1]))
        owner.prepare(LocalBackend(repos[1]), "parent-wake")
        wakes.append((repos[1] / "child-file").read_text())

    monkeypatch.setattr(threads, "_complete_child_handoff", wake)
    run = threads._create_run("sub-child", "probe", mode="child", parent_thread_id="state",
                              parent_run_id="parent", dispatch_key="probe-child",
                              assistant_id="delegate-agent")
    threads._execute_child_run(run)
    assert threads._runs().get("sub-child", run.id).status == "success"
    assert wakes == ["child output\n"]
    assert git(repos[0], "show", "thread/test:child-file") == "child output"
    assert [event[0] for event in events] == ["start", "exit"] * 5


@pytest.mark.parametrize("failure", ["before-create", "during-create"])
def test_flight_fence_starts_at_docker_create_not_preconditions(repos, monkeypatch, failure):
    from unittest.mock import MagicMock
    from docker.errors import DockerException
    from assist.sandbox_manager import SandboxManager
    owner = sync.GitSync(str(repos[3]), str(repos[1]))
    client = MagicMock()
    if failure == "before-create":
        monkeypatch.setattr(SandboxManager, "_get_docker_client",
                            lambda: (_ for _ in ()).throw(DockerException("offline")))
    else:
        monkeypatch.setattr(SandboxManager, "_get_docker_client", lambda: client)
        monkeypatch.setattr(SandboxManager, "_ensure_egress_proxy_running", lambda _client: None)
        client.containers.run.side_effect = DockerException("ambiguous create")
    assert SandboxManager.get_pi_sandbox_backend(
        str(repos[1]), before_start=owner.sandbox_started) is None
    assert bool(sync.read_state(str(repos[3])).get("sandbox_in_flight")) == (failure == "during-create")
    if failure == "before-create":
        client.containers.run.assert_not_called()
    else:
        client.containers.run.assert_called_once()


def test_shared_repository_labels_are_basename_only():
    from manage.web import state
    assert state._domain_label("git@secret-host:repo.git") == "repo"


def test_child_publication_preserves_suspended_parent_preflight(repos):
    remote, thread, _, binding = repos
    parent = sync.GitSync(str(binding), str(thread))
    parent.prepare(LocalBackend(thread), "parent")
    parent_floor = git(thread, "rev-parse", "HEAD")
    child = sync.GitSync(str(binding), str(thread))
    child.prepare(LocalBackend(thread), "child")
    (thread / "child").write_text("child\n")
    child.commit(LocalBackend(thread), "child")
    child.publish()
    child_tip = git(remote, "rev-parse", "thread/test")
    remaining = sync.read_state(str(binding))["preflights"]
    assert list(remaining) == ["parent"]
    assert remaining["parent"] == {"branch": "thread/test", "expected": child_tip, "base": parent_floor}
    resumed = sync.GitSync(str(binding), str(thread))
    resumed.select_work("parent")  # A resume must not fetch or fast-forward.
    (thread / "parent").write_text("parent after child\n")
    resumed.commit(LocalBackend(thread), "parent resume")
    resumed.publish()
    assert git(remote, "merge-base", child_tip, "thread/test") == child_tip
    assert not sync.read_state(str(binding))["preflights"]


def test_failed_child_preflight_never_erases_parent_work(repos):
    _, thread, _, binding = repos
    parent = sync.GitSync(str(binding), str(thread))
    parent.prepare(LocalBackend(thread), "parent")
    (thread / "parent").write_text("preserve parent edits\n")
    with pytest.raises(sync.GitSyncError, match="uncommitted"):
        sync.GitSync(str(binding), str(thread)).prepare(LocalBackend(thread), "child")
    assert list(sync.read_state(str(binding))["preflights"]) == ["parent"]
    parent.commit(LocalBackend(thread), "parent")
    parent.publish()


def test_post_commit_hook_dirty_work_never_claims_publication(repos):
    remote, thread, _, _ = repos
    owner, backend = turn(repos)
    hook = thread / ".git" / "hooks" / "post-commit"
    hook.write_text("#!/bin/sh\nprintf 'hook dirt\\n' > tracked\npython -c 'print(\"x\" * 200000)'\n")
    hook.chmod(0o700)
    (thread / "agent").write_text("agent\n")
    with pytest.raises(sync.GitSyncError, match="uncommitted"):
        owner.commit(backend, "agent")
    assert (thread / "tracked").read_text() == "hook dirt\n"
    assert git(remote, "for-each-ref", "--format=%(refname)") == "refs/heads/main"


@pytest.mark.parametrize("failure", ["model-teardown", "commit-start"])
def test_child_git_failure_preserves_result_without_success_wake(repos, monkeypatch, tmp_path, failure):
    def model():
        (repos[1] / "child").write_text("preserve child\n")
        return "saved child result"

    threads, events, _ = web_turn(repos, monkeypatch, model)
    child_dir = tmp_path / "sub-child"
    child_dir.mkdir()
    monkeypatch.setattr(threads.MANAGER, "thread_dir",
                        lambda tid: str(child_dir if tid == "sub-child" else repos[3]))
    monkeypatch.setattr(threads, "_child_waits_for_egress", lambda _run: False)
    outcomes = []
    monkeypatch.setattr(threads, "_complete_child_handoff", lambda run: outcomes.append(run))
    if failure == "model-teardown":
        original = threads.SandboxManager.cleanup_verified

        def cleanup(worktree, generation):
            if len(events) == 5:
                raise RuntimeError("daemon failure")
            original(worktree, generation)

        monkeypatch.setattr(threads.SandboxManager, "cleanup_verified", cleanup)
    else:
        original = threads.SandboxManager.get_pi_sandbox_backend
        calls = []

        def backend(*args, **kwargs):
            calls.append(None)
            return original(*args, **kwargs) if len(calls) == 1 else None

        monkeypatch.setattr(threads.SandboxManager, "get_pi_sandbox_backend", backend)
    run = threads._create_run("sub-child", "probe", mode="child", parent_thread_id="state",
                              parent_run_id="parent", dispatch_key="failure-child",
                              assistant_id="delegate-agent")
    threads._execute_child_run(run)
    assert outcomes[-1].status == "error"
    assert outcomes[-1].result == "saved child result"
    assert (repos[1] / "child").read_text() == "preserve child\n"
    assert not sync.read_commit_receipt(str(child_dir), run.work_id)
    threads.SandboxManager._containers.pop(str(repos[1]), None)


def test_child_crash_after_commit_recovers_saved_result_without_model_replay(repos, monkeypatch, tmp_path):
    calls = []

    def model():
        calls.append(None)
        (repos[1] / "child").write_text("committed child\n")
        return "saved child result"

    threads, _, _ = web_turn(repos, monkeypatch, model)
    child_dir = tmp_path / "sub-child"
    child_dir.mkdir()
    monkeypatch.setattr(threads.MANAGER, "thread_dir",
                        lambda tid: str(child_dir if tid == "sub-child" else repos[3]))
    monkeypatch.setattr(threads, "_child_waits_for_egress", lambda _run: False)
    wakes = []

    def wake(run):
        assert run.status == "success"
        owner = sync.GitSync(str(repos[3]), str(repos[1]))
        owner.prepare(LocalBackend(repos[1]), "parent-wake")
        owner.publish()
        wakes.append(run.result)

    monkeypatch.setattr(threads, "_complete_child_handoff", wake)
    run = threads._create_run("sub-child", "probe", mode="child", parent_thread_id="state",
                              parent_run_id="parent", dispatch_key="crash-child",
                              assistant_id="delegate-agent")
    with monkeypatch.context() as patch:
        original = sync._Store.fetch

        def crash_after_receipt(store, *args):
            if (child_dir / "git-commit-receipt.json").exists():
                raise SystemExit("crash")
            return original(store, *args)

        patch.setattr(sync._Store, "fetch", crash_after_receipt)
        with pytest.raises(SystemExit, match="crash"):
            threads._execute_child_run(run)
    saved = threads._runs().get("sub-child", run.id)
    assert saved.status == "running" and saved.result == "saved child result"
    assert sync.read_commit_receipt(str(child_dir), run.work_id)
    threads._recover_child_run(saved)
    assert calls == [None]
    assert wakes == ["saved child result"]
    assert git(repos[0], "show", "thread/test:child") == "committed child"


def test_pi_lock_io_error_terminalizes_exact_pending_ticket(repos, monkeypatch):
    threads, _, _ = web_turn(repos, monkeypatch, lambda: pytest.fail("Model ran"))
    monkeypatch.setattr(threads, "PI_PREVIEW", SimpleNamespace(admits=lambda _engine: True))
    original = sync.os.open

    def fail_lock(path, *args, **kwargs):
        if path == "git-sync.lock":
            raise PermissionError("lock denied")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(sync.os, "open", fail_lock)
    run = threads._create_run("state", "probe")
    threads._execute_pi_run(run, user_priority=False)
    saved = threads._runs().get("state", run.id)
    assert saved.status == "error" and "ownership" in saved.error


def test_unresolved_child_intent_blocks_parent_resume_and_push(repos, monkeypatch):
    remote, thread, _, binding = repos
    backend = LocalBackend(thread)
    parent = sync.GitSync(str(binding), str(thread))
    parent.prepare(backend, "parent")
    child = sync.GitSync(str(binding), str(thread))
    child.prepare(backend, "child")
    (thread / "child").write_text("child\n")
    child.commit(backend, "child")
    original = sync._Store.git

    def lose_response(store, *args, **kwargs):
        if args[0] == "push":
            raise sync.GitSyncError("unknown push outcome")
        return original(store, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(sync._Store, "git", lose_response)
        with pytest.raises(sync.GitSyncError, match="unknown"):
            child.publish()
    intent = sync.read_state(str(binding))["intent"]
    resumed = sync.GitSync(str(binding), str(thread))
    resumed.select_work("parent")
    with pytest.raises(sync.GitSyncError, match="unknown"):
        resumed.admit()
    with pytest.raises(sync.GitSyncError, match="unknown"):
        resumed.publish()
    assert sync.read_state(str(binding))["intent"] == intent
    assert git(remote, "for-each-ref", "--format=%(refname)") == "refs/heads/main"


def test_unpublished_child_floor_survives_rejection_and_parent_reset(repos, monkeypatch):
    remote, thread, _, binding = repos
    backend = LocalBackend(thread)
    base = git(thread, "rev-parse", "HEAD")
    parent = sync.GitSync(str(binding), str(thread))
    parent.prepare(backend, "parent")
    child = sync.GitSync(str(binding), str(thread))
    child.prepare(backend, "child")
    (thread / "child").write_text("child\n")
    child.commit(backend, "child")
    child_revision = git(thread, "rev-parse", "HEAD")
    original = sync._Store.git

    def reject(store, *args, **kwargs):
        if args[0] == "push":
            return "!\tHEAD:refs/heads/thread/test\t[remote rejected]"
        return original(store, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(sync._Store, "git", reject)
        with pytest.raises(sync.GitSyncError, match="rejected"):
            child.publish()
    child.forget_work()
    assert sync.read_state(str(binding))["local_revision"] == child_revision
    git(thread, "reset", "--hard", base)  # Disposable adversarial resumed-model action.
    (thread / "parent").write_text("wrong ancestry\n")
    git(thread, "add", ".")
    git(thread, "commit", "-m", "parent reset")
    resumed = sync.GitSync(str(binding), str(thread))
    resumed.select_work("parent")
    with pytest.raises(sync.GitSyncError, match="history"):
        resumed.admit()
    with pytest.raises(sync.GitSyncError, match="history"):
        resumed.publish()
    assert git(remote, "for-each-ref", "--format=%(refname)") == "refs/heads/main"
    assert sync.read_state(str(binding))["local_revision"] == child_revision


def test_stale_child_receipt_cannot_certify_dropped_commit(repos, tmp_path):
    _, thread, _, binding = repos
    base = git(thread, "rev-parse", "HEAD")
    owner, backend = turn(repos)
    (thread / "child").write_text("child\n")
    owner.commit(backend, "child")
    child_dir = tmp_path / "child-state"
    child_dir.mkdir()
    owner.publish(on_committed=lambda: owner.receipt(str(child_dir)))
    assert owner.verified_receipt(str(child_dir))
    git(thread, "reset", "--hard", base)
    with pytest.raises(sync.GitSyncError, match="Child commit"):
        owner.verified_receipt(str(child_dir))
    assert sync.read_commit_receipt(str(child_dir), "work-1")["revision"] != base


def test_child_crash_after_teardown_before_floor_keeps_parent_fenced(repos, monkeypatch, tmp_path):
    def model():
        (repos[1] / "child").write_text("keep committed child\n")
        return "saved child result"

    threads, _, _ = web_turn(repos, monkeypatch, model)
    child_dir = tmp_path / "sub-child"
    child_dir.mkdir()
    monkeypatch.setattr(threads.MANAGER, "thread_dir",
                        lambda tid: str(child_dir if tid == "sub-child" else repos[3]))
    monkeypatch.setattr(threads, "_child_waits_for_egress", lambda _run: False)
    outcomes = []
    monkeypatch.setattr(threads, "_complete_child_handoff", lambda run: outcomes.append(run))
    run = threads._create_run("sub-child", "probe", mode="child", parent_thread_id="state",
                              parent_run_id="parent", dispatch_key="gap-child",
                              assistant_id="delegate-agent")
    with monkeypatch.context() as patch:
        patch.setattr(sync.GitSync, "publish", lambda _owner, **_kw: (
            _ for _ in ()).throw(SystemExit("after teardown")))
        with pytest.raises(SystemExit, match="after teardown"):
            threads._execute_child_run(run)
    saved = threads._runs().get("sub-child", run.id)
    assert saved.status == "running" and saved.result == "saved child result"
    assert sync.read_state(str(repos[3]))["sandbox_in_flight"]
    assert not sync.read_commit_receipt(str(child_dir), run.work_id)
    with pytest.raises(sync.GitSyncError, match="verification"):
        sync.GitSync(str(repos[3]), str(repos[1]))
    threads._recover_child_run(saved)
    assert outcomes[-1].status == "error" and outcomes[-1].result == "saved child result"
    assert (repos[1] / "child").read_text() == "keep committed child\n"


def test_missing_bound_local_floor_fails_closed(repos):
    state = sync.read_state(str(repos[3]))
    state.pop("local_revision")
    sync._write_state(str(repos[3]), state)
    with pytest.raises(sync.GitSyncError, match="binding"):
        sync.GitSync(str(repos[3]), str(repos[1]))
    assert sync.workspace(str(repos[3]), str(repos[1]))["repo_label"] == "Repository unavailable"


def test_standard_git_repack_bitmap_does_not_block_snapshot(repos):
    git(repos[1], "repack", "-adb", "--write-bitmap-index")
    assert list((repos[1] / ".git" / "objects" / "pack").glob("*.bitmap"))
    owner, _ = turn(repos)
    owner.publish()
    assert git(repos[0], "rev-parse", "thread/test") == git(repos[1], "rev-parse", "HEAD")


@pytest.mark.parametrize("alias", ["keep", "untracked", "scratch", "private"])
def test_legacy_relocated_source_hardlink_cannot_hide_from_authorization(repos, tmp_path, alias):
    remote, _, _, binding = repos
    git(remote, "repack", "-ad")
    legacy = tmp_path / "legacy-clone"
    subprocess.run(["git", "clone", str(remote), str(legacy)], check=True, capture_output=True)
    git(legacy, "checkout", "-b", "legacy")
    pack = next((legacy / ".git" / "objects" / "pack").glob("*.pack"))
    original = pack.read_bytes()
    if alias == "keep":
        target = pack.with_suffix(".keep")
    elif alias == "untracked":
        target = legacy / "untracked-source-inode"
    else:
        directory = tmp_path / "tmp" if alias == "scratch" else binding / "agent"
        directory.mkdir(exist_ok=True)
        target = directory / "source-inode"
    pack.rename(target)  # Old sandbox can move its source-linked inode within a mount.
    pack.write_bytes(original)
    for path in (legacy / ".git" / "objects" / "pack").iterdir():
        if path == target:
            continue
        content = path.read_bytes()
        path.unlink()
        path.write_bytes(content)
    with sync.tempfile.TemporaryDirectory() as directory:
        sync._Store(directory).snapshot(str(legacy))  # Object-only validation misses the alias.
    with pytest.raises(sync.GitSyncError, match="hardlinks"):
        sync.authorize_branch(str(binding), str(legacy))
    assert sync.read_state(str(binding))["branch"] == "thread/test"
    assert target.stat().st_nlink > 1 and target.read_bytes() == original


def test_git_verification_profile_mounts_only_readonly_workspace(repos, monkeypatch):
    from unittest.mock import MagicMock
    from assist.sandbox_manager import SandboxManager
    client = MagicMock()
    container = client.containers.run.return_value
    container.id = "verify-container"
    monkeypatch.setattr(SandboxManager, "_containers", {})
    monkeypatch.setattr(SandboxManager, "_get_docker_client", lambda: client)
    monkeypatch.setattr(SandboxManager, "_ensure_egress_proxy_running", lambda _client: None)
    # This mount-profile fixture has no real proxy or inspectable attribution map.
    monkeypatch.setattr(SandboxManager, "_record_egress_client", lambda *_args, **_kw: None)
    starts = []
    SandboxManager.get_git_verification_backend(str(repos[1]), before_start=lambda: starts.append(True))
    call = client.containers.run.call_args.kwargs
    assert starts == [True]
    assert call["volumes"] == {str(repos[1]): {"bind": "/workspace", "mode": "ro"}}
    assert not any(key.startswith("ASSIST_") for key in call["environment"])
    assert call["user"] != "0:0"


def test_recovered_initializer_preserves_authorized_workspace_and_ancestry(repos, monkeypatch):
    threads, _, _ = web_turn(repos, monkeypatch, lambda: "unused")
    owner, _ = turn(repos)
    before = sync.read_state(str(repos[3]))
    (repos[1] / "tracked").write_text("preserve dirty work\n")
    run = threads._create_run("state", "first visible message")
    executed = []
    monkeypatch.setattr(threads, "_execute_run", lambda *args: executed.append(args))
    monkeypatch.setattr(threads, "_settle_cancelled_initializer", lambda *_: False)
    monkeypatch.setattr(threads._INITIALIZATION_SCHEDULER, "complete", lambda *_: None)
    monkeypatch.setattr(threads, "_reset_unexecuted_workspace", lambda *_: (
        _ for _ in ()).throw(AssertionError("Must not discard authorized workspace")))
    monkeypatch.setattr(threads, "DomainManager", lambda *_args, **_kw: (
        _ for _ in ()).throw(AssertionError("Must not rebuild authorized clone")))
    threads._initialize_thread("state", run.id, str(repos[0]))
    assert executed == [(run.id, "state")]
    assert sync.read_state(str(repos[3])) == before
    assert (repos[1] / "tracked").read_text() == "preserve dirty work\n"


def test_preflight_crash_after_writer_exit_keeps_resume_fenced(repos, monkeypatch):
    threads, _, _ = web_turn(repos, monkeypatch, lambda: "must not run")
    owner = sync.GitSync(str(repos[3]), str(repos[1]))
    monkeypatch.setattr(threads, "_git_verify", lambda *_: (
        _ for _ in ()).throw(SystemExit("before readonly verification")))
    with pytest.raises(SystemExit, match="readonly verification"):
        threads._git_prepare(owner, "crashed-preflight", None)
    state = sync.read_state(str(repos[3]))
    assert "crashed-preflight" in state["preflights"]
    assert state["sandbox_in_flight"]
    with pytest.raises(sync.GitSyncError, match="verification"):
        sync.GitSync(str(repos[3]), str(repos[1]))


@pytest.mark.parametrize("phase", ["preflight", "commit"])
def test_late_writer_after_clean_probe_cannot_reach_child_success(repos, monkeypatch, tmp_path, phase):
    calls = []

    def model():
        calls.append(True)
        (repos[1] / "child").write_text("child\n")
        return "saved child result"

    threads, events, _ = web_turn(repos, monkeypatch, model)
    child_dir = tmp_path / "sub-child"
    child_dir.mkdir()
    monkeypatch.setattr(threads.MANAGER, "thread_dir",
                        lambda tid: str(child_dir if tid == "sub-child" else repos[3]))
    monkeypatch.setattr(threads, "_child_waits_for_egress", lambda _run: False)
    outcomes = []
    monkeypatch.setattr(threads, "_complete_child_handoff", lambda run: outcomes.append(run))
    original = threads.SandboxManager.cleanup_verified

    def late_write(worktree, generation):
        # Deterministic cut: the mutable generation's clean check has returned,
        # while its background writer is not yet reaped. No sleep race fixture.
        if len(events) == (1 if phase == "preflight" else 7):
            (repos[1] / "tracked").write_text("late writer dirt\n")
        original(worktree, generation)

    monkeypatch.setattr(threads.SandboxManager, "cleanup_verified", late_write)
    run = threads._create_run("sub-child", "probe", mode="child", parent_thread_id="state",
                              parent_run_id="parent", dispatch_key="late-writer-child",
                              assistant_id="delegate-agent")
    threads._execute_child_run(run)
    assert outcomes[-1].status == "error"
    assert outcomes[-1].result == ("saved child result" if phase == "commit" else None)
    assert calls == ([True] if phase == "commit" else [])
    assert (repos[1] / "tracked").read_text() == "late writer dirt\n"
    assert not sync.read_commit_receipt(str(child_dir), run.work_id)


@pytest.mark.parametrize("engine", ["deep", "pi"])
def test_visible_commit_failure_preserves_answer_and_reports_error(repos, monkeypatch, engine):
    def model():
        (repos[1] / "answer-work").write_text("preserve work\n")
        return "saved answer"

    threads, _, outcomes = web_turn(repos, monkeypatch, model)
    monkeypatch.setattr(sync.GitSync, "commit", lambda *_args: (
        _ for _ in ()).throw(sync.GitSyncError("restricted commit failed")))
    run = threads._create_run("state", "probe")
    if engine == "pi":
        from assist.pi_runtime import PiRuntimeResult
        from assist.pi_conversation import PiConversationStore
        monkeypatch.setattr(threads, "PI_PREVIEW", SimpleNamespace(admits=lambda _engine: True))
        monkeypatch.setattr(threads, "_pi_system_prompt", lambda: "be useful")
        monkeypatch.setattr(threads, "_PI_CONVERSATIONS", PiConversationStore())

        def runtime(**kwargs):
            backend = threads._get_sandbox_backend("state", before_start=kwargs["sandbox_starting"])
            result = PiRuntimeResult(model(), 1)
            kwargs["commit"](result)
            kwargs["sandbox_cleanup"](backend.container)
            return result

        monkeypatch.setattr(threads, "_PI_RUNTIME", SimpleNamespace(run=runtime))
        threads._execute_pi_run(run, user_priority=False)
        assert threads._PI_CONVERSATIONS.completed_reply(str(repos[3]), run.id).text == "saved answer"
    else:
        threads._process_message("state", "probe", _run=run)
        assert outcomes[-1][1:4] == ("error", None, "saved answer")
    saved = threads._runs().get("state", run.id)
    assert saved.status == "error" and saved.result == "saved answer"
    assert (repos[1] / "answer-work").read_text() == "preserve work\n"
    assert sync.read_state(str(repos[3]))["sandbox_in_flight"]
    assert git(repos[0], "for-each-ref", "--format=%(refname)") == "refs/heads/main"


@pytest.mark.parametrize("pending", [False, True])
def test_authorization_rejects_rebranch_without_erasing_publication_fences(repos, pending):
    owner, backend = turn(repos)
    owner.commit(backend, "noop")
    owner.publish()
    before = sync.read_state(str(repos[3]))
    if pending:
        before["intent"] = {"branch": "thread/test", "expected": before["published_revision"],
                            "desired": before["local_revision"]}
        before["preflights"]["retained"] = {"branch": "thread/test", "expected": before["published_revision"],
                                             "base": before["local_revision"]}
        sync._write_state(str(repos[3]), before)
    git(repos[1], "checkout", "-b", "different-thread")
    with pytest.raises(sync.GitSyncError, match="branch"):
        sync.authorize_branch(str(repos[3]), str(repos[1]))
    assert sync.read_state(str(repos[3])) == before


@pytest.mark.parametrize("error", ["x" * 257, {"private": "invalid type"}, 7])
def test_malformed_sync_error_metadata_is_unavailable(repos, error):
    state = sync.read_state(str(repos[3]))
    state["error"] = error
    sync._write_state(str(repos[3]), state)
    with pytest.raises(sync.GitSyncError, match="binding"):
        sync.read_state(str(repos[3]))
    value = sync.workspace(str(repos[3]), str(repos[1]))
    assert value["repo_label"] == "Repository unavailable"
    assert isinstance(value["sync_error"], str) and len(value["sync_error"]) <= 256


@pytest.mark.parametrize("failure", ["rejected", "unknown"])
def test_remote_only_child_failure_keeps_local_success_and_retry_fences(repos, monkeypatch, tmp_path, failure):
    threads, _, _ = web_turn(repos, monkeypatch, lambda: "unused")
    backend = LocalBackend(repos[1])
    parent = sync.GitSync(str(repos[3]), str(repos[1]))
    parent.prepare(backend, "parent")
    child = sync.GitSync(str(repos[3]), str(repos[1]))
    child.prepare(backend, "child")
    (repos[1] / "child").write_text("saved child work\n")
    child_dir = tmp_path / "child-state"
    child_dir.mkdir()
    original = sync._Store.git

    def fail_push(store, *args, **kwargs):
        if args[0] == "push":
            if failure == "unknown":
                raise sync.GitSyncError("push outcome unknown")
            return "!\tHEAD:refs/heads/thread/test\t[remote rejected]"
        return original(store, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(sync._Store, "git", fail_push)
        assert threads._git_finish(child, "saved child result", None,
                                   on_committed=lambda: child.receipt(str(child_dir)))
    assert child.verified_receipt(str(child_dir))
    child.forget_work()
    state = sync.read_state(str(repos[3]))
    assert list(state["preflights"]) == ["parent"]
    assert state["local_revision"] == git(repos[1], "rev-parse", "HEAD")
    assert state["error"] and state["published_revision"] is None
    parent = sync.GitSync(str(repos[3]), str(repos[1]))
    parent.select_work("parent")
    if failure == "unknown":
        with pytest.raises(sync.GitSyncError, match="unknown"):
            parent.admit()
        with pytest.raises(sync.GitSyncError, match="unknown"):
            parent.prepare(backend, "next")
        assert sync.read_state(str(repos[3]))["intent"] == state["intent"]
    else:
        parent.admit()
        later = sync.GitSync(str(repos[3]), str(repos[1]))
        later.prepare(backend, "next")
        later.commit(backend, "no further change")
        later.publish()
        assert git(repos[0], "show", "thread/test:child") == "saved child work"


def test_local_commit_failure_cancels_new_continuation_and_rejournals_interjection(repos, monkeypatch):
    created = {}

    def model():
        created["continuation"] = threads._create_run("state", "new promise", origin="continuation")
        injected = threads._create_run("state", "interjected user message")
        created["interjection"] = injected
        threads._consume_interjections("state", {injected.id})
        return "saved answer"

    threads, _, outcomes = web_turn(repos, monkeypatch, model)
    retained = threads._create_run("state", "earlier promise", origin="continuation")
    queued = []
    monkeypatch.setattr(threads._RESUME_SCHEDULER, "submit", lambda *args: queued.append(args))
    monkeypatch.setattr(sync.GitSync, "commit", lambda *_args: (
        _ for _ in ()).throw(sync.GitSyncError("restricted commit failed")))
    run = threads._create_run("state", "probe")
    threads._process_message("state", "probe", _run=run)
    assert outcomes[-1][1:4] == ("error", None, "saved answer")
    assert threads._runs().get("state", created["continuation"].id).status == "cancelled"
    assert threads._runs().get("state", retained.id).status == "pending"
    fresh = [item for item in threads._runs().list("state")
             if item.text == "interjected user message" and item.status == "pending"]
    assert len(fresh) == 1 and fresh[0].id != created["interjection"].id
    assert queued == [(fresh[0].id, "state")]
    assert "state" not in threads._TURN_INTERJECTION
