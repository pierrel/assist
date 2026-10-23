"""Failed workers retain real Git edits without publishing or losing the failure."""
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from assist.git_sync import GitSyncError, GitWorkspaceBusyError, _workspace_lock, require_clean
from assist.pi_runtime import PiRuntimeError
from assist.thread_queue import ThreadAffinityQueue, ThreadPauseRequested
from manage.web import threads
from manage.web.thread_git_lifecycle import ThreadGitLifecycle
from tests.test_thread_git_simple import LocalSandbox, bound_thread, repositories  # noqa: F401
from tests.test_local_git import git


@pytest.fixture
def worker_workspace(bound_thread, monkeypatch, tmp_path):
    owner, source, server, _phone = bound_thread
    monkeypatch.setenv("ASSIST_DOMAINS", str(source))
    monkeypatch.setattr(threads.MANAGER, "root_dir", str(tmp_path))
    monkeypatch.setattr(threads.MANAGER, "thread_default_working_dir", lambda _tid: str(server))
    monkeypatch.setattr(threads, "THREAD_QUEUE", ThreadAffinityQueue())
    monkeypatch.setattr(threads, "_dispatch_pending_after", lambda *_args: None)
    monkeypatch.setattr(threads, "_complete_child_handoff", lambda *_args: None)
    monkeypatch.setattr(threads, "_get_domain_manager", lambda _tid: None)
    monkeypatch.setattr(threads, "get_cached_description", lambda _tid: "description")
    monkeypatch.setattr(threads.BrowserManager, "reconcile_startup", lambda *_args: False)
    monkeypatch.setattr(threads.BrowserManager, "cleanup", lambda *_args: None)
    monkeypatch.setattr(threads, "EGRESS_STORE", None)
    stopped = []
    generation = SimpleNamespace(id="model-generation")
    model = LocalSandbox()
    model.container = generation
    monkeypatch.setattr(threads, "_get_sandbox_backend", lambda *_args, **_kwargs: model)
    monkeypatch.setattr(threads.SandboxManager, "cleanup_verified",
                        lambda _worktree, gen: stopped.append(gen))

    def backend(_self, _timezone):
        # Recovery must use a fresh restricted sandbox only after model teardown.
        assert generation in stopped
        assert threads.THREAD_QUEUE.peek_holder() is not None
        with pytest.raises(GitWorkspaceBusyError):
            with _workspace_lock(str(server.parent / "thread")):
                pass
        return LocalSandbox()

    monkeypatch.setattr(ThreadGitLifecycle, "_backend", backend)
    monkeypatch.setattr(ThreadGitLifecycle, "_verify_clean",
                        lambda _self, _timezone: require_clean(LocalSandbox()))
    # prepare uses a sandbox before the model starts as well.
    original_prepare = ThreadGitLifecycle.prepare

    def prepare(self, timezone, **kwargs):
        stopped.append(generation)
        try:
            return original_prepare(self, timezone, **kwargs)
        finally:
            stopped.clear()

    monkeypatch.setattr(ThreadGitLifecycle, "prepare", prepare)
    monkeypatch.setattr(ThreadGitLifecycle, "_host_fence", lambda _self: nullcontext())
    return owner, source, server, stopped, generation


@pytest.mark.parametrize(("engine", "commit_fails"), [
    ("deep", False), ("pi", False), ("child", False), ("deep", True),
])
def test_failed_worker_commits_tracked_and_new_files(worker_workspace, monkeypatch, engine, commit_fails):
    owner, source, server, stopped, generation = worker_workspace
    original = git("rev-parse", "HEAD", cwd=server)
    failure = PiRuntimeError("provider failed") if engine == "pi" else RuntimeError("provider failed")

    continuations = []

    def fail(*_args, **_kwargs):
        assert generation not in stopped
        (server / "README").write_text("changed before failure\n")
        (server / "batch.org").write_bytes(b"new batch\n")
        if commit_fails:
            continuations.append(threads._create_run(
                "thread", "Follow up", origin="continuation"))
            (server / ".git" / "index.lock").write_text("contended index\n")
        raise failure

    class Chat:
        thread_id = "thread"
        def pending_reply(self):
            return None
        message = staticmethod(fail)

    monkeypatch.setattr(threads.MANAGER, "get", lambda *_args, **_kwargs: Chat())
    monkeypatch.setattr(threads, "_is_pi_thread", lambda _tid: engine == "pi")
    if engine == "pi":
        monkeypatch.setattr(threads, "PI_PREVIEW", SimpleNamespace(admits=lambda _engine: True))
        monkeypatch.setattr(threads, "_pi_system_prompt", lambda: "test prompt")
        def runtime(**kwargs):
            try:
                fail()
            finally:
                kwargs["sandbox_cleanup"](generation)
        monkeypatch.setattr(threads, "_PI_RUNTIME", SimpleNamespace(run=runtime))
    if engine == "child":
        (server.parent / "sub-task").mkdir()
        run = threads._create_run("sub-task", "Do the task", mode="child",
                                  parent_thread_id="thread", parent_run_id="parent-run",
                                  dispatch_key="task")
    else:
        run = threads._create_run("thread", "Do the task")
    threads._execute_run_active(run.id, run.thread_id)

    saved = threads._runs().get(run.thread_id, run.id)
    assert saved.status == "error"
    if commit_fails:
        assert saved.error.startswith("provider failed")
        assert "Working-file commit could not be verified" in saved.error
        assert "reconcile the Git workspace" in saved.error
        assert git("rev-parse", "HEAD", cwd=server) == original
        assert git("rev-parse", "refs/heads/assist/thread", cwd=source) == original
        assert (server / "README").read_text() == "changed before failure\n"
        assert (server / "batch.org").read_bytes() == b"new batch\n"
        assert threads._get_status("thread")["error"] == saved.error
        assert threads._runs().get("thread", continuations[0].id).status == "cancelled"
        return
    assert saved.error == "provider failed"
    assert git("status", "--porcelain", cwd=server) == ""
    recovered = git("rev-parse", "HEAD", cwd=server)
    assert recovered != original
    assert git("rev-parse", "HEAD^", cwd=server) == original
    assert git("branch", "--show-current", cwd=server) == "assist/thread"
    assert git("show", "HEAD:README", cwd=server) == "changed before failure"
    assert git("show", "HEAD:batch.org", cwd=server) == "new batch"
    assert git("rev-parse", "refs/heads/assist/thread", cwd=source) == original
    assert generation in stopped
    # The next normal prepare publishes the retained commit, without model replay.
    owner.apply_prepare(LocalSandbox(), owner.plan_prepare())
    assert git("rev-parse", "refs/heads/assist/thread", cwd=source) == recovered


def test_recovery_commit_failure_retains_bytes_and_both_errors(worker_workspace, monkeypatch):
    owner, source, server, stopped, generation = worker_workspace
    original = git("rev-parse", "HEAD", cwd=server)
    lifecycle = ThreadGitLifecycle(owner, str(server))
    stopped.append(generation)

    def commit_failure(*_args):
        raise GitSyncError("disk full")

    monkeypatch.setattr(owner, "commit", commit_failure)
    primary = RuntimeError("provider failed")
    with threads.THREAD_QUEUE.acquire("thread"), _workspace_lock(str(server.parent / "thread")), \
            pytest.raises(RuntimeError, match="provider failed") as caught:
        with lifecycle.preserve_on_error(None):
            (server / "README").write_text("tracked draft\n")
            (server / "batch.org").write_bytes(b"new draft\n")
            raise primary
    assert caught.value is primary
    assert "commit could not be verified: disk full" in primary.__notes__[0]
    assert (server / "README").read_text() == "tracked draft\n"
    assert (server / "batch.org").read_bytes() == b"new draft\n"
    assert git("rev-parse", "HEAD", cwd=server) == original
    assert git("rev-parse", "refs/heads/assist/thread", cwd=source) == original


def test_fair_pause_does_not_finalize_partial_work(worker_workspace):
    owner, _source, server, _stopped, _generation = worker_workspace
    lifecycle = ThreadGitLifecycle(owner, str(server))
    original = git("rev-parse", "HEAD", cwd=server)
    with pytest.raises(ThreadPauseRequested):
        with lifecycle.preserve_on_error(None):
            (server / "batch.org").write_text("partial work\n")
            raise ThreadPauseRequested()
    assert git("rev-parse", "HEAD", cwd=server) == original
    assert (server / "batch.org").read_text() == "partial work\n"
