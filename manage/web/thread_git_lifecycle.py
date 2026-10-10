"""Per-turn Git coordination using the checkout and remote refs as authority."""

from __future__ import annotations

from contextlib import contextmanager
import os

from assist.git_sync import (GitDirtyWorktreeError, GitSyncError, _workspace_lock,
                             read_state, require_clean)
from assist.sandbox_manager import SandboxManager
from assist.thread_git import ThreadGit


def _legacy_hold(thread_dir: str, worktree: str) -> str | None:
    """Treat retained old-generation ambiguity as an operator hold, not progress."""
    try:
        binding = read_state(thread_dir)
    except (GitSyncError, OSError):
        return "Git source binding needs operator verification"
    if not os.path.lexists(os.path.join(worktree, ".git")):
        return ("Bound Git workspace is unavailable" if binding is not None else None)
    if binding is None:
        return "Existing Git repository needs an operator-verified source binding"
    if binding.get("sandbox_in_flight") or binding.get("quarantine"):
        return "Previous Git turn needs operator reconciliation"
    return None


def recovery_error(thread_dir: str, worktree: str) -> str | None:
    """A saved answer without a verified push cannot become a successful Run."""
    hold = _legacy_hold(thread_dir, worktree)
    return hold or ("Git publication after restart needs verification"
                    if os.path.lexists(os.path.join(worktree, ".git")) else None)


def preflight_fence_error(thread_dir: str, worktree: str) -> str | None:
    return _legacy_hold(thread_dir, worktree)


class ThreadGitLifecycle:
    """A checkout fence, verified sandbox teardown and ordinary Git operations."""

    def __init__(self, owner: ThreadGit | None, worktree: str,
                 thread_scope: tuple[str, str] | None = None,
                 owner_run_id: str | None = None):
        self._owner = owner
        self._worktree = worktree
        self._thread_scope = thread_scope
        self._owner_run_id = owner_run_id
        self._teardown_failed = False

    @classmethod
    @contextmanager
    def acquire(cls, thread_dir: str, worktree: str, work_id: str | None = None,
                *, thread_scope: tuple[str, str] | None = None,
                owner_run_id: str | None = None):
        """Serialize all writers, including hidden child work on the parent checkout."""
        if not os.path.lexists(os.path.join(worktree, ".git")):
            if read_state(thread_dir) is not None:
                raise GitSyncError("Bound Git workspace is unavailable")
            yield cls(None, worktree)
            return
        with _workspace_lock(thread_dir):
            sources = tuple(source.strip() for source in os.getenv("ASSIST_DOMAINS", "").split(",")
                            if source.strip())
            yield cls(ThreadGit(thread_dir, worktree, sources), worktree,
                      thread_scope, owner_run_id)

    @property
    def bound(self) -> bool:
        return self._owner is not None

    @property
    def model_starting(self):
        # The scoped sandbox authority begins before Docker create and owns its
        # exact generation. Git needs no second persistent flight marker.
        return None

    @property
    def pi_model_cleanup(self):
        return self.cleanup_model if self.bound else None

    def require_sandbox(self, backend) -> None:
        if self.bound and backend is None:
            raise GitSyncError("Git turn requires the restricted sandbox")

    def _cleanup(self, generation) -> None:
        try:
            SandboxManager.cleanup_verified(self._worktree, generation)
        except Exception as error:
            self._teardown_failed = True
            raise GitSyncError("Git sandbox teardown needs operator verification") from error

    def cleanup_model(self, generation) -> None:
        if self.bound and generation is not None:
            self._cleanup(generation)
        else:
            SandboxManager.cleanup(self._worktree, generation)

    def _backend(self, timezone: str | None):
        if self._thread_scope is None:
            raise GitSyncError("Git sandbox thread authority is unavailable")
        backend = SandboxManager.get_pi_sandbox_backend(
            self._worktree, tz=timezone, before_start=self.model_starting,
            thread_scope=self._thread_scope, owner_run_id=self._owner_run_id)
        if backend is None:
            raise GitSyncError("Git turn requires the restricted sandbox")
        return backend

    def _verify_clean(self, timezone: str | None) -> None:
        if self._thread_scope is None:
            raise GitSyncError("Git verifier thread authority is unavailable")
        backend = SandboxManager.get_git_verification_backend(
            self._worktree, tz=timezone, before_start=self.model_starting,
            thread_scope=self._thread_scope, owner_run_id=self._owner_run_id)
        if backend is None:
            raise GitSyncError("Git clean verification requires the restricted sandbox")
        try:
            require_clean(backend)
        finally:
            self._cleanup(backend.container)

    @contextmanager
    def _host_fence(self):
        """Keep managed writable generations absent through credentialed Git."""
        if self._thread_scope is None:
            raise GitSyncError("Git sandbox thread authority is unavailable")
        from assist.browser import authority
        from assist.browser.manager import BrowserManager, BrowserUnavailable
        root, tid = self._thread_scope
        try:
            with authority.generation_fence(root, tid):
                BrowserManager.confirm_owner_stopped(root, tid, None,
                                                     before_replacement=True)
                yield
        except GitSyncError:
            raise
        except (BrowserUnavailable, OSError, RuntimeError, TimeoutError) as error:
            raise GitSyncError("Git workspace generation needs verification") from error

    def prepare(self, timezone: str | None, *, terminalize_dirty=None) -> None:
        if not self.bound:
            return
        if self._teardown_failed:
            raise GitSyncError("Git sandbox teardown needs operator verification")
        # A retained local commit may be republished by plan_prepare. Establish
        # the clean gate before any such remote ref change.
        try:
            self._verify_clean(timezone)
        except GitDirtyWorktreeError as error:
            if terminalize_dirty is not None:
                terminalize_dirty(error)
            raise
        with self._host_fence():
            plan = self._owner.plan_prepare()
        backend = self._backend(timezone)
        try:
            self._owner.apply_prepare(backend, plan)
        except GitDirtyWorktreeError as error:
            if terminalize_dirty is not None:
                terminalize_dirty(error)
            raise
        finally:
            self._cleanup(backend.container)
        self._verify_clean(timezone)

    def resume(self) -> None:
        if self._teardown_failed:
            raise GitSyncError("Git sandbox teardown needs operator verification")

    def handoff(self, timezone: str | None) -> None:
        """Continue a child wake on its parent's base only if that checkout is clean."""
        self.resume()
        if self.bound:
            self._verify_clean(timezone)

    def _finish(self, message: str, timezone: str | None) -> str | None:
        if not self.bound:
            return
        if self._teardown_failed:
            raise GitSyncError("Git sandbox teardown needs operator verification")
        backend = self._backend(timezone)
        try:
            self._owner.commit(backend, message)
        finally:
            self._cleanup(backend.container)
        self._verify_clean(timezone)
        with self._host_fence():
            return self._owner.publish()

    def finish_visible(self, message: str, timezone: str | None) -> str | None:
        return self._finish(message, timezone)

    def finish_child(self, message: str, child_dir: str) -> str | None:
        return self._finish(message, None)

    def recover_child(self, message: str, child_dir: str) -> None:
        if self.bound:
            raise GitSyncError("Child Git publication after restart needs verification")

    def child_terminal(self) -> None:
        pass

    def validate_merge_candidate(self) -> None:
        if not self.bound:
            raise GitSyncError("Thread merge source is unavailable")

    def merge_and_push(self) -> str:
        if not self.bound:
            raise GitSyncError("Thread merge validation is unavailable")
        with self._host_fence():
            plan = self._owner.plan_merge_main()
        backend = self._backend(None)
        try:
            self._owner.import_merge(backend, plan)
        finally:
            self._cleanup(backend.container)
        with self._host_fence():
            return self._owner.publish_merge(plan)
