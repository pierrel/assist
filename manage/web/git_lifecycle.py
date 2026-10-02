"""Git workspace lifecycle used by synchronous web turn orchestration."""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager

from assist.git_sync import (GitDirtyWorktreeError, GitSync, GitSyncError, ownership, read_state,
                             require_clean)
from assist.sandbox_manager import SandboxManager


def recovery_error(thread_dir: str, worktree: str) -> str | None:
    """A saved reply alone cannot authenticate a Git-backed turn's finalization."""
    try:
        git_bound = read_state(thread_dir) is not None or os.path.lexists(
            os.path.join(worktree, ".git"))
    except (GitSyncError, OSError):
        git_bound = True
    if git_bound:
        return ("Git finalization after restart is unverified; saved answer and files are preserved. "
                "Reconcile Git before continuing.")
    return None


def preflight_fence_error(thread_dir: str, worktree: str) -> str | None:
    """Project a retained fence or unverified Git state before a Run's dirty error."""
    reason = "Previous Git teardown or commit finalization needs operator verification"
    try:
        state = read_state(thread_dir)
        if state is None:
            return reason if os.path.lexists(os.path.join(worktree, ".git")) else None
        return reason if state.get("sandbox_in_flight") or state.get("quarantine") else None
    except (GitSyncError, OSError):
        return reason


def _reap(owner: GitSync, generation) -> None:
    """Prove exact generation exit, retaining the Git flight/finalization fence."""
    try:
        SandboxManager.cleanup_verified(owner.worktree, generation)
    except Exception as error:
        owner.quarantine()
        owner.failed(GitSyncError("Git sandbox teardown needs operator verification"))
        raise GitSyncError("Git sandbox teardown needs operator verification") from error


def _cleanup(owner: GitSync, generation) -> None:
    """Prove model generation exit and clear its flight fence."""
    _reap(owner, generation)
    owner.sandbox_stopped()


def _admit(owner: GitSync):
    """Fail closed with a bounded persisted reason before a resumed model runs."""
    try:
        return owner.admit()
    except Exception as error:
        reason = error if isinstance(error, GitSyncError) else GitSyncError(
            "Git admission failed; preserve the workspace and reconcile before retrying")
        owner.failed(reason)
        raise reason from error


def _verify(owner: GitSync, timezone: str | None) -> None:
    """After writer exit, prove clean Git state on a kernel-read-only workspace."""
    backend = SandboxManager.get_git_verification_backend(
        owner.worktree, tz=timezone, before_start=owner.sandbox_started)
    if backend is None:
        raise GitSyncError("Git clean verification requires the restricted sandbox")
    try:
        require_clean(backend)
    finally:
        _reap(owner, backend.container)


def _prepare(owner: GitSync, work_id: str, timezone: str | None,
             terminalize_dirty=None) -> None:
    """Reconcile Git before model admission using the credential-free profile."""
    try:
        backend = SandboxManager.get_pi_sandbox_backend(
            owner.worktree, tz=timezone, before_start=owner.sandbox_started)
        if backend is None:
            raise GitSyncError("Git sync requires the restricted sandbox")
        try:
            owner.prepare(backend, work_id)
        finally:
            _reap(owner, backend.container)
        _verify(owner, timezone)
        _, revision = owner.admit()
        if revision != owner.state["preflights"][work_id]["base"]:
            raise GitSyncError("Git checkout changed during preflight teardown; reconcile before the turn")
        owner.sandbox_stopped()
    except GitDirtyWorktreeError as error:
        try:
            if terminalize_dirty is None:
                raise GitSyncError("Git dirty preflight has no durable Run outcome")
            terminalize_dirty(error)
        except Exception as outcome_error:
            reason = GitSyncError("Git dirty preflight outcome needs operator verification")
            owner.failed(reason)
            raise reason from outcome_error
        owner.failed(error)
        owner.sandbox_stopped()
        raise
    except Exception as error:
        reason = error if isinstance(error, GitSyncError) else GitSyncError(
            "Git preflight failed; preserve the workspace and reconcile before retrying")
        owner.failed(reason)
        raise reason from error


def _finish(owner: GitSync, message: str, timezone: str | None, *, on_committed=None) -> bool:
    """Attempt publication, returning whether local commit and teardown completed."""
    committed = False

    def verified_local_commit():
        nonlocal committed
        if on_committed is not None:
            on_committed()
        owner.sandbox_stopped()
        committed = True

    try:
        if owner.state.get("quarantine") or owner.state.get("sandbox_in_flight"):
            raise GitSyncError("Git teardown or commit finalization needs operator verification")
        backend = SandboxManager.get_pi_sandbox_backend(
            owner.worktree, tz=timezone, before_start=owner.sandbox_started)
        if backend is None:
            raise GitSyncError("Git sync requires the restricted sandbox")
        try:
            owner.commit(backend, message)
        finally:
            _reap(owner, backend.container)
        _verify(owner, timezone)
        owner.publish(on_committed=verified_local_commit)
    except Exception as error:
        owner.failed(error if isinstance(error, GitSyncError) else GitSyncError(
            "Branch publication is pending; work and answer are preserved"))
        logging.warning("Thread branch publication pending", exc_info=True)
    return committed


class GitLifecycle:
    """Turn lifecycle that fences a bound Git workspace."""

    def __init__(self, owner: GitSync | None, worktree: str):
        self._owner = owner
        self._worktree = worktree

    @classmethod
    @contextmanager
    def acquire(cls, thread_dir: str, worktree: str, work_id: str | None = None):
        """Enter ownership, taking the nonblocking workspace flock when Git-bound."""
        with ownership(thread_dir, worktree) as owner:
            if owner is not None and work_id is not None:
                owner.select_work(work_id)
            yield cls(owner, worktree)

    @property
    def bound(self) -> bool:
        return self._owner is not None

    @property
    def model_starting(self):
        return self._owner.sandbox_started if self._owner is not None else None

    @property
    def pi_model_cleanup(self):
        return self._cleanup_git_model if self._owner is not None else None

    def require_sandbox(self, backend) -> None:
        if self._owner is not None and backend is None:
            raise GitSyncError("Git sync requires the restricted sandbox")

    def prepare(self, timezone: str | None, *, terminalize_dirty=None) -> None:
        if self._owner is not None:
            _prepare(self._owner, self._owner.work_id, timezone, terminalize_dirty)

    def resume(self) -> None:
        if self._owner is not None:
            _admit(self._owner)

    def _cleanup_git_model(self, generation) -> None:
        _cleanup(self._owner, generation)

    def cleanup_model(self, generation) -> None:
        if self._owner is not None and generation is not None:
            self._cleanup_git_model(generation)
        else:
            SandboxManager.cleanup(self._worktree, generation)

    def finish_visible(self, message: str, timezone: str | None) -> None:
        if self._owner is not None and not _finish(self._owner, message, timezone):
            raise GitSyncError("Git commit is pending; saved answer and files are preserved")

    def finish_child(self, message: str, child_dir: str) -> None:
        if self._owner is not None and not _finish(
                self._owner, message, None,
                on_committed=lambda: self._owner.receipt(child_dir)):
            raise GitSyncError("Child Git commit is pending; saved result and files are preserved")

    def recover_child(self, message: str, child_dir: str) -> None:
        if self._owner is not None and not self._owner.verified_receipt(child_dir):
            self.finish_child(message, child_dir)

    def child_terminal(self) -> None:
        if self._owner is not None:
            self._owner.forget_work()
