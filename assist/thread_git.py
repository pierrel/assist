"""Ordinary thread-branch Git fetch, commit and publication.

The source binding is server-private; the checkout's Git configuration is never
used by credentialed host Git. Turn progress is determined from Git refs rather
than a second durable publication journal.
"""

from __future__ import annotations

import os
from pathlib import Path
import shlex
import tempfile

from assist.git_sync import (
    GitDirtyWorktreeError, GitSyncError, MAX_BYTES, _Store, _WORKTREE_GIT, _sandbox_git, _write_state,
    identity, read_state, require_clean,
)


class ThreadGit:
    """One locked checkout and its fixed, operator-authorized source."""

    def __init__(self, thread_dir: str, worktree: str, configured_sources: tuple[str, ...]):
        self.thread_dir = thread_dir
        self.worktree = worktree
        self.branch = identity(worktree)[0]
        binding = read_state(thread_dir)
        if binding is None:
            raise GitSyncError("Existing Git repository needs an operator-verified source binding")
        if binding["source"] not in configured_sources:
            raise GitSyncError("Configured Git source is unavailable")
        # Old publication progress describes a historical attempt, not a live
        # writer. The remote ref determines what the next turn can safely do.
        if binding.get("sandbox_in_flight") or binding.get("quarantine"):
            raise GitSyncError("Previous Git turn needs operator reconciliation")
        if binding.get("branch") != self.branch:
            raise GitSyncError("Thread branch needs operator verification")
        self.source = binding["source"]
        self.initial_revision = binding["local_revision"]
        self.binding = binding

    def _snapshot(self, store: _Store) -> tuple[str, str]:
        branch, revision = store.snapshot(self.worktree)
        if self.branch is not None and branch != self.branch:
            raise GitSyncError("Thread branch changed during this turn")
        if self.initial_revision and not store.ancestor(self.initial_revision, revision):
            raise GitSyncError("Thread branch left its authorized history")
        self.branch = branch
        return branch, revision

    def prepare(self, sandbox) -> bool:
        """Fast-forward clean work, or retain dirty work at an equal remote tip."""
        try:
            require_clean(sandbox)
        except GitDirtyWorktreeError:
            self._reject_hidden_index_flags(sandbox)
            # The earlier failed turn may have left user-visible files. There is
            # nothing to merge when the server and remote already share HEAD.
            # This branch performs no checkout, index or ref mutation.
            with tempfile.TemporaryDirectory(prefix="assist-git-") as path:
                store = _Store(path)
                branch, local = self._snapshot(store)
                if store.remote_ref(self.source, branch) != local:
                    raise GitDirtyWorktreeError(
                        "Remote branch differs while local files are uncommitted")
            return True
        with tempfile.TemporaryDirectory(prefix="assist-git-") as path:
            store = _Store(path)
            branch, local = self._snapshot(store)
            remote = store.fetch(self.source, branch)
            local_ahead = bool(remote and local != remote and store.ancestor(remote, local))
            if remote and local != remote and not (local_ahead or store.ancestor(local, remote)):
                raise GitSyncError("Local and remote thread branches diverged")

            refs = ["refs/remotes/origin/main"]
            if remote:
                refs.append("refs/remotes/origin/thread")
            if int(store.git("rev-list", "--count", *refs, "--not", local)):
                bundle = os.path.join(path, "incoming.bundle")
                store.git("bundle", "create", bundle, *refs, "--not", local)
                data = Path(bundle).read_bytes()
                if len(data) > MAX_BYTES:
                    raise GitSyncError("Incoming Git bundle is too large")
                transfer = "/tmp/assist-git-" + os.path.basename(path) + ".bundle"
                response = sandbox.upload_files([(transfer, data)])[0]
                if response.error:
                    raise GitSyncError("Could not import remote Git objects")
                _sandbox_git(sandbox, _WORKTREE_GIT + " bundle unbundle " + shlex.quote(transfer)
                             + " && rm -- " + shlex.quote(transfer))
            main = store.git("rev-parse", "refs/remotes/origin/main")
            commands = [_WORKTREE_GIT + " update-ref refs/remotes/origin/main " + main]
            if remote:
                commands.append(_WORKTREE_GIT + " update-ref "
                                + shlex.quote("refs/remotes/origin/" + branch) + " " + remote)
                if local != remote and not local_ahead:
                    commands.append(_WORKTREE_GIT
                                    + " merge --ff-only --no-overwrite-ignore --no-autostash " + remote)
            _sandbox_git(sandbox, " && ".join(commands))
            expected = local if local_ahead else remote or local
            if identity(self.worktree) != (branch, expected):
                raise GitSyncError("Git preflight did not leave the expected branch")
            require_clean(sandbox)
            return False

    @staticmethod
    def _reject_hidden_index_flags(sandbox) -> None:
        """Ordinary dirty files may proceed, but hidden index entries may not."""
        probe = '''import subprocess, sys
args = sys.argv[1:]
with subprocess.Popen(args + ["ls-files", "-v", "-z"], stdout=subprocess.PIPE,
                      stderr=subprocess.DEVNULL) as process:
    total = 0
    pending = b""
    hidden = False
    while chunk := process.stdout.read(65536):
        total += len(chunk)
        if total > 134217728:
            process.kill()
            sys.exit(1)
        entries = (pending + chunk).split(b"\\0")
        pending = entries.pop()
        if len(pending) > 8192:
            process.kill()
            sys.exit(1)
        hidden |= any(not item.startswith(b"H ") for item in entries)
    if pending or process.wait():
        sys.exit(1)
if hidden:
    sys.exit(42)
'''
        result = sandbox.execute("timeout -k 2s 25s sh -c " + shlex.quote(
            "python -I -c " + shlex.quote(probe) + " " + _WORKTREE_GIT)
            + " >/dev/null 2>&1")
        if result.exit_code != 0 or getattr(result, "truncated", False):
            raise GitSyncError("Hidden Git index state needs reconciliation")

    def commit(self, sandbox, message: str) -> None:
        """Stage and commit inside the credential-free restricted sandbox."""
        if identity(self.worktree)[0] != self.branch:
            raise GitSyncError("Thread branch changed during this turn")
        _sandbox_git(sandbox, _WORKTREE_GIT + " add -A && { " + _WORKTREE_GIT
                     + " diff --cached --quiet; code=$?; if [ \"$code\" = 1 ]; then "
                     + "GIT_AUTHOR_NAME=Assist GIT_AUTHOR_EMAIL=assist@localhost "
                     + "GIT_COMMITTER_NAME=Assist GIT_COMMITTER_EMAIL=assist@localhost "
                     + _WORKTREE_GIT + " -c commit.gpgSign=false commit -m "
                     + shlex.quote(message[:4096] or "assistant update")
                     + "; else exit \"$code\"; fi; }")
        require_clean(sandbox)

    def publish(self) -> None:
        """Push an exact fast-forward to the configured source and verify its ref."""
        with tempfile.TemporaryDirectory(prefix="assist-git-") as path:
            store = _Store(path)
            branch, desired = self._snapshot(store)
            remote = store.fetch(self.source, branch)
            if remote and not store.ancestor(remote, desired):
                raise GitSyncError("Remote thread branch advanced; reconcile before publication")
            if remote != desired:
                try:
                    lease = (["--force-with-lease=refs/heads/" + branch + ":"]
                             if remote is None else [])
                    store.git("push", "--porcelain", "--no-verify", *lease, self.source,
                              desired + ":refs/heads/" + branch)
                except GitSyncError:
                    if store.remote_ref(self.source, branch) != desired:
                        raise
            if store.remote_ref(self.source, branch) != desired:
                raise GitSyncError("Thread branch publication is not verified")

    def record_merged_branch(self) -> None:
        """Authorize only the new branch left by a successful gated main merge."""
        with tempfile.TemporaryDirectory(prefix="assist-git-") as path:
            store = _Store(path)
            branch, revision = store.snapshot(self.worktree)
            if branch == self.branch:
                return  # A pending main push did not rebranch the thread.
            if store.remote_ref(self.source, "main") != revision:
                raise GitSyncError("Post-merge thread branch needs verification")
            if identity(self.worktree) != (branch, revision):
                raise GitSyncError("Post-merge branch changed during verification")
        self.binding.update(branch=branch, local_revision=revision,
                            published_branch=None, published_revision=None)
        _write_state(self.thread_dir, self.binding)
        self.branch = branch
        self.initial_revision = revision
