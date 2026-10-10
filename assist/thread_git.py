"""Thread checkout follows published Git refs, without a publication journal."""

from __future__ import annotations

from dataclasses import dataclass
import shlex

from assist.git_sync import (GitSyncError, _WORKTREE_GIT, _sandbox_git, identity,
                             read_state, require_clean)
from assist.local_git import LocalGit


@dataclass(frozen=True)
class PreparePlan:
    main: str
    remote: str
    bundle: bytes | None


class ThreadGit:
    """One locked checkout and its private, operator-authorized source."""

    def __init__(self, thread_dir: str, worktree: str, configured_sources: tuple[str, ...]):
        self.worktree = worktree
        self.branch, self.initial_revision = identity(worktree)
        binding = read_state(thread_dir)
        if binding is None:
            raise GitSyncError("Existing Git repository needs an operator-verified source binding")
        if binding["source"] not in configured_sources:
            raise GitSyncError("Configured Git source is unavailable")
        if binding.get("sandbox_in_flight") or binding.get("quarantine"):
            raise GitSyncError("Previous Git turn needs operator reconciliation")
        if binding.get("branch") != self.branch:
            raise GitSyncError("Thread branch needs operator verification")
        self.source = binding["source"]
        self.floor = binding.get("floor") or binding.get("local_revision")

    def _verified_tip(self, repository: LocalGit) -> str:
        branch, tip = repository.same_tip()
        if branch != self.branch:
            raise GitSyncError("Thread branch changed during this turn")
        if self.floor and not repository.ancestor(self.floor, tip):
            raise GitSyncError("Thread branch left its authorized history")
        return tip

    def plan_prepare(self) -> PreparePlan:
        """Use exact remote refs; retry a retained commit without model replay."""
        with LocalGit(self.source, self.worktree) as repository:
            local = self._verified_tip(repository)
            main, remote = repository.fetch(self.branch)
            if remote is None or remote != local and repository.ancestor(remote, local):
                repository.publish(self.branch, local)
                main, remote = repository.fetch(self.branch)
            if remote is None:
                raise GitSyncError("Published thread branch is unavailable")
            if not repository.ancestor(local, remote):
                result = "assist-result/" + local
                repository.publish(result, local, create_only=True)
                raise GitSyncError("Previous committed result is on " + result
                                   + "; fetch and merge it locally before the next turn")
            return PreparePlan(main, remote, repository.incoming_bundle(local, remote))

    def apply_prepare(self, sandbox, plan: PreparePlan) -> None:
        """Fast-forward only clean checkout files inside the restricted sandbox."""
        require_clean(sandbox)
        local_branch, local = identity(self.worktree)
        if local_branch != self.branch:
            raise GitSyncError("Thread branch changed during preflight")
        if plan.bundle is not None:
            transfer = "/tmp/assist-git-incoming.bundle"
            response = sandbox.upload_files([(transfer, plan.bundle)])[0]
            if response.error:
                raise GitSyncError("Could not import remote Git objects")
            _sandbox_git(sandbox, _WORKTREE_GIT + " bundle unbundle "
                         + shlex.quote(transfer) + " && rm -- " + shlex.quote(transfer))
        commands = [(_WORKTREE_GIT + " update-ref refs/remotes/origin/main " + plan.main),
                    (_WORKTREE_GIT + " update-ref "
                     + shlex.quote("refs/remotes/origin/" + self.branch) + " " + plan.remote)]
        if local != plan.remote:
            commands.append(_WORKTREE_GIT + " merge --ff-only --no-autostash "
                            "--no-overwrite-ignore " + plan.remote)
        _sandbox_git(sandbox, " && ".join(commands))
        if identity(self.worktree) != (self.branch, plan.remote):
            raise GitSyncError("Git preflight did not leave the published branch")
        require_clean(sandbox)

    def commit(self, sandbox, message: str) -> None:
        """Stage and commit only inside the credential-free restricted sandbox."""
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

    def publish(self) -> str | None:
        """Return a result branch only when the exact thread ref lost a race."""
        with LocalGit(self.source, self.worktree) as repository:
            desired = self._verified_tip(repository)
            _, remote = repository.fetch(self.branch)
            if remote == desired or remote is not None and repository.ancestor(desired, remote):
                return None
            if remote is None or repository.ancestor(remote, desired):
                repository.publish(self.branch, desired)
                return None
            result = "assist-result/" + desired
            repository.publish(result, desired, create_only=True)
            return result

    def merge_main(self) -> str:
        """Merge only published refs privately; leave server/client checkouts alone."""
        with LocalGit(self.source) as repository:
            main, thread = repository.fetch(self.branch)
            if thread is None:
                raise GitSyncError("Published thread branch is unavailable")
            if repository.ancestor(thread, main):
                raise ValueError("No published thread changes to merge")
            try:
                tree = repository.git("merge-tree", "--write-tree", main, thread)
            except GitSyncError as error:
                raise GitSyncError("Main integration conflicted; resolve locally") from error
            merged = repository.git("-c", "user.name=Assist", "-c",
                                    "user.email=assist@localhost", "commit-tree", tree,
                                    "-p", main, "-p", thread,
                                    "-m", "Merge thread " + self.branch)
            if not repository.ancestor(main, merged) or not repository.ancestor(thread, merged):
                raise GitSyncError("Main integration ancestry is invalid")
            repository.publish("main", merged)
            return "Merged published thread " + self.branch
