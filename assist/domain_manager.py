import logging
import os
import subprocess
import tempfile
from typing import List
from pydantic import BaseModel
from datetime import datetime

logger = logging.getLogger(__name__)


class MergeConflictError(Exception):
    """Raised by ``DomainManager.merge_to_main`` when the rebase onto
    ``origin/main`` produces a conflict.  The rebase is aborted before
    this is raised, so the working tree returns to the thread branch in
    a clean state — the agent can resolve the conflict by re-running
    the merge after fixing the underlying disagreement.
    """

    def __init__(self, branch: str, files: list[str]):
        self.branch = branch
        self.files = files
        super().__init__(
            f"Rebase conflict on {branch}: {len(files)} unmerged file(s)"
        )


class OriginAdvancedError(Exception):
    """Raised by ``DomainManager.push_main`` when ``origin/main`` has
    advanced past the local ``main`` ancestor — pushing would not be a
    fast-forward.  The caller surfaces this to the user with a message
    explaining that they must re-merge before re-attempting the push.
    """


class Change(BaseModel):
    path: str
    diff: str


def create_timestamped_branch(repo_dir: str, suffix: str | None = None) -> str:
    """Create and checkout a new assist/[timestamp][-suffix] branch from main.

    ``suffix`` is appended (with a leading hyphen) when supplied.  Pass
    the last 4 chars of the thread id to avoid collisions when two
    threads are created within the same UTC second.

    Returns the new branch name.
    """
    from datetime import timezone
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    branch_name = f'assist/{ts}-{suffix}' if suffix else f'assist/{ts}'
    subprocess.run(['git', '-C', repo_dir, 'checkout', '-b', branch_name, 'main'], check=True)
    return branch_name


def clone_repo(repo_url: str, dest_dir: str, branch_suffix: str | None = None) -> None:
    """Always clone the repository into dest_dir.

    ``branch_suffix`` is forwarded to :func:`create_timestamped_branch`
    so the per-thread branch carries an unambiguous identifier.
    """
    parent = os.path.dirname(dest_dir)
    os.makedirs(parent, exist_ok=True)
    subprocess.run(['git', 'clone', '--branch', 'main', repo_url, dest_dir], check=True)
    create_timestamped_branch(dest_dir, suffix=branch_suffix)


def git_diff(repo_dir: str) -> List[Change]:
    """Return structured diffs including file path and content diff.

    Includes tracked changes and untracked new files (compared to /dev/null).
    """
    changes: List[Change] = []

    # Tracked changes: list changed files, then diff each
    names = subprocess.run(
        ['git', '-C', repo_dir, 'diff', '--name-only'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if names.returncode not in (0, 1):
        raise RuntimeError(f"git diff --name-only failed: {names.stderr.strip()}")

    for path in [l.strip() for l in names.stdout.splitlines() if l.strip()]:
        d = subprocess.run(
            ['git', '-C', repo_dir, 'diff', '--no-color', '--', path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding='utf-8',
            errors='replace',
            check=False,
        )
        if d.returncode not in (0, 1):
            raise RuntimeError(f"git diff failed for {path}: {d.stderr.strip()}")
        if d.stdout:
            changes.append(Change(path=path, diff=d.stdout))

    # Untracked files: show as diff from /dev/null
    ls = subprocess.run(
        ['git', '-C', repo_dir, 'ls-files', '--others', '--exclude-standard'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if ls.returncode != 0:
        raise RuntimeError(f"git ls-files failed: {ls.stderr.strip()}")

    for path in [line.strip() for line in ls.stdout.splitlines() if line.strip()]:
        d = subprocess.run(
            ['git', '-C', repo_dir, 'diff', '--no-index', '--no-color', '--', '/dev/null', path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding='utf-8',
            errors='replace',
            check=False,
        )
        if d.returncode not in (0, 1):
            raise RuntimeError(f"git diff --no-index failed for {path}: {d.stderr.strip()}")
        if d.stdout:
            changes.append(Change(path=path, diff=d.stdout))

    return changes


def git_diff_main(repo_dir: str) -> List[Change]:
    """Return diffs of current working tree compared to ``main``.

    Includes tracked changes versus ``main`` and untracked files as added.
    """
    changes: List[Change] = []

    # Files changed relative to main
    names = subprocess.run(
        ['git', '-C', repo_dir, 'diff', '--name-only', 'main...'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if names.returncode not in (0, 1):
        raise RuntimeError(f"git diff --name-only main... failed: {names.stderr.strip()}")

    for path in [l.strip() for l in names.stdout.splitlines() if l.strip()]:
        d = subprocess.run(
            ['git', '-C', repo_dir, 'diff', '--no-color', 'main...', '--', path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding='utf-8',
            errors='replace',
            check=False,
        )
        if d.returncode not in (0, 1):
            raise RuntimeError(f"git diff main... failed for {path}: {d.stderr.strip()}")
        if d.stdout:
            changes.append(Change(path=path, diff=d.stdout))

    # Untracked files: show as diff from /dev/null
    ls = subprocess.run(
        ['git', '-C', repo_dir, 'ls-files', '--others', '--exclude-standard'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if ls.returncode != 0:
        raise RuntimeError(f"git ls-files failed: {ls.stderr.strip()}")

    for path in [line.strip() for line in ls.stdout.splitlines() if line.strip()]:
        d = subprocess.run(
            ['git', '-C', repo_dir, 'diff', '--no-index', '--no-color', '--', '/dev/null', path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding='utf-8',
            errors='replace',
            check=False,
        )
        if d.returncode not in (0, 1):
            raise RuntimeError(f"git diff --no-index failed for {path}: {d.stderr.strip()}")
        if d.stdout:
            changes.append(Change(path=path, diff=d.stdout))

    return changes


def git_commit(repo_dir: str, message: str) -> None:
    """Stage all changes (including new files) and commit with message.

    If there are no staged changes, do nothing instead of failing.
    """
    subprocess.run(['git', '-C', repo_dir, 'add', '-A'], check=True)
    # Check if there are staged changes; 'git diff --cached --quiet' exits 1 when there are diffs
    check = subprocess.run(['git', '-C', repo_dir, 'diff', '--cached', '--quiet'])
    if check.returncode == 0:
        return
    subprocess.run(['git', '-C', repo_dir, 'commit', '-m', message], check=True)


def is_git_repo(path: str) -> bool:
    """Check if path is a git repository (with or without remote)."""
    try:
        inside = subprocess.run(
            ['git', '-C', path, 'rev-parse', '--is-inside-work-tree'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        return inside.returncode == 0 and inside.stdout.strip() == 'true'
    except Exception:
        return False


def git_repo(path: str) -> str | None:
    """Return the remote URL of the git repository at path, or None if not a repo."""
    try:
        if not is_git_repo(path):
            return None
        remote = subprocess.run(
            ['git', '-C', path, 'remote', 'get-url', 'origin'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if remote.returncode != 0:
            return None
        return remote.stdout.strip() or None
    except Exception:
        return None

class DomainManager:
    """Manages domain git repositories."""

    def __init__(self,
                 repo_path: str | None = None,
                 repo: str | None = None,
                 branch_suffix: str | None = None):
        """Initialize DomainManager with a repository.

        Args:
            repo_path: Path to local repository
            repo: Remote repository URL (optional — sandbox works without git)
            branch_suffix: Forwarded to :func:`create_timestamped_branch`
                during clone and post-merge re-branch.  Pass the last 4
                chars of the owning thread id.

        Raises:
            ValueError: If repo_path has a remote but it conflicts with repo arg
        """
        if repo_path:
            self.repo_path = repo_path
        else:
            self.repo_path = tempfile.mkdtemp()

        # If repo_path is already a git repo, use its remote
        existing_remote = git_repo(self.repo_path)
        self.repo = existing_remote or repo
        self.branch_suffix = branch_suffix

        # Git setup is optional — only clone if we have a remote URL
        if self.repo:
            repo_exists = os.path.isdir(self.repo_path)
            is_empty = not os.listdir(self.repo_path) if repo_exists else True

            if not existing_remote and (not repo_exists or is_empty):
                clone_repo(self.repo, self.repo_path, branch_suffix=branch_suffix)
        else:
            os.makedirs(self.repo_path, exist_ok=True)

    def changes(self) -> List[Change]:
        if not self.repo:
            return []
        if not is_git_repo(self.repo_path):
            logger.warning(
                "changes() skipped: %s has self.repo=%s but is not a git repo",
                self.repo_path, self.repo,
            )
            return []
        return git_diff(self.repo_path)

    def main_diff(self) -> List[Change]:
        if not self.repo:
            return []
        if not is_git_repo(self.repo_path):
            logger.warning(
                "main_diff() skipped: %s has self.repo=%s but is not a git repo",
                self.repo_path, self.repo,
            )
            return []
        return git_diff_main(self.repo_path)

    def has_changes_vs_main(self) -> bool:
        """True iff the working tree has unmerged work compared to ``main``.

        Cheaper than :meth:`main_diff` when only a bool is needed —
        used by the index page to decide whether to render an
        "unmerged" status badge per thread.  Three independent checks
        — ``True`` if any signals dirty:

        - ``git diff --quiet HEAD`` exits 1 if the working tree or
          index has any uncommitted edits to tracked files.  Critical
          because the assist flow can sit between an edit and the
          end-of-turn ``sync()`` commit; the user expects the badge
          to fire on tracked-but-uncommitted dirt too.
        - ``git diff --quiet main...`` exits 1 if any *committed* work
          on this branch is not in ``main``.  Catches the post-sync
          "ready to merge" steady state.
        - ``git ls-files --others --exclude-standard`` lists any
          untracked-but-not-ignored files; if non-empty, the thread
          has new files that haven't been committed yet.

        For each git invocation we treat any returncode other than
        the documented 0 (clean) / 1 (dirty) as "no info, assume
        clean" rather than falsely badging every thread as unmerged
        — e.g., when the repo has no ``main`` branch yet.
        """
        if not self.repo:
            return False
        if not is_git_repo(self.repo_path):
            return False
        # 1. Working tree + index vs HEAD (tracked but uncommitted).
        worktree = subprocess.run(
            ['git', '-C', self.repo_path, 'diff', '--quiet', 'HEAD'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
        if worktree.returncode == 1:
            return True
        # 2. Branch commits vs main merge-base.
        committed = subprocess.run(
            ['git', '-C', self.repo_path, 'diff', '--quiet', 'main...'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
        if committed.returncode == 1:
            return True
        # 3. Untracked files (independent of either diff).
        untracked = subprocess.run(
            ['git', '-C', self.repo_path, 'ls-files', '--others', '--exclude-standard'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False,
        )
        return bool(untracked.stdout.strip())

    def domain(self) -> str:
        return self.repo_path

    def sync(self, commit_message: str) -> None:
        """Commit any pending work on the thread branch.  Does NOT push.

        Pushes are gated to the user-initiated merge → push-main flow;
        the agent must not be able to publish to ``origin`` directly,
        and we extend that contract to host-side automated pushes too
        (decision logged in =docs/2026-05-07-per-thread-web-git-isolation.org=).
        Trade-off: an in-flight thread is local-only until the user
        merges, so a deploy-box loss mid-thread loses the work.
        """
        if not self.repo:
            return
        git_commit(self.repo_path, commit_message)

    def merge_to_main(self, summary_model=None) -> str:
        """Rebase the thread branch onto ``origin/main`` and squash-merge it
        into local ``main`` with an AI-generated commit summary.

        Sequence:

        1. Generate the merge commit summary from the diff (model
           invocation, optional fallback).
        2. ``git fetch origin``.
        3. Refuse if local ``main`` has unpushed commits from a previous
           merge — the user must push to ``origin`` before another
           merge can land cleanly on top.
        4. ``git rebase origin/main`` on the thread branch.  On
           conflict: abort the rebase and raise
           :class:`MergeConflictError` with the unmerged file list.
        5. ``git checkout main`` and ``git reset --hard origin/main``
           (safe — step 3 verified local ``main`` had no extra
           commits).
        6. ``git merge --squash <thread-branch>`` and commit with the
           summary.
        7. Re-create a fresh ``assist/<ts>-<suffix>`` thread branch off
           ``main`` so the user can keep chatting in the same thread
           after the merge.

        Does **not** push — that's the caller's job, via
        :meth:`push_main`, gated to a user-initiated UI action.

        Returns the merge commit summary.

        Raises:
            ValueError: HEAD is on ``main``, no changes to merge, or
                local ``main`` has unpushed commits.
            MergeConflictError: rebase produced a conflict; the rebase
                was aborted and the working tree is back on the thread
                branch in a clean state.
            subprocess.CalledProcessError: any other git failure.
        """
        cur = subprocess.run(
            ['git', '-C', self.repo_path, 'rev-parse', '--abbrev-ref', 'HEAD'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
        )
        current_branch = cur.stdout.strip()

        if current_branch == 'main':
            raise ValueError("Already on main branch - nothing to merge")

        diffs = self.main_diff()
        if not diffs:
            raise ValueError("No changes to merge")

        # Fetch the latest origin/main so the rebase target is current.
        subprocess.run(['git', '-C', self.repo_path, 'fetch', 'origin'], check=True)

        # Refuse if local main is ahead of origin/main — a previous
        # merge hasn't been pushed yet.  Layering another merge on top
        # would commingle two threads' work in a single non-ff state
        # the user couldn't recover from.  Force the user to push first.
        ahead = subprocess.run(
            ['git', '-C', self.repo_path, 'rev-list', '--count', 'origin/main..main'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False,
        )
        ahead_count = 0
        if ahead.returncode == 0:
            try:
                ahead_count = int(ahead.stdout.strip())
            except ValueError:
                pass
        if ahead_count > 0:
            raise ValueError(
                "Local main has unpushed commits from a previous merge. "
                "Push to origin first, then retry the merge."
            )

        # Rebase the thread branch onto origin/main.
        rebase = subprocess.run(
            ['git', '-C', self.repo_path, 'rebase', 'origin/main'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False,
        )
        if rebase.returncode != 0:
            unmerged = subprocess.run(
                ['git', '-C', self.repo_path, 'diff', '--name-only', '--diff-filter=U'],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False,
            )
            conflicted = [l.strip() for l in unmerged.stdout.splitlines() if l.strip()]
            subprocess.run(['git', '-C', self.repo_path, 'rebase', '--abort'], check=False)
            if conflicted:
                raise MergeConflictError(current_branch, conflicted)
            # Non-conflict rebase failure (e.g. dirty working tree at
            # merge time, corrupt branch state).  Don't paper this
            # over as a "conflict" — the agent would chase a
            # nonexistent conflict.  Bubble up the underlying git
            # failure so the route returns a 500 with the stderr.
            raise subprocess.CalledProcessError(
                rebase.returncode,
                ['git', '-C', self.repo_path, 'rebase', 'origin/main'],
                output=rebase.stdout,
                stderr=rebase.stderr,
            )

        # Rebase succeeded — now commit to spending the LLM call on
        # the summary.  Doing this *after* the rebase avoids burning
        # wall-clock and queue capacity on merges that would have
        # hit the unpushed-main check or a conflict.
        summary = self._summarize_merge(diffs, current_branch, summary_model)

        # Switch to main and bring it level with origin/main before
        # squashing the rebased thread branch on top.
        subprocess.run(['git', '-C', self.repo_path, 'checkout', 'main'], check=True)
        subprocess.run(['git', '-C', self.repo_path, 'reset', '--hard', 'origin/main'], check=True)
        subprocess.run(
            ['git', '-C', self.repo_path, 'merge', '--squash', current_branch],
            check=True,
        )
        subprocess.run(['git', '-C', self.repo_path, 'commit', '-m', summary], check=True)

        # Roll the thread forward onto a new branch off main so the
        # user can keep chatting; preserves pre-feature UX.
        create_timestamped_branch(self.repo_path, suffix=self.branch_suffix)

        return summary

    def _summarize_merge(self, diffs: List[Change], branch: str, model) -> str:
        """Build the merge commit summary — model-driven if available,
        deterministic fallback otherwise.  Extracted for testability.
        """
        if model is None:
            # Honest fallback: summarise by file count rather than
            # naming the first file as if it's representative — the
            # original "Merge X: README.md" style was misleading on
            # multi-file merges.  Truncated to the historical
            # 72-char commit-message ceiling.
            return f"Merge {branch}: {len(diffs)} file(s)"[:72]

        log_result = subprocess.run(
            ['git', '-C', self.repo_path, 'log', 'main', '--oneline', '-20'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False,
        )
        recent_commits = log_result.stdout if log_result.returncode == 0 else ""
        diff_content = "\n\n".join([f"File: {c.path}\n{c.diff}" for c in diffs])

        prompt = f"""You are summarizing a git merge for a commit message. Be concise and clear.

Recent commit messages from this repository:
{recent_commits}

Changes in this merge:
{diff_content}

Write a single-line commit message (max 72 characters) that summarizes these changes.
Follow the style of recent commits if applicable. Do not include any explanation, just the commit message."""

        from langchain_core.messages import HumanMessage
        response = model.invoke([HumanMessage(content=prompt)])
        return response.content.strip().split('\n')[0][:72]

    def push_main(self) -> None:
        """Fast-forward push local ``main`` to ``origin/main``.

        Refuses (raises :class:`OriginAdvancedError`) if ``origin/main``
        has advanced past local ``main``'s ancestor — pushing would not
        be a fast-forward, and silently overwriting remote work is the
        worst possible failure mode.  The caller is expected to surface
        the error to the user with the "click Merge to Main again, then
        click Push again" copy.

        No-op (raises ``ValueError``) if no remote is configured.
        """
        if not self.repo:
            raise ValueError("No remote configured for this domain")
        subprocess.run(['git', '-C', self.repo_path, 'fetch', 'origin'], check=True)
        # is-ancestor exits 0 if origin/main is reachable from local main
        # (i.e. local main is at-or-ahead of origin/main and nothing
        # external moved).  Anything else is the divergent state.
        check = subprocess.run(
            ['git', '-C', self.repo_path, 'merge-base', '--is-ancestor', 'origin/main', 'main'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
        if check.returncode != 0:
            raise OriginAdvancedError(
                "origin/main has advanced past local main. "
                "Click 'Merge to Main' again to rebase, then click Push again."
            )
        subprocess.run(['git', '-C', self.repo_path, 'push', 'origin', 'main'], check=True)

    def has_unpushed_main(self) -> bool:
        """True iff local ``main`` is strictly ahead of ``origin/main``.

        Drives the visibility of the "Push to origin" UI button —
        rendered only after a successful merge has put unpushed work on
        local ``main``.  Cheap (no fetch) — staleness is acceptable
        because the button is paired with a server-side fetch when the
        user clicks.
        """
        if not self.repo or not is_git_repo(self.repo_path):
            return False
        result = subprocess.run(
            ['git', '-C', self.repo_path, 'rev-list', '--count', 'origin/main..main'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False,
        )
        if result.returncode != 0:
            return False
        try:
            return int(result.stdout.strip()) > 0
        except ValueError:
            return False
