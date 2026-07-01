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


def current_branch(repo_dir: str) -> str:
    """Return the checked-out branch name.

    ``git rev-parse --abbrev-ref HEAD`` returns the branch name, the
    literal ``HEAD`` when detached, and we return '' when the ref can't
    be read at all (e.g. not a git repo).  Callers compare against
    ``'main'`` and treat everything else as "leave it alone".
    """
    result = subprocess.run(
        ['git', '-C', repo_dir, 'rev-parse', '--abbrev-ref', 'HEAD'],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ''


def _branch_exists(repo_dir: str, name: str) -> bool:
    """True if a local branch ``name`` already exists in ``repo_dir``."""
    return subprocess.run(
        ['git', '-C', repo_dir, 'show-ref', '--verify', '--quiet', f'refs/heads/{name}'],
        check=False,
    ).returncode == 0


def create_timestamped_branch(repo_dir: str, suffix: str | None = None,
                              start_point: str = 'main') -> str:
    """Create and checkout a new assist/[timestamp][-suffix] branch from ``start_point``.

    ``suffix`` is appended (with a leading hyphen) when supplied.  Pass
    the last 4 chars of the thread id to avoid collisions when two
    threads are created within the same UTC second.  ``start_point`` is
    normally ``main`` (a fresh thread branch) but is ``HEAD`` when
    re-attaching a detached HEAD so the detached commit is preserved.

    Returns the new branch name.
    """
    from datetime import timezone
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    base = f'assist/{ts}-{suffix}' if suffix else f'assist/{ts}'
    # Second-resolution timestamps plus a per-thread-constant suffix can
    # still collide when two branches are cut within the same UTC second
    # (e.g. a heal firing in the same second as a merge re-branch).
    # ``checkout -b`` fails hard on an existing name, which would abort
    # the turn and strand the work on whatever branch HEAD is on, so
    # disambiguate with a counter rather than letting it raise.
    branch_name = base
    n = 1
    while _branch_exists(repo_dir, branch_name):
        branch_name = f'{base}-{n}'
        n += 1
    subprocess.run(['git', '-C', repo_dir, 'checkout', '-b', branch_name, start_point],
                   check=True)
    return branch_name


def ensure_thread_branch(repo_dir: str, suffix: str | None = None) -> str:
    """Guarantee HEAD is on a thread branch so per-turn commits stay reviewable.

    The web flow renders ``git diff main...HEAD`` and only shows the
    Review / Merge buttons when that diff is non-empty.  If a thread is
    left on ``main`` — a merge that failed between checking out ``main``
    and re-branching, or an older flow that stranded it — every later
    commit lands on ``main``, the diff is always empty, and the work can
    never be reviewed.

    When HEAD is on ``main``, re-branch onto a fresh ``assist/<ts>``
    branch off ``main``.  ``git checkout -b`` carries the working tree's
    uncommitted edits forward, so an in-flight turn lands on the new
    branch.  When HEAD is DETACHED (the agent ran ``git checkout <sha>`` /
    ``origin/main``, or left a rebase state), re-branch AT the current
    commit so a commit doesn't orphan onto no branch — the detached work
    is preserved on a reviewable/pushable thread branch.  Already on a
    thread branch, or on some other named branch: left untouched.  Returns
    the branch HEAD ends up on.
    """
    branch = current_branch(repo_dir)
    if branch == 'HEAD':   # detached — re-attach at the current commit
        new_branch = create_timestamped_branch(repo_dir, suffix=suffix, start_point='HEAD')
        logger.warning("Thread was on detached HEAD; re-branched to %s at the current "
                       "commit so its work stays on a branch", new_branch)
        return new_branch
    if branch != 'main':
        return branch
    new_branch = create_timestamped_branch(repo_dir, suffix=suffix)
    logger.warning(
        "Thread was stranded on main; re-branched to %s so its work stays reviewable",
        new_branch,
    )
    return new_branch


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


def git_diff_range(repo_dir: str, range_spec: str) -> List[Change]:
    """Committed diffs across a commit range (e.g. ``origin/main..main`` = what local
    main has that origin/main doesn't). No working-tree or untracked files — a range
    diff is commit-to-commit."""
    changes: List[Change] = []
    names = subprocess.run(
        ['git', '-C', repo_dir, 'diff', '--name-only', range_spec],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False,
    )
    if names.returncode not in (0, 1):
        raise RuntimeError(f"git diff --name-only {range_spec} failed: {names.stderr.strip()}")
    for path in [l.strip() for l in names.stdout.splitlines() if l.strip()]:
        d = subprocess.run(
            ['git', '-C', repo_dir, 'diff', '--no-color', range_spec, '--', path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            encoding='utf-8', errors='replace', check=False,
        )
        if d.returncode not in (0, 1):
            raise RuntimeError(f"git diff {range_spec} failed for {path}: {d.stderr.strip()}")
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
        """Restore the thread-branch invariant, commit pending work, and push the THREAD
        branch to origin.

        Before committing we ensure HEAD is on an ``assist/<ts>`` thread
        branch (see :func:`ensure_thread_branch`): if a prior flow left
        the thread on ``main``, the commit would otherwise land on
        ``main`` and the work would never render in the review UI.

        Then it host-pushes the *thread branch* to origin every turn (see
        :meth:`_push_thread_branch`) so a branch is recoverable/fixable from a real
        computer (Pierre, PR #162). Two distinct pushes, do not conflate: the **thread
        branch** push here is host-side + automatic; the **``main``** publish stays
        user-gated (the "Push to origin" button → :meth:`push_main`). The **agent** can
        push neither (no creds + push-blocker middleware + the git-shim). So a deploy-box
        loss mid-thread no longer loses committed work — origin has the branch.
        """
        if not self.repo:
            return
        if not self._abort_inprogress_rebase():
            # The abort itself failed (corrupt/unwritable rebase state) — the repo is in a
            # bad rebase mid-flight. Committing/pushing into it would make it worse; leave
            # it as-is for manual attention (already logged at ERROR). Skip this sync.
            return
        branch = ensure_thread_branch(self.repo_path, suffix=self.branch_suffix)
        git_commit(self.repo_path, commit_message)
        self._push_thread_branch(branch)

    def _abort_inprogress_rebase(self) -> bool:
        """If the agent left a rebase in progress (its turn ended mid-conflict-loop),
        HEAD is detached and the end-of-turn commit would orphan onto no branch. Abort it:
        HEAD returns to the thread branch with all *committed* work intact (the design's
        un-stuck guarantee). The partial, uncommitted conflict resolution is discarded —
        the agent re-syncs next turn against a clean branch.

        Returns True if there was nothing to abort OR the abort succeeded; False if a
        rebase was in progress and ``git rebase --abort`` FAILED (so the caller must not
        commit into the broken state)."""
        git_dir = os.path.join(self.repo_path, ".git")
        if not (os.path.isdir(os.path.join(git_dir, "rebase-merge")) or
                os.path.isdir(os.path.join(git_dir, "rebase-apply"))):
            return True
        r = subprocess.run(['git', '-C', self.repo_path, 'rebase', '--abort'],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        if r.returncode != 0:
            logger.error("could not abort an in-progress rebase in %s: %s — skipping this "
                         "sync (repo needs manual attention)", self.repo_path, r.stderr.strip())
            return False
        logger.warning("aborted an agent-left in-progress rebase in %s (re-sync next turn)",
                       self.repo_path)
        return True

    def _push_thread_branch(self, branch: str) -> None:
        """Push the thread branch to origin after each turn so Pierre can check it out and
        fix it from a real computer. Best-effort: a push failure must not break the turn
        (origin briefly unreachable, etc.) — it's logged, not raised. NOT the agent (host
        holds the creds); NOT ``main`` (that's the user-gated publish). ``--force-with-
        lease`` because the agent may have rebased the branch this turn."""
        if branch in ("", "HEAD"):
            # Detached HEAD (e.g. the agent left a rebase mid-flight) or an unreadable
            # ref — no thread branch to publish; pushing "HEAD" would junk a ref on origin.
            return
        pushed = subprocess.run(
            ['git', '-C', self.repo_path, 'push', '--force-with-lease',
             'origin', branch],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False,
        )
        if pushed.returncode != 0:
            logger.warning("thread-branch push failed for %s: %s",
                           branch, pushed.stderr.strip())

    def fetch_origin(self) -> None:
        """Refresh ``origin/main`` in the clone so the agent can rebase onto a current
        origin/main. The agent CANNOT fetch from inside the sandbox — the git
        privilege-separation that blocks push also makes ``git-upload-pack`` inaccessible
        to the non-root agent — so the host (which has full git + origin access, incl.
        local-path "life"-style domains) keeps the ref fresh here, at turn start.
        Best-effort: a failure (origin briefly unreachable) must not break the turn."""
        if not self.repo or not is_git_repo(self.repo_path):
            return
        r = subprocess.run(
            ['git', '-C', self.repo_path, 'fetch', 'origin'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False,
        )
        if r.returncode != 0:
            logger.warning("origin pre-fetch failed for %s: %s",
                           self.repo_path, r.stderr.strip())

    def merge_to_main(self) -> str:
        """Rebase the thread branch onto ``origin/main`` and squash-merge it
        into local ``main`` with a deterministic commit summary.

        Sequence:

        1. ``git fetch origin``.
        2. Refuse if local ``main`` has unpushed commits from a previous
           merge — the user must push to ``origin`` before another
           merge can land cleanly on top.
        3. ``git rebase origin/main`` on the thread branch.  On
           conflict: abort the rebase and raise
           :class:`MergeConflictError` with the unmerged file list.
        4. ``git checkout main`` and ``git reset --hard origin/main``
           (safe — step 2 verified local ``main`` had no extra
           commits).
        5. ``git merge --squash <thread-branch>``, build the deterministic
           squash summary, and commit with it.
        6. Re-create a fresh ``assist/<ts>-<suffix>`` thread branch off
           ``main`` so the user can keep chatting in the same thread
           after the merge.

        Steps 4-6 are wrapped so a failure anywhere in that window never
        leaves the thread stranded on ``main`` (the state that hides its
        work from the review UI): if the squash already committed, the
        thread is re-branched off ``main`` (the merge is preserved);
        otherwise HEAD is restored to the thread branch.

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
        branch_name = current_branch(self.repo_path)

        if branch_name == 'main':
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
                raise MergeConflictError(branch_name, conflicted)
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

        # Switch to main and bring it level with origin/main before
        # squashing the rebased thread branch on top.  From the checkout
        # below until the post-merge re-branch, HEAD sits on ``main``; if
        # any step in between raises (squash, the LLM summary call, the
        # commit, or the re-branch), we must not return with the thread
        # stranded on ``main`` — that's the exact state that makes work
        # unreviewable.  The ``squashed`` flag tells the recovery path
        # whether a real merge landed on ``main`` (keep it, just put the
        # thread on a fresh branch) or not (return to the thread branch
        # so its work is preserved).
        squashed = False
        try:
            subprocess.run(['git', '-C', self.repo_path, 'checkout', 'main'], check=True)
            subprocess.run(['git', '-C', self.repo_path, 'reset', '--hard', 'origin/main'], check=True)
            subprocess.run(
                ['git', '-C', self.repo_path, 'merge', '--squash', branch_name],
                check=True,
            )

            summary = self._summarize_merge(diffs, branch_name)
            subprocess.run(['git', '-C', self.repo_path, 'commit', '-m', summary], check=True)
            squashed = True

            # Roll the thread forward onto a new branch off main so the
            # user can keep chatting; preserves pre-feature UX.
            create_timestamped_branch(self.repo_path, suffix=self.branch_suffix)
        except Exception:
            try:
                if squashed:
                    # The squash committed onto main (a real merge that
                    # just needs pushing); only the re-branch failed.
                    # Put the thread on a fresh branch off main.
                    ensure_thread_branch(self.repo_path, suffix=self.branch_suffix)
                else:
                    # The merge never committed; main is back at
                    # origin/main.  Force-return to the thread branch so
                    # its (committed) work stays intact and reviewable —
                    # ``-f`` discards the transient squash staging, which
                    # is just a duplicate of that branch's own diff.
                    subprocess.run(
                        ['git', '-C', self.repo_path, 'checkout', '-f', branch_name],
                        check=False,
                    )
            except Exception:
                logger.exception("Failed to restore thread branch after merge error")
            raise

        return summary

    def _summarize_merge(self, diffs: List[Change], branch: str) -> str:
        """The squash-commit message — deterministic (by file count, truncated to the
        72-char ceiling). The per-turn commits already carry the agent's prose and the
        reviewer sees the full diff, so the host LLM round-trip that used to write this
        was dropped in the agent-driven-git rework (one fewer model call, less code)."""
        return f"Merge {branch}: {len(diffs)} file(s)"[:72]

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

    def push_preview(self) -> List[Change]:
        """The diff a push would send: local ``main`` vs ``origin/main`` — what the user
        reviews before clicking "Push to origin", distinct from the per-thread review diff
        (``main_diff``). Fetches origin first so the comparison is against the current
        remote. Empty when nothing is unpushed."""
        if not self.repo or not is_git_repo(self.repo_path):
            return []
        r = subprocess.run(['git', '-C', self.repo_path, 'fetch', 'origin'],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        if r.returncode != 0:
            logger.warning("push-preview fetch failed for %s: %s (preview may be stale)",
                           self.repo_path, r.stderr.strip())
        return git_diff_range(self.repo_path, 'origin/main..main')
