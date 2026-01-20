import os
import subprocess
import tempfile
from typing import List
from pydantic import BaseModel
from datetime import datetime

class Change(BaseModel):
    path: str
    diff: str

def clone_repo(repo_url: str, dest_dir: str) -> None:
    """Always clone the repository into dest_dir."""
    parent = os.path.dirname(dest_dir)
    os.makedirs(parent, exist_ok=True)
    subprocess.run(['git', 'clone', '--branch', 'main', repo_url, dest_dir], check=True)
    # Create a new branch assist/[timestamp]
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    subprocess.run(['git', '-C', dest_dir, 'checkout', '-b', f'assist/{ts}'], check=True)


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
            text=True,
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
            text=True,
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
            text=True,
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
            text=True,
            check=False,
        )
        if d.returncode not in (0, 1):
            raise RuntimeError(f"git diff --no-index failed for {path}: {d.stderr.strip()}")
        if d.stdout:
            changes.append(Change(path=path, diff=d.stdout))

    return changes


def git_push(repo_dir: str) -> None:
    """Push current branch to origin, setting upstream if needed."""
    # Determine current branch
    cur = subprocess.run(['git', '-C', repo_dir, 'rev-parse', '--abbrev-ref', 'HEAD'],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    branch = cur.stdout.strip()
    subprocess.run(['git', '-C', repo_dir, 'push', '--set-upstream', 'origin', branch], check=True)


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


def merge_main_into_current_and_push(repo_path: str) -> None:
    """Merge latest main into current branch and push the branch to origin."""
    import subprocess
    # Ensure we have latest remote refs
    subprocess.run(['git', '-C', repo_path, 'fetch', 'origin'], check=True)
    # Determine current branch
    cur = subprocess.run(['git', '-C', repo_path, 'rev-parse', '--abbrev-ref', 'HEAD'], stdout=subprocess.PIPE, text=True, check=True)
    branch = cur.stdout.strip()
    # Merge origin/main into current branch
    subprocess.run(['git', '-C', repo_path, 'merge', '--no-edit', 'origin/main'], check=True)
    # Push current branch
    subprocess.run(['git', '-C', repo_path, 'push', '--set-upstream', 'origin', branch], check=True)

class DomainManager:
    def __init__(self,
                 root: str | None = None,
                 repo: str | None = None):
        if root:
            self.root = root
        else:
            self.root = tempfile.mkdtemp()

        self.repo = repo
        self.repo_path = os.path.join(self.root, 'domain')
        # Clone only if the repo does not already exist
        repo_exists = os.path.isdir(self.repo_path)
        if repo and not repo_exists:
            clone_repo(repo, self.repo_path)
        elif not repo and not repo_exists:
            os.makedirs(self.repo_path, exist_ok=True)

    def changes(self) -> List[Change]:
        if self.repo:
            return git_diff(self.repo_path)
        else:
            return []

    def main_diff(self) -> List[Change]:
        if self.repo:
            return git_diff_main(self.repo_path)
        else:
            return []

    def domain(self) -> str:
        return self.repo_path

    def sync(self, commit_message: str) -> None:
        if self.repo:
            git_commit(self.repo_path, commit_message)
            git_push(self.repo_path)
