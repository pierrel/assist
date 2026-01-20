import os
import subprocess
import tempfile
from typing import List
from pydantic import BaseModel

class Change(BaseModel):
    path: str
    diff: str

def clone_repo(repo_url: str, dest_dir: str) -> None:
    """Always clone the repository into dest_dir."""
    parent = os.path.dirname(dest_dir)
    os.makedirs(parent, exist_ok=True)
    subprocess.run(['git', 'clone', repo_url, dest_dir], check=True)


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


class DomainManager:
    def __init__(self,
                 root: str | None = None,
                 repo: str | None = None):
        if root:
            self.root = root
        else:
            self.root = tempfile.mkdtemp()
        
        self.repo_path = os.path.join(self.root, 'domain')
        # Clone only if the repo does not already exist
        repo_exists = os.path.isdir(os.path.join(self.repo_path, '.git'))
        if repo and not repo_exists:
            clone_repo(self.repo_url, self.repo_path)
        elif not repo and not repo_exists:
            
            

    def changes(self) -> List[Change]:
        return git_diff(self.repo_path)

    def domain(self) -> str:
        return self.repo_path


