import logging
import os
import subprocess
import tempfile
from typing import List
from pydantic import BaseModel
from datetime import datetime

logger = logging.getLogger(__name__)

SANDBOX_IMAGE = "assist-sandbox"

class Change(BaseModel):
    path: str
    diff: str


def create_timestamped_branch(repo_dir: str) -> str:
    """Create and checkout a new assist/[timestamp] branch from main.

    Returns the new branch name.
    """
    from datetime import timezone
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    branch_name = f'assist/{ts}'
    subprocess.run(['git', '-C', repo_dir, 'checkout', '-b', branch_name, 'main'], check=True)
    return branch_name


def clone_repo(repo_url: str, dest_dir: str) -> None:
    """Always clone the repository into dest_dir."""
    parent = os.path.dirname(dest_dir)
    os.makedirs(parent, exist_ok=True)
    subprocess.run(['git', 'clone', '--branch', 'main', repo_url, dest_dir], check=True)
    # Create a new branch assist/[timestamp]
    create_timestamped_branch(dest_dir)


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
    """Manages domain repositories and optional Docker sandbox containers.

    Docker lifecycle is class-level: one Docker client shared across all instances,
    with a container registry keyed by repo_path for cleanup.
    """

    _docker_client = None
    _containers: dict[str, "docker.models.containers.Container"] = {}  # type: ignore[name-defined]

    def __init__(self,
                 repo_path: str | None = None,
                 repo: str | None = None):
        """Initialize DomainManager with a repository.

        Args:
            repo_path: Path to local repository
            repo: Remote repository URL (optional — sandbox works without git)

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

        # Git setup is optional — only clone if we have a remote URL
        if self.repo:
            repo_exists = os.path.isdir(self.repo_path)
            is_empty = not os.listdir(self.repo_path) if repo_exists else True

            if not existing_remote and (not repo_exists or is_empty):
                clone_repo(self.repo, self.repo_path)
        else:
            os.makedirs(self.repo_path, exist_ok=True)

    def changes(self) -> List[Change]:
        if not self.repo:
            return []
        return git_diff(self.repo_path)

    def main_diff(self) -> List[Change]:
        if not self.repo:
            return []
        return git_diff_main(self.repo_path)

    def domain(self) -> str:
        return self.repo_path

    def sync(self, commit_message: str) -> None:
        if not self.repo:
            return
        git_commit(self.repo_path, commit_message)
        git_push(self.repo_path)

    def merge_to_main(self, summary_model=None) -> str:
        """Merge current branch into main with AI-generated summary.

        Steps:
        1. Generate merge commit summary using AI model
        2. Fetch and pull latest main from origin
        3. Squash merge current branch into main
        4. Push main to origin
        5. Delete old remote branch
        6. Create new timestamped branch
        7. Push new branch to origin

        Returns the merge commit message.
        Raises subprocess.CalledProcessError if any git command fails.
        """

        # Get current branch
        cur = subprocess.run(
            ['git', '-C', self.repo_path, 'rev-parse', '--abbrev-ref', 'HEAD'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
        )
        current_branch = cur.stdout.strip()

        if current_branch == 'main':
            raise ValueError("Already on main branch - nothing to merge")

        # Get the full diff vs main
        diffs = self.main_diff()
        if not diffs:
            raise ValueError("No changes to merge")

        # Generate summary using model
        if summary_model:
            # Get recent commit messages from git log for context
            log_result = subprocess.run(
                ['git', '-C', self.repo_path, 'log', 'main', '--oneline', '-20'],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False
            )
            recent_commits = log_result.stdout if log_result.returncode == 0 else ""

            # Build diff content
            diff_content = "\n\n".join([f"File: {c.path}\n{c.diff}" for c in diffs])

            # Ask model to summarize
            prompt = f"""You are summarizing a git merge for a commit message. Be concise and clear.

Recent commit messages from this repository:
{recent_commits}

Changes in this merge:
{diff_content}

Write a single-line commit message (max 72 characters) that summarizes these changes.
Follow the style of recent commits if applicable. Do not include any explanation, just the commit message."""

            from langchain_core.messages import HumanMessage
            response = summary_model.invoke([HumanMessage(content=prompt)])
            summary = response.content.strip().split('\n')[0][:72]  # First line, max 72 chars
        else:
            # Fallback: use first file changed
            summary = f"Merge {current_branch}: {diffs[0].path}"

        # Fetch latest from origin
        subprocess.run(['git', '-C', self.repo_path, 'fetch', 'origin'], check=True)

        # Checkout main
        subprocess.run(['git', '-C', self.repo_path, 'checkout', 'main'], check=True)

        # Pull main to make sure we're up to date
        subprocess.run(['git', '-C', self.repo_path, 'pull', 'origin', 'main'], check=True)

        # Merge current branch into main (now that we're on main)
        try:
            subprocess.run(
                ['git', '-C', self.repo_path, 'merge', '--squash', current_branch],
                check=True
            )
        except subprocess.CalledProcessError as e:
            # Merge conflict - abort and re-raise
            subprocess.run(['git', '-C', self.repo_path, 'merge', '--abort'], check=False)
            subprocess.run(['git', '-C', self.repo_path, 'checkout', current_branch], check=False)
            raise ValueError(f"Merge conflict detected. Please resolve conflicts manually.") from e

        # Commit the squash merge
        subprocess.run(['git', '-C', self.repo_path, 'commit', '-m', summary], check=True)

        # Push main to origin with the squashed merge
        subprocess.run(['git', '-C', self.repo_path, 'push', 'origin', 'main'], check=True)

        # Delete the old remote branch since it's been merged
        # This happens AFTER pushing main to ensure the merge is safely on the remote first
        subprocess.run(
            ['git', '-C', self.repo_path, 'push', 'origin', '--delete', current_branch],
            check=False  # Don't fail if branch doesn't exist on remote
        )

        # Create a new branch off main for future work (reuses thread creation logic)
        # This ensures new prompts don't affect main directly
        new_branch = create_timestamped_branch(self.repo_path)

        # Push the new branch to origin
        subprocess.run(['git', '-C', self.repo_path, 'push', '--set-upstream', 'origin', new_branch], check=True)

        return summary

    # --- Docker sandbox lifecycle ---

    @classmethod
    def _get_docker_client(cls):
        """Lazily create and cache a Docker client."""
        if cls._docker_client is None:
            import docker
            cls._docker_client = docker.from_env()
        return cls._docker_client

    def get_sandbox_backend(self):
        """Return a DockerSandboxBackend for this domain, creating a container if needed.

        Returns None if Docker is not available.
        """
        if self.repo_path in self._containers:
            container = self._containers[self.repo_path]
            # Check container is still running
            try:
                container.reload()
                if container.status == "running":
                    from assist.sandbox import DockerSandboxBackend
                    return DockerSandboxBackend(container)
            except Exception:
                # Container gone, remove from registry
                self._containers.pop(self.repo_path, None)

        try:
            client = self._get_docker_client()
            container = client.containers.run(
                SANDBOX_IMAGE,
                detach=True,
                remove=True,
                volumes={self.repo_path: {"bind": "/workspace", "mode": "rw"}},
                working_dir="/workspace",
                stdin_open=True,
                tty=False,
                labels={"assist.sandbox": "true"},
            )
            self._containers[self.repo_path] = container
            logger.info("Started sandbox container %s for %s", container.id[:12], self.repo_path)
            from assist.sandbox import DockerSandboxBackend
            return DockerSandboxBackend(container)
        except Exception as e:
            logger.warning("Docker sandbox unavailable: %s", e)
            return None

    def cleanup(self) -> None:
        """Stop the container for this domain. Removal is automatic (--rm)."""
        container = self._containers.pop(self.repo_path, None)
        if container:
            try:
                container.stop(timeout=5)
                logger.info("Cleaned up container for %s", self.repo_path)
            except Exception as e:
                logger.warning("Container cleanup failed: %s", e)

    @classmethod
    def cleanup_all(cls) -> None:
        """Stop all tracked sandbox containers. Removal is automatic (--rm)."""
        for path, container in list(cls._containers.items()):
            try:
                container.stop(timeout=5)
                logger.info("Cleaned up container for %s", path)
            except Exception as e:
                logger.warning("Container cleanup failed for %s: %s", path, e)
        cls._containers.clear()
