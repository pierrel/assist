"""Git helper utilities.

This module centralizes all Git interactions so that tools can rely on a
simple, well-tested interface instead of invoking ``git`` directly.  It
provides helpers to locate the repository root, check whether files are
tracked, and commit changes.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def repo_root(path: Path) -> Path:
    """Return the root of the Git repository containing ``path``.

    Args:
        path: A directory believed to be inside a Git repository.

    Returns:
        The root directory of the repository.

    Raises:
        ValueError: If ``path`` is not inside a Git repository.
    """

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=path,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as exc:  # pragma: no cover - simple
        raise ValueError("Directory is not inside a Git repository") from exc
    return Path(result.stdout.strip())


def is_tracked(path: Path) -> bool:
    """Return ``True`` if ``path`` is tracked by Git.

    Args:
        path: Path to the file to check. The file's parent directory must be
            inside a Git repository.

    Returns:
        ``True`` if the file is tracked; ``False`` otherwise.
    """

    root = repo_root(path.parent)
    rel = path.resolve().relative_to(root)
    try:
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(rel)],
            cwd=root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError:
        return False
    return True


def commit_file(path: Path, message: str) -> None:
    """Stage ``path`` and commit it with ``message``.

    Args:
        path: Path to the file to commit. The file's parent directory must be
            inside a Git repository.
        message: Commit message to use.
    """

    root = repo_root(path.parent)
    rel = path.resolve().relative_to(root)
    subprocess.run(["git", "add", str(rel)], cwd=root, check=True)
    subprocess.run(["git", "commit", "-m", message], cwd=root, check=True)

