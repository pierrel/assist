from __future__ import annotations

from pathlib import Path
from pathspec import PathSpec

# Default ignore patterns for files and directories.
DEFAULT_IGNORE_PATTERNS = [
    "**/.*",  # all dotfiles and directories
    "**/__pycache__/",
    "**/*.py[cod]",
    "**/.mypy_cache/",
    "**/.pytest_cache/",
    "**/*.egg-info/",
    "**/build/",
    "**/dist/",
    "**/node_modules/",
    "**/.venv/",
    "**/venv/",
    "**/.DS_Store",
    "**/Thumbs.db",
]


def load_ignore_spec(root: Path) -> PathSpec:
    """Return a PathSpec ignoring ``.gitignore`` entries and default patterns."""
    patterns: list[str] = []
    gitignore = root / ".gitignore"
    if gitignore.exists():
        try:
            patterns.extend(gitignore.read_text().splitlines())
        except OSError:
            pass
    patterns.extend(DEFAULT_IGNORE_PATTERNS)
    return PathSpec.from_lines("gitwildmatch", patterns)


def apply_vgrep_ignore() -> None:
    """Apply default ignore patterns to ``vgrep``'s filesystem walker."""
    import vgrep.fs

    vgrep.fs.DEFAULT_IGNORE_PATTERNS = list(DEFAULT_IGNORE_PATTERNS)
