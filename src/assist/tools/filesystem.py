from langchain_core.tools import tool
import os
from datetime import datetime
from pathlib import Path
from pathspec import PathSpec

# Tools for working with the filesystem


@tool
def list_files(root: str) -> list[str]:
    """Recursively list files under ``root`` with creation and modification times.

    Files matching patterns from a ``.gitignore`` file in ``root`` are skipped.
    The results are sorted by last modified date in descending order and each
    entry includes the absolute path, creation date, and last modified date.

    Args:
        root: Directory to search.

    Returns:
        list[str]: ``"<path> (created: <cdate>, modified: <mdate>)"`` entries
        for every file under ``root`` that isn't ignored.
    """
    root_path = Path(root)
    ignore_spec: PathSpec | None = None
    gitignore = root_path / ".gitignore"
    if gitignore.exists():
        try:
            ignore_spec = PathSpec.from_lines("gitwildmatch", gitignore.read_text().splitlines())
        except OSError:
            ignore_spec = None

    files: list[tuple[str, float, float]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = Path(os.path.relpath(dirpath, root_path))
        if str(rel_dir) == ".":
            rel_dir = Path()

        if ignore_spec:
            dirnames[:] = [
                d for d in dirnames
                if not ignore_spec.match_file((rel_dir / d).as_posix())
            ]

        for name in filenames:
            rel_file = (rel_dir / name).as_posix()
            if ignore_spec and ignore_spec.match_file(rel_file):
                continue
            path = Path(dirpath) / name
            try:
                stat = path.stat()
            except OSError:
                continue
            files.append((str(path.resolve()), stat.st_ctime, stat.st_mtime))

    files.sort(key=lambda x: x[2], reverse=True)

    def fmt(ts: float) -> str:
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")

    return [f"{p} (created: {fmt(c)}, modified: {fmt(m)})" for p, c, m in files]


@tool
def file_contents(path: str) -> str:
    """Returns the contents of the file at `path`.

    Args:
        path (str): The path to the desired file.

    Returns:
        str: The content of the specified file as a string.
    """
    with open(path, 'r') as f:
        return f.read()


@tool
def project_context(root: str) -> str:
    """Return the contents of README and AGENTS files under ``root``.

    Searches ``root`` recursively for files whose names begin with ``README`` or
    ``AGENTS`` (case-insensitive) and returns their contents, each preceded by
    the file's path.

    Args:
        root: Directory to search.

    Returns:
        str: Concatenated contents of matching files, each section prefixed with
        the file path.
    """
    paths: list[Path] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            upper = name.upper()
            if upper.startswith("README") or upper.startswith("AGENTS"):
                paths.append(Path(dirpath) / name)

    contents: list[str] = []
    for p in paths:
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        contents.append(f"# {p}\n{text}")

    return "\n\n".join(contents)
