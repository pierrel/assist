from langchain_core.tools import tool
import os
from datetime import datetime
from pathlib import Path
from pathspec import PathSpec

# Tools for working with the filesystem


def _gitignore_spec(root_path: Path) -> tuple[PathSpec, Path] | None:
    """Return a ``PathSpec`` and its base directory for ``root_path``.

    ``root_path`` may be any directory within the project tree.  This function
    searches upwards from ``root_path`` until a ``.gitignore`` file is found and
    returns the parsed ``PathSpec`` along with the directory containing the
    ``.gitignore``.  ``None`` is returned if no ``.gitignore`` is found or the
    file cannot be read.
    """
    current = root_path
    while True:
        gitignore = current / ".gitignore"
        if gitignore.exists():
            try:
                spec = PathSpec.from_lines("gitwildmatch", gitignore.read_text().splitlines())
            except OSError:
                return None
            return spec, current
        if current.parent == current:
            break
        current = current.parent
    return None


def _is_ignored(rel_path: Path, root: Path, ignore_spec: tuple[PathSpec, Path] | None) -> bool:
    """Return ``True`` if ``rel_path`` under ``root`` matches ``ignore_spec``."""
    if not ignore_spec:
        return False
    spec, base = ignore_spec
    abs_path = (root / rel_path).resolve()
    try:
        rel_to_base = abs_path.relative_to(base)
    except ValueError:
        return False
    return spec.match_file(rel_to_base.as_posix())


@tool
def list_files(root: str) -> tuple[list[str], str | None]:
    """List up to 200 files and directories under ``root``.

    Files matching patterns from a ``.gitignore`` file in ``root`` or any
    ancestor directory are skipped.
    Only entries within four levels below ``root`` are considered.  The results
    are sorted by last modified date in descending order and each entry includes
    the absolute path, creation date, and last modified date.

    Args:
        root: Directory to search.

    Returns:
        tuple[list[str], str | None]: ``["<path> (created: <cdate>, modified: <mdate>)"]``
        entries for files and directories.  If more than 200 entries are found,
        only the first 200 are returned along with a message indicating that the
        results were truncated.
    """
    root_path = Path(root)
    ignore_spec = _gitignore_spec(root_path)

    entries: list[tuple[str, float, float]] = []
    for dirpath, dirnames, filenames in os.walk(root_path):
        rel_dir = Path(os.path.relpath(dirpath, root_path))
        if str(rel_dir) == ".":
            rel_dir = Path()

        depth = len(rel_dir.parts)
        if depth >= 4:
            dirnames[:] = []

        if ignore_spec:
            dirnames[:] = [d for d in dirnames if not _is_ignored(rel_dir / d, root_path, ignore_spec)]

        if depth > 0 and not _is_ignored(rel_dir, root_path, ignore_spec):
            path = Path(dirpath)
            try:
                stat = path.stat()
            except OSError:
                pass
            else:
                entries.append((str(path.resolve()), stat.st_ctime, stat.st_mtime))

        for name in filenames:
            rel_file = rel_dir / name
            if _is_ignored(rel_file, root_path, ignore_spec):
                continue
            path = Path(dirpath) / name
            try:
                stat = path.stat()
            except OSError:
                continue
            entries.append((str(path.resolve()), stat.st_ctime, stat.st_mtime))

    entries.sort(key=lambda x: x[2], reverse=True)

    def fmt(ts: float) -> str:
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")

    formatted = [f"{p} (created: {fmt(c)}, modified: {fmt(m)})" for p, c, m in entries]
    if len(formatted) > 200:
        return formatted[:200], "Over 200 files found, only returned the first 200 files"
    return formatted, None


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
