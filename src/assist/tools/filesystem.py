from langchain_core.tools import tool
import os
from datetime import datetime
from pathlib import Path
from pathspec import PathSpec

from assist import git
from assist.tools.safeguard import in_server_project, ensure_outside_server

# Tools for working with the filesystem


@tool
def list_files(root: str) -> list[str]:
    """Recursively list files under ``root`` with creation and modification times.

    Files matching patterns from a ``.gitignore`` file in ``root`` are skipped.
    The results are sorted by last modified date in descending order and each
    entry includes the absolute path, creation date, and last modified date.
    Only the 200 most recently modified files are returned. If more than 200
    files are found, the final entry will be the string ``"Limit of 200 files"
    ``"reached"`` to indicate truncation.

    Args:
        root: Directory to search.

    Returns:
        list[str]: ``"<path> (created: <cdate>, modified: <mdate>)"``
    """
    root_path = Path(root)
    if in_server_project(root_path):
        return ["Access to server project files is not allowed"]
    ignore_spec: PathSpec | None = None
    gitignore = root_path / ".gitignore"
    if gitignore.exists():
        try:
            ignore_spec = PathSpec.from_lines("gitwildmatch", gitignore.read_text().splitlines())
        except OSError:
            ignore_spec = None

    files: list[tuple[str, float, float]] = []
    for dirpath, dirnames, filenames in os.walk(root_path):
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

    result = [f"{p} (created: {fmt(c)}, modified: {fmt(m)})" for p, c, m in files[:200]]
    if len(files) > 200:
        result.append("Limit of 200 files reached")
    return result


@tool
def file_contents(path: str) -> str:
    """Returns the contents of the file at `path`.

    Args:
        path (str): The path to the desired file.

    Returns:
        str: The content of the specified file as a string.
    """
    p = Path(path)
    if in_server_project(p):
        return "Access to server project files is not allowed"
    with open(p, 'r') as f:
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
    root_path = Path(root)
    if in_server_project(root_path):
        return "Access to server project files is not allowed"
    paths: list[Path] = []
    for dirpath, _dirnames, filenames in os.walk(root_path):
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
@tool
def write_file(
    path: str,
    content: str,
    overwrite: bool = False,
    append: bool = False,
) -> str:
    """Write ``content`` to ``path`` ensuring repository safety.

    The parent directory of ``path`` must be within a Git repository. If the
    file already exists it must be tracked by Git. Existing files are not
    modified unless ``overwrite`` or ``append`` is set.

    Args:
        path: Destination file path.
        content: Text to write.
        overwrite: Replace the file if it already exists.
        append: Append to the file if it already exists.

    Returns:
        str: A status message describing the action taken.
    """

    p = Path(path).expanduser().resolve()
    ensure_outside_server(p)
    parent = p.parent

    git.repo_root(parent)

    if p.exists():
        if not git.is_tracked(p):
            raise ValueError("File exists but is not tracked by Git")
        if not (overwrite or append):
            raise ValueError("File exists; set overwrite=True or append=True to modify")
        mode = "a" if append else "w"
    else:
        if not parent.exists():
            raise ValueError("Parent directory does not exist")
        mode = "w"

    with open(p, mode, encoding="utf-8") as f:
        f.write(content)

    return f"Wrote {p}"

