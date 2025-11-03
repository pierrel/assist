from langchain_core.tools import tool
import os
import tempfile
from datetime import datetime
from pathlib import Path
from assist import git
from assist.tools.safeguard import in_server_project, ensure_outside_server
from .ignore import load_ignore_spec

from assist.study_agent import study_file

# Tools for working with the filesystem


def _atomic_write(path: Path, content: str) -> None:
    fd, tmp_path = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp:
            tmp.write(content)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


@tool
def list_files(root: str) -> list[str]:
    """Recursively list files under ``root`` with creation and modification times.

    Results are sorted by last modified date in descending order and each entry
    includes the absolute path, creation date, and last modified date. Only the
    200 most recently modified files are returned. If more than 200 files are
    found, the final entry will be the string ``"Limit of 200 files"`` ``"reached"``
    to indicate truncation.

    Args:
        root: Directory to search.

    Returns:
        list[str]: ``"<path> (created: <cdate>, modified: <mdate>)"``
    """
    root_path = Path(root)
    if in_server_project(root_path):
        return ["Access to server project files is not allowed"]
    ignore_spec = load_ignore_spec(root_path)

    files: list[tuple[str, float, float]] = []
    for dirpath, dirnames, filenames in os.walk(root_path):
        rel_dir = Path(os.path.relpath(dirpath, root_path))
        if str(rel_dir) == ".":
            rel_dir = Path()

        dirnames[:] = [
            d for d in dirnames
            if not ignore_spec.match_file((rel_dir / d).as_posix())
        ]

        for name in filenames:
            rel_file = (rel_dir / name).as_posix()
            if ignore_spec.match_file(rel_file):
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
def read_file(absolute_path: str, task: str = "", request: str = "") -> str:
    """Reads or summarizes the contents of an absolute file path.

    when_to_use:
    - Inspect the contents of a single local file.
    - Summarize a large file for a specific task.
    when_not_to_use:
    - Only a relative path is available.
    - Need to write or modify files.
    args_schema:
    - absolute_path (str): Absolute path to the file, e.g. "/tmp/example.txt".
    - task (str): Description of the current execution step.
    - request (str): Original user request or broader context.
    preconditions_permissions:
    - Path must be absolute and outside the server project.
    side_effects:
    - Reads file from disk; idempotent: true; retry_safe: true.
    cost_latency: "~1-100ms; free"
    pagination_cursors:
    - input_cursor: none
    - next_cursor: none
    errors:
    - relative_path: Only absolute file paths are allowed.
    returns:
    - content (str): File text or a summary suitable for the task.
    examples:
    - input: {"absolute_path": "/tmp/foo.txt"}
      output: "Hello"
    """
    p = Path(absolute_path)
    if not p.is_absolute():
        raise ValueError("Only absolute file paths are allowed")
    if in_server_project(p):
        return "Access to server project files is not allowed"
    return study_file(p, task=task, request=request)


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
    ignore_spec = load_ignore_spec(root_path)
    paths: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root_path):
        rel_dir = Path(os.path.relpath(dirpath, root_path))
        if str(rel_dir) == ".":
            rel_dir = Path()
        dirnames[:] = [
            d for d in dirnames
            if not ignore_spec.match_file((rel_dir / d).as_posix())
        ]
        for name in filenames:
            rel_file = (rel_dir / name).as_posix()
            if ignore_spec.match_file(rel_file):
                continue
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
def write_file_user(
    path: str,
    content: str,
    overwrite: bool = False,
    append: bool = False,
) -> str:
    """Write ``content`` to ``path`` for user-visible results.

    Use this tool only when the user explicitly requests a file to be written.
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
        if append:
            with open(p, "a", encoding="utf-8") as f:
                f.write(content)
        else:
            _atomic_write(p, content)
    else:
        if not parent.exists():
            raise ValueError("Parent directory does not exist")
        _atomic_write(p, content)

    return f"Wrote {p}"


@tool
def write_file_tmp(
    path: str,
    content: str,
    overwrite: bool = False,
    append: bool = False,
) -> str:
    """Write ``content`` to ``path`` for temporary internal use.

    Intended for files that are not shown to the user, such as intermediate
    data stored between steps. If ``path`` is not already located within a
    temporary directory, a new unique temporary directory is created and the
    file is written relative to that directory. Unlike
    :func:`write_file_user`, this tool does not require the target to be part of
    a Git repository.

    Args:
        path: Destination file path. When not within a temporary directory the
            file will be created inside a new temporary directory unique to this
            request.
        content: Text to write.
        overwrite: Replace the file if it already exists.
        append: Append to the file if it already exists.

    Returns:
        str: A status message describing the action taken and the full path to
        the file.
    """

    given = Path(path).expanduser()

    def _in_temp_dir(p: Path) -> bool:
        temp_root = Path(tempfile.gettempdir()).resolve()
        try:
            p.resolve().relative_to(temp_root)
            return True
        except ValueError:
            return False

    if not _in_temp_dir(given):
        temp_dir = Path(tempfile.mkdtemp(prefix="assist_tmp_"))
        relative = given.relative_to(given.anchor) if given.is_absolute() else given
        p = temp_dir / relative
    else:
        p = given

    p = p.resolve()
    ensure_outside_server(p)
    parent = p.parent
    parent.mkdir(parents=True, exist_ok=True)

    if p.exists():
        if not (overwrite or append):
            raise ValueError("File exists; set overwrite=True or append=True to modify")
        if append:
            with open(p, "a", encoding="utf-8") as f:
                f.write(content)
        else:
            _atomic_write(p, content)
    else:
        _atomic_write(p, content)

    return f"Wrote {p}"

