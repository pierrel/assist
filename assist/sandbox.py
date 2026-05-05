"""Docker sandbox backend for isolated code execution.

Implements BaseSandbox from deepagents, providing execute() via Docker
container.exec_run(). All file operations (read, write, edit, grep, glob, ls)
are inherited from BaseSandbox and work via execute().

Paths are prefixed with work_dir (/workspace) so that agent paths like
/myfile.txt map to /workspace/myfile.txt inside the container, which is
where the host bind mount lives.
"""

import logging
import os
import shlex

from deepagents.backends.sandbox import BaseSandbox
from deepagents.backends.protocol import (
    EditResult,
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
    GlobResult,
    GrepResult,
    LsResult,
    WriteResult,
)
from docker.errors import NotFound as DockerNotFound

logger = logging.getLogger(__name__)

MAX_OUTPUT_CHARS = 100_000

# Wall-clock cap on a single sandbox command. Without this, a runaway
# subprocess (e.g. small-model agent firing `glob("**/test*", cwd="/")`)
# can spin at 99% CPU indefinitely while the agent loop blocks waiting
# for output. Empirically observed 2026-05-03 — see
# docs/2026-05-03-execute-tool-timeout.md.
#
# 600s matches the per-test pytest-timeout we use for evals. Anything
# legitimately longer (a full suite run inside the sandbox, an unusually
# slow `pip install`) can override via the env var.
EXEC_TIMEOUT_SECONDS = int(os.getenv("ASSIST_SANDBOX_EXEC_TIMEOUT", "600"))

# Coreutils `timeout` first sends SIGTERM, then SIGKILL after a grace
# period. 5s gives a well-behaved process room to flush; an unkillable
# busy loop hits SIGKILL and surfaces as exit 137.
EXEC_KILL_GRACE_SECONDS = 5


_TIMEOUT_GUIDANCE = (
    "[Sandbox terminated this command after {timeout}s wall-clock limit.]\n"
    "Try one of these adjustments and retry:\n"
    "  - Narrow the scope: target a specific file or directory rather "
    "than walking the whole tree (e.g. `grep PATTERN /workspace/specific/file` "
    "instead of `grep -r PATTERN /workspace`).\n"
    "  - Pass a tighter root: the workspace at `/workspace` may hold "
    "thousands of files (cloned repos, bundled PDFs, generated dirs). "
    "Pass a specific subdirectory (e.g. `/workspace/notes/`) so you skip "
    "trees that aren't relevant to the task.\n"
    "  - Bound recursion: `find ... -maxdepth N`, `glob` with a tighter "
    "pattern, or limit lines with `| head -N`.\n"
    "  - If the command is genuinely long-running (a full pip install or "
    "test suite), break it into smaller invocations.\n"
    "Partial output (may be empty if the command was silent):\n"
    "----\n"
)


class SandboxContainerLostError(RuntimeError):
    """The Docker sandbox container disappeared mid-thread.

    Raised from ``DockerSandboxBackend.execute`` when ``container.exec_run``
    sees a 404 from the Docker API — meaning the container was stopped or
    removed (manually, by an OOM kill, by a daemon restart, by the
    container's own 1h ``sleep 3600`` TTL expiring) while the agent was
    still running.

    Distinct from a regular per-command exec failure: a dead container
    can never recover, so every subsequent tool call would also fail.
    Returning the 404 as an ``ExecuteResponse`` (the way other exec
    errors are returned) lets the model see a tool result it can ignore
    — empirically Qwen3.6 happily replies "Done. Created X" after
    repeated 404s, hallucinating the work it never did.

    The agent loop should NOT catch this; it should bubble up to the
    web layer's per-request exception handler so the thread is marked
    errored with a clear message.
    """


class DockerSandboxBackend(BaseSandbox):
    """Sandbox backend that executes commands inside a Docker container.

    Args:
        container: A running Docker container object (docker.models.containers.Container).
        work_dir: Root directory inside the container where the bind mount lives.
    """

    def __init__(self, container, work_dir: str = "/workspace",
                 strip_prefixes: tuple[str, ...] = ()):
        """Args:
            container: A running Docker container object.
            work_dir: Root directory inside the container that paths are
                resolved against.
            strip_prefixes: Optional tuple of leading path prefixes to
                strip *before* resolution.  Used by the research-agent's
                references-confined sibling so the agent writing to
                ``references/foo.org`` doesn't nest under an already-
                references-rooted ``work_dir``.  Each prefix is checked
                in both bare (``"references/"``) and absolute
                (``"/references/"``) form.
        """
        self.container = container
        self.work_dir = work_dir.rstrip("/")
        self._strip_prefixes = strip_prefixes

    def _strip(self, path: str | None) -> str | None:
        if not path or not self._strip_prefixes:
            return path
        for prefix in self._strip_prefixes:
            for variant in (f"/{prefix.strip('/')}/", f"{prefix.strip('/')}/"):
                if path.startswith(variant):
                    stripped = path[len(variant):]
                    return "/" + stripped if path.startswith("/") else stripped
        return path

    def _resolve(self, path: str | None) -> str | None:
        """Prefix path with work_dir if not already under it.

        Applies ``strip_prefixes`` first so an agent's accidental
        ``references/foo.org`` (when the work_dir is already
        ``/workspace/references``) gets flattened to ``foo.org`` before
        the work_dir prefix is added.
        """
        path = self._strip(path)
        if not path:
            return path
        if path.startswith(self.work_dir):
            return path
        return self.work_dir + (path if path.startswith("/") else "/" + path)

    @property
    def id(self) -> str:
        return self.container.id[:12]

    def execute(self, command: str) -> ExecuteResponse:
        """Execute a shell command inside the Docker container.

        The command is wrapped in coreutils `timeout` so that a runaway
        subprocess can't pin the agent loop indefinitely.  On wall-clock
        timeout the response prepends concrete adjustment guidance to the
        partial output so the model can recover and try a different
        approach instead of just seeing a generic "Error".
        """
        bounded = (
            f"timeout --kill-after={EXEC_KILL_GRACE_SECONDS}s "
            f"{EXEC_TIMEOUT_SECONDS}s bash -c {shlex.quote(command)}"
        )
        try:
            exit_code, output_bytes = self.container.exec_run(
                ["bash", "-c", bounded],
                demux=False,
                workdir=self.work_dir,
            )
        except DockerNotFound as e:
            # Container is gone (stopped, removed, daemon-restarted, TTL
            # expired).  Don't return this as a tool result — see the
            # SandboxContainerLostError docstring for why.
            logger.error(
                "Sandbox container %s no longer exists; failing thread: %s",
                self.container.id[:12], e,
            )
            raise SandboxContainerLostError(
                f"Sandbox container {self.container.id[:12]} disappeared "
                "mid-thread — please retry."
            ) from e
        except Exception as e:
            logger.error("Docker exec failed: %s", e)
            return ExecuteResponse(output=f"Error executing command: {e}", exit_code=1)

        output = output_bytes.decode("utf-8", errors="replace") if output_bytes else ""

        # 124 = `timeout` fired and SIGTERM'd; 137 = SIGKILL'd after the
        # grace window because the subprocess didn't honor SIGTERM.  Both
        # mean "we hit the wall-clock cap" from the agent's perspective.
        if exit_code in (124, 137):
            logger.warning(
                "Sandbox command timed out (exit %d) after %ds: %r",
                exit_code, EXEC_TIMEOUT_SECONDS, command[:200],
            )
            output = _TIMEOUT_GUIDANCE.format(timeout=EXEC_TIMEOUT_SECONDS) + (
                output if output else "(no output)"
            )

        truncated = len(output) > MAX_OUTPUT_CHARS
        if truncated:
            output = output[:MAX_OUTPUT_CHARS] + "\n... [output truncated]"

        return ExecuteResponse(
            output=output,
            exit_code=exit_code,
            truncated=truncated,
        )

    # --- File operations with path prefixing ---
    #
    # IMPORTANT: deepagents' SandboxBackendProtocol uses the methods
    # named ``ls``, ``grep``, ``glob`` (returning structured Result
    # types).  The sibling ``ls_info`` / ``grep_raw`` / ``glob_info``
    # entry points in ``protocol.py`` are deprecated for v0.7 removal
    # and warn on call.  Overriding the deprecated names — as we did
    # before 2026-05-04 — left ``_resolve`` as dead code: every glob
    # call from the agent went straight through ``BaseSandbox.glob``
    # with no ``/workspace`` prefixing, so ``glob(path="/")`` walked
    # the entire container filesystem (Python venv, system dirs) and
    # ran the host CPU into the floor for minutes.  Override the new
    # names so ``_resolve`` actually fires.

    def ls(self, path: str) -> LsResult:
        return super().ls(self._resolve(path))

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> str:
        return super().read(self._resolve(file_path), offset, limit)

    def write(self, file_path: str, content: str) -> WriteResult:
        return super().write(self._resolve(file_path), content)

    def edit(self, file_path: str, old_string: str, new_string: str,
             replace_all: bool = False) -> EditResult:
        return super().edit(self._resolve(file_path), old_string, new_string, replace_all)

    def grep(self, pattern: str, path: str | None = None,
             glob: str | None = None) -> GrepResult:
        return super().grep(pattern, self._resolve(path or "/"), glob)

    def glob(self, pattern: str, path: str = "/") -> GlobResult:
        return super().glob(pattern, self._resolve(path))

    # --- File transfer with path prefixing ---

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload files to the container via tar archive."""
        import io
        import tarfile

        responses = []
        for path, content in files:
            resolved = self._resolve(path)
            try:
                tar_stream = io.BytesIO()
                with tarfile.open(fileobj=tar_stream, mode="w") as tar:
                    info = tarfile.TarInfo(name=resolved.lstrip("/"))
                    info.size = len(content)
                    tar.addfile(info, io.BytesIO(content))
                tar_stream.seek(0)
                self.container.put_archive("/", tar_stream)
                responses.append(FileUploadResponse(path=path))
            except Exception as e:
                logger.error("Upload failed for %s: %s", path, e)
                responses.append(FileUploadResponse(path=path, error="permission_denied"))
        return responses

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download files from the container via tar archive."""
        import io
        import tarfile

        responses = []
        for path in paths:
            resolved = self._resolve(path)
            try:
                bits, _ = self.container.get_archive(resolved)
                tar_stream = io.BytesIO()
                for chunk in bits:
                    tar_stream.write(chunk)
                tar_stream.seek(0)
                with tarfile.open(fileobj=tar_stream, mode="r") as tar:
                    member = tar.getmembers()[0]
                    f = tar.extractfile(member)
                    if f is None:
                        responses.append(FileDownloadResponse(
                            path=path, error="is_directory"))
                        continue
                    content = f.read()
                responses.append(FileDownloadResponse(path=path, content=content))
            except Exception as e:
                logger.error("Download failed for %s: %s", path, e)
                responses.append(FileDownloadResponse(path=path, error="file_not_found"))
        return responses
