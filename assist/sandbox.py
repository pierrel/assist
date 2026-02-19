"""Docker sandbox backend for isolated code execution.

Implements BaseSandbox from deepagents, providing execute() via Docker
container.exec_run(). All file operations (read, write, edit, grep, glob, ls)
are inherited from BaseSandbox and work via execute().

Paths are prefixed with work_dir (/workspace) so that agent paths like
/myfile.txt map to /workspace/myfile.txt inside the container, which is
where the host bind mount lives.
"""

import logging

from deepagents.backends.sandbox import BaseSandbox
from deepagents.backends.protocol import (
    EditResult,
    ExecuteResponse,
    FileDownloadResponse,
    FileInfo,
    FileUploadResponse,
    GrepMatch,
    WriteResult,
)

logger = logging.getLogger(__name__)

MAX_OUTPUT_CHARS = 100_000


class DockerSandboxBackend(BaseSandbox):
    """Sandbox backend that executes commands inside a Docker container.

    Args:
        container: A running Docker container object (docker.models.containers.Container).
        work_dir: Root directory inside the container where the bind mount lives.
    """

    def __init__(self, container, work_dir: str = "/workspace"):
        self.container = container
        self.work_dir = work_dir.rstrip("/")

    def _resolve(self, path: str | None) -> str | None:
        """Prefix path with work_dir if not already under it."""
        if not path:
            return path
        if path.startswith(self.work_dir):
            return path
        return self.work_dir + (path if path.startswith("/") else "/" + path)

    @property
    def id(self) -> str:
        return self.container.id[:12]

    def execute(self, command: str) -> ExecuteResponse:
        """Execute a shell command inside the Docker container."""
        try:
            exit_code, output_bytes = self.container.exec_run(
                ["bash", "-c", command],
                demux=False,
                workdir=self.work_dir,
            )
        except Exception as e:
            logger.error("Docker exec failed: %s", e)
            return ExecuteResponse(output=f"Error executing command: {e}", exit_code=1)

        output = output_bytes.decode("utf-8", errors="replace") if output_bytes else ""
        truncated = len(output) > MAX_OUTPUT_CHARS
        if truncated:
            output = output[:MAX_OUTPUT_CHARS] + "\n... [output truncated]"

        return ExecuteResponse(
            output=output,
            exit_code=exit_code,
            truncated=truncated,
        )

    # --- File operations with path prefixing ---

    def ls_info(self, path: str) -> list[FileInfo]:
        return super().ls_info(self._resolve(path))

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> str:
        return super().read(self._resolve(file_path), offset, limit)

    def write(self, file_path: str, content: str) -> WriteResult:
        return super().write(self._resolve(file_path), content)

    def edit(self, file_path: str, old_string: str, new_string: str,
             replace_all: bool = False) -> EditResult:
        return super().edit(self._resolve(file_path), old_string, new_string, replace_all)

    def grep_raw(self, pattern: str, path: str | None = None,
                 glob: str | None = None) -> list[GrepMatch] | str:
        return super().grep_raw(pattern, self._resolve(path or "/"), glob)

    def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        return super().glob_info(pattern, self._resolve(path))

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
