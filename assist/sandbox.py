"""Docker sandbox backend for isolated code execution.

Implements BaseSandbox from deepagents, providing execute() via Docker
container.exec_run(). All file operations (read, write, edit, grep, glob, ls)
are inherited from BaseSandbox and work via execute().
"""

import logging

from deepagents.backends.sandbox import BaseSandbox
from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
)

logger = logging.getLogger(__name__)

MAX_OUTPUT_CHARS = 100_000


class DockerSandboxBackend(BaseSandbox):
    """Sandbox backend that executes commands inside a Docker container.

    Args:
        container: A running Docker container object (docker.models.containers.Container).
    """

    def __init__(self, container):
        self.container = container

    @property
    def id(self) -> str:
        return self.container.id[:12]

    def execute(self, command: str) -> ExecuteResponse:
        """Execute a shell command inside the Docker container."""
        try:
            exit_code, output_bytes = self.container.exec_run(
                ["bash", "-c", command],
                demux=False,
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

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload files to the container via tar archive."""
        import io
        import tarfile

        responses = []
        for path, content in files:
            try:
                tar_stream = io.BytesIO()
                with tarfile.open(fileobj=tar_stream, mode="w") as tar:
                    info = tarfile.TarInfo(name=path.lstrip("/"))
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
            try:
                bits, _ = self.container.get_archive(path)
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
