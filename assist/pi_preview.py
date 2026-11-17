"""Server-owned availability policy for bounded native previews."""
from __future__ import annotations

import json
import os
import stat
import tempfile
from pathlib import Path

from assist.thread_engine import EngineName


_STATE_FILE = ".pi-preview.json"


class PiPreviewUnavailable(RuntimeError):
    """A native preview operator setting cannot be safely changed."""


class NativePreviewPolicy:
    """Persist one native engine's operator setting and default it to disabled.

    This class owns no worker, Run, or cancellation lifecycle. Dispatch and
    provider admission check it independently of client rendering.
    """

    def __init__(self, root_dir: str | Path, *, engine: str = "pi"):
        if engine not in {"pi", "hermes"}:
            raise ValueError("Unknown native preview")
        self.root = Path(root_dir)
        self.engine = engine

    @property
    def path(self) -> Path:
        return self.root / (_STATE_FILE if self.engine == "pi" else ".hermes-preview.json")

    def _root_is_safe(self) -> bool:
        try:
            metadata = self.root.lstat()
        except OSError:
            return False
        return (stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode)
                and metadata.st_uid == os.getuid() and not metadata.st_mode & 0o022)

    def _ensure_safe_root(self) -> None:
        try:
            self.root.mkdir(mode=0o700)
        except FileExistsError:
            pass
        if not self._root_is_safe():
            raise PiPreviewUnavailable("Native preview state directory is unavailable or unsafe.")

    def enabled(self) -> bool:
        """Return the persisted operator setting; missing or malformed is off."""
        if not self._root_is_safe():
            return False
        try:
            descriptor = os.open(
                self.path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK)
        except FileNotFoundError:
            return False
        except OSError:
            return False
        try:
            metadata = os.fstat(descriptor)
            if (not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid()
                    or metadata.st_mode & 0o022 or metadata.st_size > 1024):
                return False
            data = os.read(descriptor, 1025)
            if len(data) > 1024:
                return False
            return json.loads(data.decode("utf-8")) == {"version": 1, "enabled": True}
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return False
        finally:
            os.close(descriptor)

    def set_enabled(self, enabled: bool) -> None:
        """Durably set the local operator switch, defaulting to disabled."""
        if type(enabled) is not bool:
            raise TypeError("Native preview enabled must be a bool")
        self._ensure_safe_root()
        temporary = tempfile.NamedTemporaryFile(
            mode="wb", dir=self.root, prefix=f".{self.engine}-preview-", delete=False)
        try:
            with temporary:
                os.fchmod(temporary.fileno(), 0o600)
                temporary.write(json.dumps(
                    {"version": 1, "enabled": enabled}, sort_keys=True,
                    separators=(",", ":")).encode())
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary.name, self.path)
            directory = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if os.path.lexists(temporary.name):
                os.unlink(temporary.name)

    def admits(self, engine: EngineName) -> bool:
        """Return whether the persisted operator setting permits an engine."""
        if engine == "deepagents":
            return True
        return engine == self.engine and self.enabled()


# Keep the operator-setting reader importable for retained Pi state and tools.
PiPreviewPolicy = NativePreviewPolicy
