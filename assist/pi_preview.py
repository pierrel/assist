"""Server-owned availability policy for the bounded Pi web preview."""
from __future__ import annotations

import json
import os
import stat
import tempfile
from pathlib import Path

from assist.thread_engine import EngineName


_STATE_FILE = ".pi-preview.json"


class PiPreviewUnavailable(RuntimeError):
    """The Pi preview operator setting cannot be safely changed."""


class PiPreviewPolicy:
    """Persist the operator setting and admit Pi only when it is enabled.

    This class deliberately owns no worker, Run, or cancellation lifecycle.  The
    later runtime manager will call it while holding its own transition gate.
    """

    def __init__(self, root_dir: str | Path):
        self.root = Path(root_dir)

    @property
    def path(self) -> Path:
        return self.root / _STATE_FILE

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
            raise PiPreviewUnavailable("Pi preview state directory is unavailable or unsafe.")

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
            raise TypeError("Pi preview enabled must be a bool")
        self._ensure_safe_root()
        temporary = tempfile.NamedTemporaryFile(
            mode="wb", dir=self.root, prefix=".pi-preview-", delete=False)
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
        return engine == "pi" and self.enabled()
