from __future__ import annotations

"""Utilities to prevent tools from modifying the running server's source tree."""

import os
from pathlib import Path


def server_root() -> Path | None:
    """Return the project root for the running server if available."""
    root = os.getenv("ASSIST_SERVER_PROJECT_ROOT")
    return Path(root).resolve() if root else None


def in_server_project(path: Path | str) -> bool:
    """True if ``path`` is inside the server's own project directory."""
    root = server_root()
    if root is None:
        return False
    try:
        p = Path(path).resolve()
    except FileNotFoundError:
        return False
    return root == p or root in p.parents


def ensure_outside_server(path: Path | str) -> None:
    """Raise ``PermissionError`` if ``path`` is in the server project."""
    if in_server_project(path):
        raise PermissionError("Access to the server project is not allowed")


__all__ = ["in_server_project", "ensure_outside_server", "server_root"]
