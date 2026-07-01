"""Host-maintained read-only mirror of a domain's origin, for the sandboxed agent.

The agent (in the container) must be able to fetch ``main`` to rebase, but must NEVER be
able to push and must not depend on the real origin being reachable from the egress
network (local-path domains like "life" are not). So the HOST keeps one bare mirror per
domain under ``<root>/.mirrors/<label>.git`` (host has the creds/reachability), and
bind-mounts it **read-only** at ``/srv/domain.git`` in every container of that domain.
The clone gets a second remote ``mirror = file:///srv/domain.git`` and the agent fetches
``mirror/main``. Read-only by kernel construction (the ro mount) — a push barrier no
credential or shim bypass can defeat.

This is Resource Access: it encapsulates *where/how the real origin lives and is reached*
(a local bare path today, Forgejo tomorrow) so the container never learns it.
"""
from __future__ import annotations

import os
import re
import subprocess
import threading

# The container-side path (ro bind-mount target) + the remote the agent fetches.
CONTAINER_MOUNT = "/srv/domain.git"
MIRROR_REMOTE = "mirror"
CONTAINER_MIRROR_URL = f"file://{CONTAINER_MOUNT}"

# One lock per mirror path so two same-domain turns starting together don't race the
# fetch (a half-updated bare repo). Keyed by absolute path.
_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _lock_for(path: str) -> threading.Lock:
    with _locks_guard:
        return _locks.setdefault(path, threading.Lock())


def _safe_label(label: str) -> str:
    """A domain label reduced to a safe single path segment for the mirror dir."""
    slug = re.sub(r"[^A-Za-z0-9._-]", "-", label).strip("-.") or "domain"
    return slug


class DomainMirror:
    """A bare mirror of one domain's origin, refreshed host-side."""

    def __init__(self, root_dir: str, repo_url: str, label: str):
        self._repo = repo_url
        self._path = os.path.join(root_dir, ".mirrors", f"{_safe_label(label)}.git")

    @property
    def path(self) -> str:
        """Host path to bind-mount (ro) at ``CONTAINER_MOUNT``."""
        return self._path

    def refresh(self) -> None:
        """Create the bare mirror if missing, else fetch origin — under the per-mirror
        lock. Raises ``CalledProcessError`` LOUDLY on failure (e.g. a moved local repo):
        that surfaces host-side rather than as a silent container mystery, and the
        last-good mirror stays in place for the container to keep using."""
        with _lock_for(self._path):
            if not os.path.isdir(self._path):
                os.makedirs(os.path.dirname(self._path), exist_ok=True)
                subprocess.run(["git", "clone", "--mirror", self._repo, self._path],
                               check=True)
            else:
                subprocess.run(["git", "-C", self._path, "fetch", "--prune", "origin"],
                               check=True)
