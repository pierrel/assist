"""Many-threads persistence management for the dev web app.

APP-SIDE module (see docs/2026-06-11-embedder-contract.org): the
``ThreadManager`` — threads.db ``SqliteSaver``, per-thread directories,
soft/hard delete, retention hooks — is the *web app's* persistence
policy, not part of the embedder contract.  Other clients own their
persistence directly through ``Thread(thread_id=..., checkpointer=...)``
(emacsos-server keeps a fixed-id conversation in its own SqliteSaver).
It lives in the ``assist`` package (rather than ``manage``) because the
eval harness uses it independently of the web app.

Moved verbatim from ``assist.thread`` in the embedder-contract refactor;
behavior unchanged.  Two semantics here are load-bearing and must not
be "improved":

- the sqlite connect happens in ``__init__`` (server *startup*), not
  lazily on first request — a blocking connect must never land on a
  request path;
- the ``model`` property stays lazy under ``_model_lock`` so the web
  server can boot before the LLM endpoint is reachable.
"""

import logging
import os
import shutil
import sqlite3
import subprocess
import tempfile
import threading
from datetime import datetime
from typing import Callable, List

from langgraph.checkpoint.sqlite import SqliteSaver

from assist.model_manager import select_assistant_model
from assist.sandbox_manager import SandboxManager
from assist.spec import AgentSpec
from assist.thread import Thread
from assist.tools import show_file

logger = logging.getLogger(__name__)

# The web app's main agent gets show_file (render a file in the web UI).  Scoped
# here BY DESIGN: ThreadManager is the web app's agent builder (emacsos builds
# its own Thread/spec; the eval harness uses create_agent directly), so the
# web-only show_file tool rides the web-only builder rather than a universal
# default that would also reach surfaces with no web view.  The AgentSpec is
# constructed per-call (in get/new) — its docstring cautions against caching a
# spec as a module constant — though this one closes over no per-request state.
_WEB_TOOLS = (show_file,)


class InvalidThreadId(ValueError):
    """A thread id that isn't a single safe path segment (traversal/separator).

    Raised by ``ThreadManager.thread_dir`` so every tid->path method rejects a
    crafted id by construction; the web layer maps it to 404."""


class ThreadManager:
    """Manage ``Thread`` instances persisted under a directory tree.

    At the root directory, a sqlite DB named 'threads.db' is used for LangGraph
    checkpointing via SqliteSaver.
    """

    DEFAULT_THREAD_WORKING_DIRECTORY = "domain"

    def __init__(self, root_dir: str | None = None):
        if root_dir:
            self.root_dir = root_dir
        else:
            self.root_dir = tempfile.mkdtemp()

        os.makedirs(self.root_dir, exist_ok=True)
        self.db_path = os.path.join(self.root_dir, "threads.db")
        # Ensure DB file exists upfront
        if not os.path.exists(self.db_path):
            open(self.db_path, "a").close()
        # SqliteSaver expects a sqlite3.Connection
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.checkpointer = SqliteSaver(self.conn)
        # Lazily resolve the chat model so the web server can boot before
        # the LLM endpoint is reachable.  First request triggers the probe;
        # the lock prevents two concurrent first-requests from probing twice.
        self._model = None
        self._model_lock = threading.Lock()

    @property
    def model(self):
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    self._model = select_assistant_model(0.1)
        return self._model

    def list(self) -> list[str]:
        """Return thread IDs filtered (no soft-deleted) and sorted by mtime descending."""
        dirs = []
        for name in os.listdir(self.root_dir):
            dpath = os.path.join(self.root_dir, name)
            if not os.path.isdir(dpath) or name == "__pycache__":
                continue
            if os.path.exists(os.path.join(dpath, ".deleted")):
                continue
            dirs.append((name, os.path.getmtime(dpath)))
        dirs.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in dirs]

    def soft_delete(self, thread_id: str) -> None:
        """Mark a thread as deleted by writing a .deleted marker file."""
        tdir = self.thread_dir(thread_id)
        if os.path.isdir(tdir):
            marker = os.path.join(tdir, ".deleted")
            with open(marker, "w") as f:
                f.write(datetime.now().isoformat())

    def hard_delete(
        self,
        tid: str,
        on_delete: List[Callable[[str], None]] | None = None,
    ) -> None:
        """Permanently delete a thread: sandbox container, DB rows, dir.

        Layer 0 of the threads.db growth plan
        (docs/2026-05-04-threads-db-layer-0-thread-retention.org).

        The order of operations is load-bearing.  See the design doc
        "Approach" section for why each step happens before the next.
        Briefly:

        1. ``SandboxManager.cleanup`` first so any in-flight agent run
           hits the existing ``SandboxContainerLostError`` path
           cleanly instead of ENOENT/EIO from a yanked bind mount.
        2. ``checkpointer.delete_thread`` — uses upstream
           SqliteSaver's atomic per-schema DELETE
           (langgraph/checkpoint/sqlite/__init__.py:477-494).
        3. ``shutil.rmtree`` the per-thread directory.  Tolerates
           ``FileNotFoundError`` so re-running on a half-deleted
           thread succeeds (idempotency).
        4. ``on_delete`` callbacks (if any) fire last, each guarded
           by try/except so a misbehaving consumer can't break the
           sweep.  ``manage/web.py`` passes one that evicts the
           in-process domain/description caches; the retention CLI
           passes none.
        """
        tdir = self.thread_dir(tid)
        work_dir = self.thread_default_working_dir(tid)

        # 1. Stop the sandbox container before yanking its bind mount.
        try:
            SandboxManager.cleanup(work_dir)
        except Exception as e:
            logger.warning("Sandbox cleanup failed for %s: %s", tid, e)

        # 2. Delete checkpointer rows via the upstream public API.
        try:
            self.checkpointer.delete_thread(tid)
        except Exception as e:
            logger.warning(
                "checkpointer.delete_thread failed for %s: %s", tid, e
            )

        # 3. Wipe the on-disk directory.  Idempotent: a missing dir is
        # fine — re-running on a half-deleted thread must succeed.
        # On EACCES, fall back to a privileged-rm via a one-shot Docker
        # container.  This path is *legacy-compat*: it covers thread
        # workspaces created before the non-root-sandbox layer
        # (docs/2026-05-08-restrict-git-real-via-non-root-sandbox.org)
        # shipped, which still hold root-owned files in
        # ``domain/references/`` and ``domain/**/__pycache__``.
        # Threads created after that deploy run the sandbox as the
        # invoking user, so files are user-owned and ``shutil.rmtree``
        # succeeds without the alpine fallback.  Once all such
        # legacy threads age out via the retention sweep, this
        # PermissionError branch becomes dead code and can be removed.
        try:
            shutil.rmtree(tdir, ignore_errors=False)
        except FileNotFoundError:
            pass
        except PermissionError:
            parent = os.path.dirname(tdir)
            basename = os.path.basename(tdir)
            try:
                subprocess.run(
                    ["docker", "run", "--rm",
                     "-v", f"{parent}:/work",
                     "alpine", "rm", "-rf", f"/work/{basename}"],
                    capture_output=True, check=True,
                )
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                # If Docker isn't available either, log and continue —
                # the DB rows are gone (step 2), so the thread is
                # invisible to the UI even with the dir lingering.
                # Manual cleanup needed.
                logger.warning(
                    "Privileged rmtree fallback failed for %s: %s. "
                    "DB rows are gone; the working tree at %s remains "
                    "and will need manual cleanup.",
                    tid, e, tdir,
                )

        # 4. Run consumer-supplied callbacks.  Each is isolated so
        # one bad callback can't break others or the sweep.
        if on_delete:
            for cb in on_delete:
                try:
                    cb(tid)
                except Exception as e:
                    logger.warning(
                        "on_delete callback %r failed for %s: %s",
                        cb, tid, e,
                    )

    def touch(self, thread_id: str) -> None:
        """Update mtime of thread dir so it sorts to the top of list()."""
        tdir = self.thread_dir(thread_id)
        if os.path.isdir(tdir):
            os.utime(tdir, None)

    def get(self,
            thread_id: str,
            working_dir: str | None = None,
            sandbox_backend=None,
            on_queue_state: Callable[[str], None] | None = None) -> Thread:
        tdir = self.thread_dir(thread_id)
        if not os.path.isdir(tdir):
            raise FileNotFoundError(f"thread directory not found: {thread_id}, {tdir}")
        if not working_dir:
            working_dir = self.make_default_working_dir(tdir)

        return Thread(working_dir,
                      thread_id=thread_id,
                      checkpointer=self.checkpointer,
                      model=self.model,
                      sandbox_backend=sandbox_backend,
                      on_queue_state=on_queue_state,
                      spec=AgentSpec(tools=_WEB_TOOLS))

    def remove(self, thread_id: str) -> None:
        tdir = self.thread_dir(thread_id)
        if os.path.isdir(tdir):
            # Best-effort delete
            for root, dirs, files in os.walk(tdir, topdown=False):
                for f in files:
                    try:
                        os.remove(os.path.join(root, f))
                    except Exception:
                        pass
                for d in dirs:
                    try:
                        os.rmdir(os.path.join(root, d))
                    except Exception:
                        pass
            try:
                os.rmdir(tdir)
            except Exception:
                pass

    def new(self, working_dir: str|None = None, sandbox_backend=None,
            on_queue_state: Callable[[str], None] | None = None) -> Thread:
        # Derive a clean ID for directory: prefer timestamp+rand
        tid = datetime.now().strftime("%Y%m%d%H%M%S") + "-" + os.urandom(4).hex()
        tdir = os.path.join(self.root_dir, tid)
        os.makedirs(tdir, exist_ok=True)
        if not working_dir:
            working_dir = self.make_default_working_dir(tdir)

        return Thread(working_dir, thread_id=tid, checkpointer=self.checkpointer,
                      model=self.model, sandbox_backend=sandbox_backend,
                      on_queue_state=on_queue_state, spec=AgentSpec(tools=_WEB_TOOLS))

    def close(self) -> None:
        try:
            if hasattr(self, "conn") and self.conn:
                self.conn.close()
        except Exception:
            pass

    def thread_dir(self, tid: str) -> str:
        # Validate by construction: a thread id is a single path segment, so a
        # crafted id ("..", "a/b", a NUL) can't escape root_dir in any caller's
        # filesystem op. Every tid->path method below routes through here.
        if tid in ("", ".", "..") or "/" in tid or "\\" in tid or "\0" in tid:
            raise InvalidThreadId(f"invalid thread id: {tid!r}")
        return os.path.join(self.root_dir, tid)

    def thread_default_working_dir(self, tid: str) -> str:
        return os.path.join(self.thread_dir(tid),
                            self.DEFAULT_THREAD_WORKING_DIRECTORY)

    def make_default_working_dir(self, tdir: str) -> str:
        wdir = self.DEFAULT_THREAD_WORKING_DIRECTORY
        working_dir = os.path.join(tdir, wdir)
        os.makedirs(working_dir, exist_ok=True)

        return working_dir

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
