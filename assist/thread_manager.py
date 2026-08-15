"""Many-threads persistence management for the dev web app.

APP-SIDE module (see docs/2026-06-11-embedder-contract.org): the
``ThreadManager`` — threads.db ``SqliteSaver``, per-thread directories,
soft/hard delete, retention hooks — is the *web app's* persistence
policy, not part of the embedder contract.  Other clients own their
persistence directly through ``Thread(thread_id=..., checkpointer=...)``
(emacsos-server keeps a fixed-id conversation in its own SqliteSaver).
It lives in the ``assist`` package (rather than ``manage``) because the
eval harness uses it independently of the web app.

Two semantics here are load-bearing and must not be "improved":

- the sqlite connect happens in ``__init__`` (server *startup*), not
  lazily on first request — a blocking connect must never land on a
  request path;
- the ``model`` property stays lazy under ``_model_lock`` so the web
  server can boot before the LLM endpoint is reachable.
"""

import logging
import json
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
from assist.thread_engine import (
    EngineName, ThreadEngine, ThreadEngineError, read_thread_engine,
    write_new_thread_engine)
from assist.async_subagents import async_task_tools
from assist.agent import create_context_agent, create_research_agent

logger = logging.getLogger(__name__)

# The web app's main agent gets web-only skills: ``render`` emits a workspace-file
# block for the web view (parsed by manage/web/threads.py), and ``send-email`` uses
# the web host's approval transport.  Scoped here BY DESIGN: ThreadManager is the
# web app's agent builder (emacsos builds its own Thread/spec; the eval harness uses
# create_agent directly), so these web-only skills never reach surfaces with no web
# view.  The eval harness may mount the route explicitly for render coverage.  The
# route keeps its historical render-skill name.
_RENDER_SKILL_ROUTE = "/render-skill/"
_RENDER_SKILLS_DIR = os.path.join(os.path.dirname(__file__), "web_skills")
_render_skill_sources = None
_triage_skill_sources_cache = None


def _web_skill_sources() -> dict:
    """Route -> backend for the web AgentSpec's web-only skills.

    The route keeps its historical ``render-skill`` name although it serves both
    the render and send-email skills.  Build it lazily so bundled-backend
    construction and deepagents' transitive imports stay off module load (same
    pattern as emacsos-server's ``_skill_sources``).
    """
    global _render_skill_sources
    if _render_skill_sources is None:
        from assist.backends import create_bundled_skills_backend
        _render_skill_sources = {
            _RENDER_SKILL_ROUTE: create_bundled_skills_backend(_RENDER_SKILLS_DIR)
        }
    return _render_skill_sources


def _triage_skill_sources() -> dict:
    """The unchanged pre-P2b skill backends used by inbound SMS triage."""
    global _triage_skill_sources_cache
    if _triage_skill_sources_cache is None:
        from deepagents.backends import FilesystemBackend
        from assist.backends import (
            SKILLS_DIR, SKILLS_ROUTE, create_legacy_skills_backend)
        _triage_skill_sources_cache = {
            SKILLS_ROUTE: create_legacy_skills_backend(SKILLS_DIR),
            _RENDER_SKILL_ROUTE: FilesystemBackend(
                root_dir=_RENDER_SKILLS_DIR, virtual_mode=True)
        }
    return _triage_skill_sources_cache


# The web app's main agent also gets the schedule tools, injected by the web composition
# root (manage/web/state) — scoped to the web-only builder for the same reason as the
# render skill: a schedule's effect needs the web's co-resident Scheduler, so emacsos and
# the eval agents (which build their own agents) must NOT get a tool that never fires.
_web_tools: tuple = ()
# A message-triage turn (inbound SMS) runs on UNTRUSTED input, so it gets a SEPARATE,
# reduced tool set: only the reply tool (HITL-gated) — NOT the schedule/subscription tools,
# which are host-effect and NOT sandbox-contained (an injected text must not be able to
# plant/delete a subscription or schedule). See docs/2026-07-01-inbound-sms-interception.org.
_web_triage_tools: tuple = ()
# HITL gating for the normal web AgentSpec (e.g. {"send_email": {...}}); the eval agents
# build their own agents and must NOT inherit it.
_web_interrupt_on: dict | None = None
# Triage has a distinct gate because its untrusted tool profile is reply-only.
_web_triage_interrupt_on: dict | None = None


def set_web_tools(tools) -> None:
    global _web_tools
    _web_tools = tuple(tools)


def set_web_triage_tools(tools) -> None:
    global _web_triage_tools
    _web_triage_tools = tuple(tools)


def set_web_interrupt_on(interrupt_on: dict | None) -> None:
    global _web_interrupt_on
    _web_interrupt_on = interrupt_on


def set_web_triage_interrupt_on(interrupt_on: dict | None) -> None:
    global _web_triage_interrupt_on
    _web_triage_interrupt_on = interrupt_on


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
        # Every directory publisher in the one web process shares this lock.
        # It prevents generic and engine-marked reservation from racing the
        # final no-overwrite directory publication.
        self._thread_reservation_lock = threading.Lock()

    @property
    def model(self):
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    self._model = select_assistant_model(0.1)
        return self._model

    def list(self) -> list[str]:
        """Return visible, published thread IDs sorted by mtime descending."""
        dirs = []
        for name in os.listdir(self.root_dir):
            dpath = os.path.join(self.root_dir, name)
            if (not os.path.isdir(dpath) or name == "__pycache__"
                    or name.startswith(".subagent-")
                    or name.startswith(".thread-")):
                continue
            if os.path.exists(os.path.join(dpath, ".subagent")):
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
           sweep.  ``manage/web/threads.py`` passes callbacks that evict
           the in-process domain/description and egress caches; the
           retention CLI passes none.
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
            on_queue_state: Callable[[str], None] | None = None,
            configurable: dict | None = None,
            triage: bool = False,
            assistant_id: str = "general-agent") -> Thread:
        tdir = self.thread_dir(thread_id)
        if not os.path.isdir(tdir):
            raise FileNotFoundError(f"thread directory not found: {thread_id}, {tdir}")
        if not working_dir:
            working_dir = self.make_default_working_dir(tdir)

        # A triage turn (untrusted inbound message) gets the reduced reply-only tool set +
        # its distinct reply HITL gate; normal turns get the full web tools and email HITL.
        tools = _web_triage_tools if triage else _web_tools
        interrupt_on = _web_triage_interrupt_on if triage else _web_interrupt_on
        specialized = None
        if assistant_id == "context-agent":
            specialized = create_context_agent(
                self.model, working_dir, self.checkpointer,
                sandbox_backend=sandbox_backend)
        elif assistant_id == "critique-agent":
            specialized = create_context_agent(
                self.model, working_dir, self.checkpointer,
                sandbox_backend=sandbox_backend,
                prompt_template="deepagents/dev_critique.md.j2")
        elif assistant_id == "research-agent":
            specialized = create_research_agent(
                self.model, working_dir, self.checkpointer,
                sandbox_backend=sandbox_backend, leaf=True)
        elif assistant_id not in {"general-agent", "delegate-agent"}:
            raise ValueError(f"unknown assistant: {assistant_id}")
        async_tools = (async_task_tools if assistant_id == "general-agent"
                       and not triage else ())
        thread_kwargs = dict(
                      thread_id=thread_id,
                      checkpointer=self.checkpointer,
                      model=self.model,
                      sandbox_backend=sandbox_backend,
                      on_queue_state=on_queue_state,
                      configurable=configurable)
        if specialized is not None:
            return Thread(working_dir, agent=specialized, **thread_kwargs)
        if assistant_id == "delegate-agent":
            # A whole-task worker is the same Assist graph with a narrower web
            # composition: shared built-in/domain capabilities and synchronous
            # specialists stay; the main-only planning skill, supervisor
            # lifecycle/front-end tools, render skill, HITL, and
            # self-delegation are simply absent.
            return Thread(
                working_dir, **thread_kwargs,
                spec=AgentSpec(role="delegate", async_subagent_tools=None))
        if sandbox_backend is None:
            agent_dir = self.thread_agent_dir(thread_id)
            os.makedirs(agent_dir, exist_ok=True)
            thread_kwargs["agent_dir"] = agent_dir
        return Thread(
            working_dir, **thread_kwargs,
            spec=AgentSpec(
                skill_sources=(_triage_skill_sources() if triage
                               else _web_skill_sources()),
                tools=tools,
                async_subagent_tools=async_tools,
                web_main=(assistant_id == "general-agent" and not triage),
                main_guidance_skills=(assistant_id == "general-agent" and not triage),
                interrupt_on=interrupt_on))

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
        tid = self.reserve()
        return self.get(
            tid, working_dir=working_dir, sandbox_backend=sandbox_backend,
            on_queue_state=on_queue_state)

    def reserve(self, thread_id: str | None = None,
                *, hidden: dict | None = None) -> str:
        with self._thread_reservation_lock:
            return self._reserve(thread_id, hidden=hidden)

    def _reserve(self, thread_id: str | None = None,
                 *, hidden: dict | None = None) -> str:
        """Reserve a thread directory and return its id without building an agent.

        Agent Protocol thread creation is a cheap persistence operation. The first run
        constructs the graph/model off the request loop through :meth:`get`. Hidden
        thread identity is published atomically and an existing identity is validated,
        never rewritten.
        """
        tid = thread_id or (
            datetime.now().strftime("%Y%m%d%H%M%S") + "-" + os.urandom(4).hex())
        tdir = self.thread_dir(tid)
        if hidden is not None:
            marker = os.path.join(tdir, ".subagent")
            if os.path.isdir(tdir):
                with open(marker) as stream:
                    if json.load(stream) != hidden:
                        raise ValueError("task metadata conflict")
                return tid
            pending = tempfile.mkdtemp(prefix=".subagent-", dir=self.root_dir)
            try:
                with open(os.path.join(pending, ".subagent"), "w") as stream:
                    json.dump(hidden, stream)
                    stream.flush()
                    os.fsync(stream.fileno())
                directory_fd = os.open(pending, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
                os.rename(pending, tdir)
                directory_fd = os.open(self.root_dir, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            finally:
                if os.path.isdir(pending):
                    shutil.rmtree(pending)
            return tid
        os.makedirs(tdir, exist_ok=True)
        return tid

    def reserve_visible(self, engine: EngineName,
                        thread_id: str | None = None) -> str:
        """Atomically reserve a visible manual-web thread with its engine marker.

        The ordinary :meth:`reserve` remains the generic/legacy primitive.  New
        web threads use this method so no caller can publish a visible directory
        and create its first Run before the immutable engine identity exists.
        """
        tid = thread_id or (
            datetime.now().strftime("%Y%m%d%H%M%S") + "-" + os.urandom(4).hex())
        tdir = self.thread_dir(tid)
        identity = ThreadEngine(engine, "manual-web")
        with self._thread_reservation_lock:
            if os.path.lexists(tdir):
                if os.path.islink(tdir) or not os.path.isdir(tdir):
                    raise ThreadEngineError("visible thread path is not a directory")
                if os.path.isfile(os.path.join(tdir, ".subagent")):
                    raise ThreadEngineError("a hidden task cannot become a visible thread")
                if read_thread_engine(tdir) != identity:
                    raise ThreadEngineError("visible thread engine conflicts with existing identity")
                return tid

            pending = tempfile.mkdtemp(prefix=".thread-", dir=self.root_dir)
            try:
                write_new_thread_engine(pending, engine)
                os.rename(pending, tdir)
                root_fd = os.open(self.root_dir, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(root_fd)
                finally:
                    os.close(root_fd)
                return tid
            finally:
                if os.path.isdir(pending):
                    shutil.rmtree(pending)

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

    # A persistent scratch dir mounted at the sandbox's /tmp (sandbox_manager),
    # SIBLING of the working dir (not under it, so it isn't part of the agent's
    # /workspace project tree).  Unlike the per-turn container's ephemeral /tmp,
    # this lives on the host and survives across turns — so a file the agent
    # writes to /tmp (e.g. a converted copy or generated image artifact) is still
    # there next turn and is reachable by the web renderer.  Removed with the
    # thread dir on hard_delete.
    def thread_tmp_dir(self, tid: str) -> str:
        return os.path.join(self.thread_dir(tid), "tmp")

    # Private main-agent state, outside both the repo and renderable scratch.
    # The thread directory owns its lifecycle, so hard_delete removes it naturally.
    def thread_agent_dir(self, tid: str) -> str:
        return os.path.join(self.thread_dir(tid), "agent")

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
