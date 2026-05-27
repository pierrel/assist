import logging
import os
import shutil
import subprocess
import threading
import time
import tempfile
from datetime import datetime
from typing import Literal, Dict, Any, Callable, List, Iterator, Sequence

import sqlite3
from langchain.messages import HumanMessage, AIMessage, AnyMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.language_models.chat_models import BaseChatModel

from deepagents.backends.protocol import BackendProtocol

from assist.promptable import base_prompt_for
from assist.model_manager import select_chat_model
from assist.agent import create_research_agent, create_agent
from assist.checkpoint_rollback import invoke_with_rollback
from assist.sandbox_manager import SandboxManager
from assist.thread_queue import THREAD_QUEUE

logger = logging.getLogger(__name__)

def render_tool_calls(message: AIMessage) -> str:
    calls = getattr(message, "tool_calls", None)
    if calls:
        calls_str = " -- ".join(map(lambda c: render_tool_call(c), calls))
        if getattr(message, "content", None):
            return f"{calls_str} \n> {message.content}"

        return calls_str
    return ""


def render_tool_call(call: dict) -> str:
    name = call.get("name", "none")
    args = call.get("args", {})
    if name == "task" and call.get("args", None):
        subagent = args.get("subagent_type", "none")
        return f"Calling subagent {subagent} with {args}"
    else:
        return f"Calling {name} with {args}"

class Thread:
    """Reusable chat-like interface that mimics the CLI back-and-forth.

    Initialize with a working directory; it derives a thread id from cwd + timestamp,
    keeps a rolling messages list, and exposes a message() method that returns the
    assistant reply as a string.
    """

    def __init__(self,
                 working_dir: str,
                 thread_id: str | None = None,
                 checkpointer=None,
                 model: BaseChatModel | None = None,
                 max_concurrency: int = 5,
                 sandbox_backend=None,
                 default_backend: BackendProtocol | None = None,
                 on_queue_state: Callable[[str], None] | None = None,
                 extra_tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None,
                 loop_exploration_tools: frozenset[str] | None = None,
                 extra_skill_sources: dict[str, BackendProtocol] | None = None,
                 extra_config: dict[str, Any] | None = None):
        """`extra_tools` is forwarded to ``create_agent(extra_tools=...)``
        — embedder-supplied tools the main agent can call.  See
        ``assist.agent.create_agent`` docstring for the subagent
        scope.

        `loop_exploration_tools` is forwarded to
        ``create_agent(loop_exploration_tools=...)`` — tool names whose
        distinct-args breadth gets a relaxed (but still finite) loop-
        detection threshold because probing many forms is their normal
        shape (eg. an embedder's ``eval_elisp``).  Default ``None``.

        `extra_skill_sources` is forwarded to
        ``create_agent(extra_skill_sources=...)`` — a mapping of additional
        virtual-path routes to backends that hold ``SKILL.md`` files, so an
        embedder can ship skills that live outside the assist repo (see
        ``assist.agent.create_agent``).  Default ``None``.

        `default_backend` is forwarded to ``create_agent(default_backend=...)``
        — the composite backend's default (the target for non-routed paths),
        instead of a FilesystemBackend rooted at ``working_dir``.  Mutually
        exclusive with ``sandbox_backend``; if it implements
        ``SandboxBackendProtocol`` the ``execute`` tool is enabled for the
        main agent.  Default ``None``.

        `extra_config` is merged into ``self.runconfig`` so per-Thread
        embedder context (eg. langgraph ``configurable`` values that
        tools read via the ``config: RunnableConfig`` parameter) reaches
        every invocation without the embedder having to call
        ``.with_config(...)`` at every entry point.  Default ``None``
        preserves prior behavior.

        *Merge semantics:* two-level, not recursive.  The nested
        ``configurable`` dict gets a shallow `.update()` from the
        embedder's ``configurable`` (so adding a key alongside
        ``thread_id`` works as expected; passing a key whose value
        is itself a dict overwrites any existing dict at that key
        wholesale — no deep merge).  Top-level keys other than
        ``configurable`` are overridden wholesale.

        Constructor-owned keys (``configurable.thread_id``,
        ``max_concurrency``) are NOT overridable via ``extra_config``
        — `self.thread_id` / `self.max_concurrency` would diverge
        from what the runconfig says, and downstream code
        (THREAD_QUEUE affinity, ``message()``'s log lines) reads the
        attribute not the config.  Pass those via the dedicated
        ``thread_id=`` / ``max_concurrency=`` constructor params
        instead; the merge silently drops the protected keys to keep
        the attribute and runconfig in sync.
        """
        self.working_dir = working_dir
        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        self.thread_id = thread_id or f"{working_dir}:{ts}"
        self.model = model or select_chat_model(0.1, enable_thinking=False)
        self.max_concurrency = max_concurrency
        # Notified with "queued" if another thread is holding the LLM
        # queue when this Thread.message() runs, then "running" once
        # acquired.  Callers (e.g. manage.web) wire this to the
        # status.json so the UI can show "queued".
        self.on_queue_state = on_queue_state
        self.runconfig = {
            "configurable": {"thread_id": self.thread_id},
            "max_concurrency": self.max_concurrency
        }
        if extra_config is not None:
            # `is not None` (vs `if extra_config:`) so a falsy-but-
            # wrong-type value like `[]` or `""` still trips the
            # isinstance check below instead of silently skipping the
            # whole merge.  An explicit `{}` no-ops harmlessly either
            # way.  Validate up front so an embedder gets a clear
            # error instead of a downstream AttributeError on
            # `.items()`.
            if not isinstance(extra_config, dict):
                raise TypeError(
                    f"extra_config must be a dict, got {type(extra_config).__name__}"
                )
            inner = extra_config.get("configurable")
            if inner is not None and not isinstance(inner, dict):
                raise TypeError(
                    f"extra_config['configurable'] must be a dict, "
                    f"got {type(inner).__name__}"
                )
            # Two-level merge (NOT recursive): the inner `configurable`
            # dict gets a shallow `.update()` from the embedder's
            # `configurable`; top-level keys are overridden wholesale.
            # Protected keys (`thread_id` inside `configurable`, and
            # top-level `max_concurrency`) are silently dropped from
            # the embedder's input to keep `self.thread_id` /
            # `self.max_concurrency` in sync with the runconfig —
            # see docstring.
            #
            # The dict-comprehension below also serves as a defensive
            # shallow copy of `extra_config["configurable"]` — if the
            # embedder mutates its own dict later, our `runconfig`
            # isn't affected.  Nested values themselves are not
            # deep-copied (consistent with the documented "two-level,
            # not recursive" merge semantics).
            extra_configurable = {
                k: v for k, v in (inner or {}).items()
                if k != "thread_id"
            }
            self.runconfig["configurable"].update(extra_configurable)
            for k, v in extra_config.items():
                if k in ("configurable", "max_concurrency"):
                    continue
                self.runconfig[k] = v

        self.agent = create_agent(self.model,
                                  working_dir=working_dir,
                                  checkpointer=checkpointer,
                                  sandbox_backend=sandbox_backend,
                                  default_backend=default_backend,
                                  extra_tools=extra_tools,
                                  loop_exploration_tools=loop_exploration_tools,
                                  extra_skill_sources=extra_skill_sources)

    def message(self, text: str) -> str:
        """Continue the thread and return the last response.

        Acquires the per-thread LLM affinity queue for the duration of
        the agent loop so concurrent threads don't thrash llama.cpp's
        single KV-cache slot.  See ``assist/thread_queue.py``.
        """
        with THREAD_QUEUE.acquire(self.thread_id, on_state_change=self.on_queue_state):
            result = invoke_with_rollback(
                self.agent,
                {"messages": [{"role": "user", "content": text}]},
                self.runconfig,
            )
        # Extract content from the last AIMessage
        messages = result.get("messages", [])
        if messages:
            last_msg = messages[-1]
            if isinstance(last_msg, AIMessage):
                return last_msg.content
        return ""

    def stream_message(self, text: str) -> Iterator[dict[str, Any] | Any]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        # Continue the thread by sending only the latest human message; prior state is in the checkpointer.
        # Wrap the iterator in a generator so the queue is held for the
        # full streaming lifetime, not just the call to .stream().
        def _gen():
            with THREAD_QUEUE.acquire(self.thread_id, on_state_change=self.on_queue_state):
                yield from self.agent.stream(
                    {"messages": [{"role": "user", "content": text}]},
                    self.runconfig,
                    stream_mode=["messages", "updates"],
                    # durability="sync": persist each checkpoint synchronously
                    # before the next step instead of langgraph's default
                    # "async" background writes.  The async path submits every
                    # checkpoint put to a BackgroundExecutor and chains them via
                    # futures (`_checkpointer_put_after_previous`); under the
                    # streamed agent's deep nesting (general.stream → sub-agent
                    # .invoke → nested trio + concurrent tools) that pool gets
                    # exhausted — every worker blocks on the previous put's
                    # future with none left to complete them, deadlocking the
                    # turn (confirmed by py-spy, 2026-05-25).  "sync" keeps one
                    # put in flight at a time, so the pool can't pile up, while
                    # still writing per-step checkpoints (RollbackRunnable needs
                    # them).  langgraph propagates this to the task-tool
                    # sub-agents via the config, so the whole tree is covered.
                    durability="sync",
                )
        return _gen()

    def get_messages(self) -> list[dict]:
        """Return user/assistant messages from checkpointer state as role/content dicts."""
        state = self.agent.get_state(self.runconfig)
        msgs = []
        for m in state.values.get("messages", []):
            if isinstance(m, HumanMessage):
                msgs.append({"role": "user", "content": m.content})
            elif isinstance(m, AIMessage):
                calls = getattr(m, "tool_calls", None)
                if calls:
                    msgs.append({"role": "tools",
                                 "content": render_tool_calls(m)})
                elif m.content:
                    msgs.append({"role": "assistant", "content": m.content})
        return msgs


    def get_raw_messages(self) -> List[AnyMessage]:
        state = self.agent.get_state(self.runconfig)
        return state.values.get("messages", [])

    
    def description(self) -> str:
        """Return a short (<=5 words) description of the conversation so far.

        Uses the underlying chat model directly. Raises ValueError if there
        are no messages yet.
        If description.txt exists in the thread directory, return it; otherwise compute and cache.
        """
        msgs = self.get_messages()
        if not msgs:
            raise ValueError("no messages to describe")

        # Filter out "tools" role messages - LangChain doesn't recognize this role
        # Only include user/assistant messages for description generation
        filtered_msgs = [m for m in msgs if m.get("role") in ("user", "assistant")]

        if not filtered_msgs:
            raise ValueError("no user/assistant messages to describe")

        prompt = {
            "role": "system",
            "content": base_prompt_for("deepagents/describe_system.md.j2"),
        }
        request = {
            "role": "user",
            "content": "Describe the conversation up until now",
        }
        resp = self.model.invoke([prompt] + filtered_msgs + [request])
        desc = resp.content.strip()

        return desc


class ThreadManager:
    """Manage DeepAgentsThread instances persisted under a directory tree.

    At the root directory, a sqlite DB named 'threads.db' is used for LangGraph
n    checkpointing via SqliteSaver.
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
                    self._model = select_chat_model(0.1, enable_thinking=False)
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
        tdir = os.path.join(self.root_dir, thread_id)
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
        tdir = os.path.join(self.root_dir, tid)
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
        tdir = os.path.join(self.root_dir, thread_id)
        if os.path.isdir(tdir):
            os.utime(tdir, None)

    def get(self,
            thread_id: str,
            working_dir: str | None = None,
            sandbox_backend=None,
            on_queue_state: Callable[[str], None] | None = None) -> Thread:
        tdir = os.path.join(self.root_dir, thread_id)
        if not os.path.isdir(tdir):
            raise FileNotFoundError(f"thread directory not found: {thread_id}, {tdir}")
        if not working_dir:
            working_dir = self.make_default_working_dir(tdir)

        return Thread(working_dir,
                      thread_id=thread_id,
                      checkpointer=self.checkpointer,
                      model=self.model,
                      sandbox_backend=sandbox_backend,
                      on_queue_state=on_queue_state)

    def remove(self, thread_id: str) -> None:
        tdir = os.path.join(self.root_dir, thread_id)
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
                      on_queue_state=on_queue_state)

    def close(self) -> None:
        try:
            if hasattr(self, "conn") and self.conn:
                self.conn.close()
        except Exception:
            pass

    def thread_dir(self, tid: str) -> str:
        return os.path.join(self.root_dir, tid)

    def thread_default_working_dir(self, tid: str) -> str:
        return os.path.join(os.path.join(self.root_dir, tid),
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
