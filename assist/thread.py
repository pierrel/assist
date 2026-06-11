import contextvars
import logging
from datetime import datetime
from typing import Literal, Dict, Any, Callable, List, Iterator, Mapping, Sequence

from langchain.messages import HumanMessage, AIMessage, AnyMessage
from langchain_core.tools import BaseTool
from langchain_core.language_models.chat_models import BaseChatModel

from deepagents.backends.protocol import BackendProtocol

from assist.promptable import base_prompt_for
from assist.model_manager import select_assistant_model
from assist.agent import create_agent
from assist.spec import AgentSpec
from assist.checkpoint_rollback import invoke_with_rollback
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


class _ContextBoundIterator:
    """Iterator that routes every step through a captured ``contextvars.Context``.

    Used by :meth:`Thread.stream_message` to keep its generator's
    ``__enter__`` / ``__exit__`` (and the THREAD_QUEUE contextvar token
    they bind) on the same Context, even when ``__next__`` / ``close`` are
    called from a different OS thread.  See the comment in
    ``stream_message`` for the failure mode this closes.

    ``__del__`` routes the GC-driven generator finalization through
    ``ctx.run`` too: CPython's generator finalizer (``_PyGen_Finalize``)
    runs on whichever thread the collector happens to be on, which
    typically isn't the construction thread.  Without this, a consumer
    that drops the iterator without calling ``close()`` (e.g. emacsos-
    server client disconnect) re-introduces the contextvar-mismatch
    leak that PR #114 closed for the explicit-iteration path.
    """
    __slots__ = ("_ctx", "_gen")

    def __init__(self, ctx: contextvars.Context, gen: Iterator) -> None:
        self._ctx = ctx
        self._gen = gen

    def __iter__(self) -> "_ContextBoundIterator":
        return self

    def __next__(self):
        return self._ctx.run(self._gen.__next__)

    def close(self) -> None:
        self._ctx.run(self._gen.close)

    def __del__(self) -> None:
        # GC-driven finalization.  Errors are swallowed because Python
        # routes them to sys.unraisablehook anyway; raising here just
        # pollutes the unraisable channel without any caller to handle it.
        try:
            self._ctx.run(self._gen.close)
        except Exception:
            pass


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
                 on_queue_state: Callable[[str], None] | None = None,
                 extra_tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None,
                 loop_exploration_tools: frozenset[str] | None = None,
                 extra_skill_sources: dict[str, BackendProtocol] | None = None,
                 extra_config: dict[str, Any] | None = None,
                 default_backend: BackendProtocol | None = None,
                 *,
                 spec: AgentSpec | None = None,
                 configurable: Mapping[str, Any] | None = None):
        """`spec` is the embedder contract (``assist.spec.AgentSpec``):
        one declaration object carrying embedder-supplied tools, skill
        sources, and default backend.  Forwarded to
        ``create_agent(spec=...)``, which validates that it isn't
        combined with the legacy per-need kwargs below.  ``None``
        means "no embedder additions" — today's defaults.

        `configurable` is the narrowed replacement for ``extra_config``:
        a mapping merged (shallow, one level) into
        ``self.runconfig["configurable"]`` so per-request embedder
        context (e.g. emacsos's phone identity) reaches every
        invocation.  The reserved langgraph keys ``thread_id`` /
        ``checkpoint_ns`` / ``checkpoint_id`` raise ``ValueError`` —
        pass identity via the ``thread_id=`` constructor param.
        Mutually exclusive with ``extra_config``.

        `extra_tools` is forwarded to ``create_agent(extra_tools=...)``
        — embedder-supplied tools the main agent can call.  See
        ``assist.agent.create_agent`` docstring for the subagent
        scope.

        `loop_exploration_tools` is DEPRECATED and now a no-op (it fed the
        since-removed Pattern-C distinct-args breadth threshold).  Still
        forwarded to ``create_agent`` and accepted there so embedders that
        pass it don't break, but ignored — loop detection now catches only
        exact-repeat loops (A/B), which apply uniformly to every tool.
        Default ``None``.

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

        `extra_config` is the DEPRECATED predecessor of
        ``configurable``: only its ``{"configurable": {...}}`` shape
        was ever used by a client (verified across manage.web,
        emacsos-server, and the eval harness), so it is now a thin
        adapter over the same narrowed merge.  Top-level keys other
        than ``configurable`` raise (they used to pass through to the
        runconfig; no client did that).  ``configurable.thread_id``
        keeps its historical silent-drop (the constructor param owns
        it).  Mutually exclusive with ``configurable``; removed once
        the known embedders are ported to the new kwarg.
        """
        self.working_dir = working_dir
        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        self.thread_id = thread_id or f"{working_dir}:{ts}"
        self.model = model or select_assistant_model(0.1)
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
        if configurable is not None and extra_config is not None:
            raise TypeError(
                "Thread: pass configurable= OR the deprecated extra_config=, "
                "not both — extra_config's replacement is configurable")
        if extra_config is not None:
            # DEPRECATED adapter: normalize the legacy shape onto the
            # narrowed merge below.  `is not None` (vs `if extra_config:`)
            # so a falsy-but-wrong-type value like `[]` or `""` still
            # trips the isinstance check instead of silently skipping.
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
            stray = [k for k in extra_config if k != "configurable"]
            if stray:
                raise TypeError(
                    f"extra_config top-level keys are no longer merged into "
                    f"the runconfig (got {stray[0]!r}); only the "
                    f"'configurable' key is supported — and its replacement "
                    f"is the configurable= kwarg")
            # Historical silent-drop of thread_id is preserved for the
            # deprecated kwarg (the new `configurable=` raises instead).
            configurable = {
                k: v for k, v in (inner or {}).items() if k != "thread_id"
            }

        if configurable is not None:
            if not isinstance(configurable, Mapping):
                raise TypeError(
                    f"configurable must be a mapping, got "
                    f"{type(configurable).__name__}"
                )
            # Reserved langgraph checkpoint-addressing keys: letting an
            # embedder set these would silently retarget checkpoint
            # resolution (thread_id additionally diverging from
            # self.thread_id, which THREAD_QUEUE affinity reads).
            reserved = {"thread_id", "checkpoint_ns", "checkpoint_id"} \
                & set(configurable)
            if reserved:
                raise ValueError(
                    f"configurable must not set reserved langgraph keys "
                    f"{sorted(reserved)}; pass identity via the thread_id= "
                    f"constructor param")
            # Shallow copy: an embedder mutating its own mapping after
            # construction must not change this Thread's runconfig.
            self.runconfig["configurable"].update(dict(configurable))

        self.agent = create_agent(self.model,
                                  working_dir=working_dir,
                                  checkpointer=checkpointer,
                                  sandbox_backend=sandbox_backend,
                                  default_backend=default_backend,
                                  extra_tools=extra_tools,
                                  loop_exploration_tools=loop_exploration_tools,
                                  extra_skill_sources=extra_skill_sources,
                                  spec=spec)

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
        #
        # The iterator is bound to a captured contextvars.Context (see
        # _ContextBoundIterator below).  Without that, a consumer that
        # drives `next()` / `close()` from a different OS thread (e.g.
        # the emacsos-server's `run_in_executor` → `next()` pump, or
        # any FastAPI streaming response handler) exits THREAD_QUEUE's
        # `with` block in a different Context than it entered — and
        # `_active_handle.reset(token)` raises ValueError, leaking
        # `_holder` until the watchdog (see assist/thread_queue.py)
        # force-releases it.  The binding makes that bug structurally
        # impossible: every step runs in the construction context.
        ctx = contextvars.copy_context()
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
        return _ContextBoundIterator(ctx, _gen())

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
