import contextvars
import logging
from datetime import datetime
from typing import Any, Callable, List, Iterator, Mapping

from langchain.messages import HumanMessage, AIMessage, AnyMessage
from langchain_core.language_models.chat_models import BaseChatModel

from assist.promptable import base_prompt_for
from assist.model_manager import select_assistant_model
from assist.agent import create_agent
from assist.spec import AgentSpec
from assist.checkpoint_rollback import invoke_with_rollback
from assist.thread_queue import THREAD_QUEUE
from langgraph.types import Command

logger = logging.getLogger(__name__)

def render_tool_calls(message: AIMessage) -> str:
    """The tool-call text line for a message's calls, or "" when it has none.
    The CLI prints this for every AIMessage, so a message with no tool calls
    must render empty — otherwise its plain content (already streamed/printed
    separately) would be duplicated."""
    calls = getattr(message, "tool_calls", None)
    if not calls:
        return ""
    return _render_calls(calls, getattr(message, "content", None))


def _messages_to_dicts(raw: list) -> list[dict]:
    """Convert checkpointer messages to the role/content dicts the web UI renders.

    Pure (no agent/state access) so it's unit-testable. An AIMessage's tool calls
    become the ``"tools"`` text line (which also carries the message's own content
    when it has calls); a message with content and no tool calls is an
    ``"assistant"`` message (its content may carry a ```render block the web layer
    turns into a file embed); a HumanMessage is ``"user"``."""
    msgs: list[dict] = []
    for m in raw:
        if isinstance(m, HumanMessage):
            msgs.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage):
            calls = getattr(m, "tool_calls", None)
            if calls:
                msgs.append({"role": "tools",
                             "content": _render_calls(calls, m.content)})
            elif m.content:
                msgs.append({"role": "assistant", "content": m.content})
    return msgs


def _render_calls(calls: list, content) -> str:
    """The tool-call text line for an AIMessage's calls (+ optional prose).
    Shared by ``render_tool_calls`` and ``_messages_to_dicts``."""
    s = " -- ".join(render_tool_call(c) for c in calls)
    if content:
        return f"{s} \n> {content}" if s else str(content)
    return s


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
                 *,
                 spec: AgentSpec | None = None,
                 configurable: Mapping[str, Any] | None = None,
                 thread_id: str | None = None,
                 checkpointer=None,
                 model: BaseChatModel | None = None,
                 max_concurrency: int = 5,
                 sandbox_backend=None,
                 on_queue_state: Callable[[str], None] | None = None):
        """The embedder surface (docs/2026-06-11-embedder-contract.org):

        `spec` — the embedder contract (``assist.spec.AgentSpec``): one
        declaration object carrying embedder-supplied tools, skill
        sources, and default backend (canonical semantics live on the
        spec's field docs).  Forwarded to ``create_agent(spec=...)``.
        ``None`` means "no embedder additions" — the defaults.

        `configurable` — per-request context: a mapping merged
        (shallow, one level) into ``self.runconfig["configurable"]`` so
        embedder context (e.g. emacsos's phone identity) reaches every
        invocation.  The reserved langgraph keys ``thread_id`` /
        ``checkpoint_ns`` / ``checkpoint_id`` raise ``ValueError``
        (thread identity belongs to the ``thread_id=`` param; the
        checkpoint keys are not settable at all).

        The remaining params are per-instance/run wiring: identity
        (``thread_id``), persistence (``checkpointer``), model,
        concurrency, sandbox, and the queue-state callback.  The split
        rule: *spec = the agent's shape; kwargs = instance wiring*.
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
                    f"{sorted(reserved)} — thread identity belongs to the "
                    f"thread_id= constructor param; checkpoint_ns/"
                    f"checkpoint_id are not settable")
            # .update() copies the entries into runconfig's own dict, so
            # an embedder mutating its mapping later can't change this
            # Thread's runconfig (nested values stay shared, as ever).
            self.runconfig["configurable"].update(configurable)

        self.agent = create_agent(self.model,
                                  working_dir=working_dir,
                                  checkpointer=checkpointer,
                                  sandbox_backend=sandbox_backend,
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

    def pending_reply(self) -> dict | None:
        """If this thread is paused awaiting approval of a ``send_reply`` (HITL interrupt),
        return ``{"text": <draft>}`` — read from the durable checkpoint, so it survives the
        request that produced it. ``None`` when the thread isn't awaiting an approval.
        """
        try:
            snap = self.agent.get_state(self.runconfig)
        except Exception:
            return None
        for intr in (getattr(snap, "interrupts", None) or ()):
            value = intr.value or {}
            for ar in value.get("action_requests", []):
                if ar.get("name") == "send_reply":
                    return {"text": ar.get("args", {}).get("text", "")}
        return None

    def resume_reply(self, decision: dict) -> str:
        """Resume a paused ``send_reply`` with a HITL decision — ``{"type": "approve"}``,
        ``{"type": "reject", "message": …}``, or ``{"type": "edit", "edited_action":
        {"name": "send_reply", "args": {"text": …}}}``. On approve/edit the tool body runs
        (the reply is sent); returns the agent's final content."""
        with THREAD_QUEUE.acquire(self.thread_id, on_state_change=self.on_queue_state):
            result = invoke_with_rollback(
                self.agent,
                Command(resume={"decisions": [decision]}),
                self.runconfig,
            )
        messages = result.get("messages", [])
        if messages and isinstance(messages[-1], AIMessage):
            return messages[-1].content
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
        return _messages_to_dicts(state.values.get("messages", []))


    def get_raw_messages(self) -> List[AnyMessage]:
        state = self.agent.get_state(self.runconfig)
        return state.values.get("messages", [])

    
    def description(self) -> str:
        """Return a short (<=5 words) description of the conversation so far.

        Uses the underlying chat model directly. Raises ValueError if there
        are no messages yet.  Caching is the caller's concern — the web
        app caches descriptions app-side (manage/web/state.py).
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
