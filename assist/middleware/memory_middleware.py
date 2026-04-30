"""Small-model-friendly memory middleware.

Subclasses deepagents' ``MemoryMiddleware`` so the small models we run
(e.g. Qwen3-Coder-30B) reliably read AND write the user's persistent
memory file (``AGENTS.md``).

Two changes from the upstream behavior:

1. A dedicated ``save_memory(content=...)`` tool. The small model is
   much more reliable invoking a single-arg named tool than constructing
   a path string for ``edit_file`` — empirically it invents file names
   (``/user_info.md``) rather than writing to the configured memory path.
   The tool appends to the fixed memory file, so the model never has to
   pick a path or a strategy (insert vs. replace).
2. A short, imperative system prompt template. The upstream template
   wraps every loaded source in ``<agent_memory>`` tags (kept — needed
   so the model recognizes loaded memory) but follows it with ~50 lines
   of guidelines and three full prose examples. Our variant trims that
   to a ~15-line block: how to save, when to save, when not to save, and
   a mandatory pre-action check that scans each user message for facts
   to remember before any other tool call.

The model interacts with memory through ``save_memory(content=...)`` —
no paths anywhere. Read happens automatically: the upstream
``before_agent`` loads the memory file into state on the first turn,
and our ``_format_agent_memory`` injects the contents into the system
message. After ``save_memory`` writes new content the tool returns a
``Command`` that updates ``memory_contents`` in state, so the next turn
on the same thread sees the freshly-saved fact (without re-reading the
file from the backend).
"""
from __future__ import annotations

from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.types import Command

from deepagents.middleware.memory import MemoryMiddleware


SMALL_MODEL_MEMORY_PROMPT = """<agent_memory>
{agent_memory}
</agent_memory>

## Memory

The block above between `<agent_memory>` tags is what you remember
about the user across conversations. Treat it as authoritative when
answering questions about the user's identity, preferences, or past
statements.

### How to save a memory

Call `save_memory(content="<one short sentence>")`. The tool appends
that sentence to the user's memory file. Keep `content` to plain
prose: do not put literal `<agent_memory>` tags in it, do not put
markdown headings (`#`, `##`) in it, and do not put YAML frontmatter
in it. Multi-line text is fine; structural framing is not.

Examples:
- User: "I have 3 cats."
- You: call `save_memory(content="User has 3 cats.")`, then reply.

- User: "I prefer Python over JavaScript."
- You: call `save_memory(content="User prefers Python over JavaScript.")`, then reply.

### When to save

- The user explicitly says "remember", "save", "commit to memory", or
  similar.
- The user states a persistent fact about themselves (identity,
  possessions, preferences, environment) that is NOT already in
  `<agent_memory>` above.
- The user gives forward-looking feedback or a behavioral rule.
  Examples: "in the future ...", "from now on ...", "always ...",
  "never ...", "I prefer X over Y", "I'd rather see ...", "don't do
  X again", "next time ...". Save the rule even when the user did
  not say "remember".

### When NOT to save

- Transient state ("I'm running late", "I'm on my phone").
- One-off task requests ("find me a recipe", "what's 25 * 4?").
- Simple questions that reveal no lasting preference.
- Acknowledgments and small talk ("thanks", "sounds good").
- Credentials, API keys, passwords — never echo or save these.

### Pre-action check (MANDATORY — apply on every turn)

Before issuing your first tool call on a turn, scan the user's latest
message for a fact about the user, an explicit save request, or
forward-looking feedback that is NOT already in `<agent_memory>`
above. If you find one, your FIRST tool call this turn MUST be
`save_memory(content=...)`. Do not run `write_todos`, `task`,
`read_file`, `edit_file`, or any other tool first.

`save_memory` and `load_skill` are both pre-action tools — call either
or both before any other tool call. Order between them does not
matter, but neither is allowed to come AFTER `write_todos`, `task`,
`read_file`, or `edit_file`.

**`save_memory` is required even when nothing else is.** If the
user's whole turn is just a preference, fact, or feedback ("I prefer
Python over JavaScript", "I have 3 cats", "in the future, do X"),
the correct turn is `save_memory(content=...)` followed by a short
prose reply. A prose-only reply ("Got it, I'll remember that") is
NOT a substitute — the memory is lost.

This is the single most common bug: the model acknowledges the fact
in prose but never calls the tool. The check exists to prevent that.
"""


def _make_save_memory_tool(backend, memories_path: str):
    """Build a ``save_memory`` tool that appends to the fixed memory file.

    Closes over the backend and memory path so the tool itself takes
    only ``content`` — easier for the small model to invoke reliably
    than ``edit_file(file_path=..., old_string=..., new_string=...)``.
    Append (not replace) is the only operation the small model gets to
    pick from, which eliminates the "invented filename / wrong write
    strategy" failure mode.

    The tool returns a ``Command`` that updates ``memory_contents`` in
    agent state alongside the tool's reply message, so a subsequent
    turn on the same thread sees the freshly-saved fact in the system
    prompt (the upstream ``before_agent`` only loads on the first turn
    of a session, so without this update later turns would render
    stale content).

    The read-then-write sequence is not concurrency-safe. In the
    deepagents single-threaded tool loop this is fine: the model
    issues at most one ``save_memory`` call per turn and waits for the
    result before issuing the next. If the model ever emits two
    parallel ``save_memory`` tool calls in a single assistant message,
    the second can clobber the first. We accept that constraint
    rather than reach for file locking.
    """

    @tool
    def save_memory(
        content: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        """Append a short fact about the user to persistent memory.

        Use this whenever the user states something worth remembering
        across conversations (identity, preferences, persistent state).
        Pass plain prose — one short sentence per fact. Do not include
        `<agent_memory>` tags, markdown headings, or YAML frontmatter.

        Returns a confirmation message.
        """
        # Read existing memory. If we can't read, refuse to write —
        # otherwise a transient backend error would silently clobber
        # the user's saved facts.
        try:
            responses = backend.download_files([memories_path])
        except Exception as exc:
            return _failure_command(tool_call_id, f"Memory save failed (read error): {exc}")

        if not responses:
            return _failure_command(tool_call_id, "Memory save failed: backend returned no response")

        resp = responses[0]
        if resp.error and resp.error != "file_not_found":
            return _failure_command(tool_call_id, f"Memory save failed (read error): {resp.error}")

        if resp.error == "file_not_found" or resp.content is None:
            existing = ""
        else:
            try:
                existing = resp.content.decode("utf-8")
            except UnicodeDecodeError as exc:
                return _failure_command(tool_call_id, f"Memory save failed (decode error): {exc}")

        new_content = existing
        if new_content and not new_content.endswith("\n"):
            new_content += "\n"
        new_content += content.strip("\n") + "\n"

        try:
            backend.upload_files([(memories_path, new_content.encode("utf-8"))])
        except Exception as exc:
            return _failure_command(tool_call_id, f"Memory save failed (write error): {exc}")

        return Command(
            update={
                "memory_contents": {memories_path: new_content},
                "messages": [
                    ToolMessage(
                        content="Memory saved.",
                        tool_call_id=tool_call_id,
                        name="save_memory",
                    )
                ],
            }
        )

    return save_memory


def _failure_command(tool_call_id: str, message: str) -> Command:
    """Return a Command that surfaces a save_memory failure to the model.

    Used instead of a bare string return so the failure path matches
    the success path (both are Commands), and the failure does NOT
    update ``memory_contents`` — leaving state untouched is the whole
    point of fail-closed-on-read-error.
    """
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=message,
                    tool_call_id=tool_call_id,
                    name="save_memory",
                    status="error",
                )
            ]
        }
    )


class SmallModelMemoryMiddleware(MemoryMiddleware):
    """MemoryMiddleware variant with a ``save_memory`` tool and an
    imperative, small-model-friendly system prompt.

    Inherits ``before_agent`` / ``abefore_agent`` (file load into state)
    and ``modify_request`` / ``wrap_model_call`` (system-message
    injection) unchanged from the upstream class. Only the formatted
    prompt body and the registered tools change.
    """

    def __init__(self, *, backend, memories_path: str) -> None:
        if callable(backend):
            # SmallModelMemoryMiddleware closes over the backend at
            # construction time when wiring the save_memory tool. Backend
            # *factories* (callables resolved per-request via runtime)
            # would not survive that closure. We only run with concrete
            # backend instances today; reject factories explicitly so the
            # failure mode is loud rather than an opaque AttributeError
            # the first time save_memory fires.
            msg = (
                "SmallModelMemoryMiddleware requires a concrete backend "
                "instance, not a factory callable."
            )
            raise TypeError(msg)
        super().__init__(backend=backend, sources=[memories_path])
        self.tools = [_make_save_memory_tool(backend, memories_path)]

    def _format_agent_memory(self, contents: dict[str, str]) -> str:
        """Format loaded memory using the small-model prompt template.

        Mirrors upstream's logic (memory.py:218–236) but substitutes
        ``SMALL_MODEL_MEMORY_PROMPT`` and drops upstream's per-source
        path prefix (``"{path}\\n{content}"``). With a single configured
        source — and the model never having to address it by path —
        showing ``/AGENTS.md`` in the prompt is just a token the small
        model can copy by accident into saved memory. We emit the body
        only.

        Keeps the ``<agent_memory>...</agent_memory>`` wrapper because
        ``test_doesnt_include_tags`` and ``test_reads_memory`` both
        depend on the model seeing memory in that exact frame.
        """
        if not contents:
            return SMALL_MODEL_MEMORY_PROMPT.format(agent_memory="(No memory loaded)")

        sections = [contents[path] for path in self.sources if contents.get(path)]
        if not sections:
            return SMALL_MODEL_MEMORY_PROMPT.format(agent_memory="(No memory loaded)")

        memory_body = "\n\n".join(sections)
        return SMALL_MODEL_MEMORY_PROMPT.format(agent_memory=memory_body)
