"""Small-model-friendly memory middleware.

Subclasses deepagents' ``MemoryMiddleware`` so the small models we run
(e.g. Qwen3.6-27B) reliably read AND write the user's persistent
memory file (``AGENTS.md``).

One change from the upstream behavior: the system-prompt block injected
alongside the loaded memory is rewritten in an imperative,
small-model-friendly form.  The one-source form keeps focused
how-to-save / when-to-save / when-not-to-save / pre-action guidance.
When thread-private memory is also available, a concise scope-aware
form directs each durable fact to the repository or the current thread.

We do NOT register a dedicated ``save_memory`` tool — earlier versions
did, but the model does not need the affordance. Repository memory is
saved to ``AGENTS.md`` and web-main thread state to ``/agent/memory.md``
through the model's existing ``write_file`` / ``edit_file`` tools, with
the loaded blocks serving as ``edit_file`` anchors.

Read happens automatically: our ``before_agent`` / ``abefore_agent``
overrides discard the upstream state cache and reload every configured
file each turn; ``_format_agent_memory`` injects the contents into the
system message.
Web main agents may also load a thread-private ``/agent/memory.md``;
the established one-source prompt remains unchanged for every other
construction path.
"""
from __future__ import annotations

from deepagents.middleware.memory import MemoryMiddleware


SMALL_MODEL_MEMORY_PROMPT = """<agent_memory>
{agent_memory}
</agent_memory>

## Memory

The block above between `<agent_memory>` tags is the contents of the
file `{memory_path}`.  It is what you remember about the user across
conversations.  Treat it as authoritative for facts about the user's
identity, preferences, or past statements.

### How to save a memory

Persistent memory lives in `{memory_path}`.  To save a new fact,
append a one-sentence prose line to that file using your existing
filesystem tools.

If `<agent_memory>` above shows `(No memory loaded)` the file is
empty — use `write_file`:

  write_file(file_path="{memory_path}", content="<your sentence>\\n")

Otherwise the file already has content — use `edit_file` to append.
The current contents shown between the `<agent_memory>` tags above
are your `old_string`; the same content followed by your new sentence
is your `new_string`:

  edit_file(
    file_path="{memory_path}",
    old_string="<everything currently between the agent_memory tags>",
    new_string="<that same content>\\n<your new sentence>",
  )

Constraints on the saved sentence:
- Plain prose, one sentence per fact.
- No `<agent_memory>` tags.  No markdown headings (`#`, `##`).  No YAML.
- Never save credentials, API keys, or passwords.

### When to save

- The user explicitly asks to remember or save a fact for future conversations.
- The user states a persistent fact about themselves (identity,
  possessions, preferences, environment) NOT already in
  `<agent_memory>` above.
- The user gives forward-looking feedback or a behavioral rule.
  Examples: "in the future ...", "from now on ...", "always ...",
  "never ...", "I prefer X over Y", "I'd rather see ...", "don't do X
  again", "next time ...".  Save the rule even when the user did not
  say "remember".

### When NOT to save

- Transient state ("I'm running late", "I'm on my phone").
- One-off task requests ("find me a recipe", "what's 25 * 4?").
- Current project progress, a task's status, or a request to keep track of
  ongoing work. This is thread state, not a fact about the user across
  conversations; use private thread memory when the system provides it.
- Acknowledgments and small talk ("thanks", "sounds good").
- Credentials, API keys, passwords — never echo or save these.

### Pre-action check (MANDATORY — apply on every turn)

Before any work tool (`task`, `write_todos`, `read_file`, etc.), scan
the user's latest message for a durable fact about the user, an explicit
request to remember something across conversations, or forward-looking
feedback that is NOT already in
`<agent_memory>` above.  If you find one, your save MUST happen this
turn: `write_file(file_path="{memory_path}", ...)` if `<agent_memory>`
above shows `(No memory loaded)`, otherwise
`edit_file(file_path="{memory_path}", ...)`.  The save can run before
or after `load_skill`, but both must precede every other tool call.

The save is required even when the user's whole turn is just a
preference or rule ("I prefer Python over JavaScript", "I have 3
cats", "in the future, do X"): the correct turn is the save tool
followed by a short prose reply.  A prose-only reply ("Got it, I'll
remember that") is NOT a substitute — the memory is lost.

This is the single most common bug: the model acknowledges the fact
in prose but never persists it.  The check exists to prevent that.
"""


THREAD_MEMORY_PROMPT = """<agent_memory>
{repo_memory}
</agent_memory>

<thread_memory>
{thread_memory}
</thread_memory>

## Durable memory

When the user establishes a future or conditional response within this thread,
it is a commitment. Before acknowledging it or promising that response, write
its operative details to `{thread_memory_path}`. A prose-only acknowledgment is
not a saved commitment.
For a new commitment whose operative details are explicit in the user's
message, make that write your first tool call. A mixed request does not change
that: save a self-contained commitment before beginning unrelated work that
needs user files, personal information, or current state outside `/agent`.
Ground first only when the commitment's own operative details need that
information.
A commitment does not become optional because the same turn has other work:
after the needed grounding is complete, persist it before the final response.

When the requested response is wholly the recorded commitment, fulfill it
directly from relevant thread memory. If it also needs mutable user data or
current state outside `/agent`, follow grounding instead.

The repository memory file `{repo_memory_path}` is shared across conversations.
`{thread_memory_path}` is private to this thread. Before saving, decide the
lifetime of the information:

- Save a fact, preference, or behavioral rule in repository memory only when it
  should apply across future conversations.
- Save the current goal, process, decision, commitment, or conditional response
  in thread memory when it applies to this conversation.

A current-work checkpoint belongs in `{thread_memory_path}`, never a user file.
Do not create a user file merely to track the work.

Persist the operative details before saying they are saved or promising a
future response. For repository memory, write or append one plain-prose
sentence to `{repo_memory_path}`. For thread memory, write or append concise
Markdown to `{thread_memory_path}`. When the corresponding block is empty, use
`write_file`; otherwise use `edit_file`. Never save credentials, API keys, or
passwords. When one message contains both scopes, update the two files
separately.
"""


class SmallModelMemoryMiddleware(MemoryMiddleware):
    """``MemoryMiddleware`` variant with a small-model-friendly system prompt.

    Overrides ``before_agent`` / ``abefore_agent`` to reload files every
    turn and the formatted prompt body to distinguish the optional thread
    source. Inherits the request/model-injection hooks unchanged; no tools
    are registered. The model saves repository and optional thread
    memory through its existing filesystem tools.
    """

    def __init__(self, *, backend, memories_path: str,
                 thread_memories_path: str | None = None) -> None:
        sources = [memories_path]
        if thread_memories_path is not None:
            sources.append(thread_memories_path)
        super().__init__(backend=backend, sources=sources)

    def before_agent(self, state, runtime, config):
        # Force a fresh read every turn.  Upstream short-circuits when
        # ``memory_contents`` is already in state, which would render
        # stale content if the model wrote to the memory file via
        # ``edit_file`` / ``write_file`` on a prior turn — those tools
        # update the file on disk but not the in-state cache.
        fresh = {k: v for k, v in state.items() if k != "memory_contents"}
        return super().before_agent(fresh, runtime, config)

    async def abefore_agent(self, state, runtime, config):
        # Async twin of ``before_agent`` — same staleness fix.
        fresh = {k: v for k, v in state.items() if k != "memory_contents"}
        return await super().abefore_agent(fresh, runtime, config)

    def _format_agent_memory(self, contents: dict[str, str]) -> str:
        """Format loaded memory using the small-model prompt template.

        The one-source branch mirrors upstream's logic (memory.py:218-236) but substitutes
        ``SMALL_MODEL_MEMORY_PROMPT`` and drops upstream's per-source
        path prefix (``"{path}\\n{content}"``) — the path is rendered
        once in the prompt body via ``{memory_path}`` instead of
        per-source. The optional second source is rendered separately so
        repository and thread scope cannot be mistaken for each other.

        Keeps the ``<agent_memory>...</agent_memory>`` wrapper because
        the read-path tests (and the save-path prompt above) both
        depend on the model seeing memory in that exact frame.

        Treats whitespace-only content as empty so a stale ``"\\n"`` in
        the file does not render as a near-empty ``<agent_memory>``
        block (which would tell the model to use ``edit_file`` with an
        empty-string anchor).
        """
        memory_path = self.sources[0]
        memory_body = contents.get(memory_path, "")
        if not memory_body.strip():
            memory_body = "(No memory loaded)"
        if len(self.sources) == 1:
            return SMALL_MODEL_MEMORY_PROMPT.format(
                agent_memory=memory_body,
                memory_path=memory_path,
            )

        thread_memory_path = self.sources[1]
        thread_memory = contents.get(thread_memory_path, "")
        if not thread_memory.strip():
            thread_memory = "(No thread memory loaded)"
        return THREAD_MEMORY_PROMPT.format(
            repo_memory=memory_body,
            repo_memory_path=memory_path,
            thread_memory_path=thread_memory_path,
            thread_memory=thread_memory,
        )
