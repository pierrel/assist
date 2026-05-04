"""Small-model-friendly memory middleware.

Subclasses deepagents' ``MemoryMiddleware`` so the small models we run
(e.g. Qwen3-Coder-30B) reliably read AND write the user's persistent
memory file (``AGENTS.md``).

One change from the upstream behavior: the system-prompt block injected
alongside the loaded memory is rewritten in an imperative,
small-model-friendly form.  Upstream wraps every loaded source in
``<agent_memory>`` tags and follows that with ~50 lines of guidelines
and three full prose examples; our variant trims that to a focused
block of how-to-save / when-to-save / when-not-to-save / pre-action
check.

We do NOT register a dedicated ``save_memory`` tool — earlier versions
did, but the model does not need the affordance.  Memory is saved by
having the model invoke its existing ``write_file`` / ``edit_file``
tools against ``AGENTS.md``, with the loaded ``<agent_memory>`` block
serving as the anchor for the ``edit_file`` replace.

Read happens automatically: the upstream ``before_agent`` loads the
memory file into state on the first turn, and our
``_format_agent_memory`` injects the contents into the system message.
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

- The user explicitly says "remember", "save", "commit to memory", or
  similar.
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
- Acknowledgments and small talk ("thanks", "sounds good").
- Credentials, API keys, passwords — never echo or save these.

### Pre-action check (MANDATORY — apply on every turn)

Before any work tool (`task`, `write_todos`, `read_file`, etc.), scan
the user's latest message for a fact about the user, an explicit
save request, or forward-looking feedback that is NOT already in
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


class SmallModelMemoryMiddleware(MemoryMiddleware):
    """``MemoryMiddleware`` variant with a small-model-friendly system prompt.

    Inherits ``before_agent`` / ``abefore_agent`` (file load into state)
    and ``modify_request`` / ``wrap_model_call`` (system-message
    injection) unchanged from the upstream class.  Only the formatted
    prompt body changes; no tools are registered.  The model saves new
    memory by invoking its existing filesystem tools against
    ``AGENTS.md``.
    """

    def __init__(self, *, backend, memories_path: str) -> None:
        super().__init__(backend=backend, sources=[memories_path])

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

        Mirrors upstream's logic (memory.py:218-236) but substitutes
        ``SMALL_MODEL_MEMORY_PROMPT`` and drops upstream's per-source
        path prefix (``"{path}\\n{content}"``) — the path is rendered
        once in the prompt body via ``{memory_path}`` instead of
        per-source, since we configure exactly one source.

        Keeps the ``<agent_memory>...</agent_memory>`` wrapper because
        the read-path tests (and the save-path prompt above) both
        depend on the model seeing memory in that exact frame.

        Treats whitespace-only content as empty so a stale ``"\\n"`` in
        the file does not render as a near-empty ``<agent_memory>``
        block (which would tell the model to use ``edit_file`` with an
        empty-string anchor).
        """
        memory_path = self.sources[0]
        if not contents:
            return SMALL_MODEL_MEMORY_PROMPT.format(
                agent_memory="(No memory loaded)", memory_path=memory_path,
            )

        sections = [
            contents[path] for path in self.sources
            if contents.get(path) and contents[path].strip()
        ]
        if not sections:
            return SMALL_MODEL_MEMORY_PROMPT.format(
                agent_memory="(No memory loaded)", memory_path=memory_path,
            )

        memory_body = "\n\n".join(sections)
        return SMALL_MODEL_MEMORY_PROMPT.format(
            agent_memory=memory_body, memory_path=memory_path,
        )
