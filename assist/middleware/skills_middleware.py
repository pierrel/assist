"""Small-model-friendly skills middleware.

Subclasses deepagents' SkillsMiddleware so the small models we run (e.g.
Qwen3.6-27B) reliably load skills before acting in their domain.

Three changes from the upstream behavior:

1. A dedicated ``load_skill`` tool. The small model is much more reliable
   invoking a single-arg named tool than constructing a path string for
   ``read_file`` — the upstream mechanism. The tool takes a skill name
   (e.g. ``"org-format"``) and returns the skill body.
2. An imperative system-prompt template. The upstream template is
   verbose and example-heavy. Ours is short, name-based (no paths
   anywhere), and ends with a mandatory pre-action check that tells the
   model to scan each user message for skill triggers before any other
   tool call.
3. A generated kernel line based on the tools in the final model request.
   Application tool names supplied by the composition are excluded; no
   framework tool-name list is copied into the prompt.

The skill listing in the system prompt is name + description only — no
filesystem paths and no specific skill names baked into the surrounding
prose. The model interacts with skills through ``load_skill(name=...)``,
so paths and file structure are irrelevant from its perspective.
"""
from langchain_core.tools import tool

from deepagents.middleware.skills import SkillsMiddleware
from deepagents.middleware._utils import append_to_system_message

from assist.middleware.tool_name_sanitization import _is_valid_tool_name


SMALL_MODEL_SKILLS_PROMPT = """

## Skills

You have access to named skills. Each skill is a self-contained set of
rules for a specific domain. The list below shows each skill's *name*
and *description*; the rules themselves are revealed only when you load
the skill.

**Available skills:**

{skills_list}

### How to use a skill

1. **Match.** If a skill's description fits what you're about to do, continue to step 2.
2. **Load.** Call `load_skill(name="<skill name>")`. The tool returns the full skill body.
3. **Apply.** Use the rules from the loaded skill when you compose your action or response.

The descriptions only summarize *when* to load — they do not contain
the rules. You will not know the rules until you complete step 2. You
MUST complete step 2 before performing the matching action; relying on
the description alone leads to incorrect outcomes.

### Pre-action check (MANDATORY — apply on every turn before any tool call)

Before issuing your first tool call on a turn, scan the user's latest
message against every skill description above:

1. Look for any keyword, file extension, filename, or topic from a
   skill description that appears in the user's message.
2. If a skill matches, your FIRST tool call this turn MUST be
   `load_skill(name="<matched skill>")`. Do not run `ls`, `read_file`,
   `task` to a sub-agent, or `edit_file` first — those steps come AFTER
   the skill is loaded.
3. Only if no skill matches may you proceed directly to your task.

Skipping this check is a bug. The skill exists precisely because acting
without it produces incorrect output for that domain.
"""


def _tool_name(tool) -> str:
    """Return the model-visible name for a tool object or callable."""
    return getattr(tool, "name", None) or getattr(tool, "__name__", "")


def _without_allowed_tools(content: str) -> str:
    """Remove the ownership field while preserving every other skill byte."""
    lines = content.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        return content
    for index, line in enumerate(lines[1:], 1):
        if line.strip() == "---":
            break
        if line.startswith("allowed-tools:"):
            return "".join(lines[:index] + lines[index + 1:])
    return content


def _make_load_skill_tool(backend, sources, *, expose_allowed_tools=True,
                          assist_owned_sources=()):
    """Build a `load_skill` tool that downloads the skill body from the backend.

    Closes over the backend and source list so the tool itself takes only
    a skill name — easier for the small model to invoke reliably than
    constructing a path string for read_file.
    """
    assist_owned_sources = frozenset(assist_owned_sources)

    @tool
    def load_skill(name: str) -> str:
        """Load and return the full body of the named skill.

        Use this whenever a skill description matches your task. Pass
        only the skill's short name (e.g. "org-format") — no paths.
        Returns the full body of the skill, including the rules you
        must follow before continuing with the task.
        """
        # Resolve in REVERSE source order to match the deepagents listing,
        # which is last-source-wins (it builds a name->skill dict, so a later
        # source overwrites an earlier one).  Iterating forward here would make
        # load_skill first-source-wins and disagree with the description the
        # model just read on any name collision (e.g. a domain skill shadowing
        # a built-in).  See docs/2026-06-06-in-repo-domain-skills.org.
        for source in reversed(sources):
            path = f"{source.rstrip('/')}/{name}/SKILL.md"
            try:
                responses = backend.download_files([path])
            except Exception:
                continue
            if not responses:
                continue
            response = responses[0]
            if response.error or response.content is None:
                continue
            try:
                content = response.content.decode("utf-8")
                if not expose_allowed_tools and source in assist_owned_sources:
                    return _without_allowed_tools(content)
                return content
            except UnicodeDecodeError:
                continue
        return (
            f"Skill '{name}' not found. The system prompt's '## Skills' "
            f"section lists every available name; use one of those."
        )

    return load_skill


class SmallModelSkillsMiddleware(SkillsMiddleware):
    """SkillsMiddleware variant with a `load_skill` tool and an imperative
    system prompt — name-based throughout, no paths exposed to the model.
    """

    def __init__(self, *, backend, sources, assist_owned_sources=(),
                 non_kernel_tool_names=(), include_kernel_prompt=True):
        super().__init__(backend=backend, sources=sources)
        self.system_prompt_template = SMALL_MODEL_SKILLS_PROMPT
        self.assist_owned_sources = tuple(assist_owned_sources)
        self.tools = [_make_load_skill_tool(
            backend, sources, expose_allowed_tools=include_kernel_prompt,
            assist_owned_sources=self.assist_owned_sources)]
        self.non_kernel_tool_names = tuple(non_kernel_tool_names)
        self.include_kernel_prompt = include_kernel_prompt

    def modify_request(self, request):
        """Append the final-request kernel when enabled; otherwise keep upstream."""
        updated = super().modify_request(request)
        if not self.include_kernel_prompt:
            return updated
        non_kernel = set(self.non_kernel_tool_names)
        names = [_tool_name(tool) for tool in request.tools]
        kernel_names = [name for name in names if name not in non_kernel]
        invalid = next((name for name in kernel_names
                        if not _is_valid_tool_name(name)), None)
        if invalid is not None:
            raise ValueError(f"invalid kernel tool name: {invalid!r}")
        kernel = ", ".join(kernel_names)
        line = (f"Kernel tools: {kernel}. "
                "Declared application tools belong to skills; migration "
                "exceptions remain visible.")
        return updated.override(system_message=append_to_system_message(
            updated.system_message, line))

    def _format_skills_list(self, skills):
        """Name + description only.

        Upstream's listing appends a ``-> Read `{path}` for full
        instructions`` line, which is misleading now that we load by
        name. Strip everything except ``- **name**: description`` so the
        model has no spurious path string to copy from.
        """
        if not skills:
            return "(No skills available.)"
        return "\n".join(
            f"- **{s['name']}**: {s['description']}" for s in skills
        )
