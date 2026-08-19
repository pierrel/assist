"""The embedder contract: what a client declares about its agent.

``AgentSpec`` is the single declaration surface for embedders (the dev
web app, emacsos-server, a future CLI) — it replaces the per-need
kwargs that used to accrete on ``Thread`` / ``create_agent``
(``extra_tools``, ``extra_skill_sources``, ``default_backend``).  See
docs/2026-06-11-embedder-contract.org for the design and the split
rule: *spec = the agent's shape, consumed by create_agent; Thread
kwargs = per-instance and per-run wiring* (identity, persistence,
model, concurrency, status callback, app-owned ``agent_dir``, and
per-request ``configurable``).

Admission rule: a field requires a real, existing client need.  Do not
add fields for needs no client has yet — deferred candidates
(``system_prompt``, middleware tuning) are
recorded in the design doc with the trigger that revives them.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

from langchain_core.tools import BaseTool
from deepagents.backends.protocol import BackendProtocol


@dataclass(frozen=True, slots=True)
class AgentSpec:
    """Declares an embedder's agent for one ``Thread``/``create_agent`` call.

    Lifecycle: a spec describes ONE agent construction.  Fields may
    close over per-request state (emacsos's ``EmacsBackend`` closes
    over the phone identity), so a spec is NOT safely cacheable across
    requests as a module constant.  "Frozen" means the declaration
    doesn't mutate after construction: ``__post_init__`` normalizes
    ``tools`` to a tuple and ``skill_sources`` to a read-only mapping
    over a copy.

    Construction is pure CPU — no I/O, no backend listing, no probing.
    Callers may build a spec anywhere, including code adjacent to an
    asyncio event loop (the expensive work happens later, in
    ``create_agent``, which must stay off the loop).

    Not hashable or picklable: ``skill_sources`` is a mappingproxy and
    fields may hold closures — don't use specs as dict/set keys or
    send them across processes.
    """

    # ADDITIVE to the selected built-in tool profile (filesystem and execute;
    # delegation is selected independently below). () means "no extra tools",
    # not "no tools". Reaches
    # the constructed agent only (deepagents' auto-injected general-purpose
    # subagent — which used to inherit these — is disabled harness-wide;
    # see the profile registration in ``assist.agent``); the bespoke
    # context/research/critique subagents do not see these
    # (see ``create_agent``).
    tools: tuple[BaseTool | Callable | dict[str, Any], ...] = ()

    # Which Assist role this construction serves. ``main`` is the ordinary
    # user-facing supervisor. ``delegate`` is a whole-task worker built by the
    # same constructor; the web composition root gives it synchronous
    # specialists and withholds supervisor-only tools.
    role: Literal["main", "delegate"] = "main"

    # ADDITIVE skill routes: virtual path -> backend holding SKILL.md
    # trees, merged with built-in and domain skills.  Precedence on a
    # name collision is main guidance < caller guidance < main-only < domain <
    # built-in < embedder sources for the async main, caller guidance < domain
    # < built-in < embedder sources for delegates, and domain < built-in <
    # embedder sources otherwise (the deepagents listing is last-source-wins).
    # Re-passing the built-in SKILLS_ROUTE as a key overrides the built-in
    # backend.
    skill_sources: Mapping[str, BackendProtocol] = field(default_factory=dict)

    # The composite backend's DEFAULT ROUTE target — where non-routed
    # paths go — instead of a FilesystemBackend rooted at working_dir.
    # assist still wraps it with the standard STATEFUL_PATHS routing.
    # Mutually exclusive with the ``sandbox_backend`` param (validated
    # in ``create_agent``).
    default_backend: BackendProtocol | None = None

    # Deep Agents-compatible subagent selection surface for this graph.
    # ``None`` selects the established synchronous subagents for one-source legacy
    # embedders; thread-memory construction suppresses them so they cannot inherit
    # its private backend. An explicit empty sequence disables delegation (web
    # triage), and web main supplies all five lifecycle tools. The delegate profile
    # uses ``None`` for synchronous specialists. This field selects the delegation
    # mechanism without a second mode flag.
    async_subagent_tools: tuple[BaseTool, ...] | None = None

    # Closed identity for the ordinary visible web assistant.  It is not
    # inferred from tools or asynchronous delegation because embedders may
    # legitimately use those capabilities with a different prompt contract.
    # The prompt-rewrite composition seam uses this identity to leave legacy,
    # delegate, triage, and embedder agents unchanged.
    web_main: bool = False

    # Closed opt-in for the ordinary main assistant's progressive grounding and
    # research guidance. It selects only prompt-source material: the compact
    # Assist core and three read-only main-guidance skills, including the
    # direct-results research workflow. It stays separate from ``web_main`` so
    # evals can compare the pre-migration and candidate prompts with the same
    # production-shaped agent.
    main_guidance_skills: bool = False

    # Tools to gate with human-in-the-loop: a mapping ``{tool_name -> True |
    # InterruptOnConfig}`` installed as LangChain's HumanInTheLoopMiddleware. Web
    # profiles gate outward-effect tools (normal: ``send_email``; inbound triage:
    # ``send_reply``) so they can propose an action but never perform it without approval.
    # ``None`` = no HITL.
    interrupt_on: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        # The class is frozen; normalization goes through
        # object.__setattr__ by design.  Everything here is pure CPU.
        # The str/bytes half catches the certainly-wrong scalar that
        # tuple() would silently shred into characters.
        if isinstance(self.tools, (str, bytes)) \
                or not isinstance(self.tools, Sequence):
            raise TypeError(
                f"AgentSpec.tools must be a sequence of tools, got "
                f"{type(self.tools).__name__}"
            )
        object.__setattr__(self, "tools", tuple(self.tools))

        if self.role not in {"main", "delegate"}:
            raise ValueError(
                f"AgentSpec.role must be 'main' or 'delegate', got {self.role!r}")

        if self.async_subagent_tools is not None and (
                isinstance(self.async_subagent_tools, (str, bytes))
                or not isinstance(self.async_subagent_tools, Sequence)):
            raise TypeError(
                "AgentSpec.async_subagent_tools must be a sequence of tools, got "
                f"{type(self.async_subagent_tools).__name__}"
            )
        if self.async_subagent_tools is not None:
            object.__setattr__(self, "async_subagent_tools",
                               tuple(self.async_subagent_tools))

        if not isinstance(self.web_main, bool):
            raise TypeError(
                f"AgentSpec.web_main must be bool, got {type(self.web_main).__name__}")
        if self.web_main and (self.role != "main" or not self.async_subagent_tools):
            raise ValueError(
                "AgentSpec.web_main requires a main role with async lifecycle tools")
        if not isinstance(self.main_guidance_skills, bool):
            raise TypeError(
                "AgentSpec.main_guidance_skills must be bool, got "
                f"{type(self.main_guidance_skills).__name__}")
        if self.main_guidance_skills and not self.web_main:
            raise ValueError(
                "AgentSpec.main_guidance_skills requires web_main")

        if not isinstance(self.skill_sources, Mapping):
            raise TypeError(
                f"AgentSpec.skill_sources must be a mapping of route -> "
                f"backend, got {type(self.skill_sources).__name__}"
            )
        # Copy, then wrap read-only: the embedder mutating its own dict
        # later must not change the spec, and the spec must not be
        # mutable through this field either.
        object.__setattr__(
            self, "skill_sources", MappingProxyType(dict(self.skill_sources))
        )
