"""The embedder contract: what a client declares about its agent.

``AgentSpec`` is the single declaration surface for embedders (the dev
web app, emacsos-server, a future CLI) — it replaces the per-need
kwargs that used to accrete on ``Thread`` / ``create_agent``
(``extra_tools``, ``extra_skill_sources``, ``default_backend``).  See
docs/2026-06-11-embedder-contract.org for the design and the split
rule: *spec = the agent's shape, consumed by create_agent; Thread
kwargs = per-instance and per-run wiring* (identity, persistence,
model, concurrency, status callback, per-request ``configurable``).

Admission rule: a field requires a real, existing client need.  Do not
add fields for needs no client has yet — deferred candidates
(``subagents`` selection, ``system_prompt``, middleware tuning) are
recorded in the design doc with the trigger that revives them.
"""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

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
    """

    # ADDITIVE to assist's built-in tool surface (filesystem, execute,
    # task, ...) — () means "no extra tools", not "no tools".  Reaches
    # the main agent and the auto-injected general-purpose subagent;
    # the bespoke context/research/critique subagents do not see these
    # (see ``create_agent``).
    tools: tuple[BaseTool | Callable | dict[str, Any], ...] = ()

    # ADDITIVE skill routes: virtual path -> backend holding SKILL.md
    # trees, merged with built-in and domain skills.  Precedence on a
    # name collision is domain < built-in < embedder sources (the
    # deepagents listing is last-source-wins).  Re-passing the built-in
    # SKILLS_ROUTE as a key overrides the built-in backend.
    skill_sources: Mapping[str, BackendProtocol] = field(default_factory=dict)

    # The composite backend's DEFAULT ROUTE target — where non-routed
    # paths go — instead of a FilesystemBackend rooted at working_dir.
    # assist still wraps it with the standard STATEFUL_PATHS routing.
    # Mutually exclusive with the ``sandbox_backend`` param (validated
    # in ``create_agent``).
    default_backend: BackendProtocol | None = None

    def __post_init__(self) -> None:
        # The class is frozen; normalization goes through
        # object.__setattr__ by design.  Everything here is pure CPU.
        if isinstance(self.tools, (str, bytes)):
            # tuple("ab") silently becomes ("a", "b") — catch the
            # certainly-wrong scalar instead of producing nonsense.
            raise TypeError(
                f"AgentSpec.tools must be a sequence of tools, got "
                f"{type(self.tools).__name__}"
            )
        if not isinstance(self.tools, Sequence):
            raise TypeError(
                f"AgentSpec.tools must be a sequence of tools, got "
                f"{type(self.tools).__name__}"
            )
        object.__setattr__(self, "tools", tuple(self.tools))

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
