import os
import uuid
import logging
from typing import Any, Callable, Sequence

from deepagents import create_deep_agent, CompiledSubAgent
from langchain_core.tools import BaseTool
from deepagents.backends.protocol import BackendProtocol
from langchain.messages import AIMessage, AnyMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph
from langchain.agents.middleware import ModelRetryMiddleware
from openai import APIConnectionError, InternalServerError

from assist.promptable import base_prompt_for
from assist.tools import read_url, search_internet
from assist.backends import create_composite_backend, create_sandbox_composite_backend, create_references_backend, STATEFUL_PATHS, SKILLS_ROUTE
from assist.checkpoint_rollback import invoke_with_rollback, RollbackRunnable
from assist.research_cleanup import ReferencesCleanupRunnable
from assist.sandbox import DockerSandboxBackend, SandboxContainerLostError
from docker.errors import NotFound as DockerNotFound
from assist.middleware.model_logging_middleware import ModelLoggingMiddleware
from assist.middleware.json_validation_middleware import JsonValidationMiddleware
from assist.middleware.tool_name_sanitization import ToolNameSanitizationMiddleware
from assist.middleware.output_sanitization import OutputSanitizationMiddleware
from assist.middleware.bad_request_retry import BadRequestRetryMiddleware
from assist.middleware.loop_detection import LoopDetectionMiddleware
from assist.middleware.empty_response_recovery import EmptyResponseRecoveryMiddleware
from assist.middleware.read_only_enforcer import ReadOnlyEnforcerMiddleware
from assist.middleware.git_push_blocker import GitPushBlockerMiddleware
from assist.middleware.skills_middleware import SmallModelSkillsMiddleware
from assist.middleware.memory_middleware import SmallModelMemoryMiddleware
from assist.middleware.write_collision import WriteCollisionMiddleware
from assist.middleware.thread_queue_middleware import ThreadQueueMiddleware
from assist.env import env_int


logger = logging.getLogger(__name__)


def _create_standard_backend(working_dir: str,
                             extra_routes: dict[str, BackendProtocol] | None = None,
                             default_backend: BackendProtocol | None = None,
                             ):
    """Create the standard composite backend with state exclusions.

    This backend excludes ephemeral files like question.txt and large_tool_results/
    from the stateful filesystem, using StateBackend instead.  ``extra_routes``
    is threaded through to ``create_composite_backend`` for embedders that
    need to register additional virtual-path routes (e.g. external skill dirs).
    ``default_backend``, if given, replaces the FilesystemBackend default (see
    ``create_composite_backend``) so an embedder can supply a custom default
    while keeping the standard STATEFUL_PATHS routing.
    """
    return create_composite_backend(working_dir, STATEFUL_PATHS,
                                    extra_routes=extra_routes,
                                    default_backend=default_backend)


# Single source of truth for the retry-on tuple.  When adding a new
# transient exception type to retry, change it here and every agent /
# sub-agent factory picks it up.  Previously the tuple was duplicated
# across `create_agent` and the dict-spec sub-agents — commit 1264c77
# updated the top-level but missed the sub-agents, which left the
# vegan-pizza thread (2026-05-03 18:09) unprotected against an
# `APITimeoutError` (a subclass of `APIConnectionError`) inside a
# sub-research-agent loop.
def _make_retry_middleware():
    # Default 3 retries (4 total attempts).  Overridable via
    # ASSIST_LLM_MAX_RETRIES so an operator can tighten this without
    # a redeploy when an endpoint outage means retrying just spends
    # wall-clock budget before the inevitable failure.  Bad values
    # silently fall back to the default — see ``env_int``.
    return ModelRetryMiddleware(
        max_retries=env_int("ASSIST_LLM_MAX_RETRIES", 3),
        retry_on=(APIConnectionError, InternalServerError, TimeoutError, ConnectionError),
        backoff_factor=2,
    )


class AgentHarness:
    """Makes it easier to have conversations"""
    
    def __init__(self, agent: CompiledStateGraph, thread_id: str | None = None):
        self.agent = agent
        self.thread_id = thread_id or uuid.uuid1()

    def message(self, text: str) -> AIMessage:
        resp = invoke_with_rollback(
            self.agent,
            {"messages": [{"role": "user", "content": text}]},
            {
                "configurable": {"thread_id": self.thread_id},
                "recursion_limit": 5000,
            },
        )
        return resp["messages"][-1].content

    def all_messages(self) -> list[AnyMessage]:
        state = self.agent.get_state({
            "configurable": {"thread_id": self.thread_id}
        })
        return state.values.get("messages", [])
        



_MEMORY_FILE = "AGENTS.md"


def create_agent(model: BaseChatModel,
                 working_dir: str,
                 checkpointer=None,
                 sandbox_backend=None,
                 extra_skill_sources: dict[str, BackendProtocol] | None = None,
                 extra_tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None,
                 loop_exploration_tools: frozenset[str] | None = None,
                 default_backend: BackendProtocol | None = None,
                 ) -> CompiledStateGraph:
    """Build the general-purpose agent.

    ``extra_skill_sources`` is a mapping of additional virtual-path
    routes to backends that hold ``SKILL.md`` files.  Embedders (e.g.
    emacsos-server) use this to register skills that live outside the
    assist repo.  Each route is registered with the composite backend
    AND added to ``SmallModelSkillsMiddleware``'s sources list, so the
    routed files are reachable AND the middleware knows to list them
    as available skills.  Default ``None`` preserves the current
    behavior of loading skills only from ``SKILLS_ROUTE``.  An
    embedder that explicitly re-passes ``SKILLS_ROUTE`` as a key gets
    its backend swapped in (intentional override mechanism); the
    sources list de-duplicates so the middleware doesn't scan twice.

    ``extra_tools`` is a sequence of ``BaseTool | Callable | dict[str,
    Any]`` passed through to ``create_deep_agent(tools=...)`` — the
    additive surface for embedder-supplied tools the main agent can
    call.  Reaches the main agent and the auto-injected
    general-purpose subagent; the bespoke ``context`` / ``research`` /
    ``critique`` subagents are built separately and do not see these
    tools (correctly, given their roles).  Default ``None`` is empty.

    ``loop_exploration_tools`` is forwarded to the MAIN agent's
    ``LoopDetectionMiddleware(exploration_tools=...)``: tool names whose
    distinct-args breadth gets a higher Pattern-C threshold (they probe
    many forms legitimately — e.g. emacsos's ``eval_elisp``).  They are
    NOT exempt from loop detection (still subject to Patterns A/B and a
    finite Pattern-C cap).  Only the main agent gets it — the bespoke
    subagents don't receive ``extra_tools`` so the exploration tool can't
    reach them.  Default ``None`` leaves the dev/code agent unchanged.

    ``default_backend`` lets an embedder supply the composite backend's
    *default* — the target for every non-routed path — instead of the
    ``FilesystemBackend`` rooted at ``working_dir``.  assist still wraps it
    with the standard STATEFUL_PATHS -> ``StateBackend`` routing (so
    summarization/scratch stay ephemeral) and any ``extra_skill_sources``.
    Mutually exclusive with ``sandbox_backend``.  If the supplied backend
    implements ``SandboxBackendProtocol``, deepagents' ``supports_execution``
    enables the ``execute`` tool for it automatically.  This affects only
    the MAIN agent; the context/research subagents build their own backends
    from ``working_dir`` (+ ``sandbox_backend``) as before.  Default ``None``
    preserves the FilesystemBackend default.
    """
    if sandbox_backend is not None and default_backend is not None:
        raise ValueError(
            "create_agent: pass sandbox_backend OR default_backend, not both")
    # Core middleware: retry, tool call limiting, JSON validation, and logging.
    # See `_make_retry_middleware` for the retry-on tuple rationale.
    retry_middle = _make_retry_middleware()
    # Catch BadRequestError (e.g. context overflow), sanitize & truncate, retry.
    bad_request_mw = BadRequestRetryMiddleware(max_retries=3)
    # Validate and fix JSON in tool call arguments
    json_validation_mw = JsonValidationMiddleware(strict=False)
    # Strip tool calls with invalid names (e.g. '[]' hallucinated by small models)
    tool_name_mw = ToolNameSanitizationMiddleware()
    logging_mw = ModelLoggingMiddleware("general-agent")

    # Rewrite write_file collision errors so the small model is redirected to
    # edit_file instead of inventing a new filename.  Must run before
    # loop_detection_mw so the rewritten error is what the loop detector sees.
    write_collision_mw = WriteCollisionMiddleware()
    # Reject `git push` invocations from the agent's `execute` tool —
    # the agent must not be able to publish to origin; pushes go
    # through the web UI's "Push to origin" button only.  Sits ahead
    # of `loop_detection_mw` so the rejection is what the loop
    # detector sees if the model retries.
    git_push_blocker_mw = GitPushBlockerMiddleware()
    # subagent_dispatch_threshold caps re-dispatch of the same sub-agent
    # (context / research / critique) to once — the general-agent prompt's
    # "call each sub-agent ONCE" made deterministic.  Stops the general
    # agent from re-dispatching the research orchestrator, which multiplies
    # the inner search volume, and reinforces the #119 rate-limit handoff.
    loop_detection_mw = LoopDetectionMiddleware(
        exploration_tools=loop_exploration_tools,
        subagent_dispatch_threshold=_SUBAGENT_DISPATCH_CAP,
    )
    # Innermost wrap_model_call middleware — recovers from empty terminal
    # AIMessages after every outer retry/sanitization layer has had its turn.
    empty_response_recovery_mw = EmptyResponseRecoveryMiddleware()

    # Note: context-aware compaction is delegated to deepagents 0.6.1's
    # built-in SummarizationMiddleware (trigger fraction=0.85, offloads
    # to /conversation_history/{thread_id}.md).  Per-result tool-output
    # eviction is delegated to deepagents' FilesystemMiddleware
    # (default 20k-token cap).  Our previous ContextAwareToolEvictionMiddleware
    # was redundant with both and was deleted on 2026-05-16 — see
    # docs/2026-05-16-context-management-overhaul.org.  We kept its
    # ANSI/control-char sanitization in OutputSanitizationMiddleware
    # (proactive, before content lands in state).
    mw = [retry_middle, bad_request_mw, json_validation_mw, tool_name_mw,
          OutputSanitizationMiddleware(),
          write_collision_mw, git_push_blocker_mw,
          loop_detection_mw, ThreadQueueMiddleware(), empty_response_recovery_mw]

    workspace_dir = sandbox_backend.work_dir if sandbox_backend else "/"
    # Single-slashed path that's safe to interpolate without producing
    # `//references/` in local mode (where workspace_dir == "/").
    references_dir = os.path.join(workspace_dir, "references")

    memories_path = os.path.join(workspace_dir, _MEMORY_FILE)

    if sandbox_backend:
        backend = create_sandbox_composite_backend(sandbox_backend,
                                                   extra_routes=extra_skill_sources)
    else:
        backend = _create_standard_backend(working_dir,
                                           extra_routes=extra_skill_sources,
                                           default_backend=default_backend)

    skill_sources = [SKILLS_ROUTE]
    if extra_skill_sources:
        # De-dupe: an embedder that re-passes SKILLS_ROUTE as a key
        # has overridden the built-in *backend* (the route map
        # update wins), but shouldn't make the middleware scan the
        # same prefix twice.
        skill_sources.extend(
            k for k in extra_skill_sources if k != SKILLS_ROUTE
        )
    skills_mw = SmallModelSkillsMiddleware(backend=backend, sources=skill_sources)
    memory_mw = SmallModelMemoryMiddleware(backend=backend, memories_path=memories_path)

    context_sub = CompiledSubAgent(
        name="context-agent",
        description="Discovers and surfaces relevant context from the user's local filesystem. Use this agent to find files, read content, and understand the user's file structure before taking action. It is read-only — it will not modify files.",
        runnable=create_context_agent(model,
                                      working_dir,
                                      checkpointer,
                                      [retry_middle, json_validation_mw, tool_name_mw],
                                      sandbox_backend=sandbox_backend,
                                      default_backend=default_backend)
    )

    # NOTE: research-agent is confined to <working_dir>/references/ via
    # `create_references_backend` and does NOT inherit `default_backend`.
    # For embedders using `default_backend` (e.g. emacsos file-chat), the
    # research-agent's report writes land on the SERVER's working_dir, not
    # the embedder's filesystem.  Acceptable v1 — research is rare and
    # reports are server-side documents.  When file-chat starts needing
    # research reports on the embedder's FS, add `default_backend` to
    # `create_research_agent` and to `create_references_backend`'s routing.
    research_sub = CompiledSubAgent(
        name="research-agent",
        description=(
            "Used to conduct thorough research on external topics. "
            f"Reports are saved under '{references_dir}/'. "
            "The result of the research will be placed in a file and the file "
            "name/path will be returned. Provide a filename for more control."
        ),
        runnable=create_research_agent(model,
                                       working_dir,
                                       checkpointer,
                                       [retry_middle, json_validation_mw, tool_name_mw],
                                       sandbox_backend=sandbox_backend)
    )

    critique_sub_agent = {
        "name": "critique-agent",
        "description": "Reviews code diffs for bugs, missing tests, style issues, and security concerns. Provide the full git diff output when calling this agent.",
        "system_prompt": base_prompt_for("deepagents/dev_critique.md.j2",
                                         workspace_dir=workspace_dir),
        # Safety middleware — same rationale as the research-flow dict
        # subagents.  Includes the retry layers so a transient
        # APIConnectionError (incl. APITimeoutError) inside the critique
        # call doesn't kill the parent thread the way it killed the
        # vegan-pizza thread on 2026-05-03 (which lost a sub-research-agent
        # call to an unretried APITimeoutError).
        "middleware": [_make_retry_middleware(),
                       BadRequestRetryMiddleware(max_retries=3),
                       OutputSanitizationMiddleware(),
                       LoopDetectionMiddleware(),
                       EmptyResponseRecoveryMiddleware()],
    }

    agent = create_deep_agent(
        model=model,
        checkpointer=checkpointer or InMemorySaver(),
        system_prompt=base_prompt_for(
            "deepagents/general_instructions.md.j2",
            workspace_dir=workspace_dir,
            references_dir=references_dir,
        ),
        middleware=mw + [skills_mw, memory_mw, logging_mw],
        backend=backend,
        subagents=[context_sub, research_sub, critique_sub_agent],
        tools=list(extra_tools) if extra_tools else [],
    )

    return agent

def create_context_agent(model: BaseChatModel,
                         working_dir: str,
                         checkpointer=None,
                         middleware=[],
                         sandbox_backend=None,
                         default_backend: BackendProtocol | None = None,
                         ) -> RollbackRunnable:
    """Create a read-only context agent for codebase exploration.

    Returns a RollbackRunnable-wrapped agent — on BadRequestError the agent
    rolls back to a previous checkpoint rather than crashing.  This is safe
    because the context-agent is read-only (no filesystem side effects).

    ``default_backend`` mirrors the parent ``create_agent`` parameter: when
    an embedder supplies a custom default (e.g. emacsos's EmacsBackend for
    file-backed chat), the subagent inherits it.  Without this plumbing the
    parent agent's filesystem changes wouldn't be visible to the subagent
    that does most of the "find files" work — see the bug surfaced on
    2026-05-28 where file-chat's context-agent listed StateBackend paths
    (/skills/) instead of the phone's workdir.  Mutually exclusive with
    ``sandbox_backend`` at the parent level; ignored when sandbox_backend
    is set."""
    # Only add JSON validation if not already provided
    has_json_validation = any(isinstance(m, JsonValidationMiddleware) for m in middleware)

    workspace_dir = sandbox_backend.work_dir if sandbox_backend else "/"

    base_mw = []
    if not has_json_validation:
        base_mw.append(JsonValidationMiddleware(strict=False))

    # Catch BadRequestError, sanitize & truncate messages, retry.
    base_mw.append(BadRequestRetryMiddleware(max_retries=3))
    # Context compaction delegated to deepagents' SummarizationMiddleware
    # (auto-installed by create_deep_agent at fraction=0.85).  Per-result
    # eviction delegated to deepagents' FilesystemMiddleware (20k cap).
    # Proactive ANSI/control-char strip from tool output:
    base_mw.append(OutputSanitizationMiddleware())
    base_mw.append(LoopDetectionMiddleware())
    base_mw.append(ThreadQueueMiddleware())
    base_mw.append(EmptyResponseRecoveryMiddleware())
    # Enforce the read-only contract at the tool layer.
    base_mw.append(ReadOnlyEnforcerMiddleware())

    if sandbox_backend:
        backend = create_sandbox_composite_backend(sandbox_backend)
    else:
        backend = _create_standard_backend(working_dir,
                                           default_backend=default_backend)
    logging_mw = ModelLoggingMiddleware("context-agent")

    agent = create_deep_agent(
        model=model,
        checkpointer=checkpointer or InMemorySaver(),
        system_prompt=base_prompt_for("deepagents/context_agent.md.j2",
                                      workspace_dir=workspace_dir),
        backend=backend,
        middleware=base_mw + middleware + [logging_mw],
    )

    # 500 graph steps ≈ 45 model calls with deepagents' ~11 nodes per cycle.
    return RollbackRunnable(agent, recursion_limit=500)


# LoopDetection Pattern-E volume cap for the research flow.  The small
# model reads "conduct thorough research" as "search dozens of times"
# (prod: 50-100 search_internet calls for one trivial query), which the
# args/error-based loop patterns don't catch because each query is
# distinct and succeeds.  This caps any single tool (search_internet,
# task re-dispatch, read_url) to ~this many calls per agent within the
# detection window, then makes the agent finalize with what it has.
# Deliberately higher than a healthy research pass (~4 searches) so it
# only fires on genuine runaway, not normal multi-query exploration.
# Per-AGENT cap; combined with the per-subagent re-dispatch cap below
# (which holds the orchestrator to one research dispatch), this bounds the
# aggregate search volume to ~this many for a research turn.
_RESEARCH_TOOL_VOLUME_CAP = 6

# Pattern E only caps these tools.  Just search_internet — NOT read_url:
# reading several sources is normal research (and is per-host throttled +
# Pattern-D-protected against the 403 storm), and capping read_url stripped
# AI messages that batched a read with the report write, leaving no report.
_RESEARCH_VOLUME_TOOLS = frozenset({"search_internet"})

# Pattern F: max dispatches of any single subagent by an orchestrating
# agent.  1 = each subagent at most once — the budget both the general
# agent ("call each sub-agent ONCE") and the research orchestrator
# ("research once, critique once, fact-check once") already state in
# prose.  Made deterministic because the soft prompt rule did not hold:
# the model re-dispatched the research-agent ~3x (at the orchestrator)
# and, on some runs, re-dispatched the research orchestrator at the
# general-agent level — each multiplying the inner search volume.
# Enabled on BOTH the general agent and the research orchestrator.
_SUBAGENT_DISPATCH_CAP = 1


def create_research_agent(model: BaseChatModel,
                          working_dir: str,
                          checkpointer=None,
                          middleware=[],
                          sandbox_backend=None) -> RollbackRunnable:
    """Create a DeepAgents-based agent suitable for general-purpose research replies.

    Includes DuckDuckGo web search and a critique/research/fact-check subagent trio.

    Returns a RollbackRunnable-wrapped agent — on BadRequestError the agent
    rolls back to a previous checkpoint.  Research agents only write additive
    report files, so rollback is low-risk.
    """
    workspace_dir = sandbox_backend.work_dir if sandbox_backend else "/"

    # Only add JSON validation if not already provided
    has_json_validation = any(isinstance(m, JsonValidationMiddleware) for m in middleware)

    base_mw = []
    if not has_json_validation:
        base_mw.append(JsonValidationMiddleware(strict=False))

    # Catch BadRequestError, sanitize & truncate messages, retry.
    base_mw.append(BadRequestRetryMiddleware(max_retries=3))
    # Context compaction delegated to deepagents' SummarizationMiddleware
    # (auto-installed by create_deep_agent at fraction=0.85, with LLM-
    # summarization + offload to /conversation_history/{thread_id}.md).
    # Per-result tool-output eviction delegated to FilesystemMiddleware.
    # Proactive ANSI/control-char strip from tool output:
    base_mw.append(OutputSanitizationMiddleware())
    # Rewrite write_file collision errors before loop detection sees them —
    # research-agent is the most likely path to hit the filename-mutation
    # trap (multi-pass critique → "I have completed the research" → another
    # write_file).
    base_mw.append(WriteCollisionMiddleware())
    # The orchestrator gets BOTH the per-agent volume cap AND the
    # per-subagent re-dispatch cap (Pattern F): each subagent
    # (research / critique / fact-check) at most once, matching the
    # prompt's stated budget and deterministically stopping the
    # research-agent re-dispatch that multiplies inner search volume.
    base_mw.append(LoopDetectionMiddleware(
        volume_threshold=_RESEARCH_TOOL_VOLUME_CAP,
        volume_tools=_RESEARCH_VOLUME_TOOLS,
        subagent_dispatch_threshold=_SUBAGENT_DISPATCH_CAP,
    ))
    base_mw.append(ThreadQueueMiddleware())
    base_mw.append(EmptyResponseRecoveryMiddleware())

    # Confine the research agent's filesystem reach to <working_dir>/references/.
    # The agent (and its critique/fact-check sub-sub-agents, which inherit
    # this backend via deepagents' subagent middleware) cannot
    # accidentally resolve paths outside the references subdirectory in
    # normal use (local mode blocks `..` traversal; sandbox mode prepends
    # the prefix but does not normalize, see assist/research_cleanup.py).
    # Cleanup of intermediate files happens in `ReferencesCleanupRunnable`
    # outside the agent's loop.
    if sandbox_backend:
        # Pre-create the references dir eagerly — the sibling
        # ``DockerSandboxBackend`` below uses it as its ``work_dir``, which
        # Docker exec sets as ``chdir`` on every command.  If the dir
        # doesn't exist, every tool call from this agent (and its
        # sub-sub-agents that inherit this backend) crashes with
        # ``OCI runtime exec failed: chdir to cwd``.  Bit the 2026-05-16
        # winged-horse-flag thread.  Failure here raises ``RuntimeError``
        # back to the caller — the parent sandbox container is NOT torn
        # down (it belongs to the top-level agent, and a research-init
        # issue shouldn't kill the user's whole thread).  Idempotent
        # belt-and-suspenders alongside ``ReferencesCleanupRunnable._ensure_dir``
        # in local mode and as a lazy fallback in sandbox mode.
        references_path = sandbox_backend.work_dir + "/references"
        try:
            exit_code, output = sandbox_backend.container.exec_run(
                ["mkdir", "-p", references_path]
            )
        except DockerNotFound as e:
            # Container vanished between sandbox acquisition and research-
            # agent construction (TTL expiry, manual rm, daemon restart).
            # Translate to the typed error the web layer special-cases —
            # otherwise the raw docker exception escapes and the web layer
            # misses its dedicated cleanup + user-message path.  Matches
            # ``DockerSandboxBackend.execute``'s handling.
            raise SandboxContainerLostError(
                f"Sandbox container {sandbox_backend.container.id[:12]} "
                "disappeared before research-agent init — please retry."
            ) from e
        if exit_code != 0:
            output_str = output.decode("utf-8", errors="replace") if output else ""
            raise RuntimeError(
                f"Failed to create references dir {references_path!r} in "
                f"sandbox container {sandbox_backend.container.id[:12]}: "
                f"exit_code={exit_code} output={output_str!r}"
            )
        # Sibling sandbox rooted at /workspace/references.  ``strip_prefixes``
        # flattens any agent-supplied ``references/`` (or ``/references/``)
        # so writes don't nest under the already-references-rooted workdir.
        references_sandbox = DockerSandboxBackend(
            sandbox_backend.container,
            work_dir=references_path,
            strip_prefixes=("references",),
        )
        backend = create_sandbox_composite_backend(references_sandbox)
    else:
        backend = create_references_backend(working_dir)
    logging_mw = ModelLoggingMiddleware("research-agent")

    # Safety middleware installed on every dict-spec subagent below.
    # Includes the same retry/bad-request layers the top-level agent has,
    # so a transient APIConnectionError (incl. APITimeoutError) deep in a
    # sub-research-agent loop doesn't kill the parent thread.  Without
    # the retry layers, a 600s LLM timeout on a single sub-agent call
    # bubbles up as an unretried APITimeoutError and ends the thread —
    # observed on the vegan-pizza thread 2026-05-03 18:09 after 90 min
    # of work.  Without LoopDetection + EmptyResponseRecovery, the
    # subagent's compiled graph runs only the deepagents defaults
    # (TodoList, Filesystem, Summarization, PatchToolCalls), which once
    # left the fact-check-agent unbounded — it ran 200+ `read_url`
    # calls in a diag because nothing would short-circuit a model that
    # kept "thinking of more references to verify".
    def _subagent_safety_mw():
        return [_make_retry_middleware(),
                BadRequestRetryMiddleware(max_retries=3),
                # Strip ANSI from sub-tool output (read_url HTML can carry
                # raw escape sequences) before it lands in subagent state.
                OutputSanitizationMiddleware(),
                LoopDetectionMiddleware(volume_threshold=_RESEARCH_TOOL_VOLUME_CAP,
                                        volume_tools=_RESEARCH_VOLUME_TOOLS),
                ThreadQueueMiddleware(),
                EmptyResponseRecoveryMiddleware()]

    research_sub_agent = {
        "name": "research-agent",
        "description": "Used to research more in depth questions. Only give this researcher one topic at a time. It will return research results.",
        "system_prompt": base_prompt_for("deepagents/sub_research.txt.j2"),
        "tools": [search_internet, read_url],
        "middleware": _subagent_safety_mw(),
    }

    critique_sub_agent = {
        "name": "critique-agent",
        "description": "Used to critique the final report. You MUST provide the file it should critique.",
        "system_prompt": base_prompt_for("deepagents/sub_critique.txt.j2"),
        "middleware": _subagent_safety_mw(),
    }

    fact_check_sub_agent = {
        "name": "fact-check-agent",
        "description": "Used to check all references for alignment with claims and statements. You MUST provide the file it should fact-check.",
        "system_prompt": base_prompt_for("deepagents/fact_checker.md.j2"),
        "tools": [read_url],
        "middleware": _subagent_safety_mw(),
    }

    # The orchestrator DELEGATES searching to the research-agent (see its
    # prompt) — it does not search directly.  Giving it search_internet too
    # was redundant and doubled the over-search (orchestrator-direct +
    # inner agent both ran capped search passes).  It keeps read_url for
    # reading specific URLs while writing/fact-checking the report.
    agent = create_deep_agent(
        model=model,
        tools=[read_url],
        checkpointer=checkpointer or InMemorySaver(),
        system_prompt=base_prompt_for("deepagents/research_instructions.txt.j2",
                                      workspace_dir=workspace_dir),
        backend=backend,
        middleware=base_mw + middleware + [logging_mw],
        subagents=[critique_sub_agent,
                   research_sub_agent,
                   fact_check_sub_agent]
    )

    # 300 graph steps ≈ 150 model calls — research is multi-step but bounded.
    rollback_runnable = RollbackRunnable(agent, recursion_limit=300)

    # Wrap with the references-cleanup runnable so intermediate drafts
    # and sub-sub-agent scratch files get pruned after the research call
    # returns — only the final report stays in references/.  The wrapper
    # uses the *parent* sandbox (work_dir=/workspace) for its lazy
    # _ensure_dir fallback.  In sandbox mode the eager mkdir above is the
    # load-bearing creator; _ensure_dir is the load-bearing creator in
    # local mode (no container to exec_run against).
    if sandbox_backend:
        references_path = sandbox_backend.work_dir + "/references"
        cleanup_sandbox = sandbox_backend
    else:
        references_path = os.path.join(working_dir, "references")
        cleanup_sandbox = None
    return ReferencesCleanupRunnable(
        rollback_runnable,
        references_path=references_path,
        sandbox_backend=cleanup_sandbox,
    )


