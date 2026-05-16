import os
import uuid
import logging

from deepagents import create_deep_agent, CompiledSubAgent
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
from assist.sandbox import DockerSandboxBackend
from assist.middleware.model_logging_middleware import ModelLoggingMiddleware
from assist.middleware.json_validation_middleware import JsonValidationMiddleware
from assist.middleware.tool_name_sanitization import ToolNameSanitizationMiddleware
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


def _create_standard_backend(working_dir: str):
    """Create the standard composite backend with state exclusions.

    This backend excludes ephemeral files like question.txt and large_tool_results/
    from the stateful filesystem, using StateBackend instead.
    """
    return create_composite_backend(working_dir, STATEFUL_PATHS)


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
                 sandbox_backend=None) -> CompiledStateGraph:
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
    loop_detection_mw = LoopDetectionMiddleware()
    # Innermost wrap_model_call middleware — recovers from empty terminal
    # AIMessages after every outer retry/sanitization layer has had its turn.
    empty_response_recovery_mw = EmptyResponseRecoveryMiddleware()

    # Note: context-aware compaction is delegated to deepagents 0.6.1's
    # built-in SummarizationMiddleware (trigger fraction=0.85, offloads
    # to /conversation_history/{thread_id}.md).  Per-result tool-output
    # eviction is delegated to deepagents' FilesystemMiddleware
    # (default 20k-token cap).  Our previous ContextAwareToolEvictionMiddleware
    # was redundant with both and was deleted on 2026-05-16 — see
    # docs/2026-05-16-context-management-overhaul.org.
    mw = [retry_middle, bad_request_mw, json_validation_mw, tool_name_mw,
          write_collision_mw, git_push_blocker_mw,
          loop_detection_mw, ThreadQueueMiddleware(), empty_response_recovery_mw]

    workspace_dir = sandbox_backend.work_dir if sandbox_backend else "/"
    # Single-slashed path that's safe to interpolate without producing
    # `//references/` in local mode (where workspace_dir == "/").
    references_dir = os.path.join(workspace_dir, "references")

    memories_path = os.path.join(workspace_dir, _MEMORY_FILE)

    if sandbox_backend:
        backend = create_sandbox_composite_backend(sandbox_backend)
    else:
        backend = _create_standard_backend(working_dir)

    skills_mw = SmallModelSkillsMiddleware(backend=backend, sources=[SKILLS_ROUTE])
    memory_mw = SmallModelMemoryMiddleware(backend=backend, memories_path=memories_path)

    context_sub = CompiledSubAgent(
        name="context-agent",
        description="Discovers and surfaces relevant context from the user's local filesystem. Use this agent to find files, read content, and understand the user's file structure before taking action. It is read-only — it will not modify files.",
        runnable=create_context_agent(model,
                                      working_dir,
                                      checkpointer,
                                      [retry_middle, json_validation_mw, tool_name_mw],
                                      sandbox_backend=sandbox_backend)
    )

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
    )

    return agent

def create_context_agent(model: BaseChatModel,
                         working_dir: str,
                         checkpointer=None,
                         middleware=[],
                         sandbox_backend=None) -> RollbackRunnable:
    """Create a read-only context agent for codebase exploration.

    Returns a RollbackRunnable-wrapped agent — on BadRequestError the agent
    rolls back to a previous checkpoint rather than crashing.  This is safe
    because the context-agent is read-only (no filesystem side effects).
    """
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
    base_mw.append(LoopDetectionMiddleware())
    base_mw.append(ThreadQueueMiddleware())
    base_mw.append(EmptyResponseRecoveryMiddleware())
    # Enforce the read-only contract at the tool layer.
    base_mw.append(ReadOnlyEnforcerMiddleware())

    if sandbox_backend:
        backend = create_sandbox_composite_backend(sandbox_backend)
    else:
        backend = _create_standard_backend(working_dir)
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
    # Rewrite write_file collision errors before loop detection sees them —
    # research-agent is the most likely path to hit the filename-mutation
    # trap (multi-pass critique → "I have completed the research" → another
    # write_file).
    base_mw.append(WriteCollisionMiddleware())
    base_mw.append(LoopDetectionMiddleware())
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
        # Sibling sandbox rooted at /workspace/references.  ``strip_prefixes``
        # flattens any agent-supplied ``references/`` (or ``/references/``)
        # so writes don't nest under the already-references-rooted workdir.
        references_sandbox = DockerSandboxBackend(
            sandbox_backend.container,
            work_dir=sandbox_backend.work_dir + "/references",
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
                LoopDetectionMiddleware(),
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

    agent = create_deep_agent(
        model=model,
        tools=[search_internet, read_url],
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
    # uses the *parent* sandbox (work_dir=/workspace), not the references
    # sibling, so its `mkdir -p /workspace/references` runs with a
    # workdir that exists on the first call.
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


