"""Middleware to infer missing subagent_type in task tool calls.

Ministral-3-8B reliably fills `description` but omits `subagent_type`.
This middleware repairs those calls before the tool node fires, using the
same model_copy() + return-new-state pattern as ToolNameSanitizationMiddleware.
"""
import logging
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

logger = logging.getLogger(__name__)

# Keyword → subagent name mapping (order matters: first match wins).
# More specific/longer phrases before shorter ones to reduce false positives.
_KEYWORD_MAP: list[tuple[list[str], str]] = [
    (
        ["context-agent", "context agent", "filesystem", "file structure",
         "project structure", "discover", "explore the", "find files",
         "read the", "local files", "directory", "codebase structure",
         "analyze the", "inspect the", "identify", "locate", "examine"],
        "context-agent",
    ),
    (
        ["research-agent", "research agent", "search the web", "internet",
         "best practice", "how to", "what is", "recommend", "token bucket",
         "algorithm", "external knowledge", "look up", "investigate",
         "research the", "research how", "learn about", "find out"],
        "research-agent",
    ),
    (
        ["dev-agent", "dev agent", "implement", "write code", "fix bug",
         "add feature", "write tests", "refactor", "debug", "deploy"],
        "dev-agent",
    ),
]


def _infer_subagent_type(description: str, valid_types: set[str]) -> str | None:
    """Return the best-matching subagent name from description keywords, or None."""
    lower = description.lower()
    for keywords, name in _KEYWORD_MAP:
        if name not in valid_types:
            continue
        if any(kw in lower for kw in keywords):
            return name
    return None


class SubagentTypeInferenceMiddleware(AgentMiddleware):
    """Repair task tool calls where subagent_type is missing or empty.

    Uses the same model_copy() + return-new-state pattern as
    ToolNameSanitizationMiddleware so additional_kwargs stays in sync
    with the patched tool_calls list.

    Args:
        valid_subagent_types: Names of subagents registered with this agent.
        default_subagent_type: Fallback when keyword inference finds no match.
    """

    def __init__(
        self,
        valid_subagent_types: set[str],
        default_subagent_type: str = "general-purpose",
    ):
        self.valid_subagent_types = valid_subagent_types
        self.default_subagent_type = default_subagent_type

    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        messages = state.get("messages", [])
        if not messages:
            return None

        last = messages[-1]
        if not isinstance(last, AIMessage) or not getattr(last, "tool_calls", None):
            return None

        patched_calls = []
        modified = False

        for tc in last.tool_calls:
            if tc.get("name") != "task":
                patched_calls.append(tc)
                continue

            args = dict(tc.get("args") or {})
            subagent_type = args.get("subagent_type") or ""

            if subagent_type and subagent_type in self.valid_subagent_types:
                patched_calls.append(tc)
                continue

            description = args.get("description", "")
            inferred = _infer_subagent_type(description, self.valid_subagent_types)
            chosen = inferred or self.default_subagent_type

            logger.warning(
                "SubagentTypeInference: task call had subagent_type=%r, "
                "inferred %r from description %.80r",
                subagent_type, chosen, description,
            )

            new_tc = dict(tc)
            new_tc["args"] = {**args, "subagent_type": chosen}
            patched_calls.append(new_tc)
            modified = True

        if not modified:
            return None

        new_last = last.model_copy() if hasattr(last, "model_copy") else last.copy()
        new_last.tool_calls = patched_calls

        # Keep additional_kwargs in sync (same as ToolNameSanitizationMiddleware)
        if hasattr(new_last, "additional_kwargs"):
            ak_calls = new_last.additional_kwargs.get("tool_calls")
            if ak_calls:
                patched_ids = {tc["id"] for tc in patched_calls}
                new_last.additional_kwargs = dict(new_last.additional_kwargs)
                new_last.additional_kwargs["tool_calls"] = [
                    tc for tc in ak_calls if tc.get("id") in patched_ids
                ]

        return {"messages": messages[:-1] + [new_last]}
