"""Middleware to sanitize invalid tool call names.

Small models (e.g. vLLM-served Ministral) sometimes hallucinate tool calls
with names that violate the OpenAI function-name spec (a-zA-Z0-9_- , max 64
chars).  For example, when search results return ``[]`` the model may emit a
tool call whose name is literally ``[]``.

This causes two problems:
1. LangGraph's tool node fails with "[] is not a valid tool".
2. The invalid tool call stays in the conversation history and vLLM rejects the
   next request with a 400 ("Function name was [] but must be …").

The middleware fixes both problems:
* **after_model** – strips tool calls with invalid names from the model's
  response *before* LangGraph routes to the tool node.
* **before_model** – strips any leftover invalid tool calls (and their
  corresponding tool-result messages) from the conversation history so they
  never reach the model again.
"""

import logging
import re
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.runtime import Runtime

logger = logging.getLogger(__name__)

# OpenAI / vLLM function-name regex
_VALID_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


def _is_valid_tool_name(name: str | None) -> bool:
    if not name:
        return False
    return _VALID_NAME_RE.match(name) is not None


class ToolNameSanitizationMiddleware(AgentMiddleware):
    """Strip tool calls whose names are not valid OpenAI function names."""

    # ------------------------------------------------------------------
    # after_model: intercept the *new* AI message
    # ------------------------------------------------------------------
    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        messages = state.get("messages", [])
        if not messages:
            return None

        last = messages[-1]

        # Only inspect AI messages that carry tool calls
        if not isinstance(last, AIMessage):
            return None
        if not getattr(last, "tool_calls", None):
            return None

        valid_calls = []
        stripped = []

        for tc in last.tool_calls:
            name = tc.get("name") or ""
            if _is_valid_tool_name(name):
                valid_calls.append(tc)
            else:
                stripped.append(name)

        if not stripped:
            return None  # everything was fine

        logger.warning(
            "Stripped %d invalid tool call(s) from model response: %s",
            len(stripped),
            stripped,
        )

        # Rebuild the AI message without the invalid calls
        new_last = last.model_copy() if hasattr(last, "model_copy") else last.copy()
        new_last.tool_calls = valid_calls

        # Also clean additional_kwargs which carries the raw OpenAI format
        if hasattr(new_last, "additional_kwargs"):
            ak_calls = new_last.additional_kwargs.get("tool_calls")
            if ak_calls:
                valid_ids = {tc["id"] for tc in valid_calls}
                new_last.additional_kwargs = dict(new_last.additional_kwargs)
                new_last.additional_kwargs["tool_calls"] = [
                    tc for tc in ak_calls if tc.get("id") in valid_ids
                ]

        return {"messages": messages[:-1] + [new_last]}

    # ------------------------------------------------------------------
    # before_model: scrub history of any past invalid tool calls
    # ------------------------------------------------------------------
    def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        messages = state.get("messages", [])
        if not messages:
            return None

        # First pass: collect tool-call IDs to remove
        bad_ids: set[str] = set()

        for msg in messages:
            if not isinstance(msg, AIMessage):
                continue
            for tc in getattr(msg, "tool_calls", None) or []:
                name = tc.get("name") or ""
                if not _is_valid_tool_name(name):
                    tc_id = tc.get("id")
                    if tc_id:
                        bad_ids.add(tc_id)

        if not bad_ids:
            return None

        # Second pass: rebuild messages, stripping bad calls and their results
        cleaned: list = []

        for msg in messages:
            if isinstance(msg, ToolMessage):
                if getattr(msg, "tool_call_id", None) in bad_ids:
                    continue  # drop this tool result

            if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                kept = [
                    tc for tc in msg.tool_calls
                    if tc.get("id") not in bad_ids
                ]
                if len(kept) != len(msg.tool_calls):
                    new_msg = msg.model_copy() if hasattr(msg, "model_copy") else msg.copy()
                    new_msg.tool_calls = kept

                    # Also clean additional_kwargs
                    if hasattr(new_msg, "additional_kwargs"):
                        ak_calls = new_msg.additional_kwargs.get("tool_calls")
                        if ak_calls:
                            valid_ids = {tc["id"] for tc in kept}
                            new_msg.additional_kwargs = dict(new_msg.additional_kwargs)
                            new_msg.additional_kwargs["tool_calls"] = [
                                tc for tc in ak_calls if tc.get("id") in valid_ids
                            ]

                    # If no tool calls remain and no text content, skip message
                    if not kept and not (new_msg.content and str(new_msg.content).strip()):
                        continue

                    cleaned.append(new_msg)
                    continue

            cleaned.append(msg)

        logger.warning(
            "Sanitized %d invalid tool call(s) from conversation history",
            len(bad_ids),
        )
        return {"messages": cleaned}
