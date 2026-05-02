"""Recover from empty terminal AIMessages.

The LangChain agent loop exits as soon as an AIMessage carries no tool
calls — see ``langchain/agents/factory.py:1722``::

    if len(last_ai_message.tool_calls) == 0:
        return end_destination

It does not check whether the message has any *content*.  When the model
returns ``AIMessage(content="", tool_calls=[])`` (or content that
contains only a Qwen3 ``<think>...</think>`` block, post-strip empty),
the agent terminates cleanly and the user gets back ``""``.

This was observed in production-shape eval:  ``test_finance_synthesis``
ran for 50 minutes, made 13 model calls, and ended with an empty
``response_preview``.  The bug class is *agent behavior* (the loop's
exit condition is permissive) rather than mode-specific (the
``thinking_off`` branch was never measured on that case).  This
middleware defends against the class.

Strategy — three stages:

1. **Detect** an empty terminal AIMessage in ``wrap_model_call``.
   "Empty" means: no tool calls AND, after stripping any
   ``<think>...</think>`` block plus surrounding whitespace, the
   content has zero characters.

2. **Retry once** with the request augmented by a short instruction
   asking the model to summarise its work.  The augmenting
   ``HumanMessage`` is local to this single handler call — it goes in
   via ``request.override(messages=...)`` and never enters checkpoint
   history because we only return the retry's ``result`` messages in
   the ``ModelResponse``.

3. **Fallback** if the retry is also empty: synthesise a brief
   ``AIMessage`` that cites the most recent successful
   ``write_file``/``edit_file`` artifact when one exists in the
   current turn (reusing ``loop_detection._last_successful_artifact``
   so the bounding semantics stay aligned).  When no artifact is
   visible, the fallback is honest about the failure rather than
   inventing a summary — synthesising fake content from the message
   tail invites hallucination, and the user can re-ask.

Composition notes:

- Position: **innermost** ``wrap_model_call`` middleware in every
  agent's stack.  Recovery should react to the final model response
  *after* outer middleware (``BadRequestRetryMiddleware``,
  deepagents' ``SummarizationMiddleware``) has had its turn.  If
  ``BadRequestRetryMiddleware`` exhausts retries it returns a
  synthetic ``AIMessage`` whose content is non-empty (its error
  template), so recovery does not fire on that path — by design.

- Loop detection (``after_model``) and recovery (``wrap_model_call``)
  do not compete on the same hook, so their relative list position
  only matters for readability.  We keep them adjacent so the
  family of "agent-behavior defenses" is grouped.

- Worst-case retry amplification: if recovery's retry call raises an
  exception and ``BadRequestRetryMiddleware`` (outer, max_retries=3)
  catches and re-invokes, recovery will re-run on each of those
  retries.  Worst case is bounded at ~(1 + max_retries) * (1 + outer
  retries) handler calls per model node — currently 2*4 = 8.  Not
  catastrophic, but worth knowing when reading a long log.

- Subagents have their own compiled graph + middleware stack.  Each
  subagent that should benefit from recovery installs its own
  instance — ``assist.agent`` does this for both ``context-agent``
  and ``research-agent``.  The shared chat model is fine because
  every model call flows through the *invoking* agent's middleware
  chain.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Callable

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage

# Import via the loop_detection module so any future tweaks to the
# turn-slicing semantics stay in one place.
from assist.middleware.loop_detection import _last_successful_artifact

logger = logging.getLogger(__name__)


# DOTALL so the regex spans newlines; IGNORECASE so we still strip an
# unusually-cased ``<Think>`` (Qwen3 emits lowercase, but defensively).
# Non-greedy so successive blocks don't merge into one giant match.
_THINK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)


_AUGMENT_INSTRUCTION = (
    "Your previous response had no content and no tool calls. "
    "Provide a concise (1-3 sentence) summary of the work you "
    "completed and the result you reached. Do not call any tools."
)


def _content_to_text(content: Any) -> str:
    """Normalise message content into a single string for length checks.

    Handles both ``str`` content and the multi-part ``list[dict|str]``
    shape that some providers emit.  Mirrors the dict-with-text-key
    handling in ``BadRequestRetryMiddleware._sanitize_message_content``.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                txt = part.get("text")
                if isinstance(txt, str):
                    parts.append(txt)
        return "".join(parts)
    return str(content)


def _strip_think(text: str) -> str:
    """Remove all ``<think>...</think>`` blocks.

    An *unclosed* ``<think>`` (truncated mid-trace) does NOT match,
    intentionally — falsely matching would let recovery swallow real
    content.  ``_is_empty_terminal`` only treats post-strip-zero as
    empty, so an unclosed-think message with at least one trailing
    character is treated (correctly) as non-empty.
    """
    return _THINK_RE.sub("", text)


def _is_empty_terminal(msg: AIMessage) -> bool:
    """True iff ``msg`` would terminate the agent loop with no useful text.

    Terminal: no tool calls (the loop exits when ``tool_calls`` is
    empty).  Empty: post ``<think>`` strip and whitespace strip, the
    content is the empty string.
    """
    if getattr(msg, "tool_calls", None):
        return False
    text = _content_to_text(getattr(msg, "content", None))
    return _strip_think(text).strip() == ""


def _first_ai_message(result: list) -> tuple[int, AIMessage] | None:
    """Find the first ``AIMessage`` in a ``ModelResponse.result`` list.

    The contract permits structured-output ``ToolMessage`` companions
    so we cannot index ``result[0]`` blindly.  Returns ``(index, msg)``
    or ``None`` if no AIMessage is present.
    """
    for i, m in enumerate(result):
        if isinstance(m, AIMessage):
            return i, m
    return None


def _compose_fallback(messages: list) -> str:
    """Synthesise the user-facing AIMessage for the fallback path.

    Two-sentence cap.  Honest about the failure mode without
    apologising profusely.  Cites a successful artifact when available
    (matching the ``loop_detection._compose_terminal_message``
    pattern).
    """
    artifact = _last_successful_artifact(messages)
    if artifact:
        return (
            f"I completed the work and saved the result to `{artifact}`. "
            "Let me know if you'd like changes or follow-up."
        )
    return (
        "I wasn't able to produce a final summary for this turn. "
        "The work history is in the conversation; please ask for "
        "the specific result you need (e.g. the projected number, "
        "the file path, the next step)."
    )


class EmptyResponseRecoveryMiddleware(AgentMiddleware):
    """Catch empty terminal AIMessages, retry once, then synthesise.

    Args:
        max_retries: Number of additional handler calls after the
            first empty response.  Default 1.  ``0`` disables retry
            and goes straight to fallback synthesis on detection.
    """

    def __init__(self, max_retries: int = 1):
        super().__init__()
        self.max_retries = max_retries
        self.tools = []
        self._intervention_count = 0
        self._fallback_count = 0

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse | AIMessage],
    ) -> ModelResponse | AIMessage:
        response = handler(request)

        # Both AIMessage and ModelResponse are valid handler returns.
        # Normalise to (ai_msg, ai_idx, result_list, structured).
        if isinstance(response, AIMessage):
            ai = response
            result_list: list = [ai]
            ai_idx = 0
            structured = None
        else:
            structured = getattr(response, "structured_response", None)
            result_list = list(response.result)
            found = _first_ai_message(result_list)
            if found is None:
                # Unusual shape — nothing we can do.
                return response
            ai_idx, ai = found

        if not _is_empty_terminal(ai):
            return response

        self._intervention_count += 1
        logger.warning(
            "EmptyResponseRecovery: intervention #%d — empty terminal "
            "AIMessage detected (content_len=%d); retrying with "
            "augmenting instruction (max_retries=%d)",
            self._intervention_count,
            len(_content_to_text(ai.content)),
            self.max_retries,
        )

        # Retry path — augmenter is local to each handler call.  Any
        # exception (e.g. BadRequestError) propagates to the outer
        # middleware, which is the intended composition.
        for attempt in range(self.max_retries):
            augmented = request.override(
                messages=list(request.messages) + [
                    HumanMessage(content=_AUGMENT_INSTRUCTION),
                ]
            )
            retry_response = handler(augmented)

            if isinstance(retry_response, AIMessage):
                retry_ai = retry_response
            else:
                found = _first_ai_message(list(retry_response.result))
                if found is None:
                    continue
                _, retry_ai = found

            if not _is_empty_terminal(retry_ai):
                logger.info(
                    "EmptyResponseRecovery: intervention #%d recovered "
                    "on retry %d (new content_len=%d)",
                    self._intervention_count,
                    attempt + 1,
                    len(_content_to_text(retry_ai.content)),
                )
                # Return retry_response directly so structured_response
                # and any companion ToolMessages are preserved as-is.
                return retry_response

        # Fallback synthesis.
        self._fallback_count += 1
        fallback_text = _compose_fallback(request.messages)
        preview = fallback_text.replace("\n", " ")
        if len(preview) > 160:
            preview = preview[:159] + "…"
        logger.error(
            "EmptyResponseRecovery: fallback #%d — retry exhausted, "
            "synthesising response: %r",
            self._fallback_count,
            preview,
        )

        synth = AIMessage(content=fallback_text)
        if isinstance(response, AIMessage):
            return synth
        synth_result = list(result_list)
        synth_result[ai_idx] = synth
        return ModelResponse(result=synth_result, structured_response=structured)
