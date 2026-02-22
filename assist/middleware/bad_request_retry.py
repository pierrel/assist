"""Middleware that retries model calls on BadRequestError with message sanitization.

When vLLM (or another provider) returns a 400 Bad Request — typically because
messages contain control characters, malformed JSON escapes, or other
unparseable content — this middleware catches the error, aggressively sanitizes
the request messages, and retries.

Unlike checkpoint rollback, this keeps the agent moving forward: no state is
lost and the agent can self-correct.  This is the preferred error-handling
strategy for agents that write to the filesystem or run commands (e.g. the
dev-agent), where rollback would leave orphaned side effects.

Usage:
    mw = BadRequestRetryMiddleware(max_retries=3)
    agent = create_deep_agent(model=model, middleware=[mw, ...])
"""
import json
import logging
import re
from typing import Any

from langchain_core.messages import AIMessage
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
)
from openai import BadRequestError

logger = logging.getLogger(__name__)


class BadRequestRetryMiddleware(AgentMiddleware):
    """Catch BadRequestError from the model API, sanitize messages, and retry.

    On each retry the middleware applies increasingly aggressive sanitization:

    1. Strip control characters and fix invalid JSON escapes in all messages.
    2. (Same sanitization — non-determinism alone may resolve it.)
    3. Truncate very large tool-result messages that may have triggered the
       issue.

    If all retries are exhausted the middleware returns an ``AIMessage`` with
    the error details so the agent loop can continue rather than crash.
    """

    def __init__(self, max_retries: int = 3):
        super().__init__()
        self.max_retries = max_retries
        self.tools = []
        self._retry_count = 0

    # ------------------------------------------------------------------
    # Sanitization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_control_chars(text: str) -> str:
        """Remove control characters that break JSON serialization.

        Keeps \\n (0x0A), \\r (0x0D), \\t (0x09) — valid JSON whitespace.
        """
        return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    @staticmethod
    def _fix_json_escapes(text: str) -> str:
        r"""Double-escape lone backslashes that aren't valid JSON escapes.

        Valid JSON escapes: \" \\ \/ \b \f \n \r \t \uXXXX
        """
        return re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', text)

    def _sanitize_text(self, text: str) -> str:
        """Apply all text-level sanitization passes."""
        text = self._strip_control_chars(text)
        text = self._fix_json_escapes(text)
        return text

    def _sanitize_message_content(self, content):
        """Sanitize message content (str or list of content parts)."""
        if isinstance(content, str):
            return self._sanitize_text(content)
        if isinstance(content, list):
            result = []
            for part in content:
                if isinstance(part, dict):
                    new_part = dict(part)
                    if 'text' in new_part and isinstance(new_part['text'], str):
                        new_part['text'] = self._sanitize_text(new_part['text'])
                    result.append(new_part)
                else:
                    result.append(part)
            return result
        return content

    def _truncate_large_content(self, content, max_chars: int = 20_000) -> str:
        """Truncate content that exceeds max_chars."""
        text = str(content)
        if len(text) > max_chars:
            return text[:max_chars] + "\n\n[Content truncated due to size]"
        return content

    def _sanitize_messages(self, messages, aggressive: bool = False):
        """Return a sanitized copy of the message list.

        Args:
            messages: List of LangChain message objects.
            aggressive: If True, also truncate large tool results.
        """
        sanitized = []
        for msg in messages:
            # Create a shallow copy
            if hasattr(msg, 'model_copy'):
                new_msg = msg.model_copy()
            elif hasattr(msg, 'copy'):
                new_msg = msg.copy()
            else:
                new_msg = msg
                # Can't copy — just sanitize in place (risky but last resort)

            # Sanitize content
            if hasattr(new_msg, 'content') and new_msg.content is not None:
                new_msg.content = self._sanitize_message_content(new_msg.content)
                if aggressive:
                    new_msg.content = self._truncate_large_content(new_msg.content)

            # Sanitize tool call arguments
            if hasattr(new_msg, 'tool_calls') and new_msg.tool_calls:
                sanitized_calls = []
                for tc in new_msg.tool_calls:
                    new_tc = dict(tc)
                    if 'args' in new_tc and isinstance(new_tc['args'], dict):
                        new_args = {}
                        for k, v in new_tc['args'].items():
                            if isinstance(v, str):
                                new_args[k] = self._sanitize_text(v)
                            else:
                                new_args[k] = v
                        new_tc['args'] = new_args
                    sanitized_calls.append(new_tc)
                new_msg.tool_calls = sanitized_calls

            # Sanitize additional_kwargs tool calls (raw OpenAI format)
            if hasattr(new_msg, 'additional_kwargs'):
                ak_calls = new_msg.additional_kwargs.get('tool_calls', [])
                if ak_calls:
                    fixed_calls = []
                    for tc in ak_calls:
                        func = tc.get('function', {})
                        args_str = func.get('arguments')
                        if isinstance(args_str, str):
                            fixed_str = self._sanitize_text(args_str)
                            if fixed_str != args_str:
                                tc = dict(tc)
                                tc['function'] = dict(func)
                                tc['function']['arguments'] = fixed_str
                        fixed_calls.append(tc)
                    new_msg.additional_kwargs = dict(new_msg.additional_kwargs)
                    new_msg.additional_kwargs['tool_calls'] = fixed_calls

            sanitized.append(new_msg)
        return sanitized

    # ------------------------------------------------------------------
    # Middleware hook
    # ------------------------------------------------------------------

    def wrap_model_call(self, request: ModelRequest, handler) -> ModelResponse | AIMessage:
        """Intercept model calls and retry on BadRequestError with sanitization."""
        last_error = None
        num_messages = len(request.messages)

        for attempt in range(self.max_retries + 1):
            try:
                return handler(request)
            except BadRequestError as exc:
                last_error = exc
                self._retry_count += 1
                remaining = self.max_retries - attempt

                if remaining <= 0:
                    logger.error(
                        "BadRequestRetry: exhausted %d retries (%d messages in context, "
                        "%d total retries across session). Returning error to agent. "
                        "Error: %s",
                        self.max_retries, num_messages, self._retry_count,
                        str(exc)[:200],
                    )
                    # Return an AIMessage so the agent loop continues
                    return AIMessage(
                        content=(
                            f"[Error: The model API rejected the request after "
                            f"{self.max_retries + 1} attempts due to malformed content "
                            f"in the conversation history. Error: {str(exc)[:200]}. "
                            f"I'll try a different approach.]"
                        )
                    )

                # Aggressive sanitization on later attempts
                aggressive = attempt >= 1
                logger.warning(
                    "BadRequestRetry: attempt %d/%d failed, %d messages, "
                    "aggressive=%s. Error: %s",
                    attempt + 1, self.max_retries + 1,
                    num_messages, aggressive,
                    str(exc)[:200],
                )

                sanitized_messages = self._sanitize_messages(
                    request.messages, aggressive=aggressive
                )
                request = request.override(messages=sanitized_messages)

        # Should not reach here
        raise last_error  # type: ignore[misc]
