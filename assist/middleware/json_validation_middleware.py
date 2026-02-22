"""Middleware to validate and sanitize JSON in tool calls."""
import json
import logging
import re
from typing import Any
from langchain.agents.middleware import AgentMiddleware, AgentState
from langgraph.runtime import Runtime

logger = logging.getLogger(__name__)


class JsonValidationMiddleware(AgentMiddleware):
    """Middleware that validates and sanitizes JSON in tool call arguments.

    This middleware operates in two phases:

    Before model (prevents vLLM/API errors):
    1. Sanitizes string content to ensure JSON-safe encoding
    2. Escapes problematic characters (trailing backslashes, control chars)
    3. Ensures all message content can be serialized to JSON

    After model (fixes model output):
    1. Validates that all tool call arguments are valid JSON
    2. Attempts to fix common JSON errors (trailing commas, quote issues)
    3. Logs warnings when issues are found

    Usage:
        agent = create_agent(
            model=model,
            middleware=[JsonValidationMiddleware(strict=False)]
        )
    """

    def __init__(self, strict: bool = False):
        """Initialize the JSON validation middleware.

        Args:
            strict: If True, raise errors on invalid JSON. If False (default),
                   attempt to fix and log warnings.
        """
        self.strict = strict
        self._validation_count = 0
        self._fix_count = 0

    def _fix_json_invalid_escapes(self, json_str: str) -> str:
        r"""Fix invalid backslash escape sequences in a JSON string.

        JSON only allows these escape sequences: \" \\ \/ \b \f \n \r \t \uXXXX
        Models sometimes produce arguments with invalid escapes like \ followed
        by a space or other characters. This fixes them by escaping the backslash.

        Args:
            json_str: A JSON string that may contain invalid escape sequences

        Returns:
            Fixed JSON string with invalid escapes corrected
        """
        try:
            json.loads(json_str)
            return json_str
        except json.JSONDecodeError:
            pass

        # Fix invalid escapes: replace \X where X is not a valid JSON escape char
        # Valid JSON escape chars after \: " \ / b f n r t u
        fixed = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', json_str)

        try:
            json.loads(fixed)
            return fixed
        except json.JSONDecodeError:
            logger.warning(f"Could not fix JSON escapes: {json_str[:100]}...")
            return json_str

    def _strip_control_chars(self, text: str) -> str:
        """Strip control characters that break JSON serialization.

        Removes null bytes and other non-printable control characters
        (U+0000–U+001F) that are not valid JSON whitespace (\\n, \\r, \\t).
        These characters cause BadRequestError 400 "Expecting ':' delimiter"
        from vLLM even when passed inside a JSON string value, because
        vLLM's own JSON parser trips over them.

        Args:
            text: Input string

        Returns:
            String with problematic control characters removed
        """
        # Keep \n (0x0A), \r (0x0D), \t (0x09) — they are valid JSON whitespace.
        # Remove everything else in 0x00–0x1F range plus the lone DEL (0x7F).
        return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    def _sanitize_string_content(self, text: str) -> str:
        """Sanitize string content to ensure it's JSON-safe.

        This ensures that when the string is embedded in JSON, it won't cause
        parsing errors. Handles:
        - Null bytes and control characters (primary cause of 400 errors from vLLM)
        - Trailing backslashes (common in markdown)
        - Already-escaped sequences are preserved

        Args:
            text: The string to sanitize

        Returns:
            Sanitized string that's safe to embed in JSON
        """
        if not isinstance(text, str):
            return text

        # First pass: strip control characters that vLLM can't handle even
        # when properly JSON-encoded (null bytes cause "Expecting ':' delimiter").
        text = self._strip_control_chars(text)

        # Second pass: ensure the result is JSON-serialisable.
        try:
            json.dumps(text)
            return text
        except (TypeError, ValueError):
            pass

        # If json.dumps still fails, escape bare backslashes.
        sanitized = text.replace('\\', '\\\\')
        try:
            json.dumps(sanitized)
            return sanitized
        except (TypeError, ValueError):
            logger.warning(f"Could not sanitize string content: {text[:100]}...")
            return text

    def _sanitize_content(self, content) -> tuple[Any, bool]:
        """Sanitize message content (string or list of content parts).

        Returns:
            (sanitized_content, was_modified)
        """
        if isinstance(content, str):
            sanitized = self._sanitize_string_content(content)
            return sanitized, sanitized != content

        if isinstance(content, list):
            modified = False
            result = []
            for part in content:
                if isinstance(part, dict):
                    new_part = dict(part)
                    if 'text' in new_part and isinstance(new_part['text'], str):
                        san = self._sanitize_string_content(new_part['text'])
                        if san != new_part['text']:
                            new_part['text'] = san
                            modified = True
                    result.append(new_part)
                else:
                    result.append(part)
            return result, modified

        return content, False


    def _validate_tool_call(self, tool_call: dict) -> tuple[bool, str | None]:
        """Validate a single tool call's arguments.

        Args:
            tool_call: The tool call dictionary

        Returns:
            Tuple of (is_valid, error_message)
        """
        if 'function' not in tool_call:
            return True, None

        function = tool_call['function']

        # Get arguments - might be string or dict
        arguments = function.get('arguments')

        if arguments is None:
            return True, None

        # If already a dict, validate it can be serialized
        if isinstance(arguments, dict):
            try:
                json.dumps(arguments)
                return True, None
            except (TypeError, ValueError) as e:
                return False, f"Cannot serialize arguments dict: {e}"

        # If string, try to parse as JSON
        if isinstance(arguments, str):
            try:
                json.loads(arguments)
                return True, None
            except json.JSONDecodeError as e:
                return False, f"Invalid JSON in arguments: {e}"

        return False, f"Unknown arguments type: {type(arguments)}"

    def _attempt_fix_tool_call(self, tool_call: dict) -> dict:
        """Attempt to fix common JSON issues in a tool call.

        Args:
            tool_call: The tool call dictionary

        Returns:
            Fixed tool call dictionary
        """
        if 'function' not in tool_call:
            return tool_call

        function = tool_call['function']
        arguments = function.get('arguments')

        if not isinstance(arguments, str):
            return tool_call

        try:
            # Try to parse - if it works, no fix needed
            json.loads(arguments)
            return tool_call
        except json.JSONDecodeError:
            # Try common fixes
            original = arguments

            # Fix 1: Replace single quotes with double quotes for property names
            # This is a common issue when models generate {"query": 'value'}
            fixed = re.sub(r"'([^']*)':", r'"\1":', arguments)

            # Fix 2: Try to handle unescaped newlines
            fixed = fixed.replace('\n', '\\n').replace('\r', '\\r')

            # Fix 3: Remove trailing commas before closing braces/brackets
            fixed = re.sub(r',(\s*[}\]])', r'\1', fixed)

            # Fix 4: Fix invalid backslash escapes (e.g. \  or \- )
            fixed = self._fix_json_invalid_escapes(fixed)

            try:
                # Validate the fix worked
                json.loads(fixed)
                logger.warning(
                    f"Fixed JSON in tool call {tool_call.get('id', 'unknown')}: "
                    f"Original: {original[:100]}... "
                    f"Fixed: {fixed[:100]}..."
                )
                self._fix_count += 1

                # Update the tool call
                function['arguments'] = fixed
                return tool_call
            except json.JSONDecodeError:
                # Fix didn't work
                logger.error(
                    f"Could not fix JSON in tool call {tool_call.get('id', 'unknown')}: "
                    f"{original[:200]}..."
                )
                return tool_call

    def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Sanitize content before sending to the model.

        This prevents JSON serialization errors by escaping problematic
        characters in message content and tool call arguments.

        Args:
            state: The current agent state
            runtime: The runtime context

        Returns:
            Modified state if sanitization was needed, None otherwise
        """
        messages = state.get("messages", [])
        if not messages:
            return None

        modified = False
        sanitized_messages = []

        for msg in messages:
            # Create a copy to avoid modifying the original
            sanitized_msg = msg

            # Sanitize message content (handles both str and list[content_part])
            if hasattr(msg, 'content') and msg.content is not None:
                sanitized_content, content_changed = self._sanitize_content(msg.content)

                if content_changed:
                    if hasattr(msg, 'model_copy'):
                        sanitized_msg = msg.model_copy()
                    elif hasattr(msg, 'copy'):
                        sanitized_msg = msg.copy()
                    else:
                        sanitized_msg = msg
                    sanitized_msg.content = sanitized_content
                    modified = True

            # Also check tool calls in the message (LangChain dict format)
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                sanitized_calls = []
                calls_modified = False

                for tool_call in msg.tool_calls:
                    sanitized_call = tool_call

                    # Check if the tool call has arguments that need sanitization
                    if 'args' in tool_call:
                        args = tool_call['args']
                        sanitized_args = {}
                        args_modified = False

                        for key, value in args.items():
                            if isinstance(value, str):
                                # Sanitize string arguments (strip control chars first)
                                sanitized_value = self._sanitize_string_content(value)

                                if sanitized_value != value:
                                    args_modified = True

                                sanitized_args[key] = sanitized_value
                            else:
                                sanitized_args[key] = value

                        if args_modified:
                            # Create a modified tool call
                            if hasattr(tool_call, 'model_copy'):
                                sanitized_call = tool_call.model_copy()
                            elif hasattr(tool_call, 'copy'):
                                sanitized_call = tool_call.copy()
                            else:
                                sanitized_call = dict(tool_call)
                            sanitized_call['args'] = sanitized_args
                            calls_modified = True

                    sanitized_calls.append(sanitized_call)

                if calls_modified:
                    if hasattr(msg, 'model_copy'):
                        sanitized_msg = msg.model_copy()
                    elif hasattr(msg, 'copy'):
                        sanitized_msg = msg.copy()
                    else:
                        sanitized_msg = msg
                    sanitized_msg.tool_calls = sanitized_calls
                    modified = True

            # Also check additional_kwargs tool calls (OpenAI raw format)
            # This is what actually gets sent to the API - if function.arguments
            # contains invalid JSON escapes, vLLM will reject with 400 Bad Request
            if hasattr(msg, 'additional_kwargs'):
                ak_tool_calls = msg.additional_kwargs.get('tool_calls', [])
                if ak_tool_calls:
                    ak_modified = False
                    fixed_ak_calls = []

                    for tc in ak_tool_calls:
                        func = tc.get('function', {})
                        args_str = func.get('arguments')

                        if isinstance(args_str, str):
                            fixed_str = self._fix_json_invalid_escapes(args_str)
                            if fixed_str != args_str:
                                tc = dict(tc)
                                tc['function'] = dict(func)
                                tc['function']['arguments'] = fixed_str
                                ak_modified = True
                                logger.info(
                                    f"Fixed invalid JSON escape in additional_kwargs "
                                    f"tool call {tc.get('id', 'unknown')}"
                                )

                        fixed_ak_calls.append(tc)

                    if ak_modified:
                        if sanitized_msg is msg:
                            if hasattr(msg, 'model_copy'):
                                sanitized_msg = msg.model_copy()
                            elif hasattr(msg, 'copy'):
                                sanitized_msg = msg.copy()
                            else:
                                sanitized_msg = msg
                        sanitized_msg.additional_kwargs = dict(sanitized_msg.additional_kwargs)
                        sanitized_msg.additional_kwargs['tool_calls'] = fixed_ak_calls
                        modified = True

            sanitized_messages.append(sanitized_msg)

        if modified:
            logger.info(f"Sanitized {len(messages)} messages before sending to model")
            return {"messages": sanitized_messages}

        return None

    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Validate tool calls after the model generates them.

        Args:
            state: The current agent state
            runtime: The runtime context

        Returns:
            Modified state if fixes were applied, None otherwise
        """
        messages = state.get("messages", [])

        if not messages:
            return None

        # Get the last message (should be from the assistant)
        last_message = messages[-1]

        # Check if it has tool calls
        if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
            return None

        tool_calls = last_message.tool_calls
        self._validation_count += len(tool_calls)

        # Validate each tool call
        modified = False
        validated_calls = []

        for tool_call in tool_calls:
            is_valid, error = self._validate_tool_call(tool_call)

            if not is_valid:
                logger.warning(
                    f"Invalid JSON in tool call {tool_call.get('id', 'unknown')} "
                    f"for function {tool_call.get('function', {}).get('name', 'unknown')}: "
                    f"{error}"
                )

                if self.strict:
                    raise ValueError(
                        f"Invalid JSON in tool call: {error}. "
                        f"Set strict=False to attempt automatic fixes."
                    )

                # Attempt to fix
                fixed_call = self._attempt_fix_tool_call(tool_call)
                validated_calls.append(fixed_call)
                modified = True
            else:
                validated_calls.append(tool_call)

        # Log statistics periodically
        if self._validation_count % 100 == 0:
            logger.info(
                f"JSON validation stats: {self._validation_count} tool calls validated, "
                f"{self._fix_count} fixes applied"
            )

        # If we modified anything, update the state
        if modified:
            # Update the tool calls in the last message
            last_message.tool_calls = validated_calls
            return {"messages": messages}

        return None
