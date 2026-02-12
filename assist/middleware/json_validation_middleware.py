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

    def _sanitize_string_content(self, text: str) -> str:
        """Sanitize string content to ensure it's JSON-safe.

        This ensures that when the string is embedded in JSON, it won't cause
        parsing errors. Handles:
        - Trailing backslashes (common in markdown)
        - Control characters
        - Already-escaped sequences are preserved

        Args:
            text: The string to sanitize

        Returns:
            Sanitized string that's safe to embed in JSON
        """
        if not isinstance(text, str):
            return text

        # Python's json.dumps will handle proper escaping, but we need to
        # ensure the content doesn't have issues before it gets there.
        # The main issue is trailing backslashes which aren't followed by
        # a valid escape sequence.

        # Test if the string is already properly escapable
        try:
            json.dumps(text)
            return text
        except (TypeError, ValueError):
            # If there's an issue, try to fix it
            pass

        # Escape backslashes that aren't part of valid escape sequences
        # This is a bit tricky - we want to escape lone backslashes but not
        # valid escape sequences like \n, \t, etc.

        # Simple approach: if json.dumps fails, replace all backslashes
        # and let json.dumps re-escape properly
        sanitized = text.replace('\\', '\\\\')

        # Test if it works now
        try:
            json.dumps(sanitized)
            return sanitized
        except (TypeError, ValueError):
            # If still failing, just return original and log warning
            logger.warning(f"Could not sanitize string content: {text[:100]}...")
            return text


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

            # Check if this message has content that needs sanitization
            if hasattr(msg, 'content') and isinstance(msg.content, str):
                original_content = msg.content
                sanitized_content = self._sanitize_string_content(original_content)

                if sanitized_content != original_content:
                    # Need to modify the message
                    if hasattr(msg, 'model_copy'):
                        sanitized_msg = msg.model_copy()
                    elif hasattr(msg, 'copy'):
                        sanitized_msg = msg.copy()
                    else:
                        sanitized_msg = msg
                    sanitized_msg.content = sanitized_content
                    modified = True

            # Also check tool calls in the message
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
                                # Sanitize string arguments
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
