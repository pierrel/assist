"""Context-aware tool result eviction middleware.

This middleware monitors context usage and evicts large tool results to the filesystem
BEFORE they're sent to the model, preventing context overflow errors.

Unlike the built-in FilesystemMiddleware which seems to have issues detecting large
results, this middleware:
1. Calculates total context size (messages + incoming tool result)
2. Checks against model's max_input_tokens at runtime
3. Evicts to /large_tool_results/ if it would exceed threshold
4. Works with the agent's backend (StateBackend, StoreBackend, CompositeBackend)
"""
import logging
import json
from typing import Callable, Any

from langchain.agents.middleware import AgentMiddleware
from langchain.tools.tool_node import ToolCallRequest, ToolRuntime
from langchain_core.messages import ToolMessage, AIMessage, HumanMessage, SystemMessage
from langgraph.types import Command


logger = logging.getLogger(__name__)


class ContextAwareToolEvictionMiddleware(AgentMiddleware):
    """Middleware that evicts large tool results to prevent context overflow.

    This middleware intercepts tool results BEFORE they're sent to the model and:
    1. Calculates current context usage from conversation history
    2. Calculates incoming tool result size
    3. Checks if combined size exceeds threshold (default: 75% of max_input_tokens)
    4. If yes, writes result to /large_tool_results/{tool_call_id} and replaces
       with a reference message instructing the model to read the file

    Args:
        trigger_fraction: Context fraction to trigger eviction (default: 0.75)
        backend_factory: Optional backend factory (will use agent's backend if None)

    Example:
        middleware = ContextAwareToolEvictionMiddleware(
            trigger_fraction=0.70,  # Evict if context would reach 70%
        )
    """

    def __init__(
        self,
        trigger_fraction: float = 0.75,
        backend_factory: Callable[[ToolRuntime], Any] | None = None,
    ):
        """Initialize the middleware.

        Args:
            trigger_fraction: Fraction of max_input_tokens to trigger eviction (0.0-1.0)
            backend_factory: Optional backend factory (uses agent's backend if None)
        """
        if not 0.0 <= trigger_fraction <= 1.0:
            raise ValueError(f"trigger_fraction must be between 0.0 and 1.0, got {trigger_fraction}")

        self.trigger_fraction = trigger_fraction
        self.backend_factory = backend_factory
        self._eviction_count = 0

        logger.info(
            f"ContextAwareToolEvictionMiddleware initialized: trigger={trigger_fraction:.0%}"
        )

    def _get_backend(self, runtime: ToolRuntime) -> Any:
        """Get the backend for file operations.

        Uses the agent's backend if backend_factory is None.
        """
        if self.backend_factory is not None:
            return self.backend_factory(runtime)

        # Use agent's backend from state
        # The backend is typically available in the agent state
        state = getattr(runtime, 'state', {})
        backend = state.get('_backend')

        if backend is None:
            # Fallback: try to get from config
            from deepagents.backends import StateBackend
            logger.warning("Backend not found in state, using StateBackend as fallback")
            return StateBackend(runtime)

        return backend

    def _estimate_tokens(self, content: Any) -> int:
        """Estimate token count using conservative approximation.

        Uses ~4 characters per token, which works well for most content types.
        """
        if isinstance(content, str):
            return len(content) // 4
        return len(str(content)) // 4

    def _count_message_tokens(self, msg: Any) -> int:
        """Count approximate tokens in a message.

        Simply converts the entire message to a string and estimates tokens
        based on character count (~4 chars per token).
        """
        # Convert entire message to string representation
        msg_str = str(msg)

        # Add some overhead for message structure
        total_chars = len(msg_str) + 50  # +50 for message metadata/structure

        return total_chars // 4

    def _get_context_size(self, runtime: ToolRuntime) -> tuple[int, int]:
        """Calculate current context usage and max tokens.

        Returns:
            Tuple of (current_tokens, max_tokens)
        """
        # Get messages from state
        state = getattr(runtime, 'state', {})
        messages = state.get('messages', [])

        # Count tokens in all messages
        current_tokens = sum(self._count_message_tokens(msg) for msg in messages)

        # Get max_input_tokens from model profile
        # This should be set in the agent's model configuration
        max_tokens = 128000  # Default fallback

        # Try to get from agent config
        config = getattr(runtime, 'config', {})
        configurable = config.get('configurable', {})

        # Check various places where max_input_tokens might be
        if 'max_input_tokens' in configurable:
            max_tokens = configurable['max_input_tokens']
        elif '_model_profile' in state:
            profile = state['_model_profile']
            if isinstance(profile, dict) and 'max_input_tokens' in profile:
                max_tokens = profile['max_input_tokens']

        return current_tokens, max_tokens

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Intercept tool results and evict to filesystem if they would cause overflow.

        This is called AFTER tool execution but BEFORE the result is added to messages.
        """
        # Execute the tool first
        tool_result = handler(request)

        # Only process ToolMessage results (Commands are already handled)
        if not isinstance(tool_result, ToolMessage):
            return tool_result

        # Get current context and limits
        try:
            current_tokens, max_tokens = self._get_context_size(request.runtime)
        except Exception as e:
            logger.warning(f"Failed to get context size: {e}, skipping eviction check")
            return tool_result

        # Estimate tokens in this tool result
        result_tokens = self._estimate_tokens(tool_result.content)

        # Calculate what total would be after adding this result
        projected_tokens = current_tokens + result_tokens
        threshold_tokens = int(max_tokens * self.trigger_fraction)

        # Check if we should evict based on projected context size
        should_evict = projected_tokens >= threshold_tokens

        if not should_evict:
            # No eviction needed
            logger.debug(
                f"Tool result OK: {result_tokens} tokens, "
                f"context: {current_tokens}/{max_tokens} ({current_tokens/max_tokens:.1%}, "
                f"projected: {projected_tokens}/{max_tokens} ({projected_tokens/max_tokens:.1%})"
            )
            return tool_result

        # Evict to filesystem
        tool_call_id = tool_result.tool_call_id
        tool_name = tool_result.name or "unknown_tool"

        logger.info(
            f"Evicting large tool result: {result_tokens} tokens from {tool_name}, "
            f"context would be {projected_tokens}/{max_tokens} "
            f"({projected_tokens/max_tokens:.1%})"
        )

        try:
            # Get backend
            backend = self._get_backend(request.runtime)

            # Sanitize tool_call_id for filename (remove invalid chars)
            safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in tool_call_id)

            # Write to same location as FilesystemMiddleware would use
            file_path = f"/large_tool_results/{safe_id}"

            # Convert content to string
            content_str = str(tool_result.content)

            # Write the file
            write_result = backend.write(file_path, content_str)

            if write_result.error:
                logger.error(f"Failed to write large result to {file_path}: {write_result.error}")
                # Return original result if write fails
                return tool_result

            # Calculate how many tokens we have available for the replacement message
            # We want: current_tokens + replacement_tokens < threshold_tokens
            # So: replacement_tokens < threshold_tokens - current_tokens
            available_tokens = threshold_tokens - current_tokens

            # Reserve tokens for the message structure (header, footer, etc.)
            # Rough estimate: ~150 chars for the structure
            structure_chars = 250
            available_chars_for_preview = (available_tokens * 4) - structure_chars

            # Create preview, but guard against it being too large
            preview = ""
            include_preview = available_chars_for_preview > 100  # Only include if we have >100 chars available

            if include_preview:
                max_preview_chars = min(500, max(0, int(available_chars_for_preview)))
                preview = content_str[:max_preview_chars]
                if len(content_str) > max_preview_chars:
                    preview += "..."

            # Create replacement message that tells model to read the file
            if include_preview:
                replacement_content = f"""Tool result too large ({result_tokens:,} tokens), saved to filesystem.

File: {write_result.path}
Size: {len(content_str):,} characters (~{result_tokens:,} tokens)

Preview:
{preview}

To access the full result, use the read_file tool with path: {write_result.path}"""
            else:
                # No preview - context too tight
                replacement_content = f"""Tool result too large ({result_tokens:,} tokens), saved to filesystem.

File: {write_result.path}
Size: {len(content_str):,} characters (~{result_tokens:,} tokens)

To access the full result, use the read_file tool with path: {write_result.path}"""

            replacement_message = ToolMessage(
                content=replacement_content,
                tool_call_id=tool_result.tool_call_id,
                name=tool_result.name,
                status=tool_result.status,
            )

            self._eviction_count += 1
            logger.info(
                f"✓ Evicted {result_tokens:,} tokens to {write_result.path} "
                f"(total evictions: {self._eviction_count})"
            )

            # Return Command with state update if needed
            if write_result.files_update is not None:
                return Command(
                    update={
                        "files": write_result.files_update,
                        "messages": [replacement_message],
                    }
                )

            return replacement_message

        except Exception as e:
            logger.error(f"Error during tool result eviction: {e}", exc_info=True)
            # Return original result if eviction fails
            return tool_result

    def before_model(self, messages: list, **kwargs) -> list:
        """Hook called before messages are sent to model.

        This allows us to see what's actually being sent and potentially
        identify where tool results are being truncated.
        """
        logger.debug(f"before_model called with {len(messages)} messages")

        # Log any ToolMessage content lengths
        for i, msg in enumerate(messages):
            if hasattr(msg, '__class__') and msg.__class__.__name__ == 'ToolMessage':
                content_len = len(str(msg.content)) if hasattr(msg, 'content') else 0
                logger.debug(f"  Message {i}: ToolMessage with {content_len} chars (~{content_len//4} tokens)")

        return messages
