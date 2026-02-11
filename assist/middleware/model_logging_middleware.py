"""
Logging middleware for model interactions.

This middleware logs all prompts sent to the model and responses received,
including information about which agent the interaction is happening in.
"""
import logging
from typing import Any
from langchain.agents.middleware import AgentMiddleware, AgentState
from langgraph.runtime import Runtime
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# Create a dedicated logger for model interactions
logger = logging.getLogger("assist.model")


class ModelLoggingMiddleware(AgentMiddleware):
    """Middleware that logs all model prompts and responses at DEBUG level.

    Usage:
        # Configure the logger level
        import logging
        logging.getLogger("assist.model").setLevel(logging.DEBUG)

        # Add to agent
        agent = create_agent(
            model=model,
            working_dir=working_dir,
pp            middleware=[ModelLoggingMiddleware()]
        )
    """

    def __init__(self, agent_name: str | None = None):
        """Initialize the logging middleware.

        Args:
            agent_name: Optional name to identify this agent in logs.
                       If not provided, will attempt to extract from runtime.
        """
        self.agent_name = agent_name
        self._model_call_count = 0

    def _get_agent_name(self, runtime: Runtime) -> str:
        """Extract agent name from runtime or use default."""
        if self.agent_name:
            return self.agent_name

        # Try to get agent name from runtime context
        try:
            # Check if runtime has graph or node information
            if hasattr(runtime, 'graph') and hasattr(runtime.graph, 'name'):
                return runtime.graph.name
            if hasattr(runtime, 'node') and runtime.node:
                return str(runtime.node)
            # Check for config
            if hasattr(runtime, 'config'):
                config = runtime.config
                if isinstance(config, dict):
                    if 'configurable' in config:
                        thread_id = config['configurable'].get('thread_id', 'unknown')
                        return f"agent-{thread_id}"
        except Exception:
            pass

        return "default-agent"

    def _format_tool_call(self, tool_call: dict) -> str:
        name = tool_call.get('name', 'unknown')
        subagent = tool_call.get('arguments', {}).get('subagent_type', None)
        if name == "task" and subagent:
            return f"Subagent {subagent}"
        return name
            

    def _format_message(self, msg: Any) -> str:
        """Format a message for logging."""
        if isinstance(msg, SystemMessage):
            return f"[SYSTEM] {msg.content[:200]}..." if len(msg.content) > 200 else f"[SYSTEM] {msg.content}"
        elif isinstance(msg, HumanMessage):
            return f"[USER] {msg.content[:200]}..." if len(msg.content) > 200 else f"[USER] {msg.content}"
        elif isinstance(msg, AIMessage):
            content = msg.content if msg.content else "[No content]"
            tool_calls = ""
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                tool_names = [self._format_tool_call(tc) for tc in msg.tool_calls]
                tool_calls = f" [Tools: {', '.join(tool_names)}]"
            return f"[AI] {content[:200]}{tool_calls}" if len(str(content)) > 200 else f"[AI] {content}{tool_calls}"
        elif hasattr(msg, 'type'):
            return f"[{msg.type.upper()}] {str(msg)[:200]}"
        else:
            return f"[UNKNOWN] {str(msg)[:200]}"

    def _count_approx_tokens_message(self, msg: Any) -> int:
        """Count approximate tokens in a message.

        Uses the common approximation of ~4 characters per token.
        Accounts for message content, tool calls, and message metadata.

        Args:
            msg: A message object (SystemMessage, HumanMessage, AIMessage, etc.)

        Returns:
            Approximate token count for the message
        """
        total_chars = 0

        # Count message role/type overhead (~10 tokens for role formatting)
        total_chars += 40

        # Count content
        if hasattr(msg, 'content'):
            content = msg.content
            if isinstance(content, str):
                total_chars += len(content)
            elif isinstance(content, list):
                # Handle multi-part content (e.g., text + images)
                for part in content:
                    if isinstance(part, dict):
                        # Text part
                        if 'text' in part:
                            total_chars += len(str(part['text']))
                        # Image or other part (rough estimate)
                        elif 'type' in part:
                            total_chars += 100  # Overhead for non-text content
                    elif isinstance(part, str):
                        total_chars += len(part)

        # Count tool calls for AI messages
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            for tool_call in msg.tool_calls:
                # Tool name
                if 'name' in tool_call:
                    total_chars += len(str(tool_call['name']))

                # Tool arguments (JSON serialized)
                if 'args' in tool_call or 'arguments' in tool_call:
                    args = tool_call.get('args') or tool_call.get('arguments', {})
                    # Rough estimate: serialize to string and count
                    import json
                    try:
                        args_str = json.dumps(args)
                        total_chars += len(args_str)
                    except (TypeError, ValueError):
                        # If can't serialize, rough estimate
                        total_chars += len(str(args))

                # Tool call ID overhead
                if 'id' in tool_call:
                    total_chars += len(str(tool_call['id']))

                # Fixed overhead per tool call (~20 tokens for structure)
                total_chars += 80

        # Convert characters to approximate tokens (1 token ≈ 4 chars)
        return total_chars // 4

    def _count_approx_tokens_messages(self, msgs: list[Any]) -> int:
        return sum([self._count_approx_tokens_message(msg) for msg in msgs])

    def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Log the prompt before sending to the model."""
        self._model_call_count += 1
        agent_name = self._get_agent_name(runtime)

        # Always log at INFO level that a model call is happening
        logger.info(f"[{agent_name}] Model Call #{self._model_call_count} starting")

        # Detailed logging only at DEBUG level
        if logger.isEnabledFor(logging.DEBUG):
            messages = state.get("messages", [])

            logger.debug("=" * 80)
            logger.debug(f"[{agent_name}] Model Call #{self._model_call_count} - PROMPT ({self._count_approx_tokens_messages(messages)} approx tokens)")
            logger.debug("-" * 80)

            # Log each message
            for i, msg in enumerate(messages):
                logger.debug(f"  Message {i + 1}: {self._format_message(msg)}")

            logger.debug("-" * 80)

        return None

    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Log the response after receiving from the model."""
        agent_name = self._get_agent_name(runtime)

        # Always log at INFO level that a model call completed
        logger.info(f"[{agent_name}] Model Call #{self._model_call_count} completed")

        # Detailed logging only at DEBUG level
        if logger.isEnabledFor(logging.DEBUG):
            messages = state.get("messages", [])

            # The last message should be the model's response
            if messages:
                last_message = messages[-1]
                logger.debug(f"[{agent_name}] Model Call #{self._model_call_count} - RESPONSE")
                logger.debug("-" * 80)
                logger.debug(f"  {self._format_message(last_message)}")
                logger.debug("=" * 80)
                logger.debug("")  # Empty line for readability

        return None
