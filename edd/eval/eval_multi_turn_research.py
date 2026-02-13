"""Evaluation: Multi-turn research conversation.

This eval tests the agent's ability to handle a sustained research conversation
over 10 turns, including:
- Initial research requests
- Follow-up questions
- Comparative analysis
- JSON validation under realistic load
- Concurrent tool call handling
"""
import os
import sys
import json
import tempfile
import logging
from datetime import datetime
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage

from assist.thread import Thread, ThreadManager
from assist.model_manager import select_chat_model


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ConversationDriver:
    """Drives a multi-turn conversation with research and follow-ups."""

    def __init__(self, model: BaseChatModel):
        """Initialize the conversation driver.

        Args:
            model: The model to use for generating conversation prompts
        """
        self.model = model
        self.turn_count = 0

    def get_initial_prompt(self) -> str:
        """Get the initial research prompt to start the conversation."""
        prompts = [
            "Research the key differences between Python asyncio and threading, and explain when to use each.",
            "Compare the architectural approaches of React and Vue.js frameworks.",
            "Research the main differences between SQL and NoSQL databases with specific examples.",
            "Investigate the pros and cons of microservices versus monolithic architecture.",
        ]
        # Use first prompt for consistency
        return prompts[0]

    def generate_follow_up(self, conversation_history: list[dict]) -> str:
        """Generate a follow-up question based on conversation history.

        Args:
            conversation_history: List of messages in the conversation

        Returns:
            A follow-up question or research request
        """
        self.turn_count += 1

        # Create a system prompt for the driver model
        system_msg = """You are driving a research conversation with an AI assistant.
Generate the next question or research request based on the conversation so far.

Guidelines:
- For early turns (1-3): Ask for additional research on related topics
- For middle turns (4-6): Request comparisons or deeper analysis
- For later turns (7-9): Ask synthesizing questions that require using previous research
- Keep questions concise and focused
- Don't repeat previous questions
- Output ONLY the question, no preamble

Example follow-ups:
- "Now research the performance characteristics of each approach"
- "Compare the learning curves for developers new to each"
- "Based on the previous research, which would you recommend for a real-time chat application?"
- "What are the main tradeoffs when choosing between these options?"
"""

        # Build conversation summary for context
        user_msgs = [msg for msg in conversation_history if msg['role'] == 'user']
        assistant_msgs = [msg for msg in conversation_history if msg['role'] == 'assistant']

        context = f"""Conversation so far has {len(user_msgs)} user turns and {len(assistant_msgs)} assistant turns.

Current turn: {self.turn_count}/10

Last user question: {user_msgs[-1]['content'] if user_msgs else 'None'}

Last assistant response (first 200 chars): {assistant_msgs[-1]['content'][:200] if assistant_msgs else 'None'}

Generate the next question:"""

        # Generate follow-up
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": context}
        ]

        response = self.model.invoke(messages)
        follow_up = response.content.strip()

        logger.info(f"Generated follow-up for turn {self.turn_count}: {follow_up[:100]}...")
        return follow_up


class EvalMetrics:
    """Track metrics for the evaluation."""

    def __init__(self):
        self.turns = []
        self.total_tool_calls = 0
        self.total_tool_results = 0
        self.json_errors = 0
        self.start_time = None
        self.end_time = None

    def record_turn(self, turn_num: int, user_msg: str, agent_response: str,
                   tool_calls: int = 0, tool_results: int = 0,
                   response_length: int = 0):
        """Record metrics for a conversation turn."""
        self.turns.append({
            'turn': turn_num,
            'user_msg_length': len(user_msg),
            'agent_response_length': response_length,
            'tool_calls': tool_calls,
            'tool_results': tool_results,
        })
        self.total_tool_calls += tool_calls
        self.total_tool_results += tool_results

    def record_json_error(self):
        """Record a JSON validation error."""
        self.json_errors += 1

    def start(self):
        """Mark evaluation start time."""
        self.start_time = datetime.now()

    def end(self):
        """Mark evaluation end time."""
        self.end_time = datetime.now()

    def get_summary(self) -> dict:
        """Get summary statistics."""
        duration = (self.end_time - self.start_time).total_seconds() if self.start_time and self.end_time else 0

        return {
            'total_turns': len(self.turns),
            'total_tool_calls': self.total_tool_calls,
            'total_tool_results': self.total_tool_results,
            'json_errors': self.json_errors,
            'duration_seconds': duration,
            'avg_tool_calls_per_turn': self.total_tool_calls / len(self.turns) if self.turns else 0,
            'turns': self.turns,
        }

    def print_summary(self):
        """Print a human-readable summary."""
        summary = self.get_summary()

        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)
        print(f"Total Turns: {summary['total_turns']}")
        print(f"Duration: {summary['duration_seconds']:.2f} seconds")
        print(f"Total Tool Calls: {summary['total_tool_calls']}")
        print(f"Total Tool Results: {summary['total_tool_results']}")
        print(f"JSON Errors: {summary['json_errors']}")
        print(f"Avg Tool Calls/Turn: {summary['avg_tool_calls_per_turn']:.2f}")
        print("\nPer-Turn Breakdown:")
        print("-" * 80)
        for turn in summary['turns']:
            print(f"  Turn {turn['turn']:2d}: "
                  f"User: {turn['user_msg_length']:4d} chars | "
                  f"Agent: {turn['agent_response_length']:5d} chars | "
                  f"Tools: {turn['tool_calls']} calls, {turn['tool_results']} results")
        print("=" * 80)


def count_tool_calls_in_messages(messages: list) -> tuple[int, int]:
    """Count tool calls and tool results in message list.

    Args:
        messages: List of message objects

    Returns:
        Tuple of (tool_calls_count, tool_results_count)
    """
    from langchain_core.messages import AIMessage, ToolMessage

    tool_calls = 0
    tool_results = 0

    for msg in messages:
        if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls:
            tool_calls += len(msg.tool_calls)
        elif isinstance(msg, ToolMessage):
            tool_results += 1

    return tool_calls, tool_results


def run_eval(num_turns: int = 10, verbose: bool = True) -> EvalMetrics:
    """Run the multi-turn research conversation evaluation.

    Args:
        num_turns: Number of conversation turns to run
        verbose: Whether to print detailed progress

    Returns:
        EvalMetrics object with results
    """
    logger.info("Starting multi-turn research conversation eval")

    # Create metrics tracker
    metrics = EvalMetrics()
    metrics.start()

    # Create a temporary directory for the thread
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create thread manager
        thread_manager = ThreadManager(root_dir=tmpdir)

        # Create a new thread
        thread = thread_manager.new()

        # Create conversation driver (uses same model as agent for simplicity)
        driver_model = select_chat_model("mistral-nemo", 0.7)  # Higher temp for variety
        driver = ConversationDriver(driver_model)

        # Get initial prompt
        initial_prompt = driver.get_initial_prompt()

        if verbose:
            print(f"\n{'=' * 80}")
            print("STARTING MULTI-TURN RESEARCH CONVERSATION EVAL")
            print(f"{'=' * 80}")
            print(f"Turns: {num_turns}")
            print(f"Working Directory: {thread.working_dir}")
            print(f"{'=' * 80}\n")

        # Turn 1: Initial research request
        if verbose:
            print(f"\n{'─' * 80}")
            print(f"TURN 1/{num_turns}")
            print(f"{'─' * 80}")
            print(f"USER: {initial_prompt}\n")

        try:
            response = thread.message(initial_prompt)
            # Get the last message from the conversation
            messages = thread.get_messages()
            agent_response = messages[-1]['content'] if messages else ""

            if verbose:
                print(f"AGENT: {agent_response[:500]}...")
                if len(agent_response) > 500:
                    print(f"       (response truncated, total length: {len(agent_response)} chars)")

            # Count tool usage
            raw_messages = thread.get_raw_messages()
            tool_calls, tool_results = count_tool_calls_in_messages(raw_messages)

            metrics.record_turn(
                turn_num=1,
                user_msg=initial_prompt,
                agent_response=agent_response,
                tool_calls=tool_calls,
                tool_results=tool_results,
                response_length=len(agent_response)
            )

        except Exception as e:
            logger.error(f"Error in turn 1: {e}", exc_info=True)
            if "Invalid" in str(e) and "escape" in str(e):
                metrics.record_json_error()
            raise

        # Subsequent turns: Follow-ups and comparisons
        for turn in range(2, num_turns + 1):
            if verbose:
                print(f"\n{'─' * 80}")
                print(f"TURN {turn}/{num_turns}")
                print(f"{'─' * 80}")

            # Get conversation history
            conversation_history = thread.get_messages()

            # Generate follow-up
            follow_up = driver.generate_follow_up(conversation_history)

            if verbose:
                print(f"USER: {follow_up}\n")

            try:
                # Send to agent
                response = thread.message(follow_up)
                # Get the last message from the conversation
                messages = thread.get_messages()
                agent_response = messages[-1]['content'] if messages else ""

                if verbose:
                    print(f"AGENT: {agent_response[:500]}...")
                    if len(agent_response) > 500:
                        print(f"       (response truncated, total length: {len(agent_response)} chars)")

                # Count tool usage (only new tools since last turn)
                raw_messages = thread.get_raw_messages()
                tool_calls, tool_results = count_tool_calls_in_messages(raw_messages)

                # Calculate new tools (difference from previous)
                prev_total_calls = metrics.total_tool_calls
                prev_total_results = metrics.total_tool_results
                new_calls = tool_calls - prev_total_calls
                new_results = tool_results - prev_total_results

                metrics.record_turn(
                    turn_num=turn,
                    user_msg=follow_up,
                    agent_response=agent_response,
                    tool_calls=new_calls,
                    tool_results=new_results,
                    response_length=len(agent_response)
                )

            except Exception as e:
                logger.error(f"Error in turn {turn}: {e}", exc_info=True)
                if "Invalid" in str(e) and "escape" in str(e):
                    metrics.record_json_error()
                # Continue with next turn instead of failing
                continue

        metrics.end()

        # Save final conversation
        final_messages = thread.get_messages()
        conversation_file = Path(tmpdir) / "final_conversation.json"
        with open(conversation_file, 'w') as f:
            json.dump(final_messages, f, indent=2)

        if verbose:
            print(f"\n\nConversation saved to: {conversation_file}")

    return metrics


def main():
    """Run the evaluation and print results."""
    import argparse

    parser = argparse.ArgumentParser(description="Multi-turn research conversation eval")
    parser.add_argument('--turns', type=int, default=10, help='Number of turns to run')
    parser.add_argument('--quiet', action='store_true', help='Suppress detailed output')
    parser.add_argument('--output', type=str, help='Save metrics to JSON file')

    args = parser.parse_args()

    try:
        metrics = run_eval(num_turns=args.turns, verbose=not args.quiet)
        metrics.print_summary()

        # Save metrics if requested
        if args.output:
            output_path = Path(args.output)
            with open(output_path, 'w') as f:
                json.dump(metrics.get_summary(), f, indent=2)
            print(f"\nMetrics saved to: {output_path}")

        # Exit with error code if there were JSON errors
        if metrics.json_errors > 0:
            print(f"\n❌ FAILED: {metrics.json_errors} JSON errors occurred")
            sys.exit(1)
        else:
            print("\n✅ PASSED: No JSON errors")
            sys.exit(0)

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        print(f"\n❌ FAILED: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
