"""Evaluation: Large tool results context management.

This eval tests that the agent can handle tool results exceeding the context limit
by using FilesystemMiddleware to evict large results to files.

The test mocks the internet search tool to return a payload >78k tokens, then verifies:
1. The agent doesn't crash with a context overflow error
2. Large tool results are written to /large_tool_results/
3. The agent can still complete the research task
4. Context management middleware is working correctly
"""
import os
import sys
import json
import tempfile
import logging
from pathlib import Path
from unittest.mock import patch

from assist.thread import Thread, ThreadManager
from assist.model_manager import select_chat_model


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_large_payload(target_tokens: int = 80000) -> str:
    """Create a large text payload exceeding the target token count.

    Args:
        target_tokens: Target token count (default: 80k tokens, over the 78k limit)

    Returns:
        A large string that will exceed the token limit
    """
    # Rough estimate: 1 token ~= 4 characters for English text
    target_chars = target_tokens * 4

    # Create realistic-looking research content
    base_paragraph = """
    The President of France (Président de la République française) is the head of state
    and head of government of the French Republic. The current president is Emmanuel Macron,
    who has been serving since May 14, 2017. He was re-elected for a second term in 2022.

    Emmanuel Macron was born on December 21, 1977, in Amiens, France. Before entering politics,
    he worked as an investment banker at Rothschild & Cie Banque. He served as Minister of
    Economy, Industry and Digital Affairs under President François Hollande from 2014 to 2016.

    Macron founded the centrist political party La République En Marche! (LREM) in April 2016.
    He ran for president in 2017 and won in the second round with 66.1% of the vote against
    Marine Le Pen. His presidency has focused on economic reforms, European integration, and
    climate change initiatives.

    Key policies during his presidency include:
    - Labor market reforms to increase flexibility and reduce unemployment
    - Tax reforms including the elimination of the wealth tax
    - Pension system reform (faced significant opposition)
    - Environmental initiatives including carbon neutrality goals
    - Strong support for the European Union and multilateralism
    - COVID-19 pandemic response including lockdowns and vaccination programs
    """

    # Repeat the content to reach target size
    repetitions = (target_chars // len(base_paragraph)) + 1
    large_content = (base_paragraph * repetitions)[:target_chars]

    return large_content


class EvalMetrics:
    """Track metrics for the evaluation."""

    def __init__(self):
        self.completed = False
        self.context_overflow_error = False
        self.large_tool_results_created = False
        self.agent_response = ""
        self.error_message = ""
        self.large_result_files = []

    def print_summary(self):
        """Print evaluation summary."""
        print("\n" + "=" * 80)
        print("LARGE TOOL RESULTS EVAL SUMMARY")
        print("=" * 80)
        print(f"Task Completed: {'✅ YES' if self.completed else '❌ NO'}")
        print(f"Context Overflow Error: {'❌ YES' if self.context_overflow_error else '✅ NO'}")
        print(f"Large Results Evicted: {'✅ YES' if self.large_tool_results_created else '❌ NO'}")

        if self.large_result_files:
            print(f"\nLarge Result Files Created: {len(self.large_result_files)}")
            for filepath in self.large_result_files:
                if Path(filepath).exists():
                    size = Path(filepath).stat().st_size
                    print(f"  - {filepath} ({size:,} bytes)")

        if self.agent_response:
            print(f"\nAgent Response Length: {len(self.agent_response)} characters")
            print(f"Response Preview: {self.agent_response[:200]}...")

        if self.error_message:
            print(f"\nError: {self.error_message}")

        print("=" * 80)

        # Determine pass/fail
        if self.completed and not self.context_overflow_error and self.large_tool_results_created:
            print("✅ EVAL PASSED: Context management working correctly")
            return 0
        else:
            print("❌ EVAL FAILED: Context management not working as expected")
            return 1


def run_eval(verbose: bool = True) -> EvalMetrics:
    """Run the large tool results evaluation.

    Args:
        verbose: Whether to print detailed progress

    Returns:
        EvalMetrics object with results
    """
    logger.info("Starting large tool results eval")

    metrics = EvalMetrics()

    # Create mock that returns very large payload
    def mock_ddg_search(query: str, **kwargs):
        logger.info(f"Mock DDG search invoked with query: '{query}'")
        large_payload = create_large_payload(target_tokens=80000)
        logger.info(f"Returning large payload: {len(large_payload)} characters (~{len(large_payload)//4} tokens)")
        # Return a list of dicts matching DDG's text() return format
        # but with a huge body so str() produces a large payload
        return [{"title": "Mock Result", "href": "https://example.com", "body": large_payload}]

    # Patch the DDGS.text method BEFORE creating the agent
    with patch('ddgs.DDGS.text', mock_ddg_search):
        with tempfile.TemporaryDirectory() as tmpdir:
            if verbose:
                print(f"\n{'=' * 80}")
                print("LARGE TOOL RESULTS CONTEXT MANAGEMENT EVAL")
                print(f"{'=' * 80}")
                print(f"Working Directory: {tmpdir}")
                print(f"Mock Tool: Returns ~80k tokens (over 78k limit)")
                print(f"Expected: FilesystemMiddleware evicts to /large_tool_results/")
                print(f"{'=' * 80}\n")

            try:
                # Create thread manager AFTER patching
                thread_manager = ThreadManager(root_dir=tmpdir)
                thread = thread_manager.new()

                if verbose:
                    print("📝 Asking agent to research president of France...")
                    print("   (This will trigger a search returning ~80k tokens)\n")

                # Ask the agent to do research
                # Make it explicit that internet research is needed
                query = "Who is the current president of France? Please search the internet for this information and provide their full name, term in office, and recent achievements."

                try:
                    response = thread.message(query)

                    # Get the response
                    messages = thread.get_messages()
                    agent_response = messages[-1]['content'] if messages else ""
                    metrics.agent_response = agent_response
                    metrics.completed = True

                    if verbose:
                        print(f"\n✅ Agent completed task without crashing!")
                        print(f"   Response length: {len(agent_response)} characters")
                        print(f"   Response preview: {agent_response[:200]}...")

                except Exception as e:
                    error_str = str(e)
                    metrics.error_message = error_str

                    # Check if it's a context overflow error
                    if "maximum context length" in error_str or "input tokens" in error_str:
                        metrics.context_overflow_error = True
                        if verbose:
                            print(f"\n❌ Context overflow error occurred:")
                            print(f"   {error_str}")
                    else:
                        if verbose:
                            print(f"\n⚠️  Different error occurred:")
                            print(f"   {error_str}")
                        raise

                # Check if large_tool_results files were created in agent state
                # FilesystemMiddleware writes to state["files"], not physical filesystem
                try:
                    # Get the final state from the thread
                    final_state = thread.agent.get_state({"configurable": {"thread_id": thread.thread_id}})
                    files_in_state = final_state.values.get("files", {})

                    # Look for files in /large_tool_results/
                    large_result_files = [path for path in files_in_state.keys() if path.startswith("/large_tool_results/")]

                    if large_result_files:
                        metrics.large_tool_results_created = True
                        metrics.large_result_files = large_result_files
                        if verbose:
                            print(f"\n✅ {len(large_result_files)} file(s) evicted to /large_tool_results/ in agent state")
                            for filepath in large_result_files[:3]:  # Show first 3
                                file_data = files_in_state[filepath]
                                if isinstance(file_data, dict):
                                    content = file_data.get("content", [])
                                    content_len = len("".join(content) if isinstance(content, list) else str(content))
                                else:
                                    content_len = len(str(file_data))
                                print(f"   - {filepath}: {content_len:,} chars")
                    else:
                        if verbose:
                            print(f"\n⚠️  No files in /large_tool_results/ in agent state")
                            if files_in_state:
                                print(f"   Files in state: {list(files_in_state.keys())[:5]}")
                            print("   (Large tool results may not have been evicted)")
                except Exception as e:
                    if verbose:
                        print(f"\n⚠️  Could not check agent state for large_tool_results: {e}")

            except Exception as e:
                logger.error(f"Eval failed with unexpected error: {e}", exc_info=True)
                metrics.error_message = str(e)

    return metrics


def main():
    """Run the evaluation and print results."""
    import argparse

    parser = argparse.ArgumentParser(description="Large tool results context management eval")
    parser.add_argument('--quiet', action='store_true', help='Suppress detailed output')
    parser.add_argument('--output', type=str, help='Save metrics to JSON file')

    args = parser.parse_args()

    try:
        metrics = run_eval(verbose=not args.quiet)
        exit_code = metrics.print_summary()

        # Save metrics if requested
        if args.output:
            output_path = Path(args.output)
            with open(output_path, 'w') as f:
                json.dump({
                    'completed': metrics.completed,
                    'context_overflow_error': metrics.context_overflow_error,
                    'large_tool_results_created': metrics.large_tool_results_created,
                    'agent_response_length': len(metrics.agent_response),
                    'large_result_files': metrics.large_result_files,
                    'error_message': metrics.error_message,
                }, f, indent=2)
            print(f"\nMetrics saved to: {output_path}")

        sys.exit(exit_code)

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        print(f"\n❌ EVAL FAILED: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
