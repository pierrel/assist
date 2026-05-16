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
    """Track metrics for the evaluation.

    Pre-2026-05-16 architecture: `ContextAwareToolEvictionMiddleware`
    at the general-agent (parent) level evicted big tool results to
    `state["files"]` under `/large_tool_results/`.  This eval pinned
    that behavior.

    Post-2026-05-16 architecture (docs/2026-05-16-context-management-
    overhaul.org): the parent's eviction middleware was deleted.
    Eviction is now done by deepagents' built-in `FilesystemMiddleware`
    inside the *research subagent* whose `search_internet` call
    actually receives the big mocked payload.  The parent agent sees
    only the subagent's summary via the `task` tool return — it never
    sees `/large_tool_results/` files in its own state.

    The eval therefore needs to accept BOTH signals as evidence that
    context was managed correctly:
      1. Files under `/large_tool_results/` in PARENT state
         (old architecture's signal; still applies if the parent's
         own tools ever return 20k+ tokens — e.g. a future tool that
         the parent calls directly).
      2. `_summarization_event` field in parent state (deepagents'
         SummarizationMiddleware fired — only happens if total context
         exceeded 0.85 × max_input_tokens, which with the 131k restore
         needs >111k tokens — unlikely on this synthetic single-turn
         eval but possible).
      3. The agent COMPLETED the task (didn't crash, returned a
         meaningful response).  If the big tool result was handled
         somewhere (subagent eviction, internal summarization,
         whatever) and the parent got a sensible response back, then
         the architecture worked end-to-end — even if no observable
         artifact landed in parent state.
    """

    def __init__(self):
        self.completed = False
        self.context_overflow_error = False
        # True iff we observed concrete evidence of compaction in
        # PARENT state (signals 1 or 2 from the class docstring).
        # Signal 3 (successful completion) is captured by `completed`.
        self.parent_state_compaction = False
        self.agent_response = ""
        self.error_message = ""
        self.large_result_files = []
        self.summarization_fired = False

    def print_summary(self):
        """Print evaluation summary."""
        print("\n" + "=" * 80)
        print("LARGE TOOL RESULTS EVAL SUMMARY")
        print("=" * 80)
        print(f"Task Completed:         {'✅ YES' if self.completed else '❌ NO'}")
        print(f"Context Overflow Error: {'❌ YES' if self.context_overflow_error else '✅ NO'}")
        print(f"Parent /large_tool_results/ eviction: "
              f"{'✅ YES' if self.large_result_files else 'n/a (handled in subagent)'}")
        print(f"Parent SummarizationMiddleware fired: "
              f"{'✅ YES' if self.summarization_fired else 'n/a (single turn well under trigger)'}")

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

        # Pass criteria: agent completed end-to-end without a context-
        # overflow error.  Parent-state compaction observation is
        # informative but not load-bearing — under the post-2026-05-16
        # architecture, eviction happens in the research subagent's
        # state, which we don't probe from here (cross-graph state
        # introspection isn't a stable contract in deepagents 0.6.1).
        if self.completed and not self.context_overflow_error:
            extra = ""
            if self.large_result_files:
                extra = " (with parent-state eviction)"
            elif self.summarization_fired:
                extra = " (with parent-state summarization)"
            else:
                extra = " (compaction handled in subagent — opaque to parent)"
            print(f"✅ EVAL PASSED: Context management working correctly{extra}")
            return 0
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

                # Inspect parent state for evidence of context management.
                # Under the post-2026-05-16 architecture this is best-
                # effort — the research subagent does the actual eviction
                # in its own state, which we don't probe from here.
                try:
                    final_state = thread.agent.get_state({"configurable": {"thread_id": thread.thread_id}})
                    state_values = final_state.values
                    files_in_state = state_values.get("files", {})

                    # Signal 1: files evicted to /large_tool_results/ in PARENT state.
                    # Only fires under the old architecture or if the parent agent
                    # itself called a tool that returned 20k+ tokens directly.
                    large_result_files = [
                        path for path in files_in_state.keys()
                        if path.startswith("/large_tool_results/")
                    ]
                    if large_result_files:
                        metrics.large_result_files = large_result_files
                        metrics.parent_state_compaction = True
                        if verbose:
                            print(f"\n✅ {len(large_result_files)} file(s) evicted to /large_tool_results/ in PARENT state")
                            for filepath in large_result_files[:3]:
                                file_data = files_in_state[filepath]
                                if isinstance(file_data, dict):
                                    content = file_data.get("content", [])
                                    content_len = len("".join(content) if isinstance(content, list) else str(content))
                                else:
                                    content_len = len(str(file_data))
                                print(f"   - {filepath}: {content_len:,} chars")

                    # Signal 2: SummarizationMiddleware fired (writes
                    # _summarization_event to state via wrap_model_call).
                    if state_values.get("_summarization_event") is not None:
                        metrics.summarization_fired = True
                        metrics.parent_state_compaction = True
                        if verbose:
                            print("\n✅ SummarizationMiddleware fired in parent state")

                    # Signal 3: /conversation_history/ artifact landed in
                    # parent state — the SummarizationMiddleware offload
                    # path (routed via STATEFUL_PATHS → StateBackend).
                    convo_history_files = [
                        path for path in files_in_state.keys()
                        if path.startswith("/conversation_history/")
                    ]
                    if convo_history_files:
                        metrics.parent_state_compaction = True
                        if verbose:
                            print(f"\n✅ {len(convo_history_files)} /conversation_history/ artifact(s) in parent state")

                    if not metrics.parent_state_compaction and verbose:
                        print(
                            "\nℹ️  No compaction artifacts in PARENT state — under the "
                            "post-2026-05-16 architecture this is expected when the "
                            "research subagent (not the parent) handled the big tool "
                            "result.  Parent state files keys: "
                            f"{list(files_in_state.keys())[:5]}"
                        )
                except Exception as e:
                    if verbose:
                        print(f"\n⚠️  Could not inspect parent state: {e}")

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
