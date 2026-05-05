"""Evaluation: Large tool results context management.

This eval tests that the agent can handle tool results exceeding the context limit
by using FilesystemMiddleware to evict large results to files.

The test mocks the internet search tool to return a payload >78k tokens, then verifies:
1. The agent invokes ``search_internet`` (precondition; otherwise INCONCLUSIVE).
2. The agent doesn't crash with a context overflow error.
3. Large tool results are evicted somewhere — either:
   - the existing path: ``/large_tool_results/`` entries in agent state (the
     ``ContextAwareToolEvictionMiddleware`` writes here via ``StateBackend``), or
   - the new path: on-disk blobs under ``<tmpdir>/<tid>/large_tool_results/``
     that Layer 3 (``EvictionSaver``) writes at checkpoint-write time.
4. The agent can still complete the research task.

The model under test is small enough to answer many factual queries from
training memory and skip search entirely.  The prompt is engineered to be
unanswerable without the search result (it asks for the title of the first
search result, which is only known via the mock) and is explicit that the
tool MUST be used.
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
        # Did the model actually invoke the mocked search?  When 0 the
        # eviction path was never exercised — the eval is INCONCLUSIVE,
        # not a fail.
        self.mock_search_invoked = 0
        # Layer 3 (EvictionSaver) writes evicted blobs to disk under
        # <tmpdir>/<tid>/large_tool_results/<sha256_16>.  Independent
        # signal from the in-state files-channel check.
        self.layer3_disk_blobs: list[str] = []

    def print_summary(self):
        """Print evaluation summary."""
        print("\n" + "=" * 80)
        print("LARGE TOOL RESULTS EVAL SUMMARY")
        print("=" * 80)
        print(f"Mock Search Invoked: {self.mock_search_invoked} time(s)")
        print(f"Task Completed: {'✅ YES' if self.completed else '❌ NO'}")
        print(f"Context Overflow Error: {'❌ YES' if self.context_overflow_error else '✅ NO'}")
        print(f"Large Results Evicted (state): {'✅ YES' if self.large_tool_results_created else '❌ NO'}")
        print(f"Layer 3 Disk Blobs: {len(self.layer3_disk_blobs)}")

        if self.large_result_files:
            print(f"\nLarge Result Files Created (state): {len(self.large_result_files)}")
            for filepath in self.large_result_files:
                if Path(filepath).exists():
                    size = Path(filepath).stat().st_size
                    print(f"  - {filepath} ({size:,} bytes)")

        if self.layer3_disk_blobs:
            print(f"\nLayer 3 Eviction Blobs ({len(self.layer3_disk_blobs)}):")
            for blob in self.layer3_disk_blobs[:3]:
                size = Path(blob).stat().st_size if Path(blob).exists() else 0
                print(f"  - {blob} ({size:,} bytes)")

        if self.agent_response:
            print(f"\nAgent Response Length: {len(self.agent_response)} characters")
            print(f"Response Preview: {self.agent_response[:200]}...")

        if self.error_message:
            print(f"\nError: {self.error_message}")

        print("=" * 80)

        if self.context_overflow_error:
            print("❌ EVAL FAILED: Context overflow error")
            return 1
        if self.mock_search_invoked == 0:
            print("⚠️  EVAL INCONCLUSIVE: model did not invoke search tool — "
                  "eviction path not exercised")
            return 2
        evicted = self.large_tool_results_created or bool(self.layer3_disk_blobs)
        if self.completed and evicted:
            print("✅ EVAL PASSED: Context management working correctly")
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
    mock_invocations = [0]
    # Sentinel string buried inside the mocked search result.  The
    # eval prompt asks the agent to find this, which it can only do by
    # actually invoking the search tool (no model would emit a random
    # 12-character token from training).  Keeps the eval honest even
    # if a future model decides to confidently fabricate.
    SENTINEL = "MOCK-RESULT-7K3F9P"

    def mock_ddg_search(query: str, **kwargs):
        mock_invocations[0] += 1
        logger.info(
            f"Mock DDG search invoked (#{mock_invocations[0]}) with query: '{query}'"
        )
        large_payload = create_large_payload(target_tokens=80000)
        # Inject the sentinel near the top of the body so the agent
        # finds it after a single read.
        large_payload = (
            f"SEARCH-VERIFICATION-CODE: {SENTINEL}\n\n" + large_payload
        )
        logger.info(
            f"Returning large payload: {len(large_payload)} characters "
            f"(~{len(large_payload)//4} tokens) with sentinel {SENTINEL}"
        )
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

                # Force the model into the search path.  Past runs
                # showed Qwen3.6-27B answers factual questions from
                # training memory and skips ``search_internet``
                # entirely, so the eviction path was never exercised.
                # The query below is unanswerable without invoking
                # the mocked tool: it asks for a verification code
                # buried in the mock's response body.
                query = (
                    "I need you to use the search_internet tool to look "
                    "up information about France's government.  The search "
                    "result will contain a SEARCH-VERIFICATION-CODE near "
                    "the top of the body.  Find the code and report it "
                    "back to me verbatim along with a brief summary of "
                    "what was returned.  You MUST use search_internet — "
                    "do not answer from your training data."
                )

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

                # Record mock invocations for the precondition check.
                metrics.mock_search_invoked = mock_invocations[0]
                if verbose:
                    print(f"\nMock search invocations: {metrics.mock_search_invoked}")

                # Check if large_tool_results files were created in agent state.
                # ContextAwareToolEvictionMiddleware writes to state["files"]
                # via StateBackend; entries land under /large_tool_results/.
                try:
                    final_state = thread.agent.get_state({"configurable": {"thread_id": thread.thread_id}})
                    files_in_state = final_state.values.get("files", {})

                    large_result_files = [path for path in files_in_state.keys() if path.startswith("/large_tool_results/")]

                    if large_result_files:
                        metrics.large_tool_results_created = True
                        metrics.large_result_files = large_result_files
                        if verbose:
                            print(f"\n✅ {len(large_result_files)} file(s) in /large_tool_results/ in agent state")
                            for filepath in large_result_files[:3]:
                                file_data = files_in_state[filepath]
                                if isinstance(file_data, dict):
                                    content = file_data.get("content", [])
                                    content_len = len("".join(content) if isinstance(content, list) else str(content))
                                else:
                                    content_len = len(str(file_data))
                                print(f"   - {filepath}: {content_len:,} chars")
                    elif verbose:
                        print(f"\n⚠️  No files in /large_tool_results/ in agent state")
                        if files_in_state:
                            print(f"   Files in state: {list(files_in_state.keys())[:5]}")
                except Exception as e:
                    if verbose:
                        print(f"\n⚠️  Could not check agent state for large_tool_results: {e}")

                # Layer 3 (EvictionSaver) writes evicted blobs to disk under
                # <tmpdir>/<tid>/large_tool_results/<sha256_16>.  Independent
                # of the in-state check above — Layer 3 evicts both the
                # messages channel and the files channel at checkpoint write
                # time, so this directory should have at least one blob if
                # eviction triggered.
                try:
                    import glob
                    pattern = os.path.join(tmpdir, "*", "large_tool_results", "*")
                    blobs = sorted(p for p in glob.glob(pattern) if os.path.isfile(p))
                    metrics.layer3_disk_blobs = blobs
                    if verbose and blobs:
                        print(f"\n✅ Layer 3 wrote {len(blobs)} eviction blob(s) to disk:")
                        for blob in blobs[:3]:
                            print(f"   - {blob} ({Path(blob).stat().st_size:,} bytes)")
                    elif verbose:
                        print(f"\n⚠️  No Layer 3 eviction blobs on disk under {tmpdir}")
                except Exception as e:
                    if verbose:
                        print(f"\n⚠️  Could not scan disk for Layer 3 blobs: {e}")

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
