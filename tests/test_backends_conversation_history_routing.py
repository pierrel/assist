"""Verify /conversation_history/ is routed via STATEFUL_PATHS to StateBackend.

The 2026-05-16 context-management overhaul adopts deepagents 0.6.1's
SummarizationMiddleware as the primary compaction path.  That middleware
offloads pre-summarization history to /conversation_history/{thread_id}.md
via the agent's backend.  Without explicit routing in STATEFUL_PATHS, the
offload would land at the default backend root — in sandbox mode that's
the host-bind-mounted workspace (user's git tree), in local mode the
working directory.  This test pins the routing so the file goes to
StateBackend (ephemeral, thread-local) instead.
"""
import pytest

from assist.backends import STATEFUL_PATHS, create_composite_backend


def test_conversation_history_in_stateful_paths():
    """All three path variants must be listed (matches the
    large_tool_results/ pattern: with-slashes, without-slashes, bare)."""
    assert "/conversation_history/" in STATEFUL_PATHS
    assert "conversation_history/" in STATEFUL_PATHS
    assert "conversation_history" in STATEFUL_PATHS


def test_large_tool_results_still_in_stateful_paths():
    """Sanity: the overhaul didn't disturb the existing
    large_tool_results/ routing that FilesystemMiddleware uses."""
    assert "/large_tool_results/" in STATEFUL_PATHS
    assert "large_tool_results/" in STATEFUL_PATHS
    assert "large_tool_results" in STATEFUL_PATHS


def test_composite_backend_routes_conversation_history_to_state(tmp_path):
    """Functional check: a /conversation_history/ write IS dispatched to
    StateBackend.  We can't actually exercise a write outside a LangGraph
    execution context (StateBackend raises RuntimeError when get_config()
    fails), but that very RuntimeError IS the proof of routing — if the
    composite backend had sent the write to the default FilesystemBackend
    instead, it would silently succeed."""
    # Must pass STATEFUL_PATHS — agent.py's _create_standard_backend does this;
    # the default is [] which routes nothing through state.
    backend = create_composite_backend(str(tmp_path), STATEFUL_PATHS)

    with pytest.raises(RuntimeError, match=r"StateBackend must be used inside"):
        backend.write("/conversation_history/test-thread.md", "# session intent")

    # The on-disk file should NOT exist either (the routed StateBackend
    # never gets to fall through to disk).
    host_path = tmp_path / "conversation_history" / "test-thread.md"
    assert not host_path.exists(), (
        f"conversation_history file leaked to disk at {host_path} despite the "
        f"StateBackend dispatch — STATEFUL_PATHS routing is fundamentally broken"
    )


def test_composite_backend_default_path_lands_on_disk(tmp_path):
    """Sanity counter-check: a non-stateful path WRITES TO disk
    (not routed via StateBackend).  Confirms the routing test above
    is meaningful — without this, every path could be silently
    routed and we wouldn't notice."""
    backend = create_composite_backend(str(tmp_path), STATEFUL_PATHS)
    result = backend.write("/notes/random.md", "scratch")
    assert result.error is None or result.error == ""
    # Default backend writes under tmp_path — this is the contrast case
    assert (tmp_path / "notes" / "random.md").exists(), (
        "non-stateful write didn't land on disk; either default backend "
        "is misconfigured or this whole test's premise is wrong"
    )
