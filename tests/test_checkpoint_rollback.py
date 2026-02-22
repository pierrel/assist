"""Tests for checkpoint-based rollback on unrecoverable model errors.

These are deterministic unit tests using mocked agents — they do not
require a running LLM or vLLM server.
"""
import httpx
import pytest
from unittest.mock import Mock, patch, PropertyMock
from openai import BadRequestError

from assist.checkpoint_rollback import invoke_with_rollback, RollbackRunnable


def _make_bad_request_error(msg="Expecting ':' delimiter"):
    """Create a realistic BadRequestError like vLLM returns."""
    req = httpx.Request("POST", "http://localhost/v1/chat/completions")
    resp = httpx.Response(400, json={"error": {"message": msg}}, request=req)
    return BadRequestError(msg, response=resp, body={"error": {"message": msg}})


def _make_checkpoint(checkpoint_id, step=0):
    """Create a mock StateSnapshot with a given checkpoint_id."""
    snapshot = Mock()
    snapshot.config = {
        "configurable": {
            "thread_id": "test-thread",
            "checkpoint_id": checkpoint_id,
        }
    }
    snapshot.metadata = {"step": step}
    return snapshot


class TestInvokeWithRollback:
    """Tests for invoke_with_rollback()."""

    def test_passes_through_on_success(self):
        """No error — should return the agent result directly."""
        agent = Mock()
        agent.invoke.return_value = {"messages": ["ok"]}

        result = invoke_with_rollback(
            agent,
            {"messages": [{"role": "user", "content": "hi"}]},
            {"configurable": {"thread_id": "t1"}},
        )

        assert result == {"messages": ["ok"]}
        assert agent.invoke.call_count == 1
        agent.get_state_history.assert_not_called()

    def test_recovers_from_single_failure(self):
        """One BadRequestError, then rollback succeeds on retry."""
        agent = Mock()
        agent.invoke.side_effect = [
            _make_bad_request_error(),            # 1st call: fails
            {"messages": ["recovered"]},          # 2nd call (from rollback): succeeds
        ]
        agent.get_state_history.return_value = [
            _make_checkpoint("cp-bad", step=1),   # Most recent (bad state)
            _make_checkpoint("cp-good", step=0),  # Previous (good state)
        ]

        result = invoke_with_rollback(
            agent,
            {"messages": [{"role": "user", "content": "hi"}]},
            {"configurable": {"thread_id": "t1"}},
        )

        assert result == {"messages": ["recovered"]}
        assert agent.invoke.call_count == 2
        # Second invoke should use the rollback checkpoint config
        second_call_config = agent.invoke.call_args_list[1][0][1]
        assert second_call_config["configurable"]["checkpoint_id"] == "cp-good"
        # Second invoke should pass None as input (resume from checkpoint)
        second_call_input = agent.invoke.call_args_list[1][0][0]
        assert second_call_input is None

    def test_retries_at_same_depth_before_going_deeper(self):
        """Same checkpoint should be retried max_retries_per_step times
        before rolling back further."""
        agent = Mock()
        agent.invoke.side_effect = [
            _make_bad_request_error(),            # 1st: initial fail
            _make_bad_request_error(),            # 2nd: retry at cp-1 fails
            _make_bad_request_error(),            # 3rd: retry at cp-1 fails again
            {"messages": ["deep recovery"]},      # 4th: cp-0 succeeds
        ]
        agent.get_state_history.return_value = [
            _make_checkpoint("cp-bad", step=2),
            _make_checkpoint("cp-1", step=1),
            _make_checkpoint("cp-0", step=0),
        ]

        result = invoke_with_rollback(
            agent,
            {"messages": [{"role": "user", "content": "hi"}]},
            {"configurable": {"thread_id": "t1"}},
            max_retries_per_step=2,
        )

        assert result == {"messages": ["deep recovery"]}
        assert agent.invoke.call_count == 4

        # 4th call should use cp-0 (the deeper checkpoint)
        fourth_config = agent.invoke.call_args_list[3][0][1]
        assert fourth_config["configurable"]["checkpoint_id"] == "cp-0"

    def test_raises_when_all_rollbacks_exhausted(self):
        """All checkpoints × retries exhausted — raises the original error."""
        agent = Mock()
        agent.invoke.side_effect = _make_bad_request_error()
        agent.get_state_history.return_value = [
            _make_checkpoint("cp-bad", step=1),
            _make_checkpoint("cp-0", step=0),
        ]

        with pytest.raises(BadRequestError):
            invoke_with_rollback(
                agent,
                {"messages": [{"role": "user", "content": "hi"}]},
                {"configurable": {"thread_id": "t1"}},
                max_retries_per_step=1,
                max_rollback_depth=1,
            )

    def test_non_matching_errors_propagate_immediately(self):
        """Errors not in rollback_on should propagate without any rollback."""
        agent = Mock()
        agent.invoke.side_effect = RuntimeError("something else entirely")

        with pytest.raises(RuntimeError, match="something else entirely"):
            invoke_with_rollback(
                agent,
                {"messages": [{"role": "user", "content": "hi"}]},
                {"configurable": {"thread_id": "t1"}},
            )

        assert agent.invoke.call_count == 1
        agent.get_state_history.assert_not_called()

    def test_raises_immediately_when_no_history(self):
        """If there are no checkpoints at all, raise without retrying."""
        agent = Mock()
        agent.invoke.side_effect = _make_bad_request_error()
        agent.get_state_history.return_value = []

        with pytest.raises(BadRequestError):
            invoke_with_rollback(
                agent,
                {"messages": [{"role": "user", "content": "hi"}]},
                {"configurable": {"thread_id": "t1"}},
            )

        assert agent.invoke.call_count == 1

    def test_raises_when_only_bad_checkpoint_exists(self):
        """If only the bad checkpoint exists (no earlier state), raise."""
        agent = Mock()
        agent.invoke.side_effect = _make_bad_request_error()
        agent.get_state_history.return_value = [
            _make_checkpoint("cp-bad", step=0),
        ]

        with pytest.raises(BadRequestError):
            invoke_with_rollback(
                agent,
                {"messages": [{"role": "user", "content": "hi"}]},
                {"configurable": {"thread_id": "t1"}},
            )

        assert agent.invoke.call_count == 1

    def test_max_rollback_depth_limits_how_far_back(self):
        """Should not go deeper than max_rollback_depth even if more
        checkpoints are available."""
        agent = Mock()
        # Fail 5 times (more than we should attempt)
        agent.invoke.side_effect = _make_bad_request_error()
        agent.get_state_history.return_value = [
            _make_checkpoint("cp-bad", step=4),
            _make_checkpoint("cp-3", step=3),
            _make_checkpoint("cp-2", step=2),
            _make_checkpoint("cp-1", step=1),
            _make_checkpoint("cp-0", step=0),
        ]

        with pytest.raises(BadRequestError):
            invoke_with_rollback(
                agent,
                {"messages": [{"role": "user", "content": "hi"}]},
                {"configurable": {"thread_id": "t1"}},
                max_retries_per_step=1,
                max_rollback_depth=2,
            )

        # 1 initial + 1 retry at cp-3 + 1 retry at cp-2 = 3 attempts
        # Should NOT try cp-1 or cp-0 because depth limit is 2
        assert agent.invoke.call_count == 3


class TestRollbackRunnable:
    """Tests for the RollbackRunnable wrapper used with subagents."""

    def test_invoke_delegates_on_success(self):
        """Successful invoke passes through to the underlying agent."""
        agent = Mock()
        agent.invoke.return_value = {"messages": ["ok"]}

        wrapper = RollbackRunnable(agent)
        result = wrapper.invoke(
            {"messages": [{"role": "user", "content": "hi"}]},
            {"configurable": {"thread_id": "sub-1"}},
        )

        assert result == {"messages": ["ok"]}
        assert agent.invoke.call_count == 1

    def test_invoke_rolls_back_on_bad_request(self):
        """BadRequestError triggers rollback inside the subagent."""
        agent = Mock()
        agent.invoke.side_effect = [
            _make_bad_request_error(),
            {"messages": ["recovered"]},
        ]
        agent.get_state_history.return_value = [
            _make_checkpoint("cp-bad", step=1),
            _make_checkpoint("cp-good", step=0),
        ]

        wrapper = RollbackRunnable(agent)
        result = wrapper.invoke(
            {"messages": [{"role": "user", "content": "hi"}]},
            {"configurable": {"thread_id": "sub-1"}},
        )

        assert result == {"messages": ["recovered"]}
        assert agent.invoke.call_count == 2

    def test_proxies_other_attributes(self):
        """Attributes not on RollbackRunnable are proxied to the agent."""
        agent = Mock()
        agent.name = "test-agent"
        agent.get_state.return_value = {"values": {}}

        wrapper = RollbackRunnable(agent)
        assert wrapper.name == "test-agent"
        wrapper.get_state({"configurable": {"thread_id": "t1"}})
        agent.get_state.assert_called_once()

    def test_respects_custom_rollback_params(self):
        """Custom max_retries_per_step and max_rollback_depth are honoured."""
        agent = Mock()
        agent.invoke.side_effect = _make_bad_request_error()
        agent.get_state_history.return_value = [
            _make_checkpoint("cp-bad", step=2),
            _make_checkpoint("cp-1", step=1),
            _make_checkpoint("cp-0", step=0),
        ]

        wrapper = RollbackRunnable(
            agent,
            max_retries_per_step=1,
            max_rollback_depth=1,
        )

        with pytest.raises(BadRequestError):
            wrapper.invoke(
                {"messages": [{"role": "user", "content": "hi"}]},
                {"configurable": {"thread_id": "sub-1"}},
            )

        # 1 initial + 1 retry at cp-1 = 2 (depth limit 1 prevents going to cp-0)
        assert agent.invoke.call_count == 2

    def test_invoke_with_no_config(self):
        """If config is None, an empty dict is passed (no crash)."""
        agent = Mock()
        agent.invoke.return_value = {"messages": ["ok"]}

        wrapper = RollbackRunnable(agent)
        result = wrapper.invoke({"messages": [{"role": "user", "content": "hi"}]})

        assert result == {"messages": ["ok"]}
