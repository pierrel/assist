"""Tests for LoopDetectionMiddleware."""
import logging
from unittest.mock import Mock

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from assist.middleware.loop_detection import (
    LoopDetectionMiddleware,
    _compose_terminal_message,
    _detect_loop,
    _extract_events,
    _last_error_excerpt,
    _last_successful_artifact,
    _normalise_args,
    _normalise_error,
)


def _ai_with_call(tc_id: str, name: str, args: dict, content: str = "") -> AIMessage:
    msg = AIMessage(content=content)
    msg.tool_calls = [{"id": tc_id, "name": name, "args": args}]
    return msg


def _tool_msg(tc_id: str, content: str, status: str = "success") -> ToolMessage:
    return ToolMessage(content=content, tool_call_id=tc_id, status=status)


# ---------------------------------------------------------------------------
# Pure-function tests
# ---------------------------------------------------------------------------

class TestNormalisation:
    def test_normalise_error_collapses_paths(self):
        a = _normalise_error(
            "Cannot write to /workspace/final_report.md because it already exists."
        )
        b = _normalise_error(
            "Cannot write to /workspace/completed_final_report.md because it already exists."
        )
        assert a == b

    def test_normalise_error_collapses_numbers(self):
        a = _normalise_error("Error: timeout after 30 seconds")
        b = _normalise_error("Error: timeout after 60 seconds")
        assert a == b

    def test_normalise_args_stable_across_key_order(self):
        a = _normalise_args({"file_path": "/x", "content": "hi"})
        b = _normalise_args({"content": "hi", "file_path": "/x"})
        assert a == b

    def test_normalise_args_distinguishes_different_paths(self):
        a = _normalise_args({"file_path": "/a", "content": "hi"})
        b = _normalise_args({"file_path": "/b", "content": "hi"})
        assert a != b


class TestExtractEvents:
    def test_pairs_tool_calls_with_results(self):
        messages = [
            HumanMessage(content="go"),
            _ai_with_call("c1", "write_file", {"file_path": "/a"}),
            _tool_msg("c1", "Cannot write to /a because it already exists."),
            _ai_with_call("c2", "write_file", {"file_path": "/b"}),
            _tool_msg("c2", "Cannot write to /b because it already exists."),
        ]
        events = _extract_events(messages, window=12)
        assert len(events) == 2
        assert all(e["completed"] for e in events)
        assert all(e["is_error"] for e in events)
        assert all(e["tool_name"] == "write_file" for e in events)

    def test_marks_unmatched_call_as_incomplete(self):
        messages = [
            _ai_with_call("c1", "write_file", {"file_path": "/a"}),
            _tool_msg("c1", "ok"),
            _ai_with_call("c2", "write_file", {"file_path": "/b"}),
        ]
        events = _extract_events(messages, window=12)
        assert len(events) == 2
        assert events[0]["completed"] is True
        assert events[1]["completed"] is False

    def test_window_caps_results(self):
        messages = []
        for i in range(20):
            tc_id = f"c{i}"
            messages.append(_ai_with_call(tc_id, "x", {"i": i}))
            messages.append(_tool_msg(tc_id, "ok"))
        events = _extract_events(messages, window=5)
        assert len(events) == 5

    def test_status_error_treated_as_error_even_without_prefix(self):
        messages = [
            _ai_with_call("c1", "x", {}),
            _tool_msg("c1", "boom", status="error"),
        ]
        events = _extract_events(messages, window=12)
        assert events[0]["is_error"] is True


# ---------------------------------------------------------------------------
# Detection tests
# ---------------------------------------------------------------------------

class TestDetectLoop:
    def _evt(self, tool="write_file", args=None, content="ok", is_error=False):
        return {
            "tool_name": tool,
            "args_sig": _normalise_args(args or {}),
            "result_content": content,
            "is_error": is_error,
            "completed": True,
        }

    def test_no_loop_on_empty_history(self):
        assert _detect_loop([], 2, 3, 3, 10) is None

    def test_pattern_a_two_same_errors_in_a_row(self):
        events = [
            self._evt(content="Cannot write to /a because it already exists.", is_error=True),
            self._evt(content="Cannot write to /b because it already exists.", is_error=True),
        ]
        result = _detect_loop(events, 2, 3, 3, 10)
        assert result is not None
        assert result["pattern"] == "same-tool-same-error"
        assert result["tools"] == {"write_file"}
        assert result["run_length"] == 2

    def test_pattern_a_does_not_fire_on_single_error(self):
        events = [
            self._evt(content="ok"),
            self._evt(content="Cannot write to /a because it already exists.", is_error=True),
        ]
        assert _detect_loop(events, 2, 3, 3, 10) is None

    def test_pattern_a_breaks_on_intervening_success(self):
        events = [
            self._evt(content="Cannot write to /a because it already exists.", is_error=True),
            self._evt(content="ok"),
            self._evt(content="Cannot write to /b because it already exists.", is_error=True),
        ]
        # Trailing run of errors is length 1 only — no Pattern A.
        # Pattern C also doesn't fire (only 2 distinct args within window).
        # But it might fire other patterns; let's just check Pattern A doesn't fire.
        result = _detect_loop(events, 2, 3, 3, 10)
        # A non-error in between breaks the trailing run; if other patterns
        # also don't fire, result should be None.
        # Pattern C: 3 distinct args (default different per-evt args) over 3 events
        # would fire. So provide identical args to isolate pattern A behaviour.
        events_same_args = [
            self._evt(args={"file_path": "/a"}, content="Cannot write to /a because it already exists.", is_error=True),
            self._evt(args={"file_path": "/a"}, content="ok"),
            self._evt(args={"file_path": "/a"}, content="Cannot write to /a because it already exists.", is_error=True),
        ]
        # Trailing run is just 1 error → Pattern A won't fire
        # 3 same-args-same-tool in a row → Pattern B fires at threshold 3
        result2 = _detect_loop(events_same_args, 2, 3, 3, 10)
        assert result2 is not None
        assert result2["pattern"] == "same-tool-same-args"

    def test_pattern_b_three_identical_calls(self):
        evt = self._evt(args={"file_path": "/x"}, content="ok")
        events = [evt, evt.copy(), evt.copy()]
        result = _detect_loop(events, 2, 3, 3, 10)
        assert result is not None
        assert result["pattern"] == "same-tool-same-args"

    def test_pattern_b_does_not_fire_on_two(self):
        evt = self._evt(args={"file_path": "/x"})
        # Need different args so no pattern fires
        events = [evt, evt.copy()]
        # Pattern A: not all errors → no
        # Pattern B: only 2 → no (threshold 3)
        # Pattern C: 1 distinct over 2 → no (threshold 3)
        assert _detect_loop(events, 2, 3, 3, 10) is None

    def test_pattern_c_filename_mutation(self):
        # Three distinct paths with mixed errors that don't normalise alike,
        # so Pattern A doesn't fire. Pattern C catches it because the tool is
        # mutating and at least one call errored.
        events = [
            self._evt(args={"file_path": "/a"},
                      content="Error: permission denied", is_error=True),
            self._evt(args={"file_path": "/b"},
                      content="Error: disk full", is_error=True),
            self._evt(args={"file_path": "/c"},
                      content="Error: already exists", is_error=True),
        ]
        result = _detect_loop(events, 2, 3, 3, 10)
        assert result is not None
        assert result["pattern"] == "distinct-args-thrash"

    def test_pattern_c_does_not_fire_with_two_distinct(self):
        events = [
            self._evt(args={"file_path": "/a"}),
            self._evt(args={"file_path": "/b"}),
        ]
        assert _detect_loop(events, 2, 3, 3, 10) is None

    def test_pattern_c_does_not_fire_for_read_only_tool(self):
        # Three distinct read_file calls is exploration, not a loop —
        # regression for thread 20260425110043-4822cf50.
        events = [
            self._evt(tool="read_file", args={"file_path": "/a"}, content="<contents of /a>"),
            self._evt(tool="read_file", args={"file_path": "/b"}, content="<contents of /b>"),
            self._evt(tool="read_file", args={"file_path": "/c"}, content="<contents of /c>"),
        ]
        assert _detect_loop(events, 2, 3, 3, 10) is None

    def test_pattern_c_does_not_fire_without_error(self):
        # Three successful distinct write_file calls is normal
        # multi-file work, not a thrash.
        events = [
            self._evt(args={"file_path": "/a"}, content="Wrote /a"),
            self._evt(args={"file_path": "/b"}, content="Wrote /b"),
            self._evt(args={"file_path": "/c"}, content="Wrote /c"),
        ]
        assert _detect_loop(events, 2, 3, 3, 10) is None

    def test_different_tools_do_not_interleave(self):
        # Pattern C identifies the offending tool name when 3 distinct
        # mutating calls share the same name. Custom tool name avoids the
        # read-only allowlist.
        events = [
            self._evt(tool="custom_mutator", args={"q": "a"},
                      content="Error: failed in mode A", is_error=True),
            self._evt(tool="custom_mutator", args={"q": "b"},
                      content="Error: failed in mode B", is_error=True),
            self._evt(tool="custom_mutator", args={"q": "c"},
                      content="Error: failed in mode C", is_error=True),
        ]
        result = _detect_loop(events, 2, 3, 3, 10)
        assert result is not None
        assert result["tools"] == {"custom_mutator"}


class TestLastSuccessfulArtifact:
    def test_returns_path_of_last_success(self):
        messages = [
            _ai_with_call("c1", "write_file", {"file_path": "/report.md", "content": "..."}),
            _tool_msg("c1", "Wrote /report.md"),
            _ai_with_call("c2", "write_file", {"file_path": "/dup.md", "content": "..."}),
            _tool_msg("c2", "Cannot write to /dup.md because it already exists."),
        ]
        assert _last_successful_artifact(messages) == "/report.md"

    def test_returns_none_when_no_successes(self):
        messages = [
            _ai_with_call("c1", "write_file", {"file_path": "/a"}),
            _tool_msg("c1", "Cannot write to /a because it already exists."),
        ]
        assert _last_successful_artifact(messages) is None


class TestLastErrorExcerpt:
    def test_returns_first_line_of_recent_error(self):
        messages = [
            _ai_with_call("c1", "write_file", {"file_path": "/a"}),
            _tool_msg("c1", "Cannot write to /a because it already exists.\nDetails..."),
        ]
        excerpt = _last_error_excerpt(messages, {"write_file"})
        assert excerpt == "Cannot write to /a because it already exists."

    def test_truncates_long_excerpts(self):
        long = "Error: " + ("x" * 500)
        messages = [
            _ai_with_call("c1", "write_file", {"file_path": "/a"}),
            _tool_msg("c1", long),
        ]
        excerpt = _last_error_excerpt(messages, {"write_file"}, max_chars=100)
        assert len(excerpt) <= 100
        assert excerpt.endswith("…")

    def test_returns_none_when_no_error(self):
        messages = [
            _ai_with_call("c1", "write_file", {"file_path": "/a"}),
            _tool_msg("c1", "Wrote /a"),
        ]
        assert _last_error_excerpt(messages, {"write_file"}) is None

    def test_skips_non_matching_tools(self):
        messages = [
            _ai_with_call("c1", "search", {"q": "x"}),
            _tool_msg("c1", "Error: search failed"),
        ]
        assert _last_error_excerpt(messages, {"write_file"}) is None


class TestComposeTerminalMessage:
    def test_with_artifact_clean_completion(self):
        detection = {
            "pattern": "same-tool-same-error",
            "reason": "...",
            "tools": {"write_file"},
            "run_length": 2,
        }
        messages = [
            _ai_with_call("c0", "write_file", {"file_path": "/report.md"}),
            _tool_msg("c0", "Wrote /report.md"),
            _ai_with_call("c1", "write_file", {"file_path": "/dup"}),
            _tool_msg("c1", "Cannot write to /dup because it already exists."),
        ]
        msg = _compose_terminal_message(detection, messages)
        assert "/report.md" in msg
        assert "saved" in msg.lower()
        # Reads as the agent's voice.
        assert msg.startswith("I")

    def test_pattern_a_without_artifact_includes_error_excerpt(self):
        detection = {
            "pattern": "same-tool-same-error",
            "reason": "...",
            "tools": {"write_file"},
            "run_length": 2,
        }
        messages = [
            _ai_with_call("c1", "write_file", {"file_path": "/a"}),
            _tool_msg("c1", "Cannot write to /a because it already exists."),
            _ai_with_call("c2", "write_file", {"file_path": "/b"}),
            _tool_msg("c2", "Cannot write to /b because it already exists."),
        ]
        msg = _compose_terminal_message(detection, messages)
        assert "write_file" in msg
        assert "already exists" in msg
        assert "won't retry" in msg
        assert "direction" in msg.lower()

    def test_pattern_b_without_artifact(self):
        detection = {
            "pattern": "same-tool-same-args",
            "reason": "...",
            "tools": {"search"},
            "run_length": 3,
        }
        messages = [
            _ai_with_call("c1", "search", {"q": "x"}),
            _tool_msg("c1", "no results"),
            _ai_with_call("c2", "search", {"q": "x"}),
            _tool_msg("c2", "no results"),
            _ai_with_call("c3", "search", {"q": "x"}),
            _tool_msg("c3", "no results"),
        ]
        msg = _compose_terminal_message(detection, messages)
        assert "search" in msg
        assert "won't repeat" in msg

    def test_pattern_c_without_artifact_includes_excerpt_when_error(self):
        detection = {
            "pattern": "distinct-args-thrash",
            "reason": "...",
            "tools": {"write_file"},
            "run_length": 3,
        }
        messages = [
            _ai_with_call("c1", "write_file", {"file_path": "/a"}),
            _tool_msg("c1", "Cannot write to /a because it already exists."),
            _ai_with_call("c2", "write_file", {"file_path": "/b"}),
            _tool_msg("c2", "Cannot write to /b because it already exists."),
            _ai_with_call("c3", "write_file", {"file_path": "/c"}),
            _tool_msg("c3", "Cannot write to /c because it already exists."),
        ]
        msg = _compose_terminal_message(detection, messages)
        assert "write_file" in msg
        assert "different inputs" in msg
        assert "already exists" in msg
        assert "won't keep trying" in msg


# ---------------------------------------------------------------------------
# Middleware integration tests
# ---------------------------------------------------------------------------

class TestLoopDetectionMiddleware:
    def test_strips_tool_calls_when_loop_detected(self):
        mw = LoopDetectionMiddleware()
        last = _ai_with_call("c3", "write_file", {"file_path": "/c"})
        messages = [
            HumanMessage(content="go"),
            _ai_with_call("c1", "write_file", {"file_path": "/a"}),
            _tool_msg("c1", "Cannot write to /a because it already exists."),
            _ai_with_call("c2", "write_file", {"file_path": "/b"}),
            _tool_msg("c2", "Cannot write to /b because it already exists."),
            last,
        ]
        result = mw.after_model({"messages": messages}, Mock())
        assert result is not None
        new_last = result["messages"][-1]
        assert new_last.tool_calls == []
        # Without an artifact, message names the tool and asks for direction.
        assert "write_file" in new_last.content
        assert "more direction" in new_last.content.lower()

    def test_no_action_when_no_loop(self):
        mw = LoopDetectionMiddleware()
        last = _ai_with_call("c2", "write_file", {"file_path": "/b"})
        messages = [
            HumanMessage(content="go"),
            _ai_with_call("c1", "search", {"q": "x"}),
            _tool_msg("c1", "result"),
            last,
        ]
        assert mw.after_model({"messages": messages}, Mock()) is None

    def test_no_action_when_last_message_not_ai(self):
        mw = LoopDetectionMiddleware()
        messages = [HumanMessage(content="hi")]
        assert mw.after_model({"messages": messages}, Mock()) is None

    def test_no_action_when_ai_has_no_tool_calls(self):
        mw = LoopDetectionMiddleware()
        messages = [
            HumanMessage(content="hi"),
            AIMessage(content="all done"),
        ]
        assert mw.after_model({"messages": messages}, Mock()) is None

    def test_does_not_strip_when_new_call_is_unrelated_tool(self):
        """Loop detected for write_file, but model switched to read_file."""
        mw = LoopDetectionMiddleware()
        last = _ai_with_call("c3", "read_file", {"file_path": "/x"})
        messages = [
            _ai_with_call("c1", "write_file", {"file_path": "/a"}),
            _tool_msg("c1", "Cannot write to /a because it already exists."),
            _ai_with_call("c2", "write_file", {"file_path": "/b"}),
            _tool_msg("c2", "Cannot write to /b because it already exists."),
            last,
        ]
        # Model is breaking out — let it run.
        assert mw.after_model({"messages": messages}, Mock()) is None

    def test_terminal_message_includes_last_artifact(self):
        mw = LoopDetectionMiddleware()
        messages = [
            _ai_with_call("c0", "write_file", {"file_path": "/canonical.md", "content": "x"}),
            _tool_msg("c0", "Wrote /canonical.md"),
            _ai_with_call("c1", "write_file", {"file_path": "/dup1"}),
            _tool_msg("c1", "Cannot write to /dup1 because it already exists."),
            _ai_with_call("c2", "write_file", {"file_path": "/dup2"}),
            _tool_msg("c2", "Cannot write to /dup2 because it already exists."),
            _ai_with_call("c3", "write_file", {"file_path": "/dup3"}),
        ]
        result = mw.after_model({"messages": messages}, Mock())
        assert result is not None
        content = result["messages"][-1].content
        assert "/canonical.md" in content
        # With an artifact, message reads as a clean completion.
        assert "saved" in content.lower()
        # No system-log voice.
        assert "loop detected" not in content.lower()
        assert "[" not in content

    def test_clears_additional_kwargs_tool_calls(self):
        mw = LoopDetectionMiddleware()
        last = _ai_with_call("c3", "write_file", {"file_path": "/c"})
        last.additional_kwargs = {
            "tool_calls": [{"id": "c3", "function": {"name": "write_file", "arguments": "{}"}}]
        }
        messages = [
            _ai_with_call("c1", "write_file", {"file_path": "/a"}),
            _tool_msg("c1", "Cannot write to /a because it already exists."),
            _ai_with_call("c2", "write_file", {"file_path": "/b"}),
            _tool_msg("c2", "Cannot write to /b because it already exists."),
            last,
        ]
        result = mw.after_model({"messages": messages}, Mock())
        assert result is not None
        new_last = result["messages"][-1]
        assert new_last.additional_kwargs.get("tool_calls") == []

    def test_logs_warning_with_full_context_on_intervention(self, caplog):
        mw = LoopDetectionMiddleware()
        last = _ai_with_call("c3", "write_file", {"file_path": "/c"})
        messages = [
            _ai_with_call("c0", "write_file", {"file_path": "/canonical.md"}),
            _tool_msg("c0", "Wrote /canonical.md"),
            _ai_with_call("c1", "write_file", {"file_path": "/a"}),
            _tool_msg("c1", "Cannot write to /a because it already exists."),
            _ai_with_call("c2", "write_file", {"file_path": "/b"}),
            _tool_msg("c2", "Cannot write to /b because it already exists."),
            last,
        ]
        with caplog.at_level(logging.WARNING, logger="assist.middleware.loop_detection"):
            mw.after_model({"messages": messages}, Mock())

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        msg = warnings[0].getMessage()
        assert "intervention #1" in msg
        assert "pattern=same-tool-same-error" in msg
        assert "write_file" in msg
        assert "run_length=2" in msg
        assert "/canonical.md" in msg  # artifact captured
        assert "terminal=" in msg

    def test_intervention_counter_increments(self, caplog):
        mw = LoopDetectionMiddleware()

        def make_history(suffix: str):
            return [
                _ai_with_call(f"c1{suffix}", "write_file", {"file_path": f"/a{suffix}"}),
                _tool_msg(f"c1{suffix}", f"Cannot write to /a{suffix} because it already exists."),
                _ai_with_call(f"c2{suffix}", "write_file", {"file_path": f"/b{suffix}"}),
                _tool_msg(f"c2{suffix}", f"Cannot write to /b{suffix} because it already exists."),
                _ai_with_call(f"c3{suffix}", "write_file", {"file_path": f"/c{suffix}"}),
            ]

        with caplog.at_level(logging.WARNING, logger="assist.middleware.loop_detection"):
            mw.after_model({"messages": make_history("x")}, Mock())
            mw.after_model({"messages": make_history("y")}, Mock())

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 2
        assert "intervention #1" in warnings[0].getMessage()
        assert "intervention #2" in warnings[1].getMessage()
        assert mw._intervention_count == 2

    def test_logs_info_when_pattern_matched_but_model_breaks_out(self, caplog):
        """Pattern is in history, but the new tool call is unrelated.

        Should log INFO (not WARNING) and not strip anything.
        """
        mw = LoopDetectionMiddleware()
        last = _ai_with_call("c3", "read_file", {"file_path": "/x"})
        messages = [
            _ai_with_call("c1", "write_file", {"file_path": "/a"}),
            _tool_msg("c1", "Cannot write to /a because it already exists."),
            _ai_with_call("c2", "write_file", {"file_path": "/b"}),
            _tool_msg("c2", "Cannot write to /b because it already exists."),
            last,
        ]
        with caplog.at_level(logging.INFO, logger="assist.middleware.loop_detection"):
            result = mw.after_model({"messages": messages}, Mock())

        assert result is None  # didn't strip
        infos = [r for r in caplog.records if r.levelno == logging.INFO]
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 0
        assert len(infos) == 1
        assert "letting model continue" in infos[0].getMessage()
        assert mw._intervention_count == 0

    def test_replays_failing_transcript_pattern(self):
        """End-to-end shape of the swimming-workout failure.

        Successful initial write, then three filename-mutation collisions.
        Middleware should strip the third extension and surface the
        original successful path.
        """
        mw = LoopDetectionMiddleware()
        messages = [
            HumanMessage(content="research swimming"),
            _ai_with_call("c0", "write_file",
                          {"file_path": "/workspace/swimming_workout_report.md",
                           "content": "..."}),
            _tool_msg("c0", "Wrote /workspace/swimming_workout_report.md"),
            _ai_with_call("c1", "write_file",
                          {"file_path": "/workspace/final_report.md", "content": "..."}),
            _tool_msg("c1", "Cannot write to /workspace/final_report.md because it already exists."),
            _ai_with_call("c2", "write_file",
                          {"file_path": "/workspace/final_complete_report.md", "content": "..."}),
            _tool_msg("c2", "Cannot write to /workspace/final_complete_report.md because it already exists."),
            _ai_with_call("c3", "write_file",
                          {"file_path": "/workspace/completed_final_report.md", "content": "..."}),
        ]
        result = mw.after_model({"messages": messages}, Mock())
        assert result is not None
        new_last = result["messages"][-1]
        assert new_last.tool_calls == []
        content = new_last.content
        assert "/workspace/swimming_workout_report.md" in content
        # User-facing wording, not system-log.
        assert "saved" in content.lower()
        assert "loop detected" not in content.lower()
