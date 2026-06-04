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

    def test_bounds_to_latest_human_message(self):
        """Per-turn boundary: events from before the latest HumanMessage
        must NOT appear in the returned list.

        This is the structural fix for the cross-turn misfire described
        in docs/loop-misfire.md.  ``_extract_events`` previously walked
        the full message list, letting prior-turn tool calls stay in the
        loop-detection window indefinitely and trigger false positives
        on every subsequent turn.
        """
        messages = [
            HumanMessage(content="turn 1"),
            _ai_with_call("a1", "write_todos", {"todos": ["x"]}),
            _tool_msg("a1", "Cannot write to /tmp/x because it already exists."),
            _ai_with_call("a2", "write_todos", {"todos": ["y"]}),
            _tool_msg("a2", "Cannot write to /tmp/y because it already exists."),
            _ai_with_call("a3", "write_todos", {"todos": ["z"]}),
            _tool_msg("a3", "Cannot write to /tmp/z because it already exists."),
            HumanMessage(content="turn 2"),
            _ai_with_call("b1", "write_todos", {"todos": ["fresh"]}),
        ]
        events = _extract_events(messages, window=12)
        # Only the new turn's tool call should remain.  The unmatched
        # b1 call is "incomplete" but still counts as one event.
        assert len(events) == 1
        assert events[0]["completed"] is False
        assert events[0]["tool_name"] == "write_todos"

    def test_no_human_message_walks_full_history(self):
        """If there is no HumanMessage in the list (e.g. a synthetic
        test or a harness that hasn't injected one yet), the function
        operates on the full list — preserving today's behavior for
        every existing test that does not lead with HumanMessage.
        """
        messages = [
            _ai_with_call("c1", "write_file", {"file_path": "/a"}),
            _tool_msg("c1", "Cannot write to /a because it already exists."),
            _ai_with_call("c2", "write_file", {"file_path": "/b"}),
            _tool_msg("c2", "Cannot write to /b because it already exists."),
        ]
        events = _extract_events(messages, window=12)
        assert len(events) == 2


# ---------------------------------------------------------------------------
# Detection tests
# ---------------------------------------------------------------------------

class TestDetectLoop:
    def _evt(self, tool="write_file", args=None, content="ok",
             is_error=False, http_failure=False):
        return {
            "tool_name": tool,
            "args_sig": _normalise_args(args or {}),
            "result_content": content,
            "is_error": is_error,
            "http_failure": http_failure,
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

    # ------------------------------------------------------------------
    # Exploration tools (eg. emacsos's eval_elisp): relaxed Pattern-C
    # breadth threshold, but NOT exempt and still subject to A/B.
    # ------------------------------------------------------------------

    def test_pattern_c_exploration_tool_gets_higher_threshold(self):
        # The live emacsos shape: a few distinct eval_elisp probes, one
        # erroring (the small model fumbles the API).  With eval_elisp
        # marked as an exploration tool, the higher threshold (default 6)
        # means this legitimate exploration is NOT terminated.
        events = [
            self._evt(tool="eval_elisp", args={"code": "(cursor-type)"}, content="box"),
            self._evt(tool="eval_elisp",
                      args={"code": "(frame-parameter nil 'cursor-color)"},
                      content="nil"),
            self._evt(tool="eval_elisp", args={"code": "(load-path)"},
                      content="error: void-function load-path", is_error=True),
        ]
        assert _detect_loop(events, 2, 3, 3, 10,
                            exploration_tools=frozenset({"eval_elisp"})) is None

    def test_pattern_c_fires_for_eval_elisp_when_not_an_exploration_tool(self):
        # Opt-in: with NO exploration_tools (the dev/code agent's config),
        # the same 3 distinct erroring probes trip Pattern C at threshold 3.
        events = [
            self._evt(tool="eval_elisp", args={"code": "(cursor-type)"}, content="box"),
            self._evt(tool="eval_elisp",
                      args={"code": "(frame-parameter nil 'cursor-color)"},
                      content="nil"),
            self._evt(tool="eval_elisp", args={"code": "(load-path)"},
                      content="error: void-function load-path", is_error=True),
        ]
        result = _detect_loop(events, 2, 3, 3, 10)
        assert result is not None
        assert result["pattern"] == "distinct-args-thrash"

    def test_pattern_c_exploration_tool_boundary_just_below_threshold(self):
        # 5 distinct erroring forms (one below the exploration threshold of
        # 6, with DISTINCT errors so A/B don't fire) → not yet a flail.
        errs = ["void-function a", "void-variable b", "wrong-type c",
                "args-range d", "scan-error e"]
        events = [
            self._evt(tool="eval_elisp", args={"code": f"(p{i})"},
                      content=f"error: {errs[i]}", is_error=True)
            for i in range(5)
        ]
        assert _detect_loop(events, 2, 3, 3, 10,
                            exploration_tools=frozenset({"eval_elisp"})) is None

    def test_pattern_c_still_catches_sustained_exploration_flail(self):
        # A sustained flail (>= the higher threshold of distinct erroring
        # forms, with DISTINCT errors so A/B don't fire) IS still caught,
        # faster than the recursion limit.
        errs = ["void-function foo", "void-variable bar", "wrong-type baz",
                "args-out-of-range qux", "scan-error quux", "no-catch corge"]
        events = [
            self._evt(tool="eval_elisp", args={"code": f"(probe-{i})"},
                      content=f"error: {errs[i]}", is_error=True)
            for i in range(6)
        ]
        result = _detect_loop(events, 2, 3, 3, 10,
                              exploration_tools=frozenset({"eval_elisp"}))
        assert result is not None
        assert result["pattern"] == "distinct-args-thrash"
        assert result["run_length"] == 6

    def test_exploration_tool_still_subject_to_pattern_a(self):
        # The SAME error repeated is a real loop — Pattern A still fires for
        # an exploration tool (it stays "mutating" for A/B).
        events = [
            self._evt(tool="eval_elisp", args={"code": "(a)"},
                      content="error: void-function frobnicate", is_error=True),
            self._evt(tool="eval_elisp", args={"code": "(b)"},
                      content="error: void-function frobnicate", is_error=True),
        ]
        result = _detect_loop(events, 2, 3, 3, 10,
                              exploration_tools=frozenset({"eval_elisp"}))
        assert result is not None
        assert result["pattern"] == "same-tool-same-error"

    def test_exploration_tool_still_subject_to_pattern_b(self):
        # The IDENTICAL form repeated 3x is a real loop — Pattern B still
        # fires for an exploration tool.
        evt = self._evt(tool="eval_elisp", args={"code": "(stuck)"}, content="nil")
        events = [evt, evt.copy(), evt.copy()]
        result = _detect_loop(events, 2, 3, 3, 10,
                              exploration_tools=frozenset({"eval_elisp"}))
        assert result is not None
        assert result["pattern"] == "same-tool-same-args"

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

    # ------------------------------------------------------------------
    # Read-only-interleaved trailing-run detection
    # Regression: 2026-05-16 thread 20260516165139-36e38c0a ("winged-
    # horse flag") oscillated 70+ times in
    # `[ls, write_file_err, ls, write_file_err, ...]` for ~1 hour
    # before the sandbox container was lost.  Pattern A trailing-run
    # walk used to break on the interleaved `ls` (non-error), so the
    # loop escaped detection.  Read-only events are now transparent
    # to the Pattern A and B walks.
    # ------------------------------------------------------------------

    def test_pattern_a_catches_loop_through_interleaved_ls(self):
        """Two write_file errors with `ls` in between — the flag-thread
        shape — must trigger Pattern A.  `ls` is read-only so it
        neither extends nor breaks the trailing mutating-tool error
        run."""
        events = [
            self._evt(tool="ls", args={"path": "/workspace"}, content="[]"),
            self._evt(
                tool="write_file",
                args={"file_path": "report.org", "content": "..."},
                content=("OCI runtime exec failed: exec failed: unable to "
                         "start container process: chdir to cwd "
                         "(\"/workspace/references\")"),
                is_error=True,
            ),
            self._evt(tool="ls", args={"path": "/workspace"}, content="[]"),
            self._evt(
                tool="write_file",
                args={"file_path": "report.org", "content": "..."},
                content=("OCI runtime exec failed: exec failed: unable to "
                         "start container process: chdir to cwd "
                         "(\"/workspace/references\")"),
                is_error=True,
            ),
        ]
        result = _detect_loop(events, 2, 3, 3, 10)
        assert result is not None, "Pattern A should fire on ls-interleaved write_file errors"
        assert result["pattern"] == "same-tool-same-error"
        assert result["tools"] == {"write_file"}
        assert result["run_length"] == 2

    def test_pattern_a_catches_long_alternating_pathology(self):
        """Full 8-event alternation (4 ls + 4 wf_err) — the exact
        N=4 mutating-error run we'd want to catch as early as
        possible.  Asserts run_length counts only the mutating
        events (4), not the interleaved ls."""
        events = []
        for _ in range(4):
            events.append(self._evt(tool="ls", args={"path": "/"}, content="[]"))
            events.append(self._evt(
                tool="write_file",
                args={"file_path": "x.org"},
                content="OCI runtime exec failed: chdir to cwd",
                is_error=True,
            ))
        result = _detect_loop(events, 2, 3, 3, 10)
        assert result is not None
        assert result["pattern"] == "same-tool-same-error"
        assert result["run_length"] == 4

    def test_pattern_b_catches_same_args_through_interleaved_glob(self):
        """Same-args trailing-run also tolerates read-only interleaving.
        Three identical write_file calls with `glob` in between count
        as Pattern B at threshold 3."""
        wf = self._evt(
            tool="write_file",
            args={"file_path": "/x.org", "content": "data"},
            content="ok",
        )
        gl = self._evt(tool="glob", args={"pattern": "*.org"}, content="[]")
        events = [wf, gl, wf.copy(), gl.copy(), wf.copy()]
        result = _detect_loop(events, 2, 3, 3, 10)
        assert result is not None
        assert result["pattern"] == "same-tool-same-args"
        assert result["tools"] == {"write_file"}
        assert result["run_length"] == 3

    def test_no_false_positive_legitimate_write_read_write(self):
        """Sanity counter-check: a real workflow of `write_file(/a) →
        ls → write_file(/b)` (different files, no errors) must NOT
        trigger any pattern.  The ls is transparent to A/B walks but
        without errors there's no Pattern A; and the two writes have
        different args so Pattern B's trailing-same-args-run is 1."""
        events = [
            self._evt(tool="write_file", args={"file_path": "/a.org"}, content="Wrote /a.org"),
            self._evt(tool="ls", args={"path": "/"}, content="['a.org']"),
            self._evt(tool="write_file", args={"file_path": "/b.org"}, content="Wrote /b.org"),
        ]
        assert _detect_loop(events, 2, 3, 3, 10) is None

    def test_pattern_b_catches_repeated_read_url(self):
        """The 2026-05-30 runaway: a sub-research-agent issued the same
        read_url(URL) ~1000 times in a row.  Pre-fix, Pattern B's walk
        was over `mutating_events` which filtered out all read_url
        calls — the runaway was invisible.  Post-fix, Pattern B walks
        ALL completed events; same-args repetition triggers regardless
        of read-only category."""
        ru = self._evt(
            tool="read_url",
            args={"url": "https://www.casio.com/products/watches/f-91w-1"},
            content="[long page content]",
        )
        events = [ru, ru.copy(), ru.copy()]
        result = _detect_loop(events, 2, 3, 3, 10)
        assert result is not None
        assert result["pattern"] == "same-tool-same-args"
        assert result["tools"] == {"read_url"}
        assert result["run_length"] == 3

    def test_pattern_b_catches_repeated_search_internet(self):
        """Same-args runaway also catches search_internet — the runaway
        log showed the same query (`Casio F-91W watch specs water
        resistance`) issued back-to-back several times in the trailing
        window before the model rephrased."""
        si = self._evt(
            tool="search_internet",
            args={"query": "Casio F-91W watch specs water resistance"},
            content="[results]",
        )
        events = [si, si.copy(), si.copy(), si.copy()]
        result = _detect_loop(events, 2, 3, 3, 10)
        assert result is not None
        assert result["pattern"] == "same-tool-same-args"
        assert result["tools"] == {"search_internet"}
        assert result["run_length"] == 4

    def test_pattern_b_read_url_transparent_to_unrelated_read_only_between(self):
        """Three same-URL read_urls with an `ls` between count via the
        same transparent-read-only rule that already protects
        interleaved-mutating loops (the 2026-05-16 case)."""
        ru = self._evt(
            tool="read_url",
            args={"url": "https://example.com/article"},
            content="[content]",
        )
        ls = self._evt(tool="ls", args={"path": "/"}, content="['x.md']")
        events = [ru, ls, ru.copy(), ls.copy(), ru.copy()]
        result = _detect_loop(events, 2, 3, 3, 10)
        assert result is not None
        assert result["pattern"] == "same-tool-same-args"
        assert result["run_length"] == 3

    def test_pattern_b_does_not_fire_on_alternating_read_urls(self):
        """Alternating between two distinct URLs is exploration, not a
        loop.  Same-tool different-args must BREAK the trailing run
        even though both args are read-only.  Without this check, the
        transparent-read-only rule would skip the `B` events and
        falsely identify a 3-run of `A`."""
        ru_a = self._evt(
            tool="read_url",
            args={"url": "https://example.com/a"},
            content="[a]",
        )
        ru_b = self._evt(
            tool="read_url",
            args={"url": "https://example.com/b"},
            content="[b]",
        )
        events = [ru_a, ru_b, ru_a.copy(), ru_b.copy(), ru_a.copy()]
        assert _detect_loop(events, 2, 3, 3, 10) is None

    def test_no_false_positive_pure_exploration(self):
        """A run of read-only tools alone (no mutating) must NOT
        trigger any pattern.  After filtering out read-only events
        the mutating-events list is empty, so all three patterns
        short-circuit."""
        events = [
            self._evt(tool="ls", args={"path": "/"}, content="[]"),
            self._evt(tool="read_file", args={"file_path": "/x"}, content="hi"),
            self._evt(tool="grep", args={"pattern": "foo"}, content="[]"),
            self._evt(tool="search_internet", args={"query": "x"}, content="[]"),
        ]
        assert _detect_loop(events, 2, 3, 3, 10) is None

    # ------------------------------------------------------------------
    # Pattern B — trailing-read-only-of-different-tool case
    # ------------------------------------------------------------------
    def test_pattern_b_catches_mutating_loop_ending_on_read_only(self):
        """``[write_file_X, ls, write_file_X, ls, write_file_X, ls]`` —
        the trailing ``ls`` would otherwise be the run anchor and any
        prior ``write_file_X`` would break it (Copilot round 4 finding
        on PR #116).  The two-pass anchor fix (anchor on the latest
        non-read-only event when the trailing one is read-only)
        recovers the loop.  Mirrors the established 2026-05-16
        winged-horse-flag case, just with one more trailing ``ls``."""
        wf = self._evt(
            tool="write_file",
            args={"file_path": "/x.org", "content": "v1"},
            content="Wrote /x.org",
        )
        ls = self._evt(tool="ls", args={"path": "/"}, content="['x.org']")
        events = [wf, ls, wf.copy(), ls.copy(), wf.copy(), ls.copy()]
        result = _detect_loop(events, 2, 3, 3, 10)
        assert result is not None, "loop missed when trailing event is read-only"
        assert result["pattern"] == "same-tool-same-args"
        assert result["tools"] == {"write_file"}
        assert result["run_length"] == 3

    def test_pattern_b_two_pass_still_catches_trailing_mutating_same_args(self):
        """Sanity: the two-pass fix must not regress the canonical case
        where the trailing event IS the mutating loop tail."""
        wf = self._evt(
            tool="write_file",
            args={"file_path": "/x.org", "content": "v1"},
            content="Wrote /x.org",
        )
        events = [wf, wf.copy(), wf.copy()]
        result = _detect_loop(events, 2, 3, 3, 10)
        assert result is not None
        assert result["pattern"] == "same-tool-same-args"
        assert result["run_length"] == 3

    def test_pattern_b_two_pass_still_catches_read_only_same_args(self):
        """Sanity: the F-91W case (``read_url(URL) x N``) must still fire
        — the second pass would skip the trailing read-only and find
        nothing, but the first pass anchors on it and the same-args
        extension keeps working regardless of read-only category."""
        ru = self._evt(
            tool="read_url",
            args={"url": "https://example.com/a"},
            content="[content]",
        )
        events = [ru, ru.copy(), ru.copy()]
        result = _detect_loop(events, 2, 3, 3, 10)
        assert result is not None
        assert result["pattern"] == "same-tool-same-args"
        assert result["run_length"] == 3

    # ------------------------------------------------------------------
    # Pattern D — HTTP-failure streak (the casio runaway shape)
    # ------------------------------------------------------------------
    def test_pattern_d_catches_distinct_url_4xx_streak(self):
        """The 2026-05-30 casio.com runaway: ``fetch_url`` returned a
        403 bot-detection HTML page on each of many distinct watch-product
        URLs.  Patterns A/B/C all short-circuit (HTML body isn't a
        Python error, args differ each call, no error flag set), so
        Pattern D specifically counts consecutive trailing tool calls
        whose body looks HTTP-failure-shaped, regardless of args."""
        events = [
            self._evt(
                tool="fetch_url",
                args={"url": f"https://www.casio.com/products/watches/p-{i}/"},
                content="<html><title>403 Forbidden</title>...",
                http_failure=True,
            )
            for i in range(5)
        ]
        result = _detect_loop(events, 2, 3, 3, 10, http_failure_threshold=4)
        assert result is not None
        assert result["pattern"] == "http-failure-streak"
        assert result["tools"] == {"fetch_url"}
        assert result["run_length"] == 5

    def test_pattern_d_does_not_fire_below_threshold(self):
        """3 consecutive failures (< default threshold 4) should not
        fire — many normal research sessions hit one or two 4xx hops
        before finding a working source."""
        events = [
            self._evt(tool="fetch_url",
                      args={"url": f"https://x.example.com/p-{i}/"},
                      content="403 Forbidden", http_failure=True)
            for i in range(3)
        ]
        assert _detect_loop(events, 2, 3, 3, 10, http_failure_threshold=4) is None

    def test_pattern_d_does_not_fire_when_streak_is_broken(self):
        """A successful response in the trailing window breaks the
        streak — the model is still making progress."""
        events = [
            self._evt(tool="fetch_url",
                      args={"url": "https://x.example.com/p-1/"},
                      content="403 Forbidden", http_failure=True),
            self._evt(tool="fetch_url",
                      args={"url": "https://x.example.com/p-2/"},
                      content="403 Forbidden", http_failure=True),
            # A real successful page — breaks the streak.
            self._evt(tool="fetch_url",
                      args={"url": "https://x.example.com/about/"},
                      content="<html><body>Welcome to our store...</body></html>"),
            self._evt(tool="fetch_url",
                      args={"url": "https://x.example.com/p-3/"},
                      content="403 Forbidden", http_failure=True),
        ]
        # The trailing failure-streak is 1; below threshold.
        assert _detect_loop(events, 2, 3, 3, 10, http_failure_threshold=4) is None

    def test_pattern_d_streak_spanning_tool_names(self):
        """Mixed-tool failure streaks (search_internet 4xx, fetch_url 4xx,
        fetch_url 4xx, …) trigger on the LATEST failing tool — the
        signature is the failure shape, not the tool identity."""
        events = [
            self._evt(tool="search_internet",
                      args={"query": "f-91w"},
                      content="Rate limit exceeded", http_failure=True),
            self._evt(tool="fetch_url",
                      args={"url": "https://casio.com/a/"},
                      content="403 Forbidden", http_failure=True),
            self._evt(tool="fetch_url",
                      args={"url": "https://casio.com/b/"},
                      content="403 Forbidden", http_failure=True),
            self._evt(tool="fetch_url",
                      args={"url": "https://casio.com/c/"},
                      content="403 Forbidden", http_failure=True),
        ]
        result = _detect_loop(events, 2, 3, 3, 10, http_failure_threshold=4)
        assert result is not None
        assert result["pattern"] == "http-failure-streak"
        assert result["tools"] == {"fetch_url"}, \
            "latest failing tool wins (most recent is what the model is currently doing)"


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

    def test_ignores_artifacts_from_prior_turn(self):
        """Artifact lookup must respect the per-turn boundary.

        Without this, a successful write from a prior turn bleeds into
        the current turn's terminal message — exactly the bug from
        docs/loop-misfire.md, where every post-loop turn cited
        ``steam_link_linux_handheld_setup.md`` even though the current
        turn never wrote it.
        """
        messages = [
            HumanMessage(content="turn 1"),
            _ai_with_call("c1", "write_file",
                          {"file_path": "/canonical.md", "content": "..."}),
            _tool_msg("c1", "Wrote /canonical.md"),
            HumanMessage(content="turn 2"),
            _ai_with_call("c2", "write_file", {"file_path": "/dup"}),
            _tool_msg("c2", "Cannot write to /dup because it already exists."),
        ]
        # Within turn 2 there is no successful artifact; the prior turn's
        # /canonical.md must not be surfaced.
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

    def test_ignores_errors_from_prior_turn(self):
        """Error-excerpt lookup must respect the per-turn boundary.

        Symmetric with the artifact-bounding fix: if an intervention
        fires in the current turn but the only matching error is from
        a prior turn, we must not quote that stale error in the
        terminal message.
        """
        messages = [
            HumanMessage(content="turn 1"),
            _ai_with_call("c1", "write_file", {"file_path": "/a"}),
            _tool_msg("c1", "Cannot write to /a because it already exists."),
            HumanMessage(content="turn 2"),
            _ai_with_call("c2", "write_file", {"file_path": "/b"}),
            _tool_msg("c2", "Wrote /b"),
        ]
        # No errors in turn 2 — must not surface turn 1's error.
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

    def test_volume_cap_fire_logs_backstop_warning(self, caplog):
        """When the tool-volume backstop fires it emits an extra warning
        pointing at an upstream cause — the cap shouldn't be the bound."""
        mw = LoopDetectionMiddleware(
            volume_threshold=4, volume_tools=frozenset({"search_internet"}))
        msgs = [HumanMessage(content="go")]
        for i in range(4):
            msgs.append(_ai_with_call(f"c{i}", "search_internet", {"q": f"q{i}"}))
            msgs.append(_tool_msg(f"c{i}", "[results]"))
        msgs.append(_ai_with_call("clast", "search_internet", {"q": "more"}))
        with caplog.at_level(logging.WARNING,
                             logger="assist.middleware.loop_detection"):
            result = mw.after_model({"messages": msgs}, Mock())
        assert result is not None
        assert result["messages"][-1].tool_calls == []
        assert any("backstop fired" in r.message for r in caplog.records), \
            f"expected backstop warning; got {[r.message for r in caplog.records]}"

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

    def test_does_not_misfire_after_completed_prior_turn(self):
        """Replays the cross-turn misfire from docs/loop-misfire.md.

        A prior turn completed and left a successful artifact + multiple
        ``write_todos`` events in the conversation history.  The current
        turn opens with a single ``write_todos`` call (typical
        small-model fresh-task behavior).  Without per-turn bounding,
        the prior turn's events stay in the loop window and pattern C
        (distinct-args-thrash) fires instantly, producing a canned
        terminal message even though the new turn has done nothing.

        With the fix, ``_extract_events`` only sees post-HumanMessage
        events — a single incomplete call — and the detector returns
        None.
        """
        mw = LoopDetectionMiddleware()
        prior_turn = [
            HumanMessage(content="turn 1: do the thing"),
            _ai_with_call("a1", "write_todos", {"todos": ["x"]}),
            _tool_msg("a1", "Cannot write to /tmp/x because it already exists."),
            _ai_with_call("a2", "write_todos", {"todos": ["y"]}),
            _tool_msg("a2", "Cannot write to /tmp/y because it already exists."),
            _ai_with_call("a3", "write_todos", {"todos": ["z"]}),
            _tool_msg("a3", "Cannot write to /tmp/z because it already exists."),
            _ai_with_call("a4", "write_file",
                          {"file_path": "/workspace/result.md", "content": "..."}),
            _tool_msg("a4", "Wrote /workspace/result.md"),
        ]
        new_turn = [
            HumanMessage(content="turn 2: a totally new question"),
            _ai_with_call("b1", "write_todos", {"todos": ["fresh"]}),
        ]
        result = mw.after_model({"messages": prior_turn + new_turn}, Mock())
        # Critical: no intervention.  The new turn is one call deep.
        assert result is None
        assert mw._intervention_count == 0

    def test_intra_turn_loop_still_fires(self):
        """Sanity: a real loop within a single turn still triggers.

        Same shape as ``test_strips_tool_calls_when_loop_detected``
        but with multi-turn history preceding it — proves the per-turn
        bound doesn't accidentally suppress legitimate loops.
        """
        mw = LoopDetectionMiddleware()
        prior_turn = [
            HumanMessage(content="turn 1"),
            _ai_with_call("p1", "ls", {"path": "/"}),
            _tool_msg("p1", "ok"),
        ]
        last = _ai_with_call("c3", "write_file", {"file_path": "/c"})
        current_turn = [
            HumanMessage(content="turn 2"),
            _ai_with_call("c1", "write_file", {"file_path": "/a"}),
            _tool_msg("c1", "Cannot write to /a because it already exists."),
            _ai_with_call("c2", "write_file", {"file_path": "/b"}),
            _tool_msg("c2", "Cannot write to /b because it already exists."),
            last,
        ]
        result = mw.after_model({"messages": prior_turn + current_turn}, Mock())
        assert result is not None
        assert result["messages"][-1].tool_calls == []


class TestPatternEVolume:
    """Pattern E: sheer per-tool call volume, regardless of args/errors.

    Off unless volume_threshold > 0 AND the tool is in volume_tools.  Uses
    search_internet (a read-only tool with distinct, successful args) so
    Patterns A/B/C/D all skip and only the volume pattern can fire —
    proving it catches a runaway the other patterns deliberately ignore
    (distinct-query exploration is normal; sheer volume is not).
    """
    SEARCH = frozenset({"search_internet"})

    def _searches(self, n):
        return [{
            "tool_name": "search_internet",
            "args_sig": _normalise_args({"query": f"q{i}"}),
            "result_content": "[{'title': 'r', 'url': 'u', 'content': 'c'}]",
            "is_error": False,
            "http_failure": False,
            "completed": True,
        } for i in range(n)]

    def test_disabled_by_default(self):
        # 20 distinct successful searches, volume cap off -> no detection.
        assert _detect_loop(self._searches(20), 2, 3, 3, 10) is None

    def test_disabled_without_volume_tools(self):
        # threshold set but no tools scoped -> no detection.
        assert _detect_loop(self._searches(20), 2, 3, 3, 10,
                            volume_threshold=8) is None

    def test_fires_at_threshold(self):
        result = _detect_loop(self._searches(8), 2, 3, 3, 10,
                              volume_threshold=8, volume_tools=self.SEARCH)
        assert result is not None
        assert result["pattern"] == "tool-volume"
        assert result["tools"] == {"search_internet"}
        assert result["run_length"] == 8

    def test_does_not_fire_below_threshold(self):
        assert _detect_loop(self._searches(7), 2, 3, 3, 10,
                            volume_threshold=8, volume_tools=self.SEARCH) is None

    def test_only_caps_tools_in_volume_tools(self):
        # 9 read_url calls but volume_tools is search-only -> not capped
        # (read_url is legitimate research, throttled + Pattern-D-protected).
        reads = [{
            "tool_name": "read_url",
            "args_sig": _normalise_args({"url": f"u{i}"}),
            "result_content": "page text",
            "is_error": False, "http_failure": False, "completed": True,
        } for i in range(9)]
        assert _detect_loop(reads, 2, 3, 3, 10,
                            volume_threshold=6, volume_tools=self.SEARCH) is None

    def test_picks_capped_tool_not_uncapped(self):
        # search (8, capped) + read_url (3, uncapped) -> fires on search.
        events = self._searches(8) + [{
            "tool_name": "read_url",
            "args_sig": _normalise_args({"url": f"u{i}"}),
            "result_content": "page text",
            "is_error": False, "http_failure": False, "completed": True,
        } for i in range(3)]
        result = _detect_loop(events, 2, 3, 3, 10,
                              volume_threshold=8, volume_tools=self.SEARCH)
        assert result is not None
        assert result["tools"] == {"search_internet"}

    def test_counts_per_tool_not_aggregate(self):
        # Two capped tools, each BELOW the threshold, summing ABOVE it:
        # 4 search + 4 read_url, both capped, threshold 6.  Per-tool the
        # max is 4 (< 6) so nothing fires.  Pins the per-tool semantics:
        # a healthy pass (~4 searches + a read or two) must not trip an
        # aggregate-counting bug.
        both = frozenset({"search_internet", "read_url"})
        events = self._searches(4) + [{
            "tool_name": "read_url",
            "args_sig": _normalise_args({"url": f"u{i}"}),
            "result_content": "page text",
            "is_error": False, "http_failure": False, "completed": True,
        } for i in range(4)]
        assert _detect_loop(events, 2, 3, 3, 10,
                            volume_threshold=6, volume_tools=both) is None

    def test_reports_all_over_threshold_tools(self):
        # Both capped tools exceed the threshold: search=7, read=7, cap=6.
        # detection["tools"] must include BOTH so after_model intervenes on
        # a latest call to either — naming only the busiest would let the
        # other slip through.
        both = frozenset({"search_internet", "read_url"})
        events = self._searches(7) + [{
            "tool_name": "read_url",
            "args_sig": _normalise_args({"url": f"u{i}"}),
            "result_content": "page text",
            "is_error": False, "http_failure": False, "completed": True,
        } for i in range(7)]
        result = _detect_loop(events, 2, 3, 3, 10,
                              volume_threshold=6, volume_tools=both)
        assert result is not None
        assert result["tools"] == {"search_internet", "read_url"}

    def test_terminal_message_is_graceful(self):
        msg = _compose_terminal_message(
            {"pattern": "tool-volume", "tools": {"search_internet"},
             "run_length": 8, "reason": "x"},
            [],
        )
        # Not an error/ask-for-direction message — a "finalize now" one.
        assert "search_internet" in msg
        assert "?" not in msg


class TestPatternFRedispatch:
    """Pattern F: per-subagent `task` re-dispatch cap (orchestrator).

    With subagent_dispatch_threshold=1, each subagent may be dispatched
    once; re-dispatching an already-used one is stripped, but dispatching
    a fresh subagent (the research -> critique -> fact-check progression)
    passes through.
    """
    def test_off_by_default(self):
        # Three research dispatches, distinct descriptions (so Pattern B
        # does not fire), default middleware (threshold 0) -> no action.
        mw = LoopDetectionMiddleware()
        last = _ai_with_call("c3", "task",
                             {"subagent_type": "research-agent", "description": "c"})
        messages = [
            HumanMessage(content="go"),
            _ai_with_call("c1", "task",
                          {"subagent_type": "research-agent", "description": "a"}),
            _tool_msg("c1", "research result one"),
            _ai_with_call("c2", "task",
                          {"subagent_type": "research-agent", "description": "b"}),
            _tool_msg("c2", "research result two"),
            last,
        ]
        assert mw.after_model({"messages": messages}, Mock()) is None

    def test_strips_redispatch_of_same_subagent(self):
        mw = LoopDetectionMiddleware(subagent_dispatch_threshold=1)
        last = _ai_with_call("c2", "task",
                             {"subagent_type": "research-agent", "description": "again"})
        messages = [
            HumanMessage(content="go"),
            _ai_with_call("c1", "task",
                          {"subagent_type": "research-agent", "description": "first"}),
            _tool_msg("c1", "research result"),
            last,
        ]
        result = mw.after_model({"messages": messages}, Mock())
        assert result is not None
        assert result["messages"][-1].tool_calls == []
        assert "gathered" in result["messages"][-1].content.lower()

    def test_in_message_batch_not_capped_known_limitation(self):
        """KNOWN LIMITATION (pinned, not a bug to fix here).

        Pattern F counts COMPLETED dispatches, so two `task` calls to the
        same subagent batched in ONE message — before either completes —
        are not capped.  The observed prod re-dispatches (and the
        2026-06-03 ablation's rdisp=2) are all CROSS-message: dispatch,
        read the result, dispatch again — which Pattern F catches on the
        next message.  In-message batching of an *identical* subagent is
        unobserved, and the whole-message strip would over-correct (drop
        the legitimate first dispatch too), so we pin current behavior
        rather than add speculative partial-strip logic.  See the design
        doc residuals.
        """
        mw = LoopDetectionMiddleware(subagent_dispatch_threshold=1)
        batched = AIMessage(content="", tool_calls=[
            {"name": "task",
             "args": {"subagent_type": "research-agent", "description": "a"},
             "id": "a"},
            {"name": "task",
             "args": {"subagent_type": "research-agent", "description": "b"},
             "id": "b"},
        ])
        messages = [HumanMessage(content="go"), batched]
        assert mw.after_model({"messages": messages}, Mock()) is None

    def test_fresh_subagent_passes(self):
        # research already dispatched; dispatching critique (fresh) is the
        # normal progression and must not be stripped.
        mw = LoopDetectionMiddleware(subagent_dispatch_threshold=1)
        last = _ai_with_call("c2", "task",
                             {"subagent_type": "critique-agent", "description": "review"})
        messages = [
            HumanMessage(content="go"),
            _ai_with_call("c1", "task",
                          {"subagent_type": "research-agent", "description": "first"}),
            _tool_msg("c1", "research result"),
            last,
        ]
        assert mw.after_model({"messages": messages}, Mock()) is None
