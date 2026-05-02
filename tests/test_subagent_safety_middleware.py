"""Regression: every dict-spec subagent must carry the safety middleware.

The dict-based subagent specs in ``assist.agent`` only get the
deepagents default middleware stack (TodoList, Filesystem,
Summarization, PatchToolCalls).  Without explicitly adding
``LoopDetectionMiddleware`` and ``EmptyResponseRecoveryMiddleware`` via
the ``"middleware"`` key, those subagents have no protection against
the runaway/empty-response failure modes the rest of the agent stack
defends against.  This test pins that protection so a future refactor
of the dict specs cannot silently drop it.
"""
from unittest import TestCase
from unittest.mock import MagicMock, patch

from assist.middleware.empty_response_recovery import (
    EmptyResponseRecoveryMiddleware,
)
from assist.middleware.loop_detection import LoopDetectionMiddleware


def _safety_mw_types_in(spec: dict) -> set[type]:
    return {type(m) for m in spec.get("middleware", [])}


class TestSubagentSafetyMiddleware(TestCase):
    """Each dict-spec subagent must carry both safety middlewares."""

    REQUIRED = {LoopDetectionMiddleware, EmptyResponseRecoveryMiddleware}

    def _capture_subagents(self, factory, *args, **kwargs):
        """Invoke ``factory`` with ``create_deep_agent`` patched to a no-op
        and return the ``subagents`` list it was called with.
        """
        captured = {}

        def fake_create_deep_agent(*a, **kw):
            captured["subagents"] = kw.get("subagents") or []
            # Return a dummy compiled-graph stand-in.  ``RollbackRunnable``
            # only proxies attributes; we don't invoke anything.
            mock_graph = MagicMock()
            return mock_graph

        with patch("assist.agent.create_deep_agent", side_effect=fake_create_deep_agent):
            factory(*args, **kwargs)

        return captured["subagents"]

    def test_research_agent_dict_subagents_have_safety_mw(self):
        from assist.agent import create_research_agent

        # Use a plain Mock model — create_research_agent accepts BaseChatModel
        # but the patched create_deep_agent won't actually run anything.
        model = MagicMock()
        subagents = self._capture_subagents(
            create_research_agent, model, working_dir="/tmp/x"
        )

        # Filter to dict-spec subagents (not CompiledSubAgent runnables).
        dict_specs = [s for s in subagents if isinstance(s, dict) and "runnable" not in s]
        self.assertGreater(len(dict_specs), 0, "Expected at least one dict subagent")

        for spec in dict_specs:
            present = _safety_mw_types_in(spec)
            missing = self.REQUIRED - present
            self.assertFalse(
                missing,
                f"Subagent '{spec.get('name')}' missing safety middleware: "
                f"{[m.__name__ for m in missing]}",
            )

    def test_main_agent_dict_subagents_have_safety_mw(self):
        from assist.agent import create_agent

        model = MagicMock()
        subagents = self._capture_subagents(
            create_agent, model, working_dir="/tmp/x"
        )

        # Only dict-spec subagents (CompiledSubAgent already inherits the
        # safety middleware via its own factory's base_mw list).
        dict_specs = [s for s in subagents if isinstance(s, dict) and "runnable" not in s]
        self.assertGreater(len(dict_specs), 0,
                           "Expected at least one dict subagent in create_agent")

        for spec in dict_specs:
            present = _safety_mw_types_in(spec)
            missing = self.REQUIRED - present
            self.assertFalse(
                missing,
                f"Subagent '{spec.get('name')}' missing safety middleware: "
                f"{[m.__name__ for m in missing]}",
            )
