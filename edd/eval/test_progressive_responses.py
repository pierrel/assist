"""Progressive responses — does the model actually split? (real-LLM eval)

The north-star shape (PRD: the 34-minute bike turn): a request that local files
partly answer and research would extend. With the continue_later tool present
(and its conditional prompt block rendered), the model should answer from local
context NOW and schedule the research as a continuation — NOT dispatch the
research-agent synchronously in the first turn.

Mechanical assertions only (tool-call presence/absence + journal writes + the
answer citing fixture content); the web dispatch/render layers are unit-tested
in tests/test_progressive_responses.py. Research is stubbed per the mocking
rule — irrelevant here anyway, since the PASS condition is that research is
NOT dispatched in-turn. Partial pass rates expected; prompts avoid the tool
docstring's vocabulary ("follow up", "background", "schedule") so the eval
probes generalization, not lexical echo.
"""
import os
import tempfile
from unittest import TestCase

from assist.agent import create_agent, AgentHarness
from assist.events.continuations import continuation_tools
from assist.model_manager import select_assistant_model
from assist.spec import AgentSpec

from .utils import create_filesystem, stub_research_subagent

_BIKE_ORG = """* My bike — Linus Roadster
** Parts I still need
- Front light — the Black MR clamp mount version
- Bell (the brass one from the shop on Valencia)
** Done
- New saddle installed in March
- Chain replaced
"""


class TestProgressiveSplit(TestCase):
    def setUp(self):
        self.model = select_assistant_model(0.1)
        self.journaled = []

    def _agent(self, files):
        root = tempfile.mkdtemp()
        create_filesystem(root, files)
        tools = continuation_tools(
            lambda tid, task: self.journaled.append(task),
            lambda tid: 0)
        return AgentHarness(create_agent(self.model, root,
                                         spec=AgentSpec(tools=tools)))

    def _research_dispatches(self, agent):
        from langchain_core.messages import AIMessage
        n = 0
        for m in agent.all_messages():
            if isinstance(m, AIMessage) and m.tool_calls:
                for tc in m.tool_calls:
                    if tc.get("name") == "task":
                        args = tc.get("args") or {}
                        sa = (args.get("subagent_type") or args.get("agent")
                              or args.get("name") or "")
                        # EXACT match: "background-research-agent" (the door —
                        # dispatching it is the PASS behavior) contains
                        # "research" and must not count as a sync dispatch.
                        if str(sa).strip() == "research-agent":
                            n += 1
        return n

    def _final_answer(self, agent):
        from langchain_core.messages import AIMessage
        for m in reversed(agent.all_messages()):
            if isinstance(m, AIMessage) and isinstance(m.content, str) and m.content.strip():
                return m.content
        return ""

    def test_files_plus_research_shape_splits(self):
        """Local files answer part; the rest is research-worthy → answer from
        the files now, journal the rest, do NOT run research in this turn."""
        with stub_research_subagent():
            agent = self._agent({"bike.org": _BIKE_ORG})
            agent.message("Is there anything I still need to get for my bike, "
                          "and what would each roughly cost these days?")
        answer = self._final_answer(agent)
        # the fast half: the from-your-files answer is in the FIRST response
        self.assertTrue("light" in answer.lower() or "bell" in answer.lower(),
                        f"answer didn't use the local file: {answer[:400]}")
        # the slow half was deferred, not done in-turn
        self.assertEqual(len(self.journaled), 1,
                         f"expected exactly one continuation; journaled={self.journaled}")
        self.assertEqual(self._research_dispatches(agent), 0,
                         "research-agent must NOT be dispatched in the fast turn")
        # the promise is visible to the user
        self.assertTrue(any(w in answer.lower() for w in ("follow", "later", "get back")),
                        f"answer never told the user more is coming: {answer[:400]}")

    def test_pure_local_request_does_not_split(self):
        """A trivially local action must stay one turn — no continuation, no
        research (US-4: no forced chattiness)."""
        with stub_research_subagent():
            agent = self._agent({"bike.org": _BIKE_ORG})
            agent.message("Add 'pump' to the parts I still need in bike.org.")
        self.assertEqual(self.journaled, [],
                         f"a local edit must not schedule background work: {self.journaled}")
        self.assertEqual(self._research_dispatches(agent), 0)
