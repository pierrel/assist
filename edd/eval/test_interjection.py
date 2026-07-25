"""Mid-turn interjection — does the model actually steer? (real-LLM eval)

The mechanics (deferred claim, sender scoping, fate-sharing, render) are
unit-pinned in tests/test_interjection.py; THIS suite evals the behavioral
contract: an interjection injected at a model-call boundary produces a
redirect / incorporate / defer / stop in the model's own voice (outcomes are
descriptive, asserted via mechanical proxies — design doc, Pierre note 1).

Harness: callbacks registered directly on the middleware with an in-memory
journal; entries become visible only from the SECOND before_model boundary
(the gate), so the model commits to the original ask first — a true mid-turn
arrival, not a two-message prompt. active_handle is patched because the eval
harness runs outside THREAD_QUEUE.acquire (production sets it per turn).
Research is stubbed per the mocking rule. Prompts avoid the guide's own
vocabulary ("redirect", "fold", "mid-turn") to probe steering, not echo.
"""
import tempfile
import uuid
from types import SimpleNamespace
from unittest import TestCase, mock

from langchain_core.messages import AIMessage, HumanMessage

from assist.agent import create_agent, AgentHarness
from assist.backlog import PendingMessage
from assist.model_manager import select_assistant_model
from assist.spec import AgentSpec
from manage.web.threads import (_INTERJECTION_FRAME, _INTERJECTION_GUIDE,
                                _INTERJECTION_DEFER)

from .utils import create_filesystem, final_answer, stub_research_subagent

_BIKE_ORG = """* My bike — Linus Roadster
** Parts I still need
- Front light — the Black MR clamp mount version
- Bell (the brass one from the shop on Valencia)
** Done
- New saddle installed in March
- Chain replaced
"""


class _Journal:
    """In-memory stand-in for MESSAGE_BACKLOG with the arrival gate: entries
    stay invisible until the second before_model boundary, so injection is
    genuinely mid-turn."""

    def __init__(self, defer_available=True):
        self.entries: list[PendingMessage] = []
        self.consumed: list[str] = []
        self.boundaries = 0
        self.defer = defer_available

    def peek(self, tid):
        self.boundaries += 1
        return list(self.entries) if self.boundaries > 1 else []

    def consume(self, tid, ids):
        self.consumed.extend(ids)
        self.entries = [r for r in self.entries if r.id not in ids]

    def frame(self, rec):
        return (_INTERJECTION_FRAME + rec.text + _INTERJECTION_GUIDE
                + (_INTERJECTION_DEFER if self.defer else "") + ")")

    def add(self, text):
        rec = PendingMessage(thread_id="eval", text=text, id=uuid.uuid4().hex)
        self.entries.append(rec)
        return rec


class TestInterjection(TestCase):
    def setUp(self):
        self.model = select_assistant_model(0.1)

    def _run(self, files, ask, interjections, defer_available=True):
        journal = _Journal(defer_available)
        for t in interjections:
            journal.add(t)
        root = tempfile.mkdtemp()
        create_filesystem(root, files)
        with mock.patch("assist.middleware.interjection.active_handle",
                        lambda: SimpleNamespace(thread_id="eval")), \
             mock.patch("assist.middleware.interjection._CALLBACKS",
                        {"peek": journal.peek, "consume": journal.consume,
                         "frame": journal.frame}), \
             stub_research_subagent():
            # agent built INSIDE the stub context (the research-mocking rule)
            agent = AgentHarness(create_agent(self.model, root, spec=AgentSpec()))
            agent.message(ask)
        return agent, journal

    def _tool_calls_after_injection(self, agent):
        """Tool calls the model made AFTER the injected frame message —
        the mechanical proxy for 'stop means stop'."""
        seen_frame = False
        n = 0
        for m in agent.all_messages():
            if isinstance(m, HumanMessage) and isinstance(m.content, str) \
                    and m.content.startswith(_INTERJECTION_FRAME):
                seen_frame = True
                continue
            if seen_frame and isinstance(m, AIMessage) and m.tool_calls:
                n += len(m.tool_calls)
        return n

    def _assert_presented(self, agent, journal):
        """Zero-loss floor: every seeded entry reached the model — its framed
        message is in graph state (consume may lag: it runs at the NEXT
        boundary or the web layer's terminal sweep, neither guaranteed in this
        harness when the injection boundary is the turn's last)."""
        in_state = set()
        for m in agent.all_messages():
            in_state.update((getattr(m, "additional_kwargs", None) or {}).get(
                "interjection_ids", []))
        missing = [r.text for r in journal.entries if r.id not in in_state]
        self.assertEqual(missing, [],
                         f"interjection never presented: {missing}")

    def test_redirect_mid_turn(self):
        """US-1: the interjection narrows the request; the final answer follows
        the interjection, not the original ask."""
        agent, journal = self._run(
            {"bike.org": _BIKE_ORG},
            "Look through my files and give me a complete maintenance plan "
            "for my bike for the rest of the year, month by month.",
            ["forget the plan — just list the parts I still need to buy"])
        self._assert_presented(agent, journal)
        answer = final_answer(agent).lower()
        self.assertTrue("light" in answer or "bell" in answer,
                        f"answer ignored the narrowed ask: {answer[:400]}")
        self.assertNotIn("december", answer)     # the month-by-month plan died

    def test_incorporate_addition(self):
        """US-2: an additive interjection folds into one coherent answer."""
        agent, journal = self._run(
            {"bike.org": _BIKE_ORG},
            "What parts do I still need to buy for my bike?",
            ["include what was already done recently too"])
        self._assert_presented(agent, journal)
        answer = final_answer(agent).lower()
        self.assertTrue("light" in answer or "bell" in answer,
                        f"lost the original ask: {answer[:400]}")
        self.assertTrue("saddle" in answer or "chain" in answer,
                        f"addition not incorporated: {answer[:400]}")

    def test_stop_halts_with_account(self):
        """US-6: stop means stop — at most one further tool call past the boundary,
        and the reply accounts for what already happened (never silent)."""
        agent, journal = self._run(
            {"bike.org": _BIKE_ORG},
            "Write a detailed file for each part my bike still needs, with a "
            "shopping checklist inside each.",
            ["never mind, don't do any of that"])
        self._assert_presented(agent, journal)
        self.assertLessEqual(self._tool_calls_after_injection(agent), 1,
                             "kept working after being told to stop")
        answer = final_answer(agent)
        self.assertTrue(answer.strip(), "stop must still produce an account")

    def test_coalesce_two_rapid_interjections(self):
        """Both entries land at one boundary; both are addressed."""
        agent, journal = self._run(
            {"bike.org": _BIKE_ORG},
            "Summarize the state of my bike.",
            ["only cover what still needs buying",
             "and mention where the bell comes from"])
        self._assert_presented(agent, journal)
        answer = final_answer(agent).lower()
        self.assertTrue("light" in answer or "bell" in answer,
                        f"first interjection lost: {answer[:400]}")
        self.assertIn("valencia", answer,
                      f"second interjection lost: {answer[:400]}")
