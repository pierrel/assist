"""Real-LLM eval: the agent triages an inbound message against a subscription template and
proposes a reply via the HITL-gated send_reply (or correctly declines), doing its own
number→name mapping. Each test renders a template + message and asserts the outcome:
whether a reply was PROPOSED (the send_reply interrupt fired) and that the turn stated a
decision (never vanishing silently).

The gate itself (an untrusted text can't cause an effect) is a deployment property of the
sandbox + HITL, not asserted here; this eval asserts the *triage behavior* generalizes
(paraphrase, number→name, don't-over-act, don't-under-act).
"""
import os
import tempfile
import shutil
from unittest import TestCase

from assist.agent import AgentHarness, create_agent
from assist.spec import AgentSpec
from assist.events.reply import reply_tools, REPLY_INTERRUPT_ON
from assist.events.model import Subscription
from assist.model_manager import select_assistant_model


def _proposed_reply(harness) -> str | None:
    """The draft from a paused send_reply (HITL interrupt), or None if the agent proposed
    no reply."""
    snap = harness.agent.get_state({"configurable": {"thread_id": harness.thread_id}})
    for intr in (getattr(snap, "interrupts", None) or ()):
        for ar in (intr.value or {}).get("action_requests", []):
            if ar.get("name") == "send_reply":
                return ar.get("args", {}).get("text", "")
    return None


class _TriageScenario(TestCase):
    TEMPLATE = ""   # subclass sets; uses {sender}/{text}

    @classmethod
    def setUpClass(cls):
        cls.model = select_assistant_model(0.1)

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="triage_eval_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _triage(self, sender: str, text: str):
        agent = AgentHarness(create_agent(
            self.model, self.tmp,
            spec=AgentSpec(tools=reply_tools(), interrupt_on=REPLY_INTERRUPT_ON)))
        sub = Subscription(id="e", thread_id="t", sender_regexp=".*", template=self.TEMPLATE)
        final = agent.message(sub.render(sender, text))
        return agent, final


class TestReplyProposedWhenRuled(_TriageScenario):
    TEMPLATE = ("A text arrived from {sender}: {text}\n"
                "If it's asking to confirm a time or appointment, propose a short reply "
                "confirming with send_reply. Otherwise say no action is needed. "
                "Always end by stating your decision.")

    def test_proposes_a_reply(self):
        agent, _ = self._triage("+15551234567", "Can you confirm you're on for 3pm tomorrow?")
        draft = _proposed_reply(agent)
        self.assertIsNotNone(draft, "agent did not propose a reply to a confirm-the-time text")
        self.assertTrue(draft.strip(), "proposed reply was empty")


class TestDeclinesWhenRuleSaysIgnore(_TriageScenario):
    TEMPLATE = ("A text arrived from {sender}: {text}\n"
                "If it looks like spam or a promotional shortcode, do NOT reply — just note "
                "it's spam. Otherwise propose a reply. Always end by stating your decision.")

    def test_no_reply_but_states_decision(self):
        agent, final = self._triage("22395", "WINNER! Claim your $1000 gift card now: bit.ly/x")
        self.assertIsNone(_proposed_reply(agent), "agent proposed a reply to spam (should ignore)")
        self.assertTrue(final.strip(), "agent produced no decision (under-acting / vanished)")


class TestNumberToNameMapping(_TriageScenario):
    TEMPLATE = ("A text arrived from {sender}: {text}\n"
                "Note: +15551234567 is Ana. If Ana asks about the school pickup, propose a "
                "reply confirming I'll be there. Otherwise no action. End by stating your decision.")

    def test_resolves_number_and_proposes(self):
        agent, _ = self._triage("+15551234567", "are you doing pickup today?")
        self.assertIsNotNone(_proposed_reply(agent),
                             "agent didn't resolve +1555… → Ana and propose the ruled reply")
