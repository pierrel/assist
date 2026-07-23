"""Eval: the phone receptionist routes callers to the right thread (voice-call
assistant, P0.5; docs/2026-07-21-voice-call-tech-design.org §4).

Real-LLM eval (the small model). The receptionist is a *vanilla* LangChain agent
(no sandbox, no filesystem) whose whole job is tool-routing over the read-only
thread catalog, so each test asserts on the OBSERVABLE routing effect — which tool
the model called, and whether the injected on_open/on_new callbacks fired with the
right thread/domain — plus the spoken-output shape. Contents are unreachable by
construction (the catalog can't read them), so the content-leak test checks the
model doesn't *fabricate* a summary instead.

Prompts deliberately avoid the system-prompt wording verbatim (probe
generalization, not lexical echo).
"""
import json
import os
import re
import shutil
import tempfile
from unittest import TestCase

from langchain_core.messages import AIMessage, HumanMessage

from assist.catalog import ThreadCatalog
from assist.receptionist import create_receptionist, receptionist_tools
from assist.thread_manager import ThreadManager


def _mk(root, tid, *, description, stage="ready", urgent=False):
    d = os.path.join(root, tid)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "description.txt"), "w") as f:
        f.write(description)
    with open(os.path.join(d, "status.json"), "w") as f:
        json.dump({"stage": stage}, f)
    if urgent:
        open(os.path.join(d, "urgent_response"), "w").close()


class _Call:
    """A live receptionist call over a fixture thread dir: records which threads
    were opened and which were created, and lets a test drive turns."""

    def __init__(self, root, domains):
        self.opened: list = []
        self.created: list = []
        catalog = ThreadCatalog(ThreadManager(root))
        tools = receptionist_tools(catalog, domains,
                                   self.opened.append,
                                   lambda d, m: self.created.append((d, m)))
        self.agent = create_receptionist(tools, domains)
        self.config = {"configurable": {"thread_id": "eval-call"}}
        self.messages: list = []

    def say(self, text) -> str:
        """One caller turn; returns the receptionist's final spoken reply."""
        state = self.agent.invoke(
            {"messages": self.messages + [HumanMessage(content=text)]}, self.config)
        self.messages = state["messages"]
        finals = [m for m in state["messages"]
                  if isinstance(m, AIMessage) and isinstance(m.content, str) and m.content.strip()]
        return finals[-1].content if finals else ""

    def tool_calls(self) -> list:
        """Every (name, args) tool call the model made this call, in order."""
        return [(tc["name"], tc.get("args", {}))
                for m in self.messages if isinstance(m, AIMessage)
                for tc in (m.tool_calls or [])]

    def called(self, name) -> bool:
        return any(n == name for n, _ in self.tool_calls())


class TestReceptionist(TestCase):
    def _dir(self, **threads):
        root = tempfile.mkdtemp(prefix="receptionist_eval_")
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        for tid, kw in threads.items():
            _mk(root, tid, **kw)
        return root

    # ── yield to intent: open directly, don't read a menu first ──────────────

    def test_picks_thread_by_topic(self):
        root = self._dir(trip={"description": "trip planning to Portugal"},
                         groc={"description": "grocery list"})
        call = _Call(root, ["Home"])
        call.say("Put me through to my Portugal trip planning conversation.")
        self.assertEqual(call.opened, ["trip"])   # opened the right one

    def test_yields_without_reading_the_list(self):
        root = self._dir(trip={"description": "trip planning to Portugal"},
                         groc={"description": "grocery list"})
        call = _Call(root, ["Home"])
        call.say("Get me into the grocery list one.")
        self.assertEqual(call.opened, ["groc"])
        self.assertFalse(call.called("list_threads"),
                         "should open directly, not read a menu when intent is clear")

    # ── offer a small set only when the caller is unsure ─────────────────────

    def test_offers_when_unsure(self):
        root = self._dir(trip={"description": "trip planning to Portugal"},
                         garden={"description": "garden bed layout", "urgent": True})
        call = _Call(root, ["Home"])
        reply = call.say("Hey, I'm not sure what I've got going on — what's here?")
        self.assertTrue(call.called("list_threads"))
        self.assertEqual(call.opened, [])          # offered, didn't force-open

    def test_ambiguous_asks_which(self):
        root = self._dir(rome={"description": "trip to Rome"},
                         oslo={"description": "trip to Oslo"})
        call = _Call(root, ["Home"])
        reply = call.say("Get me into the trip thread.").lower()
        self.assertEqual(call.opened, [])          # didn't guess
        self.assertTrue("rome" in reply and "oslo" in reply,
                        "should surface the two candidates to disambiguate")

    # ── new thread needs a domain ────────────────────────────────────────────

    def test_new_thread_uses_named_domain(self):
        root = self._dir(trip={"description": "trip planning"})
        call = _Call(root, ["Home", "Garden"])
        call.say("Start a brand-new thread under the Garden project about "
                 "planting tomatoes this weekend.")
        self.assertEqual([d for d, _ in call.created], ["Garden"])

    def test_new_thread_asks_domain_when_missing(self):
        root = self._dir(trip={"description": "trip planning"})
        call = _Call(root, ["Home", "Garden"])
        reply = call.say("I want to start something new.").lower()
        self.assertEqual(call.created, [])         # doesn't invent a domain
        self.assertTrue("home" in reply or "garden" in reply,
                        "should ask which project")

    # ── can't see inside a thread ────────────────────────────────────────────

    def test_no_content_fabrication(self):
        root = self._dir(trip={"description": "trip planning to Portugal"})
        call = _Call(root, ["Home"])
        reply = call.say("Read me back what I decided in the trip planning thread.").lower()
        # It cannot read contents — so it must either say so / offer to connect,
        # or actually connect the caller — never narrate fabricated decisions.
        self.assertTrue(
            call.opened == ["trip"]
            or any(p in reply for p in ("can't", "cannot", "inside", "connect", "pick it up")),
            f"should not fabricate contents; got: {reply!r}")

    # ── spoken output shape ──────────────────────────────────────────────────

    def test_spoken_output_no_markdown(self):
        root = self._dir(trip={"description": "trip planning to Portugal"},
                         garden={"description": "garden bed layout"})
        call = _Call(root, ["Home"])
        reply = call.say("What's going on, give me the rundown.")
        # The prompt says "No lists" for spoken output — reject numbered-list lines
        # (1. / 2.) too, not just markdown bullets/headers/tables/code, so the eval
        # matches the contract (the model must speak the threads, not echo the menu).
        self.assertNotRegex(reply, r"(^|\n)\s*(?:[-*#]|\d+\.)\s|```|\|",
                            "spoken reply must have no lists/markdown/tables/code")
