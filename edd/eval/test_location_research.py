"""Effect of the render/map skill on LOCATION research (Pierre): does loading the
render skill — whose Maps section says "research the places first, THEN map" —
DISSUADE the agent from researching a location question before mapping? The worry
(SKILL.md review): the map guidance makes the agent map made-up places instead of
researching real ones.

Runs the SAME location questions — ones that REQUIRE finding real places (not
named up front) — WITH and WITHOUT the render skill, and compares. The general
agent has no `search_internet` itself, so "did it research" == did it dispatch
the research sub-agent (a `task` call). The regression to catch: WITH the skill,
research is skipped (map_data/map block but no `task`).

Real-LLM eval (small model) — deploy venv; partial pass rates expected. Prints the
with/without metrics so the EFFECT is visible, not just pass/fail.
"""
import os
import re
import shutil
import tempfile
from unittest import TestCase

from deepagents.backends import FilesystemBackend
from langchain_core.messages import AIMessage

from assist.agent import create_agent, AgentHarness
from assist.model_manager import select_assistant_model
from assist.spec import AgentSpec

from .utils import create_filesystem, stub_research_subagent

_RENDER_SKILLS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "assist", "web_skills")

# Location questions that REQUIRE finding real places first (not named up front).
_QUESTIONS = [
    "What are the best-rated coffee shops with wifi within walking distance of "
    "Dolores Park in San Francisco?",
    "Find me a few highly-rated ramen places near Union Square in San Francisco, "
    "and roughly how far each is.",
]


def _metrics(agent) -> dict:
    tasks = map_calls = 0
    answer = ""
    for m in agent.all_messages():
        for t in (getattr(m, "tool_calls", None) or []):
            n = t.get("name", "")
            tasks += (n == "task")
            map_calls += (n == "map_data")
        if isinstance(m, AIMessage) and isinstance(m.content, str) and m.content.strip():
            answer = m.content
    map_block = bool(re.search(r"```render\b.*?type:\s*map", answer, re.S | re.I))
    return {"research_dispatched": tasks, "map_data_calls": map_calls,
            "map_block": map_block, "answer_len": len(answer)}


class TestLocationResearch(TestCase):
    def setUp(self):
        self.model = select_assistant_model(0.1)

    def _run(self, question: str, with_render: bool) -> dict:
        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        create_filesystem(root, {"README.org": "Personal assistant workspace."})
        skills = ({"/render-skill/": FilesystemBackend(root_dir=_RENDER_SKILLS_DIR,
                                                        virtual_mode=True)}
                  if with_render else {})
        # Stub the research subagent: this eval tests whether the render/map skill
        # SUPPRESSES research DISPATCH (it counts `task` calls) — not the research
        # results — so we don't need real search.  The orchestrator still issues the
        # dispatch; the stub just returns fast (rate-limit-free + deterministic).
        with stub_research_subagent("Found several highly-rated options near the area."):
            agent = AgentHarness(create_agent(self.model, root,
                                              spec=AgentSpec(skill_sources=skills)))
            agent.message(question)
        return _metrics(agent)

    def test_map_skill_does_not_suppress_research(self):
        """KEY test: WITH the render skill loaded, a location research question must
        STILL dispatch research (a `task` call) — not skip it to just map."""
        for q in _QUESTIONS:
            m = self._run(q, with_render=True)
            print(f"WITH    render | {q[:48]!r} -> {m}")
            self.assertGreaterEqual(
                m["research_dispatched"], 1,
                f"render/map skill SUPPRESSED research (0 task dispatches) for: {q}\n{m}")

    def test_without_render_skill_baseline(self):
        """Baseline: the same questions without the render skill still research."""
        for q in _QUESTIONS:
            m = self._run(q, with_render=False)
            print(f"WITHOUT render | {q[:48]!r} -> {m}")
            self.assertGreaterEqual(
                m["research_dispatched"], 1,
                f"baseline: no research dispatch for: {q}\n{m}")
