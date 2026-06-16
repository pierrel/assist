import functools
import os
import re
import tempfile
import shutil
from textwrap import dedent

from unittest import TestCase
from unittest.mock import patch

from langchain_core.messages import AIMessage

from assist.model_manager import select_assistant_model
from assist.agent import create_agent, AgentHarness
from assist.sandbox_manager import SandboxManager

from .utils import read_file, create_filesystem, AgentTestMixin, skill_was_loaded, cleanup_workspace

class TestAgent(AgentTestMixin, TestCase):
    def create_agent(self, filesystem: dict):
        root = tempfile.mkdtemp()
        create_filesystem(root, filesystem)

        return AgentHarness(create_agent(self.model,
                                         root)), root

    def setUp(self):
        self.model = select_assistant_model(0.1)

    def test_adds_item_correctly(self):
        agent, root = self.create_agent({"README.org": "All of my todos are in gtd/inbox.org",
                                         "gtd": {"inbox.org":
                                                 dedent("""\
                                                 * Tasks
                                                 ** TODO Fold laundry
                                                 Just get it done
                                                 ** TODO Buy new pants
                                                 Size 31
                                                 """)}})
        res = agent.message("I need a new washer/dryer")
        inbox_contents = read_file(f"{root}/gtd/inbox.org")
        self.assertRegex(res, "(?i)updated|added", "Should mention that a change was made.")
        self.assertRegex(inbox_contents,
                         "(?im)^\\*\\* TODO.*dryer",
                         "Should have added a TODO with dryer in the heading")
        self.assertRegex(inbox_contents,
                         "laundry\nJust get it done",
                         "Should not have split a TODO item")
        self.assertRegex(inbox_contents,
                         "pants\nSize",
                         "Should not have split a TODO item")


    def test_finds_and_updates_relevant_files_direct(self):
        agent, root = self.create_agent({"README.org": "All of my todos are in gtd/inbox.org",
                                         "fitness.org": dedent("""\
                                         * 2025
                                         I swam 20mi in 3 months
                                         ** Program
                                         January: 2 times a week, 20m each
                                         February: 2 times a week, 30m each
                                         March: 3 times a week, 30m each
                                         July: 3 times a week, 1mi each
                                         October: 3 times a week, 2mi each
                                         December: 3mi swim
                                         * 2026
                                         Goal: swim 40mi
                                         """),
                                         "gtd": {"inbox.org":
                                                 dedent("""\
                                                 * Tasks
                                                 ** TODO Fold laundry
                                                 Just get it done
                                                 ** TODO Buy new pants
                                                 Size 31
                                                 """)}})
        plan_before = read_file(f"{root}/fitness.org")
        res = agent.message("Create a plan for me to reach my swim goal for 2026.")
        plan_after = read_file(f"{root}/fitness.org")
        self.assertNotEqual(plan_before, plan_after, "It should have updated the plan")
        # Should add content under the 2026 section with a program/plan/training schedule
        self.assertRegex(plan_after, "(?is)\\* 2026.*(Program|Plan|Training|Schedule)",
                         "It should add a 2026 program/plan section")

    def test_research_saved_to_references(self):
        """Research should be saved to the references directory when it exists."""
        agent, root = self.create_agent({
            "README.org": dedent("""\
                * Research
                The result of general research is placed in the =references= directory.
                """),
            "references": {".keep": ""},
        })

        res = agent.message('What are the best practices for import statements in python? What does guido recommend?')

        self.assertIsNotNone(res)

        # The report should be written into the references directory
        report_path = os.path.join(root, "references")
        files = [f for f in os.listdir(report_path) if f != ".keep"]
        self.assertGreaterEqual(len(files), 1,
                                f"Report should be written to the references directory. Current files: {files}")

    def test_generic_quesion(self):
        agent, root = self.create_agent({
            "README.org": dedent(
                """
                Research results go in the =references= directory.
                """
            ),
            "references": {".keep": ""},
        })

        res = agent.message(
            "How do I maintain a resumable/persistent session in eMacs over tramp/ssh? For example, I want to connect from my laptop to a server with tramp/ssh, run a long-running command and then close/suspend my laptop. Then come back later and see the results. Similar to what you would do in screen or tmux."
        )

        # Assert that the agent delegates to the research subagent
        self.assertToolCall(agent, "task", "Should delegate to research agent")

        # Assert that the agent does not refuse the request
        self.assertNotIn(
            "I'm sorry, but I can't help with that.",
            res,
            "Agent should not refuse the request"
        )

    def test_combines_context_and_research(self):
        """When user asks about a topic with local context, agent should
        use both local files and research to give a complete answer."""
        agent, root = self.create_agent({
            "README.org": dedent("""\
                Research results go in the =references= directory.
                """),
            "references": {".keep": ""},
            "fitness.org": dedent("""\
                * 2025
                I swam 20mi in 3 months
                * 2026
                Goal: swim 40mi
                """),
        })
        res = agent.message(
            "I want to reach my swim goal of 40mi this year. "
            "What training plan would experts recommend for this distance?"
        )
        # Should have delegated to research for expert recommendations
        self.assertToolCall(agent, "task", "Should use subagent for research or context")
        # Response should reference the user's specific goal
        self.assertIn("40", res, "Should reference the user's specific goal")


def _extract_dollar_amounts(text: str) -> list[float]:
    """Pull dollar amounts out of free-form text in common shapes.

    Handles ``$1.5M`` / ``1.5 million`` shorthand and ``$1,234,567`` /
    ``$1234567`` numeric forms.  Used by the finance integration test to
    compare projected figures across response prose and saved files
    without caring about format.
    """
    out: list[float] = []
    for m in re.finditer(
        r"\$?\s?(\d+(?:\.\d+)?)\s?(M|million|K|thousand|B|billion)\b",
        text,
        flags=re.IGNORECASE,
    ):
        value = float(m.group(1))
        suffix = m.group(2).lower()
        if suffix in ("m", "million"):
            value *= 1_000_000
        elif suffix in ("k", "thousand"):
            value *= 1_000
        elif suffix in ("b", "billion"):
            value *= 1_000_000_000
        out.append(value)
    for m in re.finditer(
        r"\$\s?(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)",
        text,
    ):
        tok = m.group(1).replace(",", "")
        try:
            out.append(float(tok))
        except ValueError:
            pass
    return out


def _artifact_contents(workspace: str) -> list[str]:
    """Return the text content of every regular file under workspace.

    Used by the finance integration test to compare projected figures
    across whatever artifacts the agent produced — finance.org if it
    edited in place, or a new report.md/.org if it wrote elsewhere.
    Format-agnostic by design.

    Skips dotfiles, the references/.keep marker, and binary files
    (anything that fails utf-8 decode).
    """
    out: list[str] = []
    for dirpath, _dirnames, filenames in os.walk(workspace):
        for fname in filenames:
            if fname.startswith(".") or fname == ".keep":
                continue
            path = os.path.join(dirpath, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    out.append(f.read())
            except (OSError, UnicodeDecodeError):
                continue
    return out


class TestAgentSandboxIntegration(AgentTestMixin, TestCase):
    """Integration evals that need a real sandbox.

    Lives alongside ``TestAgent`` because the user thinks of these as
    "test_agent" cases — but kept in a separate class so we don't have
    to retrofit a sandbox onto the no-sandbox tests above (each of
    which would suddenly need Docker just to run).
    """

    @classmethod
    def setUpClass(cls):
        cls.model = select_assistant_model(0.1)

    def setUp(self):
        self.workspace = tempfile.mkdtemp(prefix="agent_integration_eval_")
        self.sandbox = SandboxManager.get_sandbox_backend(self.workspace)
        if self.sandbox is None:
            self.skipTest(
                "Docker sandbox unavailable — is Docker running and "
                "assist-sandbox built?"
            )

    def tearDown(self):
        SandboxManager.cleanup(self.workspace)
        cleanup_workspace(self.workspace)

    def _create_agent(self, filesystem: dict | None = None):
        if filesystem:
            create_filesystem(self.workspace, filesystem)
        return AgentHarness(create_agent(
            self.model,
            self.workspace,
            sandbox_backend=self.sandbox,
        ))

    def _ran_python_via_execute(self, agent) -> bool:
        """True iff at least one `execute` tool call invoked Python.

        The calculate skill's contract is "verify by running Python";
        the finance integration test should not pass if the agent
        loaded calculate but never actually ran any computation.
        """
        for m in agent.all_messages():
            if not isinstance(m, AIMessage) or not m.tool_calls:
                continue
            for tc in m.tool_calls:
                if tc.get("name") != "execute":
                    continue
                cmd = (tc.get("args") or {}).get("command", "")
                if re.search(r"\bpython3?\b", cmd):
                    return True
                if re.search(r"\b\w+\.py\b", cmd):
                    return True
        return False

    def test_finance_strategy_projection(self):
        """End-to-end: research strategies + apply them to the user's
        own financial picture, projecting forward with calculate.

        This is the integration the calculate skill was designed for.
        The contract:

        1. Delegate to research-agent for current best-practice
           investment strategies (index funds, bond allocation, etc.).
        2. Load the calculate skill to project the user's portfolio
           forward.
        3. Use the user's actual numbers (income, savings rate, goal)
           from the existing finance.org — not generic placeholders.
        4. Land a plausible projected figure SOMEWHERE the user can
           find it (existing finance.org or a new report file the
           agent created — both are acceptable).

        We deliberately do NOT assert on file format here.  Whether
        the agent updates finance.org or writes a new markdown report
        is its call; the org-format skill is exercised separately by
        test_org_format_skill.py and test_skill_loading.py.  The
        contract this test pins is the integration, not the format.
        """
        agent = self._create_agent({
            "README.org": dedent("""\
                * Finances
                Long-term financial plans live in finance.org.
                Research results go in the =references= directory.
                """),
            "references": {".keep": ""},
            "finance.org": dedent("""\
                * Income
                Take-home pay: $7,500/month after tax.

                * Savings
                Currently saving $2,000/month into a brokerage account.

                * Goals
                ** Retirement
                Target: $1,500,000 by age 65.
                Current age: 35.
                Current invested balance: $50,000.
                """),
        })

        res = agent.message(
            "Research the best long-term investment strategies for "
            "retirement, then project how those strategies will work "
            "on my income and savings rate to reach my retirement goal."
        )

        # 1. Research delegation happened.
        subagents = self.subagent_calls(agent)
        self.assertIn(
            "research-agent", subagents,
            f"Should have delegated strategy research to research-agent. "
            f"Task subagents called: {subagents}",
        )

        # 2. Calculate skill was loaded for the projection.
        self.assertTrue(
            skill_was_loaded(agent, "calculate"),
            "Should have loaded the calculate skill to project the user's "
            "balance forward.  A finance projection without running the "
            "math is exactly the failure mode this skill exists to prevent.",
        )

        # 2b. Calculate skill actually ran Python.  Loading the skill
        # without running execute(python) is the same failure mode the
        # skill exists to prevent — guessing a projection.
        self.assertTrue(
            self._ran_python_via_execute(agent),
            "Calculate skill loaded but no `execute` call ran Python — "
            "the projection was eyeballed, not computed.  This is the "
            "exact failure the skill is designed to catch.",
        )

        # 3. The agent used the user's specific numbers from finance.org
        # — not generic placeholders or strategy-research defaults.  We
        # check that at least one of the seed numbers (income, savings,
        # goal, current balance) shows up in either the response or any
        # file the agent produced.  This proves it read finance.org and
        # personalized the projection.
        seeds = ("$7,500", "$2,000", "$1,500,000", "$1.5M", "$50,000",
                 "7500", "2000", "1500000", "50000")
        all_artifacts = [res] + _artifact_contents(self.workspace)
        used_seed = any(
            any(s in artifact for s in seeds)
            for artifact in all_artifacts
        )
        self.assertTrue(
            used_seed,
            "Agent should have used the user's specific numbers from "
            "finance.org (income $7,500, savings $2,000, goal $1.5M, or "
            "current balance $50,000).  None appeared in the response "
            "or any file the agent produced — likely the projection ran "
            "on generic numbers, not the user's own situation.",
        )

        # 4. A plausible projected figure (>= $100k) landed somewhere
        # — either finance.org (if the agent edited it in place) or any
        # other file the agent wrote (e.g. a new report.md).  We don't
        # care about format; we care that the projection produced a
        # number the user can find.
        seed_set = {7500, 2000, 1500000, 50000}
        all_amounts: list[float] = []
        for artifact in all_artifacts:
            all_amounts.extend(_extract_dollar_amounts(artifact))
        novel_big_numbers = sorted({
            int(round(n)) for n in all_amounts
            if n >= 100_000 and int(round(n)) not in seed_set
        })
        self.assertTrue(
            novel_big_numbers,
            "Expected at least one projected dollar figure (>= $100k) "
            "in the response or in a file the agent wrote.  "
            f"Amounts found across response + artifacts: {all_amounts}.  "
            f"Response: {res[:500]}",
        )

        # 5. The response itself references at least one of the
        # projected figures within 5% — confirms the agent reported
        # the result to the user, not just buried it in a file.
        response_amounts = _extract_dollar_amounts(res)
        any_match = any(
            any(abs(r - n) / n < 0.05 for r in response_amounts)
            for n in novel_big_numbers
        )
        self.assertTrue(
            any_match,
            "Response should mention at least one of the projected "
            f"figures.  Novel projection numbers: {novel_big_numbers}. "
            f"Response amounts: {response_amounts}.  "
            f"Response: {res[:500]}",
        )


def _run_general_agent_with_search_stubs(test, model, prompt,
                                         search_stub, read_url_stub):
    """Build the general agent with search_internet/read_url patched to the
    given stubs, send ``prompt``, return ``(agent, response)``.

    Shared by the two mocked research evals.  Patches ``assist.agent.*``
    (the names bound into ``assist.agent`` at import, which
    ``create_research_agent``'s tool lists capture) — NOT ``assist.tools.*``.
    Builds the agent INSIDE the patch so those tool lists capture the
    stubs.  Stubs should carry the real functions' metadata via
    ``functools.wraps`` so deepagents wraps them as the same-named tools.
    ``test`` is the TestCase (for ``addCleanup``).
    """
    root = tempfile.mkdtemp(prefix="research_eval_")
    test.addCleanup(shutil.rmtree, root, ignore_errors=True)
    create_filesystem(root, {"README.org": "My notes live in notes/.",
                             "notes": {"misc.org": "Personal notes."}})
    with patch("assist.agent.search_internet", search_stub), \
         patch("assist.agent.read_url", read_url_stub):
        agent = AgentHarness(create_agent(model, root))
        res = agent.message(prompt)
    return agent, res


class TestResearchSearchUnavailableHandoff(AgentTestMixin, TestCase):
    """The general agent must HEED a research-agent that reports search
    is unavailable — relay that to the user, not re-dispatch or fabricate.

    Background: ``search_internet`` now goes through a self-hosted SearXNG
    with NO fallback; when the backend is down it RETURNS
    ``_SEARCH_UNAVAILABLE_MESSAGE`` (logged at ERROR) rather than raising, so
    the agent receives it as a tool result and can relay it without crashing
    the turn.  The contract this eval pins: dispatch research-agent, read the
    unavailable signal, relay it, stop — don't fabricate from the model's own
    knowledge.  This is now PROMPT-driven: the loop-detection rollback removed
    the per-subagent re-dispatch cap (old Pattern F), so a stray extra
    dispatch is tolerated (a few hops are fine) as long as the agent heeds the
    signal and relays it rather than answering from memory.

    MOCKING NOTE — this is the one eval in ``edd/eval/`` that
    monkey-patches tools (``search_internet`` and ``read_url``).  Existing
    evals hit the real LLM + real tools on purpose ("eval-first
    contracts"), but this failure mode is unobservable without forcing the
    search backend to fail.  We patch ``assist.agent.search_internet`` (NOT
    ``assist.tools.search_internet``): ``assist/agent.py`` binds the name at
    import via ``from assist.tools import search_internet``, so the research
    subagent's tool list captured that module-level reference — patching the
    ``assist.tools`` attribute would not rebind it.
    """

    def setUp(self):
        self.model = select_assistant_model(0.1)

    # The search-down circuit breaker (SearchUnavailableBreakerMiddleware)
    # caps a single research-subagent dispatch at ~threshold completed
    # searches before it terminates the turn — so even with a stray extra
    # orchestrator dispatch, total searches stay small.  WITHOUT the breaker
    # the slow model grinds through distinct failed queries to the
    # recursion_limit (observed ~75-100 searches / 14-30 min).  This bound is
    # the SYMPTOM assertion the old version of this eval was missing — it
    # pinned only "relays unavailable", which a 30-min grind also satisfies.
    MAX_SEARCHES_WHEN_DOWN = 10

    def _run_with_blocked_search(self, prompt: str):
        """Build the general agent with search forced to fail loudly (as a
        down SearXNG backend does), send ``prompt``, return
        ``(agent, res, n_searches)``."""
        import assist.tools as _tools
        calls = {"search": 0}

        @functools.wraps(_tools.search_internet)
        def _blocked_search(query, max_results=5):
            # Mirror the real failure: when the SearXNG backend is down,
            # search_internet RETURNS the explicit unavailable message (logged
            # ERROR) rather than raising — so the agent receives it as a tool
            # result and relays it, instead of an exception crashing the turn.
            calls["search"] += 1
            return _tools._SEARCH_UNAVAILABLE_MESSAGE

        # read_url returns an unavailable-flavoured error too, so the whole
        # external surface speaks with one voice and the model can't read
        # eval-awareness into a bespoke string.  Mirrors read_url's real
        # "Error fetching URL: <e>" shape.  The eval asserts nothing about
        # read_url; this is purely network safety.
        @functools.wraps(_tools.read_url)
        def _blocked_fetch(url):
            return "Error fetching URL: web search/fetch is unavailable right now."

        agent, res = _run_general_agent_with_search_stubs(
            self, self.model, prompt, _blocked_search, _blocked_fetch)
        return agent, res, calls["search"]

    def _assert_relays_unavailable(self, agent, res, n_searches):
        # Belt-and-suspenders: a recursion-killed / empty turn fails here
        # with a clearer message than the regexes' "didn't match".
        self.assertTrue(res, "Agent returned an empty response")

        # 1. research-agent dispatched at least once (zero would mean the
        #    parent answered from its own knowledge — a failure the prompt
        #    forbids).  Post-rollback we tolerate a stray extra dispatch (the
        #    re-dispatch cap is gone; a few hops are fine), but cap it loosely
        #    so a true meta-loop still fails.  The relay regex below carries
        #    the real discrimination.
        research_dispatches = [s for s in self.subagent_calls(agent)
                               if s == "research-agent"]
        self.assertGreaterEqual(
            len(research_dispatches), 1,
            "Expected research-agent dispatched at least once (it answered "
            f"without researching).  Dispatches seen: {self.subagent_calls(agent)}")
        self.assertLessEqual(
            len(research_dispatches), 3,
            "research-agent dispatched too many times — a re-dispatch meta-loop. "
            f"Dispatches seen: {self.subagent_calls(agent)}")

        # 2. Final response relays that search is unavailable (rather than
        #    fabricating an answer).  The unavailable token + dispatch==1
        #    carry the discrimination — a fabricated answer won't say the
        #    search was down.  No wait-time assertion: the backend is now a
        #    hard outage to fix, not a rate-limit to wait out, so "try again
        #    in N minutes" is no longer the expected phrasing.
        self.assertRegex(
            res,
            r"(?i)unavailable|couldn'?t search|could not search|search.{0,20}"
            r"(down|failed|unavailable|not available)|rate.?limit|temporarily",
            f"Response should tell the user search is unavailable.  Got: {res[:600]}")

        # 3. SYMPTOM (the assertion this eval used to lack): the search-down
        #    circuit breaker kept total searches BOUNDED — the agent did not
        #    grind through distinct failed queries.  A 30-min grind also
        #    "relays unavailable" eventually, so #1/#2 alone passed while the
        #    grind shipped; this is what actually catches it.
        self.assertLessEqual(
            n_searches, self.MAX_SEARCHES_WHEN_DOWN,
            f"Ran {n_searches} searches against a DOWN backend (cap "
            f"{self.MAX_SEARCHES_WHEN_DOWN}).  A high count is the grind the "
            f"breaker fixes — the model retrying distinct failed queries instead "
            f"of stopping.  MAIN dispatches: {self.subagent_calls(agent)}.")

    def test_relays_unavailable_tech_lookup(self):
        agent, res, n = self._run_with_blocked_search(
            "What's the latest LangGraph release and what changed?")
        self._assert_relays_unavailable(agent, res, n)

    def test_relays_unavailable_news_lookup(self):
        # A differently-shaped, non-technical lookup.  Same contract — if
        # the agent heeds the blocked signal here too, it's reading the
        # signal, not pattern-matching one LangGraph-shaped prompt.
        agent, res, n = self._run_with_blocked_search(
            "What were the headlines from the Federal Reserve's most "
            "recent interest-rate decision?")
        self._assert_relays_unavailable(agent, res, n)


# Loop-detection terminal stubs the FINAL answer must never be — they mean
# an exact-repeat loop (Pattern A/B) cut the turn off before synthesis.
# Lowercased substrings; see loop_detection.py:_compose_terminal_message.
_LOOPDETECT_STUBS = (
    "won't retry that approach",      # Pattern A (no artifact)
    "i won't repeat it",              # Pattern B (no artifact)
    "i've saved the output to",       # either pattern WITH a successful artifact
)


class TestResearchRunsToCompletion(AgentTestMixin, TestCase):
    """A multi-hop research turn must RUN TO COMPLETION, not be cut off.

    This is the positive contract behind the loop-detection rollback: the
    aggressive patterns (distinct-arg thrash, http-failure streak, sheer
    volume, sub-agent re-dispatch) were removed precisely so a research turn
    doing several distinct search/read hops is allowed to finish rather than
    be yanked into a confusing half-finished state.  So we assert the flow
    actually searches, produces a substantive synthesized answer, and is NOT
    a loop-detection give-up stub.

    We deliberately do NOT pin a tight upper bound on search count any more —
    "a few extra hops" is fine, and the real runaway backstop is the research
    agent's recursion_limit (see agent.py), not a per-tool cap.  A very loose
    sanity ceiling stays only to catch a true pathological runaway (the prod
    ~100-search case), not normal exploration.

    We patch `search_internet`/`read_url` to counted stubs returning canned,
    real-looking results (so the model has usable hits and isn't searching
    for lack of results).  Same patch-site reasoning as
    TestResearchSearchUnavailableHandoff: patch `assist.agent.search_internet`
    (the name bound into assist.agent at import), not
    `assist.tools.search_internet`.
    """

    # Loose sanity ceiling only — NOT a design bound.  A healthy turn searches
    # a handful of times; this just catches a true runaway (prod ran ~100).
    # The real bound is the research agent's recursion_limit.
    SEARCH_BUDGET = 30

    def setUp(self):
        self.model = select_assistant_model(0.1)

    def _run_counting_searches(self, prompt: str):
        import assist.tools as _tools
        calls = {"search": 0}

        _canned = str([
            {"title": "Example result one",
             "url": "https://example.com/a",
             "content": "A detailed, directly relevant paragraph answering "
                        "the question with specifics, figures, and names."},
            {"title": "Example result two",
             "url": "https://example.com/b",
             "content": "A second corroborating source with concrete detail "
                        "covering the same topic from another angle."},
        ])

        @functools.wraps(_tools.search_internet)
        def _counted_search(query, max_results=5):
            calls["search"] += 1
            return _canned

        @functools.wraps(_tools.read_url)
        def _canned_fetch(url):
            return ("Relevant page text with concrete, specific information "
                    "that fully answers the question. " * 20)

        agent, res = _run_general_agent_with_search_stubs(
            self, self.model, prompt, _counted_search, _canned_fetch)
        return agent, res, calls["search"]

    def _assert_completes(self, agent, res, n_searches):
        self.assertTrue(res, "Agent returned an empty response")
        # The flow must actually research — catches a regression that answers
        # from the model's own knowledge or strips the dispatch.
        self.assertGreaterEqual(
            n_searches, 1,
            "Expected the flow to search at least once (it answered without "
            f"searching).  MAIN dispatches: {self.subagent_calls(agent)}")
        # The answer must not be a loop-detection give-up stub — that's the
        # completion signal: the turn synthesized instead of being cut off.
        # (No length floor: a correct, concise answer — e.g. "tabata is 20s on
        # / 10s off x8" — has run to completion just as much as a long one, and
        # a length threshold only adds flakiness without adding signal beyond
        # the stub check + "it actually searched".)
        low = res.lower()
        for stub in _LOOPDETECT_STUBS:
            self.assertNotIn(
                stub, low,
                f"Final answer is a loop-detection give-up stub (the turn was "
                f"cut off before synthesis): matched {stub!r}.\nGot: {res[:600]}")
        # Loose sanity ceiling only — NOT the design bound (that's the
        # recursion_limit).  Catches a true pathological runaway, not the few
        # extra hops the rollback intentionally allows.
        self.assertLessEqual(
            n_searches, self.SEARCH_BUDGET,
            f"Research flow ran {n_searches} searches for one query — past the "
            f"loose sanity ceiling of {self.SEARCH_BUDGET}, a likely runaway.  "
            f"MAIN dispatches: {self.subagent_calls(agent)}.")

    def test_research_completes_product_lookup(self):
        agent, res, n = self._run_counting_searches(
            "What are good waterproof hiking boots for wet trails?")
        self._assert_completes(agent, res, n)

    def test_research_completes_howto_lookup(self):
        # A differently-shaped, non-telegraphed query.  Same contract.
        agent, res, n = self._run_counting_searches(
            "How do tabata intervals work?")
        self._assert_completes(agent, res, n)


class TestSearchDownMidflightDoesNotGrind(AgentTestMixin, TestCase):
    """A search backend that goes DOWN mid-research must STOP the turn, not
    grind through distinct retry queries.

    This is the eval the handoff test lacked.  It PRIMES the model into
    retry-mode with one unsatisfying-but-valid result (so it has already
    "learned" to retry), THEN the backend goes down — every subsequent search
    returns ``_SEARCH_UNAVAILABLE_MESSAGE``.  The strengthened prompt ("stop,
    do not retry with a different query") is the first line of defense; the
    ``SearchUnavailableBreakerMiddleware`` is the hard backstop.  Together,
    total searches must stay bounded.  WITHOUT either, the slow model grinds
    distinct queries to the recursion_limit (observed ~75-100 searches /
    14-30 min in prod).

    A/B the breaker end-to-end via ``ASSIST_SEARCH_UNAVAILABLE_THRESHOLD`` — a
    very large value effectively disables it, isolating the prompt's effect.
    """

    # Bound that catches a grind (prod hit ~75-100) while passing a healthy
    # stop.  With the breaker at the default threshold (4) and a stray extra
    # dispatch, completed searches stay well under this.
    MAX_SEARCHES = 15

    def setUp(self):
        self.model = select_assistant_model(0.1)

    def _run_primed_then_down(self, prompt: str):
        """One unsatisfying-but-valid result (primes a retry), then the backend
        goes down for every subsequent search.  Returns (agent, res, n)."""
        import assist.tools as _tools
        calls = {"search": 0}
        _unsat = str([{"title": "Unrelated background",
                       "url": "https://example.com/z",
                       "content": "A page about an unrelated topic that does "
                                  "not address the question at all."}])

        @functools.wraps(_tools.search_internet)
        def _priming_search(query, max_results=5):
            calls["search"] += 1
            if calls["search"] <= 1:
                return _unsat                          # prime a retry
            return _tools._SEARCH_UNAVAILABLE_MESSAGE   # backend goes down

        @functools.wraps(_tools.read_url)
        def _blocked_fetch(url):
            return "Error fetching URL: web search/fetch is unavailable right now."

        agent, res = _run_general_agent_with_search_stubs(
            self, self.model, prompt, _priming_search, _blocked_fetch)
        return agent, res, calls["search"]

    def _assert_bounded_and_relays(self, agent, res, n):
        self.assertTrue(res, "Agent returned an empty response")
        self.assertLessEqual(
            n, self.MAX_SEARCHES,
            f"Ran {n} searches after the backend went down (cap "
            f"{self.MAX_SEARCHES}) — the grind is not bounded.  MAIN "
            f"dispatches: {self.subagent_calls(agent)}.")
        self.assertRegex(
            res,
            r"(?i)unavailable|couldn'?t search|could not search|search.{0,20}"
            r"(down|failed|unavailable|not available)|rate.?limit|temporarily",
            f"Response should relay that search is unavailable.  Got: {res[:600]}")

    def test_search_down_midflight_does_not_grind(self):
        agent, res, n = self._run_primed_then_down(
            "What's the latest LangGraph release and what changed?")
        self._assert_bounded_and_relays(agent, res, n)
