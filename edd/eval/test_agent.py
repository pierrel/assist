import functools
import os
import re
import subprocess
import tempfile
import shutil
from textwrap import dedent

from unittest import TestCase
from unittest.mock import patch

from langchain_core.messages import AIMessage

from assist.model_manager import select_assistant_model
from assist.agent import create_agent, AgentHarness
from assist.sandbox_manager import SandboxManager

from .utils import read_file, create_filesystem, AgentTestMixin

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


def _cleanup_workspace(path: str) -> None:
    """Remove a workspace dir, using Docker for root-owned files.

    Mirrors the cleanup helpers in test_dev_agent.py and
    test_calculate_skill.py — sandbox commands write files as root, and
    plain shutil.rmtree fails on those without an intermediate chmod.
    """
    try:
        subprocess.run(
            ['docker', 'run', '--rm', '-v', f'{path}:/cleanup',
             'alpine', 'sh', '-c',
             'chmod -R 777 /cleanup 2>/dev/null; rm -rf /cleanup/*'],
            check=False, timeout=60,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass
    shutil.rmtree(path, ignore_errors=True)


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
        _cleanup_workspace(self.workspace)

    def _create_agent(self, filesystem: dict | None = None):
        if filesystem:
            create_filesystem(self.workspace, filesystem)
        return AgentHarness(create_agent(
            self.model,
            self.workspace,
            sandbox_backend=self.sandbox,
        ))

    def _skill_was_loaded(self, agent, skill_name: str) -> bool:
        path_needle = f"/skills/{skill_name}/"
        for m in agent.all_messages():
            if not isinstance(m, AIMessage) or not m.tool_calls:
                continue
            for tc in m.tool_calls:
                args = tc.get("args") or {}
                if (tc.get("name") == "load_skill"
                        and args.get("name") == skill_name):
                    return True
                for v in args.values():
                    if isinstance(v, str) and path_needle in v:
                        return True
        return False

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
            self._skill_was_loaded(agent, "calculate"),
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
    with NO fallback, and RAISES when the backend is down (failures must be
    loud, not silently degraded).  The research subagent surfaces that as a
    tool error; the contract this eval pins is unchanged in spirit: dispatch
    research-agent once, read the unavailable signal, relay it, stop — don't
    loop and don't answer from the model's own knowledge.  A deterministic
    backstop (``LoopDetectionMiddleware`` Pattern F,
    ``subagent_dispatch_threshold=1``) strips a within-turn re-dispatch; a
    cross-turn re-dispatch stays the prompt's job.

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

    def _run_with_blocked_search(self, prompt: str):
        """Build the general agent with search forced to fail loudly (as a
        down SearXNG backend does), send ``prompt``, return ``(agent, res)``."""
        import assist.tools as _tools

        @functools.wraps(_tools.search_internet)
        def _blocked_search(query, max_results=5):
            # Mirror the real failure: search_internet raises when the
            # SearXNG backend is unavailable (no fallback).
            raise RuntimeError(
                "Web search backend (SearXNG) is unavailable: connection refused"
            )

        # read_url returns an unavailable-flavoured error too, so the whole
        # external surface speaks with one voice and the model can't read
        # eval-awareness into a bespoke string.  Mirrors read_url's real
        # "Error fetching URL: <e>" shape.  The eval asserts nothing about
        # read_url; this is purely network safety.
        @functools.wraps(_tools.read_url)
        def _blocked_fetch(url):
            return "Error fetching URL: web search/fetch is unavailable right now."

        return _run_general_agent_with_search_stubs(
            self, self.model, prompt, _blocked_search, _blocked_fetch)

    def _assert_relays_unavailable(self, agent, res):
        # Belt-and-suspenders: a recursion-killed / empty turn fails here
        # with a clearer message than the regexes' "didn't match".
        self.assertTrue(res, "Agent returned an empty response")

        # 1. research-agent dispatched EXACTLY once.  Zero would mean the
        #    parent answered from its own knowledge (a different failure
        #    the prompt forbids); two-or-more is the runaway meta-loop.
        research_dispatches = [s for s in self.subagent_calls(agent)
                               if s == "research-agent"]
        self.assertEqual(
            len(research_dispatches), 1,
            "Expected research-agent dispatched exactly once (dispatch, "
            "read the blocked signal, stop).  Dispatches seen: "
            f"{self.subagent_calls(agent)}")

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

    def test_relays_unavailable_tech_lookup(self):
        agent, res = self._run_with_blocked_search(
            "What's the latest LangGraph release and what changed?")
        self._assert_relays_unavailable(agent, res)

    def test_relays_unavailable_news_lookup(self):
        # A differently-shaped, non-technical lookup.  Same contract — if
        # the agent heeds the blocked signal here too, it's reading the
        # signal, not pattern-matching one LangGraph-shaped prompt.
        agent, res = self._run_with_blocked_search(
            "What were the headlines from the Federal Reserve's most "
            "recent interest-rate decision?")
        self._assert_relays_unavailable(agent, res)


class TestResearchSearchBudget(AgentTestMixin, TestCase):
    """The research flow must converge on a bounded number of searches.

    Background: a prod thread (a trivial product lookup) did ~100
    `search_internet` calls across three nested research-agent dispatches
    for a trivial query — the research orchestrator re-dispatched the
    inner research-agent, each of which searched dozens of times under
    the only guidance it had ("conduct thorough research").  This eval
    pins an effort budget: a healthy research turn searches a handful of
    times and finalizes.

    We patch `search_internet` to a counted stub returning canned,
    real-looking results (so the model has usable hits and isn't
    searching for lack of results), and assert the TOTAL invocation count
    across the whole flow stays under budget.  The count is the right
    observable: the orchestrator's inner re-dispatches live in nested
    sub-agent namespaces invisible to the top-level message state, but
    every search — at any nesting depth — goes through the one patched
    function, so a global counter captures aggregate effort exactly.

    Same patch-site reasoning as TestResearchSearchUnavailableHandoff: patch
    `assist.agent.search_internet` (the name bound into assist.agent at
    import), not `assist.tools.search_internet`.
    """

    # Aggregate effort budget for a single research turn.  The designed
    # ceiling is mechanical: the orchestrator does not search (it delegates),
    # so the research-agent is the sole searcher, capped per-agent at
    # _RESEARCH_TOOL_VOLUME_CAP (6) and dispatched once (_SUBAGENT_DISPATCH_CAP
    # = 1).  So ~6 searches by design; 12 is 2x headroom for batch overshoot
    # (the agent can emit several search calls in one message before the cap
    # observes them).  Prod ran ~100; baseline (no fix) ran 15-50.
    SEARCH_BUDGET = 12

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

    def _assert_bounded(self, agent, res, n_searches):
        self.assertTrue(res, "Agent returned an empty response")
        # Lower bound: the flow must actually research.  Catches a
        # regression that "bounds" searches by BREAKING research (answering
        # from the model's own knowledge, or stripping the dispatch) — that
        # would still produce a non-empty res and pass the upper bound alone.
        self.assertGreaterEqual(
            n_searches, 1,
            "Expected the flow to search at least once (it answered without "
            f"searching).  MAIN dispatches: {self.subagent_calls(agent)}")
        # Upper bound: no over-search runaway.  MAIN-level dispatches in the
        # message help locate the source if this fails.
        self.assertLessEqual(
            n_searches, self.SEARCH_BUDGET,
            f"Research flow ran {n_searches} searches for one query — over "
            f"the {self.SEARCH_BUDGET} budget.  MAIN dispatches: "
            f"{self.subagent_calls(agent)}.")

    def test_search_budget_product_lookup(self):
        agent, res, n = self._run_counting_searches(
            "What are good waterproof hiking boots for wet trails?")
        self._assert_bounded(agent, res, n)

    def test_search_budget_howto_lookup(self):
        # A differently-shaped, non-telegraphed query.  Same budget.
        agent, res, n = self._run_counting_searches(
            "How do tabata intervals work?")
        self._assert_bounded(agent, res, n)
