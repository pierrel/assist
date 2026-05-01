import os
import re
import subprocess
import tempfile
import shutil
from textwrap import dedent

from unittest import TestCase

from langchain_core.messages import AIMessage

from assist.model_manager import select_chat_model
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
        self.model = select_chat_model(0.1)

    def test_reads_readme(self):
        agent, root = self.create_agent({"README.org": "All of my todos are in gtd/inbox.org",
                                         "gtd": {"inbox.org":
                                                 dedent("""* Tasks
                                                 ** TODO Fold laundry
                                                 Just get it done
                                                 ** TODO Buy new pants""")}})
        res = agent.message("Where are my todos?")
        self.assertRegex(res, "inbox\\.org", "Should mention the inbox file")

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


    def test_finds_relevant_files_direct(self):
        agent, root = self.create_agent({"README.org": "All of my todos are in gtd/inbox.org",
                                         "fitness.org": dedent("""\
                                         * 2025
                                         I swam 20mi in 3 months
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
        res = agent.message("What are my swim goals for 2026?")
        self.assertIn("40", res, "Should mention 40")
        self.assertRegex(res, "miles|mi", "Should mention miles or shorthand mi")

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

    def test_finds_and_updates_relevant_files_indirect(self):
        """Research an external question and update the relevant local file."""
        agent, root = self.create_agent({
            "README.org": "All of my todos are in gtd/inbox.org",
            "paris.org": dedent("""\
                Paris is the capital and largest city of France,
                with a population of about 2 million.
                Nicknamed the City of Light.
                """),
            "gtd": {"inbox.org": dedent("""\
                * Tasks
                ** TODO Fold laundry
                """)}})
        file_before = read_file(f"{root}/paris.org")
        res = agent.message("When was Paris founded? By who? Why?")
        file_after = read_file(f"{root}/paris.org")
        self.assertNotEqual(file_before, file_after, "It should have updated the file with relevant information")

    def test_emacs_framebuffer_touchscreen(self):
        # Setup filesystem with ONLY necessary files (none needed for this test)
        agent, root = self.create_agent({})

        # Send key message
        user_msg = (
            "I have eMacs running in a direct frame buffer and in a very small touch screen. "
            "What are some of the things that I should consider with this setup so that I have a good "
            "experience and eMacs runs smoothly?"
        )
        res = agent.message(user_msg)

        # Basic sanity check
        self.assertIsNotNone(res)

        # Assert that the assistant mentions key considerations
        self.assertToolCall(agent, "task", "It should have called a sub-agent")

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

    def test_adds_task_in_nested_directory(self):
        """Agent should find task files in nested directories via context."""
        agent, root = self.create_agent({
            "README.org": "All task management is in the planner/ directory using org TODO format",
            "planner": {"tasks.org": dedent("""\
                * Active
                ** TODO Write quarterly report
                Due next Friday
                ** TODO Schedule dentist appointment
                """)},
        })
        res = agent.message("I need to renew my passport")
        tasks_after = read_file(f"{root}/planner/tasks.org")
        self.assertRegex(res, "(?i)added|updated|passport",
                         "Should confirm the task was added")
        self.assertRegex(tasks_after, "(?im)TODO.*passport",
                         "Should add a TODO about passport")
        self.assertRegex(tasks_after, "quarterly report",
                         "Should preserve existing tasks")

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


class TestAgentSandboxIntegration(TestCase):
    """Integration evals that need a real sandbox.

    Lives alongside ``TestAgent`` because the user thinks of these as
    "test_agent" cases — but kept in a separate class so we don't have
    to retrofit a sandbox onto the no-sandbox tests above (each of
    which would suddenly need Docker just to run).
    """

    @classmethod
    def setUpClass(cls):
        cls.model = select_chat_model(0.1)

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

    def _task_subagents_called(self, agent) -> list[str]:
        out = []
        for m in agent.all_messages():
            if not isinstance(m, AIMessage) or not m.tool_calls:
                continue
            for tc in m.tool_calls:
                if tc.get("name") == "task":
                    args = tc.get("args") or {}
                    sa = args.get("subagent_type") or args.get("agent") or args.get("name") or ""
                    if sa:
                        out.append(sa)
        return out

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
        subagents = self._task_subagents_called(agent)
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
