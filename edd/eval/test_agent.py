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

    def test_finance_strategy_projection(self):
        """End-to-end: research strategies + apply them to the user's
        own financial picture, projecting forward with calculate.

        This is the integration the calculate skill was designed for —
        the agent has to:

        1. Read finance.org to understand the user's situation
           (income, savings rate, goal).
        2. Delegate to the research-agent for current best-practice
           investment strategies (index funds, bond allocation, etc.).
        3. Load the calculate skill to project the user's portfolio
           forward under those strategies.
        4. Update finance.org with the projected numbers under the
           appropriate heading (org-format skill applies for the
           edit), preserving existing content.

        Assertions intentionally focus on observable side-effects, not
        specific strategy names — the research-agent picks what's
        current, and the agent picks how to phrase it.  We check that
        SOME concrete projected balance lands in the file and that
        the existing content survives.
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

        finance_before = read_file(os.path.join(self.workspace, "finance.org"))

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

        # 3. Org-format skill was loaded — the agent is editing a .org
        # file, and the heading-body insertion rule applies here just
        # as in test_org_format_skill.  Without it, projected output is
        # likely to land between * Income and its body, orphaning all
        # the seed data underneath.
        self.assertTrue(
            self._skill_was_loaded(agent, "org-format"),
            "Should have loaded the org-format skill before editing "
            "finance.org.  Editing a .org file without it is how "
            "headings get orphaned.",
        )

        # 4. finance.org was updated with concrete numbers.
        finance_after = read_file(os.path.join(self.workspace, "finance.org"))
        self.assertNotEqual(
            finance_before, finance_after,
            "finance.org should have been updated with projection results.",
        )
        # Existing content must still be present.  This is the same
        # contract org-format enforces in test_org_format_skill — a
        # write-up under * Goals must not destroy * Income or
        # * Savings.
        for marker in ("$7,500", "$2,000", "$1,500,000", "$50,000"):
            self.assertIn(
                marker, finance_after,
                f"Existing content '{marker}' must be preserved in finance.org.",
            )

        # 4b. Body-preservation check (org-format heading rule): the
        # body of * Income must remain attached to * Income, and
        # similarly for * Savings — the projection should land under
        # * Goals or in a new top-level section, never wedged into
        # the head of an existing one.
        income_idx = finance_after.find("* Income")
        savings_idx = finance_after.find("* Savings")
        goals_idx = finance_after.find("* Goals")
        self.assertGreaterEqual(income_idx, 0)
        self.assertGreaterEqual(savings_idx, 0)
        self.assertGreaterEqual(goals_idx, 0)
        income_section = finance_after[income_idx:savings_idx]
        savings_section = finance_after[savings_idx:goals_idx]
        self.assertIn(
            "$7,500", income_section,
            "$7,500 income figure must remain inside * Income — "
            "the projection wrote ahead of it and orphaned the body.",
        )
        self.assertIn(
            "$2,000", savings_section,
            "$2,000 savings figure must remain inside * Savings — "
            "the projection wrote ahead of it and orphaned the body.",
        )

        # 5. Some plausible projected balance lands in the file — any
        # value >= $100k beyond the seed values, in any common
        # formatting (``$1,234,567``, ``1234567``, ``$1.5M``,
        # ``1.5 million``).  We normalize all of them to a number
        # before comparing.
        seed_numbers = {7500, 2000, 1500000, 50000}

        def _extract_dollar_amounts(text: str) -> list[float]:
            out: list[float] = []
            # $1.5M / 1.5M / $1.5 million / 1.5 million
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
            # $1,234,567 / $1234567 / 1,234,567 in dollar contexts
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

        all_amounts = _extract_dollar_amounts(finance_after)
        novel_big_numbers = sorted({
            int(round(n)) for n in all_amounts
            if n >= 100_000 and int(round(n)) not in seed_numbers
        })
        self.assertTrue(
            novel_big_numbers,
            "finance.org should contain at least one new dollar figure "
            "(>= $100k) added by the projection.  Existing seed figures "
            "alone are not evidence the projection ran.  "
            f"All extracted amounts: {all_amounts}.  "
            f"finance.org after:\n{finance_after}",
        )

        # 6. Response itself should reference at least one of the
        # projected figures it wrote — confirms the agent isn't just
        # stuffing numbers into the file silently.  Match either the
        # bare integer, the comma-formatted form, or M/K shorthand
        # within 5% of the value.
        response_amounts = _extract_dollar_amounts(res)
        any_match = any(
            any(abs(r - n) / n < 0.05 for r in response_amounts)
            for n in novel_big_numbers
        )
        self.assertTrue(
            any_match,
            "Response should mention at least one of the projected "
            f"balances it wrote to finance.org.  File novel: "
            f"{novel_big_numbers}.  Response amounts: "
            f"{response_amounts}.  Response: {res[:500]}",
        )
