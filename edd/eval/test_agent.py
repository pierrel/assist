import os
import tempfile
import shutil
from textwrap import dedent

from unittest import TestCase

from assist.model_manager import select_chat_model
from assist.agent import create_agent, AgentHarness

from .utils import read_file, create_filesystem, AgentTestMixin

class TestAgent(AgentTestMixin, TestCase):
    def create_agent(self, filesystem: dict):
        root = tempfile.mkdtemp()
        create_filesystem(root, filesystem)

        return AgentHarness(create_agent(self.model,
                                         root)), root

    def setUp(self):
        self.model = select_chat_model("gpt-oss-20b", 0.1)

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
