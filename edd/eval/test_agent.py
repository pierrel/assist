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
        self.assertIn("swim 40mi\n** Program", plan_after, "It should add a 2026 plan")
        
    def test_finds_and_updates_relevant_files_indirect(self):
        agent, root = self.create_agent({"README.org": "All of my todos are in gtd/inbox.org",
                                         "paris.org": dedent("""\n
                                         Paris is the capital and largest city of France, with an estimated city population of 2,048,472 in an area of 105.4 km2 (40.7 sq mi), and a metropolitan population of 13,171,056 as of January 2025. Located on the river Seine in the centre of the Île-de-France region, it is the largest metropolitan area and fourth-most populous city in the European Union (EU). Nicknamed the City of Light, partly because of its role in the Age of Enlightenment, Paris has been one of the world's major centres of finance, diplomacy, commerce, culture, fashion, and gastronomy since the 17th century.
                                         """),
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

    def test_scenario(self):
        # Setup filesystem with ONLY necessary files
        agent, root = self.create_agent({
            "README.org": dedent("""
These are the files that I use to manage my everyday life through emacs org mode. All files by default are written in org format.

* Todos
It is also where I keep my todo list(s). I use the gtd method and org TODOs and use the following files to manage my todos:
- gtd/inbox.org - General inbox where todo items begin
- gtd/projects.org - Todo items organized by project. TODOs are placed under the project in general order of either priority or next action.
- gtd/someday.org - Todo items that don't require any immediate attention, mostly saved for future ideas
- gtd/tickler.org - Todo items that will be important at some point in the future

All new items should start in the inbox for later triaging.
* Research
The result of general research is placed in the =references= directory and can be referenced in other files.

* Other
I have files that track fitness, finances, and Roman's (my son) progress. I also keep notes about travel plans and ideas for gifts for my wife (Ana)."""),
            "question.txt": "Research best practices for import statements in Python and Guido van Rossum's recommendations. Provide concise summary and key points.",
            "references": {".keep": ""},
        })

        # Send key message
        res = agent.message('What are the best practices for import statements in python? What does guido recommend?')

        # Assert that the agent responded
        self.assertIsNotNone(res)

        # After the agent finishes, question.txt should be removed
        self.assertFalse(os.path.exists(os.path.join(root, "question.txt")), "question.txt should be deleted after processing")

        # The report should be written into the references directory
        report_path = os.path.join(root, "references", "report_import_best_practices.org")
        self.assertTrue(os.path.exists(report_path), "Report should be created in references directory")

        # Verify that the report contains expected content
        report_content = read_file(report_path)
        self.assertIn("Import Statements Best Practices in Python", report_content, "Report should contain key heading")

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
            "I’m sorry, but I can’t help with that.",
            res,
            "Agent should not refuse the request"
        )



