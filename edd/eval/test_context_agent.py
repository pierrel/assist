import tempfile
from textwrap import dedent

from unittest import TestCase

from assist.agent import create_context_agent, AgentHarness

from assist.model_manager import select_chat_model

from .utils import read_file, create_filesystem

class TestContextAgent(TestCase):
    def create_agent(self, filesystem: dict):
        root = tempfile.mkdtemp()
        create_filesystem(root, filesystem)

        return AgentHarness(create_context_agent(self.model,
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
        res = agent.message(dedent("""\
        The Parisii people inhabited the Paris area from around the middle of the 3rd century BC. One of the area's major north–south trade routes crossed the Seine on the Île de la Cité, which gradually became an important trading centre. The Parisii traded with many river towns (some as far away as the Iberian Peninsula) and minted their own coins.

        Julius Caesar conquered the Paris Basin for the Roman Republic in 52 BC and began the Roman settlement on Paris's Left Bank. The Roman town was originally called Lutetia (more fully, Lutetia Parisiorum, \"Lutetia of the Parisii\", modern French Lutèce). It became a prosperous city with a forum, baths, temples, theatres, and an amphitheatre.
        """))
        file_after = read_file(f"{root}/paris.org")
        self.assertRegex(res, "(?i)updated|added", "It should have an update")
        self.assertNotEqual(file_before, file_after, "It should have updated the file with relevant information")
