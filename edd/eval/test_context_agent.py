import tempfile
from textwrap import dedent

from unittest import TestCase

from assist.agent import create_context_agent, AgentHarness

from assist.model_manager import select_chat_model

from .utils import create_filesystem


class TestContextAgent(TestCase):
    """Evals for the context agent's ability to surface relevant context
    from the local filesystem."""

    def create_agent(self, filesystem: dict):
        # Create a temporary directory for testing
        root = tempfile.mkdtemp()
        create_filesystem(root, filesystem)

        return AgentHarness(create_context_agent(self.model,
                                                  root)), root

    def setUp(self):
        self.model = select_chat_model("gpt-oss-20b", 0.1)

    def test_surfaces_todo_files_for_task_request(self):
        """When the query implies a task, surface the task files directly."""
        agent, root = self.create_agent({
            "README.org": "All important files are located in the /workspace directory. Key files include:
            - Todos: gtd/inbox.org
            - Fitness goals: fitness.org",
            "gtd": {"inbox.org": dedent("""\
                * Tasks
                ** TODO Fold laundry
                Just get it done
                ** TODO Buy new pants
                Size 31
                """)},
            "fitness.org": "* 2026\nGoal: swim 40mi\n",
        })
        res = agent.message("I need to add a reminder to buy groceries")
        self.assertRegex(res, "inbox\\.org",
                         "Should surface the todo file path")
        self.assertRegex(res, "(?i)TODO",
                         "Should include existing TODO content so caller knows the format")

    def test_surfaces_relevant_file_for_topic_query(self):
        """Given a topic query, find and surface the right file."""
        agent, root = self.create_agent({
            "README.org": "Fitness tracking in fitness.org, travel notes in paris.org",
            "fitness.org": dedent("""\
                * 2025
                I swam 20mi in 3 months
                * 2026
                Goal: swim 40mi
                """),
            "paris.org": dedent("""\
                Paris is the capital of France.
                Population: 2 million.
                """),
        })
        res = agent.message("What are my fitness goals?")
        self.assertIn("40", res,
                      "Should surface the swim goal from fitness.org")
        self.assertRegex(res, "(?i)fitness\\.org",
                         "Should reference the fitness file")

    def test_does_not_surface_irrelevant_files(self):
        """Should only surface files relevant to the query, not everything."""
        agent, root = self.create_agent({
            "README.org": "Various notes in org files",
            "fitness.org": "* 2026\nGoal: swim 40mi\n",
            "finances.org": "* Budget\nMonthly: $5000\n",
            "paris.org": dedent("""\
                Paris is the capital of France.
                Known as the City of Light.
                """),
        })
        res = agent.message("Tell me about Paris")
        self.assertRegex(res, "(?i)capital|City of Light",
                         "Should surface Paris content")
        self.assertNotRegex(res, "(?i)swim|budget",
                            "Should not surface unrelated fitness or finance content")

    def test_answers_direct_question_with_evidence(self):
        """Should answer a factual question from files and cite the source."""
        agent, root = self.create_agent({
            "README.org": dedent("""\
                Fitness tracking in fitness.org.
                I have files that track fitness and my son Roman's progress.
                My wife's name is Ana.
                """),
            "fitness.org": dedent("""\
                * 2025
                I swam 20mi in 3 months
                ** Program
                January: 2 times a week, 20m each
                February: 2 times a week, 30m each
                March: 3 times a week, 30m each
                * 2026
                Goal: swim 40mi
                """),
        })
        res = agent.message("How many miles did I swim in 2025?")
        self.assertIn("20", res,
                      "Should answer with the correct distance")
        self.assertRegex(res, "(?i)fitness\\.org",
                         "Should reference the source file")

    def test_follows_readme_to_find_files(self):
        """Should use README to discover where files are, not guess."""
        agent, root = self.create_agent({
            "README.org": "All task management is in the planner/ directory",
            "planner": {"tasks.org": dedent("""\
                * Active
                ** TODO Write quarterly report
                Due next Friday
                ** TODO Schedule dentist appointment
                """)},
        })
        res = agent.message("What tasks do I have?")
        self.assertRegex(res, "(?i)quarterly report|dentist",
                         "Should find tasks via README guidance")
        self.assertRegex(res, "(?i)planner|tasks\\.org",
                         "Should reference the file path")

    def test_surfaces_user_personal_info(self):
        """Should surface personal information about the user from files."""
        agent, root = self.create_agent({
            "README.org": dedent("""\
                I have files that track fitness, finances, and
                Roman's (my son) progress. I also keep notes about
                travel plans and ideas for gifts for my wife (Ana).
                """),
            "roman.org": dedent("""\
                * Roman's Progress
                ** 2026
                Started kindergarten in September.
                Loves dinosaurs and building blocks.
                """),
        })
        res = agent.message("What is my son interested in?")
        self.assertRegex(res, "(?i)dinosaur",
                         "Should surface Roman's interests")
        self.assertRegex(res, "(?i)roman",
                         "Should identify the son by name")

    # --- No README: agent must explore on its own ---

    def test_no_readme_finds_file_by_name(self):
        """No README present. Agent should ls and find files by name."""
        agent, root = self.create_agent({
            "recipes.org": dedent("""\
                * Favorites
                ** Pasta Carbonara
                Eggs, pecorino, guanciale, black pepper
                ** Thai Green Curry
                Coconut milk, green curry paste, chicken, basil
                """),
            "fitness.org": "* 2026\nGoal: swim 40mi\n",
        })
        res = agent.message("What are my favorite recipes?")
        self.assertRegex(res, "(?i)carbonara|curry",
                         "Should find recipes without README guidance")

    def test_no_readme_finds_file_by_content(self):
        """No README. File name doesn't directly hint at the answer;
        agent must grep or read files to find relevant content."""
        agent, root = self.create_agent({
            "notes.org": dedent("""\
                * House Projects
                ** Kitchen renovation
                Budget: $15,000
                Timeline: March-May 2026
                Contractor: Mike's Remodeling (555-0123)
                ** Backyard fence
                Need quotes from 3 companies
                """),
            "goals.org": "* 2026\n- Run a marathon\n- Read 24 books\n",
        })
        res = agent.message("What's the budget for my kitchen renovation?")
        self.assertRegex(res, "15.?000",
                         "Should find the budget amount")

    def test_no_readme_nested_directories(self):
        """No README. Files are nested in subdirectories.
        Agent must explore directory structure."""
        agent, root = self.create_agent({
            "work": {
                "projects": {
                    "alpha.org": dedent("""\
                        * Project Alpha
                        Status: In progress
                        Deadline: 2026-03-15
                        Lead: Sarah
                        """),
                    "beta.org": "* Project Beta\nStatus: Planning\n",
                },
            },
            "personal": {
                "journal.org": "* January 2026\nStarted new job.\n",
            },
        })
        res = agent.message("When is Project Alpha due?")
        self.assertRegex(res, "2026-03-15|March",
                         "Should find the deadline in nested dirs")

    def test_includes_org_format_guidance_for_org_files(self):
        """When surfacing .org files, should include formatting instructions
        so the caller knows how to properly edit them."""
        agent, root = self.create_agent({
            "README.org": "All important files are located in the /workspace directory. Key files include:
            - Todos: gtd/inbox.org
            - Fitness goals: fitness.org",
            "gtd": {"inbox.org": dedent("""\
                * Tasks
                ** TODO Fold laundry
                Just get it done
                ** TODO Buy new pants
                Size 31
                """)},
        })
        res = agent.message("I need to add a reminder to buy groceries")
        # Should surface the file
        self.assertRegex(res, "inbox\\.org",
                         "Should surface the todo file path")
        # Should include org formatting guidance about headings/structure
        self.assertRegex(res, "(?i)(\\*.*heading|heading.*\\*|asterisk|org.*(format|structure))",
                         "Should include org formatting guidance when surfacing .org files")
        # Should mention the heading-body relationship (critical for correct edits)
        self.assertRegex(res, "(?i)(body|content.*between|section|before the next heading)",
                         "Should explain heading-body structure")

    def test_no_org_format_guidance_for_non_org_files(self):
        """When surfacing non-.org files, should NOT include org formatting instructions."""
        agent, root = self.create_agent({
            "README.md": "Tasks are tracked in tasks.md",
            "tasks.md": dedent("""\
                # Tasks
                - [ ] Fold laundry
                - [ ] Buy new pants (size 31)
                """),
        })
        res = agent.message("I need to add a task about buying groceries")
        # Should surface the file
        self.assertRegex(res, "tasks\\.md",
                         "Should surface the todo file path")
        # Should NOT include actual org formatting guide content
        self.assertNotRegex(res, "(?i)Headings/Subheadings.*asterisk",
                            "Should not include org formatting guide content for .md files")

    def test_no_readme_cross_file_context(self):
        """No README. Answer requires info from multiple files."""
        agent, root = self.create_agent({
            "family.org": dedent("""\
                * Family
                ** Ana (wife)
                Birthday: June 15
                Loves gardening and French cuisine
                ** Roman (son, age 5)
                Birthday: September 3
                Loves dinosaurs
                """),
            "gift-ideas.org": dedent("""\
                * Gift Ideas
                ** Ana
                - Herb garden kit
                - French cookbook by Julia Child
                - Spa day voucher
                ** Roman
                - Dinosaur puzzle set
                - Building blocks
                """),
        })
        res = agent.message("What should I get Ana for her birthday?")
        self.assertRegex(res, "(?i)herb garden|cookbook|spa",
                         "Should surface gift ideas for Ana")
        self.assertRegex(res, "(?i)gardening|French cuisine",
                         "Should connect interests to gift suggestions")
