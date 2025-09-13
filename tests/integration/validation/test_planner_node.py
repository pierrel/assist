from unittest import TestCase
from langchain_core.messages import HumanMessage

from assist.reflexion_agent import build_plan_node, ReflexionState

from .utils import thinking_llm, graphiphy, base_tools_for_test

class TestPlannerNode(TestCase):
    def setUp(self) -> None:
        llm = thinking_llm("")
        self.graph = graphiphy(build_plan_node(llm,
                                               base_tools_for_test(),
                                               []))

    def ask_node(self, query: str) -> ReflexionState:
        message = HumanMessage(content=query)
        return self.graph.invoke({"messages": [message]})

    def assertNotInPlan(self,
                        thing: str,
                        state: ReflexionState):
        plan = state["plan"]
        self.assertFalse(any([thing in s.action for s in plan.steps]),
                        f"{thing} should not be in a plan step")

    def assertInPlan(self,
                        thing: str,
                        state: ReflexionState):
        plan = state["plan"]
        self.assertTrue(any([thing in s.action for s in plan.steps]),
                        f"{thing} should not be in a plan step")
    
    def test_dont_write_file(self) -> None:
        state = self.ask_node("What is the capital of France?")

        self.assertNotInPlan("write_file_user", state)

    def test_should_write_file(self) -> None:
        state = self.ask_node("Write me a python script to use in my project at /home/pierre/myproject")

        self.assertInPlan("write_file_user", state)

    def test_search_website(self) -> None:
        state = self.ask_node("I remember seeing something about college campuses with the best food on this website: https://www.mentalfloss.com. What's the URL for that article?")

        self.assertInPlan("site_search", state)


    def test_search_webpage(self) -> None:
        state = self.ask_node("Which campus has the best food according to this website: https://www.mentalfloss.com/food/best-and-worst-college-campus-food?utm_source=firefox-newtab-en-us ?")

        self.assertInPlan("page_search", state)


    def test_project_context_without_project(self):
        state = self.graph.invoke({"messages": [HumanMessage(content="Hello, can you explain to me what's in the README file?")]})
        plan = state["plan"]

        self.assertGreater(len(plan.steps), 1, "Should have multiple steps")
        # the following should really be false...
        self.assertInPlan("project_context", state)

    def test_project_context_with_projecct(self):
        state = self.graph.invoke({"messages": [HumanMessage(content="Hello, can you explain to me what's in the README file?\n\nThe context for this request is /home/myhome/project")]})
        plan = state["plan"]

        
        self.assertInPlan("project_context", state)
        self.assertInPlan("README", state)


    def test_tea_brew(self) -> None:
        state = self.graph.invoke({"messages": [HumanMessage(content="How do I brew a cup of tea?")]})
        
        plan = state["plan"]
        has_assumptions = bool(plan.assumptions)
        has_risks = bool(plan.risks)
        has_over_2_steps = len(plan.steps) > 2
        uses_tavily = any("tavily_search" in s.action.lower() for s in plan.steps)

        self.assertTrue(has_assumptions, "Has assumptions")
        self.assertTrue(has_risks, "Has risks")
        self.assertGreater(len(plan.steps), 2, "Should have more than 2 steps")
        self.assertTrue(
                any(
                    "tavily" in step.action.lower()
                    or "search" in step.action.lower()
                    for step in plan.steps
                ),
                "Uses search or reference tool",
            )

    def test_rewrite_more_professional(self) -> None:
        query = "Rewrite this to be more professional."
        examples = [
            "hey—need that report asap. thx.",
            "We kinda dropped the ball on the Q3 metrics.",
        ]
        for example in examples:
            state = self.graph.invoke({
                "messages": [HumanMessage(content=f"{query} {example}")]
            })
            plan = state["plan"]

            self.assertGreater(len(plan.steps), 1, "has at least 2 steps")
            self.assertTrue(
                any(
                    "tavily" in step.action.lower()
                    or "search" in step.action.lower()
                    for step in plan.steps
                ),
                "Uses search or reference tool",
            )

    def test_rephrase_for_ninth_grade(self) -> None:
        query = "Rephrase for a 9th-grade reading level."
        examples = [
            "The municipality’s fiscal posture necessitates austerity measures.",
            "Our platform leverages distributed systems to optimize throughput.",
        ]
        for example in examples:
            state = self.graph.invoke({
                "messages": [HumanMessage(content=f"{query} {example}")]
            })
            plan = state["plan"]
            self.assertGreater(len(plan.steps), 1, "Has at least 2 steps")
            self.assertTrue(
                any(
                    "tavily" in step.action.lower()
                    or "search" in step.action.lower()
                    for step in plan.steps
                ),
                "Uses search or reference tool",
            )

    def test_extract_entities_to_json(self) -> None:
        query = (
            "Extract all dates, people, and organizations from this text into JSON and "
            "consult external references for JSON schema or entity recognition guidelines."
        )
        examples = [
            "On March 2, 2024, Mayor London Breed met with leaders from SFUSD.",
            "Apple hired Sam Patel on 2023-11-14 after interviews at UCSF.",
        ]
        for example in examples:
            state = self.graph.invoke({
                "messages": [HumanMessage(content=f"{query} {example}")]
            })
            plan = state["plan"]
            self.assertGreater(len(plan.steps), 1, "Has at least 2 steps")
            self.assertFalse(
                any(
                    "tavily" in step.action.lower()
                    or "search" in step.action.lower()
                    for step in plan.steps
                ),
                "Uses search or reference tool",
            )

    def test_classify_customer_messages(self) -> None:
        query = (
            "Classify these customer messages into issue categories; return CSV."
        )
        examples = [
            "App crashes when I upload a photo.",
            "How do I reset my password?",
            "Please cancel my subscription.",
            "Charged twice for August.",
            "Search results are super slow.",
            "Two-factor code never arrives.",
            "Dark mode text is unreadable.",
        ]
        for example in examples:
            state = self.graph.invoke({
                "messages": [HumanMessage(content=f"{query} {example}")]
            })
            plan = state["plan"]
            self.assertGreater(len(plan.steps), 1, "Has at least 2 steps")
            self.assertNotInPlan("write_file", state)

    def test_refactor_function_readability(self) -> None:
        query = (
            "Refactor this function for readability and add docstrings and type hints."
        )
        examples = [
            """def f(a,b):
    r=[]
    for i in a:
        if i not in r: r.append(i)
    for j in b:
        if j not in r: r.append(j)
    return r""",
            """def calc(x):
    t=0
    for i in range(len(x)):
        t=t+x[i]
    return t/len(x)""",
        ]
        for example in examples:
            state = self.graph.invoke({
                "messages": [HumanMessage(content=f"{query}\n{example}")]
            })
            plan = state["plan"]
            self.assertGreater(len(plan.steps), 1, "Has at least 2 steps")
            self.assertFalse(
                any(
                    "tavily" in step.action.lower()
                    or "search" in step.action.lower()
                    for step in plan.steps
                ),
                "Uses search or reference tool",
            )

    def test_build_python_cli(self) -> None:
        query = "Implement a small Python CLI with argparse that performs tasks X and Y."
        examples = [
            "X = convert a .txt file to uppercase, Y = count words and print top-5 by frequency.",
            "X = merge two CSVs by 'id', Y = filter rows where 'amount' > 100 and save.",
        ]
        for example in examples:
            state = self.graph.invoke({
                "messages": [HumanMessage(content=f"{query} {example}")]
            })
            plan = state["plan"]
            self.assertGreater(len(plan.steps), 1, "Has at least 2 steps")
            self.assertTrue(
                any(
                    "tavily" in step.action.lower()
                    or "search" in step.action.lower()
                    for step in plan.steps
                ),
                "Uses search or reference tool",
            )

    def test_research_watches(self) -> None:
        query = "Research the best minimalist mechanical watches under $3k; compare and cite."
        examples = [
            "Field watches under $1.5k, 38–40 mm, sapphire, no date.",
            "Dress watches under $2.5k, <10 mm thick, Bauhaus aesthetics.",
        ]
        for example in examples:
            state = self.graph.invoke({
                "messages": [HumanMessage(content=f"{query} {example}")]
            })
            plan = state["plan"]
            self.assertGreater(len(plan.steps), 1, "Has at least 2 steps")
            self.assertTrue(
                any(
                    "tavily" in step.action.lower()
                    or "search" in step.action.lower()
                    for step in plan.steps
                ),
                "Uses search or reference tool",
            )

    def test_day_trip_plan(self) -> None:
        query = (
            "Create a day-trip plan using rideshare only; estimate times/costs; output a tweakable sheet."
        )
        examples = [
            "Sonoma plaza stroll + one tasting + lunch, 4 adults, Saturday 9/20.",
            "Half Moon Bay coastal walk + café lunch, 2 adults, Sunday 10/5.",
        ]
        for example in examples:
            state = self.graph.invoke({
                "messages": [HumanMessage(content=f"{query} {example}")]
            })
            plan = state["plan"]
            self.assertGreater(len(plan.steps), 1, "Has at least 2 steps")
            self.assertTrue(
                any(
                    "tavily" in step.action.lower()
                    or "search" in step.action.lower()
                    for step in plan.steps
                ),
                "Uses search or reference tool",
            )

    def test_file_expense_report(self) -> None:
        query = (
            "File an expense report from provided PDFs: extract line items, code them, total, attach, submit, "
            "and reference IRS guidelines for expense categories."
        )
        examples = [
            "Receipts = 'Lyft $28.34 (08/12), Coffee $6.50 (08/12), Lunch w/ client $54.20 (08/12).',",
            "Receipts = 'SFO⇄LAX airfare $216.90 (08/25), Hotel 1 night $189.00 (08/25), Per-diem dinner $35.00.',",
        ]
        for example in examples:
            state = self.graph.invoke({
                "messages": [HumanMessage(content=f"{query} {example}")]
            })
            plan = state["plan"]
            self.assertGreater(len(plan.steps), 1, "Has at least 2 steps")
            self.assertTrue(
                any(
                    "tavily" in step.action.lower()
                    or "search" in step.action.lower()
                    for step in plan.steps
                ),
                "Uses search or reference tool",
            )

    def test_run_llm_benchmark(self) -> None:
        query = (
            "Run a benchmark comparing three LLMs on a supplied prompt suite; chart quality/latency; memo."
        )
        examples = [
            "Coding suite with tasks like writing a JSON Schema and fixing a failing pytest.",
            "Reasoning suite with summarization, extraction, and classification prompts.",
        ]
        for example in examples:
            state = self.graph.invoke({
                "messages": [HumanMessage(content=f"{query} {example}")]
            })
            plan = state["plan"]
            self.assertGreater(len(plan.steps), 1, "Has at least 2 steps")
            self.assertTrue(
                any(
                    "tavily" in step.action.lower()
                    or "search" in step.action.lower()
                    for step in plan.steps
                ),
                "Uses search or reference tool",
            )

