from unittest import TestCase
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from assist.reflexion_agent import build_execute_node, ReflexionState, Plan, Step
from assist.general_agent import general_agent

from ..utils import actual_llm, graphiphy


@tool
def find_file(directory: str, pattern: str) -> list[str]:
    """Find files matching a pattern in a directory.
    
    Args:
        directory: The directory to search in
        pattern: The filename pattern to search for
    
    Returns:
        list[str]: List of matching file paths
    """
    # Placeholder implementation
    return []


class TestFindFileTool(TestCase):
    def setUp(self) -> None:
        llm = actual_llm()
        agent = general_agent(llm, [find_file])
        self.graph = graphiphy(build_execute_node(agent))
        
    def test_find_file_execution(self) -> None:
        """Test that the execution node can use the find_file tool."""
        plan = Plan(
            goal="Find Python files",
            steps=[
                Step(
                    action="Find all Python files in /tmp",
                    objective="Locate Python files for analysis"
                )
            ],
            assumptions=[],
            risks=[]
        )
        
        state: ReflexionState = {
            "messages": [HumanMessage(content="Find all Python files in /tmp")],
            "plan": plan,
            "step_index": 0,
            "history": [],
            "needs_replan": False,
            "plan_check_needed": False,
            "learnings": [],
            "replan_count": 0,
        }
        
        result = self.graph.invoke(state)
        
        # Verify execution completed
        self.assertEqual(result["step_index"], 1)
        self.assertEqual(len(result["history"]), 1)
        
        # Verify the step was executed
        resolution = result["history"][0]
        self.assertEqual(resolution.action, "Find all Python files in /tmp")
        self.assertTrue(len(resolution.resolution) > 0)
