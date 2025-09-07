"""Demonstrate using SafePythonTool with a real LLM."""

from assist.general_agent import general_agent
from assist.tools.safe_python import SafePythonTool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI


def main() -> None:
    """Run a simple agent that computes compound savings."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)
    agent = general_agent(llm, [SafePythonTool()])

    prompt = (
        "If I save $100 every month in an account that started with $10,000 and "
        "provides an interest rate of 3%, how much will the account have in 10 years?"
    )
    resp = agent.invoke({"messages": [HumanMessage(content=prompt)]})
    print(resp)


if __name__ == "__main__":
    main()

