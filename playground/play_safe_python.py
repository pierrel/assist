"""Demonstrate using :class:`SafePythonTool` with a real LLM."""

from langchain.callbacks.tracers.stdout import ConsoleCallbackHandler
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from assist.tools.safe_python import SafePythonTool


def main() -> None:
    """Run a simple agent that computes compound savings."""
    tool = SafePythonTool()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4).bind_tools([tool])
    preagent = create_react_agent(llm, [tool])
    agent = preagent.with_config({"callbacks": [ConsoleCallbackHandler()]})

    prompt = (
        "If I save $100 every month in an account that started with $10,000 and "
        "provides an interest rate of 3%, how much will the account have in 10 years?"
    )
    resp = agent.invoke({"messages": [HumanMessage(content=prompt)]})
    print(resp["messages"][-1].content)


if __name__ == "__main__":
    main()

