import os
from typing import List
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable

from assist.fake_runnable import FakeRunnable
from assist import general_agent


def make_test_agent(responses: List[List[BaseMessage]], temperature: float = 0.4) -> Runnable:
    """Return FakeRunnable by default, or an Agent using a real LLM if configured.

    Use environment variable TEST_LLM to switch the backend:
    - "chatollama" -> use ChatOllama
    - "huggingface" -> use HuggingFaceChat
    Any other value or unset -> FakeRunnable with ``responses``.
    """
    llm_choice = os.getenv("TEST_LLM", "").lower()
    if llm_choice == "chatollama":
        from langchain_ollama import ChatOllama
        llm = ChatOllama(model=os.getenv("TEST_MODEL", "mistral"), temperature=temperature)
        return general_agent.general_agent(llm, [])
    elif llm_choice == "huggingface":
        from langchain_community.chat_models import HuggingFaceChat
        repo_id = os.getenv("HF_REPO_ID", "HuggingFaceH4/zephyr-7b-beta")
        llm = HuggingFaceChat(repo_id=repo_id, temperature=temperature)
        return general_agent.general_agent(llm, [])
    else:
        return FakeRunnable(responses)
