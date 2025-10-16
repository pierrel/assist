import os
from collections.abc import Iterable
from typing import List

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import Runnable

from assist import general_agent
from assist.agent_types import AgentInvokeResult


def _ensure_sequence(messages: BaseMessage | Iterable[BaseMessage]) -> List[BaseMessage]:
    if isinstance(messages, BaseMessage):
        return [messages]
    if isinstance(messages, Iterable) and not isinstance(messages, (str, bytes)):
        return list(messages)
    raise TypeError("messages must be a BaseMessage or an iterable of BaseMessage instances")


class _FakeAgent(Runnable):
    """Runnable backed by ``GenericFakeChatModel`` for deterministic tests."""

    def __init__(self, responses: List[List[BaseMessage]]) -> None:
        if not responses:
            raise ValueError("Fake agent requires at least one response sequence")

        final_messages: List[AIMessage] = []
        self._responses: List[List[BaseMessage]] = []

        for seq in responses:
            if not seq:
                raise ValueError("Each response sequence must contain at least one message")
            final = seq[-1]
            if not isinstance(final, AIMessage):
                raise TypeError("The final message in each sequence must be an AIMessage")
            self._responses.append(seq)
            final_messages.append(final)

        self._llm = GenericFakeChatModel(messages=iter(final_messages))
        self._idx = 0

    def invoke(self, inputs: dict, *args, **kwargs) -> AgentInvokeResult:
        if self._idx >= len(self._responses):
            raise IndexError("No more fake responses")

        provided = inputs.get("messages", [])
        input_messages = _ensure_sequence(provided)

        sequence = self._responses[self._idx]
        self._idx += 1

        ai_message = self._llm.invoke(input_messages)
        history = [*input_messages, *sequence[:-1], ai_message]
        return AgentInvokeResult(messages=history)

    def stream(self, inputs: dict, *args, **kwargs):
        result = self.invoke(inputs, *args, **kwargs)
        for message in result.messages:
            yield message, {}


def make_test_agent(responses: List[List[BaseMessage]], temperature: float = 0.4) -> Runnable:
    """Return a deterministic fake agent by default or build a real LLM agent.

    Use environment variable TEST_LLM to switch the backend:
    - "chatollama" -> use ChatOllama
    - "huggingface" -> use HuggingFaceChat
    Any other value or unset -> in-memory fake backed by ``GenericFakeChatModel``.
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
        return _FakeAgent(responses)
