from typing import List, Any
from langchain_core.messages import BaseMessage


class FakeRunnable:
    """A simple runnable that returns predetermined message sequences.

    It optionally appends the provided input messages to the response so it can
    mimic the behaviour of agents that return the entire conversation. All
    invocations are recorded to ``calls`` for inspection in tests.
    """

    def __init__(self, responses: List[List[BaseMessage]], append: bool = False):
        self._responses = responses
        self._idx = 0
        self.append = append
        self.calls: List[Any] = []

    def invoke(self, inputs, *_args, **_kwargs):
        self.calls.append(inputs)
        if self._idx >= len(self._responses):
            raise IndexError("No more fake responses")
        resp = self._responses[self._idx]
        self._idx += 1
        if self.append and isinstance(inputs, dict) and "messages" in inputs:
            return {"messages": list(inputs["messages"]) + resp}
        return {"messages": resp}
