from typing import List
from langchain_core.messages import BaseMessage

class FakeRunnable:
    """A simple runnable that returns predetermined message sequences."""

    def __init__(self, responses: List[List[BaseMessage]]):
        self._responses = responses
        self._idx = 0

    def invoke(self, *_args, **_kwargs):
        if self._idx >= len(self._responses):
            raise IndexError("No more fake responses")
        resp = self._responses[self._idx]
        self._idx += 1
        return {"messages": resp}
