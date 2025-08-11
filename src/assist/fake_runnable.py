from typing import List, Tuple, Iterator, Dict, Any
from langchain_core.messages import BaseMessage

class FakeInvocation:
    """Container that behaves like both an iterable of messages and a dict."""

    def __init__(self, messages: List[BaseMessage]):
        self._messages = messages

    def __iter__(self) -> Iterator[Tuple[BaseMessage, Dict[str, Any]]]:
        for m in self._messages:
            yield m, {}

    def __getitem__(self, key: str):
        if key == "messages":
            return self._messages
        raise KeyError(key)


class FakeRunnable:
    """A simple runnable that returns predetermined message sequences."""

    def __init__(self, responses: List[List[BaseMessage]]):
        self._responses = responses
        self._idx = 0
        self._schema = None

    def with_structured_output(self, schema):
        self._schema = schema
        return self

    def invoke(self, *_args, **_kwargs):
        if self._idx >= len(self._responses):
            raise IndexError("No more fake responses")
        resp = self._responses[self._idx]
        self._idx += 1
        if self._schema:
            content = resp[0].content
            steps = [line.split('. ', 1)[1] if '. ' in line else line for line in content.splitlines() if line]
            schema = self._schema
            self._schema = None
            return schema(goal="", steps=steps)
        return FakeInvocation(resp)
