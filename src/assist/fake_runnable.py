from typing import Any, Dict, Iterator, List, Tuple

from langchain_core.messages import BaseMessage
from pydantic import BaseModel

class FakeInvocation:
    """Container that behaves like both an iterable of messages and a dict."""

    def __init__(self, messages: List[BaseMessage]) -> None:
        self._messages = messages

    def __iter__(self) -> Iterator[Tuple[BaseMessage, Dict[str, Any]]]:
        for m in self._messages:
            yield m, {}

    def __getitem__(self, key: str) -> List[BaseMessage]:
        if key == "messages":
            return self._messages
        raise KeyError(key)


class FakeRunnable:
    """A simple runnable that returns predetermined message sequences."""

    def __init__(self, responses: List[List[BaseMessage]]) -> None:
        self._responses = responses
        self._idx = 0
        self._schema: type[BaseModel] | None = None

    def with_structured_output(self, schema: type[BaseModel]) -> "FakeRunnable":
        self._schema = schema
        return self

    def invoke(self, *args: Any, **kwargs: Any) -> BaseModel | FakeInvocation:
        if self._idx >= len(self._responses):
            raise IndexError("No more fake responses")
        resp = self._responses[self._idx]
        self._idx += 1
        if self._schema:
            content_raw = resp[0].content
            content = content_raw if isinstance(content_raw, str) else str(content_raw)
            steps = [
                line.split('. ', 1)[1] if '. ' in line else line
                for line in content.splitlines()
                if line
            ]
            schema = self._schema
            self._schema = None
            return schema(goal="", steps=steps)
        return FakeInvocation(resp)

    def stream(self, *args: Any, **kwargs: Any) -> Tuple[BaseModel | FakeInvocation, Dict[str, Any]]:
        res = self.invoke(*args, **kwargs)
        return (res, {})
