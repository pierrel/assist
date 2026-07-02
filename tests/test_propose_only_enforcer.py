"""Unit tests for ``ProposeOnlyEnforcerMiddleware`` — the message-triage gate.

This is the GATE test, not an eval: it proves by construction that a triage turn can only
propose/read, so an untrusted inbound text cannot cause an effect regardless of what the
small model tries to call. Deny-by-default → every effectful tool (incl. ones not named in
the middleware) is rejected.
"""
import unittest
from types import SimpleNamespace

from assist.middleware.propose_only_enforcer import ProposeOnlyEnforcerMiddleware


class TestProposeOnlyEnforcer(unittest.TestCase):
    def setUp(self):
        self.mw = ProposeOnlyEnforcerMiddleware()

    def _request(self, name: str) -> SimpleNamespace:
        return SimpleNamespace(tool_call={"name": name, "args": {}, "id": "test-id"})

    def _rejected(self, name: str) -> None:
        called = {"handler": False}

        def handler(_):
            called["handler"] = True
            return object()

        result = self.mw.wrap_tool_call(self._request(name), handler)
        self.assertFalse(called["handler"], f"{name} reached the handler (should be blocked)")
        self.assertEqual(result.status, "error", f"{name} was not rejected")
        self.assertIn(name, result.content)

    def _allowed(self, name: str) -> None:
        sentinel = object()
        result = self.mw.wrap_tool_call(self._request(name), lambda _: sentinel)
        self.assertIs(result, sentinel, f"{name} should pass through to the handler")

    def test_allows_propose_reply_and_readonly(self):
        for name in ("propose_reply", "read_file", "ls", "glob", "grep"):
            self._allowed(name)

    def test_rejects_every_effectful_tool(self):
        # The force-installed deepagents built-ins + assist tools an injected text might try.
        for name in (
            "write_file", "edit_file", "execute", "task", "write_todos",
            "search_internet", "read_url", "propose_action",
        ):
            self._rejected(name)

    def test_rejects_unknown_future_tool_by_default(self):
        # Deny-by-default: a tool the middleware has never heard of is still blocked.
        self._rejected("some_new_tool_added_later")

    def test_rejects_empty_tool_name(self):
        self._rejected("")


if __name__ == "__main__":
    unittest.main()
