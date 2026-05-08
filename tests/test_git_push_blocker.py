"""Unit tests for ``GitPushBlockerMiddleware`` — both the helper that
classifies a command and the middleware's wrap_tool_call dispatch.
"""
import unittest
from types import SimpleNamespace

from assist.middleware.git_push_blocker import (
    GitPushBlockerMiddleware,
    _command_pushes,
)


class TestCommandPushClassifier(unittest.TestCase):
    """Pure-function tests for the tokenised command classifier."""

    def assertPushes(self, command: str) -> None:
        self.assertTrue(
            _command_pushes(command),
            f"expected push to be detected: {command!r}",
        )

    def assertDoesNotPush(self, command: str) -> None:
        self.assertFalse(
            _command_pushes(command),
            f"expected no push detection: {command!r}",
        )

    def test_blocks_bare_push(self):
        self.assertPushes("git push")

    def test_blocks_push_origin(self):
        self.assertPushes("git push origin")

    def test_blocks_push_with_force(self):
        self.assertPushes("git push -f")
        self.assertPushes("git push --force")
        self.assertPushes("git push --force-with-lease")

    def test_blocks_push_with_refspec(self):
        self.assertPushes("git push origin HEAD:main")
        self.assertPushes("git push origin feature/x:main")

    def test_blocks_push_with_C_flag(self):
        self.assertPushes("git -C /workspace push")
        self.assertPushes("git -C /workspace push origin main")

    def test_blocks_push_with_long_options_before_subcommand(self):
        self.assertPushes("git --git-dir=/repo/.git push origin")
        self.assertPushes("git --no-pager push")

    def test_blocks_push_after_chain(self):
        # Shell operators tokenise to their own tokens; the `git push`
        # adjacency still surfaces.
        self.assertPushes("cd /tmp && git push")
        self.assertPushes("git status; git push")
        self.assertPushes("ls | grep foo && git push origin")

    def test_allows_other_git_commands(self):
        self.assertDoesNotPush("git status")
        self.assertDoesNotPush("git fetch")
        self.assertDoesNotPush("git fetch origin")
        self.assertDoesNotPush("git pull")
        self.assertDoesNotPush("git diff main..feature")
        self.assertDoesNotPush("git log --oneline")
        self.assertDoesNotPush("git rebase origin/main")

    def test_allows_non_git_commands(self):
        self.assertDoesNotPush("echo push")
        self.assertDoesNotPush("rm -rf /")
        self.assertDoesNotPush("python -c 'print(\"git push not real\")'")

    def test_allows_pushd_command(self):
        # `pushd` starts with "push" but is not preceded by `git`.
        self.assertDoesNotPush("pushd /tmp && ls")

    def test_allows_remote_named_push(self):
        # `git fetch push` would be valid if a remote were named "push" —
        # the subcommand here is `fetch`, not `push`.
        self.assertDoesNotPush("git fetch push")

    def test_handles_malformed_quoting_pessimistically(self):
        # A quote-mismatched string falls back to whitespace split; we
        # would rather false-positive than miss a real push attempt.
        self.assertPushes("git push 'unclosed")

    # --- Recursion into shell-out forms -------------------------------------

    def test_blocks_bash_dash_c_push(self):
        self.assertPushes('bash -c "git push"')
        self.assertPushes("bash -c 'git push origin main'")

    def test_blocks_sh_dash_c_push(self):
        self.assertPushes('sh -c "git push"')
        self.assertPushes('zsh -c "git push --force"')

    def test_blocks_shell_dash_lc_push(self):
        # ``-lc`` is bash's "login + command" cluster; the c-flag still
        # consumes the next arg as a command string.
        self.assertPushes('bash -lc "git push"')
        self.assertPushes('sh -ic "git push"')

    def test_blocks_eval_push(self):
        self.assertPushes('eval "git push"')
        self.assertPushes("eval 'git push origin'")

    def test_blocks_chain_inside_shell_out(self):
        # The recursion re-tokenises the inner string, so chain
        # operators inside the quoted command work the same way.
        self.assertPushes('bash -c "echo hi && git push"')
        self.assertPushes('sh -c "git fetch; git push"')

    def test_allows_shell_out_without_push(self):
        self.assertDoesNotPush('bash -c "git status"')
        self.assertDoesNotPush('sh -c "ls /tmp"')
        self.assertDoesNotPush('eval "git fetch"')

    def test_allows_non_command_shell_invocation(self):
        # ``bash myscript.sh`` doesn't have a -c flag; the script
        # contents aren't inspected (out of scope for v1).  This
        # is the documented limitation.
        self.assertDoesNotPush("bash myscript.sh")
        self.assertDoesNotPush("bash --version")


class TestMiddlewareDispatch(unittest.TestCase):
    """``wrap_tool_call`` must reject ``execute`` calls whose command
    pushes, and otherwise delegate to the handler.
    """

    def setUp(self):
        self.mw = GitPushBlockerMiddleware()

    def _request(self, name: str, command: str = "") -> SimpleNamespace:
        return SimpleNamespace(
            tool_call={
                "name": name,
                "args": {"command": command} if command else {},
                "id": "test-id",
            }
        )

    def test_rejects_execute_with_push(self):
        sentinel = object()

        def handler(_):
            return sentinel

        result = self.mw.wrap_tool_call(self._request("execute", "git push origin"), handler)
        self.assertIsNot(result, sentinel)
        self.assertEqual(result.status, "error")
        self.assertIn("not allowed", result.content)

    def test_passes_through_execute_without_push(self):
        sentinel = object()

        def handler(_):
            return sentinel

        result = self.mw.wrap_tool_call(self._request("execute", "git status"), handler)
        self.assertIs(result, sentinel)

    def test_passes_through_non_execute_tools(self):
        sentinel = object()

        def handler(_):
            return sentinel

        # write_file with command-shaped content shouldn't be inspected.
        result = self.mw.wrap_tool_call(self._request("write_file", "git push"), handler)
        self.assertIs(result, sentinel)

    def test_passes_through_when_args_missing(self):
        sentinel = object()

        def handler(_):
            return sentinel

        result = self.mw.wrap_tool_call(self._request("execute"), handler)
        self.assertIs(result, sentinel)


if __name__ == '__main__':
    unittest.main()
