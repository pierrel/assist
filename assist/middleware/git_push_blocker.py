"""Git push blocker middleware.

Refuses ``execute`` tool calls that would run ``git push`` (in any
form) before they reach the shell.  The agent runs inside a sandbox
that has no SSH key or git credentials, so a real push attempt would
already fail at the credential layer — but a clean tool-result error
beats a garbled credential failure for two reasons:

1. The small model can reason about the message and stop retrying.
2. Defence in depth — if a future tool inadvertently hands credentials
   to the sandbox, the wrapper still says no.

Push to ``origin`` is exclusively user-initiated through the web UI's
"Push to origin" button.  Documented in
``docs/2026-05-07-per-thread-web-git-isolation.org``.
"""
import logging
import shlex
from typing import Any, Callable

from langchain.agents.middleware import AgentMiddleware
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command


logger = logging.getLogger(__name__)


_REJECTION_MESSAGE = (
    "Error: direct git push is not allowed.  The user controls pushes "
    "from the web UI.  To publish your work to origin, ask the user to "
    "click 'Push to origin' in their browser."
)


# Git top-level options that take a separate (non-``=``-joined) value;
# any other ``--name`` option is treated as a flag for the purpose of
# locating the subcommand.  Keeping the set small means the rare
# never-seen-in-practice option falls through as "flag" and the worst
# outcome is one extra step of tokenisation — we never miss a push.
_OPTIONS_WITH_VALUE: frozenset[str] = frozenset({
    "-C", "-c",
    "--git-dir", "--work-tree", "--namespace",
    "--exec-path", "--super-prefix",
})

# Shells that take a command string via a flag containing ``c`` —
# ``bash -c "git push"``, ``sh -lc "git push"``.  We re-tokenise the
# argument string and look for a push inside it.
_SHELL_C_FORMS: frozenset[str] = frozenset({
    "bash", "sh", "zsh", "ash", "ksh", "dash",
})


def _command_pushes(command: str) -> bool:
    """Return True iff ``command`` invokes ``git push`` in any form.

    Tokenises with :func:`shlex.split` (POSIX), then walks looking for
    a ``git`` token followed — possibly after ``git``-level option
    flags like ``-C <path>``, ``--no-pager``, or ``--git-dir=...`` —
    by ``push``.  Shell operators (``;``, ``&&``, ``|``) become their
    own tokens, so chained commands surface the same way.

    Recurses into nested shell-out forms so ``bash -c "git push"``,
    ``sh -lc "git push origin"``, and ``eval "git push"`` are all
    caught — the literal string after ``-c`` (or ``eval``) is
    re-tokenised and re-classified.

    Tolerant of malformed commands: a quote-mismatched string falls
    back to a whitespace split, which is intentionally pessimistic —
    we'd rather false-positive on a weird command than miss a real
    push attempt.
    """
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        tokens = command.split()

    i = 0
    while i < len(tokens):
        tok = tokens[i]

        # ``bash -c "<cmd>"`` / ``sh -lc "<cmd>"`` etc. — recurse
        # into the command-string argument.
        if tok in _SHELL_C_FORMS:
            j = i + 1
            while j < len(tokens) and tokens[j].startswith("-"):
                # Any flag whose short cluster contains 'c' (or the
                # long-form ``--command``) takes the next token as
                # the command string.
                if "c" in tokens[j] and j + 1 < len(tokens):
                    if _command_pushes(tokens[j + 1]):
                        return True
                j += 1
            i += 1
            continue

        # ``eval "<cmd>"`` — the next non-flag token is a command
        # string.
        if tok == "eval":
            j = i + 1
            while j < len(tokens):
                if not tokens[j].startswith("-"):
                    if _command_pushes(tokens[j]):
                        return True
                    break
                j += 1
            i += 1
            continue

        if tok != "git":
            i += 1
            continue

        # Walk past git-level options after `git` to find the subcommand.
        j = i + 1
        while j < len(tokens):
            t = tokens[j]
            # ``--name=value`` form: one token, skip it.
            if t.startswith("--") and "=" in t:
                j += 1
                continue
            # Options that take a separate value: skip option + value.
            if t in _OPTIONS_WITH_VALUE:
                j += 2
                continue
            # Any other flag (short or long, e.g. ``--no-pager``,
            # ``-p``): one token.
            if t.startswith("-"):
                j += 1
                continue
            # First non-flag token after `git` is the subcommand.
            break

        if j < len(tokens) and tokens[j] == "push":
            return True
        i = j + 1 if j > i else i + 1
    return False


class GitPushBlockerMiddleware(AgentMiddleware):
    """Reject ``execute`` tool calls that would invoke ``git push``."""

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        tool_call = request.tool_call
        if tool_call.get("name", "") != "execute":
            return handler(request)

        args: Any = tool_call.get("args") or tool_call.get("arguments") or {}
        command = args.get("command", "") if isinstance(args, dict) else ""

        if command and _command_pushes(command):
            logger.warning("GitPushBlocker rejected execute call: %s", command)
            return ToolMessage(
                content=_REJECTION_MESSAGE,
                tool_call_id=tool_call.get("id", ""),
                name="execute",
                status="error",
            )
        return handler(request)
