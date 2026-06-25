import ast
import functools
import os
import shutil
import subprocess
from unittest.mock import patch
from unittest import TestCase
from langchain_core.messages import ToolMessage, AIMessage


class AgentTestMixin:
    """
    Mixin for TestCase classes that adds agent-specific assertions.

    Usage:
        class MyTest(AgentTestMixin, TestCase):
            def test_something(self):
                agent, root = self.create_agent({...})
                agent.message("Write a file")
                self.assertToolCall(agent, "write_file", "Should have written")
    """

    def assertToolCall(self, agent, tool_name: str, msg: str = None):
        """
        Assert that a specific tool was called by the agent.

        Args:
            agent: The AgentHarness instance
            tool_name: The name of the tool to check for
            msg: Optional custom assertion message
        """
        tool_calls = [m.name for m in agent.all_messages() if isinstance(m, ToolMessage)]

        if msg is None:
            msg = f"Tool '{tool_name}' should have been called. Called tools: {tool_calls}"

        self.assertIn(tool_name, tool_calls, msg)

    def subagent_calls(self, agent) -> list[str]:
        """Names of every subagent dispatched via the `task` tool, in order.

        Reads AIMessage tool_calls (outgoing) rather than ToolMessages
        (results), so a dispatch counts even if the subagent errored.
        Returns a list (not a set) so callers can assert on *how many*
        times a subagent was dispatched, not just whether it was.
        """
        calls = []
        for m in agent.all_messages():
            if isinstance(m, AIMessage) and m.tool_calls:
                for tc in m.tool_calls:
                    if tc.get("name") == "task":
                        # deepagents' task tool names the target via
                        # `subagent_type`, but the small model sometimes
                        # emits it under `agent`/`name` instead — the
                        # dev-agent evals (test_dev_agent.py:167,
                        # test_dev_agent_planning_flow.py:157) carry the
                        # same fallback against that observed shape, so
                        # match it here for consistent counting.  The
                        # `or` chain also recovers an empty `subagent_type`
                        # (which SubagentTypeInferenceMiddleware would
                        # otherwise default to general-purpose).
                        args = tc.get("args") or {}
                        sa = (args.get("subagent_type")
                              or args.get("agent")
                              or args.get("name") or "")
                        if sa:
                            calls.append(sa)
        return calls

    def assertSubAgentCall(self, agent, subagent_name: str, msg: str = None):
        """
        Assert that a specific subagent was called by the agent via the task tool.

        Checks AIMessage tool_calls (outgoing calls) rather than ToolMessages (results),
        so this passes even if the subagent itself errors out.

        Args:
            agent: The AgentHarness instance
            subagent_name: The subagent_type value to look for (e.g. "dev-agent")
            msg: Optional custom assertion message
        """
        calls = self.subagent_calls(agent)

        if msg is None:
            msg = f"Subagent '{subagent_name}' should have been called via task tool. Called subagents: {calls}"

        self.assertIn(subagent_name, calls, msg)


def assertToolCall(test_case, agent, tool_name: str, msg: str = None):
    """
    Assert that a specific tool was called by the agent.

    This function can be used directly or as a helper to add to TestCase classes.

    Args:
        test_case: The TestCase instance (pass self from the test)
        agent: The AgentHarness instance
        tool_name: The name of the tool to check for
        msg: Optional custom assertion message

    Usage (direct):
        from tests.integration.validation.utils import assertToolCall

        agent, root = self.create_agent({...})
        agent.message("Do something")
        assertToolCall(self, agent, "write_file", "Should have written a file")

    Usage (as method - add to TestCase setUp):
        from tests.integration.validation.utils import assertToolCall

        class MyTest(TestCase):
            def setUp(self):
                # Add as instance method
                self.assertToolCall = lambda agent, tool, msg=None: assertToolCall(self, agent, tool, msg)

            def test_something(self):
                agent, root = self.create_agent({...})
                agent.message("Write a file")
                self.assertToolCall(agent, "write_file", "Should have written")
    """
    tool_calls = [m.name for m in agent.all_messages() if isinstance(m, ToolMessage)]

    if msg is None:
        msg = f"Tool '{tool_name}' should have been called. Called tools: {tool_calls}"

    test_case.assertIn(tool_name, tool_calls, msg)


def read_file(path: str) -> str:
    """Returns the full contents of file at path"""
    with open(path, 'r') as f:
        return f.read()

def files_in_directory(path: str) -> list[str]:
    """Returns the files in path as a list"""
    return os.listdir(path)

def skill_was_loaded(agent, skill_name: str) -> bool:
    """True iff a tool call loaded the named skill's body.

    Recognizes both routes the SkillsMiddleware exposes:

    - ``load_skill(name=skill_name)`` — the small-model tool registered by
      ``SmallModelSkillsMiddleware``.
    - ``read_file`` / ``read`` with a path containing ``/skills/<name>/`` —
      the upstream deepagents path.

    Shared by the skill-loading evals; grep tool-call args rather than results,
    since the model proves intent the moment it issues the call.
    """
    path_needle = f"/skills/{skill_name}/"
    for m in agent.all_messages():
        if not isinstance(m, AIMessage) or not m.tool_calls:
            continue
        for tc in m.tool_calls:
            args = tc.get("args") or {}
            if tc.get("name") == "load_skill" and args.get("name") == skill_name:
                return True
            for v in args.values():
                if isinstance(v, str) and path_needle in v:
                    return True
    return False


def executed_commands(agent) -> list[str]:
    """Command strings from every ``execute`` tool call, in order."""
    cmds = []
    for m in agent.all_messages():
        if not isinstance(m, AIMessage) or not m.tool_calls:
            continue
        for tc in m.tool_calls:
            if tc.get("name") == "execute":
                cmd = (tc.get("args") or {}).get("command", "")
                if cmd:
                    cmds.append(cmd)
    return cmds


def cleanup_workspace(path: str) -> None:
    """Remove a sandbox workspace, using Docker to delete root-owned files.

    Sandbox commands (pip, pytest, emacs, ...) write files as root inside the
    bind mount; a plain ``shutil.rmtree`` fails on those without an
    intermediate chmod, so run one in a throwaway alpine container first.
    """
    try:
        subprocess.run(
            ['docker', 'run', '--rm', '-v', f'{path}:/cleanup', 'alpine',
             'sh', '-c', 'chmod -R 777 /cleanup 2>/dev/null; rm -rf /cleanup/*'],
            check=False, timeout=60,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass
    shutil.rmtree(path, ignore_errors=True)


def create_filesystem(root_dir: str,
                      structure: dict):
    """Creates a directory structure and files according to `structure`. For example:
    {"README.org": "This is the readme file",
    "gtd": {"inbox.org": "This is the inbox file"},
           {"projects": {"project1.org": "This is a project file"}}}

    Creates:
    a README.org file with content "This is the readme file"
    a gtd directory
    a gtd/inbox.org file with content "This is the inbox file"
    ..."""
    for name, content in structure.items():
        path = os.path.join(root_dir, name)

        if isinstance(content, str):
            # Create a file with the given content
            with open(path, 'w') as f:
                f.write(content)
        elif isinstance(content, dict):
            # Create a directory and recursively process its contents
            os.makedirs(path, exist_ok=True)
            create_filesystem(path, content)


# -------------------- research URL-provenance spy --------------------

# One source of truth for "the same URL" — the prod guard defines it, the eval
# imports it, so the spy's provenance accounting can't drift from the guard's.
from assist.middleware.url_provenance import normalize_url


def _urls_in_search_result(result_str: str) -> list[str]:
    """Extract URLs from a search_internet result string
    (``[{'title','url','content'}, ...]``). Returns [] for non-list results
    (e.g. the unavailable message or ``"[]"``)."""
    try:
        items = ast.literal_eval(result_str)
    except (ValueError, SyntaxError):
        return []
    if not isinstance(items, list):
        return []
    return [it["url"] for it in items
            if isinstance(it, dict) and it.get("url")]


class ResearchToolSpy:
    """Context manager that records every ``search_internet`` / ``read_url``
    call made anywhere in the research agent (it patches the names where
    ``assist.agent`` binds them, so nested sub-agent calls are captured too —
    ``all_messages()`` only exposes the top-level thread). Calls pass through to
    the real tools, so research runs for real.

    Optionally injects a fetch failure to test dead-fetch recovery:
      - ``fail_first=N`` — the first N *distinct* URLs fetched return a 404.
      - ``fail_urls={...}`` — these exact URLs return a 404.

    After the ``with`` block:
      ``fetched``        — list[str], every read_url url arg in call order
      ``searched``       — list[str], every search query
      ``search_results`` — set[str], normalized URLs returned by all searches
      ``failed_first``   — list[str] normalized URLs failed via ``fail_first``
    """

    def __init__(self, fail_first: int = 0, fail_urls=None,
                 search_fixture=None, read_fixture=None):
        """``search_fixture`` (list of ``{title,url,content}`` dicts) makes
        ``search_internet`` return that canned set for every query instead of
        hitting SearXNG — deterministic and immune to the upstream-engine
        rate-limits/CAPTCHAs that make live research evals flaky. In canned mode
        ``read_url`` also returns canned text (``read_fixture`` maps normalized
        url->text; otherwise a generic body), so the eval exercises the model's
        URL-choice behavior without any network."""
        self.fetched: list[str] = []
        self.searched: list[str] = []
        self.search_results: set[str] = set()
        self.failed_first: list[str] = []
        self._fail_first = fail_first
        self._fail = {normalize_url(u) for u in (fail_urls or [])}
        self._search_fixture = search_fixture
        self._read_fixture = {normalize_url(k): v
                              for k, v in (read_fixture or {}).items()}
        self._patches: list = []

    @staticmethod
    def _err(url: str) -> str:
        return f"Error fetching URL: 404 Client Error: Not Found for url: {url}"

    def __enter__(self):
        import assist.agent as ag
        real_read, real_search = ag.read_url, ag.search_internet

        # @wraps so deepagents/langchain wrap these as tools named "read_url" /
        # "search_internet" (it derives the tool name + description from
        # __name__/__doc__), not "spy_read"/"spy_search".
        @functools.wraps(real_read)
        def spy_read(url):
            self.fetched.append(url)
            n = normalize_url(url)
            if n in self._fail:
                return self._err(url)
            if (self._fail_first and n not in self.failed_first
                    and len(self.failed_first) < self._fail_first):
                self.failed_first.append(n)
                return self._err(url)
            if self._search_fixture is not None:  # canned mode: no network
                return self._read_fixture.get(n, f"Reference page at {url}.")
            return real_read(url)

        @functools.wraps(real_search)
        def spy_search(query, max_results=5):
            self.searched.append(query)
            if self._search_fixture is not None:
                res = str(self._search_fixture[:max_results])
            else:
                res = real_search(query, max_results)
            for u in _urls_in_search_result(res):
                self.search_results.add(normalize_url(u))
            return res

        self._patches = [patch.object(ag, "read_url", spy_read),
                         patch.object(ag, "search_internet", spy_search)]
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in self._patches:
            p.stop()
        return False

    def guessed_fetches(self) -> list[str]:
        """Fetched URLs that came from NO search result — i.e. the agent
        constructed/guessed them rather than picking a search hit."""
        return [u for u in self.fetched
                if normalize_url(u) not in self.search_results]

    def fetch_count(self, url: str) -> int:
        n = normalize_url(url)
        return sum(1 for u in self.fetched if normalize_url(u) == n)
