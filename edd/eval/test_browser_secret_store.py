"""Natural browser login and private-entry requests in an offline repository."""

import json
import os
import re
import signal
import tempfile
from dataclasses import replace
from pathlib import Path
from unittest import TestCase, mock

from assist.agent import AgentHarness, create_agent
from assist.browser.manager import browser_tools
from assist.checkpoint_rollback import invoke_with_rollback
from assist.git_sync import _initial_binding, _write_state
from assist.model_manager import select_assistant_model
from assist.secret_store import SecretStore
from assist.thread_manager import web_main_skill_sources
from manage.web import state

from .test_browser_agent import _BrowserSite, _OfflineBackend, _StaticShell, _browser_task_fixture
from .utils import agent_tool_calls, create_filesystem, prompt_rewrite_web_main_spec, stub_research_subagent


LOGIN = "https://accounts.fern.example/login"
USERNAME = "synthetic-user"
PASSWORD = "synthetic-river-8642"
PHRASE = "synthetic unlock phrase only"
NEW_TOOLS = {"browser_list_secrets", "browser_request_secret", "browser_fill_secret"}


class _LoginSite:
    def __init__(self, repo: Path, threads_root: Path):
        self.work_dir = str(repo)
        self.threads_root = str(threads_root)
        self.thread_id = "thread"
        self.calls = []
        self.filled = False
        self.signed_in = False
        self.opened = False
        self.last_snapshot = None
        self.last_targets = set()

    def _page(self):
        if self.signed_in:
            page = {"result": {
                "page_id": "page-1", "snapshot_id": "dashboard",
                "url": "https://accounts.fern.example/dashboard?user=" + USERNAME,
                "snapshot": "Dashboard heading: Account overview. User: " + USERNAME,
                "targets": [], "pages": [], "downloads": [], "network_errors": []}}
        else:
            page = {"result": {
                "page_id": "page-1", "snapshot_id": "filled" if self.filled else "login",
                "url": LOGIN,
                "snapshot": ("Username: " + USERNAME + "; Password: " + PASSWORD
                             if self.filled else "Sign in to view your dashboard."),
                "targets": [
                    {"ref": "username", "role": "textbox", "name": "Username"},
                    {"ref": "password", "role": "textbox", "name": "Password"},
                    {"ref": "submit", "role": "button", "name": "Sign in"}],
                "pages": [], "downloads": [], "network_errors": []}}
        self.last_snapshot = page["result"]["snapshot_id"]
        self.last_targets = {target["ref"] for target in page["result"]["targets"]}
        return page

    def command(self, operation, **args):
        self.calls.append((operation, args))
        if operation == "open":
            if args.get("url") != LOGIN:
                return {"error": "page unavailable"}
            self.opened = True
            return self._page()
        if operation in {"observe", "wait"}:
            if not self.opened or args.get("page_id") != "page-1":
                return {"error": "page unavailable"}
            return self._page()
        if operation == "fill_secret":
            if (args.get("page_id") != "page-1"
                    or args.get("snapshot_id") != self.last_snapshot
                    or self.last_snapshot != "login"
                    or {"username", "password"} - self.last_targets
                    or args.get("username_ref") != "username"
                    or args.get("login_page") != LOGIN or args.get("username") != USERNAME
                    or args.get("password") != PASSWORD
                    or args.get("password_ref") != "password"):
                return {"error": "credential field changed"}
            self.filled = True
            self.last_snapshot = None
            self.last_targets.clear()
            return {"result": {"filled": True}}
        if operation == "act":
            if (args.get("action") == "click" and self.filled
                    and args.get("page_id") == "page-1"
                    and self.last_snapshot == "filled"
                    and args.get("snapshot_id") == self.last_snapshot
                    and "submit" in self.last_targets
                    and (args.get("target") or {}).get("ref") == "submit"):
                self.signed_in = True
                return self._page()
            return {"error": "action unavailable"}
        if operation == "preflight":
            return {"result": {"top_level": {"origin": "https://accounts.fern.example",
                                              "allowed": True, "reason": "allowed"},
                               "dependencies": [], "partial": False}}
        if operation == "close":
            self.opened = False
            self.filled = False
            self.last_snapshot = None
            self.last_targets.clear()
            return {"result": {"closed_page_id": "page-1", "pages": []}}
        return {"error": "operation unavailable"}


class TestLoginSiteContract(TestCase):
    def test_login_needs_observed_fields_and_fresh_submit(self):
        site = _LoginSite(Path("/tmp/synthetic-repo"), Path("/tmp"))
        fill = {
            "page_id": "page-1", "snapshot_id": "login", "login_page": LOGIN,
            "username": USERNAME, "password": PASSWORD,
            "username_ref": "username", "password_ref": "password"}
        submit = {"page_id": "page-1", "snapshot_id": "filled",
                  "action": "click", "target": {"ref": "submit"}}

        self.assertIn("error", site.command("observe", page_id="page-1"))
        self.assertIn("error", site.command("fill_secret", **fill))
        site.command("open", url=LOGIN)
        self.assertIn("error", site.command("observe", page_id="unknown"))
        self.assertIn("error", site.command(
            "fill_secret", **{**fill, "snapshot_id": "stale"}))
        self.assertIn("error", site.command(
            "fill_secret", **{**fill, "username_ref": "unknown"}))
        self.assertEqual(site.command("fill_secret", **fill),
                         {"result": {"filled": True}})
        self.assertIn("error", site.command("act", **submit))
        site.command("close", page_id="page-1")
        self.assertEqual(site.command("open", url=LOGIN)["result"]["snapshot_id"],
                         "login")
        self.assertIn("error", site.command("act", **submit))
        self.assertEqual(site.command("fill_secret", **fill),
                         {"result": {"filled": True}})
        site.command("observe", page_id="page-1")
        self.assertEqual(site.command("act", **submit)["result"]["snapshot_id"],
                         "dashboard")


class TestBrowserSecretStore(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = select_assistant_model(0.1)

    def _turn(self, prompt: str, *, stored: bool = False, status: bool = False):
        with tempfile.TemporaryDirectory(prefix="browser_secret_eval_") as temporary:
            root = Path(temporary)
            repo = root / "repo"
            repo.mkdir()
            create_filesystem(str(repo), {"README.org": "Synthetic local workspace."})
            thread = root / "thread"
            thread.mkdir()
            source = "ssh://synthetic.invalid/repository.git"
            _write_state(str(thread), _initial_binding(source))
            store = SecretStore(repo, thread)
            if stored:
                store.unlock(PHRASE, create=True)
                store.put("Fern account", LOGIN, USERNAME, PASSWORD)
            site = (_BrowserSite("status", str(repo)) if status
                    else _LoginSite(repo, root))
            if status:
                site.work_dir = str(repo)
                site.threads_root = str(root)
                site.thread_id = "thread"
            baseline = os.environ.get("ASSIST_SECRET_EVAL_BASELINE") == "1"
            tools = tuple(tool for tool in browser_tools(site)
                          if not baseline or tool.__name__ not in NEW_TOOLS)
            spec = replace(
                prompt_rewrite_web_main_spec(tools=tools),
                web_main=True, main_guidance_skills=True,
                skill_sources=web_main_skill_sources(browser=True))
            with mock.patch.object(state, "DOMAINS", [source]), \
                 mock.patch("assist.tools.requests.get",
                            return_value=_StaticShell("status" if status else "login")), \
                 mock.patch("assist.tools._query_engine", return_value=([], False)), \
                 mock.patch("requests.sessions.Session.request",
                            side_effect=AssertionError("synthetic fixture has no network")), \
                 _browser_task_fixture(), stub_research_subagent():
                agent = AgentHarness(create_agent(
                    self.model, str(repo), sandbox_backend=_OfflineBackend(str(repo)),
                    spec=spec))
                previous = signal.signal(
                    signal.SIGALRM,
                    lambda *_args: (_ for _ in ()).throw(
                        TimeoutError("natural secret browser eval exceeded 150 seconds")))
                signal.alarm(150)
                try:
                    result = invoke_with_rollback(
                        agent.agent, {"messages": [{"role": "user", "content": prompt}]},
                        {"configurable": {"thread_id": agent.thread_id},
                         "recursion_limit": 500})
                    answer = str(result["messages"][-1].content)
                    calls = agent_tool_calls(agent)
                    model_visible = "\n".join({
                        json.dumps({
                            "content": message.content,
                            "tool_calls": getattr(message, "tool_calls", None),
                        }, default=str)
                        for checkpoint in agent.agent.get_state_history({
                            "configurable": {"thread_id": agent.thread_id}})
                        for message in checkpoint.values.get("messages", [])
                    })
                    pending = store.pending_request()
                    metrics_path = os.environ.get("ASSIST_SECRET_EVAL_METRICS")
                    if metrics_path:
                        credential_question = bool(re.search(
                            r"(?:send|share|reply with|give|tell)\b.{0,160}"
                            r"(?:password|credentials)", answer, re.I | re.S))
                        Path(metrics_path).write_text(json.dumps({
                            "baseline": baseline,
                            "pending_request": pending is not None,
                            "signed_in": site.signed_in if not status else False,
                            "heading_in_answer": "Account overview" in answer,
                            "password_in_model_messages": PASSWORD in model_visible,
                            "username_in_model_messages": USERNAME in model_visible,
                            "chat_credential_question": credential_question,
                            "secret_tool_calls": [call["name"] for call in calls
                                                  if call.get("name") in NEW_TOOLS],
                        }, sort_keys=True))
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, previous)
            return baseline, answer, calls, model_visible, pending, site

    def test_natural_private_entry_request(self):
        baseline, answer, calls, visible, pending, site = self._turn(
            "Please set up a login for https://accounts.fern.example/login "
            "so you can sign in for me later.")
        self.assertNotIn(PASSWORD, visible)
        self.assertNotIn(USERNAME, visible)
        if baseline:
            self.assertIsNone(pending, (answer, calls))
        else:
            self.assertIsNotNone(pending, (answer, calls))
            self.assertEqual(pending["login_page"], LOGIN)
            self.assertTrue(agent_tool_calls_from(calls, "browser_request_secret"))

    def test_natural_login(self):
        baseline, answer, calls, visible, pending, site = self._turn(
            "Sign in to https://accounts.fern.example/login and tell me the "
            "heading on my dashboard.", stored=True)
        self.assertNotIn(PASSWORD, visible)
        self.assertNotIn(USERNAME, visible)
        if baseline:
            self.assertFalse(site.signed_in, (answer, calls))
        else:
            self.assertTrue(site.signed_in, (answer, calls))
            self.assertIn("Account overview", answer)
            self.assertTrue(agent_tool_calls_from(calls, "browser_fill_secret"))

    def test_adjacent_rendered_status(self):
        baseline, answer, calls, visible, pending, site = self._turn(
            "What is the current service status on https://status.fern.example/?",
            status=True)
        self.assertIn("green", answer.lower())
        self.assertTrue(agent_tool_calls_from(calls, "browser_open"))
        self.assertFalse(agent_tool_calls_from(calls, "browser_request_secret"))


def agent_tool_calls_from(calls, name):
    return [call for call in calls if call.get("name") == name]
