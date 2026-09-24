"""Natural web-main browser journeys against a deterministic JavaScript site.

Only the site/browser boundary is faked. The real main graph chooses whether
to load its skill, try static navigation, and use its stateful browser tools.
Set ASSIST_BROWSER_EVAL_BASELINE=1 to remove browser tools for a comparison.
"""
import os
import signal
import tempfile
from dataclasses import replace
from unittest import TestCase, mock

from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends.protocol import ExecuteResponse, SandboxBackendProtocol

from assist.agent import AgentHarness, create_agent
from assist.browser.manager import browser_tools
from assist.checkpoint_rollback import invoke_with_rollback
from assist.model_manager import select_assistant_model
from assist.thread_manager import web_main_skill_sources

from .utils import (agent_tool_calls, create_filesystem,
                    prompt_rewrite_web_main_spec, stub_research_subagent)


class _StaticShell:
    text = "<html><main>This page requires JavaScript to display its content.</main></html>"
    status_code = 200

    def raise_for_status(self):
        pass


class _OfflineBackend(FilesystemBackend, SandboxBackendProtocol):
    def __init__(self, root):
        super().__init__(root_dir=root, virtual_mode=False)
        self.work_dir = root

    def execute(self, command, timeout=None):
        return ExecuteResponse(output="This fixture has no shell network access.", exit_code=1)


class _BrowserSite:
    def __init__(self, scenario):
        self.scenario = scenario
        self.calls = []

    def _page(self, snapshot, targets=(), downloads=(), snapshot_id="first"):
        return {"result": {"page_id": "page-1", "snapshot_id": snapshot_id,
                "url": f"https://{self.scenario}.fern.example/",
                "snapshot": snapshot, "targets": list(targets),
                "pages": [{"page_id": "page-1",
                           "url": f"https://{self.scenario}.fern.example/"}],
                "downloads": list(downloads), "network_errors": []}}

    def command(self, operation, **args):
        self.calls.append((operation, args))
        if operation == "open":
            if self.scenario == "status":
                return self._page("Service status: Green. No active incidents.")
            if self.scenario == "reports":
                return self._page("Fern reports. Choose a section.",
                                  [{"ref": "reports-button", "role": "button",
                                    "name": "Reports"}])
            return self._page("Fern product support.",
                              [{"ref": "manual-button", "role": "button",
                                "name": "Download manual"}])
        if operation == "act":
            target = args.get("target") or {}
            report_target = (target.get("ref") == "reports-button" or
                             (target.get("role"), target.get("name")) ==
                             ("button", "Reports"))
            manual_target = (target.get("ref") == "manual-button" or
                             (target.get("role"), target.get("name")) ==
                             ("button", "Download manual"))
            if self.scenario == "reports" and report_target:
                return self._page("Latest report: October Reliability Review.",
                                  snapshot_id="second")
            if self.scenario == "manuals" and manual_target:
                return self._page("Manual download started.",
                                  downloads=[{"download_id": "manual-download",
                                              "name": "Fern-Manual.pdf"}],
                                  snapshot_id="second")
        if operation == "observe":
            return self._page("This page requires interaction.")
        return {"error": "unsupported fixture action"}

    def save_download(self, download_id, filename=None):
        self.calls.append(("save_download", {"download_id": download_id,
                                             "filename": filename}))
        if download_id != "manual-download":
            raise ValueError("unknown download")
        return {"path": "/workspace/downloads/Fern-Manual.pdf", "size": 1280}


class TestBrowserAgent(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = select_assistant_model(0.1)

    def _run(self, scenario, prompt):
        with tempfile.TemporaryDirectory(prefix="browser_agent_eval_") as root:
            create_filesystem(root, {"README.org": "Personal workspace."})
            site = _BrowserSite(scenario)
            baseline = os.getenv("ASSIST_BROWSER_EVAL_BASELINE") == "1"
            spec = replace(
                prompt_rewrite_web_main_spec(
                    tools=() if baseline else tuple(browser_tools(site))),
                web_main=True, main_guidance_skills=True,
                skill_sources=web_main_skill_sources(browser=not baseline))
            with mock.patch("assist.tools.requests.get", return_value=_StaticShell()), \
                 stub_research_subagent():
                agent = AgentHarness(create_agent(
                    self.model, root, sandbox_backend=_OfflineBackend(root), spec=spec))
                previous = signal.signal(
                    signal.SIGALRM,
                    lambda _signum, _frame: (_ for _ in ()).throw(
                        TimeoutError("browser eval exceeded 150 seconds")))
                signal.alarm(150)
                try:
                    result = invoke_with_rollback(
                        agent.agent, {"messages": [{"role": "user", "content": prompt}]},
                        {"configurable": {"thread_id": agent.thread_id},
                         "recursion_limit": 120})
                except Exception as error:
                    raise AssertionError(
                        f"browser journey stopped ({type(error).__name__}: {error}); "
                        f"browser calls={site.calls[:20]}; "
                        f"graph calls={agent_tool_calls(agent)[:20]}") from error
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, previous)
                answer = result["messages"][-1].content
            return agent, site, str(answer)

    def test_reports_rendered_service_status(self):
        agent, site, answer = self._run(
            "status", "What is the current service status on "
            "https://status.fern.example/?")
        self.assertTrue(agent_tool_calls(agent, "browser_open"), agent.all_messages())
        self.assertIn("green", answer.lower())
        self.assertFalse(agent_tool_calls(agent, "task"), agent.all_messages())

    def test_opens_reports_menu_for_latest_title(self):
        agent, site, answer = self._run(
            "reports", "What's the latest report called on "
            "https://reports.fern.example/?")
        self.assertTrue(agent_tool_calls(agent, "browser_open"), agent.all_messages())
        self.assertTrue(agent_tool_calls(agent, "browser_act"), agent.all_messages())
        self.assertIn("October Reliability Review", answer)
        self.assertFalse(agent_tool_calls(agent, "task"), agent.all_messages())

    def test_saves_manual_from_interactive_support_page(self):
        agent, site, answer = self._run(
            "manuals", "Download the manual from https://manuals.fern.example/ "
            "and tell me where you saved it.")
        self.assertTrue(agent_tool_calls(agent, "browser_open"), agent.all_messages())
        self.assertTrue(agent_tool_calls(agent, "browser_act"), agent.all_messages())
        self.assertTrue(agent_tool_calls(agent, "browser_save_download"),
                        agent.all_messages())
        self.assertIn("/workspace/downloads/Fern-Manual.pdf", answer)
        self.assertFalse(agent_tool_calls(agent, "task"), agent.all_messages())
