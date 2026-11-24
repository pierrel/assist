"""Natural model-choice coverage for requested personal browsing.

The static site and browser are offline fixtures. Real host link publication,
checkpoint retention and Return are covered separately by test_browser_handoff;
this eval measures whether the model discovers and requests that capability.
"""
import tempfile
from dataclasses import replace
from unittest import TestCase

from assist.agent import AgentHarness, create_agent
from assist.browser.manager import browser_tools
from assist.model_manager import select_assistant_model
from assist.thread_manager import web_main_skill_sources
from .test_browser_agent import _BrowserSite, _OfflineBackend, _browser_task_fixture, _offline_url_fetch
from .utils import (create_filesystem, prompt_rewrite_web_main_spec,
                    stub_research_subagent)


class _PersonalBrowser(_BrowserSite):
    def __init__(self, root):
        super().__init__('static', root)
        self.handed_off = False
        self.control_url = 'https://assist.fern.example/thread/exhibit/browser'

    def command(self, operation, **args):
        if operation == 'close':
            self.calls.append((operation, args))
            self.session_open = False
            return {'result': {'pages': []}}
        if self.handed_off:
            self.calls.append((operation, args))
            return {'error': 'Browser control is reserved for the user.'}
        result = super().command(operation, **args)
        if operation == 'open' and 'error' not in result:
            self.session_open = True
        return result

    def offer_control(self):
        self.calls.append(('handoff', {}))
        if not self.session_open:
            return {'error': 'No live website page is available.'}
        self.handed_off = True
        return {'handoff_url': self.control_url, 'status': 'waiting_for_user'}


class TestBrowserRequestedControl(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = select_assistant_model(0.1)

    def test_personal_browsing_opens_and_hands_over_a_static_page(self):
        with tempfile.TemporaryDirectory(prefix='browser_personal_eval_') as root:
            create_filesystem(root, {'README.org': 'Personal workspace.'})
            site = _PersonalBrowser(root)
            spec = replace(prompt_rewrite_web_main_spec(tools=tuple(browser_tools(site))),
                           web_main=True, main_guidance_skills=True,
                           default_backend=_OfflineBackend(root),
                           skill_sources=web_main_skill_sources(browser=True))
            with _offline_url_fetch('static'), \
                 stub_research_subagent(), _browser_task_fixture():
                agent = AgentHarness(create_agent(self.model, root, spec=spec))
                agent.message('Open https://static.fern.example/ so I can look around myself.')
            self.assertTrue(site.handed_off, (site.calls, agent.all_messages()))
            self.assertTrue(site.session_open, site.calls)
            self.assertFalse(any(operation == 'close' for operation, _ in site.calls), site.calls)
            self.assertEqual(sum(operation == 'handoff' for operation, _ in site.calls), 1)
            self.assertTrue(any(operation == 'open' for operation, _ in site.calls), site.calls)
