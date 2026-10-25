"""Natural SMS-history acceptance eval with synthetic private archive fixtures."""
import shutil
import tempfile
from pathlib import Path
from unittest import TestCase, mock
from functools import wraps

from assist.agent import AgentHarness, create_agent
from assist.model_manager import select_assistant_model
from assist.events.sms_archive import SmsArchive
from assist.events.sms_tools import sms_archive_tools
from .utils import prompt_rewrite_web_main_spec, skill_was_loaded, stub_research_subagent


class TestSmsHistorySkill(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = select_assistant_model(0.1)

    def test_find_and_read_shop_pickup_text(self):
        root = Path(tempfile.mkdtemp(prefix='sms_history_eval_'))
        try:
            archive = SmsArchive(root / 'private', thread_root=root / 'thread')
            archive.record('shop_receipt', 'incoming', '+15555550100',
                           'Your bicycle will be ready at 16:37. Collection code LARK72.', 'received')
            archive.record('other_receipt', 'incoming', '+15555550101',
                           'Lunch is at noon.', 'received')
            observed = []
            tools = []
            for function in sms_archive_tools():
                def tracked(function):
                    @wraps(function)
                    def call(*args, **kwargs):
                        observed.append((function.__name__, kwargs))
                        return function(*args, **kwargs)
                    return call
                tools.append(tracked(function))
            with mock.patch.dict('os.environ', {'ASSIST_SMS_ARCHIVE_DIR': str(root / 'private'), 'ASSIST_THREADS_DIR': str(root / 'thread')}), stub_research_subagent():
                harness = AgentHarness(create_agent(self.model, str(root / 'thread'),
                    spec=prompt_rewrite_web_main_spec(tools=tools)))
                output = harness.message('What pickup time did the bicycle shop text me, and what collection code do I need? Their number is +15555550100.')
            self.assertTrue(skill_was_loaded(harness, 'sms-history'))
            self.assertTrue(any(name == 'search_sms_messages' for name, _ in observed))
            self.assertTrue(any(name == 'read_sms_message' and args.get('message_id') == 'shop_receipt' for name, args in observed))
            self.assertIn('16:37', str(output))
            self.assertIn('LARK72', str(output))
        finally:
            shutil.rmtree(root, ignore_errors=True)
