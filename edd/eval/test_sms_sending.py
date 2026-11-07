"""Natural ordinary-thread SMS staging; synthetic stores and zero live effects."""
import json
import tempfile
from pathlib import Path
from unittest import TestCase, mock

from assist.model_manager import select_assistant_model
from assist.thread_manager import ThreadManager
from edd.prompt_census import SyntheticSandbox, _web_config, _web_toolset
from .test_async_subagents import _TOOLS
from .utils import stub_research_subagent


class TestOrdinarySmsSending(TestCase):
    def test_literal_number_stages_exact_pending_message(self):
        self._stage('Send a text to +15555550100 saying "hey now".', "hey now")

    def test_trusted_contact_stages_exact_pending_message(self):
        self._stage('Send Alex a text saying "running ten minutes late".',
                    "running ten minutes late", contact=True)

    def _stage(self, prompt, expected_text, contact=False):
        with tempfile.TemporaryDirectory(prefix='ordinary-sms-eval-') as temporary:
            root=Path(temporary);sandbox_root=root/'sandbox'
            (sandbox_root/'workspace').mkdir(parents=True)
            (sandbox_root/'workspace'/'AGENTS.md').write_text('Synthetic personal workspace.\n')
            if contact:
                (sandbox_root/'user').mkdir()
                (sandbox_root/'user'/'contacts.org').write_text('* Alex\nPhone: +15555550100\n')
            (sandbox_root/'agent').mkdir()
            (sandbox_root/'agent'/'memory.md').write_text('')
            sandbox=SyntheticSandbox(str(sandbox_root))
            if contact:
                self.assertFalse(sandbox.read('/user/contacts.org').error)
            stores=root/'threads'
            manager=ThreadManager(str(root/'threads'))
            manager._model=select_assistant_model(0.1)
            tid=manager.reserve(thread_id='synthetic-sms-thread')
            try:
                tools=_web_toolset(stores,full=True)
                with mock.patch.dict('os.environ',{'ASSIST_SMS_ARCHIVE_DIR':str(root/'archive'),'ASSIST_THREADS_DIR':str(root/'threads')}), _web_config(tools), mock.patch('assist.thread_manager.async_task_tools',_TOOLS), stub_research_subagent(), mock.patch('requests.post',side_effect=AssertionError('No SMS/network effect permitted in staging eval')) as send:
                    thread=manager.get(tid,sandbox_backend=sandbox)
                    result=thread.message(prompt)
                    output=str(result)
                    calls=[{'name':call['name'],'args':call.get('args')}
                           for message in thread.agent.get_state(thread.runconfig).values.get('messages',[])
                           for call in (getattr(message,'tool_calls',None) or [])]
                    # The proposed SMS is private server state, not a sent archive row.
                    try:
                        from assist.events.sms_outbox import SmsOutbox
                        pending=SmsOutbox(str(stores)).for_thread(tid)
                    except ModuleNotFoundError:
                        pending=[]
                    print(json.dumps({'profile':'ordinary web-main with guidance','output':output,'pending_count':len(pending),'transport_calls':send.call_count,'sandbox_commands':sandbox.commands,'tool_calls':calls}))
                    self.assertEqual(send.call_count,0)
                    self.assertEqual(len(pending),1,'Ordinary request must stage a durable pending SMS')
                    self.assertEqual(pending[0].number,'+15555550100')
                    self.assertEqual(pending[0].text,expected_text)
                    self.assertEqual(pending[0].state,'pending')
                    self.assertRegex(output.lower(),r'approval|confirm|propos|review')
            finally:
                manager.close()
