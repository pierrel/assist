import json,tempfile
from pathlib import Path
from unittest import mock
from assist.model_manager import select_assistant_model
from assist.thread_manager import ThreadManager
from assist.events.sms_outbox import SmsOutbox
from edd.prompt_census import SyntheticSandbox,_web_config,_web_toolset
from edd.eval.test_async_subagents import _TOOLS
from edd.eval.utils import stub_research_subagent

def test_explicit_user_retry_is_new_pending_attempt_not_automatic_send():
 _exercise_retry(True)

def test_unknown_outcome_without_new_request_stops_without_another_attempt():
 _exercise_retry(False)

def _exercise_retry(explicit_retry):
 with tempfile.TemporaryDirectory(prefix='sms-explicit-retry-') as temporary:
  root=Path(temporary);sandbox_root=root/'sandbox'
  (sandbox_root/'workspace').mkdir(parents=True)
  (sandbox_root/'workspace'/'AGENTS.md').write_text('Synthetic personal workspace.\n')
  (sandbox_root/'agent').mkdir();(sandbox_root/'agent'/'memory.md').write_text('')
  sandbox=SyntheticSandbox(str(sandbox_root));stores=root/'threads';manager=ThreadManager(str(stores));manager._model=select_assistant_model(0.1)
  tid=manager.reserve(thread_id='synthetic-retry-thread')
  try:
   with mock.patch.dict('os.environ',{'ASSIST_SMS_ARCHIVE_DIR':str(root/'archive'),'ASSIST_THREADS_DIR':str(stores)}),_web_config(_web_toolset(stores,full=True)),mock.patch('assist.thread_manager.async_task_tools',_TOOLS),stub_research_subagent(),mock.patch('requests.post',side_effect=AssertionError('No SMS/network effects')) as transport:
    thread=manager.get(tid,sandbox_backend=sandbox)
    prompts=['Send a text to +15555550100 saying "hey now".',('That came back with an unknown send result. Please try sending the same text again.' if explicit_retry else 'My phone reports that the send outcome is unknown.')]
    first=thread.message(prompts[0]);box=SmsOutbox(str(stores));records=box.for_thread(tid);assert len(records)==1
    box.decide(tid,records[0].id,'approved');box.decide(tid,records[0].id,'unknown')
    second=thread.message(prompts[1]);records=box.for_thread(tid)
    print(json.dumps({'prompts':prompts,'responses':[str(first),str(second)],'states':[r.state for r in records],'transport_calls':transport.call_count}))
    assert transport.call_count==0
    assert len(records)==(2 if explicit_retry else 1)
    assert [r.state for r in records]==(['unknown','pending'] if explicit_retry else ['unknown'])
    if explicit_retry:
     assert records[0].id!=records[1].id
    else:
     assert 'unknown' in str(second).lower() or 'uncertain' in str(second).lower()
    assert all((r.number,r.text)==('+15555550100','hey now') for r in records)
  finally:manager.close()
