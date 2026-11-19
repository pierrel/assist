"""Repeatable native client capability check, with fake upstream and network:none.

Run from the project venv after building Dockerfile.hermes-runtime. Optionally set
ASSIST_HERMES_TEST_IMAGE and HERMES_PROOF_ARTIFACTS. No production proxy, model,
phone device or external Git remote is used. All launched containers are stopped
in finally, including when an outer timeout sends TERM.
"""
import http.server
import json
import logging
import os
from pathlib import Path
import sys
import signal
import shlex
import tempfile
import threading
import time

ROOT=Path(os.environ.get('HERMES_PROOF_ARTIFACTS') or tempfile.mkdtemp(prefix='hermes-check-'))
ROOT.mkdir(exist_ok=True)
for name in ('threads','runtime','maps','approvals'):(ROOT/name).mkdir(exist_ok=True,mode=0o700)
os.environ.update(ASSIST_THREADS_DIR=str(ROOT/'threads'),ASSIST_EGRESS_RUNTIME_DIR=str(ROOT/'runtime'),ASSIST_EGRESS_CLIENT_MAP_DIR=str(ROOT/'maps'),ASSIST_EGRESS_APPROVALS_DIR=str(ROOT/'approvals'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from manage.web import threads
from assist import sandbox_manager as sm
from assist.browser.manager import BrowserManager
from assist.hermes_runtime import HermesRuntimeManager
from assist.hermes_provider_relay import HermesProviderRelay
from assist.model_manager import OpenAIConfig
from assist.thread_queue import ThreadAffinityQueue

logging.getLogger('assist.thread_queue').setLevel(logging.INFO)
sm.HERMES_SANDBOX_IMAGE=os.environ.get('ASSIST_HERMES_TEST_IMAGE','assist-hermes-runtime')
sm.EGRESS_NETWORK='none'
# The actual sandbox creator, mounts, authority and verified cleanup run below.
# Egress setup is excluded: this test has no network and never touches the live proxy.
sm.SandboxManager._ensure_egress_proxy_running=classmethod(lambda cls,client:None)
sm.SandboxManager._record_egress_client=classmethod(lambda cls,*args,**kwargs:None)
BrowserManager.reconcile_startup=classmethod(lambda cls,root:True)
threads.THREAD_QUEUE=ThreadAffinityQueue(hold_timeout_s=120,wait_timeout_s=30,quantum_s=10)
threads._recovery_prep=lambda q:None
threads._RESUME_SCHEDULER=threads._ResumeScheduler()
threads.MANAGER.root_dir=str(ROOT/'threads')
for tid in ('hermes-proof-a','hermes-proof-b','hermes-proof-c'):
    threads.MANAGER.reserve_visible('hermes',thread_id=tid)
    Path(threads.MANAGER.thread_default_working_dir(tid)).mkdir(exist_ok=True)
A='hermes-proof-a';B='hermes-proof-b';C='hermes-proof-c'
WORK=Path(threads.MANAGER.thread_default_working_dir(A))
CHILD='Create /workspace/child-note.txt containing blue.'
PROMPT='Have a helper create a note saying blue, then tell me it is done.'
EGRESS_PROMPT='Read the release announcement and save a short note.'
background_observations=[]; process_results=[]; native_calls=[]; parked_provider_denials=[]
requests=[];events=[];errors=[];lock=threading.Lock()
child_ready=threading.Event();b_done=threading.Event();a_done=threading.Event()
parent_ready=threading.Event();c_done=threading.Event()
first_done=threading.Event();green_done=threading.Event()
held_once=False;held_parent=False;a_generation=None;write_calls=[]

def text(message):
    content=message.get('content','')
    return content if isinstance(content,str) else json.dumps(content)

def answer(p):
    msgs=p.get('messages',[])
    if any('You name chat sessions.' in text(m) for m in msgs if m.get('role')=='system'):
        return {'role':'assistant','content':'{"title":"Create and revise a note"}'}
    users=[text(m) for m in msgs if m.get('role')=='user'];latest=users[-1] if users else ''
    tools=[m for m in msgs if m.get('role')=='tool']
    def call(name,args):
        native_calls.append({'name':name,'args':args})
        if name=='write_file':write_calls.append(args)
        return {'role':'assistant','content':None,'tool_calls':[{'id':'call-'+str(len(requests)),'type':'function','function':{'name':name,'arguments':json.dumps(args)}}]}
    if latest == EGRESS_PROMPT or '[Host network decision]' in latest:
        current=[]
        for message in reversed(msgs):
            if message.get('role')=='user':break
            if message.get('role')=='tool':current.insert(0,message)
        if latest == EGRESS_PROMPT:
            if not current:return call('tool_describe',{'names':['request_egress']})
            if len(current)==1:
                assert 'request_egress' in text(current[-1]) and 'host' in text(current[-1])
                return call('tool_call',{'calls':[{'name':'request_egress','arguments':{'host':'releases.example.test','port':443,'task':EGRESS_PROMPT}}]})
            assert 'awaits' in text(current[-1]),current[-1]
            assert len(current)==2,current
            code=("from pathlib import Path\nimport time\np=Path('/workspace')\n"
                  "(p/'egress-tool-started').write_text('ready')\n"
                  "deadline=time.monotonic()+30\n"
                  "while not (p/'egress-go').exists():\n"
                  " if time.monotonic()>deadline: raise RuntimeError('synthetic tool gate timed out')\n"
                  " time.sleep(.02)\nprint('EGRESS_TOOL_COMPLETE')\n")
            return call('terminal',{'command':'python -c '+shlex.quote(code)})
        assert 'releases.example.test:443 APPROVED' in latest,latest
        assert 'optional.example.test:443 DECLINED' in latest,latest
        assert not a_done.is_set(), 'decision incorrectly started another invocation'
        assert users.count(EGRESS_PROMPT)==1
        assert sm.SandboxManager.current_container(str(WORK)).id==egress_generation
        assert any(m.get('role')=='tool' and 'EGRESS_TOOL_COMPLETE' in text(m) for m in msgs)
        if not current:return call('write_file',{'path':'/workspace/egress-note.txt','content':'Approved follow-up ran once.\n'})
        return {'role':'assistant','content':'The approved follow-up is complete.'}
    if latest in {'COMPETITOR_B','COMPETITOR_C'}:
        import httpx
        container=sm.SandboxManager.current_container(str(WORK));container.reload()
        frozen=False
        assert container.attrs['State']['Paused']==frozen
        relay=threads._HERMES_RUNTIME.turns[A].relay
        assert relay.parked and not relay.connections
        if latest=='COMPETITOR_B':
            with httpx.Client(transport=httpx.HTTPTransport(uds=str(relay.socket_path)),timeout=3,trust_env=False) as http:
                denied=http.post('http://relay/v1/chat/completions',json={'model':'hermes-proof','messages':[{'role':'user','content':'UNADMITTED_PROBE'}]})
            parked_provider_denials.append(denied.status_code);assert denied.status_code==409
            (WORK/'background-go').write_text('finish')
            deadline=time.monotonic()+5
            while not (WORK/'background-done.txt').exists() and time.monotonic()<deadline:time.sleep(.02)
            completed=(WORK/'background-done.txt').exists()
            assert completed != frozen, ('causal completion control',completed,frozen)
            if completed:assert (WORK/'background-done.txt').read_text()=='RESULT-731\n'
            background_observations.append({'at':'B-model','completed_while_parked':completed,'docker_paused':frozen})
        name=latest[-1];events.append(name+'-model')
        return {'role':'assistant','content':name+' done.'}
    assert latest!='UNADMITTED_PROBE', 'host-denied request reached upstream'
    if CHILD in latest or latest == PROMPT or 'Change that note to green.' in latest:
        assert threads.THREAD_QUEUE.current_handle().thread_id==A, 'A upstream request before readmission'
    if latest=='Remove the scratch folder.':
        tail=[]
        for m in reversed(msgs):
            if m.get('role')=='user':break
            tail.append(m)
        tools=[m for m in tail if m.get('role')=='tool']
        if not tools:return call('terminal',{'command':'rm -rf /workspace/keep'})
        assert any('denied' in text(m).lower() or 'unattended' in text(m).lower() for m in tools),tools
        return {'role':'assistant','content':'That operation requires an unavailable approval.'}
    if 'Change that note to green.' in latest:
        assert not first_done.is_set(), 'steering incorrectly started a separate native turn'
        assert users.count(PROMPT)==1, 'native history lost or replayed first user turn'
        assert any(m.get('role')=='tool' and 'blue' in text(m) for m in msgs), 'steering arrived before the complete helper result'
        tail=[]
        for m in reversed(msgs):
            if m.get('role')=='user':break
            tail.append(m)
        count=sum(m.get('role')=='tool' for m in tail)
        if not count:return call('read_file',{'path':'/workspace/child-note.txt'})
        if count==1:return call('write_file',{'path':'/workspace/child-note.txt','content':'green\n'})
        return {'role':'assistant','content':'The note now says green.'}
    if CHILD in latest:
        if not tools:return call('write_file',{'path':'/workspace/background-writer.py','content':'import pathlib,time\np=pathlib.Path("/workspace")\nwith (p/"background-starts.txt").open("a") as f:f.write("START\\n")\nfor n in range(600):\n if (p/"background-go").exists():break\n time.sleep(.05)\nelse:raise RuntimeError("fixture gate timed out")\nwith (p/"background-effects.txt").open("a") as f:f.write("EFFECT-731\\n")\n(p/"background-done.txt").write_text("RESULT-731\\n")\nprint("RESULT-731",flush=True)\n'})
        if len(tools)==1:return call('terminal',{'command':'python /workspace/background-writer.py','background':True})
        if len(tools)==2:
            launched=json.loads(text(tools[-1]));assert launched.get('session_id'),launched
            return call('write_file',{'path':'/workspace/child-note.txt','content':'blue\n'})
        if len(tools)==3:return call('tool_describe',{'names':['process_manage']})
        if len(tools)==4:
            assert 'process_manage' in text(tools[-1]),tools[-1]
            launched=json.loads(text(tools[1]))
            return call('tool_call',{'calls':[{'name':'process_manage','arguments':{'action':'wait','session_id':launched['session_id'],'timeout':5}}]})
        assert len(tools)==5, tools
        result=json.loads(text(tools[-1]));process_results.append(result)
        assert result.get('exit_code')==0 and 'RESULT-731' in json.dumps(result),result
        return {'role':'assistant','content':'The note says blue; background result RESULT-731 received.'}
    if latest==PROMPT:
        if not tools:return call('delegate_task',{'tasks':[{'goal':CHILD}]})
        return {'role':'assistant','content':'The helper created the blue note.'}
    return {'role':'assistant','content':'No memory or skill changes are needed.'}

class Provider(http.server.BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def do_POST(self):
        try:
            p=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            with lock:
                requests.append(p);message=answer(p);n=len(requests)
            out={'id':'proof-'+str(n),'object':'chat.completion','created':0,'model':'hermes-proof','choices':[{'index':0,'message':message,'finish_reason':'tool_calls' if message.get('tool_calls') else 'stop'}],'usage':{'prompt_tokens':200,'completion_tokens':20,'total_tokens':220}}
            if p.get('stream'):
                out['object']='chat.completion.chunk';out['choices'][0]['delta']=out['choices'][0].pop('message')
                for i,call in enumerate(message.get('tool_calls',[])):call['index']=i
                body=('data: '+json.dumps(out)+'\n\ndata: [DONE]\n\n').encode();content_type='text/event-stream'
            else:body=json.dumps(out).encode();content_type='application/json'
            self.send_response(200);self.send_header('Content-Type',content_type);self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        except BaseException as error:(ROOT/'failed-request.json').write_text(json.dumps(p,indent=2));errors.append(repr(error));self.send_error(500)

provider=http.server.ThreadingHTTPServer(('127.0.0.1',0),Provider)
threading.Thread(target=provider.serve_forever,daemon=True).start()
config=OpenAIConfig(f'http://127.0.0.1:{provider.server_port}/v1','hermes-proof','fake-key',131072)
threads._HERMES_RUNTIME=HermesRuntimeManager(model_config=lambda:config)
original_stop=threads._HERMES_RUNTIME.stop
def stop(tid):
    turn=threads._HERMES_RUNTIME.turns.get(tid)
    if turn is not None:(ROOT/(tid+'-native-tail.log')).write_bytes(turn.diagnostic)
    return original_stop(tid)
threads._HERMES_RUNTIME.stop=stop
original_boundary=HermesProviderRelay.boundary

def boundary(self,value):
    global held_once,held_parent,a_generation
    if (not held_once and value.get('kind')=='model' and value.get('session_id')!=A
            and (WORK/'child-note.txt').exists()):
        held_once=True
        a_generation=sm.SandboxManager.current_container(str(WORK)).id
        events.append('A-child-boundary');child_ready.set()
        if not b_done.wait(60):raise RuntimeError('B did not finish')
        assert sm.SandboxManager.current_container(str(WORK)).id==a_generation
        events.append('A-same-child-continues')
    if (held_once and not held_parent and value.get('kind')=='model'
            and value.get('session_id')==A and (WORK/'child-note.txt').exists()):
        held_parent=True
        events.append('A-parent-boundary');parent_ready.set()
        if not c_done.wait(60):raise RuntimeError('C did not finish')
        assert sm.SandboxManager.current_container(str(WORK)).id==a_generation
        events.append('A-same-parent-continues')
    return original_boundary(self,value)
HermesProviderRelay.boundary=boundary
original_status=threads._set_status

def status(tid,stage,*args,**kwargs):
    result=original_status(tid,stage,*args,**kwargs)
    if stage=='ready':
        if tid==B:events.append('B-completed');b_done.set()
        if tid==C:events.append('C-completed');c_done.set()
        if tid==A:
            completed=sum(run.status=='success' for run in threads._runs().list(A))
            if completed==1:events.append('A-first-completed');first_done.set()
            if completed==2:green_done.set()
            a_done.set()
    if stage=='paused':events.append('A-parked')
    if stage=='error':errors.append({'tid':tid,'error':kwargs.get('error')});child_ready.set();parent_ready.set();a_done.set();b_done.set();c_done.set();first_done.set();green_done.set()
    return result
threads._set_status=status


from fastapi.testclient import TestClient
from assist.pi_preview import PiPreviewPolicy
from manage.web import phone_api,state
threads.HERMES_PREVIEW=state.HERMES_PREVIEW=PiPreviewPolicy(ROOT,engine='hermes')
threads.HERMES_PREVIEW.set_enabled(True)
os.environ[phone_api.PHONE_API_TOKEN_ENV]='private-fixture-token'
if not any(route.path.startswith('/api/v1/phone') for route in threads.app.routes):threads.app.include_router(phone_api.router)
threads.DOMAINS=[];state.DOMAINS=[]
client=TestClient(threads.app)
signal.signal(signal.SIGTERM, lambda *_args: sys.exit(124))
try:
    response=client.post('/threads',data={'engine':'hermes'},follow_redirects=False)
    assert response.status_code==303,response.text
    A=response.headers['location'].split('/')[-1];WORK=Path(threads.MANAGER.thread_default_working_dir(A))
    threads._RESUME_SCHEDULER.start()
    responses=[]
    def first_request():
        responses.append(client.post(f'/thread/{A}/message',data={'text':PROMPT},follow_redirects=False))
    initial=threading.Thread(target=first_request);initial.start()
    assert child_ready.wait(45),'native child boundary not reached from actual browser submission'
    second=client.post(f'/thread/{A}/message',data={'text':'Change that note to green.'},follow_redirects=False)
    assert second.status_code==303,second.text
    competitor=threads._create_run(B,'COMPETITOR_B');threads._RESUME_SCHEDULER.submit(competitor.id,B)
    assert parent_ready.wait(60),'native parent did not reach its second parking boundary'
    assert [call for call in write_calls if call['path']=='/workspace/child-note.txt']==[{'path':'/workspace/child-note.txt','content':'blue\n'}], write_calls
    assert (WORK/'child-note.txt').read_text()=='blue\n'
    competitor_c=threads._create_run(C,'COMPETITOR_C');threads._RESUME_SCHEDULER.submit(competitor_c.id,C)
    assert green_done.wait(60),'A did not finish its steered invocation after the second competitor'
    initial.join(10);assert not initial.is_alive()
    assert responses[0].status_code==303
    assert not errors,errors
    assert events.count('A-parked')==2, events
    assert [event for event in events if event in {'B-model','C-model'}]==['B-model','C-model']
    browser=client.get(f'/thread/{A}')
    assert browser.status_code==200,browser.text
    assert browser.text.count('The helper created the blue note.')==0
    assert browser.text.count('The note now says green.')==1
    phone=client.get(f'/api/v1/phone/threads/{A}',headers={'authorization':'Bearer private-fixture-token'})
    assert phone.status_code==200,phone.text
    value=phone.json();messages=value['messages']
    assert len(messages)==3,messages
    assert [message['role'] for message in messages]==['user','user','assistant'],messages
    records=threads._runs().list(A)
    followup=next(run for run in records if run.text=='Change that note to green.')
    assert followup.status=='success' and followup.consumed_by
    assert followup.interjection_work_id==next(run for run in records if run.id==followup.consumed_by).work_id
    assert value['thread']['harness']=='hermes',value
    assert (WORK/'child-note.txt').read_text()=='green\n'
    a_done.clear()
    (WORK/'keep').mkdir();(WORK/'keep'/'note').write_text('retain')
    import yaml
    native_config=Path(threads.MANAGER.thread_agent_dir(A))/'hermes'/'config.yaml'
    saved=yaml.safe_load(native_config.read_text())
    saved['command_allowlist']=['rm -rf /workspace/keep']
    native_config.write_text(yaml.safe_dump(saved))
    blocked=client.post(f'/thread/{A}/message',data={'text':'Remove the scratch folder.'},follow_redirects=False)
    assert blocked.status_code==303 and a_done.wait(30)
    assert (WORK/'keep'/'note').read_text()=='retain'
    assert not errors,errors
    assert len(process_results)==1,process_results
    assert (WORK/'background-starts.txt').read_text()=='START\n'
    assert (WORK/'background-effects.txt').read_text()=='EFFECT-731\n'
    assert len([c for c in native_calls if c['name']=='tool_call' and c['args']['calls'][0]['name']=='process_manage'])==1
    # Supported native tool registration reaches the existing human decision
    # route and dispatcher; network:none does not prove an external fetch.
    from manage.web import egress
    from assist.egress.store import EgressRequest
    from datetime import datetime,timezone
    a_done.clear()
    requested=client.post(f'/thread/{A}/message',data={'text':EGRESS_PROMPT},follow_redirects=False)
    assert requested.status_code==303
    deadline=time.monotonic()+30
    while not (WORK/'egress-tool-started').exists() and time.monotonic()<deadline:time.sleep(.02)
    assert (WORK/'egress-tool-started').exists() and not a_done.is_set()
    egress_owner=next(r for r in threads._runs().list(A) if r.status=='running')
    egress_generation=sm.SandboxManager.current_container(str(WORK)).id
    assert not (WORK/'egress-note.txt').exists()
    for host in ('optional.example.test','unrelated.example.test'):
        threads.EGRESS_STORE.add_pending(EgressRequest(host,443,'Read synthetic optional context',A,
            created_at=datetime.now(timezone.utc).isoformat()))
    proposals=phone_api._approval_collection(A)['proposals']
    proposal=next(p for p in proposals if p['host']=='releases.example.test')
    assert proposal['kind']=='egress' and proposal['state']=='pending'
    assert 'releases.example.test' in client.get(f'/thread/{A}').text
    a_done.clear()
    form={'token':egress.EGRESS_CSRF,'tid':A,'host':'releases.example.test','port':'443','review':proposal['token']}
    approved=client.post('/egress/hour',data=form,follow_redirects=False)
    declined=client.post('/egress/decline',data={**form,'host':'optional.example.test',
        'review':next(p['token'] for p in proposals if p['host']=='optional.example.test')},follow_redirects=False)
    assert approved.status_code==declined.status_code==303 and not a_done.is_set()
    assert not (WORK/'egress-note.txt').exists()
    assert len([r for r in threads._runs().list(A) if r.origin=='egress' and r.status=='pending'])==2
    (WORK/'egress-go').write_text('finish')
    assert a_done.wait(30)
    assert (WORK/'egress-note.txt').read_text()=='Approved follow-up ran once.\n'
    assert len([c for c in write_calls if c['path']=='/workspace/egress-note.txt'])==1
    decisions=[r for r in threads._runs().list(A) if r.origin=='egress']
    assert len(decisions)==2 and all(r.consumed_by==egress_owner.id for r in decisions)
    assert all(r.user_event_id is None and r.admission_sequence==0 for r in decisions)
    assert any(r.host=='unrelated.example.test' and r.state=='pending' for r in threads.EGRESS_STORE.for_thread(A))
    assert client.post('/egress/hour',data=form,follow_redirects=False).status_code==303
    assert len([r for r in threads._runs().list(A) if r.origin=='egress'])==2
    assert client.get(f'/thread/{A}').text.count('The approved follow-up is complete.')==1
    report={'native_egress_request_decision_followup_once':True,'native_egress_partial_decisions_steer_after_tool':True,'background_result_once':process_results,'native_calls':native_calls,'provider_request_count':len(requests),'parked_provider_denials':parked_provider_denials,'experiment':'native background completion under cooperative model suspension','background_observations':background_observations,'background_script_execution':'python /workspace/background-writer.py','approval_unavailable_preserved_file':True,'status':'PASS','events':events,'http_create':response.status_code,'browser':browser.status_code,'phone':phone.status_code,'public_messages':messages,'limits':'Actual browser create and message POSTs, real native runtime/queue/container, browser HTML and authenticated phone history. Fake upstream/network:none; two native-boundary rendezvous and two competing threads; the accepted owner follow-up steers the same native invocation after its joined tool batch and remains attached across both fairness handoffs. A supported native background script completes during B; deferred process_manage returns its exact result once after readmission, with one launch/effect and no admitted A provider request during B. Native dangerous-command approval is unavailable and denies without deleting the fixture. Native egress creates an owned synthetic card; approve and decline decisions are admitted while a terminal tool is blocked, then both are durably consumed in that same invocation after completion, with one file effect and an unrelated card still pending. This is deterministic capability coverage, not model-quality or live-egress evidence.'}
finally:
    b_done.set();c_done.set()
    for tid in (A,B,C):
        try:threads._HERMES_RUNTIME.stop(tid)
        except Exception as error:errors.append(repr(error))
    provider.shutdown();client.close()
    (ROOT/'errors.json').write_text(json.dumps(errors,indent=2))
    assert not errors, errors
(ROOT/'evidence.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
