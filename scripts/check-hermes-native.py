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
for tid in ('hermes-proof-a','hermes-proof-b'):
    threads.MANAGER.reserve_visible('hermes',thread_id=tid)
    Path(threads.MANAGER.thread_default_working_dir(tid)).mkdir(exist_ok=True)
A='hermes-proof-a';B='hermes-proof-b'
WORK=Path(threads.MANAGER.thread_default_working_dir(A))
CHILD='Create /workspace/child-note.txt containing blue.'
PROMPT='Have a helper create a note saying blue, then tell me it is done.'
requests=[];events=[];errors=[];lock=threading.Lock()
child_ready=threading.Event();b_done=threading.Event();a_done=threading.Event()
held_once=False;a_generation=None;write_calls=[]

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
        if name=='write_file':write_calls.append(args)
        return {'role':'assistant','content':None,'tool_calls':[{'id':'call-'+str(len(requests)),'type':'function','function':{'name':name,'arguments':json.dumps(args)}}]}
    if latest=='COMPETITOR_B':
        container=sm.SandboxManager.current_container(str(WORK));container.reload()
        assert container.attrs['State']['Paused'],'A generation must be frozen before B model'
        events.append('B-model')
        return {'role':'assistant','content':'B done.'}
    if latest=='Remove the scratch folder.':
        tail=[]
        for m in reversed(msgs):
            if m.get('role')=='user':break
            tail.append(m)
        tools=[m for m in tail if m.get('role')=='tool']
        if not tools:return call('terminal',{'command':'rm -rf /workspace/keep'})
        assert any('denied' in text(m).lower() or 'unattended' in text(m).lower() for m in tools),tools
        return {'role':'assistant','content':'That operation requires an unavailable approval.'}
    if latest=='Change that note to green.':
        tail=[]
        for m in reversed(msgs):
            if m.get('role')=='user':break
            tail.append(m)
        count=sum(m.get('role')=='tool' for m in tail)
        if not count:return call('read_file',{'path':'/workspace/child-note.txt'})
        if count==1:return call('write_file',{'path':'/workspace/child-note.txt','content':'green\n'})
        return {'role':'assistant','content':'The note now says green.'}
    if CHILD in latest:
        if not tools:return call('write_file',{'path':'/workspace/child-note.txt','content':'blue\n'})
        return {'role':'assistant','content':'The note says blue.'}
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
        except BaseException as error:errors.append(repr(error));self.send_error(500)

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
    global held_once,a_generation
    if (not held_once and value.get('kind')=='model' and value.get('session_id')!=A
            and (WORK/'child-note.txt').exists()):
        held_once=True
        a_generation=sm.SandboxManager.current_container(str(WORK)).id
        events.append('A-child-boundary');child_ready.set()
        if not b_done.wait(60):raise RuntimeError('B did not finish')
        assert sm.SandboxManager.current_container(str(WORK)).id==a_generation
        events.append('A-same-child-continues')
    return original_boundary(self,value)
HermesProviderRelay.boundary=boundary
original_status=threads._set_status

def status(tid,stage,*args,**kwargs):
    result=original_status(tid,stage,*args,**kwargs)
    if stage=='ready':
        if tid==B:events.append('B-completed');b_done.set()
        if tid==A:a_done.set()
    if stage=='paused':events.append('A-parked')
    if stage=='error':errors.append({'tid':tid,'error':kwargs.get('error')});child_ready.set();a_done.set();b_done.set()
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
    competitor=threads._create_run(B,'COMPETITOR_B');threads._RESUME_SCHEDULER.submit(competitor.id,B)
    assert a_done.wait(60),'A did not resume and finish'
    initial.join(10);assert not initial.is_alive()
    assert responses[0].status_code==303
    assert not errors,errors
    a_done.clear()
    second=client.post(f'/thread/{A}/message',data={'text':'Change that note to green.'},follow_redirects=False)
    assert second.status_code==303,second.text
    assert a_done.wait(45)
    assert not errors,errors
    browser=client.get(f'/thread/{A}')
    assert browser.status_code==200,browser.text
    assert browser.text.count('The helper created the blue note.')==1
    assert browser.text.count('The note now says green.')==1
    phone=client.get(f'/api/v1/phone/threads/{A}',headers={'authorization':'Bearer private-fixture-token'})
    assert phone.status_code==200,phone.text
    value=phone.json();messages=value['messages']
    assert len(messages)==4,messages
    assert value['thread']['harness']=='hermes',value
    assert (WORK/'child-note.txt').read_text()=='green\n'
    a_done.clear()
    (WORK/'keep').mkdir();(WORK/'keep'/'note').write_text('retain')
    blocked=client.post(f'/thread/{A}/message',data={'text':'Remove the scratch folder.'},follow_redirects=False)
    assert blocked.status_code==303 and a_done.wait(30)
    assert (WORK/'keep'/'note').read_text()=='retain'
    assert not errors,errors
    report={'approval_unavailable_preserved_file':True,'status':'PASS','events':events,'http_create':response.status_code,'browser':browser.status_code,'phone':phone.status_code,'public_messages':messages,'limits':'Actual browser create and two message POSTs, real native runtime/queue/container, browser HTML and authenticated phone history. Fake upstream/network:none; one child-boundary rendezvous. Native dangerous-command approval is unavailable and denies without deleting the fixture. This is deterministic capability coverage, not model-quality or live-egress evidence.'}
    (ROOT/'evidence.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
finally:
    b_done.set()
    for tid in (A,B):
        try:threads._HERMES_RUNTIME.stop(tid)
        except Exception as error:errors.append(repr(error))
    provider.shutdown();client.close()
    (ROOT/'errors.json').write_text(json.dumps(errors,indent=2))
