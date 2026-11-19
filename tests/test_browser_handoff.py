"""Same-page viewer acceptance across the actual Run and affinity queue paths."""
import base64
import io
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import re
import socket
from pathlib import Path
from threading import Event, Thread, current_thread
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from assist.browser import authority, manager as browser
from assist.browser.runner import BrowserWorker
from assist.browser import runner
from assist.middleware.thread_queue_middleware import ThreadQueueMiddleware
from assist.thread_queue import ThreadAffinityQueue
from assist.thread_queue import ThreadPauseRequested
from manage import web
from manage.web import threads
from tests.test_fair_scheduling_integration import wired
from tests.test_browser_lifecycle_v12 import admitted


@pytest.fixture
def controlled(admitted, monkeypatch):
    root, runs = admitted
    owner = runs.create('t', 'general-agent', 'Synthetic browser task', user_origin=True)
    runs.claim('t', owner.id)
    runs.transition('t', owner.id, 'interrupted')
    successor = runs.create('t', 'general-agent', None, work_id=owner.work_id,
                            user_event_id=owner.id, resume=True)
    container = SimpleNamespace(id='controlled-generation')
    session = browser.BrowserSession('t', owner.id, str(root/'t'), str(root),
        threads._browser_user_request(owner, runs.list('t')),work_id=owner.work_id,
        sandbox_generation=container.id, sandbox_backend=SimpleNamespace(container=container),run_service=runs)
    session.identity=browser._ContainerIdentity(str(root),'172.20.0.2',container.id)
    session.human_active=session.human_requested=True
    session.human_client='a'*32
    with authority.fence(str(root),'t') as state:
        state.begin(owner.id,owner.admission_sequence)
        state.add_generation(owner.id,container.id)
    monkeypatch.setattr(browser.BrowserManager,'_sessions',{'t':session})
    monkeypatch.setattr(browser.SandboxManager,'_containers',{str(root/'t'):container})
    stopped=[]
    monkeypatch.setattr(browser,'_kill_container_confirmed',lambda generation:stopped.append(generation))
    monkeypatch.setattr(browser,'_bounded_cli',lambda *_a,**_k:b'')
    monkeypatch.setattr(browser,'browser_records',lambda *_a:{})
    monkeypatch.setattr(browser,'disarm_sandbox_browser',lambda *_a:None)
    monkeypatch.setattr(browser,'forget_client',lambda *_a:None)
    monkeypatch.setattr(session,'_capture_storage',lambda:None)
    def cleanup(work_dir, expected=None):
        browser.SandboxManager._containers.pop(work_dir,None)
    monkeypatch.setattr(browser.SandboxManager,'cleanup',cleanup)
    return root,runs,owner,successor,session,stopped


@pytest.mark.parametrize('reason',['newer_user','cancel'])
def test_human_wait_revokes_without_handback(controlled,monkeypatch,reason):
    root,runs,owner,successor,session,stopped=controlled
    token=session.viewer_token
    if reason=='newer_user':
        held=threads._create_run('t','New instruction',user_origin=True,
                                 revocation_pending=True,browser_reset_run_id=owner.id)
    else:
        runs.cancel_logical('t',owner.id,browser_owner_run_id=owner.id)
        from manage.web import phone_api
        monkeypatch.setattr(phone_api,'_finish_browser_cancel_cleanup',
                            lambda tid,rid:runs.complete_cancel_cleanup(tid,rid))
    assert threads._drain_held_browser_events('t') is True
    assert stopped==['controlled-generation']
    assert session.closed and session.viewer_token!=token
    assert browser.BrowserManager.current_session('t') is None
    with authority.fence(str(root),'t') as state:
        assert state.lease is None
    if reason=='newer_user':
        assert runs.get('t',held.id).status=='pending'
    else:
        assert runs.get('t',owner.id).cancel_cleanup=='complete'


def test_child_acquiring_model_slot_during_human_control_remains_pending(controlled,monkeypatch):
    root,runs,owner,successor,session,stopped=controlled
    (root/'sub-child').mkdir()
    child=runs.create('sub-child','delegate-agent','Read synthetic state',mode='child',
                     parent_thread_id='t',parent_run_id=owner.id,dispatch_key='held-child')
    queue=ThreadAffinityQueue()
    monkeypatch.setattr(threads,'THREAD_QUEUE',queue)
    with queue.acquire('unrelated'):
        worker=Thread(target=threads._execute_child_run,args=(child,),daemon=True)
        worker.start()
    worker.join(2)
    assert not worker.is_alive()
    assert runs.get('sub-child',child.id).status=='pending'
    assert (child.id,'sub-child') in session.held_runs
    assert session.human_active and stopped==[]


def test_control_is_exclusive_and_return_retires_input_capability(controlled):
    _root,_runs,_owner,_successor,session,_stopped=controlled
    token=session.viewer_token
    with pytest.raises(browser.BrowserUnavailable,match='another viewer'):
        session.request_human(token,'b'*32)
    session.return_control(token,'a'*32)
    assert session.paused and not session.human_active
    with pytest.raises(browser.BrowserUnavailable,match='viewer ended'):
        session.viewer_command(token,'a'*32,'tabs')


@pytest.mark.parametrize('confirmed',[True,False])
def test_return_reconciles_lost_pause_acknowledgment(controlled,monkeypatch,confirmed):
    _root,runs,_owner,successor,session,stopped=controlled
    session.held_runs[(successor.id,'t')]=None
    session.on_control_end=lambda:threads._release_browser_waiters(session)
    scheduled=[]
    monkeypatch.setattr(threads._RESUME_SCHEDULER,'submit',lambda rid,tid:scheduled.append((rid,tid)))
    def control(argv,**_kwargs):
        if argv[1]=='pause':
            raise browser.BrowserUnavailable('pause accepted; acknowledgment lost')
        assert argv[1]=='inspect'
        return b'true\n' if confirmed else b'false\n'
    monkeypatch.setattr(browser,'_bounded_cli',control)
    if confirmed:
        session.return_control(session.viewer_token,session.human_client)
        assert session.paused and not session.human_active and stopped==[]
        assert scheduled==[(successor.id,'t')]
    else:
        with pytest.raises(browser.BrowserUnavailable,match='could not confirm'):
            session.return_control(session.viewer_token,session.human_client)
        assert session.closed and stopped==['controlled-generation']
        assert runs.get('t',successor.id).status=='error'
        assert scheduled==[]


def test_failed_return_preserves_other_work_and_drains_followers(controlled,monkeypatch):
    _root,runs,owner,successor,session,_stopped=controlled
    held=runs.create('t','general-agent','Held other work',origin='task-completion')
    unnotified=runs.create('t','general-agent','Follower not yet notified',origin='continuation')
    session.held_runs[(held.id,'t')]=None
    # The same-work successor may already have a queue ticket and need not be
    # in held_runs when a fairness-paused browser is taken over and returned.
    session.viewer_error='Handback ended this live browser'
    session.human_active=session.human_requested=False
    submitted=[]
    monkeypatch.setattr(threads._RESUME_SCHEDULER,'submit',lambda rid,tid,**_k:submitted.append((rid,tid)))
    from tests.test_browser_lifecycle_v12 import _ORIGINAL_DISPATCH_PENDING
    drained=[]
    def drain(tid):
        drained.append(tid)
        _ORIGINAL_DISPATCH_PENDING(tid)
    monkeypatch.setattr(threads,'_dispatch_pending_after',drain)
    threads._release_browser_waiters(session)
    assert runs.get('t',successor.id).status=='error'
    assert runs.get('t',held.id).status=='pending'
    assert runs.get('t',unnotified.id).status=='pending'
    assert (held.id,'t') in submitted
    assert drained==['t']


def test_viewer_http_requires_origin_token_and_bounded_fixed_input(controlled,monkeypatch):
    _root,_runs,_owner,_successor,session,_stopped=controlled
    client=TestClient(web.app)
    url='/thread/t/browser/control'
    body={'client':'a'*32,'action':'status'}
    assert client.post(url,json=body).status_code==403
    headers={'Origin':'https://other.example','X-Browser-Token':session.viewer_token}
    assert client.post(url,json=body,headers=headers).status_code==403
    headers['Origin']='http://testserver'
    assert client.post(url,json=body,headers=headers).status_code==200
    assert client.post(url,json={**body,'action':[]},headers=headers).status_code==422
    assert client.post(url,json={**body,'action':'evaluate','script':'arbitrary'},headers=headers).status_code==422
    assert client.post(url,content=b'x'*12001,headers=headers).status_code==413
    headers['X-Browser-Token']='0'*32
    assert client.post(url,json=body,headers=headers).status_code==409


def test_viewer_pixels_cross_private_transport_without_widening_agent_results(tmp_path,monkeypatch):
    path=str(tmp_path/'browser.sock')
    response=json.dumps({'result':{'image':base64.b64encode(b'j'*70000).decode()}}).encode()
    output=io.BytesIO()
    with socket.socket(socket.AF_UNIX) as listener:
        listener.bind(path)
        listener.listen(1)
        def serve():
            with listener.accept()[0] as peer:
                while peer.recv(4096):pass
                peer.sendall(response)
        server=Thread(target=serve,daemon=True)
        server.start()
        monkeypatch.setattr(runner,'SOCKET',path)
        monkeypatch.setattr(runner,'sys',SimpleNamespace(
            stdin=SimpleNamespace(buffer=io.BytesIO(json.dumps({'operation':'viewer'}).encode())),
            stdout=SimpleNamespace(buffer=output)))
        runner.call()
        server.join(2)
    assert output.getvalue()==response and len(response)>runner.MAX_RESULT


@pytest.mark.parametrize('state',['empty','closed','remaining','transport'])
def test_handback_discovers_current_pages_without_admitting_stale_effects(controlled,monkeypatch,state):
    _root,_runs,_owner,_successor,session,_stopped=controlled
    pages=[] if state in {'empty','closed'} else [{'page_id':'current','url':'https://example.test/'}]
    effects=[]
    def transport(_generation,payload,**_kwargs):
        command=json.loads(payload)
        if command['operation']=='viewer':
            if state=='transport':raise OSError('private transport unavailable')
            answer={'pages':pages}
        elif command['operation']=='observe':
            if command['args']['page_id']!='current':
                return json.dumps({'error':'unknown or closed page ID'}).encode()
            answer={'page_id':'current','snapshot':'Current human page'}
        else:
            effects.append(command['operation'])
            answer={'page_id':'opened'}
        return json.dumps({'result':answer}).encode()
    monkeypatch.setattr(browser,'_docker_exec',transport)
    session.return_control(session.viewer_token,session.human_client)
    browser.BrowserManager.resume_session('t',session.work_id,session.user_request)
    if state=='transport':
        with pytest.raises(OSError,match='unavailable'):
            session.command('open',url='https://example.test/')
        session.model_boundary()
        assert session.needs_observation and not session.handoff_observed and effects==[]
        return
    if state=='closed':
        assert session.command('observe',page_id='gone')['result']['pages']==[]
    else:
        assert session.command('open',url='https://example.test/')['pages']==pages
    assert 'error' in session.command('open',url='https://example.test/')
    if state=='remaining':
        session.model_boundary()
        assert session.needs_observation
        assert session.command('observe',page_id='gone')['pages']==pages
        assert session.command('observe',page_id='current')['result']['snapshot']=='Current human page'
        assert 'error' in session.command('open',url='https://example.test/')
    session.model_boundary()
    assert session.command('open',url='https://example.test/')['result']['page_id']=='opened'
    assert effects==['open']


@pytest.fixture
def handoff_host(wired,monkeypatch):
    tid,_=wired
    root=Path(web.MANAGER.root_dir)
    queue=ThreadAffinityQueue()
    scheduler=threads._ResumeScheduler()
    containers,stopped,notifications={},{},[]
    faults={'stop':False}
    monkeypatch.setattr(threads,'THREAD_QUEUE',queue)
    monkeypatch.setattr(threads,'_RESUME_SCHEDULER',scheduler)
    monkeypatch.setattr(browser.BrowserManager,'_sessions',{})
    monkeypatch.setattr(browser.BrowserManager,'_reconciled_roots',{str(root)})
    monkeypatch.setattr(browser.BrowserManager,'reconcile_startup',lambda *_a:True)
    monkeypatch.setattr(browser.BrowserManager,'ready_for_browser',lambda *_a:True)
    monkeypatch.setattr(browser.SandboxManager,'_containers',containers)
    monkeypatch.setattr(browser,'configured_directory',lambda:str(root))
    monkeypatch.setattr(threads,'configured_directory',lambda:str(root))
    monkeypatch.setattr(browser,'browser_records',lambda *_a:{})
    monkeypatch.setattr(threads,'browser_records',lambda *_a:{})
    monkeypatch.setattr(browser,'_sandbox_generations',lambda *_a:set())
    monkeypatch.setattr(browser,'_bounded_cli',lambda *_a,**_k:b'')
    monkeypatch.setattr(browser,'disarm_sandbox_browser',lambda *_a:None)
    monkeypatch.setattr(browser,'forget_client',lambda *_a:None)
    monkeypatch.setattr(browser.BrowserSession,'_capture_storage',lambda _s:None)
    def backend(t,**kwargs):
        container=SimpleNamespace(id='host-'+t)
        containers[str(root/t)]=container
        run=threads._runs().get(t,kwargs['owner_run_id'])
        with authority.fence(str(root),t) as lease:
            sequence=authority.execution_sequence(threads._runs().list(t),run.id)
            lease.begin(run.id,sequence)
            lease.add_generation(run.id,container.id)
        return SimpleNamespace(container=container)
    def start(session):
        session.identity=browser._ContainerIdentity(str(root),'172.20.0.2',session.sandbox_generation)
    def stop(generation):
        if faults['stop']:raise browser.BrowserUnavailable('exact stop unconfirmed')
        stopped[generation]=stopped.get(generation,0)+1
    def cleanup(work_dir,expected=None):
        if faults['stop']:raise browser.BrowserUnavailable('exact stop unconfirmed')
        container=containers.pop(work_dir,None)
        if container:
            with authority.fence(str(root),Path(work_dir).name) as lease:
                if lease.lease:
                    owner=lease.lease['owner_run_id']
                    lease.remove_generation(owner,container.id)
                    lease.clear(owner)
    monkeypatch.setattr(threads,'_get_sandbox_backend',backend)
    monkeypatch.setattr(browser.BrowserSession,'_start',start)
    monkeypatch.setattr(browser,'_kill_container_confirmed',stop)
    monkeypatch.setattr(browser.SandboxManager,'cleanup',cleanup)
    monkeypatch.setattr(browser,'_docker_exec',lambda *_a,**_k:json.dumps({'result':{
        'page_id':'page','snapshot_id':'snapshot','snapshot':'Synthetic form'}}).encode())
    monkeypatch.setattr(threads,'_queue_browser_revocation',lambda t:notifications.append(t))
    return tid,root,scheduler,stopped,notifications,faults


@pytest.mark.parametrize('phase',['startup','tool','fairness'])
@pytest.mark.parametrize('system',[False,True])
def test_active_browser_followup_reaches_existing_model_boundary(handoff_host,monkeypatch,phase,system):
    tid,root,scheduler,stopped,_notifications,_faults=handoff_host
    from assist.middleware import interjection
    monkeypatch.setattr(interjection,'_turn_sender',lambda _runtime:None)
    hook=interjection.InterjectionMiddleware()
    backend=threads._get_sandbox_backend
    messages=[]
    followups=[]
    generations=[]
    def followup():
        followups.append(threads._accept_message_run(tid,'Keep the current page and adjust the task')[0])
    def starting(t,**kwargs):
        result=backend(t,**kwargs)
        if phase=='startup' and not followups:followup()
        return result
    monkeypatch.setattr(threads,'_get_sandbox_backend',starting)
    transport=browser._docker_exec
    def during_tool(*args,**kwargs):
        if phase=='tool' and not followups:
            with ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(followup).result(timeout=2)
        return transport(*args,**kwargs)
    monkeypatch.setattr(browser,'_docker_exec',during_tool)
    class Chat:
        agent=None
        pending_reply=lambda self:None
        get_messages=lambda self:messages
        get_raw_messages=get_messages
        def boundary(self):
            session=browser.BrowserManager.current_session(tid)
            generations.append(session.sandbox_backend.container)
            result=hook.before_model({'messages':messages},None)
            assert result and followups[0].id in interjection.collect_interjection_ids(result['messages'])
            messages.extend(result['messages'])
            hook.before_model({'messages':messages},None)
            session.command('open',url='https://example.test/after')
            return 'Steering handled in the same execution'
        def message(self,text):
            session=browser.BrowserManager.current_session(tid)
            session.command('open',url='https://example.test/before')
            if phase=='fairness':
                generations.append(session.sandbox_backend.container)
                raise ThreadPauseRequested('ordinary fairness slice')
            return self.boundary()
        def resume(self):return self.boundary()
    monkeypatch.setattr(web.MANAGER,'get',lambda *_a,**_k:Chat())
    if system:
        previous=threads._create_run(tid,'Earlier owner request',user_origin=True)
        threads._runs().claim(tid,previous.id)
        threads._runs().transition(tid,previous.id,'success')
    owner=threads._create_run(tid,'Continue the browser task',origin='system' if system else None,user_origin=not system)
    threads._execute_run(owner.id,tid)
    target=owner
    if phase=='fairness':
        target=next(r for r in threads._runs().list(tid) if r.resume)
        followup()
        threads._execute_run(target.id,tid)
    assert threads._runs().get(tid,target.id).status=='success'
    assert len(followups)==1
    admitted=threads._runs().get(tid,followups[0].id)
    assert admitted.status=='success' and admitted.consumed_by==owner.id
    assert admitted.interjection_work_id==owner.work_id
    assert admitted.work_id!=owner.work_id
    assert len({id(container) for container in generations})==1
    assert stopped=={'host-'+tid:1}


def test_unconsumed_browser_steering_runs_once_as_fallback(handoff_host,monkeypatch):
    tid,_root,scheduler,_stopped,_notifications,_faults=handoff_host
    from tests.test_browser_lifecycle_v12 import _ORIGINAL_DISPATCH_PENDING
    monkeypatch.setattr(threads,'_dispatch_pending_after',_ORIGINAL_DISPATCH_PENDING)
    calls=[]
    followups=[]
    class Chat:
        agent=None
        pending_reply=lambda self:None
        get_messages=lambda self:[]
        get_raw_messages=get_messages
        def message(self,text):
            calls.append(text)
            if not followups:
                for text in ['First unconsumed follow-up','Second unconsumed follow-up']:
                    followups.append(threads._accept_message_run(tid,text)[0])
            elif text=='First unconsumed follow-up':
                followups.append(threads._accept_message_run(tid,'Third unconsumed follow-up')[0])
            return 'Completed this execution'
    monkeypatch.setattr(web.MANAGER,'get',lambda *_a,**_k:Chat())
    owner=threads._create_run(tid,'Original browser work',user_origin=True)
    threads._execute_run(owner.id,tid)
    assert threads._runs().get(tid,owner.id).status=='success'
    assert threads._runs().get(tid,followups[0].id).status=='pending'
    for followup in followups:
        item=scheduler._q.get_nowait()
        assert item['run_id']==followup.id
        threads._execute_run(item['run_id'],tid)
        threads._execute_run(item['run_id'],tid)
        assert threads._runs().get(tid,followup.id).status=='success'
    assert calls==['Original browser work','First unconsumed follow-up','Second unconsumed follow-up','Third unconsumed follow-up']


def test_failed_turn_claimed_steering_retries_both_messages(handoff_host,monkeypatch):
    tid,_root,scheduler,_stopped,_notifications,_faults=handoff_host
    calls=[]
    followups=[]
    class Chat:
        agent=None
        pending_reply=lambda self:None
        get_messages=lambda self:[]
        get_raw_messages=get_messages
        def message(self,text):
            calls.append(text)
            if not followups:
                for message in ['First rescue','Second rescue']:
                    followups.append(threads._accept_message_run(tid,message)[0])
                threads._consume_interjections(tid,{run.id for run in followups})
                raise RuntimeError('model failed after consuming both instructions')
            return 'Recovered this instruction'
    monkeypatch.setattr(web.MANAGER,'get',lambda *_a,**_k:Chat())
    owner=threads._create_run(tid,'Original browser work',user_origin=True)
    threads._execute_run(owner.id,tid)
    assert threads._runs().get(tid,owner.id).status=='error'
    retries=[run for run in threads._runs().list(tid)
             if run.id not in {owner.id,*(followup.id for followup in followups)}]
    assert [run.text for run in retries]==['First rescue','Second rescue']
    for retry in retries:
        item=scheduler._q.get_nowait()
        assert item['run_id']==retry.id
        threads._execute_run(retry.id,tid)
        threads._execute_run(retry.id,tid)
        assert threads._runs().get(tid,retry.id).status=='success'
    assert calls==['Original browser work','First rescue','Second rescue']


def test_targeted_steering_cannot_enter_another_execution(handoff_host,monkeypatch):
    tid,_root,_scheduler,_stopped,_notifications,_faults=handoff_host
    owner=threads._create_run(tid,'Active owner',user_origin=True)
    threads._runs().claim(tid,owner.id)
    followup,_=threads._accept_message_run(tid,'Steer the active owner')
    assert followup.interjection_work_id==owner.work_id
    threads._runs().transition(tid,owner.id,'success')
    other=threads._create_run(tid,'Independent background work',origin='system')
    threads._runs().claim(tid,other.id)
    monkeypatch.setitem(threads._TURN_INTERJECTION,tid,{'run_id':other.id,'claimed':[]})
    assert followup.id not in {record.id for record in threads._pending_run_records(tid)}
    assert threads._runs().get(tid,followup.id).status=='pending'


def test_owner_message_cannot_steer_sender_scoped_execution(handoff_host):
    tid,_root,_scheduler,_stopped,_notifications,_faults=handoff_host
    triage=threads._create_run(tid,'Sender-scoped work',sender='+15550001111')
    threads._runs().claim(tid,triage.id)
    followup,_=threads._accept_message_run(tid,'Owner instruction')
    assert followup.interjection_work_id is None
    assert followup.status=='revocation_pending'


def test_visible_approval_pause_is_not_active_steering(handoff_host):
    tid,_root,_scheduler,_stopped,_notifications,_faults=handoff_host
    owner=threads._create_run(tid,'Work reaching a reply approval',user_origin=True)
    threads._runs().claim(tid,owner.id)
    threads._set_status(tid,'awaiting_approval',pending_reply='Proposed reply')
    followup,_=threads._accept_message_run(tid,'New instruction at approval')
    assert followup.interjection_work_id is None
    assert followup.status=='revocation_pending'


@pytest.mark.parametrize('terminal',['success','error','cancelled'])
def test_terminal_logical_owner_cannot_keep_browser_authority(controlled,terminal):
    _root,runs,_owner,successor,session,_stopped=controlled
    runs.claim('t',successor.id)
    runs.transition('t',successor.id,terminal)
    with pytest.raises(browser.BrowserUnavailable,match='terminal|cancelled'):
        session._fence_command()


@pytest.mark.parametrize('boundary',['steering','replacement','terminal','generation','cancel'])
def test_runtime_and_session_share_execution_owner_fences(controlled,boundary):
    root,runs,owner,successor,session,_stopped=controlled
    from assist.browser.turn_runtime import _check_lease
    if boundary in {'steering','replacement'}:
        runs.create('t','general-agent','New direct message',user_origin=True,
                    interjection_work_id=owner.work_id if boundary=='steering' else None)
    if boundary=='terminal':
        runs.claim('t',successor.id)
        runs.transition('t',successor.id,'success')
    if boundary=='cancel':runs.cancel_logical('t',owner.id)
    request={'threads_root':str(root),'thread_id':'t','run_id':owner.id,
             'generation':'wrong-generation' if boundary=='generation' else session.sandbox_generation}
    if boundary=='generation':
        session.sandbox_generation='wrong-generation'
    if boundary=='steering':
        _check_lease(request)
        session._fence_command()
    else:
        with pytest.raises(RuntimeError):_check_lease(request)
        with pytest.raises(browser.BrowserUnavailable):session._fence_command()


@pytest.mark.parametrize('stop_failure',[False,True])
def test_takeover_fence_rejection_settles_only_after_exact_stop_and_dispatches_new_user(handoff_host,monkeypatch,stop_failure):
    tid,root,scheduler,stopped,notifications,faults=handoff_host
    newer=[]
    class Chat:
        agent=None
        on_queue_state=None
        pending_reply=lambda self:None
        get_messages=lambda self:[]
        get_raw_messages=get_messages
        def message(self,text):
            if text=='New instruction':return 'New instruction processed'
            session=browser.BrowserManager.current_session(tid)
            session.command('open',url='https://example.test/')
            session.request_human(session.viewer_token,'a'*32)
            newer.append(threads._accept_message_run(tid,'New instruction')[0])
            faults['stop']=stop_failure
            raise ThreadPauseRequested('boundary after newer direct admission')
    monkeypatch.setattr(web.MANAGER,'get',lambda *_a,**_k:Chat())
    old=threads._create_run(tid,'Old work',user_origin=True)
    if stop_failure:
        with pytest.raises(browser.BrowserUnavailable,match='unconfirmed'):
            threads._execute_run(old.id,tid)
        assert threads._runs().get(tid,old.id).status=='running'
        assert threads._runs().get(tid,newer[0].id).status=='revocation_pending'
        with authority.fence(str(root),tid) as lease:
            assert lease.lease['generations']==['host-'+tid]
        assert scheduler._q.empty()
        return
    threads._execute_run(old.id,tid)
    assert threads._runs().get(tid,old.id).status=='error'
    assert stopped=={'host-'+tid:1} and len(notifications)>=2
    assert threads._drain_held_browser_events(tid)
    item=scheduler._q.get_nowait()
    assert item['run_id']==newer[0].id
    threads._execute_run(item['run_id'],item['tid'])
    assert threads._runs().get(tid,newer[0].id).status=='success'


def test_immediate_return_after_human_activation_cannot_run_failed_handoff_cleanup(handoff_host,monkeypatch):
    tid,_root,scheduler,stopped,_notifications,_faults=handoff_host
    original_gate=browser.BrowserManager.bounded_thread_gate
    returned=Event()
    sessions=[]
    @contextmanager
    def return_at_gate_exit(t):
        with original_gate(t):yield
        session=browser.BrowserManager.current_session(t)
        if session and session.human_active and not returned.is_set():
            returned.set()
            session.return_control(session.viewer_token,session.human_client)
    monkeypatch.setattr(browser.BrowserManager,'bounded_thread_gate',return_at_gate_exit)
    class Chat:
        agent=None
        on_queue_state=None
        pending_reply=lambda self:None
        get_messages=lambda self:[]
        get_raw_messages=get_messages
        def message(self,_text):
            session=browser.BrowserManager.current_session(tid)
            session.command('open',url='https://example.test/')
            session.request_human(session.viewer_token,'a'*32)
            sessions.append(session)
            raise ThreadPauseRequested('human boundary')
        def resume(self):return 'Same browser resumed'
    monkeypatch.setattr(web.MANAGER,'get',lambda *_a,**_k:Chat())
    old=threads._create_run(tid,'Take over',user_origin=True)
    threads._execute_run(old.id,tid)
    assert returned.is_set() and stopped=={}
    assert browser.BrowserManager.current_session(tid) is sessions[0]
    assert sessions[0].paused and not sessions[0].closed
    item=scheduler._q.get_nowait()
    threads._execute_run(item['run_id'],item['tid'])
    assert threads._runs().get(tid,item['run_id']).status=='success'
    assert stopped=={'host-'+tid:1}


@pytest.mark.parametrize('early_resume',[False,True])
@pytest.mark.parametrize('stop_failure,registry_lost',[(False,False),(True,False),(True,True)])
def test_failed_return_before_successor_publication_never_queues_destroyed_work(handoff_host,monkeypatch,stop_failure,registry_lost,early_resume):
    tid,root,scheduler,stopped,_notifications,faults=handoff_host
    from tests.test_browser_lifecycle_v12 import _ORIGINAL_DISPATCH_PENDING
    monkeypatch.setattr(threads,'_dispatch_pending_after',_ORIGINAL_DISPATCH_PENDING)
    original_gate=browser.BrowserManager.bounded_thread_gate
    depth=0
    armed=False
    fail_pause=False
    returned=[]
    failed_sessions=[]
    follower=[]
    prior_work_ids=set()
    def control(argv,**_kwargs):
        if fail_pause and argv[1]=='pause':
            raise browser.BrowserUnavailable('Return pause outcome unconfirmed')
        if argv[1]=='inspect':return b'false\n'
        return b''
    monkeypatch.setattr(browser,'_bounded_cli',control)
    def fail_return(session):
        nonlocal fail_pause
        assert {r.id for r in threads._runs().list(tid) if r.work_id==owner.work_id}==prior_work_ids
        if not session.human_active:
            session.request_human(session.viewer_token,'a'*32)
        assert session.human_active and not session.paused
        fail_pause=True
        faults['stop']=stop_failure
        with pytest.raises(browser.BrowserUnavailable) as error:
            session.return_control(session.viewer_token,'a'*32)
        returned.append(str(error.value))
        failed_sessions.append(session)
        if registry_lost:
            browser.BrowserManager._sessions.pop(tid)
            browser.SandboxManager._containers.pop(str(root/tid))
    @contextmanager
    def failed_return_between_handlers(t):
        nonlocal depth,armed
        with original_gate(t):
            depth+=1
            try:yield
            finally:depth-=1
        session=browser.BrowserManager.current_session(t)
        if armed and depth==0 and session and session.paused:
            armed=False
            fail_return(session)
    monkeypatch.setattr(browser.BrowserManager,'bounded_thread_gate',failed_return_between_handlers)
    class Chat:
        agent=None
        on_queue_state=None
        pending_reply=lambda self:None
        get_messages=lambda self:[]
        get_raw_messages=get_messages
        def message(self,text):
            nonlocal armed
            if follower:return 'Follower completed'
            session=browser.BrowserManager.current_session(tid)
            session.command('open',url='https://example.test/')
            follower.append(threads._create_run(tid,'Accepted follower',origin='continuation'))
            armed=not early_resume
            raise ThreadPauseRequested('fairness pause before handoff publication')
    monkeypatch.setattr(web.MANAGER,'get',lambda *_a,**_k:Chat())
    owner=threads._create_run(tid,'Browser work',user_origin=True)
    prior_work_ids.add(owner.id)
    target=owner
    if early_resume:
        threads._execute_run(owner.id,tid)
        target=next(r for r in threads._runs().list(tid) if r.resume)
        prior_work_ids.add(target.id)
        while not scheduler._q.empty():scheduler._q.get_nowait()
        original_status=threads._set_status
        takeover_once=[True]
        def takeover_before_resume(t,stage,**kwargs):
            original_status(t,stage,**kwargs)
            if stage=='starting_sandbox' and takeover_once[0]:
                takeover_once[0]=False
                session=browser.BrowserManager.current_session(t)
                session.request_human(session.viewer_token,'a'*32)
        monkeypatch.setattr(threads,'_set_status',takeover_before_resume)
        original_pop=threads.THREAD_QUEUE.pop_hold
        def failed_return_after_capture(handle):
            nonlocal armed
            session=browser.BrowserManager.current_session(tid)
            if session and session.human_active and not returned:
                # The early resume branch has unwound; Return runs before its
                # outer pause handler can publish another successor.
                fail_return(session)
            return original_pop(handle)
        monkeypatch.setattr(threads.THREAD_QUEUE,'pop_hold',failed_return_after_capture)
    if stop_failure:
        with pytest.raises(browser.BrowserUnavailable,match='exact stop unconfirmed'):
            threads._execute_run(target.id,tid)
        assert threads._runs().get(tid,target.id).status=='running'
        session=failed_sessions[0]
        assert session.closed and session.identity and session.viewer_error
        assert threads._runs().get(tid,follower[0].id).status=='pending'
        with authority.fence(str(root),tid) as state:
            assert state.lease['generations']==['host-'+tid]
        assert scheduler._q.empty()
    else:
        threads._execute_run(target.id,tid)
        assert threads._runs().get(tid,target.id).status=='error'
        assert browser.BrowserManager.current_session(tid) is None
        assert stopped=={'host-'+tid:1}
        queued=scheduler._q.get_nowait()
        assert queued['run_id']==follower[0].id
        fail_pause=False
        threads._execute_run(queued['run_id'],queued['tid'])
        assert threads._runs().get(tid,follower[0].id).status=='success'
    assert returned
    assert {r.id for r in threads._runs().list(tid) if r.work_id==owner.work_id}==prior_work_ids


@pytest.mark.parametrize('path',['parent','child'])
@pytest.mark.parametrize('takeover',['before','during','none'])
def test_takeover_and_fairness_pause_preserve_usable_controller(handoff_host,monkeypatch,path,takeover):
    tid,root,scheduler,stopped,_notifications,_faults=handoff_host
    sessions=[]
    controls=[]
    container_paused=False
    requested,attempting=Event(),Event()
    errors=[]
    requester=None
    original_gate=browser.BrowserManager.bounded_thread_gate
    @contextmanager
    def observed_gate(t):
        if current_thread() is requester:attempting.set()
        with original_gate(t):yield
    monkeypatch.setattr(browser.BrowserManager,'bounded_thread_gate',observed_gate)
    def control(argv,**_kwargs):
        nonlocal container_paused
        controls.append(argv[1])
        if argv[1]=='pause':container_paused=True
        elif argv[1]=='unpause':container_paused=False
        return b''
    monkeypatch.setattr(browser,'_bounded_cli',control)
    def transport(*_a,**_k):
        assert not container_paused, 'viewer transport reached a frozen container'
        return json.dumps({'result':{'pages':[{'page_id':'page','title':'Synthetic form'}]}}).encode()
    monkeypatch.setattr(browser,'_docker_exec',transport)
    class Chat:
        agent=None
        on_queue_state=None
        pending_reply=lambda self:None
        get_messages=lambda self:[]
        get_raw_messages=get_messages
        def __init__(self,t):self.thread_id=t
        def message(self,_text):
            session=browser.BrowserManager.current_session(tid)
            if self.thread_id==tid:
                session.command('open',url='https://example.test/')
                sessions.append(session)
            if takeover=='before' and (path=='parent' or self.thread_id!=tid):
                session.request_human(session.viewer_token,'a'*32)
            raise ThreadPauseRequested('fairness boundary')
        def resume(self):return 'same execution resumed'
    monkeypatch.setattr(web.MANAGER,'get',lambda t,**_k:Chat(t))
    owner=threads._create_run(tid,'Synthetic browser work',user_origin=True)
    original_pause=browser.BrowserSession.pause
    armed=False
    def request_control():
        try:sessions[0].request_human(sessions[0].viewer_token,'a'*32)
        except Exception as error:errors.append(error)
        finally:requested.set()
    def pause_at_request(session):
        nonlocal armed,requester
        if armed:
            armed=False
            requester=Thread(target=request_control,daemon=True)
            requester.start()
            assert attempting.wait(3)
            # Force the old check/pause gap when it is unlocked. With an outer
            # gate, the competing request instead waits until pause is complete.
            if not browser.BrowserManager.thread_gate(tid)._is_owned():
                assert requested.wait(3)
        original_pause(session)
    monkeypatch.setattr(browser.BrowserSession,'pause',pause_at_request)
    if path=='parent':
        armed=takeover=='during'
        threads._execute_run(owner.id,tid)
        pending=next(r for r in threads._runs().list(tid) if r.status=='pending')
    else:
        threads._execute_run(owner.id,tid)
        session=sessions[0]
        assert session.paused and not session.human_requested
        child_tid='sub-race'
        (root/child_tid).mkdir()
        (root/child_tid/'.subagent').touch()
        child=threads._runs().create(child_tid,'delegate-agent','Synthetic child work',mode='child',
            parent_thread_id=tid,parent_run_id=owner.id,dispatch_key='race-child')
        armed=takeover=='during'
        threads._execute_child_run(child)
        pending=next(r for r in threads._runs().list(child_tid) if r.status=='pending')
    if requester:
        requester.join(3)
        assert not requester.is_alive() and not errors
    session=sessions[0]
    if takeover=='none':
        assert session.paused and container_paused and not session.human_active
    else:
        assert session.human_active and not session.paused and not container_paused
        assert session.viewer_command(session.viewer_token,'a'*32,'tabs')['result']['pages']
        # A queued successor must remain pending while the usable viewer owns it.
        if path=='parent':threads._execute_run(pending.id,pending.thread_id)
        else:threads._execute_child_run(pending)
        assert threads._runs().get(pending.thread_id,pending.id).status=='pending'
        session.return_control(session.viewer_token,'a'*32)
        assert session.paused and container_paused and not session.human_active
    if path=='parent':threads._execute_run(pending.id,pending.thread_id)
    else:threads._execute_child_run(pending,resume=True)
    assert threads._runs().get(pending.thread_id,pending.id).status=='success'
    if path=='parent':assert stopped=={'host-'+tid:1}
    else:assert session.paused and not session.closed


@pytest.mark.skipif(not os.getenv("ASSIST_BROWSER_HANDOFF_CHROMIUM"),
                    reason="explicit disposable local Chromium acceptance")
def test_same_page_human_edit_handback_and_competing_run(wired, monkeypatch):
    tid, _ = wired
    root = Path(web.MANAGER.root_dir)
    other = "competitor"
    (root / other).mkdir()
    authority.mark_new_thread(str(root), other)
    queue = ThreadAffinityQueue()
    scheduler = threads._ResumeScheduler()
    monkeypatch.setattr(threads, "THREAD_QUEUE", queue)
    monkeypatch.setattr(threads, "_RESUME_SCHEDULER", scheduler)
    monkeypatch.setattr(browser.BrowserManager, "_sessions", {})
    monkeypatch.setattr(browser.BrowserManager, "_reconciled_roots", {str(root)})
    monkeypatch.setattr(browser.BrowserManager, "reconcile_startup", lambda *_a: True)
    monkeypatch.setattr(browser.BrowserManager, "ready_for_browser", lambda *_a: True)
    monkeypatch.setattr(browser, "configured_directory", lambda: str(root))
    monkeypatch.setattr(browser, "browser_records", lambda *_a: {})
    monkeypatch.setattr(threads, "_require_deep_thread", lambda *_a: None)
    submissions = []

    class Fixture(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith('/result'):
                submissions.append(self.path)
                content = b'<h1>Submitted once</h1>'
            elif self.path == '/second':
                content = b'<h1>Second tab</h1>'
            else:
                content = b'''<form action="/result"><input name="query" placeholder="Artist"><button>Submit</button></form>
                <script>window.heap='live-heap';sessionStorage.setItem('canary','live-tab');</script>'''
            self.send_response(200)
            self.send_header('Content-Type','text/html')
            self.send_header('Set-Cookie','synthetic_login=yes; HttpOnly; SameSite=Strict')
            self.end_headers()
            self.wfile.write(content)
        def log_message(self, *_a):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Fixture)
    Thread(target=server.serve_forever, daemon=True).start()
    origin = f'http://127.0.0.1:{server.server_port}'
    executor = ThreadPoolExecutor(max_workers=1)

    def bootstrap():
        from playwright.sync_api import sync_playwright
        worker = BrowserWorker()
        worker.playwright = sync_playwright().start()
        worker.browser = worker.playwright.chromium.launch(
            executable_path=os.environ['ASSIST_BROWSER_HANDOFF_CHROMIUM'], headless=True,
            args=['--no-sandbox'])
        worker.context = worker.browser.new_context(viewport={'width':390,'height':600})
        worker.context.on('page', worker._register_page)
        return worker

    worker = executor.submit(bootstrap).result(timeout=15)
    containers, controls = {}, []
    monkeypatch.setattr(browser.SandboxManager, "_containers", containers)
    monkeypatch.setattr(browser, "_bounded_cli", lambda argv, **_k: controls.append(argv) or b'')
    monkeypatch.setattr(browser, "disarm_sandbox_browser", lambda *_a: None)
    monkeypatch.setattr(browser, "forget_client", lambda *_a: None)
    monkeypatch.setattr(browser, "_kill_container_confirmed", lambda *_a: None)

    def backend(thread_id, **kwargs):
        container = SimpleNamespace(id='synthetic-'+thread_id)
        containers[str(root/thread_id)] = container
        run = threads._runs().get(thread_id, kwargs['owner_run_id'])
        with authority.fence(str(root),thread_id) as state:
            state.begin(run.id,run.admission_sequence)
            state.add_generation(run.id,container.id)
        return SimpleNamespace(container=container)

    def cleanup(work_dir, expected=None):
        container = containers.pop(work_dir,None)
        if container:
            with authority.fence(str(root),Path(work_dir).name) as state:
                owner=state.lease['owner_run_id']
                if container.id in state.lease['generations']:
                    state.remove_generation(owner,container.id)
                state.clear(owner)

    def start(session):
        session.identity=browser._ContainerIdentity(str(root),'172.20.0.2',session.sandbox_generation)

    def transport(_generation,payload,**_kwargs):
        command=json.loads(payload)
        def execute():
            try:
                operation=command['operation']
                return {'result': getattr(worker,operation)(**command.get('args',{}))}
            except Exception as error:
                return {'error':str(error)}
        return json.dumps(executor.submit(execute).result(timeout=15)).encode()

    monkeypatch.setattr(threads,'_get_sandbox_backend',backend)
    monkeypatch.setattr(browser.SandboxManager,'cleanup',cleanup)
    monkeypatch.setattr(browser.BrowserSession,'_start',start)
    monkeypatch.setattr(browser.BrowserSession,'_capture_storage',lambda _s:None)
    monkeypatch.setattr(browser,'_docker_exec',transport)
    opened,boundary,done=Event(),Event(),Event()
    observations={}

    class Chat:
        agent=None
        on_queue_state=None
        pending_reply=lambda self:None
        get_messages=lambda self:[]
        get_raw_messages=get_messages
        def __init__(self, thread_id, tools):
            self.thread_id=thread_id
            self.tools={tool.__name__:tool for tool in tools}
        def invoke(self,name,**args):
            return json.loads(self.tools[name](**args))
        def message(self,text):
            if self.thread_id==other:
                return 'unrelated thread progressed'
            observations.update(self.invoke('browser_open',url=origin+'/')['result'])
            observations.update(self.invoke('browser_act',page_id=observations['page_id'],
                snapshot_id=observations['snapshot_id'],action='fill',
                target={'role':'textbox','name':'Artist','text':'Agent draft'})['result'])
            self.invoke('browser_open',url=origin+'/second')
            opened.set()
            assert boundary.wait(5)
            ThreadQueueMiddleware().after_model(None,None)
            raise AssertionError('takeover did not request cooperative pause')
        def resume(self):
            old=dict(observations)
            assert 'error' in self.invoke('browser_open',url=origin+'/')
            assert 'error' in self.invoke('browser_act',page_id=old['page_id'],snapshot_id=old['snapshot_id'],
                action='fill',target={'role':'textbox','name':'Artist','text':'stale overwrite'})
            fresh=self.invoke('browser_observe',page_id=old['page_id'])['result']
            assert 'Agent draft + human edit' in fresh['snapshot']
            # Even an old batch containing observe then navigation cannot mutate.
            assert 'error' in self.invoke('browser_open',url=origin+'/')
            ThreadQueueMiddleware().after_model(None,None)
            self.invoke('browser_act',page_id=fresh['page_id'],snapshot_id=fresh['snapshot_id'],
                action='click',target={'role':'button','name':'Submit'})
            return 'submitted once after fresh observation'

    monkeypatch.setattr(web.MANAGER,'get',lambda t,**kwargs:Chat(t,kwargs.get('browser_tools',())))
    first=threads._create_run(tid,'Work on the synthetic form',user_origin=True)
    def run_first():
        try:threads._execute_run(first.id,tid)
        finally:done.set()
    process=Thread(target=run_first,daemon=True)
    process.start()
    client=TestClient(web.app)
    identity='a'*32
    try:
        assert opened.wait(15)
        session=browser.BrowserManager.current_session(tid)
        token=session.viewer_token
        headers={'Origin':'http://testserver','X-Browser-Token':token}
        def control(action,**args):
            if action not in {'status','takeover','return'}:
                session._viewer_at=0  # This test exercises ownership, not wall-clock rate limiting.
            return client.post(f'/thread/{tid}/browser/control',headers=headers,
                json={'client':identity,'action':action,**args})
        assert control('takeover').status_code==200
        # Admission can own authority while takeover unwinds. The handoff must
        # wait there without holding the global admission lock in reverse order.
        original_fence=authority.fence
        attempting_authority=Event()
        @contextmanager
        def observed_fence(*args,**kwargs):
            if current_thread() is process and boundary.is_set():
                attempting_authority.set()
            with original_fence(*args,**kwargs) as state:
                yield state
        monkeypatch.setattr(authority,'fence',observed_fence)
        with original_fence(str(root),tid):
            boundary.set()
            assert attempting_authority.wait(2)
            acquired=threads._RUN_ADMISSION_LOCK.acquire(timeout=0.5)
            assert acquired
            threads._RUN_ADMISSION_LOCK.release()
        assert done.wait(5)
        assert session.human_active and not session.paused
        with pytest.raises(browser.BrowserUnavailable,match='human viewer'):
            session.command('open',url=origin+'/')
        competitor=threads._create_run(other,'Quick unrelated work',user_origin=True)
        threads._execute_run(competitor.id,other)
        assert threads._runs().get(other,competitor.id).status=='success'
        assert queue.peek_holder() is None
        pages=control('tabs').json()['pages']
        assert len(pages)==2
        page_id=observations['page_id']
        frame=control('frame',page_id=page_id)
        assert frame.status_code==200 and frame.content.startswith(b'\xff\xd8')
        firefox_path=os.getenv('ASSIST_BROWSER_HANDOFF_FIREFOX')
        if firefox_path:
            delay_frame,frame_waiting,release_frame=Event(),Event(),Event()
            viewer_inputs=[]
            class ViewerBridge(BaseHTTPRequestHandler):
                def handle_request(self):
                    payload=self.rfile.read(int(self.headers.get('Content-Length','0')))
                    command=json.loads(payload) if payload else {}
                    if command.get('action') in {'click','text','key'}:
                        viewer_inputs.append(command)
                    response=client.request(self.command,
                        f'http://127.0.0.1:{self.server.server_port}'+self.path,
                        headers=dict(self.headers),
                        content=payload)
                    if delay_frame.is_set() and command.get('action')=='frame' and command.get('page_id')==page_id:
                        frame_waiting.set()
                        assert release_frame.wait(5)
                    self.send_response(response.status_code)
                    for name,value in response.headers.items():
                        self.send_header(name,value)
                    self.end_headers()
                    self.wfile.write(response.content)
                do_GET=handle_request
                do_POST=handle_request
                def log_message(self,*_a):pass
            bridge=ThreadingHTTPServer(('127.0.0.1',0),ViewerBridge)
            Thread(target=bridge.serve_forever,daemon=True).start()
            from playwright.sync_api import sync_playwright, expect
            try:
                with sync_playwright() as playwright:
                    firefox=playwright.firefox.launch(executable_path=firefox_path,headless=True)
                    context=firefox.new_context(viewport={'width':390,'height':844})
                    context.add_init_script(f"sessionStorage.setItem('assist-viewer:/thread/{tid}/browser','{identity}')")
                    viewer=context.new_page()
                    viewer.goto(f'http://127.0.0.1:{bridge.server_port}/thread/{tid}/browser')
                    expect(viewer.locator('#screen')).to_have_attribute('src',re.compile('^blob:'))
                    expect(viewer.locator('#screen')).to_have_js_property('complete',True)
                    assert viewer.locator('#tabs option').count()==2
                    expect(viewer.locator('#screen')).to_have_attribute('data-page',page_id)
                    delay_frame.set()
                    viewer.locator('#refresh').click()
                    assert frame_waiting.wait(3)
                    viewer.locator('#tabs').select_option(pages[1]['page_id'])
                    viewer.locator('#typing').fill('must not reach wrong tab')
                    viewer.locator('#type').click()
                    assert viewer_inputs==[]
                    delay_frame.clear()
                    release_frame.set()
                    expect(viewer.locator('#screen')).to_have_attribute('data-page',pages[1]['page_id'])
                    viewer.locator('#tabs').select_option(page_id)
                    expect(viewer.locator('#screen')).to_have_attribute('data-page',page_id)
                    # Actual Firefox pointer events and text toolbar drive the route.
                    position=executor.submit(lambda:worker.pages[page_id].locator('input').bounding_box()).result()
                    viewer.locator('#screen').click(position={'x':position['x']+position['width']-10,'y':position['y']+8})
                    viewer.locator('#typing').fill(' + human edit')
                    with viewer.expect_response(lambda r:r.request.method=='POST' and r.request.post_data_json.get('action')=='text') as sent:
                        viewer.locator('#type').click()
                    assert sent.value.status==200
                    viewer.reload()
                    expect(viewer.locator('#screen')).to_have_attribute('src',re.compile('^blob:'))
                    expect(viewer.locator('#screen')).to_have_js_property('complete',True)
                    screenshot=root/'viewer-firefox.png'
                    viewer.screenshot(path=str(screenshot))
                    print({'firefox':firefox.version,'viewer_screenshot':str(screenshot),'transport':'real Firefox HTTP pixels/input'})
                    with viewer.expect_response(lambda r:r.request.method=='POST' and r.request.post_data_json.get('action')=='return') as returned:
                        viewer.locator('#return').click()
                    assert returned.value.status==200
                    firefox.close()
            finally:
                release_frame.set()
                bridge.shutdown()
                bridge.server_close()
        else:
            position=executor.submit(lambda:worker.pages[page_id].locator('input').bounding_box()).result()
            assert control('click',page_id=page_id,x=position['x']+position['width']-10,y=position['y']+8).status_code==200
            assert control('text',page_id=page_id,text=' + human edit').status_code==200
        before=executor.submit(lambda:worker.pages[page_id].evaluate(
            '({draft:document.querySelector("input").value,heap:window.heap,tab:sessionStorage.getItem("canary"),cookie:document.cookie})')).result()
        assert before['draft']=='Agent draft + human edit'
        assert before['heap']=='live-heap' and before['tab']=='live-tab'
        assert executor.submit(lambda:worker.context.cookies()).result()[0]['name']=='synthetic_login'
        if not firefox_path:
            assert client.get(f'/thread/{tid}/browser').status_code==200
            assert control('return').status_code==200
        assert control('text',page_id=page_id,text='late').status_code==409
        item=scheduler._q.get_nowait()
        threads._execute_run(item['run_id'],item['tid'])
        assert threads._runs().get(tid,item['run_id']).status=='success'
        assert len(submissions)==1 and 'human+edit' in submissions[0]
        assert controls==[['docker','pause','synthetic-'+tid],['docker','unpause','synthetic-'+tid]]
    finally:
        boundary.set()
        process.join(5)
        executor.submit(worker.revoke).result(timeout=10)
        executor.shutdown(wait=True)
        server.shutdown()
        server.server_close()
