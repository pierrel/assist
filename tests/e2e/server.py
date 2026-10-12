"""Real Assist fixture; only graph/model inference is deterministic."""
from dataclasses import asdict
import os
from pathlib import Path
import shlex
import threading
import uuid

from docker.errors import NotFound
from fastapi import APIRouter, Request
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, MessagesState, START, END

import assist.thread as thread_module
import assist.thread_manager as manager_module
from assist.browser import authority
from assist.sandbox_manager import SandboxManager
from manage.web import app, state, threads
from manage.web.phone_api import _authenticate
from manage.web.protocol_service import SERVICE

HOLDS = {}
ROUTER = APIRouter()


def deterministic_agent(model, working_dir, checkpointer=None, sandbox_backend=None, **kwargs):
    """Replace inference while retaining the supplied real backend/checkpointer."""
    def output(values):
        text = values['messages'][-1].content
        if text.startswith('Hold '):
            hold = HOLDS.setdefault(text, threading.Event())
            try:
                if not hold.wait(120):
                    raise TimeoutError('Fixture model hold exceeded 120 seconds')
            finally:
                if HOLDS.get(text) is hold:
                    del HOLDS[text]
        if text == 'Hold publication':
            responses = sandbox_backend.upload_files([('/workspace/server-during.txt', b'server-race-result')])
            if responses[0].error:
                raise RuntimeError(responses[0].error)
        if text.startswith('Write '):
            filename, content = text[6:].split(' ', 1)
            responses = sandbox_backend.upload_files([('/workspace/' + filename, content.encode())])
            if responses[0].error:
                raise RuntimeError(responses[0].error)
            result = 'Saved ' + filename
        elif text.startswith('Read '):
            response = sandbox_backend.execute('cat -- ' + shlex.quote('/workspace/' + text[5:]))
            if response.exit_code != 0:
                raise RuntimeError(response.output)
            result = response.output
        else:
            result = 'Completed: ' + text
        return {'messages': [AIMessage(content=result)]}
    graph = StateGraph(MessagesState)
    graph.add_node('model', output)
    graph.add_edge(START, 'model')
    graph.add_edge('model', END)
    return graph.compile(checkpointer=checkpointer)


def install():
    """Install deterministic inference explicitly, without lifecycle patches."""
    state.MANAGER._model = FakeListChatModel(responses=['E2E thread'])
    thread_module.create_agent = deterministic_agent
    manager_module.create_context_agent = deterministic_agent
    manager_module.create_research_agent = deterministic_agent
    threads._llm_reachable = lambda: True
    app.include_router(ROUTER)


@ROUTER.get('/__e2e/state/{tid}')
def inspect_state(tid: str, request: Request):
    _authenticate(request)
    with authority.fence(state.MANAGER.root_dir, tid) as record:
        lease = record.lease
    container = SandboxManager.current_container(os.path.join(state.MANAGER.root_dir, tid, 'domain'))
    generation = None
    if container is not None:
        try:
            container.reload()
        except NotFound:
            pass
        else:
            if container.attrs['State']['Running']:
                generation = container.id
    with threads._BROWSER_RETRY_LOCK:
        with threads._BROWSER_RESET_QUEUE.mutex:
            reset_idle = (threads._BROWSER_RESET_QUEUE.unfinished_tasks == 0
                          and tid not in threads._BROWSER_QUEUED
                          and tid not in threads._BROWSER_REQUEUE
                          and tid not in threads._BROWSER_RETRY_DUE)
    return {'runs': [asdict(r) for r in threads._runs().list(tid)], 'holds': list(HOLDS),
            'lease': lease, 'live_registered_generation': generation,
            'reset_worker_idle': reset_idle}


@ROUTER.post('/__e2e/release')
def release(body: dict, request: Request):
    _authenticate(request)
    HOLDS[body['text']].set()
    return {'released': True}


@ROUTER.post('/__e2e/child/{tid}')
def child(tid: str, request: Request):
    _authenticate(request)
    parent = threads._runs().list(tid)[-1]
    child_id = 'sub-' + uuid.uuid4().hex
    metadata = {'parent_thread_id': tid, 'parent_run_id': parent.id,
                'dispatch_key': parent.work_id + ':e2e-child'}
    SERVICE.create_thread(child_id, metadata)
    run = SERVICE.create_run(child_id, 'context-agent', 'Hold child', metadata=metadata)
    return {'thread_id': child_id, 'run_id': run.id}


if __name__ == '__main__':
    os.environ['ASSIST_PHONE_API_TOKEN'] = Path('/secrets/token').read_text().strip()
    install()
    from manage.web.__main__ import TurnSafeServer
    import uvicorn
    TurnSafeServer(uvicorn.Config(app, host='0.0.0.0', port=8443,
        ssl_certfile='/secrets/server.crt', ssl_keyfile='/secrets/server.key',
        log_level='warning', access_log=False)).run()
