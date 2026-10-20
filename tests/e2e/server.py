"""Real Assist fixture with synthetic inference, mailbox previews and effects."""
from dataclasses import asdict
import os
from pathlib import Path
import shlex
import threading
import uuid

from docker.errors import NotFound
from fastapi import APIRouter, HTTPException, Request
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.config import get_config
from langgraph.types import interrupt

import assist.thread as thread_module
import assist.thread_manager as manager_module
from assist.browser import authority
from assist.sandbox_manager import SandboxManager
from manage.web import app, egress, geo, state, threads
from manage.web.phone_api import _authenticate
from manage.web.protocol_service import SERVICE

HOLDS = {}
EFFECTS = []
ROUTER = APIRouter()


def deterministic_agent(model, working_dir, checkpointer=None, sandbox_backend=None, **kwargs):
    """Replace inference while retaining the supplied real backend/checkpointer."""
    def output(values):
        text = values['messages'][-1].content
        if 'Synthetic approval:' in text:
            kind = text.rsplit('Synthetic approval:', 1)[1].split()[0]
            if kind == 'send_email':
                args = {'to': 'reader@example.test', 'subject': 'Synthetic review', 'body': 'Complete synthetic body'}
            elif kind in {'email_archive', 'email_delete'}:
                args = {'message_ids': ['synthetic-message']}
            elif kind == 'send_reply':
                args = {'text': 'Complete synthetic SMS reply'}
            else:
                raise ValueError('Unknown synthetic approval kind')
            decision = interrupt({'action_requests': [{'name': kind, 'args': args}]})['decisions'][0]
            if decision['type'] != 'reject':
                EFFECTS.append({'thread_id': get_config()['configurable']['thread_id'], 'kind': kind,
                                'args': decision.get('edited_action', {'args': args})['args']})
            return {'messages': [AIMessage(content='Synthetic ' + kind + ' decision complete')]}
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
    # These services own fixture-only files; provider/download boundaries are synthetic.
    from assist.egress.store import EgressStore
    from assist.geo.proposals import ProposalStore
    from assist.geo.registry import RegionRegistry
    from assist.geo.provisioner import Provisioner
    root = Path(state.MANAGER.root_dir).parent / 'synthetic-hitl'
    root.mkdir(exist_ok=True)
    state.EGRESS_STORE = threads.EGRESS_STORE = egress.EGRESS_STORE = EgressStore(str(root / 'egress'))
    proposals, registry = ProposalStore(str(root)), RegionRegistry(str(root))
    state.GEO_PROPOSALS = threads.GEO_PROPOSALS = geo.GEO_PROPOSALS = proposals
    state.GEO_REGISTRY = threads.GEO_REGISTRY = geo.GEO_REGISTRY = registry
    provisioner = Provisioner(registry, proposals,
        lambda op, slug: EFFECTS.append({'kind': 'geo_download', 'slug': slug}) or True,
        lambda *args: None, lambda: False)
    threads._PROVISIONER = geo._PROVISIONER = provisioner
    threads.gmail_action_preview = lambda action: [{'id': 'synthetic-message', 'from': 'sender@example.test',
        'to': 'reader@example.test', 'date': '2026-10-20', 'subject': 'Synthetic message',
        'body': 'Complete synthetic mailbox body'}]
    os.environ['EMAIL_FROM_ADDRESS'] = 'assistant@example.test'
    os.environ['EMAIL_FROM_NAME'] = 'Synthetic assistant'
    os.environ['EMAIL_ALWAYS_CC'] = 'oversight@example.test'
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
            'synthetic_effects': list(EFFECTS),
            'queue_holder': threads.THREAD_QUEUE.peek_holder(),
            'lease': lease, 'live_registered_generation': generation,
            'reset_worker_idle': reset_idle}


@ROUTER.post('/__e2e/approval/{tid}')
def propose_approval(tid: str, body: dict, request: Request):
    """Explicit capability setup; never a model-behavior or provider acceptance probe."""
    _authenticate(request)
    threads._existing_thread_dir(tid)
    kind = body.get('kind')
    if kind == 'egress':
        from assist.egress.store import EgressRequest
        from datetime import datetime, timezone
        rec = EgressRequest('synthetic-' + uuid.uuid4().hex[:12] + '.example.test', 443, 'Read synthetic fixture data', tid,
                            created_at=datetime.now(timezone.utc).isoformat())
        if egress.EGRESS_STORE.add_pending(rec) is not None:
            raise HTTPException(409, 'Synthetic request already exists')
        return {'kind': kind, 'status': 'pending',
                'expected_preview': [f'Network access: {rec.host}:{rec.port}\n', 'For this thread only.\n',
                                     f"Agent's stated reason (not verified):\n{rec.task}\n",
                                     'Allow for 1 hour', 'Allow for this thread', 'Decline']}
    if kind == 'geo_download':
        from assist.geo.proposals import Proposal
        from datetime import datetime, timezone
        from assist.geo.tools import DEGRADATION_WARNING
        proposal = Proposal('test/synthetic-' + uuid.uuid4().hex[:12], 'Synthetic region', (0, 0, 1, 1),
            1200, tid, 'Plan a synthetic route', datetime.now(timezone.utc).isoformat())
        geo.GEO_PROPOSALS.put(proposal)
        return {'kind': kind, 'status': 'pending',
                'expected_preview': ['Download: Synthetic region\n', f'Region: {proposal.slug}\n',
                                     'Size: 1200 bytes\n', DEGRADATION_WARNING + '\n', 'Approve download', 'Decline']}
    if kind not in {'send_email', 'email_archive', 'email_delete', 'send_reply'}:
        raise HTTPException(422, 'Unsupported synthetic kind')
    run = threads._create_run(tid, 'Synthetic approval:' + kind,
                              sender='+15555550100' if kind == 'send_reply' else None)
    threads._RESUME_SCHEDULER.submit(run.id, tid)
    return {'kind': kind, 'run_id': run.id}


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
