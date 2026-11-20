"""Execute actual tool and HTTP client gates without a provider or Docker."""
import asyncio
import socket

import httpx
import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode

from edd.security_eval import Evidence
from edd.security_execution import Budget, ExecutionStopped, _PROVIDER, instrument


@pytest.fixture(autouse=True)
def existing_control_boundary_runtime():
    # Main can carry the tests independently; execute the engine gates only on
    # a runtime that includes the existing control-service boundary.
    from pathlib import Path
    if not (Path(__file__).parents[1] / "assist/control_api.py").is_file():
        pytest.skip("Existing control boundary runtime required in this exact tree")
    pytest.importorskip("assist.control_api")


@pytest.mark.parametrize("asynchronous", [False, True])
def test_actual_tool_retry_cannot_cross_shared_step_limit(tmp_path, asynchronous):
    effects = []
    @tool
    def effect() -> str:
        """Record a synthetic effect."""
        effects.append('once')
        return 'done'
    def retry(request, execute):
        execute(request)
        return execute(request)
    from langgraph.graph import StateGraph, MessagesState, END
    graph = StateGraph(MessagesState)
    async def async_retry(request, execute):
        await execute(request)
        return await execute(request)
    graph.add_node("tools", ToolNode([effect], **({"awrap_tool_call": async_retry}
        if asynchronous else {"wrap_tool_call": retry})))
    graph.set_entry_point("tools")
    graph.add_edge("tools", END)
    node = graph.compile()
    evidence = Evidence(tmp_path)
    budget = Budget(evidence, 1, 10)
    with instrument(budget, 'http://127.0.0.1:8000/v1', '172.22.0.2'):
        with pytest.raises(ExecutionStopped):
            value = {'messages': [AIMessage(content='', tool_calls=[
                {'name': 'effect', 'args': {}, 'id': 'synthetic-1'}])]}
            if asynchronous:
                asyncio.run(node.ainvoke(value))
            else:
                node.invoke(value)
    assert effects == ['once'] and budget.reason == 'step_limit'


def test_sync_async_provider_attempts_share_budget(tmp_path):
    evidence = Evidence(tmp_path)
    budget = Budget(evidence, 2, 10)
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={'ok': True}))
    with instrument(budget, 'http://127.0.0.1:8000/v1', '172.22.0.2'):
        token = _PROVIDER.set(True)
        try:
            with httpx.Client(transport=transport) as client:
                assert client.post('http://127.0.0.1:8000/v1/chat/completions').status_code == 200
            async def call():
                async with httpx.AsyncClient(transport=transport) as client:
                    return await client.post('http://127.0.0.1:8000/v1/chat/completions')
            assert asyncio.run(call()).status_code == 200
            with httpx.Client(transport=transport) as client, pytest.raises(ExecutionStopped):
                client.post('http://127.0.0.1:8000/v1/chat/completions')
        finally:
            _PROVIDER.reset(token)
    assert budget.count == 2


def test_real_control_checks_precede_host_transport_containment(tmp_path):
    from assist import tools
    from assist.control_api import ControlApiDenied
    evidence = Evidence(tmp_path)
    budget = Budget(evidence, 2, 10)
    from unittest.mock import patch
    with patch.dict('os.environ', {'ASSIST_PORT': '9090',
            'ASSIST_BASE_URL': 'http://control.security-eval.test:9090'}):
        with instrument(budget, 'http://127.0.0.1:8000/v1', '172.22.0.2'):
            assert 'Error fetching URL' in tools.read_url('http://control.security-eval.test:9090/')
            with pytest.raises(ControlApiDenied):
                tools._read_pdf_url('http://control.security-eval.test:9090/a.pdf')
            with pytest.raises(OSError):
                socket.getaddrinfo('unknown.security-eval.test', 80)
            with socket.socket() as connection, pytest.raises(OSError):
                connection.connect(('127.0.0.1', 8000))
    rows = evidence.path.read_text()
    assert rows.count('product_control_denial') == 2
    assert 'host_transport_unavailable' in rows


def test_expired_budget_rejects_before_effect(tmp_path):
    budget = Budget(Evidence(tmp_path), 2, 1)
    budget.deadline = 0
    with pytest.raises(ExecutionStopped):
        budget.enter('tool', 'never')
    assert budget.count == 0 and budget.reason == 'wall_limit'


def test_native_denied_boundary_waits_for_confirmed_teardown(tmp_path):
    import threading
    from edd.security_execution import instrument
    from assist.hermes_provider_relay import HermesProviderRelay
    directory = tmp_path / 'relay'
    directory.mkdir()
    relay = HermesProviderRelay(directory, 'http://127.0.0.1:8000/v1',
                                'EMPTY', 'synthetic', 8192, lambda: True)
    budget = Budget(Evidence(tmp_path / 'oracle'), 1, 10)
    budget.enter('model', 'already used')
    exited = threading.Event()
    def tool_start():
        try:
            relay.boundary({'operation': 'synthetic-tool', 'kind': 'tool', 'tool': 'terminal'})
        except ExecutionStopped:
            exited.set()
    try:
        with instrument(budget, 'http://127.0.0.1:8000/v1', '172.22.0.2'):
            worker = threading.Thread(target=tool_start)
            worker.start()
            with budget.native_condition:
                assert budget.native_condition.wait_for(lambda: budget.native_active == 1, timeout=2)
                assert not exited.is_set() and not relay.operations
                budget.native_cleanup_complete = True
                budget.native_condition.notify_all()
            worker.join(2)
            assert not worker.is_alive() and exited.is_set() and budget.native_active == 0
    finally:
        relay.close()


def test_model_connection_failures_cannot_become_real_pass_at_limit(tmp_path):
    from edd.security_eval import verify
    evidence = Evidence(tmp_path)
    budget = Budget(evidence, 1, 10)
    def refused(request):
        raise httpx.ConnectError('Synthetic connection refusal', request=request)
    with instrument(budget, 'http://127.0.0.1:8000/v1', '172.22.0.2'):
        token = _PROVIDER.set(True)
        try:
            with httpx.Client(transport=httpx.MockTransport(refused)) as client:
                with pytest.raises(httpx.ConnectError):
                    client.post('http://127.0.0.1:8000/v1/chat/completions')
                with pytest.raises(ExecutionStopped):
                    client.post('http://127.0.0.1:8000/v1/chat/completions')
        finally:
            _PROVIDER.reset(token)
    assert 'provider_error' in evidence.path.read_text()
    evidence.record('isolation_verified')
    evidence.record('execution_complete', errors=[])
    evidence.close(termination='step_limit', execution_kind='real', drained=True)
    result = verify(tmp_path)
    assert result['outcome'] == 'inconclusive' and result['real_agent'] is False


def test_actual_chatopenai_stream_iterator_keeps_provider_context(tmp_path):
    import json
    from langchain_openai import ChatOpenAI
    evidence = Evidence(tmp_path)
    budget = Budget(evidence, 2, 10)
    payload = {'id': 'synthetic-response', 'object': 'chat.completion.chunk', 'created': 0,
        'model': 'synthetic', 'choices': [{'index': 0,
            'delta': {'role': 'assistant', 'content': 'Synthetic reply.'}, 'finish_reason': 'stop'}]}
    transport = httpx.MockTransport(lambda request: httpx.Response(200,
        headers={'Content-Type': 'text/event-stream'},
        content=('data: ' + json.dumps(payload) + '\n\ndata: [DONE]\n\n').encode()))
    with httpx.Client(transport=transport) as client:
        model = ChatOpenAI(model='synthetic', api_key='EMPTY',
            base_url='http://127.0.0.1:8000/v1', http_client=client,
            streaming=True, max_retries=0)
        with instrument(budget, 'http://127.0.0.1:8000/v1', '172.22.0.2'):
            assert model.invoke('Synthetic prompt.').content == 'Synthetic reply.'
    assert budget.count == 1
    assert 'model_output' in evidence.path.read_text()


def test_failed_product_guard_is_not_masked_by_fixture_transport(tmp_path, monkeypatch):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import threading
    from assist import control_api, tools
    hits = []
    class Protected(BaseHTTPRequestHandler):
        def do_GET(self):
            hits.append(self.path)
            body = b'<html>Protected synthetic listener</html>'
            self.send_response(200)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *_args):
            pass
    listener = ThreadingHTTPServer(('127.0.0.1', 0), Protected)
    worker = threading.Thread(target=listener.serve_forever, daemon=True)
    worker.start()
    monkeypatch.setenv('ASSIST_PORT', str(listener.server_port))
    monkeypatch.setattr(control_api, 'resolve_targets', lambda *_args, **_kw: ('127.0.0.1',))
    budget = Budget(Evidence(tmp_path), 2, 10)
    try:
        with instrument(budget, 'http://127.0.0.1:8000/v1', '127.0.0.1'):
            assert 'Protected synthetic listener' in tools.read_url(
                f'http://control.security-eval.test:{listener.server_port}/')
        assert hits == ['/'], 'Fixture containment masked a missing product denial'
    finally:
        listener.shutdown()
        listener.server_close()
        worker.join(2)


def test_async_control_connect_progress_is_not_an_infrastructure_gap(tmp_path, monkeypatch):
    import errno
    monkeypatch.setenv('ASSIST_PORT', '9090')
    monkeypatch.setattr(socket.socket, 'connect_ex', lambda *_args: errno.EINPROGRESS)
    evidence = Evidence(tmp_path)
    budget = Budget(evidence, 2, 10)
    with instrument(budget, 'http://127.0.0.1:8000/v1', '172.22.0.2'):
        with socket.socket() as transport:
            transport.setblocking(False)
            with pytest.raises(BlockingIOError):
                transport.connect(('172.22.0.2', 9090))
    assert 'capability_gap' not in evidence.path.read_text()


@pytest.mark.parametrize("asynchronous", [False, True])
def test_independent_provider_failure_after_stop_is_observed(tmp_path, asynchronous):
    evidence = Evidence(tmp_path)
    budget = Budget(evidence, 2, 10)
    def failed(request):
        budget.stop('step_limit')
        raise httpx.ConnectError('Independent synthetic failure', request=request)
    with instrument(budget, 'http://127.0.0.1:8000/v1', '172.22.0.2'):
        token = _PROVIDER.set(True)
        try:
            if asynchronous:
                async def call():
                    async with httpx.AsyncClient(transport=httpx.MockTransport(failed)) as client:
                        await client.post('http://127.0.0.1:8000/v1/chat/completions')
                with pytest.raises(httpx.ConnectError):
                    asyncio.run(call())
            else:
                with httpx.Client(transport=httpx.MockTransport(failed)) as client:
                    with pytest.raises(httpx.ConnectError):
                        client.post('http://127.0.0.1:8000/v1/chat/completions')
        finally:
            _PROVIDER.reset(token)
    assert 'provider_error' in evidence.path.read_text()


def test_admitted_creation_drains_without_holding_stop_lock(tmp_path):
    import threading
    budget = Budget(Evidence(tmp_path), 2, 10)
    entered, release = threading.Event(), threading.Event()
    def creation():
        with budget.creation():
            entered.set()
            assert release.wait(2)
    worker = threading.Thread(target=creation)
    worker.start()
    try:
        assert entered.wait(1)
        budget.stop('wall_limit')
        assert budget.reason == 'wall_limit'
        with budget.creation_condition:
            assert budget.creation_active == 1
        with pytest.raises(ExecutionStopped):
            with budget.creation():
                pytest.fail('Creation admitted after stop')
    finally:
        release.set()
        worker.join(2)
    assert not worker.is_alive() and budget.creation_active == 0


@pytest.mark.parametrize("phase", ['create', 'start'])
def test_actual_container_launch_is_stopped_and_owned_while_daemon_blocks(tmp_path, phase):
    import threading
    from types import SimpleNamespace
    budget = Budget(Evidence(tmp_path), 2, 10)
    entered, release = threading.Event(), threading.Event()
    calls, errors, owned, agents = [], [], [], []
    def wait(which):
        calls.append(which)
        if phase == which:
            entered.set()
            assert release.wait(2)
    container = SimpleNamespace(start=lambda: wait('start'), reload=lambda: wait('reload'))
    def create(*args, **kwargs):
        wait('create')
        return container
    def launch():
        try:
            budget.create_container(SimpleNamespace(create=create), 'synthetic', (), {}, owned, agents)
        except ExecutionStopped as error:
            errors.append(error)
    worker = threading.Thread(target=launch)
    worker.start()
    try:
        assert entered.wait(1)
        budget.stop('wall_limit')
        with budget.creation_condition:
            assert budget.creation_active == 1
            assert not budget.creation_condition.wait_for(lambda: budget.creation_active == 0, timeout=.01)
    finally:
        release.set()
        worker.join(2)
    assert not worker.is_alive() and owned == agents == [container]
    assert budget.creation_active == 0
    assert ('start' in calls) == (phase == 'start')
    assert bool(errors) == (phase == 'create')
    with pytest.raises(ExecutionStopped):
        budget.create_container(SimpleNamespace(create=lambda *_a, **_kw: pytest.fail('late create')),
                                'synthetic', (), {}, owned, agents)


def test_expired_creation_never_calls_daemon(tmp_path):
    from types import SimpleNamespace
    budget = Budget(Evidence(tmp_path), 2, 10)
    budget.deadline = 0
    with pytest.raises(ExecutionStopped):
        budget.create_container(SimpleNamespace(create=lambda *_a, **_kw: pytest.fail('expired create')),
                                'synthetic', (), {}, [], [])
    assert budget.reason == 'wall_limit'


@pytest.mark.parametrize('kind', ['tool', 'model'])
def test_stalled_journal_cannot_delay_stop_or_resume_admitted_effect(tmp_path, kind):
    import threading
    evidence = Evidence(tmp_path)
    budget = Budget(evidence, 4, 10)
    entered, release, stopped = threading.Event(), threading.Event(), threading.Event()
    effects, errors = [], []
    original_flush = evidence.stream.flush
    def stalled_flush():
        entered.set()
        assert release.wait(2)
        original_flush()
    evidence.stream.flush = stalled_flush
    left, right = socket.socketpair()
    budget.sockets.add(left)
    def work():
        try:
            budget.enter(kind, 'synthetic')
            effects.append('forbidden after stop')
        except ExecutionStopped:
            errors.append('stopped')
    worker = threading.Thread(target=work)
    stopper = threading.Thread(target=lambda: (budget.stop('wall_limit'), stopped.set()))
    try:
        worker.start()
        assert entered.wait(1)
        stopper.start()
        assert stopped.wait(.2), 'Stop waited for journal I/O'
        right.settimeout(.2)
        assert right.recv(1) == b''
    finally:
        release.set()
        worker.join(2)
        stopper.join(2)
        evidence.stream.flush = original_flush
        left.close()
        right.close()
    assert not worker.is_alive() and not stopper.is_alive()
    assert effects == [] and errors == ['stopped']


@pytest.mark.parametrize('upstream_error', [False, True])
def test_native_response_drain_keeps_host_evidence_without_dead_agent_delivery(tmp_path, monkeypatch, upstream_error):
    import io
    import json
    from types import SimpleNamespace
    from assist.hermes_provider_relay import HermesProviderRelay
    evidence = Evidence(tmp_path)
    budget = Budget(evidence, 4, 10)
    output = io.BytesIO()
    handler = SimpleNamespace(wfile=output)
    def provider(self, target, body):
        target.wfile.write(b'HTTP/1.0 200 OK\r\nContent-Type: text/event-stream\r\n\r\n')
        target.wfile.write(b'data: {"choices":[{"delta":{"content":"before"}}]}\n\n')
        before_stop = output.getvalue()
        budget.stop('wall_limit')
        payload = json.dumps({'choices': [{'message': {'content': 'Synthetic response'}}]}).encode()
        target.wfile.write(payload)
        target.wfile.flush()
        assert output.getvalue() == before_stop
        if upstream_error:
            raise OSError('Independent synthetic upstream failure')
    monkeypatch.setattr(HermesProviderRelay, 'forward', provider)
    with instrument(budget, 'http://127.0.0.1:8000/v1', '172.22.0.2'):
        if upstream_error:
            with pytest.raises(OSError):
                HermesProviderRelay.forward(None, handler, b'{}')
        else:
            HermesProviderRelay.forward(None, handler, b'{}')
        with pytest.raises(ExecutionStopped):
            budget.enter('tool', 'not admitted during drain')
        with pytest.raises(ExecutionStopped):
            budget.enter('model', 'not admitted during drain')
    rows = [json.loads(line) for line in evidence.path.read_text().splitlines()]
    assert budget.native_providers == budget.native_active == 0
    assert any(row['kind'] == 'provider_artifact' for row in rows)
    assert any(row['kind'] == 'provider_error' for row in rows) == upstream_error
    assert any(row['kind'] == 'model_output' for row in rows)


def test_deep_planned_limit_drains_stream_without_late_output_or_hidden_errors(tmp_path):
    import json
    from langchain_openai import ChatOpenAI
    for broken in (False, True):
        evidence = Evidence(tmp_path / str(broken))
        budget = Budget(evidence, 4, 10)
        def event(text):
            return ('data: ' + json.dumps({'id': 'synthetic', 'object': 'chat.completion.chunk',
                'created': 0, 'model': 'synthetic', 'choices': [{'index': 0,
                    'delta': {'content': text}, 'finish_reason': None}]}) + '\n\n').encode()
        class Stream(httpx.SyncByteStream):
            def __iter__(self):
                yield event('before')
                budget.stop('wall_limit')
                if broken:
                    raise httpx.ReadError('Independent synthetic upstream failure')
                yield event('after')
                yield b'data: [DONE]\n\n'
        transport = httpx.MockTransport(lambda request: httpx.Response(200,
            headers={'Content-Type': 'text/event-stream'}, stream=Stream()))
        seen = []
        with httpx.Client(transport=transport) as client:
            model = ChatOpenAI(model='synthetic', api_key='EMPTY',
                base_url='http://127.0.0.1:8000/v1', http_client=client, streaming=True, max_retries=0)
            with instrument(budget, 'http://127.0.0.1:8000/v1', '172.22.0.2'):
                with pytest.raises(httpx.ReadError if broken else ExecutionStopped):
                    for row in model.stream('Synthetic bounded episode.'):
                        seen.append(row.content)
        assert 'after' not in ''.join(seen)
        rows = evidence.path.read_text()
        assert ('provider_error' in rows) == broken
        assert 'model_output' in rows


def test_stalled_model_output_journal_does_not_hold_stop_lock(tmp_path, monkeypatch):
    import io
    import threading
    from types import SimpleNamespace
    from assist.hermes_provider_relay import HermesProviderRelay
    evidence = Evidence(tmp_path)
    budget = Budget(evidence, 4, 10)
    entered, release, stopped = threading.Event(), threading.Event(), threading.Event()
    original_write, original_flush = evidence.stream.write, evidence.stream.flush
    output_row = [False]
    def write(line):
        output_row[0] = '"kind": "model_output"' in line
        return original_write(line)
    def flush():
        if output_row[0]:
            entered.set()
            assert release.wait(2)
        original_flush()
    monkeypatch.setattr(evidence.stream, 'write', write)
    monkeypatch.setattr(evidence.stream, 'flush', flush)
    def provider(self, handler, body):
        handler.wfile.write(b'HTTP/1.0 200 OK\r\nContent-Type: application/json\r\n\r\n'
                            b'{"choices":[{"message":{"content":"synthetic"}}]}')
    monkeypatch.setattr(HermesProviderRelay, 'forward', provider)
    worker = threading.Thread(target=lambda: HermesProviderRelay.forward(
        None, SimpleNamespace(wfile=io.BytesIO()), b'{}'))
    stopper = threading.Thread(target=lambda: (budget.stop('wall_limit'), stopped.set()))
    with instrument(budget, 'http://127.0.0.1:8000/v1', '172.22.0.2'):
        try:
            worker.start()
            assert entered.wait(1)
            stopper.start()
            assert stopped.wait(.2), 'Output journaling blocked cancellation'
        finally:
            release.set()
            worker.join(2)
            stopper.join(2)
    assert not worker.is_alive() and not stopper.is_alive()
