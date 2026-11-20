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
