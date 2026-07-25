import threading

from fastapi import FastAPI
from fastapi.testclient import TestClient

from manage.web.agent_protocol import (
    MAX_BODY_BYTES,
    create_agent_protocol_router,
)


class _Service:
    def __init__(self):
        self.calls = []
        self.loop_thread = None

    def _record(self, *call):
        assert self.loop_thread is not None
        assert threading.get_ident() != self.loop_thread
        self.calls.append(call)

    def create_thread(self):
        self._record("create_thread")
        return {"thread_id": "child-1", "status": "idle"}

    def get_thread(self, thread_id):
        self._record("get_thread", thread_id)
        return {
            "thread_id": thread_id,
            "status": "idle",
            "values": {"messages": [{"role": "assistant", "content": "done"}]},
        }

    def create_run(self, thread_id, assistant_id, text, *, multitask_strategy=None):
        self._record(
            "create_run", thread_id, assistant_id, text, multitask_strategy)
        return {
            "id": "run-1",
            "thread_id": thread_id,
            "assistant_id": assistant_id,
            "status": "pending",
        }

    def get_run(self, thread_id, run_id):
        self._record("get_run", thread_id, run_id)
        return {
            "run_id": run_id,
            "thread_id": thread_id,
            "assistant_id": "research-agent",
            "status": "success",
            "error": None,
        }

    def cancel_run(self, thread_id, run_id):
        self._record("cancel_run", thread_id, run_id)


def _app(service):
    app = FastAPI()

    @app.middleware("http")
    async def remember_loop_thread(request, call_next):
        service.loop_thread = threading.get_ident()
        return await call_next(request)

    app.include_router(create_agent_protocol_router(service, secret="test-secret"))
    return app


def _client():
    service = _Service()
    return TestClient(_app(service), headers={
        "Authorization": "Bearer test-secret"
    }), service


def test_authentication_is_required_before_service_access():
    service = _Service()
    client = TestClient(_app(service))
    assert client.post("/threads", json={}).status_code == 401
    assert client.post(
        "/threads", json={}, headers={"Authorization": "Bearer wrong"}
    ).status_code == 401
    assert service.calls == []


def test_installed_sdk_lifecycle_shapes():
    client, service = _client()

    thread = client.post("/threads", json={})
    assert thread.status_code == 200
    assert thread.json()["thread_id"] == "child-1"

    run = client.post("/threads/child-1/runs", json={
        "assistant_id": "research-agent",
        "input": {"messages": [{"role": "user", "content": "research this"}]},
    })
    assert run.status_code == 200
    assert run.json()["run_id"] == "run-1"
    assert run.json()["status"] == "pending"

    status = client.get("/threads/child-1/runs/run-1")
    assert status.json()["status"] == "success"
    state = client.get("/threads/child-1")
    assert state.json()["values"]["messages"][-1]["content"] == "done"
    cancelled = client.post("/threads/child-1/runs/run-1/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json() is None

    assert service.calls == [
        ("create_thread",),
        ("create_run", "child-1", "research-agent", "research this", None),
        ("get_run", "child-1", "run-1"),
        ("get_thread", "child-1"),
        ("cancel_run", "child-1", "run-1"),
    ]


def test_update_accepts_only_interrupt_strategy():
    client, service = _client()
    response = client.post("/threads/child-1/runs", json={
        "assistant_id": "research-agent",
        "input": {"messages": [{"role": "user", "content": "new context"}]},
        "multitask_strategy": "interrupt",
    })
    assert response.status_code == 200
    assert service.calls[-1] == (
        "create_run", "child-1", "research-agent", "new context", "interrupt")


def test_installed_sdk_default_run_fields_are_accepted():
    """langgraph-sdk 0.3.3 sends these defaults on every runs.create call."""
    client, service = _client()
    response = client.post("/threads/child-1/runs", json={
        "assistant_id": "research-agent",
        "input": {"messages": [{"role": "user", "content": "research this"}]},
        "stream_mode": "values",
        "stream_subgraphs": False,
        "stream_resumable": False,
    })
    assert response.status_code == 200
    assert service.calls == [
        ("create_run", "child-1", "research-agent", "research this", None),
    ]


def test_unknown_assistant_and_privileged_fields_are_rejected():
    client, service = _client()
    unknown = client.post("/threads/child-1/runs", json={
        "assistant_id": "arbitrary-agent",
        "input": {"messages": [{"role": "user", "content": "go"}]},
    })
    assert unknown.status_code == 404

    metadata = client.post("/threads/child-1/runs", json={
        "assistant_id": "research-agent",
        "input": {"messages": [{"role": "user", "content": "go"}]},
        "metadata": {"origin": "root"},
    })
    assert metadata.status_code == 422
    assert service.calls == []


def test_input_is_one_nonempty_user_message():
    client, _ = _client()
    base = {"assistant_id": "research-agent", "input": {"messages": []}}
    assert client.post("/threads/t/runs", json=base).status_code == 422

    base["input"]["messages"] = [{"role": "assistant", "content": "go"}]
    assert client.post("/threads/t/runs", json=base).status_code == 422

    base["input"]["messages"] = [
        {"role": "user", "content": "one"},
        {"role": "user", "content": "two"},
    ]
    assert client.post("/threads/t/runs", json=base).status_code == 422


def test_body_size_is_bounded_before_json_validation():
    client, service = _client()
    response = client.post(
        "/threads/t/runs",
        content=b"x" * (MAX_BODY_BYTES + 1),
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 413
    assert service.calls == []


def test_resource_ids_are_bounded_before_the_service():
    client, service = _client()
    assert client.get("/threads/../runs/run-1").status_code in {404, 422}
    assert client.get(f"/threads/{'x' * 129}/runs/run-1").status_code == 422
    assert service.calls == []


def test_response_does_not_expose_execution_context():
    class _LeakyService(_Service):
        def get_run(self, thread_id, run_id):
            value = super().get_run(thread_id, run_id)
            value.update({"text": "secret", "rider": {"token": "secret"}})
            return value

    service = _LeakyService()
    body = TestClient(_app(service), headers={
        "Authorization": "Bearer test-secret"
    }).get("/threads/t/runs/r").json()
    assert "text" not in body
    assert "rider" not in body
