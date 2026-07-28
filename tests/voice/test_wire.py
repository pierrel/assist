import asyncio
import json
import threading
import time
from hashlib import sha256

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from manage.web import app
from assist.events.thread_log import read_events
from manage.voice import wire
from manage.voice.session import VoiceSession
from manage.voice.service import VoiceService
from tests.voice.fake_bridge import FakeBridge


SECRET = "test-voice-secret"
HEADERS = {"Authorization": f"Bearer {SECRET}"}
PCM = b"\0\0" * 320


@pytest.fixture(autouse=True)
def voice_wire(monkeypatch):
    monkeypatch.setenv("ASSIST_VOICE_SECRET", SECRET)
    wire.configure_call_runner(None)
    wire._release_call()
    yield
    wire.configure_call_runner(None)
    wire._release_call()


def _echo_runner(_ring, buffers):
    buffers.outbound.put({"type": "answer"})
    while True:
        item = buffers.inbound.get()
        if type(item) is bytes:
            buffers.outbound.put(item)
        elif item["type"] == "call_end":
            buffers.outbound.close_with_final({"type": "hangup"})
            return


def _blocking_runner(_ring, buffers):
    try:
        while True:
            buffers.inbound.get()
    except wire.BufferClosed:
        return


def test_codec_validates_known_controls_and_ignores_unknown():
    controls = [
        {"type": "ring", "call_id": "boot-1", "caller": "+15555550100"},
        {"type": "answered", "call_id": "boot-1"},
        {"type": "call_end", "cause": "remote"},
        {"type": "dtmf", "digit": "*"},
        {"type": "stats", "rate_hz": 16_700.0, "underruns": 0},
    ]
    for control in controls:
        assert wire.decode_phone_control(json.dumps(control)) == control
    assert wire.decode_phone_control('{"type":"future","value":1}') is None


@pytest.mark.parametrize("text", [
    '{"type":"ring","call_id":"a","call_id":"b","caller":"+1"}',
    '{"type":"stats","rate_hz":NaN,"underruns":0}',
    '{"type":"stats","rate_hz":true,"underruns":0}',
    '{"type":"stats","rate_hz":16000,"underruns":false}',
    '{"type":"dtmf","digit":"12"}',
    '{"type":"ring","call_id":"","caller":"+1"}',
    '{"type":"answered","call_id":""}',
    '{"type":"ring","call_id":"a","caller":"+1","extra":1}',
    '\ud800',
    r'{"type":"ring","call_id":"\ud800","caller":"+1"}',
    r'{"type":"dtmf","digit":"\ud800"}',
])
def test_codec_rejects_ambiguous_or_invalid_known_controls(text):
    with pytest.raises(wire.WireProtocolError):
        wire.decode_phone_control(text)


def test_codec_caps_untrusted_strings():
    for control in (
        {"type": "ring", "call_id": "x" * 129, "caller": "+1"},
        {"type": "ring", "call_id": "a", "caller": "x" * 33},
        {"type": "call_end", "cause": "x" * 65},
    ):
        with pytest.raises(wire.WireProtocolError):
            wire.decode_phone_control(json.dumps(control))


def test_auth_fails_closed_without_reserving_the_slot(monkeypatch):
    client = TestClient(app)
    for headers in ({}, {"Authorization": "Bearer wrong"}):
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/call", headers=headers):
                pass

    wire.configure_call_runner(_echo_runner)
    with client.websocket_connect("/call", headers=HEADERS) as websocket:
        bridge = FakeBridge(websocket)
        bridge.ring()
        assert bridge.receive_control() == {"type": "answer"}
        bridge.send_control("call_end", cause="remote")
        assert bridge.receive_control() == {"type": "hangup"}

    monkeypatch.delenv("ASSIST_VOICE_SECRET")
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/call", headers=HEADERS):
            pass


def test_fake_bridge_exercises_duplex_transport():
    wire.configure_call_runner(_echo_runner)
    with TestClient(app).websocket_connect(
        "/call", headers=HEADERS
    ) as websocket:
        bridge = FakeBridge(websocket)
        bridge.ring()
        assert bridge.receive_control() == {"type": "answer"}
        bridge.send_pcm(PCM)
        assert bridge.receive_pcm() == PCM
        bridge.send_control("call_end", cause="remote")
        assert bridge.receive_control() == {"type": "hangup"}
        with pytest.raises(WebSocketDisconnect):
            bridge.receive_control()


def test_app_lifespan_activates_voice_only_with_complete_configuration(monkeypatch):
    values = {
        "ASSIST_VOICE_SECRET": "test-secret",
        "ASSIST_VOICE_PIN": "123456",
        "ASSIST_VOICE_CALLERS": "+15555550100",
        "ASSIST_VOICE_CALL_LOG_DIR": "/var/lib/assist/call-log",
        "ASSIST_VOICE_PIPER_MODEL": "/opt/assist/models/voice.onnx",
        "ASSIST_VOICE_WHISPER_MODEL": "/opt/assist/models/whisper",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)

    with TestClient(app):
        assert isinstance(wire._CALL_RUNNER, VoiceService)

    assert wire._CALL_RUNNER is None


def test_fake_bridge_reconstructs_the_session_boundary_log(tmp_path):
    class Speech:
        def synthesize(self, _text):
            yield PCM

    session = VoiceSession(
        pin="000000", allowed_callers=frozenset({"+15555550100"}),
        speech=Speech(), router_turn=lambda _text: "", call_log_root=tmp_path,
    )
    wire.configure_call_runner(session.run)
    with TestClient(app).websocket_connect("/call", headers=HEADERS) as websocket:
        bridge = FakeBridge(websocket)
        bridge.ring()
        assert bridge.receive_control() == {"type": "answer"}
        bridge.send_control("answered", call_id="boot-1")
        assert bridge.receive_pcm() == PCM
        for _ in range(6):
            bridge.send_control("dtmf", digit="0")
        assert bridge.receive_pcm() == PCM
        bridge.send_control("call_end", cause="remote")
        assert bridge.receive_control() == {"type": "hangup"}

    events = read_events(str(tmp_path / "calls" / sha256(b"boot-1").hexdigest()))
    assert [event["kind"] for event in events] == [
        "ring", "answered", "pin_prompt", "tts", "pin_attempt", "auth_ok",
        "tts", "router_enter", "router_return", "hangup",
    ]
    assert "000000" not in str(events)
    assert "+15555550100" not in str(events)


@pytest.mark.parametrize("size", [639, 641, 4096])
def test_non_frame_binary_closes_the_call(size):
    wire.configure_call_runner(_blocking_runner)
    with TestClient(app).websocket_connect(
        "/call", headers=HEADERS
    ) as websocket:
        bridge = FakeBridge(websocket)
        bridge.ring()
        bridge.send_pcm(b"x" * size)
        with pytest.raises(WebSocketDisconnect) as exc:
            bridge.receive_control()
    assert exc.value.code == 1008


def test_terminal_close_exposes_a_released_call_slot(monkeypatch):
    original_close = wire._close_websocket
    slot_available = []

    async def close(websocket, code):
        available = wire._claim_call()
        slot_available.append(available)
        if available:
            wire._release_call()
        await original_close(websocket, code)

    monkeypatch.setattr(wire, "_close_websocket", close)
    wire.configure_call_runner(_blocking_runner)
    with TestClient(app).websocket_connect(
        "/call", headers=HEADERS
    ) as websocket:
        bridge = FakeBridge(websocket)
        bridge.ring()
        bridge.send_pcm(b"x")
        with pytest.raises(WebSocketDisconnect):
            bridge.receive_control()

    assert slot_available == [True]


def test_unknown_controls_count_toward_the_rate_limit(monkeypatch):
    monkeypatch.setattr(wire, "MAX_CONTROLS_PER_SECOND", 3)
    wire.configure_call_runner(_blocking_runner)
    with TestClient(app).websocket_connect(
        "/call", headers=HEADERS
    ) as websocket:
        bridge = FakeBridge(websocket)
        bridge.ring()
        bridge.send_control("future-a")
        bridge.send_control("future-b")
        bridge.send_control("future-c")
        with pytest.raises(WebSocketDisconnect) as exc:
            bridge.receive_control()
    assert exc.value.code == 1008


def test_unknown_controls_refresh_valid_inbound_idle(monkeypatch):
    monkeypatch.setattr(wire, "IDLE_SECONDS", 0.1)
    wire.configure_call_runner(_echo_runner)
    with TestClient(app).websocket_connect(
        "/call", headers=HEADERS
    ) as websocket:
        bridge = FakeBridge(websocket)
        bridge.ring()
        assert bridge.receive_control() == {"type": "answer"}
        for _ in range(5):
            time.sleep(0.03)
            bridge.send_control("future")
        bridge.send_control("call_end", cause="remote")
        assert bridge.receive_control() == {"type": "hangup"}


def test_first_ring_timeout_releases_the_slot(monkeypatch):
    monkeypatch.setattr(wire, "FIRST_RING_SECONDS", 0.01)
    client = TestClient(app)
    with client.websocket_connect("/call", headers=HEADERS) as websocket:
        with pytest.raises(WebSocketDisconnect) as exc:
            websocket.receive_text()
    assert exc.value.code == 1008

    wire.configure_call_runner(_echo_runner)
    with client.websocket_connect("/call", headers=HEADERS) as websocket:
        bridge = FakeBridge(websocket)
        bridge.ring()
        assert bridge.receive_control() == {"type": "answer"}
        bridge.send_control("call_end", cause="remote")
        assert bridge.receive_control() == {"type": "hangup"}


@pytest.mark.parametrize(
    "constant", ["IDLE_SECONDS", "MAX_CONNECTION_SECONDS"]
)
def test_call_time_bounds_close_and_release(monkeypatch, constant):
    monkeypatch.setattr(wire, constant, 0.01)
    wire.configure_call_runner(_blocking_runner)
    client = TestClient(app)
    with client.websocket_connect("/call", headers=HEADERS) as websocket:
        FakeBridge(websocket).ring()
        with pytest.raises(WebSocketDisconnect) as exc:
            websocket.receive_text()
    assert exc.value.code == 1008

    wire.configure_call_runner(_echo_runner)
    with client.websocket_connect("/call", headers=HEADERS) as websocket:
        bridge = FakeBridge(websocket)
        bridge.ring()
        assert bridge.receive_control() == {"type": "answer"}
        bridge.send_control("call_end", cause="remote")
        assert bridge.receive_control() == {"type": "hangup"}


def test_one_call_gate_rejects_a_concurrent_connection():
    wire.configure_call_runner(_blocking_runner)
    client = TestClient(app)
    with client.websocket_connect("/call", headers=HEADERS) as first:
        FakeBridge(first).ring()
        with pytest.raises(WebSocketDisconnect) as exc:
            with client.websocket_connect("/call", headers=HEADERS):
                pass
        assert exc.value.code == 1013


def test_blocked_call_workers_do_not_block_the_event_loop():
    wire.configure_call_runner(_blocking_runner)
    with TestClient(app) as client:
        with client.websocket_connect("/call", headers=HEADERS) as websocket:
            FakeBridge(websocket).ring()
            started = time.monotonic()
            response = client.get("/favicon.ico")
            elapsed = time.monotonic() - started
    assert response.status_code == 200
    assert elapsed < 0.2


def test_stalled_terminal_send_has_bounded_teardown(monkeypatch):
    monkeypatch.setattr(wire, "TERMINAL_GRACE_SECONDS", 0.01)

    def finish(_ring, buffers):
        buffers.outbound.close_with_final({"type": "hangup"})

    class StalledWebSocket:
        def __init__(self):
            self.sends = 0

        async def receive(self):
            await asyncio.Future()

        async def send_text(self, _text):
            self.sends += 1
            await asyncio.Future()

        async def send_bytes(self, _data):
            self.sends += 1
            await asyncio.Future()

    wire.configure_call_runner(finish)
    websocket = StalledWebSocket()
    buffers = wire.CallBuffers(wire.InboundBuffer(), wire.OutboundBuffer())
    started = time.monotonic()
    asyncio.run(
        wire._run_call(
            websocket,
            {"type": "ring", "call_id": "a", "caller": "+1"},
            buffers,
            wire._ControlRate(),
        )
    )
    assert time.monotonic() - started < 1
    assert websocket.sends == 1


@pytest.mark.parametrize("size", [639, 641])
def test_non_frame_outbound_binary_fails_the_call(size):
    async def exercise():
        class WebSocket:
            async def receive(self):
                await asyncio.Future()

            async def close(self, code):
                pass

            async def send_bytes(self, _data):
                raise AssertionError("invalid audio must not reach the socket")

        def runner(_ring, buffers):
            buffers.outbound.put(b"x" * size)

        wire.configure_call_runner(runner)
        buffers = wire.CallBuffers(
            wire.InboundBuffer(), wire.OutboundBuffer()
        )
        with pytest.raises(wire.WireProtocolError):
            await wire._run_call(
                WebSocket(),
                {"type": "ring", "call_id": "a", "caller": ""},
                buffers,
                wire._ControlRate(),
            )

    asyncio.run(exercise())


def test_runner_failure_cleans_up_before_returning():
    exited = threading.Event()

    def runner(_ring, _buffers):
        try:
            raise RuntimeError("runner failed")
        finally:
            exited.set()

    class WebSocket:
        async def receive(self):
            await asyncio.Future()

        async def close(self, code):
            pass

    wire.configure_call_runner(runner)
    buffers = wire.CallBuffers(wire.InboundBuffer(), wire.OutboundBuffer())
    with pytest.raises(RuntimeError, match="runner failed"):
        asyncio.run(
            wire._run_call(
                WebSocket(),
                {"type": "ring", "call_id": "a", "caller": ""},
                buffers,
                wire._ControlRate(),
            )
        )
    assert exited.is_set()


def test_runner_failure_closes_endpoint_as_internal_error():
    def runner(_ring, _buffers):
        raise RuntimeError("runner failed")

    wire.configure_call_runner(runner)
    with TestClient(app).websocket_connect(
        "/call", headers=HEADERS
    ) as websocket:
        bridge = FakeBridge(websocket)
        bridge.ring()
        with pytest.raises(WebSocketDisconnect) as exc:
            bridge.receive_control()
    assert exc.value.code == 1011


def test_hard_timeout_stops_stalled_send_and_runner_before_return(monkeypatch):
    monkeypatch.setattr(wire, "TERMINAL_GRACE_SECONDS", 0.01)
    runner_exited = threading.Event()
    send_cancelled = asyncio.Event()

    def runner(_ring, buffers):
        try:
            buffers.outbound.put(PCM)
            buffers.inbound.get()
        except wire.BufferClosed:
            pass
        finally:
            runner_exited.set()

    class WebSocket:
        async def receive(self):
            await asyncio.Future()

        async def close(self, code):
            pass

        async def send_bytes(self, _data):
            try:
                await asyncio.Future()
            finally:
                send_cancelled.set()

    async def exercise():
        wire.configure_call_runner(runner)
        buffers = wire.CallBuffers(
            wire.InboundBuffer(), wire.OutboundBuffer()
        )
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(0.02):
                await wire._run_call(
                    WebSocket(),
                    {"type": "ring", "call_id": "a", "caller": ""},
                    buffers,
                    wire._ControlRate(),
                )
        assert runner_exited.is_set()
        assert send_cancelled.is_set()

    asyncio.run(exercise())


def test_timeout_wakes_full_receive_and_empty_send_workers():
    runner_exited = threading.Event()

    def runner(_ring, buffers):
        try:
            buffers.outbound.get()
        except wire.BufferClosed:
            runner_exited.set()

    class WebSocket:
        def __init__(self):
            self.receives = 0

        async def receive(self):
            if self.receives <= wire.INBOUND_CAPACITY:
                self.receives += 1
                return {"type": "websocket.receive", "bytes": PCM}
            await asyncio.Future()

        async def close(self, code):
            pass

    async def exercise():
        wire.configure_call_runner(runner)
        buffers = wire.CallBuffers(
            wire.InboundBuffer(), wire.OutboundBuffer()
        )
        call = asyncio.create_task(
            wire._run_call(
                WebSocket(),
                {"type": "ring", "call_id": "a", "caller": ""},
                buffers,
                wire._ControlRate(),
            )
        )
        started = time.monotonic()
        await asyncio.sleep(0.02)
        assert time.monotonic() - started < 0.2
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call
        assert runner_exited.is_set()

    asyncio.run(exercise())


def test_abort_discards_a_gracefully_closed_backlog():
    outbound = wire.OutboundBuffer()
    outbound.put(PCM)
    outbound.finish()
    outbound.abort()
    with pytest.raises(wire.BufferClosed):
        outbound.get()


def test_close_wakes_blocked_get_and_put():
    inbound = wire.InboundBuffer()
    got = []
    getter = threading.Thread(target=lambda: _record_closed(inbound.get, got))
    getter.start()
    time.sleep(0.01)
    inbound.close()
    getter.join(1)
    assert got == [wire.BufferClosed]

    outbound = wire.OutboundBuffer()
    for _ in range(wire.OUTBOUND_CAPACITY):
        outbound.put(PCM)
    put = []
    putter = threading.Thread(
        target=lambda: _record_closed(lambda: outbound.put(PCM), put)
    )
    putter.start()
    time.sleep(0.01)
    outbound.abort()
    putter.join(1)
    assert put == [wire.BufferClosed]


def _record_closed(operation, result):
    try:
        operation()
    except wire.BufferClosed:
        result.append(wire.BufferClosed)


def test_barge_in_flushes_and_invalidates_a_blocked_tts_generation():
    outbound = wire.OutboundBuffer()
    generation = outbound.start_generation()
    for _ in range(wire.OUTBOUND_CAPACITY):
        assert outbound.put_audio(generation, PCM)

    accepted = []
    producer = threading.Thread(
        target=lambda: accepted.append(outbound.put_audio(generation, PCM))
    )
    producer.start()
    time.sleep(0.01)

    outbound.interrupt_audio()
    producer.join(1)

    assert accepted == [False]
    outbound.interrupt_audio()
    outbound.interrupt_audio()
    assert outbound.get() == {"type": "flush_uplink"}
    with pytest.raises(TimeoutError):
        outbound.get(timeout=0.01)


def test_control_evicts_stale_audio_when_outbound_is_full():
    outbound = wire.OutboundBuffer()
    for _ in range(wire.OUTBOUND_CAPACITY):
        outbound.put(PCM)

    outbound.put_control({"type": "hangup"})

    assert len(outbound._items) == wire.OUTBOUND_CAPACITY
    assert outbound.get() == {"type": "hangup"}


def test_terminal_inbound_control_discards_buffered_pcm():
    inbound = wire.InboundBuffer()
    for _ in range(wire.INBOUND_CAPACITY):
        inbound.put(PCM)

    final = {"type": "call_end", "cause": "remote"}
    inbound.close_with_final(final)

    assert inbound.get() == final
    with pytest.raises(wire.BufferClosed):
        inbound.get()
