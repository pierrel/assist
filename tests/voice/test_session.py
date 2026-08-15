import threading
import time
from hashlib import sha256
from types import SimpleNamespace

import pytest

from assist.events.thread_log import read_events
from manage.voice.session import VoiceSession, _Reply, _REPLIES, _frames
from manage.voice.service import PinLockout
from manage.voice.speech import Transcription
from manage.voice.wire import BufferClosed, CallBuffers, InboundBuffer, OutboundBuffer


PCM = b"\0\0" * 320


class FakeSpeech:
    def synthesize(self, _text):
        yield PCM

    def transcribe(self, _pcm):
        return Transcription("hello", 0.0)


class FakeFlow:
    def __init__(self, **_kwargs):
        self.events = []
        self.speaking = False

    def feed(self, _frame):
        events, self.events = self.events, []
        return events

    def set_speaking(self, value):
        self.speaking = value


def _eventually_get(buffer):
    return buffer.get(timeout=1)


def test_session_answers_after_allowlist_pin_and_speaks_only_its_run():
    buffers = CallBuffers(InboundBuffer(), OutboundBuffer())
    submitted = []
    session = VoiceSession(
        pin="000000",
        allowed_callers=frozenset({"+15555550100"}),
        speech=FakeSpeech(),
        flow_factory=FakeFlow,
        router_turn=lambda _text: "",
        submit_turn=lambda tid, text: submitted.append((tid, text)) or "voice-run",
    )
    runner = threading.Thread(
        target=session.run,
        args=({"type": "ring", "call_id": "call", "caller": "+15555550100"}, buffers),
    )
    runner.start()

    assert _eventually_get(buffers.outbound) == {"type": "answer"}
    buffers.inbound.put({"type": "answered", "call_id": "call"})
    assert _eventually_get(buffers.outbound) == PCM
    for _ in range(6):
        buffers.inbound.put({"type": "dtmf", "digit": "0"})
    assert _eventually_get(buffers.outbound) == PCM

    session.open_thread("thread-1")
    session._events.put(Transcription("find the forecast", 0.0))
    assert _eventually_get(buffers.outbound) == PCM
    assert submitted == [("thread-1", "find the forecast")]

    _REPLIES.deliver("thread-1", "ready", None, "It will rain.", "voice-run")
    assert _eventually_get(buffers.outbound) == PCM
    _REPLIES.deliver("thread-1", "ready", None, "wrong run", "web-run")
    time.sleep(0.05)
    with pytest.raises(TimeoutError):
        buffers.outbound.get(timeout=0.01)

    buffers.inbound.close_with_final({"type": "call_end", "cause": "remote"})
    runner.join(1)
    assert not runner.is_alive()
    assert _eventually_get(buffers.outbound) == {"type": "hangup"}


def test_session_never_answers_an_unknown_caller():
    buffers = CallBuffers(InboundBuffer(), OutboundBuffer())
    session = VoiceSession(
        pin="000000", allowed_callers=frozenset(), speech=FakeSpeech(),
    )

    session.run({"type": "ring", "call_id": "call", "caller": "+1555"}, buffers)

    with pytest.raises(BufferClosed):
        _eventually_get(buffers.outbound)


def test_locked_or_malformed_callers_do_not_construct_a_detector():
    class NoFlow:
        def __init__(self, **_kwargs):
            raise AssertionError("pre-auth call constructed Flow")

    gate = PinLockout()
    for _ in range(3):
        gate.record_failure()
    for caller in ("+15555550100", "+1 (555) 555-0100"):
        buffers = CallBuffers(InboundBuffer(), OutboundBuffer())
        VoiceSession(
            pin="000000", allowed_callers=frozenset({"+15555550100"}),
            speech=FakeSpeech(), flow_factory=NoFlow, lockout=gate,
        ).run({"type": "ring", "call_id": "call", "caller": caller}, buffers)
        with pytest.raises(BufferClosed):
            _eventually_get(buffers.outbound)


def test_call_log_hashes_hostile_bridge_ids_and_omits_caller(tmp_path):
    buffers = CallBuffers(InboundBuffer(), OutboundBuffer())
    session = VoiceSession(
        pin="000000", allowed_callers=frozenset(), speech=FakeSpeech(),
        call_log_root=tmp_path,
    )
    call_id = "../caller-control"

    session.run({"type": "ring", "call_id": call_id, "caller": "+1555"}, buffers)

    call_dir = tmp_path / "calls" / sha256(call_id.encode()).hexdigest()
    events = read_events(str(call_dir))
    assert len(events) == 1
    assert events[0]["kind"] == "decline"
    assert events[0]["reason"] == "not_allowed"
    assert "+1555" not in str(events)
    assert not (tmp_path / "caller-control").exists()


def test_unwritable_call_log_does_not_fail_the_call(tmp_path):
    root = tmp_path / "not-a-directory"
    root.write_text("occupied")
    buffers = CallBuffers(InboundBuffer(), OutboundBuffer())
    session = VoiceSession(
        pin="000000", allowed_callers=frozenset(), speech=FakeSpeech(),
        call_log_root=root,
    )

    session.run({"type": "ring", "call_id": "call", "caller": "+1555"}, buffers)

    with pytest.raises(BufferClosed):
        _eventually_get(buffers.outbound)


def test_session_logs_the_auth_boundary_without_digits_or_caller(tmp_path):
    buffers = CallBuffers(InboundBuffer(), OutboundBuffer())
    session = VoiceSession(
        pin="000000", allowed_callers=frozenset({"+15555550100"}),
        speech=FakeSpeech(), router_turn=lambda _text: "", call_log_root=tmp_path,
    )
    runner = threading.Thread(
        target=session.run,
        args=({"type": "ring", "call_id": "call", "caller": "+15555550100"}, buffers),
    )
    runner.start()
    assert _eventually_get(buffers.outbound) == {"type": "answer"}
    buffers.inbound.put({"type": "answered", "call_id": "call"})
    assert _eventually_get(buffers.outbound) == PCM
    for _ in range(6):
        buffers.inbound.put({"type": "dtmf", "digit": "0"})
    assert _eventually_get(buffers.outbound) == PCM
    buffers.inbound.close_with_final({"type": "call_end", "cause": "remote"})
    runner.join(1)

    events = read_events(str(tmp_path / "calls" / sha256(b"call").hexdigest()))
    kinds = [event["kind"] for event in events]
    assert ["ring", "answered", "pin_prompt", "pin_attempt", "auth_ok", "hangup"] == [
        kind for kind in kinds
        if kind in {"ring", "answered", "pin_prompt", "pin_attempt", "auth_ok", "hangup"}
    ]
    assert "000000" not in str(events)
    assert "+15555550100" not in str(events)


def test_greeting_constructs_only_a_dtmf_flow_and_never_transcribes():
    buffers = CallBuffers(InboundBuffer(), OutboundBuffer())
    flow_kwargs = []

    def make_flow(**kwargs):
        flow_kwargs.append(kwargs)
        return FakeFlow()

    class NoPreauthStt(FakeSpeech):
        def transcribe(self, _pcm):
            raise AssertionError("STT must not receive pre-auth PCM")

    session = VoiceSession(
        pin="000000", allowed_callers=frozenset({"+15555550100"}),
        speech=NoPreauthStt(), flow_factory=make_flow, router_turn=lambda _text: "",
    )
    runner = threading.Thread(
        target=session.run,
        args=({"type": "ring", "call_id": "call", "caller": "+15555550100"}, buffers),
    )
    runner.start()
    assert _eventually_get(buffers.outbound) == {"type": "answer"}
    buffers.inbound.put({"type": "answered", "call_id": "call"})
    assert _eventually_get(buffers.outbound) == PCM
    buffers.inbound.put(PCM)
    for _ in range(6):
        buffers.inbound.put({"type": "dtmf", "digit": "0"})
    assert _eventually_get(buffers.outbound) == PCM
    assert flow_kwargs == [{"vad": False}, {}]
    buffers.inbound.close_with_final({"type": "call_end", "cause": "remote"})
    runner.join(1)


def test_final_pin_failure_plays_recovery_then_hangs_up():
    buffers = CallBuffers(InboundBuffer(), OutboundBuffer())
    session = VoiceSession(
        pin="000000", allowed_callers=frozenset({"+15555550100"}),
        speech=FakeSpeech(), lockout=PinLockout(),
    )
    runner = threading.Thread(
        target=session.run,
        args=({"type": "ring", "call_id": "call", "caller": "+15555550100"}, buffers),
    )
    runner.start()
    assert _eventually_get(buffers.outbound) == {"type": "answer"}
    buffers.inbound.put({"type": "answered", "call_id": "call"})
    assert _eventually_get(buffers.outbound) == PCM

    for _ in range(3):
        for _ in range(6):
            buffers.inbound.put({"type": "dtmf", "digit": "1"})
        assert _eventually_get(buffers.outbound) == PCM

    runner.join(1)
    assert not runner.is_alive()
    assert _eventually_get(buffers.outbound) == {"type": "hangup"}


def test_default_submit_registers_before_its_turn_runner(monkeypatch):
    import manage.web.threads as threads

    session = VoiceSession(pin="000000", allowed_callers=frozenset(), speech=FakeSpeech())
    submitted = []
    monkeypatch.setattr(
        threads, "_create_run",
        lambda _tid, _text, **kwargs: (
            submitted.append(kwargs) or SimpleNamespace(id="voice-run")),
    )

    def finish(run_id, tid):
        _REPLIES.deliver(tid, "ready", None, "done", run_id)

    monkeypatch.setattr(threads, "_execute_run", finish)

    assert session._default_submit_turn("thread-1", "hello") == "voice-run"
    # Omitting ``assistant_id`` selects _create_run's ordinary general-agent,
    # which ThreadManager constructs with the main guidance profile.
    assert submitted == [{}]
    reply = session._events.get(timeout=1)
    assert reply == _Reply("thread-1", "ready", "done", "voice-run")


def test_router_callbacks_do_nothing_after_hangup():
    session = VoiceSession(pin="000000", allowed_callers=frozenset(), speech=FakeSpeech())
    session._stopped.set()

    session.open_thread("thread-1")
    session.new_thread("domain", "late request")

    assert session._events.empty()


def test_silence_watchdog_speaks_while_a_turn_is_thinking(monkeypatch):
    buffers = CallBuffers(InboundBuffer(), OutboundBuffer())
    session = VoiceSession(
        pin="000000", allowed_callers=frozenset(), speech=FakeSpeech(),
    )
    session._thinking = True
    session._last_audio = 0
    monkeypatch.setattr("manage.voice.session.SILENCE_SECONDS", 0)
    session._start_tts_producer(buffers)

    session._speak_if_silent(buffers)

    assert _eventually_get(buffers.outbound) == PCM
    session._stopped.set()
    with session._tts_condition:
        session._tts_condition.notify_all()


def test_tts_producer_serializes_latest_speech_request():
    first_started = threading.Event()
    release_first = threading.Event()
    synthesized = []

    class Speech:
        def synthesize(self, text):
            synthesized.append(text)
            if text == "first":
                first_started.set()
                release_first.wait(1)
            yield PCM

    buffers = CallBuffers(InboundBuffer(), OutboundBuffer())
    session = VoiceSession(pin="000000", allowed_callers=frozenset(), speech=Speech())
    session._start_tts_producer(buffers)
    session._speak("first", buffers)
    assert first_started.wait(1)
    session._speak("second", buffers)
    release_first.set()

    assert _eventually_get(buffers.outbound) == {"type": "flush_uplink"}
    assert _eventually_get(buffers.outbound) == PCM
    assert synthesized == ["first", "second"]
    session._stopped.set()
    with session._tts_condition:
        session._tts_condition.notify_all()


def test_tts_frames_preserve_chunk_remainders_and_pad_only_the_end():
    first = b"a" * 100
    second = b"b" * 600

    frames = list(_frames([first, second]))

    assert frames == [
        first + second[:540],
        second[540:] + b"\0" * 580,
    ]


def test_hangup_does_not_wait_for_a_stalled_tts_call():
    started = threading.Event()
    release = threading.Event()

    class StalledSpeech:
        def synthesize(self, _text):
            started.set()
            release.wait()
            yield PCM

    buffers = CallBuffers(InboundBuffer(), OutboundBuffer())
    session = VoiceSession(
        pin="000000", allowed_callers=frozenset({"+15555550100"}),
        speech=StalledSpeech(), router_turn=lambda _text: "",
    )
    runner = threading.Thread(
        target=session.run,
        args=({"type": "ring", "call_id": "call", "caller": "+15555550100"}, buffers),
    )
    runner.start()
    assert _eventually_get(buffers.outbound) == {"type": "answer"}
    buffers.inbound.put({"type": "answered", "call_id": "call"})
    assert started.wait(1)

    buffers.inbound.close_with_final({"type": "call_end", "cause": "remote"})
    runner.join(1)
    release.set()

    assert not runner.is_alive()
    assert _eventually_get(buffers.outbound) == {"type": "hangup"}
