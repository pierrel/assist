"""The synchronous per-call voice session.

Wire owns the WebSocket and buffers; this module owns call-local state and
never runs on the asyncio event loop.  The production runner remains disabled
until PR5 supplies its complete authentication policy.  Tests construct a
session with an explicit PIN and allowlist and install it through wire's runner
seam.
"""
from __future__ import annotations

import queue
import re
import secrets
import threading
import time
from hashlib import sha256
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from assist.events.thread_log import append_event
from manage.voice.flow import BargeIn, Dtmf, Flow, FRAME_BYTES, Utterance
from manage.voice.speech import Speech, Transcription
from manage.voice.wire import BufferClosed, CallBuffers


_PIN_PROMPT = "Please enter your PIN."
_AUTH_OK = "You're in. One second."
_THINKING = "Let me check."
_REPEAT = "Sorry, say that again."
_STILL_WORKING = "Still with me? I'm working on it."
SILENCE_SECONDS = 7.0
PIN_INTERDIGIT_SECONDS = 5.0
_E164 = re.compile(r"\+[1-9][0-9]{1,14}\Z")
_LOCKED = "Too many tries. Call back in a minute."


def normalize_e164(caller: str) -> str | None:
    """Return canonical E.164 caller identity, rejecting every other shape."""
    return caller if _E164.fullmatch(caller) else None


@dataclass(frozen=True)
class _Reply:
    tid: str
    stage: str
    text: str | None
    run_id: str


class _CallLog:
    """Best-effort call boundary history that never uses bridge input as a path."""

    def __init__(self, root: str | Path | None, call_id: str) -> None:
        self._directory: str | None = None
        if root is not None:
            safe_id = sha256(call_id.encode()).hexdigest()
            directory = Path(root) / "calls" / safe_id
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except OSError:
                return
            self._directory = str(directory)

    def append(self, kind: str, **fields: Any) -> None:
        if self._directory is not None:
            append_event(self._directory, kind, **fields)


class _ReplyRegistry:
    """Routes completed durable Runs to their one live call session."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_run: dict[str, tuple[str, queue.SimpleQueue]] = {}

    def attach(self, tid: str, run_id: str, events: queue.SimpleQueue) -> None:
        with self._lock:
            self._by_run[run_id] = (tid, events)

    def detach(self, events: queue.SimpleQueue) -> None:
        with self._lock:
            self._by_run = {
                run_id: binding
                for run_id, binding in self._by_run.items()
                if binding[1] is not events
            }

    def deliver(
        self, tid: str, stage: str, _origin: str | None,
        text: str | None, run_id: str | None,
    ) -> None:
        if not run_id:
            return
        with self._lock:
            binding = self._by_run.pop(run_id, None)
        if binding is not None and binding[0] == tid:
            binding[1].put(_Reply(tid, stage, text, run_id))


_REPLIES = _ReplyRegistry()
_observer_lock = threading.Lock()
_observer_installed = False


def _install_observer() -> None:
    """Install the one process-lifetime observer used by every voice session."""
    global _observer_installed
    with _observer_lock:
        if _observer_installed:
            return
        from manage.web.threads import register_turn_observer

        register_turn_observer(_REPLIES.deliver)
        _observer_installed = True


def _frames(chunks: Iterable[bytes]) -> Iterator[bytes]:
    """Join arbitrary Piper chunks into exact wire frames.

    Piper's chunks do not necessarily align to the bridge's fixed PCM frame
    size.  Preserve every sample across boundaries and pad only the final
    partial frame with silence.
    """
    pending = bytearray()
    for chunk in chunks:
        pending.extend(chunk)
        complete = len(pending) // FRAME_BYTES * FRAME_BYTES
        for offset in range(0, complete, FRAME_BYTES):
            yield bytes(pending[offset:offset + FRAME_BYTES])
        del pending[:complete]
    if pending:
        yield bytes(pending).ljust(FRAME_BYTES, b"\0")


class VoiceSession:
    """One fake-bridge voice call, with audio and Run delivery serialized here."""

    def __init__(
        self,
        *,
        pin: str,
        allowed_callers: frozenset[str],
        speech: Speech,
        flow_factory: Callable[..., Flow] = Flow,
        router_turn: Callable[[str], str] | None = None,
        submit_turn: Callable[[str, str], str] | None = None,
        call_log_root: str | Path | None = None,
        lockout: Any | None = None,
    ) -> None:
        self._pin = pin
        self._allowed_callers = allowed_callers
        self._speech = speech
        self._flow_factory = flow_factory
        self._router_turn = router_turn or self._default_router_turn
        self._submit_turn = submit_turn or self._default_submit_turn
        self._events: queue.SimpleQueue = queue.SimpleQueue()
        self._flow: Flow | None = None
        self._authenticated = False
        self._answered = False
        self._digits = ""
        self._stopped = threading.Event()
        self._tid: str | None = None
        self._call_log_root = call_log_root
        self._lockout = lockout
        self._pin_locked = False
        self._end_after_speech = False
        self._log: _CallLog | None = None
        self._router: Any | None = None
        self._speaking = False
        self._thinking = False
        self._last_audio = time.monotonic()
        self._last_digit_at: float | None = None
        self._tts_condition = threading.Condition()
        self._pending_tts: tuple[int, str] | None = None
        self._tts_thread: threading.Thread | None = None
        self._router_running = False
        self._pending_router_text: str | None = None
        self._default_submit = submit_turn is None

    def run(self, ring: dict[str, Any], buffers: CallBuffers) -> None:
        """Run until the bridge hangs up.  This is wire's synchronous runner."""
        _install_observer()
        log = _CallLog(self._call_log_root, ring["call_id"])
        self._log = log
        try:
            caller = normalize_e164(ring["caller"])
            if caller not in self._allowed_callers:
                log.append("decline", reason="not_allowed")
                return
            if self._lockout is not None and self._lockout.locked():
                log.append("decline", reason="pin_locked")
                return
            log.append("ring")
            buffers.outbound.put_control({"type": "answer"})
            self._start_tts_producer(buffers)
            while not self._stopped.is_set():
                self._drain_events(buffers)
                if (self._end_after_speech and not self._speaking
                        and buffers.outbound.empty()):
                    log.append("hangup", by="pin_lockout")
                    return
                try:
                    item = buffers.inbound.get(timeout=0.05)
                except TimeoutError:
                    self._check_pin_timeout(buffers)
                    self._speak_if_silent(buffers)
                    continue
                except BufferClosed:
                    return
                if isinstance(item, bytes):
                    self._handle_pcm(item, buffers)
                else:
                    if self._handle_control(item, ring, buffers):
                        log.append("hangup", by="bridge")
                        return
        finally:
            self._stopped.set()
            with self._tts_condition:
                self._pending_tts = None
                self._tts_condition.notify_all()
            _REPLIES.detach(self._events)
            if self._answered:
                buffers.outbound.close_with_final({"type": "hangup"})
            else:
                buffers.outbound.finish()

    def open_thread(self, tid: str) -> None:
        """Queue a receptionist handoff; safe to call from its model worker."""
        if self._stopped.is_set():
            return
        self._events.put(("open", tid))

    def new_thread(self, domain: str, first_message: str) -> None:
        """Create and bind a thread from a receptionist worker."""
        if self._stopped.is_set():
            return
        from manage.web.threads import (
            _initialize_thread, create_thread_with_message_core,
        )

        tid, run_id, selected = create_thread_with_message_core(
            first_message, domain)
        _REPLIES.attach(tid, run_id, self._events)
        self.open_thread(tid)
        threading.Thread(
            target=_initialize_thread, args=(tid, run_id, selected), daemon=True,
        ).start()

    def _default_router_turn(self, text: str) -> str:
        from assist.catalog import ThreadCatalog
        from assist.receptionist import create_receptionist, receptionist_tools
        from manage.web.state import DOMAINS, MANAGER

        if self._router is None:
            self._router = create_receptionist(
                receptionist_tools(ThreadCatalog(MANAGER), DOMAINS,
                                  self.open_thread, self.new_thread),
                DOMAINS,
            )
        result = self._router.invoke(
            {"messages": [{"role": "user", "content": text}]})
        message = result["messages"][-1]
        return message.content if isinstance(message.content, str) else ""

    def _default_submit_turn(self, tid: str, text: str) -> str:
        from manage.web.threads import _create_run, _execute_run

        run = _create_run(tid, text)
        _REPLIES.attach(tid, run.id, self._events)
        threading.Thread(
            target=_execute_run, args=(run.id, tid), daemon=True,
        ).start()
        return run.id

    def _handle_control(
        self, control: dict[str, Any], ring: dict[str, Any], buffers: CallBuffers
    ) -> bool:
        kind = control["type"]
        if kind == "call_end":
            return True
        if not self._answered:
            if kind == "answered" and control["call_id"] == ring["call_id"]:
                self._answered = True
                self._flow = self._flow_factory(vad=False)
                if self._log is not None:
                    self._log.append("answered")
                self._prompt_for_pin(buffers)
            return False
        if kind == "dtmf" and control["digit"] in "0123456789*":
            self._handle_digit(control["digit"], buffers)
        return False

    def _handle_pcm(self, frame: bytes, buffers: CallBuffers) -> None:
        if not self._answered or self._flow is None:
            return
        for event in self._flow.feed(frame):
            if isinstance(event, Dtmf):
                self._handle_digit(event.digit, buffers)
            elif isinstance(event, BargeIn):
                buffers.outbound.interrupt_audio()
                self._flow.set_speaking(False)
                self._speaking = False
                if self._log is not None:
                    self._log.append("barge_in", at_ms=event.at_ms)
            elif isinstance(event, Utterance) and self._authenticated:
                self._transcribe(event.pcm)

    def _handle_digit(self, digit: str, buffers: CallBuffers) -> None:
        if self._authenticated or self._pin_locked:
            return
        if digit == "*":
            self._digits = ""
            self._last_digit_at = None
            self._prompt_for_pin(buffers)
            return
        self._last_digit_at = time.monotonic()
        self._digits += digit
        if len(self._digits) < len(self._pin):
            return
        accepted = secrets.compare_digest(self._digits, self._pin)
        self._digits = ""
        self._last_digit_at = None
        if self._log is not None:
            self._log.append("pin_attempt", ok=accepted)
        if not accepted:
            if self._lockout is not None and self._lockout.record_failure():
                self._pin_locked = True
                self._end_after_speech = True
                if self._log is not None:
                    self._log.append("pin_locked")
                self._speak(_LOCKED, buffers)
                return
            self._prompt_for_pin(buffers)
            return
        self._authenticated = True
        if self._log is not None:
            self._log.append("auth_ok")
        self._flow = self._flow_factory()
        self._speak(_AUTH_OK, buffers)
        self._route("Orient the caller and ask what they need.")

    def _transcribe(self, pcm: bytes) -> None:
        try:
            self._events.put(self._speech.transcribe(pcm))
        except Exception:
            self._events.put(("stt_failed",))

    def _route(self, text: str) -> None:
        self._thinking = True
        if self._router_running:
            self._pending_router_text = text
            return
        self._router_running = True
        if self._log is not None:
            self._log.append("router_enter")

        def work() -> None:
            try:
                self._events.put(("router", self._router_turn(text)))
            except Exception:
                self._events.put(("router_error",))

        threading.Thread(target=work, daemon=True).start()

    def _drain_events(self, buffers: CallBuffers) -> None:
        while True:
            try:
                event = self._events.get_nowait()
            except queue.Empty:
                return
            if isinstance(event, _Reply):
                if self._log is not None:
                    self._log.append(
                        "turn_done", tid=event.tid, stage=event.stage,
                        run_id=event.run_id,
                    )
                if event.stage == "ready" and event.text:
                    self._thinking = False
                    self._speak(event.text, buffers)
                elif event.stage != "ready":
                    self._thinking = False
                    self._speak("Sorry, that didn't work.", buffers)
            elif isinstance(event, Transcription):
                if self._log is not None:
                    self._log.append(
                        "stt", chars=len(event.text),
                        no_speech_prob=event.no_speech_prob,
                    )
                if not event.text or (event.no_speech_prob or 0) > 0.6:
                    if self._log is not None:
                        self._log.append("noise_discard")
                    self._speak(_REPEAT, buffers)
                elif self._tid is None:
                    self._route(event.text)
                else:
                    self._speak(_THINKING, buffers)
                    self._thinking = True
                    run_id = self._submit_turn(self._tid, event.text)
                    if not self._default_submit:
                        _REPLIES.attach(self._tid, run_id, self._events)
                    if self._log is not None:
                        self._log.append("turn_submit", tid=self._tid)
            elif isinstance(event, tuple) and event[0] == "router":
                self._router_running = False
                self._thinking = False
                if self._log is not None:
                    self._log.append("router_return")
                self._speak(event[1], buffers)
                self._start_pending_router()
            elif isinstance(event, tuple) and event[0] == "router_error":
                self._router_running = False
                self._thinking = False
                if self._log is not None:
                    self._log.append("router_error")
                self._speak("Sorry, I couldn't do that.", buffers)
                self._start_pending_router()
            elif isinstance(event, tuple) and event[0] == "open":
                self._tid = event[1]
                if self._log is not None:
                    self._log.append("handoff", tid=self._tid)
            elif isinstance(event, tuple) and event[0] == "tts_done":
                if self._flow and buffers.outbound.is_current_generation(event[1]):
                    self._flow.set_speaking(False)
                    self._speaking = False
            elif isinstance(event, tuple) and event[0] == "tts_failed":
                if self._flow and buffers.outbound.is_current_generation(event[1]):
                    self._flow.set_speaking(False)
                    self._speaking = False
            elif isinstance(event, tuple) and event[0] == "stt_failed":
                self._speak(_REPEAT, buffers)

    def _speak(self, text: str, buffers: CallBuffers) -> None:
        if not text or self._stopped.is_set():
            return
        if self._speaking:
            buffers.outbound.interrupt_audio()
        if self._flow is not None:
            self._flow.set_speaking(True)
        self._speaking = True
        self._last_audio = time.monotonic()
        generation = buffers.outbound.start_generation()
        if self._log is not None:
            self._log.append("tts", chars=len(text))
        with self._tts_condition:
            self._pending_tts = (generation, text)
            self._tts_condition.notify()

    def _start_tts_producer(self, buffers: CallBuffers) -> None:
        """Start the one producer that serializes this call's Piper work."""
        if self._tts_thread is not None:
            return

        def produce() -> None:
            while not self._stopped.is_set():
                with self._tts_condition:
                    while self._pending_tts is None and not self._stopped.is_set():
                        self._tts_condition.wait()
                    if self._stopped.is_set():
                        return
                    generation, text = self._pending_tts
                    self._pending_tts = None
                try:
                    for frame in _frames(self._speech.synthesize(text)):
                        if self._stopped.is_set() or not buffers.outbound.put_audio(
                                generation, frame):
                            break
                    else:
                        self._events.put(("tts_done", generation))
                except Exception:
                    self._events.put(("tts_failed", generation))

        self._tts_thread = threading.Thread(target=produce, daemon=True)
        self._tts_thread.start()

    def _prompt_for_pin(self, buffers: CallBuffers) -> None:
        if self._log is not None:
            self._log.append("pin_prompt")
        self._speak(_PIN_PROMPT, buffers)

    def _check_pin_timeout(self, buffers: CallBuffers) -> None:
        if self._authenticated or self._last_digit_at is None:
            return
        if time.monotonic() - self._last_digit_at >= PIN_INTERDIGIT_SECONDS:
            self._digits = ""
            self._last_digit_at = None
            self._prompt_for_pin(buffers)

    def _start_pending_router(self) -> None:
        if self._pending_router_text is None:
            return
        text = self._pending_router_text
        self._pending_router_text = None
        self._route(text)

    def _speak_if_silent(self, buffers: CallBuffers) -> None:
        if self._thinking and time.monotonic() - self._last_audio >= SILENCE_SECONDS:
            self._speak(_STILL_WORKING, buffers)
