"""Startup configuration and shared security state for voice sessions."""
from __future__ import annotations

import os
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from manage.voice.session import VoiceSession, normalize_e164
from manage.voice.speech import Speech
from manage.voice.wire import CallBuffers


PIN_FAILURE_LIMIT = 3
PIN_FAILURE_WINDOW_SECONDS = 60.0
PIN_LOCKOUT_SECONDS = 60.0


class PinLockout:
    """One process-wide rolling PIN failure gate for the single voice worker."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._failures: deque[float] = deque()
        self._locked_until = 0.0
        self._lock = threading.Lock()

    def locked(self) -> bool:
        with self._lock:
            return self._clock() < self._locked_until

    def record_failure(self) -> bool:
        """Record one failure and return whether it triggered the lockout."""
        with self._lock:
            now = self._clock()
            while (self._failures
                   and now - self._failures[0] >= PIN_FAILURE_WINDOW_SECONDS):
                self._failures.popleft()
            self._failures.append(now)
            if len(self._failures) < PIN_FAILURE_LIMIT:
                return False
            self._locked_until = now + PIN_LOCKOUT_SECONDS
            self._failures.clear()
            return True


@dataclass(frozen=True)
class VoiceConfig:
    pin: str
    callers: frozenset[str]
    call_log_root: Path
    piper_model: Path
    whisper_model: Path

    @classmethod
    def from_environ(cls, environ: Mapping[str, str] = os.environ) -> VoiceConfig | None:
        names = (
            "ASSIST_VOICE_SECRET", "ASSIST_VOICE_PIN", "ASSIST_VOICE_CALLERS",
            "ASSIST_VOICE_CALL_LOG_DIR", "ASSIST_VOICE_PIPER_MODEL",
            "ASSIST_VOICE_WHISPER_MODEL",
        )
        values = {name: environ.get(name, "") for name in names}
        if not any(values.values()):
            return None
        if not all(values.values()):
            raise ValueError("voice configuration is incomplete")
        pin = values["ASSIST_VOICE_PIN"]
        if len(pin) < 6 or not pin.isascii() or not pin.isdigit():
            raise ValueError("ASSIST_VOICE_PIN must contain at least six digits")
        callers = frozenset(values["ASSIST_VOICE_CALLERS"].split(","))
        if not callers or any(normalize_e164(caller) != caller for caller in callers):
            raise ValueError("ASSIST_VOICE_CALLERS must be canonical E.164")
        return cls(
            pin=pin,
            callers=callers,
            call_log_root=Path(values["ASSIST_VOICE_CALL_LOG_DIR"]),
            piper_model=Path(values["ASSIST_VOICE_PIPER_MODEL"]),
            whisper_model=Path(values["ASSIST_VOICE_WHISPER_MODEL"]),
        )


class VoiceService:
    """Create call-local sessions over one configured security boundary."""

    def __init__(self, config: VoiceConfig) -> None:
        self._config = config
        self._lockout = PinLockout()
        self._speech = Speech(config.piper_model, config.whisper_model)

    def __call__(self, ring: dict[str, object], buffers: CallBuffers) -> None:
        VoiceSession(
            pin=self._config.pin,
            allowed_callers=self._config.callers,
            speech=self._speech,
            call_log_root=self._config.call_log_root,
            lockout=self._lockout,
        ).run(ring, buffers)
