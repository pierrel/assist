from pathlib import Path

import pytest

from manage.voice.service import PinLockout, VoiceConfig


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_global_lockout_counts_rotating_caller_attempts(monkeypatch):
    clock = Clock()
    gate = PinLockout(clock)
    monkeypatch.setattr("manage.voice.service.PIN_FAILURE_LIMIT", 3)

    assert not gate.record_failure()
    clock.now += 1
    assert not gate.record_failure()
    clock.now += 1
    assert gate.record_failure()
    assert gate.locked()
    clock.now += 60
    assert not gate.locked()


def test_voice_config_requires_complete_canonical_configuration():
    settings = {
        "ASSIST_VOICE_SECRET": "secret",
        "ASSIST_VOICE_PIN": "123456",
        "ASSIST_VOICE_CALLERS": "+15555550100",
        "ASSIST_VOICE_CALL_LOG_DIR": "/var/lib/assist/calls",
        "ASSIST_VOICE_PIPER_MODEL": "/opt/assist/models/voice.onnx",
        "ASSIST_VOICE_WHISPER_MODEL": "/opt/assist/models/whisper",
    }

    config = VoiceConfig.from_environ(settings)

    assert config == VoiceConfig(
        pin="123456", callers=frozenset({"+15555550100"}),
        call_log_root=Path("/var/lib/assist/calls"),
        piper_model=Path("/opt/assist/models/voice.onnx"),
        whisper_model=Path("/opt/assist/models/whisper"),
    )
    assert VoiceConfig.from_environ({}) is None
    settings["ASSIST_VOICE_CALLERS"] = "+1 (555) 555-0100"
    with pytest.raises(ValueError, match="canonical E.164"):
        VoiceConfig.from_environ(settings)
