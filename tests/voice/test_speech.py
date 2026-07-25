from dataclasses import dataclass
import os
from pathlib import Path
import wave

import numpy as np
import pytest

from manage.voice.speech import Speech


_WHISPER_TEST_MODEL = os.getenv("ASSIST_WHISPER_TEST_MODEL")
_PIPER_TEST_MODEL = os.getenv("ASSIST_PIPER_TEST_MODEL")
_FIXTURES = Path(__file__).with_name("fixtures")


@dataclass
class _Segment:
    text: str
    no_speech_prob: float


class _Whisper:
    def __init__(self, segments):
        self.segments = segments
        self.calls = []

    def transcribe(self, audio, **kwargs):
        self.calls.append((audio, kwargs))
        return iter(self.segments), object()


@dataclass
class _Chunk:
    audio_float_array: np.ndarray
    sample_rate: int = 16_000
    sample_width: int = 2
    sample_channels: int = 1


class _Piper:
    def __init__(self, chunks):
        self.chunks = chunks
        self.calls = []

    def synthesize(self, text):
        self.calls.append(text)
        yield from self.chunks


def test_models_load_independently_and_lazily(monkeypatch, tmp_path):
    loads = []
    whisper = _Whisper([_Segment(" hello ", 0.04)])
    piper = _Piper([_Chunk(np.array([0.5], dtype=np.float32))])

    monkeypatch.setattr(
        "manage.voice.speech.WhisperModel",
        lambda *args, **kwargs: loads.append(("stt", args, kwargs)) or whisper,
    )
    monkeypatch.setattr(
        "manage.voice.speech.PiperVoice.load",
        lambda *args, **kwargs: loads.append(("tts", args, kwargs)) or piper,
    )

    speech = Speech(tmp_path / "voice.onnx", tmp_path / "whisper")
    assert loads == []

    assert speech.transcribe(b"\0\0" * 320).text == "hello"
    assert [load[0] for load in loads] == ["stt"]

    assert list(speech.synthesize("Hello."))
    assert [load[0] for load in loads] == ["stt", "tts"]


def test_transcribe_passes_the_voice_contract(monkeypatch, tmp_path):
    whisper = _Whisper([_Segment("hello", 0.08), _Segment(" world", 0.08)])
    monkeypatch.setattr(
        "manage.voice.speech.WhisperModel", lambda *a, **k: whisper
    )

    result = Speech(tmp_path / "voice.onnx", tmp_path / "whisper").transcribe(
        np.array([-32768, 0, 32767], dtype="<i2").tobytes(),
        initial_prompt="Larochelle",
    )

    audio, kwargs = whisper.calls[0]
    np.testing.assert_allclose(audio, [-1.0, 0.0, 32767 / 32768])
    assert kwargs == {
        "language": "en",
        "vad_filter": True,
        "condition_on_previous_text": False,
        "initial_prompt": "Larochelle",
    }
    assert result.text == "hello world"
    assert result.no_speech_prob == pytest.approx(0.08)


def test_transcribe_empty_has_no_invented_probability(monkeypatch, tmp_path):
    monkeypatch.setattr("manage.voice.speech.WhisperModel",
                        lambda *a, **k: _Whisper([]))

    result = Speech(
        tmp_path / "voice.onnx", tmp_path / "whisper"
    ).transcribe(b"\0\0" * 320)

    assert result.text == ""
    assert result.no_speech_prob is None


def test_transcribe_rejects_partial_sample(tmp_path):
    with pytest.raises(ValueError, match="whole 16-bit samples"):
        Speech(tmp_path / "voice.onnx", tmp_path / "whisper").transcribe(b"\0")


def test_synthesize_streams_normalized_16k_s16le(monkeypatch, tmp_path):
    piper = _Piper([
        _Chunk(np.array([-1.0, 0.5], dtype=np.float32)),
        _Chunk(np.array([0.25], dtype=np.float32)),
    ])
    monkeypatch.setattr(
        "manage.voice.speech.PiperVoice.load", lambda *a, **k: piper
    )
    chunks = Speech(
        tmp_path / "voice.onnx", tmp_path / "whisper"
    ).synthesize("One. Two.")

    first = next(chunks)
    assert np.frombuffer(first, dtype="<i2").tolist() == [-29490, 14745]
    assert len(piper.calls) == 1
    assert list(chunks)


def test_synthesize_resamples_to_16k(monkeypatch, tmp_path):
    piper = _Piper([_Chunk(np.linspace(-0.5, 0.5, 2205, dtype=np.float32),
                           sample_rate=22_050)])
    monkeypatch.setattr(
        "manage.voice.speech.PiperVoice.load", lambda *a, **k: piper
    )

    [pcm] = Speech(
        tmp_path / "voice.onnx", tmp_path / "whisper"
    ).synthesize("Hello.")

    assert len(pcm) // 2 == 1600


@pytest.mark.skipif(not _WHISPER_TEST_MODEL,
                    reason="ASSIST_WHISPER_TEST_MODEL is not provisioned")
def test_real_whisper_transcribes_fixture_and_rejects_noise(tmp_path):
    with wave.open(str(_FIXTURES / "speech_16k.wav"), "rb") as wav:
        spoken = wav.readframes(wav.getnframes())
    speech = Speech(tmp_path / "voice.onnx", _WHISPER_TEST_MODEL)

    result = speech.transcribe(spoken)
    noise = np.random.default_rng(7).integers(
        -200, 201, 16_000, dtype=np.int16
    )
    noise_result = speech.transcribe(noise.astype("<i2").tobytes())

    assert result.text.lower().strip(".") == (
        "the quick brown fox jumps over the lazy dog"
    )
    assert result.no_speech_prob is not None and result.no_speech_prob < 0.6
    assert noise_result.text == ""


@pytest.mark.skipif(not _PIPER_TEST_MODEL,
                    reason="ASSIST_PIPER_TEST_MODEL is not provisioned")
def test_real_piper_produces_16k_pcm():
    chunks = list(
        Speech(_PIPER_TEST_MODEL, "/unused/whisper").synthesize(
            "The quick brown fox."
        )
    )
    samples = np.frombuffer(b"".join(chunks), dtype="<i2")

    assert chunks
    assert len(samples) > 16_000 // 2
    assert 20_000 < np.abs(samples).max() <= 29_490
