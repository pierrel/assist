"""In-process speech boundary for the voice-call client.

``Speech`` wraps the one STT/TTS implementation selected by the design:
faster-whisper ``base.en`` on CPU and Piper. Both models load independently on
first use from already provisioned local artifacts. Methods are synchronous and
must be called off the uvicorn event loop; session scheduling belongs in
``session.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import Iterator

import numpy as np
from faster_whisper import WhisperModel
from piper import PiperVoice
from scipy.signal import resample_poly

from manage.voice.flow import SAMPLE_RATE


@dataclass(frozen=True)
class Transcription:
    """Accepted text and Whisper's segment-level no-speech evidence.

    Endpointed utterances fit in one Whisper window. All segments from that
    window carry the same probability, so the first supplies it. If VAD removes
    the input, Whisper emits no segment and the probability is ``None`` rather
    than an invented model score; empty text is the discard signal.
    """

    text: str
    no_speech_prob: float | None


class Speech:
    """Host-owned Whisper/Piper models for the single active voice call."""

    def __init__(self, piper_model_path: str | Path,
                 whisper_model_path: str | Path):
        self._piper_model_path = Path(piper_model_path)
        self._whisper_model_path = str(whisper_model_path)
        self._whisper: WhisperModel | None = None
        self._piper: PiperVoice | None = None

    def _stt(self) -> WhisperModel:
        if self._whisper is None:
            self._whisper = WhisperModel(
                self._whisper_model_path,
                device="cpu",
                compute_type="int8",
                local_files_only=True,
            )
        return self._whisper

    def _tts(self) -> PiperVoice:
        if self._piper is None:
            self._piper = PiperVoice.load(
                self._piper_model_path, use_cuda=False
            )
        return self._piper

    def transcribe(
        self, pcm: bytes, *, initial_prompt: str | None = None
    ) -> Transcription:
        """Transcribe little-endian mono s16le/16 kHz PCM."""
        if len(pcm) % 2:
            raise ValueError("PCM must contain whole 16-bit samples")
        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        segments, _ = self._stt().transcribe(
            audio,
            language="en",
            vad_filter=True,
            condition_on_previous_text=False,
            initial_prompt=initial_prompt,
        )
        decoded = list(segments)
        return Transcription(
            text="".join(segment.text for segment in decoded).strip(),
            no_speech_prob=decoded[0].no_speech_prob if decoded else None,
        )

    def synthesize(self, text: str) -> Iterator[bytes]:
        """Yield Piper sentence chunks as normalized mono s16le/16 kHz PCM."""
        for chunk in self._tts().synthesize(text):
            audio = chunk.audio_float_array
            if chunk.sample_rate != SAMPLE_RATE:
                divisor = gcd(chunk.sample_rate, SAMPLE_RATE)
                audio = resample_poly(
                    audio,
                    SAMPLE_RATE // divisor,
                    chunk.sample_rate // divisor,
                )
            normalized = np.clip(audio, -1.0, 1.0) * (0.9 * 32767)
            yield normalized.astype("<i2").tobytes()
