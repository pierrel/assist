"""Hermetic conversation-flow engine (voice-call assistant, P1;
docs/2026-07-23-voice-p1.org). ``Flow.feed(frame) -> list[event]`` consumes 20 ms PCM
frames and emits events (Dtmf / VadOpen / Utterance / BargeIn). NO web imports, no
network, no threads, no wall clock — time is the VAD-window count, which is what makes
the fixture tests deterministic. The only I/O is lazily loading the silero VAD ONNX on
the first VAD window.

Two detectors: ``DtmfDetector`` — in-band DTMF via the Goertzel single-bin power (the
one detector interface the design mandates as swappable; in-band today, a modem-URC
detector is the hardware-gated upgrade if AMR mangles in-band tones); and the silero
VAD + endpointing + barge-in inside ``Flow``. DTMF is pure DSP (numpy only, no model);
the VAD lazy-loads onnxruntime + the ONNX model, so DTMF-only scopes never touch it.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

SAMPLE_RATE = 16_000
FRAME_MS = 20
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000   # 320
FRAME_BYTES = FRAME_SAMPLES * 2                  # 640 (s16le)

# silero VAD (v5) consumes exactly 512-sample windows at 16 kHz (32 ms). The bridge
# delivers 20 ms frames, so the engine buffers frames and drains 512-sample windows;
# time is the window count (deterministic), never a wall clock.
_VAD_WIN = 512
_VAD_CTX = 64                                     # silero v5 prepends the last 64 samples
_VAD_WIN_MS = _VAD_WIN * 1000 // SAMPLE_RATE      # 32
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "assets", "silero_vad.onnx")

# The DTMF keypad: each key is one low-group + one high-group tone.
_LOW_FREQS = (697, 770, 852, 941)
_HIGH_FREQS = (1209, 1336, 1477, 1633)
_DIGITS = {
    (697, 1209): "1", (697, 1336): "2", (697, 1477): "3", (697, 1633): "A",
    (770, 1209): "4", (770, 1336): "5", (770, 1477): "6", (770, 1633): "B",
    (852, 1209): "7", (852, 1336): "8", (852, 1477): "9", (852, 1633): "C",
    (941, 1209): "*", (941, 1336): "0", (941, 1477): "#", (941, 1633): "D",
}


class DtmfDetector:
    """Leading-edge DTMF detector over 20 ms s16le/16 kHz frames.

    ``feed(frame) -> digit | None`` returns a digit exactly ONCE per keypress: when a
    valid single tone-pair holds for ``min_frames`` consecutive frames it fires, then
    stays silent until the tone drops (an invalid/idle frame) — so a held key yields
    one digit and two presses (with a gap) yield two. Rejects speech/noise by requiring
    a dominant single tone in each group holding a large share of frame energy, within
    a lenient twist band.
    """

    def __init__(self, *, min_frames: int = 2, release_frames: int = 2,
                 energy_frac: float = 0.3, group_dominance: float = 0.25, twist=(0.1, 10.0)):
        self._min_frames = min_frames
        self._release_frames = release_frames     # non-tone frames needed to re-arm (gap tolerance)
        self._energy_frac = energy_frac
        self._group_dominance = group_dominance   # 2nd tone in a group must be < this × the top
        self._twist_lo, self._twist_hi = twist
        # Precompute the single-bin DFT basis (8 × N): power at freq f over the frame
        # equals |basis_f · samples|² — the Goertzel result without the recurrence.
        n = np.arange(FRAME_SAMPLES)
        freqs = np.array(_LOW_FREQS + _HIGH_FREQS, dtype=np.float64)
        self._basis = np.exp(-2j * np.pi * np.outer(freqs, n) / SAMPLE_RATE)
        self._cur: str | None = None
        self._count = 0
        self._emitted = False
        self._gap = 0

    def _candidate(self, samples: np.ndarray) -> str | None:
        """The valid DTMF digit in this frame, or None (silence/speech/noise)."""
        energy = float(samples @ samples)
        if energy <= 0.0:
            return None
        powers = np.abs(self._basis @ samples) ** 2
        lo, hi = powers[:4], powers[4:]
        li, hii = int(lo.argmax()), int(hi.argmax())
        lo_p, hi_p = float(lo[li]), float(hi[hii])
        if (lo_p + hi_p) / (FRAME_SAMPLES * energy) < self._energy_frac:
            return None                                     # tones don't dominate the frame
        lo_2nd = float(np.partition(lo, -2)[-2])
        hi_2nd = float(np.partition(hi, -2)[-2])
        if lo_2nd >= self._group_dominance * lo_p or hi_2nd >= self._group_dominance * hi_p:
            return None                                     # not a single tone per group
        if not (self._twist_lo <= hi_p / lo_p <= self._twist_hi):
            return None                                     # implausible high/low balance
        return _DIGITS[(_LOW_FREQS[li], _HIGH_FREQS[hii])]

    def feed(self, frame: bytes) -> str | None:
        if len(frame) != FRAME_BYTES:
            raise ValueError(f"expected a {FRAME_BYTES}-byte frame, got {len(frame)}")
        samples = np.frombuffer(frame, dtype="<i2").astype(np.float64)
        cand = self._candidate(samples)
        if cand is None:
            # Tolerate a brief tone dropout mid-hold (a lossy codec drops frames): only
            # re-arm after release_frames of continuous non-tone, so a single glitched
            # frame during a held key can't split it into a duplicate digit.
            self._gap += 1
            if self._gap >= self._release_frames:
                self._cur, self._count, self._emitted = None, 0, False
            return None
        self._gap = 0
        if cand == self._cur:
            self._count += 1
        else:
            self._cur, self._count, self._emitted = cand, 1, False
        if self._count >= self._min_frames and not self._emitted:
            self._emitted = True
            return cand
        return None


# ── events ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class VadOpen:
    t_ms: int


@dataclass(frozen=True)
class Utterance:
    pcm: bytes
    dur_ms: int


@dataclass(frozen=True)
class BargeIn:
    at_ms: int


@dataclass(frozen=True)
class Dtmf:
    digit: str


class Flow:
    """The hermetic frame-in / event-out engine. ``feed(frame) -> list[event]`` over
    20 ms s16le/16 kHz frames; emits ``Dtmf``, ``VadOpen``, ``Utterance``, ``BargeIn``.
    No web imports, no network, no threads, no wall clock — time is the VAD-window
    count, so fixture runs are deterministic. The only I/O is lazily loading the silero
    ONNX on the first VAD window (so DTMF-only scopes and scripted tests never touch it).

    Detector scoping (§8): construct with ``vad=False`` (GREETING_PIN — DTMF only) or
    ``dtmf=False``; ``set_speaking`` gates barge-in, which fires ONLY while the agent is
    speaking, on sustained speech above a raised threshold.
    """

    def __init__(self, *, vad: bool = True, dtmf: bool = True, speaking: bool = False,
                 open_ms: int = 200, close_ms: int = 700, max_utt_ms: int = 30_000,
                 vad_threshold: float = 0.5, bargein_threshold: float = 0.7,
                 bargein_sustain_ms: int = 300):
        self._dtmf = DtmfDetector() if dtmf else None
        self._vad = vad
        self._speaking = speaking
        self._vad_threshold = vad_threshold
        self._bargein_threshold = bargein_threshold
        # thresholds in whole VAD windows (ceil, so "≥200 ms" needs a full 200 ms)
        self._open_win = -(-open_ms // _VAD_WIN_MS)
        self._close_win = -(-close_ms // _VAD_WIN_MS)
        self._max_utt_win = max(1, max_utt_ms // _VAD_WIN_MS)   # ≥1: a sub-window cap can't close-on-open
        self._bargein_win = -(-bargein_sustain_ms // _VAD_WIN_MS)
        # rolling state
        self._buf = np.zeros(0, dtype=np.int16)
        self._windows = 0
        self._sess = None
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(_VAD_CTX, dtype=np.float32)   # carried between windows
        self._in_utt = False
        self._speech_run = 0
        self._silence_run = 0
        self._pending: list[bytes] = []      # onset windows held until an utterance opens
        self._utt: list[bytes] = []
        self._bargein_run = 0
        self._bargein_fired = False

    def set_speaking(self, speaking: bool) -> None:
        """The session tells the engine when TTS is playing. Leaving SPEAKING re-arms
        barge-in for the next reply."""
        self._speaking = speaking
        if not speaking:
            self._bargein_run = 0
            self._bargein_fired = False

    def _speech_prob(self, window: np.ndarray) -> float:
        """silero speech probability for one 512-sample float window (lazy-loaded).
        v5 takes 64 samples of carried context prepended to the window (576 total)."""
        if self._sess is None:
            import onnxruntime as ort
            self._sess = ort.InferenceSession(
                _MODEL_PATH, providers=["CPUExecutionProvider"])
        inp = np.concatenate([self._context, window]).reshape(1, -1)
        out = self._sess.run(None, {"input": inp, "state": self._state,
                                    "sr": np.array(SAMPLE_RATE, dtype=np.int64)})
        self._state, self._context = out[1], window[-_VAD_CTX:]
        return float(out[0][0, 0])

    def _vad_window(self, win: np.ndarray) -> list:
        """Drive endpointing + barge-in for one 512-sample s16 window."""
        self._windows += 1
        t_ms = self._windows * _VAD_WIN_MS
        prob = self._speech_prob(win.astype(np.float32) / 32768.0)
        win_bytes = win.tobytes()
        events: list = []

        if self._speaking:                              # barge-in: raised threshold + sustain
            if prob >= self._bargein_threshold:
                self._bargein_run += 1
                if self._bargein_run >= self._bargein_win and not self._bargein_fired:
                    self._bargein_fired = True
                    events.append(BargeIn(t_ms))
            else:
                self._bargein_run = 0

        is_speech = prob >= self._vad_threshold
        if not self._in_utt:
            if is_speech:
                self._speech_run += 1
                self._pending.append(win_bytes)
                if self._speech_run >= self._open_win:
                    self._in_utt, self._utt, self._pending = True, self._pending, []
                    self._silence_run = 0
                    events.append(VadOpen(t_ms))
            else:
                self._speech_run, self._pending = 0, []
        else:
            self._utt.append(win_bytes)
            self._silence_run = 0 if is_speech else self._silence_run + 1
            if self._silence_run >= self._close_win or len(self._utt) >= self._max_utt_win:
                keep = len(self._utt) - self._silence_run    # drop the trailing close-silence
                events.append(Utterance(b"".join(self._utt[:keep]), keep * _VAD_WIN_MS))
                self._in_utt, self._utt = False, []
                self._silence_run = self._speech_run = 0
        return events

    def feed(self, frame: bytes) -> list:
        if len(frame) != FRAME_BYTES:
            raise ValueError(f"expected a {FRAME_BYTES}-byte frame, got {len(frame)}")
        events: list = []
        if self._dtmf is not None:
            digit = self._dtmf.feed(frame)
            if digit is not None:
                events.append(Dtmf(digit))
        if self._vad:
            self._buf = np.concatenate([self._buf, np.frombuffer(frame, dtype="<i2")])
            while len(self._buf) >= _VAD_WIN:
                win, self._buf = self._buf[:_VAD_WIN], self._buf[_VAD_WIN:]
                events += self._vad_window(win)
        return events
