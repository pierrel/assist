"""Hermetic conversation-flow engine (voice-call assistant, P1;
docs/2026-07-23-voice-p1.org). Consumes 20 ms PCM frames, emits events. NO web
imports, no network, no threads, no wall clock — time is derived from the frame count,
which is what makes the fixture tests deterministic. The only I/O is loading the silero
VAD ONNX at construction (VAD/endpointing/barge-in land next; see the state doc).

This module first: ``DtmfDetector`` — in-band DTMF via the Goertzel single-bin power,
the one detector interface the design mandates as swappable (in-band today; a
modem-URC detector is the hardware-gated upgrade if AMR mangles in-band tones). It is
fully deterministic and dependency-light (numpy only), so it needs no ML model and is
testable today.
"""
from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16_000
FRAME_MS = 20
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000   # 320
FRAME_BYTES = FRAME_SAMPLES * 2                  # 640 (s16le)

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

    def __init__(self, *, min_frames: int = 2, energy_frac: float = 0.3,
                 group_dominance: float = 0.25, twist=(0.1, 10.0)):
        self._min_frames = min_frames
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
            self._cur, self._count, self._emitted = None, 0, False
            return None
        if cand == self._cur:
            self._count += 1
        else:
            self._cur, self._count, self._emitted = cand, 1, False
        if self._count >= self._min_frames and not self._emitted:
            self._emitted = True
            return cand
        return None
