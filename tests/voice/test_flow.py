"""Deterministic tests for the hermetic flow engine (voice-call assistant, P1).

The DTMF detector is pure DSP — no ML model, no wall clock — so we synthesize exact
tones and drive them frame-by-frame. Time is the frame count; every assertion is
reproducible.
"""
import numpy as np

from manage.voice.flow import (
    FRAME_BYTES, FRAME_SAMPLES, SAMPLE_RATE, DtmfDetector, _DIGITS,
)

# (low, high) tone pair for each digit, inverted from the module's map.
_PAIR = {d: pair for pair, d in _DIGITS.items()}


def _tone_frames(digit, n_frames, *, amp=0.3, seed=0):
    """`n_frames` 20 ms s16le frames of the two-sinusoid DTMF tone for `digit`."""
    low, high = _PAIR[digit]
    total = FRAME_SAMPLES * n_frames
    t = np.arange(total) / SAMPLE_RATE
    sig = amp * (np.sin(2 * np.pi * low * t) + np.sin(2 * np.pi * high * t)) / 2
    pcm = (sig * 32767).astype("<i2")
    return [pcm[i * FRAME_SAMPLES:(i + 1) * FRAME_SAMPLES].tobytes()
            for i in range(n_frames)]


def _silence_frame():
    return (np.zeros(FRAME_SAMPLES, dtype="<i2")).tobytes()


def _noise_frames(n_frames, *, amp=0.3, seed=1):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_frames):
        pcm = (rng.uniform(-amp, amp, FRAME_SAMPLES) * 32767).astype("<i2")
        out.append(pcm.tobytes())
    return out


def _feed(det, frames):
    return [d for d in (det.feed(f) for f in frames) if d is not None]


def test_each_digit_detected_once():
    for digit in _DIGITS.values():
        det = DtmfDetector()
        # a held key (5 frames = 100 ms) fires exactly once
        assert _feed(det, _tone_frames(digit, 5)) == [digit], f"digit {digit}"


def test_leading_edge_debounce_two_presses():
    det = DtmfDetector()
    frames = _tone_frames("5", 4) + [_silence_frame()] + _tone_frames("5", 4)
    assert _feed(det, frames) == ["5", "5"]        # gap between presses ⇒ two digits


def test_min_frames_gates_a_blip():
    det = DtmfDetector(min_frames=2)
    # a single-frame blip (20 ms) is below the 40 ms floor ⇒ nothing
    assert _feed(det, _tone_frames("7", 1) + [_silence_frame()]) == []


def test_silence_and_noise_reject():
    det = DtmfDetector()
    assert _feed(det, [_silence_frame()] * 5) == []
    det2 = DtmfDetector()
    assert _feed(det2, _noise_frames(10)) == []     # broadband noise is not a tone-pair


def test_speech_like_reject():
    # A low-frequency sweep (voiced-speech-like, not a stable tone-pair) must not fire.
    det = DtmfDetector()
    total = FRAME_SAMPLES * 10
    t = np.arange(total) / SAMPLE_RATE
    sweep = 0.3 * np.sin(2 * np.pi * (200 + 40 * t) * t)   # gliding pitch
    pcm = (sweep * 32767).astype("<i2")
    frames = [pcm[i * FRAME_SAMPLES:(i + 1) * FRAME_SAMPLES].tobytes() for i in range(10)]
    assert _feed(det, frames) == []


def test_tone_survives_additive_noise():
    # A real key over line noise: tone at amp 0.4 + noise at amp 0.05 still detects.
    det = DtmfDetector()
    low, high = _PAIR["3"]
    total = FRAME_SAMPLES * 5
    t = np.arange(total) / SAMPLE_RATE
    rng = np.random.default_rng(2)
    sig = 0.4 * (np.sin(2 * np.pi * low * t) + np.sin(2 * np.pi * high * t)) / 2
    sig = sig + rng.uniform(-0.05, 0.05, total)
    pcm = (np.clip(sig, -1, 1) * 32767).astype("<i2")
    frames = [pcm[i * FRAME_SAMPLES:(i + 1) * FRAME_SAMPLES].tobytes() for i in range(5)]
    assert _feed(det, frames) == ["3"]


def test_pin_sequence():
    # A realistic PIN punch: digits separated by short silences reconstruct in order.
    det = DtmfDetector()
    frames = []
    for d in "941073":
        frames += _tone_frames(d, 3) + [_silence_frame()] * 2
    assert _feed(det, frames) == list("941073")


def test_wrong_frame_size_raises():
    det = DtmfDetector()
    import pytest
    with pytest.raises(ValueError):
        det.feed(b"\x00" * (FRAME_BYTES - 2))
