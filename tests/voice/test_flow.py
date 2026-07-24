"""Deterministic tests for the hermetic flow engine (voice-call assistant, P1).

The DTMF detector is pure DSP — no ML model, no wall clock — so we synthesize exact
tones and drive them frame-by-frame. Time is the frame count; every assertion is
reproducible.
"""
import os

import numpy as np

from manage.voice.flow import (
    FRAME_BYTES, FRAME_SAMPLES, SAMPLE_RATE, DtmfDetector, _DIGITS,
)

# (low, high) tone pair for each digit, inverted from the module's map.
_PAIR = {d: pair for pair, d in _DIGITS.items()}


def _tone_frames(digit, n_frames, *, amp=0.3):
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
    # a real inter-digit release (≥ release_frames of silence) separates two presses
    frames = _tone_frames("5", 4) + [_silence_frame()] * 2 + _tone_frames("5", 4)
    assert _feed(det, frames) == ["5", "5"]


def test_held_key_survives_a_frame_dropout():
    # A single glitched frame mid-hold (lossy codec) must NOT split a held key into two
    # digits — only a full release_frames gap re-arms.
    det = DtmfDetector()
    frames = _tone_frames("5", 3) + [_silence_frame()] + _tone_frames("5", 3)
    assert _feed(det, frames) == ["5"]             # one keypress despite the dropout


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


# ── endpointing / barge-in (deterministic via scripted VAD probabilities) ──────

from manage.voice.flow import (  # noqa: E402
    _VAD_WIN, BargeIn, Flow, Utterance, VadOpen,
)

_ZWIN = np.zeros(_VAD_WIN, dtype=np.int16)


class _ScriptedFlow(Flow):
    """Flow whose VAD probabilities are scripted — tests the endpointing/barge-in
    timing logic with zero dependence on the ONNX model (never loaded)."""

    def __init__(self, probs, **kw):
        super().__init__(**kw)
        self._scripted = list(probs)
        self._i = 0

    def _speech_prob(self, window):
        p = self._scripted[self._i] if self._i < len(self._scripted) else 0.0
        self._i += 1
        return p


def _windows(flow, probs):
    """Drive `len(probs)` VAD windows; return the events (probs come from the flow)."""
    out = []
    for _ in probs:
        out += flow._vad_window(_ZWIN)
    return out


def _types(events, kind):
    return [e for e in events if isinstance(e, kind)]


def test_endpoints_a_full_utterance():
    # 200ms open (7 win) then 700ms trailing silence (22 win) ⇒ one open + one utterance.
    probs = [0.9] * 10 + [0.1] * 25
    flow = _ScriptedFlow(probs)
    events = _windows(flow, probs)
    assert len(_types(events, VadOpen)) == 1
    utts = _types(events, Utterance)
    assert len(utts) == 1 and utts[0].dur_ms > 0


def test_short_speech_never_opens():
    # 3 speech windows (~96ms) is below the 200ms open floor ⇒ nothing.
    probs = [0.9] * 3 + [0.1] * 10
    events = _windows(_ScriptedFlow(probs), probs)
    assert _types(events, VadOpen) == [] and _types(events, Utterance) == []


def test_barge_in_only_while_speaking():
    probs = [0.9] * 12
    on = _windows(_ScriptedFlow(probs, speaking=True), probs)
    off = _windows(_ScriptedFlow(probs, speaking=False), probs)
    assert len(_types(on, BargeIn)) == 1        # sustained speech while speaking fires once
    assert _types(off, BargeIn) == []           # not while the agent is silent


def test_barge_in_needs_sustain():
    # 5 speech windows (~160ms) is below the 300ms sustain floor ⇒ no barge-in.
    probs = [0.9] * 5 + [0.1] * 5
    events = _windows(_ScriptedFlow(probs, speaking=True), probs)
    assert _types(events, BargeIn) == []


def test_set_speaking_rearms_barge_in():
    flow = _ScriptedFlow([0.9] * 12 + [0.9] * 12, speaking=True)
    first = _windows(flow, [0.9] * 12)
    flow.set_speaking(False)                    # reply ends
    flow.set_speaking(True)                     # next reply starts
    second = _windows(flow, [0.9] * 12)
    assert len(_types(first, BargeIn)) == 1 and len(_types(second, BargeIn)) == 1


def test_max_utterance_cap_closes():
    # Continuous speech never hits trailing silence ⇒ the max-utterance cap closes it.
    probs = [0.9] * 15
    flow = _ScriptedFlow(probs, max_utt_ms=320)   # cap = 10 windows
    events = _windows(flow, probs)
    assert len(_types(events, Utterance)) == 1


def test_feed_buffers_frames_into_windows():
    # 16 frames × 320 samples = 5120 samples = exactly 10 VAD windows.
    class _Counter(Flow):
        def _speech_prob(self, window):
            return 0.0
    flow = _Counter()
    for _ in range(16):
        flow.feed(_silence_frame())
    assert flow._windows == 16 * FRAME_SAMPLES // _VAD_WIN   # 10


def test_dtmf_scope_only():
    # A GREETING_PIN-scoped flow (vad=False) still detects DTMF and NEVER touches VAD
    # (no STT/model pre-auth, §8) — _speech_prob raising proves it's never invoked.
    class _NoVad(Flow):
        def _speech_prob(self, window):
            raise AssertionError("VAD must not run in a DTMF-only scope")
    import manage.voice.flow as flowmod
    flow = _NoVad(vad=False)
    digits = []
    for f in _tone_frames("7", 5):
        digits += [e.digit for e in flow.feed(f) if isinstance(e, flowmod.Dtmf)]
    assert digits == ["7"]


def test_real_silero_silence_no_utterance():
    # The real ONNX model (lazy-loaded) on silence: near-zero prob ⇒ no utterance.
    flow = Flow()                               # vad=True, real silero
    events = []
    for _ in range(60):                         # ~1.2s of silence
        events += flow.feed(_silence_frame())
    assert _types(events, Utterance) == [] and _types(events, VadOpen) == []


def test_real_silero_fires_on_speech():
    # The real ONNX model on a real (piper-synthesized) speech clip: silero fires and
    # the utterance endpoints. Closes the model-fires integration gap the scripted
    # tests can't cover; the clip is a committed fixture (piper 16kHz s16le).
    import wave
    fix = os.path.join(os.path.dirname(__file__), "fixtures", "speech_16k.wav")
    with wave.open(fix, "rb") as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16000, 1, 2)
        pcm = w.readframes(w.getnframes())
    flow = Flow()
    events = []
    for i in range(0, len(pcm) - FRAME_BYTES + 1, FRAME_BYTES):
        events += flow.feed(pcm[i:i + FRAME_BYTES])
    for _ in range(50):                      # ≥700ms trailing silence forces the endpoint close
        events += flow.feed(_silence_frame())
    assert len(_types(events, VadOpen)) >= 1        # silero fired on the speech
    assert len(_types(events, Utterance)) >= 1      # and the utterance endpointed
