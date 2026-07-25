"""The voice client (voice-call assistant, P1). A thin client over the shared
run service + turn-observer seam and the receptionist (P0.5). See
docs/2026-07-23-voice-p1.org for the build plan and threading model.

P1 includes ``flow.py`` (the hermetic frame-in/event-out engine), ``speech.py``
(lazy in-process STT/TTS), and ``wire.py`` (the bounded WSS /call transport,
the only asyncio-loop code and a pure frame shuttle). The remaining work adds
session policy and the final security/reconstruction gate.
"""
