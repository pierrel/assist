"""The voice client (voice-call assistant, P1). A thin client over the shared
run service + turn-observer seam and the receptionist
(P0.5). See docs/2026-07-23-voice-p1.org for the build plan and threading model.

P1 currently includes ``flow.py`` (the hermetic frame-in/event-out engine) and
``speech.py`` (lazy in-process STT/TTS). The remaining slices add ``session.py``
(the per-call state machine) and ``wire.py`` (the WSS /call endpoint — the only
asyncio-loop code, a pure frame shuttle).
"""
