"""The voice client (voice-call assistant, P1). A thin client over the shared
session API (P0's plan_turn/route_turn + the turn-observer seam) and the receptionist
(P0.5). See docs/2026-07-23-voice-p1.org for the build plan and threading model.

This PR ships ``flow.py`` — the hermetic frame-in/event-out engine. The rest of the
package lands in later P1 slices: ``session.py`` (per-call state machine), ``wire.py``
(the WSS /call endpoint — the only asyncio-loop code, a pure frame-shuttle), and
``speech.py`` (in-proc STT/TTS).
"""
