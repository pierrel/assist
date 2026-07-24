"""The voice client (voice-call assistant, P1). A thin client over the shared
session API (P0's plan_turn/route_turn + the turn-observer seam) and the receptionist
(P0.5). See docs/2026-07-23-voice-p1.org for the build plan and threading model.

flow.py is the hermetic frame-in/event-out engine; session.py the per-call state
machine; wire.py the WSS /call endpoint; speech.py the in-proc STT/TTS. Nothing here
touches the asyncio event loop except wire.py's pure frame-shuttle.
"""
