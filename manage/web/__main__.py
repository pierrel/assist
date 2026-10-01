"""Package entry point for ``python -m manage.web``.

Lives in ``__main__.py`` so the systemd service (which spawns
``python -m manage.web``) keeps working after the split from a single
``web.py`` module.  The same code used to live inside an
``if __name__ == "__main__":`` block at the bottom of ``web.py`` —
that idiom only fires when a module is run directly, which a package
isn't.
"""
import os

import uvicorn

from manage.web.state import ROOT
from manage.voice.wire import MAX_WS_MESSAGE_BYTES


class TurnSafeServer(uvicorn.Server):
    """Route a signal during lifespan startup through normal shutdown."""

    def __init__(self, config: uvicorn.Config) -> None:
        super().__init__(config)
        self._main_loop_entered = False
        self._early_stop = False
        self._stop_received = False

    def handle_exit(self, sig, frame) -> None:
        if self._stop_received:
            return  # A repeated INT must not set Uvicorn's force_exit.
        self._stop_received = True
        self._captured_signals.append(sig)
        if self._main_loop_entered:
            self.should_exit = True
        else:
            self._early_stop = True

    async def main_loop(self) -> None:
        self._main_loop_entered = True
        if self._early_stop:
            self.should_exit = True
        await super().main_loop()


if __name__ == "__main__":
    os.makedirs(ROOT, exist_ok=True)
    port = int(os.getenv("ASSIST_PORT", "8000"))
    # Optional TLS: set ASSIST_SSL_CERT + ASSIST_SSL_KEY to serve HTTPS
    # directly. Needed so the browser exposes geolocation over a non-localhost
    # address: geolocation requires a secure context (HTTPS or localhost).
    # Use a trusted local cert (mkcert) so there's no browser warning.
    ssl_kwargs = {}
    cert, key = os.getenv("ASSIST_SSL_CERT"), os.getenv("ASSIST_SSL_KEY")
    if bool(cert) != bool(key):
        raise SystemExit(
            "Set BOTH ASSIST_SSL_CERT and ASSIST_SSL_KEY (or neither)."
        )
    if cert and key:
        ssl_kwargs = {"ssl_certfile": cert, "ssl_keyfile": key}
    server = TurnSafeServer(uvicorn.Config(
        "manage.web:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        reload=False,
        ws_max_size=MAX_WS_MESSAGE_BYTES,
        ws_per_message_deflate=False,
        **ssl_kwargs,
    ))
    try:
        server.run()
    except KeyboardInterrupt:
        pass  # Uvicorn replays captured SIGINT after shutdown.
    if not server.started:
        raise SystemExit(3)  # Uvicorn's startup-failure exit status.
