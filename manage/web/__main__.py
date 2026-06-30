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


if __name__ == "__main__":
    os.makedirs(ROOT, exist_ok=True)
    port = int(os.getenv("ASSIST_PORT", "8000"))
    # Optional TLS: set ASSIST_SSL_CERT + ASSIST_SSL_KEY to serve HTTPS directly.
    # Needed so the browser exposes geolocation over a non-localhost address (e.g.
    # the WireGuard IP) — geolocation requires a secure context (HTTPS or localhost).
    # Use a trusted local cert (mkcert) so there's no browser warning.
    ssl_kwargs = {}
    cert, key = os.getenv("ASSIST_SSL_CERT"), os.getenv("ASSIST_SSL_KEY")
    if cert and key:
        ssl_kwargs = {"ssl_certfile": cert, "ssl_keyfile": key}
    uvicorn.run(
        "manage.web:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        reload=False,
        **ssl_kwargs,
    )
