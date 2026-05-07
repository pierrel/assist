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
    uvicorn.run(
        "manage.web:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        reload=False,
    )
