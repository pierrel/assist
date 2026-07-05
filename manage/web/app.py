"""FastAPI app instance.

Lives in its own module so the route submodules (``diff``, ``review``,
``threads``, ``evals``) can ``from manage.web.app import app`` to
register their endpoints without circular-import gymnastics.
"""
import os

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from manage.web.state import lifespan

app = FastAPI(title="Assist Web", lifespan=lifespan)

# Serve the PWA manifest + home-screen icons (the only static assets — everything
# else is inlined).  StaticFiles reads on the threadpool (off the event loop) and
# rejects `..` traversal by construction; the dir holds only the manifest + PNGs.
app.mount("/static",
          StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")),
          name="static")
