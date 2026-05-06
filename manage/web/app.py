"""FastAPI app instance.

Lives in its own module so the route submodules (``diff``, ``review``,
``threads``, ``evals``) can ``from manage.web.app import app`` to
register their endpoints without circular-import gymnastics.
"""
from fastapi import FastAPI

from manage.web.state import lifespan

app = FastAPI(title="Assist Web", lifespan=lifespan)
