"""FastAPI app instance.

Lives in its own module so the route submodules (``diff``, ``review``,
``threads``, ``evals``) can ``from manage.web.app import app`` to
register their endpoints without circular-import gymnastics.
"""
import logging

from fastapi import FastAPI, Request
from starlette.responses import PlainTextResponse

from manage.web.state import lifespan

logger = logging.getLogger(__name__)

app = FastAPI(title="Assist Web", lifespan=lifespan)


@app.middleware("http")
async def phone_no_store(request: Request, call_next):
    """Keep every phone response, including unexpected failures, out of caches."""
    phone_prefix = "/api/v1/phone"
    is_phone_request = (request.url.path == phone_prefix
                        or request.url.path.startswith(f"{phone_prefix}/"))
    try:
        response = await call_next(request)
    except Exception:
        if not is_phone_request:
            raise
        logger.exception("Unhandled phone API request")
        return PlainTextResponse(
            "Internal Server Error", status_code=500,
            headers={"Cache-Control": "no-store"})
    if is_phone_request:
        response.headers["Cache-Control"] = "no-store"
    return response
