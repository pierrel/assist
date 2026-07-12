"""Strip image content from model requests — the local model is text-only.

The deepagents filesystem backend maps image extensions (``.png`` / ``.jpg`` / …) to an
``image`` file type, so ``read_file`` on an image asset returns an *image content block*.
Sent to the text-only llama.cpp server (no ``mmproj`` loaded), that block makes the whole
request fail with a 500 ("image input is not supported"), and after the retry layers give
up the turn dies. This bit a larochelle.io content turn that wrote + read a QR wallpaper
PNG (thread 20260711112029).

Guidance alone can't carry this — the failure is at the model-call layer, not the agent's
decision — so we guarantee it structurally: replace any image block with a short text note
right before the call, so an image can never reach the model. It's an unambiguous strip (the
text-only model literally cannot use image blocks), not a heuristic guard. The prompt also
tells the agent it can't see images so it doesn't bother reading them. If a vision model
(+ mmproj) is ever loaded, delete this middleware.
"""
import logging

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import AIMessage

logger = logging.getLogger(__name__)

_PLACEHOLDER = "[image omitted: this model is text-only and cannot view images]"
_IMAGE_TYPES = {"image", "image_url"}


def _has_image_block(content) -> bool:
    return isinstance(content, list) and any(
        isinstance(b, dict) and b.get("type") in _IMAGE_TYPES for b in content)


def _strip_images(content: list) -> list:
    """Replace each image block with a text note; keep every other block as-is."""
    return [{"type": "text", "text": _PLACEHOLDER}
            if isinstance(b, dict) and b.get("type") in _IMAGE_TYPES else b
            for b in content]


class ImageInputGuardMiddleware(AgentMiddleware):
    """Replace image content blocks with a text placeholder before the model call, so an
    image asset read into context can't 500 the text-only model. No-op when no message
    carries an image block (the common case)."""

    def wrap_model_call(self, request: ModelRequest, handler) -> ModelResponse | AIMessage:
        if not any(_has_image_block(getattr(m, "content", None)) for m in request.messages):
            return handler(request)
        new_msgs, stripped = [], 0
        for m in request.messages:
            content = getattr(m, "content", None)
            if _has_image_block(content):
                nm = m.model_copy() if hasattr(m, "model_copy") else m.copy()
                nm.content = _strip_images(content)
                stripped += 1
                new_msgs.append(nm)
            else:
                new_msgs.append(m)
        logger.info("ImageInputGuard: stripped image block(s) from %d message(s) — "
                    "text-only model", stripped)
        return handler(request.override(messages=new_msgs))
