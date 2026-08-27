"""Direct Qwen3.8 vision smoke for the local llama.cpp provider.

This sends a valid 32-by-32 solid-red PNG directly to the provider.  It does
not exercise Assist's text-only image guard: that middleware stays in place
until the Qwen3.8 provider itself has proved that it accepts and grounds image
content.
"""
from __future__ import annotations

import time
from unittest import TestCase

import httpx

from assist.model_manager import current_model_config


_RED_PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgAQMAAABJtOi3AAAAIGNIUk0AAHomAACAhAAA"
    "+gAAAIDoAAB1MAAA6mAAADqYAAAXcJy6UTwAAAAGUExURf8AAP///0EdNBEAAAABYktH"
    "RAH/Ai3eAAAADElEQVQI12NgGNwAAACgAAFhJX1HAAAAAElFTkSuQmCC"
)
_MAX_SECONDS = 120


class TestProviderVision(TestCase):
    """Require the 32k vision profile to recognize a known solid-color image."""

    def test_recognizes_red_png_at_32k(self):
        config = current_model_config()
        self.assertGreaterEqual(config.context_len, 32_768)
        headers = {"Authorization": f"Bearer {config.api_key}"}
        request = {
            "model": config.model,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "What color is this image? Answer with one word."},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{_RED_PNG}",
                }},
            ]}],
            "temperature": 0,
            "max_tokens": 32,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        started = time.monotonic()
        response = httpx.post(
            f"{config.url.rstrip('/')}/chat/completions",
            headers=headers,
            json=request,
            timeout=_MAX_SECONDS,
        )
        elapsed = time.monotonic() - started

        self.assertLessEqual(elapsed, _MAX_SECONDS)
        self.assertEqual(response.status_code, 200, response.text[:500])
        try:
            answer = response.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, ValueError) as error:
            self.fail(f"vision response was malformed: {error}")
        self.assertIn("red", str(answer).lower(), f"vision answer: {answer!r}")
        print(f"VISION-SMOKE answer={answer!r} completion_seconds={elapsed:.2f}")
