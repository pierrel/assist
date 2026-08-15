"""Host-owned web-main product prompt assembly for both agent engines."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from jinja2 import TemplateError

from assist.promptable import base_prompt_for


_MAX_BYTES = 64 * 1024
_SHARED_TEMPLATES = (
    "prompt/web_main_purpose.md.j2",
    "prompt/web_main_evidence.md.j2",
)


class WebMainPromptError(RuntimeError):
    """The trusted web-main prompt could not be rendered safely."""


class WebMainPromptUnavailable(WebMainPromptError):
    """Trusted prompt source could not be read or rendered."""


class WebMainPromptInvalid(WebMainPromptError):
    """Trusted prompt source rendered outside its required contract."""


@dataclass(frozen=True)
class WebMainPrompt:
    """One rendered Assist static prompt with attributed policy fragments."""

    text: str
    shared_fragments: tuple[str, ...]
    adapter: str

    @property
    def shared_core_sha256(self) -> str:
        """Digest the ordered common policy without conflating engine adapters."""
        return _sha("\0".join(self.shared_fragments))

    @property
    def adapter_sha256(self) -> str:
        """Digest the engine-only capability and lifecycle guidance."""
        return _sha(self.adapter)

    @property
    def sha256(self) -> str:
        """Digest this renderer's exact static Assist text."""
        return _sha(self.text)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _render(template: str, **kwargs: object) -> str:
    try:
        text = base_prompt_for(template, **kwargs)
    except (OSError, TemplateError, UnicodeError) as error:
        raise WebMainPromptUnavailable("web-main prompt is unavailable") from error
    if not text.strip() or len(text.encode("utf-8")) > _MAX_BYTES:
        raise WebMainPromptInvalid("web-main prompt is invalid")
    return text


def _assemble(template: str, adapter_template: str, **kwargs: object) -> WebMainPrompt:
    shared = tuple(_render(source) for source in _SHARED_TEMPLATES)
    adapter = _render(adapter_template, **kwargs)
    text = _render(template, **kwargs)
    components = (*zip(_SHARED_TEMPLATES, shared), (adapter_template, adapter))
    for source, fragment in components:
        if text.count(fragment) != 1:
            raise WebMainPromptInvalid(
                f"web-main prompt composition is invalid: {source}")
    return WebMainPrompt(text=text, shared_fragments=shared, adapter=adapter)


def render_deep_web_main_prompt(*, guidance_skills: bool) -> WebMainPrompt:
    """Render the Deep Agents web-main prompt without changing its text order."""
    return _assemble(
        "deepagents/assist_core.md.j2",
        "deepagents/web_main_adapter.md.j2",
        guidance_skills=guidance_skills,
    )


def render_pi_web_main_prompt() -> WebMainPrompt:
    """Render Pi's host-owned web-main prompt for one fresh bounded worker."""
    return _assemble("pi/system.md.j2", "pi/web_main_adapter.md.j2")
