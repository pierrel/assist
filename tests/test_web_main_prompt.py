"""Shared web-main prompt ownership and engine-specific capability boundaries."""
from __future__ import annotations

import pytest

from assist import web_main_prompt


@pytest.mark.parametrize(("guidance_skills", "expected_sha256"), [
    (False, "8d4107f0107d72743de0b953e445d3973b3ba1d67bba79fb089b794ed862d176"),
    (True, "9b9386688c837c16cba7cc07ac89baab41fd15d7452fe8f19e7c6de419e63d78"),
])
def test_deep_web_main_prompt_preserves_the_merged_rewrite(
    guidance_skills, expected_sha256,
):
    prompt = web_main_prompt.render_deep_web_main_prompt(
        guidance_skills=guidance_skills)

    assert prompt.sha256 == expected_sha256


def test_both_engines_include_the_same_shared_policy_once():
    deep = web_main_prompt.render_deep_web_main_prompt(guidance_skills=True)
    pi = web_main_prompt.render_pi_web_main_prompt()

    assert deep.shared_fragments == pi.shared_fragments
    assert deep.shared_core_sha256 == pi.shared_core_sha256
    assert deep.adapter_sha256 != pi.adapter_sha256
    for fragment in deep.shared_fragments:
        assert deep.text.count(fragment) == 1
        assert pi.text.count(fragment) == 1


def test_pi_adapter_never_promises_deep_only_capabilities():
    prompt = web_main_prompt.render_pi_web_main_prompt().text

    for available in ("`read`", "`write`", "`edit`", "`bash`"):
        assert available in prompt
    for unavailable in (
        "`notify`", "`grounding`", "`research`", "`load_skill`",
        "`start_async_task`", "`research-agent`", "`complex-request`",
        "`orchestrate-repeated-work`",
    ):
        assert unavailable not in prompt
    assert "no skill loader, notification, research,\nor background-task capability" in prompt


def test_missing_or_malformed_template_fails_closed(monkeypatch):
    monkeypatch.setattr(web_main_prompt, "base_prompt_for", lambda *_args, **_kwargs: "")

    with pytest.raises(web_main_prompt.WebMainPromptError, match="invalid"):
        web_main_prompt.render_pi_web_main_prompt()
