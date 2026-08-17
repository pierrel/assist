"""Shared web-main prompt ownership and engine-specific capability boundaries."""
from __future__ import annotations

import os

import pytest
from jinja2 import FileSystemLoader

from assist import web_main_prompt
from assist import promptable


@pytest.mark.parametrize(("guidance_skills", "expected_sha256"), [
    (False, "5ecea1756711f918f6a02ad7fba118e4a72b27f99b3bd13c3e81154a02199501"),
    (True, "f4d2c564ed7c034fa6473150f0b6784b1943e39e88ae970889aac14d7e38019c"),
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


def test_shared_prompt_sources_auto_reload_for_later_turns(tmp_path, monkeypatch):
    templates = {
        "prompt/web_main_purpose.md.j2": "first shared purpose\n",
        "prompt/web_main_evidence.md.j2": "shared evidence\n",
        "pi/web_main_adapter.md.j2": "Pi adapter\n",
        "pi/system.md.j2": (
            '{% include "prompt/web_main_purpose.md.j2" %}{{ "\\n" }}\n'
            '{% include "pi/web_main_adapter.md.j2" %}{{ "\\n" }}\n'
            '{% include "prompt/web_main_evidence.md.j2" %}\n'),
    }
    for name, content in templates.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    purpose = tmp_path / "prompt/web_main_purpose.md.j2"
    monkeypatch.setattr(promptable.env, "loader", FileSystemLoader(str(tmp_path)))
    monkeypatch.setattr(promptable.env, "cache", {})

    assert "first shared purpose" in web_main_prompt.render_pi_web_main_prompt().text
    purpose.write_text("second shared purpose\n", encoding="utf-8")
    stat = purpose.stat()
    os.utime(purpose, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))

    assert "second shared purpose" in web_main_prompt.render_pi_web_main_prompt().text


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


def test_composition_failure_names_its_template_without_prompt_content(monkeypatch):
    original = web_main_prompt.base_prompt_for
    purpose = original("prompt/web_main_purpose.md.j2")

    def duplicated_purpose(template, **kwargs):
        text = original(template, **kwargs)
        if template == "pi/system.md.j2":
            return f"{purpose}\n{text}"
        return text

    monkeypatch.setattr(web_main_prompt, "base_prompt_for", duplicated_purpose)

    with pytest.raises(
            web_main_prompt.WebMainPromptError,
            match=r"invalid: prompt/web_main_purpose\.md\.j2"):
        web_main_prompt.render_pi_web_main_prompt()
