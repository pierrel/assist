"""Identity proof for prompt-rewrite eval baseline and candidate profiles."""

from deepagents.graph import BASE_AGENT_PROMPT

from edd.prompt_census import capture_prompt_rewrite_profiles


def test_prompt_rewrite_profile_capture_uses_the_eval_helper_on_both_sides():
    profiles = capture_prompt_rewrite_profiles()
    baseline = profiles["baseline"]
    candidate = profiles["candidate"]
    assert baseline["candidate_env"] is False
    assert candidate["candidate_env"] is True
    assert baseline["initial_text"].startswith(
        "You are a helpful, proactive assistant that works within the user's local filesystem.")
    assert baseline["final_text"].startswith(
        "You are a helpful, proactive assistant that works within the user's local filesystem.")
    assert candidate["initial_text"].startswith("## Purpose\n\nYou are Assist")
    assert candidate["final_text"].startswith(
        f"{BASE_AGENT_PROMPT}\n\n## Purpose\n\nYou are Assist")
    assert "assist.PromptCompositionMiddleware" not in baseline["transition_owners"]
    assert "assist.PromptCompositionMiddleware" in candidate["transition_owners"]
    assert baseline["initial_sha256"] != candidate["initial_sha256"]
    assert baseline["final_sha256"] != candidate["final_sha256"]
