"""Opt-in four-value judge mechanics; behavioral grades come from frozen panels."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from edd import four_value_codex
from edd.four_value_judge import (
    CombinedAssessment, ValueAssessment, combine_grades,
    validate_assessment, validate_combined,
)
from edd.outcome_judge import Evidence, OutcomeObservation, OutcomeRequirement


def _observation() -> OutcomeObservation:
    return OutcomeObservation(
        requested=(OutcomeRequirement(id="update", description="Update the note.",
                                      evidence_ids=("initial", "final")),),
        evidence=(
            Evidence(id="request", kind="prompt", state="present", content="Update."),
            Evidence(id="initial", kind="initial", state="present", content="Unchecked."),
            Evidence(id="source", kind="event", state="present", content="Passed."),
            Evidence(id="final", kind="final", state="present", content="Updated."),
        ),
    )


def _assessments():
    return tuple(ValueAssessment(value=value, grade="Meets", evidence_ids=(evidence,),
                                 reason="Supported.", uncertainty="")
                 for value, evidence in (("Thorough", "final"),
                                         ("Focused", "request"), ("Correct", "source")))


def _combined(assessments):
    return {"overall": "Meets",
            "value_grades": [{"value": a.value, "grade": a.grade} for a in assessments],
            "evidence_ids": ["final", "source"], "disagreements": [],
            "reason": "Assessments agree."}


def test_citations_uncertainty_and_aggregation_policy():
    obs = _observation()
    assessments = _assessments()
    assert validate_assessment(obs, "Thorough", assessments[0])
    with pytest.raises(ValueError, match="citations"):
        validate_assessment(obs, "Thorough", assessments[0].model_copy(
            update={"evidence_ids": ("ghost",)}))
    with pytest.raises(ValueError, match="unknown"):
        validate_assessment(obs, "Thorough", assessments[0].model_copy(
            update={"grade": "Uncertain"}))
    good = CombinedAssessment.model_validate(_combined(assessments))
    assert validate_combined(obs, assessments, good) == good
    with pytest.raises(ValueError, match="policy"):
        validate_combined(obs, assessments, good.model_copy(
            update={"overall": "Partly meets"}))
    assert combine_grades(("Meets", "Uncertain", "Meets")) == "Uncertain"
    assert combine_grades(("Does not meet", "Uncertain", "Meets")) == "Does not meet"


def test_measured_wrapper_text_and_independent_order(tmp_path, monkeypatch):
    obs = _observation().model_dump(mode="json")
    role = (Path(four_value_codex.__file__).with_name("four_value_prompts")
            / "thorough.md").read_text()
    assert four_value_codex._prompt("Thorough", obs, []) == (
        "Apply the following independent research-value assessor instruction to "
        "the observation. Return only the JSON schema object. Do not use tools, "
        "browse, edit files, or look anything up beyond this observation.\n"
        "BEGIN ASSESSOR INSTRUCTION\n" + role + "\nEND ASSESSOR INSTRUCTION\n"
        "BEGIN UNTRUSTED OBSERVATION\n" + json.dumps(obs, ensure_ascii=False)
        + "\nEND UNTRUSTED OBSERVATION\n")
    source = tmp_path / "observation.json"
    source.write_text(json.dumps(obs))
    calls = []
    answers = _assessments()

    def fake_call(role_name, prompt, folder, **kwargs):
        calls.append((role_name, prompt))
        folder.mkdir()
        answer = (_combined(answers) if role_name == "aggregator" else
                  answers[len(calls)-1].model_dump(mode="json"))
        return {"role": role_name, "status": "pending_validation", "answer": answer}

    monkeypatch.setattr(four_value_codex, "_call", fake_call)
    summary = four_value_codex.run(source, tmp_path / "output", model="gpt-6.1-sol",
                                   effort="medium", cwd=tmp_path, cap_seconds=180,
                                   codex="codex")
    assert [x[0] for x in calls] == ["thorough", "focused", "correct", "aggregator"]
    assert all("BEGIN ASSESSMENTS" not in x[1] for x in calls[:3])
    assert "BEGIN ASSESSMENTS" in calls[3][1]
    assert summary["status"] == "validated"
    assert summary["overall"] == "Meets"
    assert [item["grade"] for item in summary["roles"][:3]] == ["Meets"] * 3


def test_failed_assessor_stops_without_overall(tmp_path, monkeypatch):
    source = tmp_path / "observation.json"
    source.write_text(_observation().model_dump_json())
    calls = []

    def fake_call(role_name, prompt, folder, **kwargs):
        calls.append(role_name)
        folder.mkdir()
        return {"role": role_name, "status": "transport_error"}

    monkeypatch.setattr(four_value_codex, "_call", fake_call)
    summary = four_value_codex.run(source, tmp_path / "nested/output", model="gpt-6.1-sol",
                                   effort="medium", cwd=tmp_path, cap_seconds=180,
                                   codex="codex")
    assert calls == ["thorough"]
    assert summary["status"] == "nonverdict"
    assert "overall" not in summary
    assert not (tmp_path / "nested/output" / "aggregator").exists()


@pytest.mark.parametrize("malformed", [None, []])
def test_nonobject_answer_is_recorded_without_overall(tmp_path, monkeypatch,
                                                      malformed):
    source = tmp_path / "observation.json"
    source.write_text(_observation().model_dump_json())

    def fake_call(role_name, prompt, folder, **kwargs):
        folder.mkdir()
        return {"role": role_name, "status": "pending_validation",
                "answer": malformed}

    monkeypatch.setattr(four_value_codex, "_call", fake_call)
    summary = four_value_codex.run(source, tmp_path / "output",
                                   model="gpt-6.1-sol", effort="medium",
                                   cwd=tmp_path, cap_seconds=180, codex="codex")
    assert summary["roles"][0]["status"] == "invalid_assessor"
    assert summary["roles"][0]["grade"] is None
    assert summary["status"] == "nonverdict"
    assert "overall" not in summary


def test_relative_paths_resolve_before_provider_invocation(tmp_path, monkeypatch):
    caller = tmp_path / "caller"
    provider = tmp_path / "provider"
    caller.mkdir()
    provider.mkdir()
    (tmp_path / "observation.json").write_text(_observation().model_dump_json())
    monkeypatch.chdir(caller)
    seen = []

    def fake_call(role_name, prompt, folder, **kwargs):
        seen.append((folder, kwargs["cwd"]))
        folder.mkdir()
        return {"role": role_name, "status": "transport_error"}

    monkeypatch.setattr(four_value_codex, "_call", fake_call)
    four_value_codex.run(Path("../observation.json"), Path("output"),
                         model="gpt-6.1-sol", effort="medium",
                         cwd=Path("../provider"), cap_seconds=180, codex="codex")
    assert seen == [(caller / "output/thorough", provider)]


def test_codex_call_records_model_mismatch(tmp_path, monkeypatch):
    def fake_run(command, **kwargs):
        Path(command[command.index("--output-last-message") + 1]).write_text("{}")
        return SimpleNamespace(returncode=0, stderr="", stdout=(
            '{"type":"thread.started","thread_id":"t"}\n'
            '{"type":"turn.completed","usage":{}}\n'))

    monkeypatch.setattr(four_value_codex.subprocess, "run", fake_run)
    monkeypatch.setattr(four_value_codex, "_observed_model", lambda _: "other-model")
    record = four_value_codex._call("focused", "prompt", tmp_path / "role",
                                    codex="codex", model="gpt-6.1-sol",
                                    effort="medium", cwd=tmp_path, cap_seconds=180)
    assert record["status"] == "model_identity_error"
    assert record["requested_model"] != record["observed_model"]


def test_codex_launch_error_is_nonverdict(tmp_path, monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError("missing codex")

    monkeypatch.setattr(four_value_codex.subprocess, "run", missing)
    record = four_value_codex._call("focused", "prompt", tmp_path / "role",
                                    codex="codex", model="gpt-6.1-sol",
                                    effort="medium", cwd=tmp_path, cap_seconds=180)
    assert record["status"] == "transport_error"
    assert (tmp_path / "role/record.json").exists()


def test_nonobject_jsonl_after_cli_error_still_records_nonverdict(tmp_path,
                                                                  monkeypatch):
    def fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=1, stderr="CLI failed", stdout="null\n[]\n")

    monkeypatch.setattr(four_value_codex.subprocess, "run", fake_run)
    record = four_value_codex._call("correct", "prompt", tmp_path / "role",
                                    codex="codex", model="gpt-6.1-sol",
                                    effort="medium", cwd=tmp_path, cap_seconds=180)
    assert record["status"] == "transport_error"
    assert json.loads((tmp_path / "role/record.json").read_text())["status"] \
           == "transport_error"


@pytest.mark.parametrize("session_bytes", (
    b"null\n", b"[]\n", b'{"type":"turn_context","payload":[]}\n',
    b"\xff\n",
))
def test_malformed_session_metadata_records_nonverdict(tmp_path, monkeypatch,
                                                       session_bytes):
    sessions = tmp_path / "codex-home/sessions"
    sessions.mkdir(parents=True)
    (sessions / "session-t.jsonl").write_bytes(session_bytes)
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex-home"))

    def fake_run(command, **kwargs):
        Path(command[command.index("--output-last-message") + 1]).write_text("{}")
        return SimpleNamespace(returncode=0, stderr="", stdout=(
            '{"type":"thread.started","thread_id":"t"}\n'
            '{"type":"turn.completed","usage":{}}\n'))

    monkeypatch.setattr(four_value_codex.subprocess, "run", fake_run)
    record = four_value_codex._call("correct", "prompt", tmp_path / "role",
                                    codex="codex", model="gpt-6.1-sol",
                                    effort="medium", cwd=tmp_path, cap_seconds=180)
    assert record["status"] == "model_identity_error"
    assert record["observed_model"] is None
    assert json.loads((tmp_path / "role/record.json").read_text())["status"] \
           == "model_identity_error"


def test_malformed_item_event_is_nonverdict(tmp_path, monkeypatch):
    def fake_run(command, **kwargs):
        Path(command[command.index("--output-last-message") + 1]).write_text("{}")
        return SimpleNamespace(returncode=0, stderr="", stdout=(
            '{"type":"thread.started","thread_id":"t"}\n'
            '{"type":"item.completed","item":[]}\n'
            '{"type":"turn.completed","usage":{}}\n'))

    monkeypatch.setattr(four_value_codex.subprocess, "run", fake_run)
    monkeypatch.setattr(four_value_codex, "_observed_model", lambda _: "gpt-6.1-sol")
    record = four_value_codex._call("correct", "prompt", tmp_path / "role",
                                    codex="codex", model="gpt-6.1-sol",
                                    effort="medium", cwd=tmp_path, cap_seconds=180)
    assert record["status"] == "unexpected_tool_use"


def test_invented_calibration_packet_keeps_host_labels_out_of_observations():
    path = Path(__file__).resolve().parents[1] / (
        "edd/eval/fixtures/outcome-judge/four_value_synthetic.json")
    packet = json.loads(path.read_text())
    assert len(packet["cases"]) == 12
    assert {pair["set"] for pair in packet["pairs"].values()} == {
        "tuning", "holdout"}
    for item in packet["cases"]:
        observation = item["observation"]
        OutcomeObservation.model_validate(observation)
        assert "expected" not in json.dumps(observation).lower()
