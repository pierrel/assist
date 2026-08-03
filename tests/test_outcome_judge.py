from __future__ import annotations

import hashlib
import json
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import ValidationError

from edd.outcome_judge import (
    Evidence,
    ForbiddenVerdict,
    OutcomeJudge,
    OutcomeObservation,
    OutcomeRequirement,
    OutcomeVerdict,
    RequestedVerdict,
    validate_verdict,
)


def _observation() -> OutcomeObservation:
    return OutcomeObservation(
        requested=(OutcomeRequirement(
            id="ready", description="The result is ready.",
            evidence_ids=("final:result.txt",),
        ),),
        evidence=(
            Evidence(
                id="prompt:0", kind="prompt", state="present",
                content="Write ready to result.txt.",
            ),
            Evidence(
                id="final:result.txt", kind="final", state="present",
                content="ready\n",
            ),
            Evidence(
                id="response:final", kind="response", state="present",
                content="Done.",
            ),
        ),
    )


def _verdict(**changes) -> OutcomeVerdict:
    value = {
        "overall": "pass",
        "requested": (
            RequestedVerdict(
                id="ready", grade="satisfied",
                evidence_ids=("final:result.txt",),
            ),
        ),
        "forbidden": (),
        "contradictions": (),
        "material_unrelated_evidence_ids": (),
        "unsafe_extra_evidence_ids": (),
        "rationale": "The file contains the requested value.",
        "rationale_evidence_ids": ("final:result.txt",),
        "confidence": "high",
    }
    value.update(changes)
    return OutcomeVerdict(**value)


def _message(content=None, **metadata) -> AIMessage:
    return AIMessage(
        content=_verdict().model_dump_json() if content is None else content,
        response_metadata={
            "finish_reason": "stop",
            "model_name": "served-model",
            **metadata,
        },
    )


def test_models_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        OutcomeObservation.model_validate({
            **_observation().model_dump(), "expected": "pass",
        })


@pytest.mark.parametrize("field", ["rationale", "contradictions"])
def test_verdict_rejects_empty_explanations(field: str) -> None:
    changes = {field: ""} if field == "rationale" else {
        field: ({"description": "", "evidence_ids": ("final:result.txt",)},)
    }
    with pytest.raises(ValidationError):
        _verdict(**changes)


def test_observation_rejects_duplicate_and_unknown_evidence() -> None:
    evidence = _observation().evidence
    with pytest.raises(ValidationError, match="unique"):
        OutcomeObservation(
            requested=_observation().requested,
            evidence=(*evidence, evidence[0]),
        )
    with pytest.raises(ValidationError, match="unknown"):
        OutcomeObservation(
            requested=(OutcomeRequirement(
                id="ready", description="Ready.", evidence_ids=("ghost",),
            ),),
            evidence=evidence,
        )
    with pytest.raises(ValidationError, match="outcome evidence IDs"):
        OutcomeObservation(
            requested=(OutcomeRequirement(
                id="ready", description="Ready.",
                evidence_ids=("final:result.txt", "final:result.txt"),
            ),),
            evidence=evidence,
        )


@pytest.mark.parametrize("field", ["evidence", "outcome"])
def test_observation_rejects_blank_ids(field: str) -> None:
    evidence = list(_observation().evidence)
    requested = _observation().requested
    if field == "evidence":
        evidence[0] = evidence[0].model_copy(update={"id": "  "})
    else:
        requested = (requested[0].model_copy(update={"id": "  "}),)
    with pytest.raises(ValidationError, match="must not be blank"):
        OutcomeObservation(requested=requested, evidence=tuple(evidence))


def test_observation_rejects_duplicate_outcomes_and_inconsistent_state() -> None:
    outcome = _observation().requested[0]
    with pytest.raises(ValidationError, match="unique"):
        OutcomeObservation(
            requested=(outcome, outcome), evidence=_observation().evidence,
        )
    with pytest.raises(ValidationError, match="disagree"):
        Evidence(
            id="final:missing.txt", kind="final", state="missing",
            content="unexpected content",
        )


@pytest.mark.parametrize(
    "verdict",
    [
        _verdict(requested=()),
        _verdict(requested=(RequestedVerdict(
            id="ready", grade="satisfied", evidence_ids=("response:final",),
        ),)),
        _verdict(requested=(RequestedVerdict(
            id="ready", grade="satisfied",
            evidence_ids=("final:result.txt", "final:result.txt"),
        ),)),
        _verdict(rationale_evidence_ids=("ghost",)),
        _verdict(overall="partial"),
    ],
)
def test_verdict_rejects_identity_citation_and_grade_errors(
    verdict: OutcomeVerdict,
) -> None:
    with pytest.raises(ValueError):
        validate_verdict(_observation(), verdict)


def test_extra_output_controls_partial_and_fail() -> None:
    partial = _verdict(
        overall="partial",
        material_unrelated_evidence_ids=("response:final",),
    )
    failed = _verdict(
        overall="fail", unsafe_extra_evidence_ids=("response:final",),
    )
    assert validate_verdict(_observation(), partial).overall == "partial"
    assert validate_verdict(_observation(), failed).overall == "fail"


def test_forbidden_outcome_controls_overall_grade() -> None:
    observation = _observation().model_copy(update={
        "forbidden": (OutcomeRequirement(
            id="sent", description="The result was sent.",
            evidence_ids=("response:final",),
        ),),
    })
    verdict = _verdict(
        overall="fail",
        forbidden=(ForbiddenVerdict(
            id="sent", grade="present", evidence_ids=("response:final",),
        ),),
    )
    assert validate_verdict(observation, verdict).overall == "fail"


def test_judge_makes_one_bounded_structured_call() -> None:
    selected = MagicMock()
    bound = selected.bind.return_value
    bound.invoke.return_value = _message()

    verdict = OutcomeJudge(selected).judge(_observation())

    assert verdict.overall == "pass"
    selected.bind.assert_called_once()
    bind = selected.bind.call_args.kwargs
    assert bind["max_tokens"] == 4096
    assert bind["seed"] == 0
    assert bind["response_format"]["json_schema"]["strict"] is True
    bound.invoke.assert_called_once()
    messages = bound.invoke.call_args.args[0]
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)
    assert "UNTRUSTED OUTCOME OBSERVATION" in messages[1].content


def test_judge_records_response_model_and_exact_prompt() -> None:
    selected = MagicMock()
    selected.bind.return_value.invoke.return_value = _message()

    with patch("edd.outcome_judge._PROMPT_PATH") as prompt_path:
        prompt_path.read_text.return_value = "exact prompt"
        result = OutcomeJudge(selected).judge_with_provenance(_observation())

    assert result.model == "served-model"
    assert result.prompt_sha256 == hashlib.sha256(b"exact prompt").hexdigest()


@pytest.mark.parametrize(
    "metadata",
    [
        {"model_name": ""},
        {"model_name": "   "},
        {"finish_reason": "length"},
    ],
)
def test_judge_rejects_incomplete_provenance(metadata) -> None:
    selected = MagicMock()
    selected.bind.return_value.invoke.return_value = _message(**metadata)
    with pytest.raises(ValueError):
        OutcomeJudge(selected).judge_with_provenance(_observation())


def test_default_model_is_temperature_zero_with_thinking() -> None:
    selected = MagicMock()
    selected.bind.return_value.invoke.return_value = _message()
    with patch("assist.model_manager.select_chat_model", return_value=selected) as pick:
        OutcomeJudge().judge(_observation())
    pick.assert_called_once_with(0, enable_thinking=True)


@pytest.mark.parametrize("content", ["not json", [{"text": "not text"}]])
def test_model_output_errors_propagate(content) -> None:
    selected = MagicMock()
    selected.bind.return_value.invoke.return_value = _message(content)
    with pytest.raises((TypeError, ValidationError)):
        OutcomeJudge(selected).judge(_observation())


def test_observation_payload_omits_labels_and_identity() -> None:
    selected = MagicMock()
    selected.bind.return_value.invoke.return_value = _message()
    OutcomeJudge(selected).judge(_observation())
    message = selected.bind.return_value.invoke.call_args.args[0][1]
    payload = message.content.splitlines()[1]
    assert set(json.loads(payload)) == {"requested", "forbidden", "evidence"}
