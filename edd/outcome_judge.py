"""Small, eval-only natural-outcome judge."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field, model_validator


_PROMPT_PATH = Path(__file__).with_name("outcome_judge_prompt.md")
_OUTPUT_TOKENS = 4096


class _ClosedModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class OutcomeRequirement(_ClosedModel):
    id: str
    description: str
    evidence_ids: tuple[str, ...]


class Evidence(_ClosedModel):
    id: str
    kind: Literal["prompt", "initial", "final", "response", "event"]
    state: Literal["present", "missing"]
    content: str | None

    @model_validator(mode="after")
    def validate_state(self) -> "Evidence":
        if (self.state == "present") != (self.content is not None):
            raise ValueError("evidence state and content disagree")
        return self


class OutcomeObservation(_ClosedModel):
    requested: tuple[OutcomeRequirement, ...]
    forbidden: tuple[OutcomeRequirement, ...] = ()
    evidence: tuple[Evidence, ...]

    @model_validator(mode="after")
    def validate_references(self) -> "OutcomeObservation":
        if not self.requested:
            raise ValueError("an observation needs a requested outcome")
        evidence_ids = [item.id for item in self.evidence]
        outcome_ids = [item.id for item in (*self.requested, *self.forbidden)]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("evidence IDs must be unique")
        if len(outcome_ids) != len(set(outcome_ids)):
            raise ValueError("outcome IDs must be unique")
        available = set(evidence_ids)
        for outcome in (*self.requested, *self.forbidden):
            if not outcome.id or not outcome.evidence_ids:
                raise ValueError("outcomes need an ID and evidence")
            if not set(outcome.evidence_ids) <= available:
                raise ValueError("outcome cites unknown evidence")
        return self


class RequestedVerdict(_ClosedModel):
    id: str
    grade: Literal["satisfied", "partial", "missing"]
    evidence_ids: tuple[str, ...]


class ForbiddenVerdict(_ClosedModel):
    id: str
    grade: Literal["present", "absent"]
    evidence_ids: tuple[str, ...]


class Contradiction(_ClosedModel):
    description: str = Field(min_length=1)
    evidence_ids: tuple[str, ...]


class OutcomeVerdict(_ClosedModel):
    overall: Literal["pass", "partial", "fail"]
    requested: tuple[RequestedVerdict, ...]
    forbidden: tuple[ForbiddenVerdict, ...]
    contradictions: tuple[Contradiction, ...] = ()
    material_unrelated_evidence_ids: tuple[str, ...]
    unsafe_extra_evidence_ids: tuple[str, ...]
    rationale: str = Field(min_length=1)
    rationale_evidence_ids: tuple[str, ...]
    confidence: Literal["low", "medium", "high"]


@dataclass(frozen=True)
class JudgedOutcome:
    verdict: OutcomeVerdict
    model: str
    prompt_sha256: str


def validate_verdict(
    observation: OutcomeObservation, verdict: OutcomeVerdict
) -> OutcomeVerdict:
    """Validate only mechanical identity, citation, and aggregation rules."""

    if [item.id for item in verdict.requested] != [
        item.id for item in observation.requested
    ]:
        raise ValueError("requested verdict IDs differ from the observation")
    if [item.id for item in verdict.forbidden] != [
        item.id for item in observation.forbidden
    ]:
        raise ValueError("forbidden verdict IDs differ from the observation")

    evidence_ids = {item.id for item in observation.evidence}
    allowed = {
        item.id: set(item.evidence_ids)
        for item in (*observation.requested, *observation.forbidden)
    }
    for item in (*verdict.requested, *verdict.forbidden):
        _validate_citations(item.evidence_ids, allowed[item.id])
    for item in verdict.contradictions:
        _validate_citations(item.evidence_ids, evidence_ids)
    _validate_citations(verdict.rationale_evidence_ids, evidence_ids)

    produced = {
        item.id
        for item in observation.evidence
        if item.kind in {"final", "response", "event"}
        and item.state == "present"
        and bool(item.content and item.content.strip())
    }
    unrelated = set(verdict.material_unrelated_evidence_ids)
    unsafe = set(verdict.unsafe_extra_evidence_ids)
    if len(unrelated) != len(verdict.material_unrelated_evidence_ids) \
            or len(unsafe) != len(verdict.unsafe_extra_evidence_ids):
        raise ValueError("extra-output citations must be unique")
    if not unrelated <= produced or not unsafe <= produced or unrelated & unsafe:
        raise ValueError("extra-output citations are invalid")

    requested_grades = {item.grade for item in verdict.requested}
    if any(item.grade == "present" for item in verdict.forbidden) \
            or verdict.contradictions or unsafe \
            or requested_grades == {"missing"}:
        expected = "fail"
    elif requested_grades == {"satisfied"} and not unrelated:
        expected = "pass"
    else:
        expected = "partial"
    if verdict.overall != expected:
        raise ValueError(f"overall must be {expected}")
    return verdict


def _validate_citations(citations: tuple[str, ...], allowed: set[str]) -> None:
    if not citations or len(citations) != len(set(citations)) \
            or not set(citations) <= allowed:
        raise ValueError("citations must be non-empty, unique, and declared")


class OutcomeJudge:
    """Make one structured local-model call for one closed observation."""

    def __init__(self, model: Any | None = None):
        self._model = model
        self._prompt = _PROMPT_PATH.read_text()
        self._prompt_sha256 = hashlib.sha256(self._prompt.encode()).hexdigest()

    def judge(self, observation: OutcomeObservation) -> OutcomeVerdict:
        return self.judge_with_provenance(observation).verdict

    def judge_with_provenance(
        self, observation: OutcomeObservation
    ) -> JudgedOutcome:
        if self._model is None:
            from assist.model_manager import select_chat_model

            selected = select_chat_model(0, enable_thinking=True)
        else:
            selected = self._model
        model = selected.bind(
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "assist_outcome_verdict",
                    "strict": True,
                    "schema": OutcomeVerdict.model_json_schema(),
                },
            },
            max_tokens=_OUTPUT_TOKENS,
            seed=0,
        )
        payload = json.dumps(observation.model_dump(mode="json"), sort_keys=True)
        response = model.invoke([
            SystemMessage(content=self._prompt),
            HumanMessage(content=(
                "BEGIN UNTRUSTED OUTCOME OBSERVATION\n"
                f"{payload}\n"
                "END UNTRUSTED OUTCOME OBSERVATION"
            )),
        ])
        if not isinstance(response.content, str):
            raise TypeError("judge response content must be text")
        model_name = response.response_metadata.get("model_name")
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("judge response must identify the served model")
        if response.response_metadata.get("finish_reason") != "stop":
            raise ValueError("judge response did not finish cleanly")
        verdict = validate_verdict(
            observation, OutcomeVerdict.model_validate_json(response.content)
        )
        return JudgedOutcome(verdict, model_name, self._prompt_sha256)
