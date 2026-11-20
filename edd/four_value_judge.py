"""Pure value-judge schemas and validation for opt-in calibration."""

from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

from edd.outcome_judge import OutcomeObservation


Value = Literal["Thorough", "Focused", "Correct"]
Grade = Literal["Meets", "Partly meets", "Does not meet", "Uncertain"]
_VALUES: tuple[Value, ...] = ("Thorough", "Focused", "Correct")


class _Closed(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ValueAssessment(_Closed):
    value: Value
    grade: Grade
    evidence_ids: tuple[str, ...] = Field(min_length=1)
    reason: str = Field(min_length=1)
    uncertainty: str


class ValueEcho(_Closed):
    value: Value
    grade: Grade


class CombinedAssessment(_Closed):
    overall: Grade
    value_grades: tuple[ValueEcho, ...] = Field(min_length=3, max_length=3)
    evidence_ids: tuple[str, ...] = Field(min_length=1)
    disagreements: tuple[str, ...]
    reason: str = Field(min_length=1)


def combine_grades(grades: tuple[Grade, Grade, Grade]) -> Grade:
    """Apply the frozen opt-in policy; no human overall labels are implied."""

    for grade in ("Does not meet", "Uncertain", "Partly meets"):
        if grade in grades:
            return grade
    return "Meets"


def validate_assessment(
    observation: OutcomeObservation, value: Value, answer: ValueAssessment
) -> ValueAssessment:
    """Reject empty, repeated, or undeclared citations, not semantic judgments."""

    if answer.value != value:
        raise ValueError("assessor changed value")
    ids = answer.evidence_ids
    known = {item.id for item in observation.evidence}
    if len(ids) != len(set(ids)) or not set(ids) <= known or not answer.reason.strip():
        raise ValueError("assessor citations or explanation invalid")
    if answer.grade == "Uncertain" and not answer.uncertainty.strip():
        raise ValueError("uncertain assessment must state what is unknown")
    return answer


def validate_combined(
    observation: OutcomeObservation,
    assessments: tuple[ValueAssessment, ValueAssessment, ValueAssessment],
    answer: CombinedAssessment,
) -> CombinedAssessment:
    """The fourth call may explain disagreement but cannot rewrite the grades."""

    expected = tuple(ValueEcho(value=a.value, grade=a.grade) for a in assessments)
    if answer.value_grades != expected:
        raise ValueError("combined answer changed assessor grades")
    if answer.overall != combine_grades(tuple(a.grade for a in assessments)):
        raise ValueError("combined answer changed frozen policy")
    ids = answer.evidence_ids
    cited = {item for a in assessments for item in a.evidence_ids}
    known = {item.id for item in observation.evidence}
    if len(ids) != len(set(ids)) or not set(ids) <= cited or not set(ids) <= known \
            or not answer.reason.strip():
        raise ValueError("combined citations or explanation invalid")
    return answer
