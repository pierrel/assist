"""Calibrate the natural-outcome judge against labelled observations."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from edd.outcome_judge import OutcomeJudge, OutcomeObservation


_DEVELOPMENT = (
    Path(__file__).with_name("fixtures") / "outcome-judge" / "development.json"
)


def _fixture_path() -> Path:
    configured = os.environ.get("ASSIST_OUTCOME_JUDGE_FIXTURE")
    return Path(configured) if configured else _DEVELOPMENT


def _cases() -> list[tuple[str, str, OutcomeObservation]]:
    payload = json.loads(_fixture_path().read_text())
    return [
        (
            item["case_id"],
            item["expected"],
            OutcomeObservation.model_validate(item["observation"]),
        )
        for item in payload["cases"]
    ]


_CASES = _cases()


@pytest.mark.parametrize(
    ("case_id", "expected", "observation"),
    _CASES,
    ids=[case_id for case_id, _, _ in _CASES],
)
def test_outcome_judge(
    case_id: str, expected: str, observation: OutcomeObservation
) -> None:
    verdict = OutcomeJudge().judge(observation)
    assert verdict.overall == expected, (case_id, verdict)
