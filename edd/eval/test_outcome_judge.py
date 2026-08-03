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
    "case_index",
    range(len(_CASES)),
    ids=[case_id for case_id, _, _ in _CASES],
)
def test_outcome_judge(case_index: int) -> None:
    case_id, expected, observation = _CASES[case_index]
    result = OutcomeJudge().judge_with_provenance(observation)
    verdict_record = result.verdict.model_dump(mode="json")
    verdict_record.pop("rationale")
    for contradiction in verdict_record["contradictions"]:
        contradiction.pop("description")
    print("\n" + json.dumps({
        "case_id": case_id,
        "model": result.model,
        "prompt_sha256": result.prompt_sha256,
        "verdict": verdict_record,
    }, sort_keys=True), flush=True)
    assert result.verdict.overall == expected, (
        case_id, expected, result.verdict.overall
    )
