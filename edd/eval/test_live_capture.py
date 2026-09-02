"""Natural judged-live-capture acceptance rows against the local model.

The interaction is already completed evidence.  These rows test only the new
reason-to-criteria interpretation plus the existing outcome judge; they never
replay a conversation through an agent graph.
"""
from __future__ import annotations

import json
from pathlib import Path
from time import monotonic, sleep

import pytest

from assist.visible_conversation import VisibleRecord
from edd.live_capture import CaptureStore, CaptureWorker


_CASES = (
    (
        "capital-correct",
        "I wanted the answer to say that Paris is the capital of France, with no different city.",
        [
            {"id": "r0001", "order": 1, "role": "user", "text": "What is the capital of France?", "source_kind": "user", "capture_eligible": True},
            {"id": "r0002", "order": 2, "role": "assistant", "text": "Paris is the capital of France.", "source_kind": "assistant", "capture_eligible": True},
        ],
        "pass",
    ),
    (
        "capital-wrong",
        "I wanted the answer to say that Paris is the capital of France, with no different city.",
        [
            {"id": "r0001", "order": 1, "role": "user", "text": "What is the capital of France?", "source_kind": "user", "capture_eligible": True},
            {"id": "r0002", "order": 2, "role": "assistant", "text": "Lyon is the capital of France.", "source_kind": "assistant", "capture_eligible": True},
        ],
        "fail",
    ),
)


@pytest.mark.parametrize(
    "case_id,reason,records,expected", _CASES,
    ids=[case[0] for case in _CASES],
)
def test_judged_live_capture(
    case_id: str, reason: str, records: list[dict], expected: str, tmp_path: Path,
) -> None:
    threads = tmp_path / "threads"
    threads.mkdir()
    store = CaptureStore(tmp_path / "captures", threads_root=threads)
    capture = store.create(
        thread_id=case_id, reason=reason, scope="last_3", turn_range=(1, 1),
        records=tuple(VisibleRecord(**record) for record in records),
        calibration={"verdict": expected, "reason": "Synthetic calibration evidence."},
    )
    capture_id = capture["request"]["capture_id"]
    worker = CaptureWorker(store)
    worker.start()
    worker.submit(case_id, capture_id)
    deadline = monotonic() + 240
    while monotonic() < deadline:
        current = store.get_for_thread(case_id, capture_id)
        if current["result"]["status"] not in {"queued", "interpreting", "judging"}:
            break
        sleep(0.1)
    worker.stop()
    judged = store.get_for_thread(case_id, capture_id)["result"]
    print(json.dumps({
        "case_id": case_id, "interpreter_model": judged.get("interpreter", {}).get("model"),
        "judge_model": judged.get("judge", {}).get("model"), "verdict": judged.get("verdict"),
    }, sort_keys=True), flush=True)
    assert judged["status"] == expected
