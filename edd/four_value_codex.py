"""Opt-in four-value calibration using the measured Codex CLI transport.

Run from a source checkout with an observation JSON and a private output directory.
The caller authorizes sending that observation to the selected Codex model.
No Assist runtime or default judge imports this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from edd.four_value_judge import (
    CombinedAssessment,
    ValueAssessment,
    validate_assessment,
    validate_combined,
)
from edd.outcome_judge import OutcomeObservation


_ASSETS = Path(__file__).with_name("four_value_prompts")
_VALUES = ("Thorough", "Focused", "Correct")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _prompt(role: str, observation: dict, assessments: list[dict]) -> str:
    instruction = (_ASSETS / f"{role.lower()}.md").read_text()
    data = json.dumps(observation, ensure_ascii=False)
    if role == "aggregator":
        return (
            "Apply the following fourth-judge aggregation instruction to the "
            "observation and three immutable independent assessments. Return only "
            "the JSON schema object. Do not use tools, browse, edit files, or look "
            "anything up.\nBEGIN AGGREGATION INSTRUCTION\n" + instruction
            + "\nEND AGGREGATION INSTRUCTION\nBEGIN UNTRUSTED OBSERVATION\n"
            + data + "\nEND UNTRUSTED OBSERVATION\nBEGIN ASSESSMENTS\n"
            + json.dumps(assessments, ensure_ascii=False) + "\nEND ASSESSMENTS\n"
        )
    return (
        "Apply the following independent research-value assessor instruction "
        "to the observation. Return only the JSON schema object. Do not use "
        "tools, browse, edit files, or look anything up beyond this observation.\n"
        "BEGIN ASSESSOR INSTRUCTION\n" + instruction
        + "\nEND ASSESSOR INSTRUCTION\nBEGIN UNTRUSTED OBSERVATION\n"
        + data + "\nEND UNTRUSTED OBSERVATION\n"
    )


def _observed_model(thread_id: str) -> str | None:
    home = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
    try:
        matches = list((home / "sessions").rglob(f"*{thread_id}.jsonl"))
        if len(matches) != 1:
            return None
        contexts = []
        for line in matches[0].read_text().splitlines():
            item = json.loads(line)
            if item.get("type") == "turn_context":
                contexts.append(item["payload"])
        return contexts[-1].get("model") if contexts else None
    except (OSError, json.JSONDecodeError, KeyError):
        return None


def _call(
    role: str, prompt: str, output: Path, *, codex: str, model: str,
    effort: str, cwd: Path, cap_seconds: int,
) -> dict:
    output.mkdir()
    (output / "prompt.txt").write_text(prompt)
    schema = _ASSETS / ("aggregator-schema.json" if role == "aggregator"
                        else "assessor-schema.json")
    command = [
        "/usr/bin/timeout", "--signal=TERM", "--kill-after=10s",
        f"{cap_seconds}s", codex, "exec", "-m", model,
        "-c", f"model_reasoning_effort={effort}", "--strict-config",
        "-C", str(cwd), "--skip-git-repo-check", "--ignore-user-config",
        "--sandbox", "read-only", "--output-schema", str(schema),
        "--output-last-message", str(output / "answer.json"), "--json", "-",
    ]
    start = time.monotonic()
    try:
        result = subprocess.run(command, input=prompt, text=True,
                                capture_output=True, cwd=cwd,
                                timeout=cap_seconds + 15, check=False)
        stdout, stderr, code = result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or b""
        stderr = exc.stderr or b""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        code = 124
    except OSError as exc:
        stdout, stderr, code = "", str(exc), 127
    (output / "events.jsonl").write_text(stdout)
    (output / "stderr.log").write_text(stderr)
    events = []
    for line in stdout.splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    starts = [item["thread_id"] for item in events
              if item.get("type") == "thread.started"]
    ends = [item for item in events if item.get("type") == "turn.completed"]
    observed = _observed_model(starts[0]) if code == 0 and len(starts) == 1 else None
    unexpected_tool = any(
        item.get("type", "").startswith("item.")
        and item.get("item", {}).get("type") not in (None, "agent_message")
        for item in events
    )
    status = ("transport_error" if code else "model_identity_error"
              if observed != model else "unexpected_tool_use"
              if unexpected_tool else "incomplete_turn" if not ends
              else "pending_validation")
    record = {
        "role": role, "status": status, "exit_code": code,
        "duration_seconds": round(time.monotonic() - start, 3),
        "prompt_sha256": _sha(prompt.encode()),
        "schema_sha256": _sha(schema.read_bytes()),
        "requested_model": model, "observed_model": observed,
        "reasoning_effort": effort, "codex_cwd": str(cwd),
        "thread_id": starts[0] if len(starts) == 1 else None,
        "usage": ends[-1].get("usage") if ends else None,
    }
    if status == "pending_validation" and (output / "answer.json").exists():
        try:
            record["answer"] = json.loads((output / "answer.json").read_text())
        except json.JSONDecodeError as exc:
            record["status"] = "invalid_json"
            record["error"] = str(exc)
    (output / "record.json").write_text(json.dumps(record, indent=2) + "\n")
    for path in output.iterdir():
        path.chmod(0o600)
    return record


def run(
    observation_path: Path, output: Path, *, model: str, effort: str,
    cwd: Path, cap_seconds: int, codex: str,
) -> dict:
    """A failed assessor prevents the fourth call; any failure has no overall verdict."""
    observation_path = observation_path.resolve(strict=True)
    output = output.resolve()
    cwd = cwd.resolve(strict=True)
    source = observation_path.read_bytes()
    observation_data = json.loads(source)
    observation = OutcomeObservation.model_validate(observation_data)
    output.mkdir(mode=0o700, parents=True)
    records = []
    assessments: list[ValueAssessment] = []
    for value in _VALUES:
        record = _call(value.lower(), _prompt(value, observation_data, []),
                       output / value.lower(), codex=codex, model=model,
                       effort=effort, cwd=cwd, cap_seconds=cap_seconds)
        records.append(record)
        if record["status"] == "pending_validation":
            try:
                answer = ValueAssessment.model_validate(record["answer"])
                validate_assessment(observation, value, answer)
                assessments.append(answer)
                record["status"] = "validated"
            except (KeyError, ValueError) as exc:
                record["status"] = "invalid_assessor"
                record["error"] = str(exc)
            (output / value.lower() / "record.json").write_text(
                json.dumps(record, indent=2) + "\n")
        if record["status"] != "validated":
            break
    if len(assessments) == 3:
        triple = (assessments[0], assessments[1], assessments[2])
        record = _call(
            "aggregator", _prompt("aggregator", observation_data,
                                  [a.model_dump(mode="json") for a in triple]),
            output / "aggregator", codex=codex, model=model, effort=effort,
            cwd=cwd, cap_seconds=cap_seconds,
        )
        records.append(record)
        if record["status"] == "pending_validation":
            try:
                answer = CombinedAssessment.model_validate(record["answer"])
                validate_combined(observation, triple, answer)
                record["status"] = "validated"
            except (KeyError, ValueError) as exc:
                record["status"] = "invalid_aggregation"
                record["error"] = str(exc)
            (output / "aggregator" / "record.json").write_text(
                json.dumps(record, indent=2) + "\n")
    summary = {
        "observation_sha256": _sha(source), "observation_path": str(observation_path),
        "roles": [{key: record.get(key) for key in (
            "role", "status", "duration_seconds", "prompt_sha256",
            "schema_sha256", "requested_model", "observed_model", "usage",
        )} | {"grade": record["answer"]["grade"]
              if record["status"] == "validated" and record["role"] != "aggregator"
              else None}
                  for record in records],
        "status": "validated" if len(records) == 4 and all(
            record["status"] == "validated" for record in records
        ) else "nonverdict",
    }
    if summary["status"] == "validated":
        summary["overall"] = records[3]["answer"]["overall"]
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "summary.json").chmod(0o600)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("observation", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--effort", required=True)
    parser.add_argument("--codex-cwd", type=Path, required=True)
    parser.add_argument("--cap-seconds", type=int, default=180)
    args = parser.parse_args()
    if not 1 <= args.cap_seconds <= 180:
        parser.error("per-call cap must be 1..180 seconds")
    codex = shutil.which("codex")
    if not codex:
        parser.error("codex executable not found")
    print(json.dumps(run(args.observation, args.output, model=args.model,
                         effort=args.effort, cwd=args.codex_cwd,
                         cap_seconds=args.cap_seconds, codex=codex)))


if __name__ == "__main__":
    main()
