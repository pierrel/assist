"""Read-only diagnostic of existing bounded Assist research captures.

This measures delivery and graph shape. It does not grade factual quality or
claim a timeout's eventual result.
"""

import json
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


TASK_STARTS = {"task", "start_async_task"}


def _query_identity(event: dict) -> str:
    inputs = event.get("inputs")
    query = (inputs.get("query") or inputs.get("q")) if isinstance(inputs, dict) else None
    if query is None:
        query = event.get("input_str")
    return " ".join(str(query or "").casefold().split())


def _url_identity(event: dict) -> str:
    inputs = event.get("inputs")
    raw = inputs.get("url") if isinstance(inputs, dict) else event.get("input_str")
    try:
        parsed = urlsplit(str(raw or ""))
    except ValueError:
        return str(raw or "")
    if not parsed.scheme or not parsed.netloc:
        return str(raw or "")
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(),
                       parsed.path or "/", parsed.query, ""))


def _matched_time(starts: dict, ends: dict, names: set[str] | None = None) -> float:
    return round(sum(max(0.0, float(end["at"]) - float(starts[run_id]["at"]))
                     for run_id, end in ends.items() if run_id in starts
                     and (names is None or starts[run_id].get("name") in names)
                     and isinstance(end.get("at"), int | float)
                     and isinstance(starts[run_id].get("at"), int | float)), 3)


def _maximum_open_sync_tasks(trace: list[dict]) -> int:
    active = set()
    maximum = 0
    for event in sorted(trace, key=lambda item: item.get("at", 0)):
        run_id = event.get("run_id")
        if event.get("name") != "task" or not run_id:
            continue
        if event.get("event") == "tool_start":
            active.add(run_id)
            maximum = max(maximum, len(active))
        elif event.get("event") in {"tool_end", "tool_error"}:
            active.discard(run_id)
    return maximum


def _handoff_contains(batches: list, marker: str) -> bool:
    for batch in batches:
        for message in batch:
            content = str(message.get("data", {}).get("content", ""))
            if marker not in content:
                continue
            if message.get("type") == "tool":
                return True
            if (message.get("type") == "human" and
                    content.startswith("[Background task finished] Task ID:")):
                return True
    return False


def _durable_graph(runs: list[dict]) -> dict:
    if not runs or any("parent_run_id" not in run for run in runs):
        return {"durable_ancestry_available": False,
                "child_runs": None, "missing_durable_parent_ids": None,
                "cyclic_durable_run_ids": None,
                "max_durable_child_depth": None}
    by_id = {run["id"]: run for run in runs if run.get("id")}
    children = [run for run in runs if run.get("parent_run_id")]
    missing = sorted({str(run["parent_run_id"]) for run in children
                      if run["parent_run_id"] not in by_id})
    depths = []
    cycles = []
    for run in runs:
        seen = {run.get("id")}
        depth = 0
        parent = run.get("parent_run_id")
        while parent in by_id:
            if parent in seen:
                cycles.append(str(run.get("id")))
                break
            seen.add(parent)
            depth += 1
            parent = by_id[parent].get("parent_run_id")
        depths.append(depth)
    return {"durable_ancestry_available": bool(runs),
            "child_runs": len(children), "missing_durable_parent_ids": missing,
            "cyclic_durable_run_ids": sorted(cycles),
            "max_durable_child_depth": max(depths, default=0)}


def _run_pending_seconds(lifecycle: list[dict]) -> float:
    """Sum observed durable pending/paused-to-running spans across Run slices."""
    waiting = {}
    elapsed = 0.0
    for event in lifecycle:
        if event.get("event") != "run_status" or not isinstance(event.get("at"), int | float):
            continue
        key = (event.get("run_thread_id"), event.get("run_id"))
        status = event.get("status")
        if status in {"pending", "paused"}:
            waiting[key] = event["at"]
        elif status == "running" and key in waiting:
            elapsed += max(0.0, event["at"] - waiting.pop(key))
        elif status in {"success", "error", "timeout", "cancelled"}:
            waiting.pop(key, None)
    return round(elapsed, 3)


def _retrieved_content(event: dict) -> str | None:
    """Return only a successful source result, not a tool error or empty-search steer."""
    if (event.get("event") != "tool_end" or
            event.get("name") not in {"search_internet", "read_url"}):
        return None
    output = event.get("output")
    if isinstance(output, dict):
        data = output.get("data") or {}
        if data.get("status") == "error":
            return None
        content = str(data.get("content") or "")
    else:
        content = str(output or "")
    if content.startswith(("Error fetching URL:", "Web search is unavailable",
                           "This search returned no results.")):
        return None
    return content or None


def summarize_records(supervisor: dict, checkpoint: dict, trace: list[dict],
                      *, finding_markers: tuple[str, ...] = (),
                      artifact_ready: bool = False,
                      capability_answer_ready: bool = False,
                      lifecycle: tuple[dict, ...] | list[dict] = (),
                      parent_thread_id: str | None = None) -> dict:
    """Measure observed delivery and graph events, without grading source truth."""
    parent_id = parent_thread_id or supervisor["checkpoint_threads"][0]
    roots = [thread for thread in checkpoint["threads"]
             if thread["thread_id"] == parent_id and thread["checkpoint_ns"] == ""]
    if len(roots) != 1:
        raise ValueError("expected one parent root checkpoint")
    messages = roots[0]["messages"]
    final = messages[-1] if messages else {}
    final_data = final.get("data", {})
    final_reply = (final.get("type") == "ai"
                   and bool(str(final_data.get("content") or "").strip())
                   and not final_data.get("tool_calls"))
    all_starts = {event.get("run_id"): event for event in trace
                  if event.get("event") == "tool_start" and event.get("run_id")}
    all_ends = {event.get("run_id"): event for event in trace
                if event.get("event") in {"tool_end", "tool_error"}
                and event.get("run_id")}
    starts = [event for event in all_starts.values()
              if event.get("name") in TASK_STARTS]
    ends = [event for event in all_ends.values()
            if event.get("name") in TASK_STARTS]
    sync_starts = {event.get("run_id") for event in starts
                   if event.get("name") == "task"}
    sync_returns = {event.get("run_id") for event in ends
                    if event.get("name") == "task"}
    pending_sync = sync_starts - sync_returns
    runs = [run for group in supervisor.get("run_reconciliation", {}).get("after", {}).values()
            for run in group]
    durable_graph = _durable_graph(runs)
    terminal_statuses = {"success", "error", "timeout", "interrupted", "cancelled"}
    durable_closed = all(run.get("status") in terminal_statuses for run in runs)
    all_success = all(run.get("status") == "success" for run in runs)
    finished_events = [event for event in lifecycle
                       if event.get("event") == "worker_finished"]
    worker_delivery = (bool(finished_events[-1].get("delivered"))
                       if finished_events else None)
    capability = supervisor.get("evaluation_profile", "web-main") != "web-main"
    mismatched_parents = sorted(str(run_id) for run_id, end in all_ends.items()
                                if run_id in all_starts
                                and end.get("parent_run_id") != all_starts[run_id].get(
                                    "parent_run_id"))
    delivered = (not supervisor["timed_out"] and supervisor["worker_exit_code"] == 0
                 and final_reply and (capability_answer_ready if capability else artifact_ready)
                 and durable_closed
                 and not durable_graph["missing_durable_parent_ids"]
                 and not durable_graph["cyclic_durable_run_ids"]
                 and (worker_delivery if worker_delivery is not None else
                      all_success and not pending_sync))
    counts = Counter(event.get("name") for event in trace
                     if event.get("event") == "tool_start")
    search_inputs = [_query_identity(event)
                     for event in all_starts.values()
                     if event.get("name") == "search_internet"]
    read_inputs = [_url_identity(event)
                   for event in all_starts.values() if event.get("name") == "read_url"]
    model_starts = {event.get("run_id"): event for event in trace
                    if event.get("event") == "model_start" and event.get("run_id")}
    model_ends = {event.get("run_id"): event for event in trace
                  if event.get("event") == "model_end" and event.get("run_id")}
    token_usage = Counter()
    for event in model_ends.values():
        for batch in event.get("generations") or []:
            for generation in batch:
                usage = generation.get("data", {}).get("usage_metadata") or {}
                token_usage["input"] += usage.get("input_tokens") or 0
                token_usage["output"] += usage.get("output_tokens") or 0
    product_timeouts = sum(run.get("status") == "timeout" for run in runs)
    product_errors = sum(run.get("status") == "error" for run in runs)
    cancellations = sum(run.get("status") == "cancelled" for run in runs)
    visibility = {}
    first_source_at = None
    for marker in finding_markers:
        if not marker:
            raise ValueError("finding markers must be nonempty")
        source_events = [event for event in all_ends.values()
                         if marker in (_retrieved_content(event) or "")]
        returned = [event for event in ends if event.get("name") == "task"
                    and marker in str(event.get("output", ""))]
        source_at = min((event["at"] for event in source_events
                         if isinstance(event.get("at"), int | float)), default=None)
        if source_at is not None:
            first_source_at = (source_at if first_source_at is None
                               else min(first_source_at, source_at))
        evidence_at = min((event["at"] for event in (returned or source_events)
                           if "at" in event), default=None)
        parent_visible = [event for event in trace
                          if event.get("event") == "model_start"
                          and event.get("thread_id") == parent_id
                          and evidence_at is not None
                          and event.get("at", 0) > evidence_at
                          and _handoff_contains(event.get("batches") or [], marker)]
        visibility[marker] = {"retrieval_result": bool(source_events),
                              "task_return": bool(returned),
                              "parent_model_input": bool(parent_visible)}
    started_at = supervisor.get("started_at")
    return {
        "classification": (("capability_answer_within_window" if capability
                            else "delivered_within_window") if delivered else
                           "external_deadline_non_delivery" if supervisor["timed_out"] else
                           "product_run_timeout" if product_timeouts else
                           "product_run_error" if product_errors else
                           "cancelled_or_incomplete" if cancellations else
                           "worker_failed_before_model" if supervisor["worker_exit_code"] != 0
                           and not model_starts else
                           "incomplete_or_error"),
        "quality": "not_scored_by_this_diagnostic",
        "elapsed_seconds": supervisor["elapsed_seconds"],
        "external_timeout": supervisor["timed_out"],
        "worker_exit_code": supervisor["worker_exit_code"],
        "parent_final_reply": final_reply,
        "requested_artifact_observed": artifact_ready,
        "capability_answer_observed": capability_answer_ready,
        "durable_runs": len(runs),
        "durable_runs_all_terminal": durable_closed if runs else None,
        "durable_runs_all_success": all_success if runs else None,
        "worker_reported_delivery": worker_delivery,
        "durable_run_active_ms_sum_reported": sum(run.get("active_ms") or 0
                                                   for run in runs),
        "product_timeout_runs": product_timeouts,
        "product_error_runs": product_errors,
        "cancelled_runs": cancellations,
        **durable_graph,
        "sync_task_starts": len(sync_starts),
        "sync_task_returns": len(sync_returns),
        "async_task_launches": sum(event.get("name") == "start_async_task"
                                   for event in starts),
        "async_task_launch_tool_returns": sum(event.get("name") == "start_async_task"
                                               for event in ends),
        "pending_sync_tasks": len(pending_sync),
        "max_open_sync_tasks": _maximum_open_sync_tasks(trace),
        "mismatched_tool_parent_ids": mismatched_parents,
        "unreturned_tool_calls": sorted(str(run_id) for run_id in all_starts
                                        if run_id not in all_ends),
        "max_checkpoint_task_depth": max(
            (thread["checkpoint_ns"].count("tools:") for thread in checkpoint["threads"]),
            default=0),
        "model_starts": sum(event.get("event") == "model_start" for event in trace),
        "model_ends": sum(event.get("event") == "model_end" for event in trace),
        "model_calls_without_end": len(model_starts.keys() - model_ends.keys()),
        "matched_model_seconds_sum": _matched_time(model_starts, model_ends),
        "matched_retrieval_seconds_sum": _matched_time(
            all_starts, all_ends, {"search_internet", "read_url"}),
        "durable_run_pending_seconds_sum": _run_pending_seconds(lifecycle),
        "model_slot_wait_seconds_sum": round(sum(
            float(event["waiting_ms"]) / 1000 for event in lifecycle
            if event.get("event") == "model_slot_acquired"), 3),
        "model_input_tokens_reported": token_usage["input"],
        "model_output_tokens_reported": token_usage["output"],
        "search_starts": counts["search_internet"],
        "distinct_search_inputs": len(set(search_inputs)),
        "repeated_search_inputs": len(search_inputs) - len(set(search_inputs)),
        "read_url_starts": counts["read_url"],
        "repeated_read_url_inputs": len(read_inputs) - len(set(read_inputs)),
        "finding_visibility": visibility,
        "first_source_finding_seconds": (round(first_source_at - started_at, 3)
                                         if first_source_at is not None and
                                         isinstance(started_at, int | float) else None),
        "first_sync_task_return_seconds": (
            round(min(event["at"] for event in ends if event.get("name") == "task")
                  - started_at, 3)
            if isinstance(started_at, int | float) and
            any(event.get("name") == "task" and "at" in event for event in ends)
            else None),
    }


def summarize(folder: Path, *, finding_markers: tuple[str, ...] = ()) -> dict:
    supervisor = json.loads((folder / "supervisor.json").read_text())
    checkpoint_file = folder / "checkpoint-latest.json"
    if not checkpoint_file.is_file():
        stderr_file = folder / "worker.stderr.log"
        stderr = (stderr_file.read_text(errors="replace")[-4096:]
                  if stderr_file.is_file() else "")
        admission_denied = ("production is busy" in stderr or
                            "resource is busy: llm" in stderr)
        trace_file = folder / "trace.jsonl"
        trace = ([json.loads(line) for line in trace_file.read_text().splitlines()
                  if line.strip()] if trace_file.is_file() else [])
        model_starts = sum(event.get("event") == "model_start" for event in trace)
        return {"capture": folder.name,
                "classification": ("external_deadline_non_delivery"
                                   if supervisor["timed_out"] else
                                   "admission_denied" if admission_denied else
                                   "checkpoint_missing_after_model"
                                   if model_starts else
                                   "setup_failed_before_checkpoint"
                                   if supervisor["worker_exit_code"] != 0 else
                                   "checkpoint_missing_after_worker"),
                "quality": "not_scored_by_this_diagnostic",
                "elapsed_seconds": supervisor["elapsed_seconds"],
                "external_timeout": supervisor["timed_out"],
                "worker_exit_code": supervisor["worker_exit_code"],
                "checkpoint_available": False,
                "model_starts": model_starts,
                "parent_final_reply": False,
                "requested_artifact_observed": False}
    if not finding_markers and supervisor.get("case_id"):
        manifest = json.loads(Path(__file__).with_name("research_live_cases.json").read_text())
        case = next((item for item in manifest["cases"]
                     if item["id"] == supervisor["case_id"]), None)
        if case is not None:
            finding_markers = tuple(case.get("diagnostic_source_markers", ()))
    checkpoint = json.loads(checkpoint_file.read_text())
    trace = [json.loads(line) for line in (folder / "trace.jsonl").read_text().splitlines()
             if line.strip()]
    lifecycle_file = folder / "lifecycle.jsonl"
    lifecycle = ([json.loads(line) for line in lifecycle_file.read_text().splitlines()
                  if line.strip()] if lifecycle_file.is_file() else [])
    owning_event = next((event for event in lifecycle if event.get("event") in {
        "submitted", "leaf_started", "research_manager_capability_started",
        "delegate_capability_started"}), None)
    parent_thread_id = ((owning_event.get("parent_thread_id") or
                         owning_event.get("thread_id")) if owning_event else None)
    artifact_ready = any(event.get("event") in {"note_update_available", "report_available"}
                         for event in lifecycle)
    capability_answer_ready = any(event.get("event") in {
        "leaf_answer_available", "research_manager_answer_available",
        "delegate_answer_available"} for event in lifecycle)
    provider_file = folder / "provider-calls.jsonl"
    provider = ([json.loads(line) for line in provider_file.read_text().splitlines()
                 if line.strip()] if provider_file.is_file() else [])
    requests = [row for row in provider if row.get("event") == "provider_request"]
    finished = [row for row in provider if row.get("event") == "provider_finished"]
    submitted = next((row for row in lifecycle if row.get("event") == "submitted"), None)
    first_model = next((row for row in trace if row.get("event") == "model_start"), None)
    return {"capture": folder.name,
            "checkpoint_available": True,
            **summarize_records(supervisor, checkpoint, trace,
                                finding_markers=finding_markers,
                                artifact_ready=artifact_ready,
                                capability_answer_ready=capability_answer_ready,
                                lifecycle=lifecycle,
                                parent_thread_id=parent_thread_id),
            "provider_requests": len(requests),
            "provider_finished": sum(bool(row.get("completed")) for row in finished),
            "provider_elapsed_seconds_sum": round(sum(
                float(row.get("elapsed_ms") or 0) / 1000 for row in finished), 3),
            "provider_message_json_bytes_sum": sum(len(json.dumps(row.get("messages"),
                                                           ensure_ascii=False).encode("utf-8"))
                                                   for row in requests),
            "submit_to_first_model_seconds": (
                round(first_model["at"] - submitted["at"], 3)
                if submitted and first_model else None)}


if __name__ == "__main__":
    print(json.dumps([summarize(Path(name)) for name in sys.argv[1:]], indent=2))
