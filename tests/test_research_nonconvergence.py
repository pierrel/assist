"""CPU controls for delivery and ancestry diagnostics; no model is called."""

import json

from edd.eval.research_nonconvergence import summarize, summarize_records
from pathlib import Path

import pytest

from edd.eval.research_live_supervisor import _reconcile_private_runs, preflight
from edd.eval.research_live_trace import _Trace, install_model_and_tool_trace
from edd.eval.research_live_outcome import _completed_child_id, _completed_parent_run
from assist.run_service import RunService


def _records(*, timeout=False, final=True, artifact=True, nested=False, trace=()):
    supervisor = {"checkpoint_threads": ["parent"], "timed_out": timeout,
                  "worker_exit_code": -15 if timeout else 0,
                  "elapsed_seconds": 90.0,
                  "run_reconciliation": {"after": {}}}
    messages = [{"type": "human", "data": {"content": "Update my notes."}}]
    if final:
        messages.append({"type": "ai", "data": {"content": "Updated the notes.",
                                                "tool_calls": []}})
    checkpoint = {"threads": [{"thread_id": "parent", "checkpoint_ns": "",
                               "messages": messages}]}
    if nested:
        checkpoint["threads"].append({"thread_id": "parent",
                                      "checkpoint_ns": "tools:x|tools:y|",
                                      "messages": []})
    return summarize_records(supervisor, checkpoint, list(trace),
                             finding_markers=("DECISIVE-42",), artifact_ready=artifact)


def _tool(event, name, run_id, at, *, parent="model", output="", inputs=None):
    return {"event": event, "name": name, "run_id": run_id,
            "parent_run_id": parent, "thread_id": "parent", "at": at,
            "output": output, "inputs": inputs}


def test_nested_external_cutoff_is_non_delivery_even_with_a_draft():
    trace = [_tool("tool_start", "task", "lead", 1),
             _tool("tool_start", "task", "worker", 2, parent="lead"),
             _tool("tool_end", "task", "worker", 3, parent="lead",
                   output="DECISIVE-42")]
    result = _records(timeout=True, nested=True, trace=trace)
    assert result["classification"] == "external_deadline_non_delivery"
    assert result["max_checkpoint_task_depth"] == 2
    assert result["sync_task_starts"] == 2
    assert result["sync_task_returns"] == 1
    assert result["pending_sync_tasks"] == 1
    assert result["max_open_sync_tasks"] == 2


def test_completed_complex_path_requires_saved_artifact_and_terminal_reply():
    trace = [_tool("tool_start", "task", "lead", 1),
             _tool("tool_end", "task", "lead", 2, output="DECISIVE-42"),
             {"event": "model_start", "thread_id": "parent", "at": 3,
              "batches": [[{"type": "tool", "data": {"content": "DECISIVE-42"}}]]}]
    result = _records(trace=trace)
    assert result["classification"] == "delivered_within_window"
    assert result["finding_visibility"]["DECISIVE-42"] == {
        "retrieval_result": False, "task_return": True, "parent_model_input": True}
    assert _records(trace=trace, artifact=False)["classification"] == "incomplete_or_error"
    assert _records(trace=trace, final=False)["classification"] == "incomplete_or_error"


def test_direct_capability_answer_is_distinct_from_web_delivery():
    supervisor = {"checkpoint_threads": ["leaf"], "evaluation_profile": "research-leaf",
                  "timed_out": False, "worker_exit_code": 0,
                  "elapsed_seconds": 2.0, "run_reconciliation": {"after": {}}}
    checkpoint = {"threads": [{"thread_id": "leaf", "checkpoint_ns": "",
                               "messages": [{"type": "ai", "data": {
                                   "content": "Sourced result", "tool_calls": []}}]}]}
    result = summarize_records(supervisor, checkpoint, [],
                               capability_answer_ready=True)
    assert result["classification"] == "capability_answer_within_window"
    assert not result["requested_artifact_observed"]
    assert summarize_records(supervisor, checkpoint, [
        _tool("tool_start", "task", "pending", 1)],
        capability_answer_ready=True)["classification"] == "incomplete_or_error"


def test_async_completion_rider_counts_only_after_returned_finding():
    trace = [_tool("tool_start", "search_internet", "s", 1,
                   inputs={"query": "source"}),
             _tool("tool_end", "search_internet", "s", 2,
                   output="DECISIVE-42"),
             {"event": "model_start", "thread_id": "parent", "at": 3,
              "batches": [[{"type": "human", "data": {"content":
                  "[Background task finished] Task ID: sub-1\n"
                  "Agent: research-agent\nResult: DECISIVE-42"}}]]}]
    result = _records(trace=trace)
    assert result["finding_visibility"]["DECISIVE-42"]["parent_model_input"]


def test_wrong_parent_and_lost_returned_finding_are_reported():
    trace = [_tool("tool_start", "task", "lead", 1),
             _tool("tool_end", "task", "lead", 2, parent="wrong-model",
                   output="DECISIVE-42"),
             {"event": "model_start", "thread_id": "parent", "at": 3,
              "batches": [[{"type": "human", "data": {
                  "content": "User note already mentioned DECISIVE-42."}}]]}]
    result = _records(trace=trace)
    assert result["mismatched_tool_parent_ids"] == ["lead"]
    assert result["finding_visibility"]["DECISIVE-42"]["task_return"]
    assert not result["finding_visibility"]["DECISIVE-42"]["parent_model_input"]


@pytest.mark.parametrize("earlier_status", ["interrupted", "cancelled", "error"])
def test_delivered_worker_receipt_survives_earlier_terminal_slices(earlier_status):
    supervisor = {"checkpoint_threads": ["parent"], "timed_out": False,
                  "worker_exit_code": 0, "elapsed_seconds": 9,
                  "run_reconciliation": {"after": {"parent": [
                      {"id": "old", "status": earlier_status,
                       "parent_run_id": None},
                      {"id": "final", "status": "success",
                       "parent_run_id": None}]}}}
    checkpoint = {"threads": [{"thread_id": "parent", "checkpoint_ns": "",
                               "messages": [{"type": "ai", "data": {
                                   "content": "Final", "tool_calls": []}}]}]}
    result = summarize_records(supervisor, checkpoint, [], artifact_ready=True,
                               lifecycle=[{"event": "worker_finished",
                                           "delivered": True}])
    assert result["classification"] == "delivered_within_window"
    assert result["durable_runs_all_terminal"]
    assert not result["durable_runs_all_success"]


def test_echoed_marker_in_failed_retrieval_is_not_a_source_finding():
    trace = [_tool("tool_start", "read_url", "u1", 1),
             _tool("tool_end", "read_url", "u1", 2,
                   output={"type": "tool", "data": {
                       "status": "error", "content": "DECISIVE-42 failed"}}),
             _tool("tool_start", "read_url", "u2", 3),
             _tool("tool_end", "read_url", "u2", 4,
                   output={"type": "tool", "data": {
                       "status": "success",
                       "content": "Error fetching URL: DECISIVE-42 unavailable"}})]
    result = _records(trace=trace)
    assert not result["finding_visibility"]["DECISIVE-42"]["retrieval_result"]
    assert result["first_source_finding_seconds"] is None


def test_repeat_query_diagnostic_counts_same_request_without_forcing_route():
    query = {"query": "Aurora specifications"}
    trace = [_tool("tool_start", "search_internet", "s1", 1, inputs=query),
             _tool("tool_end", "search_internet", "s1", 2,
                   output="DECISIVE-42"),
             _tool("tool_start", "search_internet", "s2", 3,
                   inputs={"query": " aurora   SPECIFICATIONS "}),
             _tool("tool_end", "search_internet", "s2", 4,
                   output="DECISIVE-42"),
             _tool("tool_start", "read_url", "u1", 5,
                   inputs={"url": "https://EXAMPLE.test/path#section"}),
             _tool("tool_end", "read_url", "u1", 6),
             _tool("tool_start", "read_url", "u2", 7,
                   inputs={"url": "https://example.test/path"}),
             _tool("tool_end", "read_url", "u2", 8)]
    result = _records(trace=trace)
    assert result["search_starts"] == 2
    assert result["distinct_search_inputs"] == 1
    assert result["repeated_search_inputs"] == 1
    assert result["repeated_read_url_inputs"] == 1
    assert result["matched_retrieval_seconds_sum"] == 4.0
    assert result["finding_visibility"]["DECISIVE-42"]["retrieval_result"]


def test_observer_reports_model_span_and_usage_without_wall_time_inference():
    trace = [{"event": "model_start", "thread_id": "parent", "run_id": "m1",
              "at": 1.0, "batches": []},
             {"event": "model_end", "thread_id": "parent", "run_id": "m1",
              "at": 6.0, "generations": [[{"data": {"usage_metadata": {
                  "input_tokens": 40, "output_tokens": 12}}}]]}]
    result = _records(trace=trace)
    assert result["matched_model_seconds_sum"] == 5.0
    assert result["model_input_tokens_reported"] == 40
    assert result["model_output_tokens_reported"] == 12


def test_observer_keeps_durable_and_model_slot_wait_separate():
    supervisor = {"checkpoint_threads": ["parent"], "timed_out": False,
                  "worker_exit_code": 0, "elapsed_seconds": 10,
                  "run_reconciliation": {"after": {}}}
    checkpoint = {"threads": [{"thread_id": "parent", "checkpoint_ns": "",
                               "messages": [{"type": "ai", "data": {
                                   "content": "Done", "tool_calls": []}}]}]}
    lifecycle = [
        {"event": "run_status", "run_thread_id": "parent", "run_id": "a",
         "status": "pending", "at": 1.0},
        {"event": "run_status", "run_thread_id": "parent", "run_id": "a",
         "status": "running", "at": 4.0},
        {"event": "model_slot_acquired", "waiting_ms": 750, "at": 4.75},
    ]
    result = summarize_records(supervisor, checkpoint, [], lifecycle=lifecycle)
    assert result["durable_run_pending_seconds_sum"] == 3.0
    assert result["model_slot_wait_seconds_sum"] == 0.75


def test_actual_model_slot_callback_records_acquisition_without_model(tmp_path):
    from assist.thread import Thread
    from assist.thread_queue import THREAD_QUEUE
    from langchain_openai import ChatOpenAI

    original = (Thread.__init__, Thread._run, ChatOpenAI._stream,
                ChatOpenAI._generate)
    try:
        install_model_and_tool_trace(tmp_path)
        with THREAD_QUEUE.acquire("private-cpu-smoke"):
            pass
    finally:
        Thread.__init__, Thread._run, ChatOpenAI._stream, ChatOpenAI._generate = original
        if "acquire" in vars(THREAD_QUEUE):
            del THREAD_QUEUE.acquire
    events = [json.loads(line) for line in
              (tmp_path / "lifecycle.jsonl").read_text().splitlines()]
    assert [event["event"] for event in events] == [
        "model_slot_acquired", "model_slot_released"]
    assert events[0]["waiting_ms"] >= 0


def test_actual_callback_keeps_nested_task_start_and_return(tmp_path):
    from langchain_core.messages import HumanMessage, ToolMessage

    callback = _Trace(tmp_path)
    callback.on_chat_model_start({}, [[HumanMessage(content="Investigate")]],
                                 run_id="model-1", parent_run_id=None)
    callback.on_tool_start({"name": "task"},
                           '{"subagent_type":"research-agent","description":"Check"}',
                           run_id="task-1", parent_run_id="model-1")
    callback.on_tool_end(ToolMessage(content="DECISIVE-42", tool_call_id="task-1"),
                         run_id="task-1", parent_run_id="model-1")
    trace = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert [event["event"] for event in trace] == [
        "model_start", "tool_start", "tool_end"]
    assert trace[1]["parent_run_id"] == trace[2]["parent_run_id"] == "model-1"
    assert "DECISIVE-42" in str(trace[2]["output"])
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert events[1]["subagent"] == "research-agent"


def test_provider_trace_records_actual_extra_body_without_network(tmp_path):
    from assist.thread import Thread
    from assist.thread_queue import THREAD_QUEUE
    from langchain_core.messages import HumanMessage
    from langchain_openai import ChatOpenAI

    original = (Thread.__init__, Thread._run, ChatOpenAI._stream,
                ChatOpenAI._generate)
    try:
        ChatOpenAI._generate = lambda _model, _messages, **_kwargs: object()
        install_model_and_tool_trace(tmp_path)
        model = ChatOpenAI(model="fixture-model", api_key="fixture-key",
                           base_url="http://127.0.0.1:1",
                           extra_body={"top_k": 20})
        model._generate([HumanMessage(content="Synthetic")])
    finally:
        Thread.__init__, Thread._run, ChatOpenAI._stream, ChatOpenAI._generate = original
        if "acquire" in vars(THREAD_QUEUE):
            del THREAD_QUEUE.acquire
    events = [json.loads(line) for line in
              (tmp_path / "provider-calls.jsonl").read_text().splitlines()]
    assert events[0]["event"] == "provider_request"
    assert events[0]["extra_body"] == {"top_k": 20}
    assert events[1]["event"] == "provider_finished"
    assert events[1]["completed"]


def test_artifact_observer_keeps_provider_bytes_and_time_separate(tmp_path):
    (tmp_path / "supervisor.json").write_text(json.dumps({
        "checkpoint_threads": ["parent"], "timed_out": False,
        "worker_exit_code": 0, "elapsed_seconds": 5,
        "run_reconciliation": {"after": {}},
    }))
    (tmp_path / "checkpoint-latest.json").write_text(json.dumps({
        "threads": [{"thread_id": "parent", "checkpoint_ns": "",
                     "messages": [{"type": "ai", "data": {
                         "content": "Done", "tool_calls": []}}]}]}))
    (tmp_path / "trace.jsonl").write_text("")
    (tmp_path / "lifecycle.jsonl").write_text(json.dumps({
        "event": "note_update_available", "at": 4.0}) + "\n")
    (tmp_path / "provider-calls.jsonl").write_text("\n".join(map(json.dumps, [
        {"event": "provider_request", "request_id": "m1",
         "messages": [{"type": "human", "data": {"content": "Question"}}]},
        {"event": "provider_finished", "request_id": "m1",
         "elapsed_ms": 1250, "completed": True},
    ])) + "\n")
    result = summarize(tmp_path)
    assert result["classification"] == "delivered_within_window"
    assert result["provider_requests"] == 1
    assert result["provider_finished"] == 1
    assert result["provider_elapsed_seconds_sum"] == 1.25
    assert result["provider_message_json_bytes_sum"] > 0


def test_observer_uses_submitted_parent_not_sorted_child_checkpoint(tmp_path):
    supervisor = {"checkpoint_threads": ["a-child", "z-parent"],
                  "timed_out": False, "worker_exit_code": 0,
                  "elapsed_seconds": 5, "run_reconciliation": {"after": {}}}
    checkpoint = {"threads": [
        {"thread_id": "a-child", "checkpoint_ns": "", "messages": [
            {"type": "ai", "data": {"content": "Child report"}}]},
        {"thread_id": "z-parent", "checkpoint_ns": "", "messages": [
            {"type": "ai", "data": {"content": "Final parent reply"}}]},
    ]}
    result = summarize_records(supervisor, checkpoint, [], artifact_ready=True,
                               parent_thread_id="z-parent")
    assert result["classification"] == "delivered_within_window"
    assert result["parent_final_reply"]


@pytest.mark.parametrize("stderr,expected", [
    ("production is busy (running=1); not starting llm work", "admission_denied"),
    ("ImportError: missing fixture", "setup_failed_before_checkpoint"),
])
def test_no_checkpoint_distinguishes_admission_from_setup(tmp_path, stderr, expected):
    (tmp_path / "supervisor.json").write_text(json.dumps({
        "checkpoint_threads": [], "timed_out": False,
        "worker_exit_code": 1, "elapsed_seconds": 0.5,
    }))
    (tmp_path / "worker.stderr.log").write_text(stderr)
    result = summarize(tmp_path)
    assert result["classification"] == expected
    assert not result["checkpoint_available"]


def test_no_checkpoint_after_model_is_not_called_setup(tmp_path):
    (tmp_path / "supervisor.json").write_text(json.dumps({
        "checkpoint_threads": [], "timed_out": False,
        "worker_exit_code": 1, "elapsed_seconds": 2.0,
    }))
    (tmp_path / "worker.stderr.log").write_text("worker failed after inference")
    (tmp_path / "trace.jsonl").write_text(json.dumps({
        "event": "model_start", "at": 1.0}) + "\n")
    result = summarize(tmp_path)
    assert result["classification"] == "checkpoint_missing_after_model"
    assert result["model_starts"] == 1


def test_no_model_run_store_to_supervisor_to_observer_smoke(tmp_path):
    """Use the real durable Run store and eval postflight, with no agent inference."""
    root = tmp_path / "threads"
    root.mkdir()
    (root / "parent").mkdir()
    service = RunService(str(root))
    parent = service.create("parent", "general-agent", "Update my notes")
    child = service.create("sub-research", "research-agent", "Check sources",
                           mode="child", parent_thread_id="parent",
                           parent_run_id=parent.id, dispatch_key="research-1")
    assert _completed_parent_run(service.list("parent"),
                                 service.list("sub-research")) is None
    service.transition(child.thread_id, child.id, "running")
    service.transition(child.thread_id, child.id, "success", result="DECISIVE-42")
    service.transition(parent.thread_id, parent.id, "running")
    service.transition(parent.thread_id, parent.id, "success")
    continuation = service.create("parent", "general-agent", None,
                                  work_id="task-completion:sub-research")
    service.transition("parent", continuation.id, "running")
    service.transition("parent", continuation.id, "success")
    assert _completed_parent_run(service.list("parent"),
                                 service.list("sub-research")).id == continuation.id
    reconciliation = _reconcile_private_runs(root)
    supervisor = {"checkpoint_threads": ["parent"], "timed_out": False,
                  "worker_exit_code": 0, "elapsed_seconds": 1.0,
                  "run_reconciliation": reconciliation}
    checkpoint = {"threads": [{"thread_id": "parent", "checkpoint_ns": "",
                               "messages": [{"type": "human", "data": {"content": "Update my notes"}},
                                            {"type": "ai", "data": {"content": "Done",
                                                                    "tool_calls": []}}]}]}
    observed = summarize_records(supervisor, checkpoint, [], artifact_ready=True)
    assert observed["classification"] == "delivered_within_window"
    assert observed["child_runs"] == 1
    assert observed["max_durable_child_depth"] == 1
    assert observed["missing_durable_parent_ids"] == []
    after = reconciliation["after"]
    assert after["sub-research"][0]["parent_run_id"] == parent.id
    assert after["sub-research"][0]["assistant_id"] == "research-agent"
    assert after["parent"][-1]["status"] == "success"
    assert after["parent"][-1]["work_id"] == "task-completion:sub-research"


def test_completed_child_id_is_absent_for_all_cancelled_final():
    from types import SimpleNamespace

    assert _completed_child_id(SimpleNamespace(work_id="task-completion:sub-1")) == "sub-1"
    assert _completed_child_id(SimpleNamespace(work_id="initial-opaque")) is None


def test_observer_reports_missing_parent_without_inventing_ancestry():
    supervisor = {"checkpoint_threads": ["parent"], "timed_out": False,
                  "worker_exit_code": 0, "elapsed_seconds": 1,
                  "run_reconciliation": {"after": {"sub-x": [
                      {"id": "child", "status": "success", "parent_run_id": "lost"}]}}}
    checkpoint = {"threads": [{"thread_id": "parent", "checkpoint_ns": "",
                               "messages": [{"type": "ai", "data": {"content": "Done"}}]}]}
    observed = summarize_records(supervisor, checkpoint, [], artifact_ready=True)
    assert observed["missing_durable_parent_ids"] == ["lost"]
    assert observed["max_durable_child_depth"] == 0
    assert observed["classification"] == "incomplete_or_error"


def test_old_run_rows_without_parent_fields_report_unknown_ancestry():
    supervisor = {"checkpoint_threads": ["parent"], "timed_out": False,
                  "worker_exit_code": 0, "elapsed_seconds": 5,
                  "run_reconciliation": {"after": {"parent": [
                      {"id": "p1", "status": "success"},
                      {"id": "p2", "status": "success"}]}}}
    checkpoint = {"threads": [{"thread_id": "parent", "checkpoint_ns": "",
                               "messages": [{"type": "ai", "data": {
                                   "content": "Done", "tool_calls": []}}]}]}
    result = summarize_records(supervisor, checkpoint, [], artifact_ready=True,
                               lifecycle=[{"event": "worker_finished",
                                           "delivered": True}])
    assert result["classification"] == "delivered_within_window"
    assert result["durable_ancestry_available"] is False
    assert result["child_runs"] is None
    assert result["max_durable_child_depth"] is None


@pytest.mark.parametrize("status,classification", [
    ("timeout", "product_run_timeout"),
    ("error", "product_run_error"),
    ("cancelled", "cancelled_or_incomplete"),
])
def test_run_terminal_reason_stays_distinct_from_external_cutoff(status, classification):
    supervisor = {"checkpoint_threads": ["parent"], "timed_out": False,
                  "worker_exit_code": 0, "elapsed_seconds": 1,
                  "run_reconciliation": {"after": {"parent": [
                      {"id": "one", "status": status, "parent_run_id": None}]}}}
    checkpoint = {"threads": [{"thread_id": "parent", "checkpoint_ns": "",
                               "messages": [{"type": "ai", "data": {"content": "draft"}}]}]}
    assert summarize_records(supervisor, checkpoint, [],
                             artifact_ready=False)["classification"] == classification


def test_preflight_rejects_runtime_or_eval_overlay_drift(tmp_path):
    product = tmp_path / "product"
    deployed = tmp_path / "deployed"
    for root in (product, deployed):
        for name in ("assist", "manage", "dockerfiles"):
            (root / name).mkdir(parents=True)
            (root / name / "sentinel.py").write_text("same\n")
    local = Path(__file__).resolve().parents[1] / "edd" / "eval"
    overlay = product / "edd" / "eval"
    overlay.mkdir(parents=True)
    for name in ("research_trace.py", "research_nonconvergence.py"):
        (overlay / name).write_bytes((local / name).read_bytes())
    for path in local.glob("research_live_*"):
        if path.is_file() and path.suffix in {".py", ".json"}:
            (overlay / path.name).write_bytes(path.read_bytes())
    source, duration = preflight("music_bundle", product, deployed)
    assert duration == 2700
    assert source["eval_files"]["research_live_cases.json"]
    _, direct_duration = preflight("music_bundle", product, deployed,
                                   profile="research-manager-capability")
    assert direct_duration == 600
    (overlay / "research_live_cases.json").write_text("changed")
    with pytest.raises(RuntimeError, match="changed eval overlay"):
        preflight("music_bundle", product, deployed)
    (overlay / "research_live_cases.json").write_bytes(
        (local / "research_live_cases.json").read_bytes())
    (product / "assist" / "sentinel.py").write_text("changed\n")
    with pytest.raises(RuntimeError, match="differs from the declared"):
        preflight("music_bundle", product, deployed)
