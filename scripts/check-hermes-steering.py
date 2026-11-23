"""Check the real packaged Hermes steering/persistence contract on every upgrade.

Run with /opt/hermes/.venv/bin/python in a network:none runtime container,
mounting this script and the candidate worker read-only. Uses actual AIAgent,
ticket drain, formatter and SessionDB; client construction, tool discovery and
requirements, plus injected DB write/ack failures are stubbed. No provider/model/tool request is made. The
separate check-hermes-native.py covers the complete callback/tool/model ordering.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sqlite3
import sys
import tempfile
from unittest.mock import patch

signal.alarm(45)
root = Path(tempfile.mkdtemp(prefix="hermes-steering-compat-"))
home = root / "hermes"
home.mkdir()
os.environ.update(HERMES_HOME=str(home), HERMES_SESSION_PLATFORM="api_server",
                  TERMINAL_CWD=str(root))
sys.path.insert(0, "/opt/hermes")
from run_agent import AIAgent
from hermes_state import SessionDB
from agent.context_compressor import _DB_PERSISTED_MARKER
from agent.prompt_builder import steer_user_row

spec = importlib.util.spec_from_file_location("worker", "/opt/assist/hermes_worker.py")
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)
database = SessionDB(home / "session.db")
with (patch("model_tools.get_tool_definitions", return_value=[]),
      patch("model_tools.check_toolset_requirements", return_value={}),
      patch("agent.process_bootstrap.OpenAI")):
    agent = AIAgent(api_key="synthetic", base_url="http://127.0.0.1:9/v1",
                    provider="custom", quiet_mode=True, skip_context_files=True,
                    skip_memory=True, enabled_toolsets=[], session_db=database,
                    session_id="assist-steering-compat")
original_flush = agent._flush_messages_to_session_db
original_drain = agent._drain_pending_steer
original_step, original_tool = agent.step_callback, agent.tool_complete_callback
tickets, acknowledgments = [], []
lose_ack = True


def stored():
    with sqlite3.connect(home / "session.db") as connection:
        return [json.loads(raw)["assist_steering"] for (raw,) in connection.execute(
            "SELECT display_metadata FROM messages WHERE display_kind='steer'")]


def control(_path, payload):
    if payload.get("consumed"):
        durable = {rid for receipt in stored() for rid in receipt["run_ids"]}
        assert set(payload["consumed"]) <= durable, "Acknowledged before native commit"
        if lose_ack:
            raise OSError("Synthetic lost acknowledgment")
        acknowledgments.extend(payload["consumed"])
    return {"messages": list(tickets)}


worker.control = control
delivery = worker._Steering(agent, "synthetic-work")


def drain(*ids):
    tickets[:] = [{"id": rid, "text": f"[Assist follow-up id={rid}]\nSame text\n"}
                  for rid in ids]
    agent.tool_complete_callback("synthetic-tool", "synthetic", {}, "synthetic result")
    agent.step_callback(1, [])
    text = agent._drain_pending_steer()
    assert agent._steer_last_drain == (text, [(item["id"], item["text"].strip())
                                            for item in tickets])
    return steer_user_row(text)


try:
    row = drain("same-a", "same-b")
    # Empty-response retries can drain again before the first row flush.
    assert agent._drain_pending_steer() is None
    assert agent._steer_last_drain == (None, [])
    assert not row.get(_DB_PERSISTED_MARKER)
    assert agent._flush_messages_to_session_db([row]) is True
    assert row[_DB_PERSISTED_MARKER] is True
    assert (row.get("display_metadata") or {}).get("assist_steering") == {
        "work_id": "synthetic-work", "run_ids": ["same-a", "same-b"]
    }, "First native steer row lost its receipt after an empty drain"
    assert stored() == [{"work_id": "synthetic-work", "run_ids": ["same-a", "same-b"]}]
    assert not delivery.drained, "Committed associations retained in memory after lost ack"
    assert not acknowledgments and set(delivery.offered) == {"same-a", "same-b"}
    lose_ack = False
    assert agent._flush_messages_to_session_db([row]) is True
    assert acknowledgments == ["same-a", "same-b"] and not delivery.offered
    assert len(stored()) == 1, "Repeated native flush duplicated the durable row"

    failed = drain("retry")
    with patch.object(database, "append_messages_batch",
                      side_effect=sqlite3.OperationalError("Synthetic write failure")):
        assert agent._flush_messages_to_session_db([failed]) is False
    assert not failed.get(_DB_PERSISTED_MARKER) and "retry" not in acknowledgments
    assert len(delivery.drained) == 1
    assert len(stored()) == 1
    assert agent._flush_messages_to_session_db([failed]) is True
    assert failed[_DB_PERSISTED_MARKER] is True and "retry" in acknowledgments
    assert not delivery.drained

    disabled = drain("disabled")
    agent._persist_disabled = True
    try:
        assert agent._flush_messages_to_session_db([disabled]) is None
    finally:
        agent._persist_disabled = False
    assert not disabled.get(_DB_PERSISTED_MARKER) and "disabled" not in acknowledgments
    assert len(delivery.drained) == 1
    assert agent._flush_messages_to_session_db([disabled]) is True
    assert disabled[_DB_PERSISTED_MARKER] is True and "disabled" in acknowledgments
    assert not delivery.drained
finally:
    delivery.finish()
    assert agent._flush_messages_to_session_db == original_flush
    assert agent._drain_pending_steer == original_drain
    assert agent.step_callback is original_step and agent.tool_complete_callback is original_tool
    agent.close()
    database.close()

print(json.dumps({"status": "PASS", "marker": _DB_PERSISTED_MARKER,
    "acknowledgments": acknowledgments, "provider_calls": 0,
    "native_sources": {name: hashlib.sha256((Path('/opt/hermes') / name).read_bytes()).hexdigest()
        for name in ("agent/session_persistence.py", "agent/interrupt_control.py", "agent/prompt_builder.py")},
    "coverage": "Actual native True/False/None flush results, marker, exact drain tickets, durable receipt metadata, lost-ack retry and restoration; no model judgment or host admission."}))
