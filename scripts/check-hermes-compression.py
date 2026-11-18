"""Pinned native compression/next-turn capability, run inside the runtime image.

Mount this script read-only and run it with /opt/hermes/.venv/bin/python in a
network:none container. It uses temporary native state, a private scripted HTTP
provider and scripted compressor output. Native compression publication, close,
tip resolution and the subsequent worker conversation remain real. This does
not exercise the host queue, model judgment or summary quality.
"""
import http.server
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
from unittest.mock import patch

signal.alarm(120)
root = Path(tempfile.mkdtemp(prefix="hermes-compression-"))
os.environ["HERMES_HOME"] = str(root)
os.environ["TERMINAL_CWD"] = str(root)
# Default Hermes compression is in-place. Exercise its supported rotation mode.
(root / "config.yaml").write_text("compression:\n  in_place: false\n")
sys.path.insert(0, "/opt/hermes")
spec = importlib.util.spec_from_file_location("worker", "/opt/assist/hermes_worker.py")
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)
PROMPT1 = "Remember that the note is blue."
PROMPT2 = "What color is the note?"
SUMMARY = "The user asked to remember that the note is blue."
seen, results, rotation = [], [], []


class Provider(http.server.BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        messages = request.get("messages", [])
        if any("You name chat sessions." in str(m.get("content")) for m in messages):
            answer = '{"title":"Blue note"}'
        else:
            if any(m.get("role") == "user" and m.get("content") == PROMPT2 for m in messages):
                assert any(SUMMARY in str(m.get("content")) for m in messages), "Provider lost compressed history"
            seen.append(messages)
            answer = "The note is blue."
            if len(seen) == 1:
                answer += " ".join(f"Background record {i}: field value {i*i}." for i in range(500))
        response = {
            "id": "fixture", "object": "chat.completion", "model": "fixture",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": answer},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 100, "total_tokens": 200},
        }
        content_type = "application/json"
        if request.get("stream"):
            response["object"] = "chat.completion.chunk"
            response["choices"][0]["delta"] = response["choices"][0].pop("message")
            body = ("data: " + json.dumps(response) + "\n\ndata: [DONE]\n\n").encode()
            content_type = "text/event-stream"
        else:
            body = json.dumps(response).encode()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        body = b'{"data":[{"id":"fixture","context_length":131072}]}'
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


# This isolates persistence. The separate native capability exercises real host
# admission and hooks; here the worker's provider listener serves scripted data.
worker._ProviderBridge = Provider
worker.control = lambda path, value: results.append((path, value))
worker._register_boundaries = lambda: None
from run_agent import AIAgent
from hermes_state import SessionDB

original = AIAgent.run_conversation


def conversation(self, prompt, **kwargs):
    if prompt == PROMPT2:
        assert self.session_id != "compression-proof", "Worker reopened original ID after compression"
        assert any(SUMMARY in str(m.get("content")) for m in kwargs["conversation_history"]), "Summary missing from next turn"
    result = original(self, prompt, **kwargs)
    assert not result.get("failed"), result
    if prompt == PROMPT1:
        self._compression_feasibility_checked = True
        old = self.session_id
        database = SessionDB(root / "session.db")
        history = database.get_messages_as_conversation(old, include_ancestors=True)
        database.close()
        # Script only the summary output. The native forced-compression path
        # still acquires its lease, publishes the child and rotates the session.
        with patch.object(type(self.context_compressor), "compress",
                          return_value=[{"role": "user", "content": SUMMARY}]):
            self._compress_context(history, "fixture", approx_tokens=20000, force=True)
        rotation.append((old, self.session_id))
        assert self.session_id != old, "Native forced compression did not rotate"
    return result


AIAgent.run_conversation = conversation
for index, prompt in enumerate((PROMPT1, PROMPT2)):
    request = root / f"request{index}.json"
    request.write_text(json.dumps({"session_id": "compression-proof", "model": "fixture",
                                   "toolsets": [], "prompt": prompt}))
    worker.main(str(request))
    if index == 0:
        database = SessionDB(root / "session.db")
        tip = database.get_compression_tip("compression-proof")
        assert tip != "compression-proof"
        assert database.get_session(tip)["ended_at"] is not None, "Worker did not close rotated tip"
        database.close()
assert len(seen) == 2
assert sum(path == "/result" for path, _value in results) == 2
print(json.dumps({
    "status": "PASS", "rotation": rotation, "provider_calls": len(seen),
    "limits": "Actual pinned worker and native forced compression/rotation/close; scripted summary "
              "output and private provider. No host queue, real model or summary-quality acceptance.",
}))
