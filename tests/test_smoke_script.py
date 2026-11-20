"""Run the existing smoke shell flow against a private deterministic HTTP fixture."""
from email import policy
from email.parser import BytesParser
import http.server
import os
from pathlib import Path
import re
import subprocess
import threading
from urllib.parse import parse_qs, urlsplit

import pytest


@pytest.mark.parametrize("hermes, failure", [(False, None), (True, None), (True, "missing-file"),
                                             (True, "error"), (True, "disabled")])
def test_smoke_keeps_deep_flow_and_verifies_hermes_file_and_cleanup(tmp_path, hermes, failure):
    state = {"threads": {}, "created": [], "deleted": [], "prompts": [], "file_reads": 0}

    class Server(http.server.BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def respond(self, status, body="", location=None):
            self.send_response(status)
            if location:
                self.send_header("Location", location)
            raw = body.encode()
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_POST(self):
            data = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            message = BytesParser(policy=policy.default).parsebytes(
                ("Content-Type: " + self.headers.get("Content-Type", "") + "\r\n\r\n").encode() + data)
            fields = {part.get_param("name", header="content-disposition"): part.get_content()
                      for part in message.iter_parts()} if message.is_multipart() else {}
            if self.path == "/threads/with-message":
                engine = fields.get("engine", "deepagents")
                if engine == "hermes" and failure == "disabled":
                    return self.respond(503)
                tid = f"smoke-{len(state['created']) + 1}"
                state["created"].append((tid, engine))
                state["threads"][tid] = {"engine": engine, "count": 1}
                state["prompts"].append(fields["text"])
                if engine == "hermes":
                    match = re.search(r"at (/tmp/(hermes-smoke-[a-f0-9]+)\.txt)", fields["text"])
                    assert match
                    state["path"], state["token"] = match.groups()
                return self.respond(303, location=f"/thread/{tid}")
            _, _, tid, action = self.path.split("/")
            assert tid in state["threads"], "smoke touched a thread it did not create"
            if action == "message":
                state["prompts"].append(fields["text"])
                state["threads"][tid]["count"] += 1
                return self.respond(303, location=f"/thread/{tid}")
            assert action == "delete"
            state["deleted"].append(tid)
            del state["threads"][tid]
            return self.respond(303, location="/")

        def do_GET(self):
            parsed = urlsplit(self.path)
            if parsed.path == "/":
                return self.respond(200, "unrelated-existing-thread " + " ".join(state["threads"]))
            tid = parsed.path.split("/")[2]
            if tid not in state["threads"]:
                return self.respond(404)
            record = state["threads"][tid]
            if parsed.path.endswith("/status"):
                return self.respond(200, '{"stage":"' + ("error" if failure == "error" else "ready") + '"}')
            if parsed.path.endswith("/show"):
                state["file_reads"] += 1
                assert parse_qs(parsed.query)["path"] == [state["path"]]
                if failure == "missing-file":
                    return self.respond(404)
                return self.respond(200, '<pre class="show-text">' + state["token"] + '\n</pre>')
            return self.respond(200, '\n'.join('<div class="msg assistant">Done</div>'
                                               for _ in range(record["count"])))

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Server)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        env = dict(os.environ, ASSIST_SMOKE_HERMES="1" if hermes else "0")
        result = subprocess.run(["bash", str(Path(__file__).parents[1] / "scripts/smoke_test.sh"),
                                 f"http://127.0.0.1:{server.server_port}"],
                                env=env, capture_output=True, text=True, timeout=20)
    finally:
        server.shutdown()
        server.server_close()
        worker.join(2)
    assert (result.returncode == 0) == (failure is None), result.stdout + result.stderr
    assert state["created"][0][1] == "deepagents"
    assert state["prompts"][:2] == [
        "This is an automated smoke test. Please respond with a short greeting.",
        "Thanks! Can you repeat what you just said in one sentence?"]
    assert not state["threads"]
    assert state["deleted"] == [tid for tid, _engine in state["created"]]
    if hermes and failure != "disabled":
        assert state["created"][1][1] == "hermes"
    if hermes and failure in {None, "missing-file"}:
        assert state["file_reads"] == 1
    if hermes and failure is None:
        assert "Hermes provider, file write and final HTTP reply completed" in result.stdout
