"""Run one native Hermes conversation inside the owning Assist sandbox.

The host owns the Unix relay and the container lifetime. Native callbacks are
cooperative parking points, not authorization: the relay must independently
fence every provider request, including requests after a failed callback.
"""
from __future__ import annotations

import http.server
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import uuid

import httpx


_MAX_BODY = 2 * 1024 * 1024
_CONTROL = "/run/assist-hermes/provider.sock"


def control(path: str, payload: dict) -> dict:
    """Wait at a host coordination point without starting a provider deadline."""
    with httpx.Client(transport=httpx.HTTPTransport(uds=_CONTROL), timeout=None,
                      trust_env=False) as client:
        response = client.post("http://host" + path, json=payload)
        response.raise_for_status()
        return response.json()


class _ProviderBridge(http.server.BaseHTTPRequestHandler):
    """Expose the private Unix provider relay only on container loopback."""

    def log_message(self, *_args):
        pass

    def do_GET(self):
        self._forward()

    def do_POST(self):
        self._forward()

    def _forward(self):
        try:
            size = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_error(400)
            return
        if not 0 <= size <= _MAX_BODY or self.headers.get("Transfer-Encoding"):
            self.send_error(413)
            return
        self.connection.settimeout(5)
        body = self.rfile.read(size)
        if len(body) != size:
            self.send_error(400)
            return
        with httpx.Client(transport=httpx.HTTPTransport(uds=_CONTROL), timeout=None,
                          trust_env=False) as client:
            with client.stream(self.command, "http://host" + self.path, content=body,
                               headers={"Content-Type": "application/json"}) as response:
                self.send_response(response.status_code)
                self.send_header("Content-Type", response.headers.get("Content-Type", "application/json"))
                self.send_header("Connection", "close")
                self.end_headers()
                transferred = 0
                for chunk in response.iter_raw():
                    transferred += len(chunk)
                    if transferred > _MAX_BODY:
                        raise RuntimeError("Hermes provider response exceeds its bound")
                    self.wfile.write(chunk)
                    self.wfile.flush()


def _register_boundaries():
    from hermes_cli.plugins import PluginContext, PluginManifest, _delivery_manager

    context = PluginContext(PluginManifest(name="assist-host", version="1"), _delivery_manager())

    def execution(request, next_call, **metadata):
        operation = uuid.uuid4().hex
        control("/boundary", {"operation": operation, "kind": "model",
                              "session_id": metadata.get("session_id")})
        try:
            return next_call(request)
        finally:
            control("/settled", {"operation": operation})

    def auxiliary_start(**metadata):
        control("/boundary", {"operation": metadata.get("api_request_id"),
                              "kind": "auxiliary", "aux_task": metadata.get("aux_task"),
                              "session_id": metadata.get("session_id")})

    def auxiliary_end(**metadata):
        control("/settled", {"operation": metadata.get("api_request_id")})

    def tool(tool_name, args, next_call, **metadata):
        operation = uuid.uuid4().hex
        control("/boundary", {"operation": operation, "kind": "tool", "tool": tool_name,
                              "session_id": metadata.get("session_id")})
        try:
            return next_call(args)
        finally:
            control("/settled", {"operation": operation})

    context.register_middleware("llm_execution", execution)
    context.register_middleware("tool_execution", tool)
    context.register_hook("pre_auxiliary_call", auxiliary_start)
    context.register_hook("post_auxiliary_call", auxiliary_end)


def main(request_path: str) -> None:
    """Consume one host request; native SessionDB supplies previous user turns."""
    with open(request_path, "rb") as stream:
        raw = stream.read(_MAX_BODY + 1)
    if len(raw) > _MAX_BODY:
        raise RuntimeError("Hermes turn request exceeds its bound")
    request = json.loads(raw)
    import hermes_yaml as yaml
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    bridge = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _ProviderBridge)
    provider_url = f"http://127.0.0.1:{bridge.server_port}/v1"

    config_path = Path(os.environ["HERMES_HOME"]) / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(config_path.read_text()) if config_path.exists() else {}
    config = config or {}
    # Preserve native preferences while selecting its supported bounded-host
    # execution settings. Provider RPC deadlines remain native and unchanged.
    config.setdefault("plugins", {})["hook_callback_timeout"] = 0
    config.setdefault("compression", {})["context_timeout_seconds"] = 0
    config.setdefault("delegation", {})["child_timeout_seconds"] = None
    config.setdefault("auxiliary", {}).setdefault("background_review", {})["defer"] = "never"
    # This client has no native interactive approval transport. Use Hermes's
    # unattended denial policy instead of waiting on an invisible stdin prompt.
    config.setdefault("approvals", {}).update(mode="manual", unattended_mode="deny")
    config["command_allowlist"] = []
    os.environ["HERMES_SESSION_PLATFORM"] = "api_server"
    # Host admission covers every native model call. Preserve task preferences
    # and deadlines, but select the same host route for children and side work.
    route = {"provider": "custom", "base_url": provider_url,
             "api_key": "host-relay", "model": request["model"],
             "api_mode": "chat_completions"}
    config["delegation"].update(route, backend="default", fallback_providers=[])
    auxiliary = config["auxiliary"]
    for task in set(DEFAULT_CONFIG.get("auxiliary", {})) | set(auxiliary):
        if isinstance(auxiliary.get(task, DEFAULT_CONFIG.get("auxiliary", {}).get(task)), dict):
            auxiliary.setdefault(task, {}).update(route, fallback_chain=[])
    config["fallback_model"] = []
    config["fallback_providers"] = []
    # Container cancellation may happen at any instruction. Never truncate the
    # persistent native preferences before their replacement is complete.
    with tempfile.NamedTemporaryFile(mode="w", dir=config_path.parent, delete=False) as replacement:
        replacement.write(yaml.safe_dump(config))
        replacement.flush()
        os.fsync(replacement.fileno())
    os.replace(replacement.name, config_path)
    from run_agent import AIAgent
    from hermes_state import SessionDB
    from tools.delegate_tool_config import gateway_delegation_inline_scope
    from agent import title_generator

    _register_boundaries()
    threading.Thread(target=bridge.serve_forever, name="hermes-provider", daemon=True).start()
    database = SessionDB(Path(os.environ["HERMES_HOME"]) / "session.db")
    agent = None
    try:
        history = database.get_messages_as_conversation(request["session_id"], include_ancestors=True)
        # The constructor freezes the joined delegate schema. A config.yaml
        # gateway_mode entry alone does not select it for a direct AIAgent.
        with gateway_delegation_inline_scope(True):
            agent = AIAgent(
                base_url=provider_url, api_key="host-relay",
                provider="custom", api_mode="chat_completions", model=request["model"],
                platform="api_server",
                fallback_model={},
                session_id=request["session_id"], session_db=database,
                enabled_toolsets=request["toolsets"], quiet_mode=True, cwd="/workspace")
        result = agent.run_conversation(request["prompt"], conversation_history=history,
                                        system_message=request.get("system_message"))
        # Pinned native review completion and title-thread registries provide
        # joins; the host Run supplies the bound and cancellation. The host
        # stops the entire generation, including background tool processes,
        # before publishing this private worker result to the user.
        review = getattr(agent, "_background_review_run", None)
        if review is not None:
            review.request_done.wait()
        for title in tuple(title_generator._UPGRADE_THREADS):
            title.join()
        control("/result", {"failed": bool(result.get("failed") or result.get("interrupted")),
                            "reply": result.get("final_response", "")})
    finally:
        if agent is not None:
            agent.close()
        database.close()
        bridge.shutdown()
        bridge.server_close()


if __name__ == "__main__":
    main(sys.argv[1])
