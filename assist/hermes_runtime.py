"""Keep one native Hermes invocation inside its existing Run-owned generation."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import shutil
import tempfile
import threading

from assist.browser import authority
from assist.hermes_provider_relay import HermesProviderRelay, HermesRelayError
from assist.model_manager import current_model_config
from assist.sandbox_manager import SandboxManager
from assist.thread_queue import ThreadHoldExpired, ThreadPauseRequested


class HermesRuntimeError(RuntimeError):
    pass


@dataclass
class _Turn:
    work_id: str
    work_dir: str
    control_dir: Path
    relay: HermesProviderRelay
    owner_run_id: str
    thread_scope: tuple[str, str]
    backend: object = None
    finished: threading.Event = field(default_factory=threading.Event)
    error: Exception | None = None
    diagnostic: bytes = b""
    paused: bool = False


class HermesRuntimeManager:
    """Retain process handles across scheduler slices, never reconstruct a turn."""

    def __init__(self, sandbox_manager=SandboxManager, model_config=current_model_config):
        self.sandboxes = sandbox_manager
        self.model_config = model_config
        self.lock = threading.RLock()
        self.turns: dict[str, _Turn] = {}
        self.retired: set[str] = set()

    def _start(self, run, work_dir, agent_dir, thread_scope, admitted, egress_tools):
        model = self.model_config()
        # Unix socket paths are limited to 108 bytes. Keep this private,
        # ephemeral mount independent of the user's potentially long root.
        directory = Path(tempfile.mkdtemp(prefix="assist-hermes-"))
        try:
            relay = HermesProviderRelay(directory, model.url, model.api_key, model.model,
                                        model.context_len, admitted, egress_tools=egress_tools)
        except BaseException:
            shutil.rmtree(directory)
            raise
        turn = _Turn(run.work_id, work_dir, directory, relay, run.id, thread_scope)
        relay.admitted = lambda: self._provider_admitted(turn, admitted)
        with self.lock:
            if run.thread_id in self.turns or run.thread_id in self.retired:
                relay.server.server_close()
                shutil.rmtree(directory)
                raise HermesRuntimeError("A native invocation already owns this thread")
            self.turns[run.thread_id] = turn
        try:
            Path(agent_dir).mkdir(parents=True, exist_ok=True)
            from langchain_core.utils.function_calling import convert_to_openai_tool
            schemas = [convert_to_openai_tool(tool) for tool in egress_tools]
            for schema in schemas:
                function = schema["function"]
                if function["name"] in {"request_egress", "request_egress_batch"}:
                    function["description"] = (
                        "Ask the user for outbound access to exact host/port destinations for this thread. "
                        "A proxy or CONNECT 403 may mean access is not approved; inspect list_allowed_hosts. "
                        "This records approval cards, never grants access. Use task to describe the "
                        "complete follow-up after the user's decision. If pending, finish your answer "
                        "and say approval is needed; do not retry until approved. A denial means "
                        "continue without that access, not request it again.")
            (directory / "request.json").write_text(json.dumps({
                "prompt": run.text, "session_id": run.thread_id, "model": model.model,
                "system_message": os.environ.get("ASSIST_HERMES_SYSTEM_MESSAGE", ""),
                "egress_tools": schemas,
                "toolsets": ["file", "terminal", "skills", "memory", "delegation"]
                            + (["assist_egress"] if schemas else [])}))
            relay.start()
            turn.backend = self.sandboxes.get_hermes_sandbox_backend(
                work_dir, agent_dir=agent_dir, control_dir=str(directory),
                thread_scope=thread_scope, owner_run_id=run.id, tz=(run.rider or {}).get("tz"))
            if turn.backend is None:
                raise HermesRuntimeError("Hermes sandbox is unavailable")

            def execute():
                try:
                    container = turn.backend.container
                    execution = container.client.api.exec_create(container.id, [
                        "/opt/hermes/.venv/bin/python", "/opt/assist/hermes_worker.py",
                        "/run/assist-hermes/request.json"], stdout=True, stderr=True)
                    # Attaching neither stream makes Docker return before exit.
                    # Keep only a private bounded tail for failure diagnosis.
                    for chunk in container.client.api.exec_start(execution["Id"], stream=True):
                        turn.diagnostic = (turn.diagnostic + chunk)[-8192:]
                    if container.client.api.exec_inspect(execution["Id"])["ExitCode"] != 0:
                        turn.error = HermesRuntimeError("Native Hermes worker failed")
                except Exception as error:
                    turn.error = error
                finally:
                    turn.finished.set()
                    relay.changed.set()

            threading.Thread(target=execute, name="hermes-native-turn", daemon=True).start()
            return turn
        except BaseException:
            self.stop(run.thread_id)
            raise

    def run_slice(self, run, *, work_dir, agent_dir, thread_scope, handle, admitted,
                  egress_tools=()):
        """Execute or resume the same worker under the current queue handle."""
        with self.lock:
            turn = self.turns.get(run.thread_id)
            fresh = turn is None
            if fresh:
                if run.resume:
                    raise HermesRuntimeError("Native Hermes process was lost; retained files were not replayed")
                # Serialize registration and Docker creation with retirement;
                # stop cannot finish while a late generation is still starting.
                turn = self._start(run, work_dir, agent_dir, thread_scope, admitted, egress_tools)
        if not fresh:
            if turn.work_id != run.work_id or not run.resume or not turn.paused:
                raise HermesRuntimeError("Native Hermes invocation does not match this successor")
            turn.relay.admitted = lambda: self._provider_admitted(turn, admitted)
            if self.sandboxes.current_container(work_dir) is not turn.backend.container:
                raise HermesRuntimeError("Retained Hermes generation changed")
            turn.relay.resume()
            turn.paused = False
        while True:
            if handle.expired or turn.relay.stopped or not admitted():
                self.stop(run.thread_id)
                raise ThreadHoldExpired("Hermes model admission ended")
            if turn.finished.is_set():
                result, error = turn.relay.result, turn.error
                self.stop(run.thread_id)
                if error is not None or result is None or result["failed"]:
                    raise HermesRuntimeError("Native Hermes turn did not complete") from error
                return result["reply"]
            if handle.pause_requested and turn.relay.request_pause():
                turn.relay.confirm_parked()
                turn.paused = True
                raise ThreadPauseRequested()
            turn.relay.changed.wait(0.1)
            turn.relay.changed.clear()

    def _provider_admitted(self, turn, admitted):
        """Check current Run admission and the original retained generation lease."""
        if not admitted() or turn.backend is None:
            return False
        container = turn.backend.container
        if self.sandboxes.current_container(turn.work_dir) is not container:
            return False
        try:
            with authority.fence(*turn.thread_scope) as state:
                lease = state.lease
                return bool(lease and lease["owner_run_id"] == turn.owner_run_id
                            and container.id in lease["generations"])
        except (OSError, RuntimeError, ValueError):
            return False

    def release_slice(self, thread_id: str):
        """Prove native model admission is parked, or stop before queue release."""
        with self.lock:
            turn = self.turns.get(thread_id)
            if turn is None:
                return
            if turn.paused:
                if not turn.relay.parked or turn.relay.connections:
                    raise HermesRuntimeError("Hermes parking is unconfirmed")
                return
            self.stop(thread_id)

    def paused_owner(self, thread_id: str, work_id: str) -> str | None:
        """Snapshot the retained owner for a durable reset, without granting authority.

        Cancellation reads this while holding the short journal fence. It must
        not wait on the runtime lock, whose teardown needs that same fence.
        The reset subsequently verifies the exact owner and generation.
        """
        turn = self.turns.get(thread_id)
        return turn.owner_run_id if turn and turn.paused and turn.work_id == work_id else None

    def retire(self, thread_id: str):
        """Prevent registration and stop the current generation before deletion."""
        with self.lock:
            self.retired.add(thread_id)
            self.stop(thread_id)

    def retains_work(self, thread_id: str, work_id: str) -> bool:
        """Snapshot live work without reversing admission/runtime/authority locks.

        This only suppresses duplicate recovery; it grants no model authority.
        """
        turn = self.turns.get(thread_id)
        return turn is not None and turn.work_id == work_id

    def stop_work(self, thread_id: str, work_id: str) -> bool:
        """Prove matching work stopped or absent; never stop unrelated work."""
        with self.lock:
            turn = self.turns.get(thread_id)
            if turn is not None:
                if turn.work_id != work_id:
                    return False
                self.stop(thread_id)
            return True

    def revoke_model(self, thread_id: str, work_id: str) -> bool:
        """Drain this work's provider relay while retaining its cleanup ownership."""
        with self.lock:
            turn = self.turns.get(thread_id)
            if turn is not None:
                if turn.work_id != work_id:
                    return False
                turn.relay.close()
            return True

    def stop(self, thread_id: str):
        """Fence provider access and verify teardown of the exact managed generation."""
        with self.lock:
            turn = self.turns.get(thread_id)
            if turn is None:
                return
            turn.relay.stop_admission()
            if turn.backend is not None:
                self.sandboxes.cleanup_verified(turn.work_dir, turn.backend.container)
                # Preserve completed teardown if relay drain needs another attempt.
                turn.backend = None
                turn.paused = False
            turn.relay.close()
            self.turns.pop(thread_id, None)
            shutil.rmtree(turn.control_dir)
