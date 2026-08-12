from __future__ import annotations

import json
import os
import io
import urllib.request

from assist.pi_health import (
    HEALTH_TTL_SECONDS,
    _NoRedirect,
    _initial_user_messages_sha256,
    _provider_identity_snapshot,
    _sha256,
    preview_health_admits,
)


def _service():
    return {
        "pid": 42, "start_ticks": 9, "invocation": "a" * 32,
        "activation_monotonic_us": 10, "control_group": "/system.slice/provider.service",
        "memory_current": 100, "memory_peak": 200, "oom": 0, "oom_kill": 0,
    }


def _record(now: int):
    service = _service()
    snapshot = {"upstream_sha256": "d" * 64, "model": "model",
                "provider_build_sha256": "b" * 64, "service": service}
    def census(engine):
        non_user_messages = []
        return [{
            "system_count": 1, "system_sha256": "c" * 64,
            "non_user_messages": non_user_messages,
            "non_user_messages_sha256": _sha256(non_user_messages),
            "user_count": 1,
            "user_messages_sha256": _initial_user_messages_sha256(engine),
            "tools_sha256": "e" * 64, "tool_count": 0, "tool_names": [],
            "model": "model", "temperature": 0.1,
        }]

    def arm(engine):
        pre_model_commands = 0 if engine == "pi" else 1
        request_census = census(engine)
        return {
            "engine": engine, "passed": True,
            "checks": {key: True for key in (
                "controller", "containment", "driver", "one_completion", "census",
                "no_model_tools", "cleanup")},
            "provider": {
                "chat_attempts": 1, "successful_chat_responses": 1,
                "chat_failures": 0, "bound_rejections": 0,
                "request_census": request_census,
                "request_census_sha256": _sha256(request_census),
            },
            "tools": {"commands": pre_model_commands,
                      "pre_model_commands": pre_model_commands,
                      "post_model_commands": 0, "rejections": 0, "error": False},
            "driver": {"status": "completed"}, "runner_release_sha256": "a" * 64,
        }
    return {"schema": 1, "id": "provider-health-test", "status": "passed",
            "started_at_ns": now - 2, "finished_at_ns": now - 1,
            "snapshots": {"before": snapshot, "after_pi": snapshot, "final": snapshot},
            "arms": {"pi": arm("pi"), "deepagents": arm("deepagents")},
            "reason_codes": []}


def _write(root, record):
    root.mkdir(mode=0o700)
    (root / "latest.json").write_text(json.dumps(record))
    os.chmod(root / "latest.json", 0o600)
    (root / "EXECUTION.lock").touch(mode=0o600)


def _snapshot():
    return {"upstream_sha256": "d" * 64, "model": "model",
            "provider_build_sha256": "b" * 64, "service": _service()}


def test_fresh_stable_health_record_admits(tmp_path):
    now = 1_000_000_000_000
    root = tmp_path / "health"
    _write(root, _record(now))
    assert preview_health_admits(root, "provider.service", now_ns=now,
                                 snapshot=lambda _: _snapshot())


def test_stale_active_changed_and_oom_health_records_do_not_admit(tmp_path):
    now = 1_000_000_000_000
    root = tmp_path / "health"
    record = _record(now)
    _write(root, record)
    assert not preview_health_admits(root, "provider.service",
                                     now_ns=now + (HEALTH_TTL_SECONDS + 1) * 1_000_000_000,
                                     snapshot=lambda _: _snapshot())
    (root / "ACTIVE.json").write_text("active")
    assert not preview_health_admits(root, "provider.service", now_ns=now,
                                     snapshot=lambda _: _snapshot())
    (root / "ACTIVE.json").unlink()
    changed = _snapshot() | {"service": _service() | {"pid": 99}}
    assert not preview_health_admits(root, "provider.service", now_ns=now,
                                     snapshot=lambda _: changed)
    record["snapshots"]["final"]["service"]["oom_kill"] = 1
    (root / "latest.json").write_text(json.dumps(record))
    assert not preview_health_admits(root, "provider.service", now_ns=now,
                                     snapshot=lambda _: _snapshot())


def test_unsafe_health_root_and_record_do_not_admit(tmp_path):
    now = 1_000_000_000_000
    root = tmp_path / "health"
    _write(root, _record(now))
    os.chmod(root / "latest.json", 0o666)
    assert not preview_health_admits(root, "provider.service", now_ns=now,
                                     snapshot=lambda _: _snapshot())


def test_symlinked_health_record_does_not_admit(tmp_path):
    now = 1_000_000_000_000
    root = tmp_path / "health"
    _write(root, _record(now))
    target = tmp_path / "record.json"
    target.write_text(json.dumps(_record(now)))
    (root / "latest.json").unlink()
    (root / "latest.json").symlink_to(target)
    assert not preview_health_admits(root, "provider.service", now_ns=now,
                                     snapshot=lambda _: _snapshot())


def test_missing_or_changed_provider_evidence_does_not_admit(tmp_path):
    now = 1_000_000_000_000
    root = tmp_path / "health"
    assert not preview_health_admits(root, "provider.service", now_ns=now,
                                     snapshot=lambda _: _snapshot())
    _write(root, _record(now))
    changed_model = _snapshot() | {"model": "different"}
    assert not preview_health_admits(root, "provider.service", now_ns=now,
                                     snapshot=lambda _: changed_model)
    record = _record(now)
    record["arms"]["pi"] = {"engine": "pi", "passed": True}
    (root / "latest.json").write_text(json.dumps(record))
    assert not preview_health_admits(root, "provider.service", now_ns=now,
                                     snapshot=lambda _: _snapshot())


def test_malformed_health_request_census_does_not_admit(tmp_path):
    now = 1_000_000_000_000
    root = tmp_path / "health"
    record = _record(now)
    record["arms"]["pi"]["provider"]["request_census"] = [{"model": "model"}]
    record["arms"]["pi"]["provider"]["request_census_sha256"] = _sha256(
        record["arms"]["pi"]["provider"]["request_census"])
    _write(root, record)
    assert not preview_health_admits(root, "provider.service", now_ns=now,
                                     snapshot=lambda _: _snapshot())


def test_provider_identity_disables_proxy_and_redirects(monkeypatch):
    captured = {}

    class Opener:
        def open(self, request, timeout):
            captured.setdefault("requests", []).append((request.full_url, timeout))
            if request.full_url.endswith("/models"):
                return io.BytesIO(json.dumps({"data": [{"id": "model", "meta": {}}]}).encode())
            return io.BytesIO(json.dumps({
                "build_info": "build", "model_alias": "model", "model_path": "/model",
                "total_slots": 1, "chat_template": "template", "chat_template_caps": {},
                "modalities": {}, "default_generation_settings": {"params": {}, "n_ctx": 1},
            }).encode())

    def build_opener(*handlers):
        captured["handlers"] = handlers
        return Opener()

    monkeypatch.setenv("ASSIST_MODEL_URL", "http://127.0.0.1:8000/v1")
    monkeypatch.setattr(urllib.request, "build_opener", build_opener)
    assert _provider_identity_snapshot()["model"] == "model"
    assert any(isinstance(handler, urllib.request.ProxyHandler)
               and handler.proxies == {} for handler in captured["handlers"])
    assert any(isinstance(handler, _NoRedirect) for handler in captured["handlers"])
