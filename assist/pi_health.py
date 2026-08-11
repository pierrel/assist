"""Fail-closed admission evidence for the Pi web preview.

The P1 probe writes the health record; this module only verifies its current
operational meaning.  It deliberately does not import the disposable P1
controller or issue a model completion.
"""
from __future__ import annotations

import json
import hashlib
import os
import re
import stat
import subprocess
import time
import urllib.parse
import urllib.request
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Callable

import fcntl


HEALTH_TTL_SECONDS = 30 * 60
_MAX_RECORD_BYTES = 128 * 1024
_SERVICE_UNIT = re.compile(r"[A-Za-z0-9@_.-]{1,128}\Z")
_INVOCATION = re.compile(r"[0-9a-f]{32}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ARM_CHECKS = {"controller", "containment", "driver", "one_completion", "census",
               "no_model_tools", "cleanup"}
_HEALTH_PROMPT = "Reply with exactly HEALTHY. Do not use tools."


class PiHealthError(RuntimeError):
    """The Pi preview's provider-health evidence is unavailable or malformed."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Do not forward the model credential beyond the configured loopback URL."""

    def redirect_request(self, request, fp, code, message, headers, newurl):
        return None


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"),
                      sort_keys=True).encode()


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _initial_user_messages_sha256(engine: str) -> str:
    if engine == "deepagents":
        return _sha256([_HEALTH_PROMPT])
    if engine == "pi":
        return _sha256([[{"type": "text", "text": _HEALTH_PROMPT}]])
    raise PiHealthError("health record names an unknown engine")


def _census_is_valid(census: object, engine: str, model: object) -> bool:
    if not isinstance(census, list) or not census:
        return False
    required = {"system_count", "system_sha256", "non_user_messages",
                "non_user_messages_sha256", "user_count", "user_messages_sha256",
                "tools_sha256", "tool_count", "tool_names", "model", "temperature"}
    for index, item in enumerate(census):
        if not isinstance(item, dict) or not required <= set(item) or set(item) - required:
            return False
        if (any(type(item[name]) is not int or item[name] < 0
                for name in ("system_count", "user_count", "tool_count"))
                or any(not isinstance(item[name], str)
                       or _SHA256.fullmatch(item[name]) is None
                       for name in ("system_sha256", "non_user_messages_sha256",
                                    "user_messages_sha256", "tools_sha256"))
                or not isinstance(item["tool_names"], list)
                or not all(isinstance(name, str) for name in item["tool_names"])):
            return False
        messages = item["non_user_messages"]
        if (not isinstance(messages, list)
                or not all(isinstance(message, dict)
                           and isinstance(message.get("role"), str)
                           and isinstance(message.get("content_sha256"), str)
                           and _SHA256.fullmatch(message["content_sha256"]) is not None
                           and isinstance(message.get("message_sha256"), str)
                           and _SHA256.fullmatch(message["message_sha256"]) is not None
                           for message in messages)
                or item["non_user_messages_sha256"] != _sha256(messages)
                or item["model"] != model or item["temperature"] != 0.1):
            return False
        if index == 0 and (item["user_count"] != 1
                           or item["user_messages_sha256"]
                           != _initial_user_messages_sha256(engine)):
            return False
    return True


def _positive(value: str, name: str) -> int:
    if not value.isdecimal() or int(value) <= 0:
        raise PiHealthError(f"provider {name} is not positive")
    return int(value)


def _cgroup_path(value: str) -> Path:
    candidate = PurePosixPath(value)
    if (candidate == PurePosixPath("/") or not candidate.is_absolute()
            or ".." in candidate.parts):
        raise PiHealthError("provider control group is unsafe")
    root = Path("/sys/fs/cgroup").resolve()
    try:
        path = root.joinpath(*candidate.parts[1:]).resolve(strict=True)
    except OSError as error:
        raise PiHealthError("provider control group is unavailable") from error
    if not path.is_dir() or path.is_symlink() or not path.is_relative_to(root):
        raise PiHealthError("provider control group is unavailable")
    return path


def _cgroup_file(path: Path, name: str) -> str:
    candidate = path / name
    if candidate.is_symlink() or not candidate.is_file():
        raise PiHealthError(f"provider cgroup {name} is unavailable")
    try:
        data = candidate.read_bytes()
    except OSError as error:
        raise PiHealthError(f"provider cgroup {name} is unavailable") from error
    if len(data) > 4096:
        raise PiHealthError(f"provider cgroup {name} exceeds its bound")
    try:
        return data.decode("ascii")
    except UnicodeDecodeError as error:
        raise PiHealthError(f"provider cgroup {name} is malformed") from error


def _process_start_ticks(pid: int) -> int:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
    except (OSError, UnicodeDecodeError) as error:
        raise PiHealthError("provider process start time is unavailable") from error
    closing = raw.rfind(")")
    fields = raw[closing + 2:].split() if closing >= 0 else []
    if len(fields) < 20:
        raise PiHealthError("provider process start time is unavailable")
    return _positive(fields[19], "process start time")


def _process_cgroup(pid: int) -> str:
    try:
        lines = Path(f"/proc/{pid}/cgroup").read_text().splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise PiHealthError("provider process control group is unavailable") from error
    groups: list[str] = []
    for line in lines:
        hierarchy, separator, group = line.partition("::")
        if hierarchy == "0" and separator and group.startswith("/"):
            groups.append(group)
    if len(groups) != 1:
        raise PiHealthError("provider process control group is malformed")
    return groups[0]


def provider_service_snapshot(service: str) -> dict[str, object]:
    """Read the provider identity and OOM evidence without invoking the model."""
    if not _SERVICE_UNIT.fullmatch(service) or service.startswith("-"):
        raise PiHealthError("provider service name is invalid")
    try:
        completed = subprocess.run(
            ["systemctl", "show", service, "--property=LoadState",
             "--property=ActiveState", "--property=SubState", "--property=MainPID",
             "--property=InvocationID", "--property=ActiveEnterTimestampMonotonic",
             "--property=ControlGroup"],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, timeout=3, check=True,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise PiHealthError("provider service inspection failed") from error
    properties: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        key, separator, value = line.partition("=")
        if not separator or key in properties:
            raise PiHealthError("provider service inspection is malformed")
        properties[key] = value
    expected = {"LoadState", "ActiveState", "SubState", "MainPID", "InvocationID",
                "ActiveEnterTimestampMonotonic", "ControlGroup"}
    if (set(properties) != expected or properties["LoadState"] != "loaded"
            or properties["ActiveState"] != "active"
            or properties["SubState"] != "running"):
        raise PiHealthError("provider service is not running")
    pid = _positive(properties["MainPID"], "pid")
    control_group = properties["ControlGroup"]
    if _process_cgroup(pid) != control_group:
        raise PiHealthError("provider process is outside its stated control group")
    invocation = properties["InvocationID"]
    if not _INVOCATION.fullmatch(invocation):
        raise PiHealthError("provider invocation identity is malformed")
    cgroup = _cgroup_path(control_group)
    current = _positive(_cgroup_file(cgroup, "memory.current").strip(), "cgroup memory")
    peak = _positive(_cgroup_file(cgroup, "memory.peak").strip(), "cgroup memory peak")
    if peak < current:
        raise PiHealthError("provider cgroup memory peak is malformed")
    events: dict[str, int] = {}
    for line in _cgroup_file(cgroup, "memory.events").splitlines():
        key, separator, value = line.partition(" ")
        if not separator or key in events or not value.isdecimal():
            raise PiHealthError("provider cgroup memory events are malformed")
        events[key] = int(value)
    if not {"oom", "oom_kill"} <= events.keys():
        raise PiHealthError("provider cgroup lacks OOM evidence")
    return {
        "pid": pid, "start_ticks": _process_start_ticks(pid), "invocation": invocation,
        "activation_monotonic_us": _positive(
            properties["ActiveEnterTimestampMonotonic"], "activation time"),
        "control_group": control_group, "memory_current": current, "memory_peak": peak,
        "oom": events["oom"], "oom_kill": events["oom_kill"],
    }


def _provider_identity_snapshot() -> dict[str, str]:
    """Read stable llama.cpp identity endpoints, never a completion endpoint."""
    url = os.getenv("ASSIST_MODEL_URL")
    if not isinstance(url, str) or not url.startswith(("http://", "https://")):
        raise PiHealthError("model endpoint is not configured")
    parsed = urllib.parse.urlsplit(url)
    if (parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
            or parsed.username is not None or parsed.password is not None
            or parsed.query or parsed.fragment or parsed.path.rstrip("/") != "/v1"):
        raise PiHealthError("model endpoint is not a credential-free loopback llama.cpp /v1 endpoint")
    api_key = os.getenv("ASSIST_API_KEY") or os.getenv("OPENAI_API_KEY") or "EMPTY"
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}), _NoRedirect())

    def get_json(endpoint: str) -> object:
        request = urllib.request.Request(
            endpoint, headers={"Authorization": f"Bearer {api_key}"})
        try:
            with opener.open(request, timeout=3) as response:
                body = response.read(_MAX_RECORD_BYTES + 1)
        except Exception as error:
            raise PiHealthError("model identity probe failed") from error
        if len(body) > _MAX_RECORD_BYTES:
            raise PiHealthError("model identity response exceeds its bound")
        try:
            return json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise PiHealthError("model identity response is malformed") from error

    models = get_json(f"{url.rstrip('/')}/models")
    records = models.get("data") if isinstance(models, dict) else None
    model_record = records[0] if isinstance(records, list) and records else None
    model = model_record.get("id") if isinstance(model_record, dict) else None
    model_meta = model_record.get("meta") if isinstance(model_record, dict) else None
    root_path = parsed.path.rstrip("/").removesuffix("/v1").rstrip("/")
    props = get_json(urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, f"{root_path}/props", "", "")))
    if not isinstance(model, str) or not model or not isinstance(model_meta, dict) or not isinstance(props, dict):
        raise PiHealthError("model identity response lacks required fields")
    required = ("build_info", "model_alias", "model_path", "total_slots", "chat_template",
                "chat_template_caps", "modalities", "default_generation_settings")
    if any(key not in props for key in required):
        raise PiHealthError("model build response lacks required fields")
    settings = props["default_generation_settings"]
    if (not isinstance(settings, dict) or not isinstance(settings.get("params"), dict)
            or not isinstance(settings.get("n_ctx"), int)
            or not isinstance(props["build_info"], str)
            or not isinstance(props["model_alias"], str)
            or not isinstance(props["model_path"], str)
            or not isinstance(props["total_slots"], int)
            or not isinstance(props["chat_template"], str)
            or not isinstance(props["chat_template_caps"], dict)
            or not isinstance(props["modalities"], dict)):
        raise PiHealthError("model build response has malformed fields")
    return {
        "upstream_sha256": hashlib.sha256(url.encode()).hexdigest(),
        "model": model,
        "provider_build_sha256": _sha256({
            "model": model, "model_meta": model_meta, "build_info": props["build_info"],
            "model_alias": props["model_alias"], "model_path": props["model_path"],
            "total_slots": props["total_slots"], "generation_settings": settings,
            "chat_template": props["chat_template"],
            "chat_template_caps": props["chat_template_caps"], "modalities": props["modalities"],
        }),
    }


def provider_health_snapshot(service: str) -> dict[str, object]:
    """Read provider identity and OOM evidence without issuing a completion."""
    return _provider_identity_snapshot() | {"service": provider_service_snapshot(service)}


def _service_is_valid(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != {
            "pid", "start_ticks", "invocation", "activation_monotonic_us", "control_group",
            "memory_current", "memory_peak", "oom", "oom_kill"}:
        return False
    numeric = ("pid", "start_ticks", "activation_monotonic_us", "memory_current",
               "memory_peak", "oom", "oom_kill")
    if any(type(value[key]) is not int or value[key] < 0 for key in numeric):
        return False
    if any(value[key] <= 0 for key in numeric[:5]):
        return False
    group = value["control_group"]
    return (isinstance(group, str) and group.startswith("/") and ".." not in group.split("/")
            and isinstance(value["invocation"], str)
            and _INVOCATION.fullmatch(value["invocation"]) is not None
            and value["memory_peak"] >= value["memory_current"])


def _record_admits(record: object, current: dict[str, object], now_ns: int) -> bool:
    if not isinstance(record, dict) or set(record) != {
            "schema", "id", "status", "started_at_ns", "finished_at_ns", "snapshots",
            "arms", "reason_codes"}:
        return False
    snapshots = record["snapshots"]
    arms = record["arms"]
    if (record["schema"] != 1 or record["status"] != "passed"
            or not isinstance(record["id"], str)
            or not record["id"].startswith("provider-health-")
            or type(record["started_at_ns"]) is not int
            or type(record["finished_at_ns"]) is not int
            or record["finished_at_ns"] < record["started_at_ns"]
            or record["finished_at_ns"] > now_ns
            or now_ns - record["finished_at_ns"] > HEALTH_TTL_SECONDS * 1_000_000_000
            or record["reason_codes"] != []
            or not isinstance(snapshots, dict) or set(snapshots) != {"before", "after_pi", "final"}
            or not isinstance(arms, dict) or set(arms) != {"pi", "deepagents"}):
        return False
    snapshots_list: list[dict[str, object]] = []
    services: list[dict[str, object]] = []
    for snapshot in snapshots.values():
        if not isinstance(snapshot, dict) or set(snapshot) != {
                "upstream_sha256", "model", "provider_build_sha256", "service"}:
            return False
        if (not isinstance(snapshot["model"], str) or not snapshot["model"]
                or any(not isinstance(snapshot[key], str)
                       or _SHA256.fullmatch(snapshot[key]) is None
                       for key in ("upstream_sha256", "provider_build_sha256"))):
            return False
        service = snapshot["service"]
        if not _service_is_valid(service):
            return False
        assert isinstance(service, dict)
        if service["oom"] != 0 or service["oom_kill"] != 0:
            return False
        snapshots_list.append(snapshot)
        services.append(service)
    if any(any(snapshot[key] != snapshots_list[0][key]
                   for key in ("upstream_sha256", "model", "provider_build_sha256"))
           for snapshot in snapshots_list[1:]):
        return False
    identity_fields = ("pid", "start_ticks", "invocation", "activation_monotonic_us", "control_group")
    if any(any(service[key] != services[0][key] for key in identity_fields) for service in services[1:]):
        return False
    if not all(_arm_is_valid(arms[key], key, snapshots_list[0]["model"])
                   for key in arms):
        return False
    if (not isinstance(current, dict) or set(current) != {
            "upstream_sha256", "model", "provider_build_sha256", "service"}
            or not _service_is_valid(current["service"])
            or current["service"]["oom"] != 0 or current["service"]["oom_kill"] != 0):
        return False
    return (all(current[key] == snapshots_list[0][key]
                for key in ("upstream_sha256", "model", "provider_build_sha256"))
            and all(current["service"][key] == services[0][key]
                    for key in identity_fields))


def _arm_is_valid(value: object, engine: str, model: object) -> bool:
    if not isinstance(value, dict) or set(value) != {
            "engine", "passed", "checks", "provider", "tools", "driver",
            "runner_release_sha256"}:
        return False
    checks = value["checks"]
    provider = value["provider"]
    tools = value["tools"]
    if (value["engine"] != engine or value["passed"] is not True
            or not isinstance(checks, dict) or set(checks) != _ARM_CHECKS
            or any(item is not True for item in checks.values())
            or not isinstance(provider, dict) or set(provider) != {
                "chat_attempts", "successful_chat_responses", "chat_failures",
                "bound_rejections", "request_census_sha256", "request_census"}
            or [provider[key] for key in ("chat_attempts", "successful_chat_responses",
                                          "chat_failures", "bound_rejections")] != [1, 1, 0, 0]
            or not _census_is_valid(provider["request_census"], engine, model)
            or not isinstance(provider["request_census_sha256"], str)
            or _SHA256.fullmatch(provider["request_census_sha256"]) is None
            or provider["request_census_sha256"] != _sha256(provider["request_census"])
            or not isinstance(tools, dict) or set(tools) != {
                "commands", "pre_model_commands", "post_model_commands", "rejections", "error"}
            or any(type(tools[key]) is not int or tools[key] < 0
                   for key in ("commands", "pre_model_commands", "post_model_commands", "rejections"))
            or tools["commands"] != tools["pre_model_commands"] + tools["post_model_commands"]
            or (tools["pre_model_commands"] != 0 if engine == "pi"
                else tools["pre_model_commands"] > 4)
            or tools["post_model_commands"] != 0 or tools["rejections"] != 0
            or tools["error"] is not False or value["driver"] != {"status": "completed"}
            or not isinstance(value["runner_release_sha256"], str)
            or _SHA256.fullmatch(value["runner_release_sha256"]) is None):
        return False
    return True


def _safe_record(path: Path) -> object:
    try:
        descriptor = os.open(
            path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
        )
    except OSError as error:
        raise PiHealthError("Pi health record is unavailable") from error
    try:
        metadata = os.fstat(descriptor)
        if (not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid()
                or metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
                or metadata.st_size > _MAX_RECORD_BYTES):
            raise PiHealthError("Pi health record is unsafe")
        chunks: list[bytes] = []
        remaining = _MAX_RECORD_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        if len(data) > _MAX_RECORD_BYTES:
            raise PiHealthError("Pi health record exceeds its bound")
        return json.loads(data.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PiHealthError("Pi health record is unreadable") from error
    finally:
        os.close(descriptor)


@contextmanager
def _probe_lock(root: Path):
    """Hold P1's shared lock or deny while a probe changes its evidence."""
    try:
        descriptor = os.open(
            root / "EXECUTION.lock",
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
        )
    except OSError as error:
        raise PiHealthError("Pi health lock is unavailable") from error
    try:
        metadata = os.fstat(descriptor)
        if (not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid()
                or metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH)):
            raise PiHealthError("Pi health lock is unsafe")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except OSError as error:
            raise PiHealthError("Pi health probe is active") from error
        try:
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def preview_health_admits(
    root: Path, service: str, *, now_ns: int | None = None,
    snapshot: Callable[[str], dict[str, object]] = provider_health_snapshot,
) -> bool:
    """Return whether one fresh, stable P1 record safely admits a Pi turn."""
    try:
        metadata = root.lstat()
        if (not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode)
                or metadata.st_uid != os.getuid()
                or metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH)):
            return False
        with _probe_lock(root):
            if (root / "ACTIVE.json").exists():
                return False
            current = snapshot(service)
            return _record_admits(_safe_record(root / "latest.json"), current,
                                  time.time_ns() if now_ns is None else now_ns)
    except (Exception,):
        return False


def configured_preview_health_admits() -> bool:
    """Read operator configuration; absent configuration deliberately closes Pi."""
    directory = os.getenv("ASSIST_PI_HEALTH_DIR")
    service = os.getenv("ASSIST_PI_PROVIDER_SERVICE")
    try:
        return bool(directory and service and preview_health_admits(Path(directory), service))
    except (OSError, ValueError):
        return False
