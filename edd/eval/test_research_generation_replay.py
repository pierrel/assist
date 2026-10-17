"""Slow seeded next-response capability replay, separate from natural acceptance.

Opt in with ASSIST_RUN_RESEARCH_REPLAY=1 through ``resource run llm``. Choose
ASSIST_RESEARCH_REPLAY_PROFILE (baseline/capped/t1/official/presence) and SEED
(ASSIST_RESEARCH_REPLAY_SEED, default 0). The baseline's 32768-token/1200-second
observation ceilings are harness bounds, not historical production defaults.
No generated tool is executed. Native stream events and provenance stay private.
"""
import ast
from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time

import httpx
from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import AIMessage, convert_to_messages
import pytest

from assist.middleware.research_generation import ResearchGenerationMiddleware
from assist.model_manager import _server_root, select_assistant_model
from .test_research_generation import _count_sourced_recommendations


_FIXTURES = Path(__file__).parents[1] / "fixtures" / "research_generation"
_PROFILES = {
    "baseline": (0.1, {}),
    "capped": (0.1, {}),
    "t1": (1.0, {}),
    "official": (1.0, {"min_p": 0}),
    "presence": (1.0, {"min_p": 0, "presence_penalty": 0.3}),
}


def _context_sha(payload):
    context = {key: payload[key] for key in ("messages", "tools")}
    return hashlib.sha256(json.dumps(context, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def _sources(messages):
    calls = {}
    sources = []
    failed = set()
    for message in messages:
        for call in message.get("tool_calls", []):
            calls[call["id"]] = call["function"]
        if message["role"] != "tool":
            continue
        call = calls[message["tool_call_id"]]
        if call["name"] == "search_internet":
            sources.extend(ast.literal_eval(message["content"]))
        elif call["name"] == "read_url" and "error" in message["content"].lower():
            failed.add(json.loads(call["arguments"])["url"])
    return sources, failed


def _grounded_answer(text, sources):
    # Known book identities and supported formats in this captured evidence.
    identities = [
        ("Gillian Cross", "The Odyssey", r"illustrat|narrative|prose|retelling"),
        ("Marcia Williams", "The Iliad and the Odyssey", r"comic|cartoon|visual"),
        ("Russell Punter", "The Odyssey", r"comic|graphic|illustrat"),
    ]
    rows = []
    for author, title, pattern in identities:
        urls = [row["url"] for row in sources
                if author.lower() in (row["title"] + " " + row["content"]).lower()
                and title.lower() in (row["title"] + " " + row["content"]).lower()]
        if urls:
            url = next((url for url in urls if url in text), urls[0])
            rows.append((author, title, url, pattern))
        text = re.sub(re.escape(author), author, text, flags=re.I)
        text = re.sub(re.escape(title), title, text, flags=re.I)
    return _count_sourced_recommendations(text, rows) >= 1


def _useful(message, sources, failed):
    if (message.response_metadata.get("finish_reason") not in {"stop", "tool_calls"}
            or message.invalid_tool_calls):
        return False
    urls = {row["url"] for row in sources} - failed
    if message.tool_calls:
        grounded = False
        for call in message.tool_calls:
            args = call["args"]
            if not call.get("id"):
                return False
            if (call["name"] == "read_url" and set(args) == {"url"}
                    and isinstance(args["url"], str) and args["url"] in urls):
                grounded = True
                continue
            if (call["name"] == "write_file" and set(args) == {"file_path", "content"}
                    and isinstance(args["file_path"], str) and args["file_path"].startswith("/")
                    and isinstance(args["content"], str) and _grounded_answer(args["content"], sources)):
                grounded = True
                continue
            if (call["name"] == "search_internet" and "query" in args
                    and set(args) <= {"query", "max_results"}
                    and isinstance(args["query"], str) and args["query"].strip()
                    and ("max_results" not in args or type(args["max_results"]) is int)):
                continue
            return False
        return grounded
    return _grounded_answer(message.text, sources)


def _stream_response(request, fixture, artifacts, metrics, client):
    payload = request.model._get_request_payload(
        request.messages, tools=fixture["tools"], seed=metrics["seed"], stream=True,
        stream_options={"include_usage": True}, **request.model_settings)
    assert payload["messages"] == fixture["messages"], "Replay changed captured messages"
    assert payload["tools"] == fixture["tools"], "Replay changed captured tool definitions"
    chunks = 0
    reasoning = []
    content = []
    calls = {}
    finish = None
    usage = None
    started = time.monotonic()

    try:
        props = client.get(_server_root(str(client.base_url)) + "/props", cast_to=dict[str, object])
        (artifacts / "server-props.json").write_text(json.dumps(props, indent=2))
        (artifacts / "models.json").write_text(client.models.list().model_dump_json())
        with (artifacts / "stream.jsonl").open("w") as raw:
            with client.chat.completions.create(**payload) as stream:
                for chunk in stream:
                    event = chunk.model_dump(mode="json")
                    raw.write(json.dumps({"seconds": time.monotonic() - started,
                                          "event": event}) + "\n")
                    raw.flush()
                    if time.monotonic() - started > 1200:
                        raise TimeoutError("Replay exceeded 1200-second observation ceiling")
                    usage = event.get("usage") or usage
                    for choice in event["choices"]:
                        delta = choice["delta"]
                        text = delta.get("reasoning_content") or ""
                        if text:
                            reasoning.append(text)
                        content.append(delta.get("content") or "")
                        for part in delta.get("tool_calls") or []:
                            call = calls.setdefault(part["index"], {
                                "id": "", "type": "function",
                                "function": {"name": "", "arguments": ""}})
                            call["id"] += part.get("id") or ""
                            for key in ("name", "arguments"):
                                call["function"][key] += (part.get("function") or {}).get(key) or ""
                        finish = choice.get("finish_reason") or finish
                    chunks += 1
    finally:
        repeats = Counter(line.strip() for line in "".join(reasoning).splitlines()
                          if len(line.strip()) >= 32)
        metrics.update(seconds=round(time.monotonic() - started, 3), finish_reason=finish,
                       usage=usage, chunks=chunks, reasoning_delta_events=len(reasoning),
                       reasoning_chars=sum(map(len, reasoning)),
                       max_repeated_long_line=max(repeats.values(), default=0))
        (artifacts / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return AIMessage(content="".join(content), additional_kwargs={"tool_calls": list(calls.values())},
                     response_metadata={"finish_reason": finish})


@pytest.mark.skipif(os.environ.get("ASSIST_RUN_RESEARCH_REPLAY") != "1",
                    reason="Slow seeded capability replay; explicit production-priority admission required")
def test_seeded_replay_makes_source_backed_progress():
    fixture = json.loads((_FIXTURES / "request_748.json").read_text())
    provenance = json.loads((_FIXTURES / "provenance.json").read_text())
    assert _context_sha(fixture) == provenance["sanitized_context_sha256"]
    profile = os.environ.get("ASSIST_RESEARCH_REPLAY_PROFILE", "capped")
    temperature, extra = _PROFILES[profile]
    seed = int(os.environ.get("ASSIST_RESEARCH_REPLAY_SEED", "0"))
    artifact_root = os.environ.get("ASSIST_RESEARCH_REPLAY_ARTIFACT_DIR")
    if artifact_root:
        Path(artifact_root).mkdir(parents=True, exist_ok=True, mode=0o700)
    artifacts = Path(tempfile.mkdtemp(prefix=profile + "-", dir=artifact_root))
    metrics = {"profile": profile, "seed": seed, "temperature": temperature,
               "extra_sampling": extra, "context_sha256": _context_sha(fixture),
               "artifacts": str(artifacts), "observation_tokens": 32768,
               "observation_seconds": 1200}
    model = select_assistant_model(temperature)
    assert Path(model.model_name).name == fixture["model"], "Replay requires the captured model"
    request = ModelRequest(model=model, messages=convert_to_messages(deepcopy(fixture["messages"])),
                           state={}, model_settings={"max_tokens": 32768, "extra_body": extra})
    sources, failed = _sources(fixture["messages"])
    try:
        def record_wire(http_request):
            if http_request.method == "POST" and http_request.url.path.endswith("/chat/completions"):
                (artifacts / "request.json").write_text(http_request.content.decode())

        with httpx.Client(event_hooks={"request": [record_wire]}) as http_client:
            client = model.root_client.with_options(http_client=http_client, max_retries=0)
            handler = lambda bounded: _stream_response(bounded, fixture, artifacts, metrics, client)
            response = (handler(request) if profile == "baseline" else
                        ResearchGenerationMiddleware().wrap_model_call(request, handler))
        metrics["useful"] = _useful(response, sources, failed)
        metrics["tool_names"] = [call["name"] for call in response.tool_calls]
        (artifacts / "response.json").write_text(response.model_dump_json())
        assert metrics["useful"], "No valid source-derived next action or grounded book answer"
    finally:
        (artifacts / "metrics.json").write_text(json.dumps(metrics, indent=2))
        print("REPLAY_RESULT " + json.dumps(metrics), flush=True)
