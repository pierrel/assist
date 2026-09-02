from __future__ import annotations

import asyncio
import json
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import monotonic, sleep

import pytest
from fastapi.testclient import TestClient
from langchain.messages import AIMessage, HumanMessage

from assist.safe_markdown import render_markdown
from assist.visible_conversation import (
    CONTINUATION_RIDER,
    INTERJECTION_FRAME,
    INTERJECTION_GUIDE,
    VisibleRecord,
    select_completed_turns,
    visible_records,
)
from edd.live_capture import CaptureStorageFull, CaptureStore, CaptureWorker


def _records() -> tuple[VisibleRecord, ...]:
    return (
        VisibleRecord("r0001", 1, "user", "Find one fact", "user", True),
        VisibleRecord("r0002", 2, "assistant", "Here is the fact.", "assistant", True),
    )


def test_projection_selects_last_completed_turns_and_excludes_host_frames():
    records = visible_records([
        HumanMessage(content="first"), AIMessage(content="one"),
        HumanMessage(content=CONTINUATION_RIDER + "later"),
        HumanMessage(content="second"), AIMessage(content="two"),
        HumanMessage(content="unfinished"),
    ])

    selected = select_completed_turns(records)

    assert [record.text for record in selected] == ["first", "one", "second", "two"]
    assert all(record.capture_eligible for record in selected)


def test_projection_keeps_a_mid_turn_interjection_with_its_original_prompt():
    records = visible_records([
        HumanMessage(content="Find a nearby cafe"),
        HumanMessage(content=INTERJECTION_FRAME + "Make it quiet." + INTERJECTION_GUIDE),
        AIMessage(content="Here is a quiet cafe nearby."),
    ])

    selected = select_completed_turns(records)

    assert [record.text for record in selected] == [
        "Find a nearby cafe", "Make it quiet.", "Here is a quiet cafe nearby.",
    ]


def test_store_keeps_immutable_transcript_and_binds_reads_to_thread(tmp_path: Path):
    threads = tmp_path / "threads"
    threads.mkdir()
    store = CaptureStore(tmp_path / "captures", threads_root=threads)

    created = store.create(
        thread_id="thread-a", reason="It found the requested fact.",
        scope="last_3", records=_records(), source_revision="abc",
    )
    capture_id = created["request"]["capture_id"]
    before = (tmp_path / "captures" / capture_id / "transcript.json").read_bytes()
    store.update_result("thread-a", capture_id, {"status": "failed", "error": "test"})

    assert (tmp_path / "captures" / capture_id / "transcript.json").read_bytes() == before
    assert store.get_for_thread("thread-a", capture_id)["request"]["reason"].startswith("It found")
    with pytest.raises(FileNotFoundError):
        store.get_for_thread("thread-b", capture_id)
    assert (tmp_path / "captures" / capture_id / "request.json").stat().st_mode & 0o077 == 0
    newest = store.create(thread_id="thread-a", reason="A later capture.",
                          scope="last_3", records=_records())
    assert store.list_for_threads()["thread-a"][0]["capture_id"] == newest["request"]["capture_id"]


def test_dismissals_choose_the_newest_visible_capture(tmp_path: Path):
    from manage.web.state import CaptureDismissals

    threads = tmp_path / "threads"
    threads.mkdir()
    store = CaptureStore(tmp_path / "captures", threads_root=threads)
    older = store.create(thread_id="thread-a", reason="Older capture.",
                         scope="last_3", records=_records())
    newer = store.create(thread_id="thread-a", reason="Newer capture.",
                         scope="last_3", records=_records())
    dismissals = CaptureDismissals()
    dismissals.dismiss("thread-a", newer["request"]["capture_id"])

    assert dismissals.latest_visible(store.list_for_threads())["thread-a"]["capture_id"] \
        == older["request"]["capture_id"]


def test_store_skips_malformed_index_entries_during_pending_recovery(tmp_path: Path):
    store = CaptureStore(tmp_path / "captures", threads_root=tmp_path / "threads")
    valid_id = "a" * 20
    (store.root / "index.json").write_text(json.dumps({
        "threads": {"thread-a": [None, {"status": "queued"},
                                 {"status": "queued", "capture_id": 3},
                                 {"status": "queued", "capture_id": "../bad"},
                                 {"status": "queued", "capture_id": valid_id}],
                    "thread-b": "not-a-list"},
    }))

    assert store.pending() == [("thread-a", valid_id)]


def test_store_rejects_a_symlinked_capture_file(tmp_path: Path):
    threads = tmp_path / "threads"
    threads.mkdir()
    store = CaptureStore(tmp_path / "captures", threads_root=threads)
    created = store.create(thread_id="thread-a", reason="Reason with an outcome.",
                           scope="last_3", records=_records())
    capture_id = created["request"]["capture_id"]
    path = tmp_path / "captures" / capture_id / "result.json"
    path.unlink()
    path.symlink_to(tmp_path / "outside.json")

    with pytest.raises(ValueError, match="regular file"):
        store.get_for_thread("thread-a", capture_id)


def test_store_quota_does_not_follow_symlinks(tmp_path: Path):
    threads = tmp_path / "threads"
    threads.mkdir()
    store = CaptureStore(tmp_path / "captures", threads_root=threads)
    outside = tmp_path / "outside"
    outside.write_bytes(b"x" * 10_000)
    before = store._store_bytes()
    (tmp_path / "captures" / "outside-link").symlink_to(outside)

    assert store._store_bytes() == before


def test_store_refuses_capture_root_inside_thread_root(tmp_path: Path):
    threads = tmp_path / "threads"
    threads.mkdir()
    with pytest.raises(ValueError, match="outside"):
        CaptureStore(threads / "captures", threads_root=threads)


def test_store_refuses_capture_root_containing_thread_root(tmp_path: Path):
    captures = tmp_path / "captures"
    threads = captures / "threads"
    threads.mkdir(parents=True)

    with pytest.raises(ValueError, match="outside"):
        CaptureStore(captures, threads_root=threads)


def test_rejected_capture_cannot_bypass_the_store_quota(tmp_path: Path, monkeypatch):
    import edd.live_capture as captures

    threads = tmp_path / "threads"
    threads.mkdir()
    store = CaptureStore(tmp_path / "captures", threads_root=threads)
    before = {path.name for path in (tmp_path / "captures").iterdir()}
    monkeypatch.setattr(captures, "MAX_STORE_BYTES", 1)

    with pytest.raises(CaptureStorageFull):
        store.create(thread_id="thread-a", reason="Reason with an outcome.",
                     scope="entire", records=_records())

    assert {path.name for path in (tmp_path / "captures").iterdir()} == before


def test_store_full_capture_is_failed_not_a_shorter_scope_request(tmp_path: Path, monkeypatch):
    import edd.live_capture as captures

    threads = tmp_path / "threads"
    threads.mkdir()
    store = CaptureStore(tmp_path / "captures", threads_root=threads)
    monkeypatch.setattr(
        captures, "MAX_STORE_BYTES",
        store._store_bytes() + captures._REJECTED_CAPTURE_RESERVATION_BYTES + 1,
    )

    capture = store.create(thread_id="thread-a", reason="Reason with an outcome.",
                           scope="entire", records=_records())

    assert capture["result"]["status"] == "failed"
    assert capture["result"]["error"] == "private capture storage is full"


def test_result_update_cannot_grow_past_the_store_quota(tmp_path: Path, monkeypatch):
    import edd.live_capture as captures

    threads = tmp_path / "threads"
    threads.mkdir()
    store = CaptureStore(tmp_path / "captures", threads_root=threads)
    capture = store.create(thread_id="thread-a", reason="Reason with an outcome.",
                           scope="last_3", records=_records())
    capture_id = capture["request"]["capture_id"]
    original = store.get_for_thread("thread-a", capture_id)["result"]
    monkeypatch.setattr(captures, "MAX_STORE_BYTES", store._store_bytes())

    with pytest.raises(CaptureStorageFull):
        store.update_result("thread-a", capture_id, {"observation": {"evidence": "x" * 100}})

    assert store.get_for_thread("thread-a", capture_id)["result"] == original


class _Response:
    def __init__(self, content: str):
        self.content = content
        self.response_metadata = {"model_name": "fake-local", "finish_reason": "stop"}


class _Model:
    def bind(self, **kwargs):
        self.schema_name = kwargs["response_format"]["json_schema"]["name"]
        return self

    def invoke(self, messages):
        if self.schema_name == "assist_capture_criteria":
            return _Response(json.dumps({
                "status": "criteria", "requested": [{"description": "Provide the requested fact"}],
                "forbidden": [], "clarification": None,
            }))
        return _Response(json.dumps({
            "overall": "pass",
            "requested": [{"id": "requested-1", "grade": "satisfied", "evidence_ids": ["r0001", "r0002"]}],
            "forbidden": [], "contradictions": [], "material_unrelated_evidence_ids": [],
            "unsafe_extra_evidence_ids": [], "rationale": "The response provides the fact.",
            "rationale_evidence_ids": ["r0002"], "confidence": "high",
        }))


def test_worker_interprets_then_judges_one_private_capture(tmp_path: Path):
    threads = tmp_path / "threads"
    threads.mkdir()
    store = CaptureStore(tmp_path / "captures", threads_root=threads)
    capture = store.create(thread_id="thread-a", reason="It found the requested fact.",
                           scope="last_3", records=_records())
    capture_id = capture["request"]["capture_id"]
    worker = CaptureWorker(store, model_factory=_Model)
    worker.start()
    worker.submit("thread-a", capture_id)
    deadline = monotonic() + 3
    while monotonic() < deadline:
        result = store.get_for_thread("thread-a", capture_id)["result"]
        if result["status"] == "pass":
            break
        sleep(0.01)
    worker.stop()

    result = store.get_for_thread("thread-a", capture_id)["result"]
    assert result["status"] == "pass"
    assert result["interpreter"]["model"] == "fake-local"
    assert result["judge"]["model"] == "fake-local"


def test_only_one_worker_can_own_a_capture_store(tmp_path: Path):
    threads = tmp_path / "threads"
    threads.mkdir()
    store = CaptureStore(tmp_path / "captures", threads_root=threads)
    first = CaptureWorker(store, model_factory=_Model)
    second = CaptureWorker(store, model_factory=_Model)
    first.start()
    try:
        with pytest.raises(RuntimeError, match="another capture worker"):
            second.start()
    finally:
        first.stop()


def test_worker_submit_deduplicates_concurrent_notifications(tmp_path: Path):
    threads = tmp_path / "threads"
    threads.mkdir()
    store = CaptureStore(tmp_path / "captures", threads_root=threads)
    worker = CaptureWorker(store, model_factory=_Model)

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda _: worker.submit("thread-a", "capture-a"), range(32)))

    assert worker._queue.qsize() == 1


def test_worker_start_releases_ownership_when_recovery_scan_fails(tmp_path: Path, monkeypatch):
    threads = tmp_path / "threads"
    threads.mkdir()
    store = CaptureStore(tmp_path / "captures", threads_root=threads)
    worker = CaptureWorker(store, model_factory=_Model)
    monkeypatch.setattr(store, "pending", lambda: (_ for _ in ()).throw(ValueError("bad index")))

    with pytest.raises(ValueError, match="bad index"):
        worker.start()

    assert worker._thread is None
    assert worker._lock_file is None


def test_worker_refuses_a_symlinked_lock_file(tmp_path: Path):
    threads = tmp_path / "threads"
    threads.mkdir()
    store = CaptureStore(tmp_path / "captures", threads_root=threads)
    outside = tmp_path / "outside"
    outside.write_text("leave me alone")
    (store.root / "worker.lock").symlink_to(outside)

    with pytest.raises(ValueError, match="owner-only regular file"):
        CaptureWorker(store, model_factory=_Model).start()

    assert outside.read_text() == "leave me alone"


def test_worker_stop_interrupts_queue_wait_before_starting_a_model_call(tmp_path: Path, monkeypatch):
    import edd.live_capture as captures

    class BusyQueue:
        @contextmanager
        def acquire(self, *_args, **_kwargs):
            raise captures.QueueWaitTimeout("busy")
            yield

        def pop_hold(self, _queue_id):
            pass

    threads = tmp_path / "threads"
    threads.mkdir()
    store = CaptureStore(tmp_path / "captures", threads_root=threads)
    capture = store.create(thread_id="thread-a", reason="Reason with an outcome.",
                           scope="last_3", records=_records())
    worker = CaptureWorker(store, model_factory=_Model)
    monkeypatch.setattr(captures, "THREAD_QUEUE", BusyQueue())
    monkeypatch.setattr(captures, "CAPTURE_QUEUE_POLL_TIMEOUT_S", 0.01)
    worker.start()
    worker.submit("thread-a", capture["request"]["capture_id"])
    sleep(0.05)

    worker.stop()

    assert worker._thread is None


def test_safe_markdown_rejects_script_events_and_javascript_urls():
    output = render_markdown(
        '<script>steal()</script><a onclick="steal()" href=" javascript:steal()">x</a>'
        '<a href="java\nscript:steal()">y</a>'
    )

    assert "<script" not in output
    assert "onclick" not in output
    assert "javascript:" not in output
    assert "href=" not in output


def test_store_rejects_excess_records_before_building_a_transcript(tmp_path: Path, monkeypatch):
    import edd.live_capture as captures

    threads = tmp_path / "threads"
    threads.mkdir()
    store = CaptureStore(tmp_path / "captures", threads_root=threads)
    monkeypatch.setattr(captures, "_record_json", lambda _record: pytest.fail("should not serialize"))

    capture = store.create(thread_id="thread-a", reason="Reason with an outcome.",
                           scope="last_3", records=_records() * (captures.MAX_RECORDS + 1))

    assert capture["result"]["status"] == "needs_shorter_scope"
    assert capture["result"]["error"] == "too many visible records"


def test_shorter_scope_card_keeps_hostile_reason_out_of_an_event_handler(tmp_path: Path, monkeypatch):
    from manage.web.threads import _capture_card_html
    import edd.live_capture as captures

    threads = tmp_path / "threads"
    threads.mkdir()
    store = CaptureStore(tmp_path / "captures", threads_root=threads)
    monkeypatch.setattr(captures, "MAX_SNAPSHOT_BYTES", 1)
    reason = 'x" onclick="steal()'
    capture = store.create(thread_id="thread-a", reason=reason,
                           scope="entire", records=_records())

    output = _capture_card_html(capture)

    assert "captureLastThree(this)" in output
    assert "Hide for this session" in output
    assert 'data-capture-reason="x&quot; onclick=&quot;steal()"' in output
    assert "onclick=\"steal()" not in output
    assert "turns 1–1" in output


def test_capture_route_snapshots_raw_messages_and_scopes_fragment_to_thread(tmp_path: Path, monkeypatch):
    from manage.web import threads as web_threads

    threads_root = tmp_path / "threads"
    threads_root.mkdir()
    store = CaptureStore(tmp_path / "captures", threads_root=threads_root)

    class FakeThread:
        def get_raw_messages(self):
            return [
                HumanMessage(content="question"),
                HumanMessage(content=INTERJECTION_FRAME + "also this" + INTERJECTION_GUIDE),
                AIMessage(content="answer"),
            ]

    class FakeManager:
        def get(self, tid, **kwargs):
            assert tid == "t1"
            return FakeThread()

    class FakeWorker:
        def __init__(self): self.submitted = []
        def submit(self, tid, capture_id): self.submitted.append((tid, capture_id))

    worker = FakeWorker()
    monkeypatch.setattr(web_threads, "MANAGER", FakeManager())
    monkeypatch.setattr(web_threads, "CAPTURE_STORE", store)
    monkeypatch.setattr(web_threads, "CAPTURE_WORKER", worker)
    monkeypatch.setattr(web_threads, "_require_deep_thread", lambda tid: None)
    monkeypatch.setattr(web_threads, "_get_status", lambda tid: {"stage": "ready"})
    client = TestClient(web_threads.app)

    response = client.post("/thread/t1/capture", data={
        "reason": "It answered the question.", "scope": "last_3",
        "csrf_token": web_threads.CAPTURE_CSRF,
    })

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    capture_id = response.json()["capture_id"]
    assert worker.submitted == [("t1", capture_id)]
    capture = store.get_for_thread("t1", capture_id)
    assert capture["request"]["turn_range"] == [1, 1]
    assert [record["text"] for record in capture["transcript"]["records"]] == [
        "question", "also this", "answer",
    ]
    fragment = client.get(f"/thread/t1/capture/{capture_id}")
    assert fragment.status_code == 200
    assert fragment.headers["cache-control"] == "no-store"
    assert "It answered the question." in fragment.text
    assert client.get(f"/thread/other/capture/{capture_id}").status_code == 404
    assert client.post("/thread/t1/capture", data={
        "reason": "no", "scope": "last_3", "csrf_token": "wrong",
    }).status_code == 403


def test_dismiss_capture_hides_only_its_process_local_card(tmp_path: Path, monkeypatch):
    from manage.web import threads as web_threads
    from manage.web.state import CaptureDismissals

    threads_root = tmp_path / "threads"
    threads_root.mkdir()
    store = CaptureStore(tmp_path / "captures", threads_root=threads_root)
    capture = store.create(thread_id="t1", reason="Keep the evidence.",
                           scope="last_3", records=_records())
    capture_id = capture["request"]["capture_id"]
    files = {
        path.name: path.read_bytes()
        for path in (store.root / capture_id).iterdir()
    }
    index_before = (store.root / "index.json").read_bytes()
    monkeypatch.setattr(web_threads, "CAPTURE_STORE", store)
    monkeypatch.setattr(web_threads, "CAPTURE_DISMISSALS", CaptureDismissals())
    client = TestClient(web_threads.app)

    assert client.get(f"/thread/t1/capture/{capture_id}").status_code == 200
    assert client.post(f"/thread/t1/capture/{capture_id}/dismiss", data={
        "csrf_token": "wrong",
    }).status_code == 403
    assert client.post(f"/thread/other/capture/{capture_id}/dismiss", data={
        "csrf_token": web_threads.CAPTURE_CSRF,
    }).status_code == 404

    response = client.post(f"/thread/t1/capture/{capture_id}/dismiss", data={
        "csrf_token": web_threads.CAPTURE_CSRF,
    })

    assert response.status_code == 204
    assert response.headers["cache-control"] == "no-store"
    assert client.get(f"/thread/t1/capture/{capture_id}").status_code == 404
    assert {path.name: path.read_bytes() for path in (store.root / capture_id).iterdir()} == files
    assert (store.root / "index.json").read_bytes() == index_before
    assert store.update_result("t1", capture_id, {"status": "failed", "error": "later"})[
        "result"]["error"] == "later"


def test_dismissed_capture_does_not_render_after_a_thread_refresh(tmp_path: Path, monkeypatch):
    from manage import web
    from manage.web import state
    from manage.web import threads as web_threads
    from manage.web.state import CaptureDismissals

    tid = "t1"
    threads_root = tmp_path / "threads"
    (threads_root / tid).mkdir(parents=True)
    monkeypatch.setattr(web.MANAGER, "root_dir", str(threads_root))
    state.DESCRIPTION_CACHE[tid] = "Dismissal test"
    state._set_status(tid, "initializing")
    store = CaptureStore(tmp_path / "captures", threads_root=threads_root)
    capture = store.create(thread_id=tid, reason="Keep it stored.",
                           scope="last_3", records=_records())
    dismissals = CaptureDismissals()
    dismissals.dismiss(tid, capture["request"]["capture_id"])
    monkeypatch.setattr(web_threads, "CAPTURE_STORE", store)
    monkeypatch.setattr(web_threads, "CAPTURE_DISMISSALS", dismissals)

    page = asyncio.run(web_threads.get_thread(tid))

    assert capture["request"]["capture_id"] not in page
    assert "capture:" not in web_threads.render_index()


def test_pending_capture_fragment_retries_after_a_transient_failure():
    from manage.web.threads import render_thread

    page = render_thread("t1", None)

    assert ".catch(function(error){" in page
    assert "if(capturePolls[id] || hiddenCaptures[id]) return;" in page
    assert "if(hiddenCaptures[id]) return;" in page
    assert "capturePollFailures[id] < 3" in page
    assert "1200 * capturePollFailures[id]" in page
    assert "error.status=r.status" in page
    assert "if(error.status === 404) { if(card) card.remove(); return; }" in page
    assert "if(card && card.getAttribute('data-capture-pending') === '1' && capturePollFailures[id] < 3)" in page
    assert "hiddenCaptures[id]=true;" in page
    assert "var fresh=document.getElementById('capture-' + id); if(fresh) fresh.remove();" in page


def test_capture_backlog_becomes_a_visible_terminal_failure(tmp_path: Path, monkeypatch):
    from manage.web import threads as web_threads

    threads_root = tmp_path / "threads"
    threads_root.mkdir()
    store = CaptureStore(tmp_path / "captures", threads_root=threads_root)

    class FakeThread:
        def get_raw_messages(self):
            return [HumanMessage(content="question"), AIMessage(content="answer")]

    class FakeManager:
        def get(self, tid, **kwargs): return FakeThread()

    class FullWorker:
        def submit(self, tid, capture_id): raise RuntimeError("full")

    monkeypatch.setattr(web_threads, "MANAGER", FakeManager())
    monkeypatch.setattr(web_threads, "CAPTURE_STORE", store)
    monkeypatch.setattr(web_threads, "CAPTURE_WORKER", FullWorker())
    monkeypatch.setattr(web_threads, "_require_deep_thread", lambda tid: None)
    monkeypatch.setattr(web_threads, "_get_status", lambda tid: {"stage": "ready"})

    capture = web_threads._create_live_capture("t1", "It answered the question.", "last_3")

    assert capture["result"]["status"] == "failed"
    assert capture["result"]["error"] == "capture queue is full"
