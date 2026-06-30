"""The /schedules management view — render, delete, and event-loop independence."""
import os
import threading
import time

import pytest
from fastapi.testclient import TestClient

from assist.schedule.model import Cadence, Schedule
from assist.thread_queue import THREAD_QUEUE
from manage import web
from manage.web import state


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(state.SCHEDULE_STORE, "_root", str(tmp_path))
    os.makedirs(tmp_path / "t1", exist_ok=True)
    state.SCHEDULE_STORE.add(Schedule(
        id="abc123", thread_id="t1", prompt="review my inbox",
        cadence=Cadence(hour=7, minute=0), tz="America/Los_Angeles",
        next_fire_at="2026-06-15T14:00:00+00:00"))
    return TestClient(web.app)   # no `with` -> lifespan (and the scheduler) don't start


def test_schedules_page_renders_link_label_delete(client):
    html = client.get("/schedules").text
    assert '/thread/t1"' in html                      # thread link
    assert "every day at 7:00 AM" in html             # engine-derived label
    assert "review my inbox" in html                  # description/prompt
    assert "/schedules/t1/abc123/delete" in html      # delete action


def test_delete_removes_and_redirects(client):
    r = client.post("/schedules/t1/abc123/delete", follow_redirects=False)
    assert r.status_code == 303
    assert state.SCHEDULE_STORE.for_thread("t1") == []


def test_schedules_route_independent_of_thread_queue(client):
    """The view must not touch THREAD_QUEUE — holding a turn slot can't block it."""
    held = threading.Event()
    release = threading.Event()

    def hold():
        with THREAD_QUEUE.acquire("other-thread"):
            held.set()
            release.wait(timeout=5)

    t = threading.Thread(target=hold)
    t.start()
    assert held.wait(timeout=5)
    start = time.monotonic()
    assert client.get("/schedules").status_code == 200      # served while the queue is held
    assert time.monotonic() - start < 2.0
    release.set()
    t.join()
