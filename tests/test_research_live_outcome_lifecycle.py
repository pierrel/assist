"""A saved draft is not a completed research update while child work remains."""

from types import SimpleNamespace

import assist.run_service as run_service_module
from assist.run_service import RunService
from edd.eval.research_live_outcome import _completed_parent_run, _requested_updates


def run(work_id, status="success", created="20:00:00", updated=None):
    return SimpleNamespace(work_id=work_id, status=status,
                           id=f"parent-{work_id}-{created}",
                           dispatch_key=(f"task-completion:{work_id.split(':', 1)[1]}-run"
                                         if work_id.startswith("task-completion:") else None),
                           created_at=f"2026-11-10T{created}+00:00",
                           updated_at=f"2026-11-10T{updated or created}+00:00")


def child(thread_id, role, status="success", updated="20:00:00"):
    return SimpleNamespace(thread_id=thread_id, id=f"{thread_id}-run",
                           assistant_id=role, status=status,
                           updated_at=f"2026-11-10T{updated}+00:00")


def test_pending_checker_and_parent_continuation_are_not_final():
    context = child("context", "context-agent", updated="20:01:00")
    research = child("research", "research-agent", updated="20:02:00")
    checker = child("checker", "fact-check-agent", "pending", "20:04:00")
    draft = run("task-completion:research", created="20:02:01")
    parents = [run("initial", created="20:00:00"),
               run("task-completion:context", created="20:01:01"), draft]

    assert _completed_parent_run(parents, [context, research, checker]) is None
    checker.status = "success"
    checker.updated_at = "2026-11-10T20:05:00+00:00"
    assert _completed_parent_run(parents, [context, research, checker]) is None
    parents.append(run("task-completion:checker", "pending", "20:05:01"))
    assert _completed_parent_run(parents, [context, research, checker]) is None
    parents[-1].status = "success"
    assert _completed_parent_run(parents, [context, research, checker]) is parents[-1]


def test_direct_final_and_context_only_final_need_no_checker():
    initial = run("initial", created="20:01:00")
    assert _completed_parent_run([initial], []) is initial
    assert _completed_parent_run([initial, run("next", "running", "20:02:00")], []) is None
    context = child("context", "context-agent", updated="20:03:00")
    continuation = run("task-completion:context", created="20:03:01")
    assert _completed_parent_run([initial, continuation], [context]) is continuation


def test_explicit_role_and_terminal_error_paths():
    research = child("research", "research-agent", updated="20:02:00")
    checker = child("checker", "fact-check-agent", updated="20:04:00")
    research_reply = run("task-completion:research", created="20:02:01")
    checker_reply = run("task-completion:checker", created="20:04:01")
    assert _completed_parent_run([research_reply], [research, checker],
                                 "fact-check-agent") is None
    assert _completed_parent_run([research_reply, checker_reply],
                                 [research, checker], "fact-check-agent") is checker_reply
    checker.status = "error"
    checker.updated_at = "2026-11-10T20:05:00+00:00"
    honest_error_reply = run("task-completion:checker", created="20:05:01")
    assert _completed_parent_run([research_reply, honest_error_reply],
                                 [research, checker]) is honest_error_reply
    assert _completed_parent_run([research_reply, honest_error_reply],
                                 [research, checker], "fact-check-agent") is None


def test_cancelled_unneeded_sibling_needs_no_new_parent_wake():
    research = child("research", "research-agent", updated="20:02:00")
    unused = child("unused", "context-agent", "cancelled", "20:03:00")
    final = run("task-completion:research", created="20:02:01")
    assert _completed_parent_run([final], [research, unused]) is final


def test_parent_wake_order_can_differ_from_child_completion_order():
    first = child("first", "research-agent", updated="20:02:00")
    second = child("second", "research-agent", updated="20:03:00")
    delayed_first_wake = run("task-completion:first", created="20:04:00")
    second_wake = run("task-completion:second", created="20:03:01")
    assert _completed_parent_run([second_wake, delayed_first_wake],
                                 [first, second]) is delayed_first_wake
    assert _completed_parent_run([delayed_first_wake], [first, second]) is None
    premature = run("task-completion:first", created="20:02:30")
    assert _completed_parent_run([second_wake, premature], [first, second]) is None


def test_queued_parent_wake_claimed_after_sibling_settles(tmp_path, monkeypatch):
    (tmp_path / "parent").mkdir()
    service = RunService(str(tmp_path))
    times = iter(f"2026-11-10T{value}+00:00" for value in
                 ("20:02:10", "20:04:00", "20:05:00"))
    monkeypatch.setattr(run_service_module, "_now", lambda: next(times))
    first = child("first", "research-agent", updated="20:02:00")
    second = child("second", "research-agent", updated="20:03:00")
    pending = service.create("parent", "general-agent", None,
                             work_id="task-completion:first",
                             dispatch_key="task-completion:first-run")
    claimed = service.claim("parent", pending.id)
    final = service.transition("parent", pending.id, "success", result="Source-backed reply")

    assert pending.created_at < second.updated_at < claimed.updated_at
    final_wakes = [run("task-completion:second", created="20:03:01"), final]
    assert _completed_parent_run(final_wakes, [first, second]) is None
    assert _completed_parent_run(final_wakes, [first, second], parent_started_at={
        final.id: claimed.updated_at}) is final
    # A draft already running before the sibling settled remains ineligible,
    # even if it only reaches success afterward.
    assert _completed_parent_run(final_wakes, [first, second], parent_started_at={
        final.id: "2026-11-10T20:02:30+00:00"}) is None


def test_parent_can_finish_after_cancelling_all_children():
    cancelled = child("unused", "research-agent", "cancelled", "20:03:00")
    final = run("initial", created="20:01:00", updated="20:04:00")
    assert _completed_parent_run([final], [cancelled]) is final
    final.updated_at = "2026-11-10T20:02:00+00:00"
    assert _completed_parent_run([final], [cancelled]) is None


def test_component_requires_only_its_note_update():
    selected = {"workspace_files": {"notes/ca.md": "old", "notes/or.md": "old"},
                "expected_updated_files": ["notes/ca.md"]}
    assert _requested_updates(selected, {"notes/ca.md": "new"})
    assert not _requested_updates(selected, {"notes/or.md": "new"})
    assert not _requested_updates(selected, {})
