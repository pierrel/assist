"""Unit tests for the Provisioner — the approved-job state machine.

Real registry + proposal store over tmp dirs; run_job/on_complete/health_check are
recording stubs (the subprocess is increment 4). Exercises the whole lifecycle:
approve-fires-the-recorded-slug, importing→ready/failed, completion carrying the
user's request, D4 hold-and-retry on LLM-down, C1 restart durability (reconcile +
deliver_pending), remove/transit ops, and the duplicate-submit refusal.
"""
from datetime import datetime, timezone

from assist.geo.model import (
    Region, STATE_FAILED, STATE_IMPORTING, STATE_READY)
from assist.geo.proposals import Proposal, ProposalStore
from assist.geo.provisioner import Provisioner
from assist.geo.registry import RegionRegistry

_BBOX = (-124.8, 45.5, -116.9, 49.1)


def _proposal(slug="us/washington", tid="t1", request="directions in Tacoma"):
    return Proposal(slug=slug, display_name="Washington", bbox=_BBOX, size_bytes=700_000_000,
                    origin_tid=tid, user_request=request,
                    created_at=datetime.now(timezone.utc).isoformat())


class _Harness:
    """A provisioner over real stores with recording stubs."""

    def __init__(self, tmp_path, job_ok=True, healthy=True):
        self.registry = RegionRegistry(str(tmp_path))
        self.proposals = ProposalStore(str(tmp_path))
        self.jobs, self.completions = [], []
        self.job_ok, self.healthy = job_ok, healthy
        self.prov = Provisioner(
            self.registry, self.proposals,
            run_job=lambda op, slug: self.jobs.append((op, slug)) or self.job_ok,
            on_complete=lambda tid, msg: self.completions.append((tid, msg)),
            health_check=lambda: self.healthy)

    def submit_and_wait(self, op, slug):
        out = self.prov.submit(op, slug)
        self.drain()
        return out

    def drain(self):
        # flush the single worker without shutting it down (so deliver_pending, which
        # may enqueue, still works afterwards): a no-op task completes only after all
        # prior submitted tasks on the single worker.
        self.prov._executor.submit(lambda: None).result()


def test_add_full_lifecycle(tmp_path):
    h = _Harness(tmp_path)
    h.proposals.put(_proposal())
    assert h.submit_and_wait("add", "us/washington") == "queued"
    # job ran with the RECORDED slug; region is ready; completion delivered
    assert h.jobs == [("add", "us/washington")]
    r = h.registry.get("us/washington")
    assert r.state == STATE_READY and r.completion_delivered is True
    assert r.display_name == "Washington" and r.bbox == _BBOX   # seeded from the proposal
    (tid, msg), = h.completions
    assert tid == "t1"
    assert "Washington" in msg and "directions in Tacoma" in msg   # A1: carries intent
    assert h.proposals.get("us/washington") is None                # consumed


def test_reclick_approve_after_ready_keeps_held_completion(tmp_path):
    # add finished while LLM down → READY, completion_delivered=False, proposal held for
    # retry. A re-clicked approve must NOT drop that proposal (deliver_pending needs it).
    h = _Harness(tmp_path, healthy=False)
    h.proposals.put(_proposal())
    h.submit_and_wait("add", "us/washington")
    assert h.registry.get("us/washington").completion_delivered is False
    assert h.proposals.get("us/washington") is not None
    # re-click approve on the now-READY region
    assert "already loaded" in h.prov.submit("add", "us/washington")
    assert h.proposals.get("us/washington") is not None    # NOT orphaned
    # LLM back → the held completion still delivers
    h.healthy = True
    h.prov.deliver_pending()
    assert len(h.completions) == 1
    assert h.proposals.get("us/washington") is None


def test_add_without_proposal_refused(tmp_path):
    h = _Harness(tmp_path)
    out = h.prov.submit("add", "us/washington")
    assert "no pending proposal" in out
    assert h.registry.get("us/washington") is None and not h.jobs


def test_add_failure_flips_failed_and_notifies(tmp_path):
    h = _Harness(tmp_path, job_ok=False)
    h.proposals.put(_proposal())
    h.submit_and_wait("add", "us/washington")
    assert h.registry.get("us/washington").state == STATE_FAILED
    (tid, msg), = h.completions
    assert "failed" in msg and "NOT added" in msg
    assert h.proposals.get("us/washington") is None   # failure notice consumed it too


def test_llm_down_holds_completion_then_deliver_pending(tmp_path):
    h = _Harness(tmp_path, healthy=False)
    h.proposals.put(_proposal())
    h.submit_and_wait("add", "us/washington")
    # ready, but the completion is HELD (D4): nothing dispatched, flag durable
    r = h.registry.get("us/washington")
    assert r.state == STATE_READY and r.completion_delivered is False
    assert not h.completions and h.proposals.get("us/washington") is not None
    # LLM comes back → deliver_pending sends it
    h.healthy = True
    h.prov.deliver_pending()
    assert len(h.completions) == 1
    assert h.registry.get("us/washington").completion_delivered is True
    assert h.proposals.get("us/washington") is None


def test_restart_durability_reconcile_then_deliver(tmp_path):
    # simulate: import finished (ready, undelivered) + web restarted → a FRESH
    # provisioner over the same dirs delivers from disk alone (C1)
    h1 = _Harness(tmp_path, healthy=False)
    h1.proposals.put(_proposal())
    h1.submit_and_wait("add", "us/washington")
    h2 = _Harness(tmp_path)          # new process, same disk
    h2.prov.reconcile()
    h2.prov.deliver_pending()
    (tid, msg), = h2.completions
    assert tid == "t1" and "Washington" in msg


def test_reconcile_flips_orphaned_importing(tmp_path):
    h = _Harness(tmp_path)
    h.registry.put(Region(slug="socal", display_name="SoCal", bbox=_BBOX,
                          state=STATE_IMPORTING))
    assert h.prov.reconcile() == ["socal"]
    assert h.registry.get("socal").state == STATE_FAILED


def test_duplicate_submit_refused_while_importing(tmp_path):
    h = _Harness(tmp_path)
    h.registry.put(Region(slug="socal", display_name="SoCal", bbox=_BBOX,
                          state=STATE_IMPORTING))
    assert "already has a job running" in h.prov.submit("remove", "socal")
    assert not h.jobs


def test_remove_ok_and_remove_failure(tmp_path):
    h = _Harness(tmp_path)
    h.registry.put(Region(slug="socal", display_name="SoCal", bbox=_BBOX,
                          state=STATE_READY))
    h.submit_and_wait("remove", "socal")
    assert h.registry.get("socal") is None
    # failure path: data unchanged → back to ready, still served
    h2 = _Harness(tmp_path, job_ok=False)
    h2.registry.put(Region(slug="norcal", display_name="NorCal", bbox=_BBOX,
                           state=STATE_READY))
    h2.submit_and_wait("remove", "norcal")
    assert h2.registry.get("norcal").state == STATE_READY


def test_remove_unloaded_refused(tmp_path):
    h = _Harness(tmp_path)
    assert "not loaded" in h.prov.submit("remove", "socal")


def test_transit_ok_sets_flag(tmp_path):
    h = _Harness(tmp_path)
    h.registry.put(Region(slug="norcal", display_name="NorCal", bbox=_BBOX,
                          state=STATE_READY, has_transit=False))
    h.submit_and_wait("transit", "norcal")
    r = h.registry.get("norcal")
    assert r.has_transit is True and r.state == STATE_READY


def test_transit_failure_keeps_flag_off_and_ready(tmp_path):
    h = _Harness(tmp_path, job_ok=False)
    h.registry.put(Region(slug="norcal", display_name="NorCal", bbox=_BBOX,
                          state=STATE_READY, has_transit=False))
    h.submit_and_wait("transit", "norcal")
    r = h.registry.get("norcal")
    assert r.has_transit is False and r.state == STATE_READY


def test_transit_refused_on_non_ready_region(tmp_path):
    # a FAILED (or importing) region can't take transit — and it must NOT be flipped
    # out of its state for a transit job
    h = _Harness(tmp_path)
    h.registry.put(Region(slug="socal", display_name="SoCal", bbox=_BBOX,
                          state=STATE_FAILED, has_transit=False))
    assert "can't add transit" in h.prov.submit("transit", "socal")
    assert h.registry.get("socal").state == STATE_FAILED and not h.jobs


def test_remove_failure_restores_prior_failed_not_ready(tmp_path):
    # removing a FAILED region whose job returns failure must restore FAILED, never
    # promote a never-validated region to READY
    h = _Harness(tmp_path, job_ok=False)
    h.registry.put(Region(slug="socal", display_name="SoCal", bbox=_BBOX,
                          state=STATE_FAILED))
    h.submit_and_wait("remove", "socal")
    assert h.registry.get("socal").state == STATE_FAILED


def test_run_job_exception_is_failure_not_crash(tmp_path):
    def boom(op, slug):
        raise RuntimeError("docker exploded")
    h = _Harness(tmp_path)
    h.prov._run_job = boom
    h.proposals.put(_proposal())
    h.submit_and_wait("add", "us/washington")
    assert h.registry.get("us/washington").state == STATE_FAILED


def test_dispatch_exception_held_for_retry(tmp_path):
    h = _Harness(tmp_path)
    h.proposals.put(_proposal())
    calls = {"n": 0}

    def flaky(tid, msg):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("thread briefly missing")
        h.completions.append((tid, msg))
    h.prov._on_complete = flaky
    h.submit_and_wait("add", "us/washington")
    # first dispatch raised → held (flag still false, proposal kept)
    assert h.registry.get("us/washington").completion_delivered is False
    h.prov.deliver_pending()
    assert len(h.completions) == 1
    assert h.registry.get("us/washington").completion_delivered is True


def test_unknown_op_refused(tmp_path):
    h = _Harness(tmp_path)
    assert "unknown operation" in h.prov.submit("explode", "socal")
