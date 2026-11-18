"""Native queue holders retain their slot until the model-release fence succeeds."""
import threading

import pytest

from assist.thread_queue import QueueWaitTimeout, ThreadAffinityQueue


@pytest.mark.parametrize("timer", ["_on_cap", "_on_tick"])
def test_native_expiry_keeps_slot_until_cleanup(timer):
    queue = ThreadAffinityQueue(hold_timeout_s=100)
    stopped = []
    with queue.acquire("native", before_release=lambda: stopped.append(True)) as handle:
        handle.acquired_at -= 101
        getattr(queue, timer)(handle)
        assert handle.expired
        assert queue.current_handle() is handle
        assert stopped == []
    assert stopped == [True]
    assert queue.current_handle() is None


def test_failed_native_stop_cannot_admit_competitor():
    queue = ThreadAffinityQueue()

    def failed_stop():
        raise RuntimeError("generation stop not verified")

    with pytest.raises(RuntimeError, match="not verified"):
        with queue.acquire("native", before_release=failed_stop) as handle:
            pass
    assert handle.expired
    assert queue.current_handle() is handle
    with pytest.raises(QueueWaitTimeout):
        with queue.acquire("competitor", wait_timeout_s=0.01):
            pytest.fail("unconfirmed generation granted a second model slot")


def test_native_release_proof_precedes_waiter_admission():
    queue = ThreadAffinityQueue()
    queued = threading.Event()
    order = []

    def competitor():
        with queue.acquire("competitor", on_state_change=lambda stage: (
                queued.set() if stage == "queued" else None)):
            order.append("competitor")

    with queue.acquire("native", before_release=lambda: order.append("stopped")):
        waiter = threading.Thread(target=competitor)
        waiter.start()
        assert queued.wait(2)
    waiter.join(timeout=2)
    assert not waiter.is_alive()
    assert order == ["stopped", "competitor"]
