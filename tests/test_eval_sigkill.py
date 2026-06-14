"""Proves the eval harness's per-test kill mechanic actually SIGKILLs a
zombie-shaped process and frees promptly (no model, no network).

The harness (scripts/run-evals.sh) caps each test with
`timeout --kill-after=<grace> -s TERM <deadline> pytest <nodeid>`.  The
load-bearing claim: a test whose worker threads are blocked such that
SIGTERM is never serviced (the real failure: threads parked in a C-level
socket read to llama's slot) is still killed at the OS level by the
SIGKILL escalation, so the slot frees for the next test.  Here we model
that with a SIGTERM-ignoring sleeper and assert the escalation fires.
"""
import shutil
import subprocess
import sys
import textwrap
import time

import pytest

# The harness's kill mechanic is GNU coreutils `timeout`; skip cleanly
# where it isn't on PATH rather than failing with a confusing rc=127.
pytestmark = pytest.mark.skipif(
    shutil.which("timeout") is None,
    reason="GNU coreutils `timeout` not available",
)


def _run(deadline, grace, body):
    """Run `timeout --kill-after=grace -s TERM deadline python -c <body>`
    (the harness incantation, with python standing in for pytest) and
    return (rc, elapsed_seconds)."""
    start = time.monotonic()
    proc = subprocess.run(
        ["timeout", f"--kill-after={grace}", "-s", "TERM", str(deadline),
         sys.executable, "-c", textwrap.dedent(body)],
        capture_output=True,
    )
    return proc.returncode, time.monotonic() - start


def test_sigterm_ignoring_process_is_sigkilled():
    """A process that ignores SIGTERM (like a thread blocked in a C call
    that can't service it) is killed by --kill-after's SIGKILL: rc=137,
    and it dies at ~deadline+grace, NOT after its 10000s sleep."""
    rc, elapsed = _run(
        deadline=1, grace=1,
        body="""
            import signal, time
            signal.signal(signal.SIGTERM, signal.SIG_IGN)  # swallow TERM
            time.sleep(10000)
        """,
    )
    # The --kill-after SIGKILL fired.  coreutils `timeout` re-raises the
    # kill on itself to propagate it, so the bash harness sees rc=137
    # (128+9) while Python's subprocess reports the signal death as -9 —
    # both mean SIGKILL.  The harness's `[ rc -ge 124 ]` covers 137.
    assert rc in (-9, 137), f"expected SIGKILL (-9 or 137), got {rc}"
    # Killed at deadline(1) + grace(1) ≈ 2s, with generous slack — NOT 10000s.
    assert elapsed < 10, f"took {elapsed:.1f}s — SIGKILL did not free it promptly"


def test_well_behaved_process_exits_before_deadline():
    """A normal fast test exits 0 well within the deadline — the cap only
    fires on a runaway, never on healthy work."""
    rc, elapsed = _run(deadline=10, grace=5, body="import sys; sys.exit(0)")
    assert rc == 0
    assert elapsed < 5


def test_timeout_codes_distinguish_from_failure():
    """The harness treats ONLY rc 124 (TERM) and 137 (SIGKILL) as timeouts.
    A real test failure (exit 1) is neither, so the summary never miscounts
    a red test as a timeout — and `timeout`'s own 125-127 errors stay
    distinct (not folded into the timeout bucket)."""
    rc_fail, _ = _run(deadline=10, grace=5, body="import sys; sys.exit(1)")
    assert rc_fail == 1
    assert rc_fail not in (124, 137)  # a failure is not a timeout

    rc_term, _ = _run(
        deadline=1, grace=5,
        body="import time; time.sleep(10000)",  # honors TERM (no handler)
    )
    # No SIGTERM handler -> default-terminate at the deadline -> rc=124.
    assert rc_term == 124


def test_nodeid_xml_name_matches_eval_history_regex():
    """The per-test XML filename must keep the <prefix>-<YYYYMMDD-HHMM>.xml
    shape that manage/eval_history.py parses, or the live /evals page
    breaks.  Mirror the harness's sanitize and assert the regex matches."""
    import re
    run_id_re = re.compile(r"^(.+?)-(\d{8}-\d{4})\.xml$")

    nodeid = "edd/eval/test_agent.py::TestAgent::test_adds_item_correctly"
    safe = nodeid.replace("::", "__").replace("/", "_")
    filename = f"{safe}-20260614-0030.xml"

    m = run_id_re.match(filename)
    assert m is not None, f"{filename!r} does not match eval_history _RUN_ID_RE"
    # The load-bearing invariant: the run-id timestamp is parsed correctly.
    # (The prefix may contain hyphens in general; the only real hazard is a
    # prefix ending in its OWN '-<YYYYMMDD-HHMM>' — which test ids don't.)
    assert m.group(2) == "20260614-0030"
