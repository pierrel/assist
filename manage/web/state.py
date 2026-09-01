"""Module-level state and lifespan helpers shared across the web app.

This is the only module that actually owns ``MANAGER``, ``DOMAINS``,
the in-memory caches, and the status state machine.  Other submodules
import from here.  No FastAPI route handlers live in this file — those
sit alongside the screen they render (``threads.py``, ``review.py``,
``evals.py``).
"""
from __future__ import annotations

import html
import json
import logging
import os
import secrets
import sys
import tempfile
import threading
from contextlib import asynccontextmanager
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Dict

from fastapi import FastAPI
from starlette.concurrency import run_in_threadpool
import anyio

from assist.domain_manager import DomainManager
from assist.env import load_dev_env
from assist.sandbox_manager import SandboxManager
from assist.schedule.store import ScheduleStore
from assist.schedule.tools import schedule_tools
from assist.events.store import SubscriptionStore
from assist.events.tools import subscription_tools
from assist.events.reply import reply_tools, REPLY_INTERRUPT_ON
from assist.events.email import email_tools, EMAIL_INTERRUPT_ON
from assist.events.notify import notify_tools
from assist.events.inbound import InboundLog
from assist.backlog import MessageBacklog
from assist.run_service import RunService
from assist.egress.store import EgressStore, approvals_dir_is_safe
from assist.egress.tools import egress_tools
from assist.agent import set_execution_egress_tools
from assist.sandbox_manager import _load_egress_allowlist
from assist.geo.catalog import Catalog
from assist.geo.proposals import ProposalStore
from assist.geo.registry import RegionRegistry
from assist.geo.tools import geo_tools
from assist.thread_manager import (
    ThreadManager, set_web_tools, set_web_triage_tools, set_web_interrupt_on,
    set_web_triage_interrupt_on)
from assist.thread_engine import read_thread_engine
from assist.pi_preview import PiPreviewPolicy
from assist.location import LocationStore, get_location
from assist.frequency import FrequencyDecisionStore, frequency_tools
from edd.live_capture import CaptureStore, CaptureWorker


def _configure_logging() -> None:
    """Wire DEBUG-level logging to both stdout (live tail) and a per-session
    file at ``logs/web-{YYYY-MM-DD-HHMMSS}.log``.

    Stdout preserves the existing ``make web`` developer experience — you
    still see the running tail in your terminal.  The file lets a future
    session inspect what happened when the terminal scrollback is gone
    (and lets us diagnose stuck requests after the fact).

    File rotation: 50 MB per file, up to 5 backups (so a runaway session
    can't fill the disk).  One main file per server start; rotated
    overflow lands beside it as ``...log.1``, ``...log.2``, etc.
    """
    logs_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "logs",
    )
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(
        logs_dir,
        f"web-{datetime.now():%Y-%m-%d-%H%M%S}.log",
    )

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    file_handler = RotatingFileHandler(
        log_path, maxBytes=50 * 1024 * 1024, backupCount=5
    )
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    # Replace any handlers basicConfig may have left behind on import.
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(stream_handler)
    root.addHandler(file_handler)
    # Surface the chosen file so it's discoverable in the live tail.
    logging.getLogger(__name__).info("Logging to %s", log_path)


_configure_logging()
logging.getLogger("assist.model").setLevel(logging.DEBUG)

load_dev_env()

ROOT = os.getenv("ASSIST_THREADS_DIR", "/tmp/assist_threads")
MANAGER = ThreadManager(ROOT)
CAPTURES_DIR = os.getenv(
    "ASSIST_CAPTURES_DIR", os.path.join(os.path.dirname(os.path.abspath(ROOT)), "assist_captures"))
CAPTURE_STORE = CaptureStore(CAPTURES_DIR, threads_root=ROOT)
CAPTURE_WORKER = CaptureWorker(CAPTURE_STORE)
CAPTURE_CSRF = secrets.token_urlsafe(32)
# Capture filesystem work must never wait behind ordinary request handlers in
# Starlette's shared threadpool.  This limiter is used by capture routes and the
# lifespan stop path only.
CAPTURE_THREAD_LIMITER = anyio.CapacityLimiter(2)
PI_PREVIEW = PiPreviewPolicy(ROOT)
LOCATION_STORE = LocationStore(ROOT)
FREQUENCY_STORE = FrequencyDecisionStore(ROOT)

# The one shared schedule store (disk-as-truth on the thread root) — used by the
# Scheduler (started in the lifespan, see threads.py) AND the schedule tools AND the
# /schedules view, so they share its single read-modify-write lock. The web app is a
# single uvicorn worker, so exactly one Scheduler/store pair exists.
SCHEDULE_STORE = ScheduleStore(ROOT)
# The subscription store shares the thread root the same way (disk-as-truth, per-thread).
# Subscription tools let the agent set up message-event triage; the inbound-SMS route reads
# the same store to route a message to its subscription's thread.
SUBSCRIPTION_STORE = SubscriptionStore(ROOT)
# Durable inbound-message log (records before the 200; dedup by content-hash message_id).
INBOUND_LOG = InboundLog(ROOT)
# Legacy follow-up journal, read only during startup migration. New work is persisted
# in RUN_SERVICE below.
MESSAGE_BACKLOG = MessageBacklog(ROOT)
# Durable execution authority.  ``status.json`` remains the web UI projection;
# the legacy backlog is read only during startup migration.
RUN_SERVICE = RunService(ROOT)
# Normal turns get the config tools (schedule + subscription). A TRIAGE turn (untrusted
# inbound SMS) gets ONLY send_reply, HITL-gated — never the host-effect config tools — so an
# injected text can't plant/delete a subscription or schedule (MANAGER.get(triage=True)).
# notify_tools takes an injected mark-urgent callback (notify.py never imports web
# state → no import cycle).  The lambda is NOT redundant: `_mark_urgent` is defined
# ~300 lines below (kept beside the symmetric _UNSEEN block), but this runs at import
# time, so a bare `notify_tools(_mark_urgent)` would NameError — the lambda DEFERS the
# lookup to tool-call time.  (Do not "simplify" it away.)
# Lands in the NORMAL tool set ONLY — never the triage set — so an untrusted inbound
# SMS can't force an urgent badge on Pierre's phone.
# geo (region download-on-demand) — wired only when the travel-infra dir is configured
# for the web service (holds regions.json / catalog.json / proposals.json + input/). When
# unset (the default until the deploy passes it), the geo stores + tools are simply absent,
# so prod behaviour is unchanged. The Provisioner is built in threads.py (it needs the
# dispatch/urgent/health callbacks that live there — the Scheduler precedent).
GEO_DIR = os.getenv("TRAVEL_INFRA_DIR")
if GEO_DIR and not os.path.isdir(GEO_DIR):
    # Fail fast on a mis-set path rather than register tools that crash on first write
    # (ProposalStore.put → FileNotFoundError mid-turn). Disable geo, log loudly.
    logging.error("TRAVEL_INFRA_DIR=%s is not a directory — geo disabled", GEO_DIR)
    GEO_DIR = None
GEO_REGISTRY = RegionRegistry(GEO_DIR) if GEO_DIR else None
GEO_CATALOG = Catalog(GEO_DIR) if GEO_DIR else None
GEO_PROPOSALS = ProposalStore(GEO_DIR) if GEO_DIR else None
# One app-lifetime token embedded in the geo approve/decline/delete/transit forms (in the
# /geo page AND the in-thread proposal banner) + required on the POST — a cross-origin
# forge can't read a page to obtain it, so it can't fire a destructive rebuild (T1).
GEO_CSRF = secrets.token_urlsafe(16)
_geo_tools = (geo_tools(GEO_REGISTRY, GEO_CATALOG, GEO_PROPOSALS)
              if GEO_DIR else [])

# Egress approval HITL (docs/2026-07-21-egress-approval-hitl.org) — wired only
# when the approvals dir is configured; unset ⇒ tools/routes/cards absent and
# the proxy runs exactly as before (dormant, prod-safe default). The startup
# guard refuses a dir under the thread root: a sandbox bind-mounts its
# thread's work_dir rw, so an approvals dir inside the root could hand a
# sandbox write access to its own grants (self-approval).
EGRESS_DIR = os.getenv("ASSIST_EGRESS_APPROVALS_DIR")
if EGRESS_DIR and not approvals_dir_is_safe(EGRESS_DIR):
    # The same shared predicate guards the sandbox layer's proxy mount +
    # client-map writes — both readers refuse, or the guard is theater.
    logging.error(
        "ASSIST_EGRESS_APPROVALS_DIR=%s is under the thread root — a sandbox "
        "could reach it; egress approvals DISABLED", EGRESS_DIR)
    EGRESS_DIR = None
EGRESS_STORE = EgressStore(EGRESS_DIR) if EGRESS_DIR else None
EGRESS_CSRF = secrets.token_urlsafe(16)
# The committed base allowlist, for the list tool + already-allowed
# correctives. Read once at startup — the conf only changes with a deploy,
# which restarts this process.
EGRESS_BASE_HOSTS = (frozenset(_load_egress_allowlist())
                     if EGRESS_STORE else frozenset())
_egress_tools = (egress_tools(EGRESS_STORE, EGRESS_BASE_HOSTS,
                              thread_dir=MANAGER.thread_dir)
                 if EGRESS_STORE else [])

set_web_tools(schedule_tools(SCHEDULE_STORE) + subscription_tools(SUBSCRIPTION_STORE)
              + notify_tools(lambda tid: _mark_urgent(tid))
              + _geo_tools + email_tools() + [get_location]
              + frequency_tools(FREQUENCY_STORE))
set_execution_egress_tools(_egress_tools)
set_web_triage_tools(reply_tools())
set_web_interrupt_on(EMAIL_INTERRUPT_ON)
set_web_triage_interrupt_on(REPLY_INTERRUPT_ON)
_raw = os.getenv("ASSIST_DOMAINS", "")
DOMAINS: list[str] = [d.strip() for d in _raw.split(",") if d.strip()]
DESCRIPTION_CACHE: Dict[str, str] = {}
DESCRIPTION_LOCK = threading.Lock()
DOMAIN_MANAGERS: Dict[str, DomainManager] = {}  # tid -> DomainManager

# Serialises ``DomainManager.merge_to_main`` and ``push_main`` across
# concurrent web requests.  The web app runs as a single uvicorn
# worker, so an in-process lock is sufficient — without it, two
# threads merging or pushing within the same second would race the
# host's ``git fetch`` / ``git push`` sequence and one would leave
# the local repo in a partial state.  If the deploy ever runs
# multiple workers, swap this for an ``flock`` on a file in
# ``ROOT``.
MERGE_LOCK = threading.Lock()


def _domain_label(url: str) -> str:
    """'user@host:/path/to/life.git' -> 'life'"""
    return url.rstrip("/").rsplit("/", 1)[-1].removesuffix(".git")


def _domain_selector_html() -> str:
    """Return HTML for the domain selector in the new-thread form."""
    if len(DOMAINS) > 1:
        opts = "\n".join(
            f'<option value="{html.escape(d)}">{html.escape(_domain_label(d))}</option>'
            for d in DOMAINS
        )
        return (
            '<select name="domain" style="margin-bottom:.5rem; padding:.6rem; '
            'border:1px solid #ccc; border-radius:6px; font-size:16px; width:100%; '
            'min-height:44px;">'
            f"{opts}</select>"
        )
    if len(DOMAINS) == 1:
        return f'<input type="hidden" name="domain" value="{html.escape(DOMAINS[0])}" />'
    return ""


def _thread_domain_html(tid: str) -> str:
    """Return a small badge showing the domain name for a thread, if any."""
    dm = _get_domain_manager(tid)
    if dm and dm.repo:
        label = html.escape(_domain_label(dm.repo))
        return (
            f'<span style="display:inline-block; font-size:.8rem; color:#555; '
            f'background:#f0f0f0; padding:.2rem .5rem; border-radius:4px; '
            f'margin-bottom:.5rem;">{label}</span>'
        )
    return ""


def _get_domain_manager(tid: str, domain: str | None = None) -> DomainManager | None:
    """Get or create a DomainManager for a thread, caching by tid.

    For new threads pass *domain* (a git URL to clone).
    For existing threads pass None — DomainManager auto-detects the remote.

    Passes the last 4 chars of ``tid`` as ``branch_suffix`` so per-thread
    branches and post-merge re-branches are unambiguous when two threads
    are created within the same UTC second.
    """
    if tid in DOMAIN_MANAGERS:
        return DOMAIN_MANAGERS[tid]
    twdir = MANAGER.thread_default_working_dir(tid)
    try:
        dm = DomainManager(twdir, domain, branch_suffix=tid[-4:])
        DOMAIN_MANAGERS[tid] = dm
        return dm
    except Exception:
        return None


def _get_sandbox_backend(tid: str, tz: str | None = None, *,
                         include_agent: bool = True):
    """Get sandbox backend for a thread, or None if Docker is unavailable.

    ``tz`` is the per-turn context-rider timezone, so this turn's sandbox ``date``
    runs in the user's local time (else the host/server zone).

    ``include_agent`` mounts the visible thread's private main-agent directory.
    Hidden child runs pass ``False`` and receive self-contained task briefs instead.

    Runs off the event loop (from ``_process_message``'s background task), so the
    turn-start origin pre-fetch is safe here: the host refreshes ``origin/main`` in the
    clone (it has git + origin access) so the agent can rebase onto a current local
    ``origin/main`` — the agent cannot fetch from inside the sandbox itself."""
    work_dir = MANAGER.thread_default_working_dir(tid)
    dm = _get_domain_manager(tid)
    if dm is not None:
        try:
            dm.fetch_origin()
        except Exception as e:
            logging.getLogger(__name__).warning("origin pre-fetch failed for %s: %s", tid, e)
    return SandboxManager.get_sandbox_backend(
        work_dir, tz=tz,
        agent_dir=(MANAGER.thread_agent_dir(tid) if include_agent else None))


def _has_unmerged_changes(tid: str) -> bool:
    """True if this thread's working tree has unmerged work vs main.

    Used by the index page to surface an "unmerged" badge on threads
    that finished a turn but haven't been merged yet.  Wraps
    ``DomainManager.has_changes_vs_main`` and swallows any exception
    (a transient git error here mustn't 500 the index page).
    """
    dm = _get_domain_manager(tid)
    if not dm:
        return False
    try:
        return dm.has_changes_vs_main()
    except Exception as e:
        # Log at debug so a future "why is the badge wrong?" debug
        # session has a paper trail without spamming the live tail.
        logging.getLogger(__name__).debug(
            "has_changes_vs_main failed for %s: %s", tid, e,
        )
        return False


def _conflict_path(tid: str) -> str:
    return os.path.join(MANAGER.thread_dir(tid), "merge_conflict.json")


def _get_conflict(tid: str) -> dict | None:
    """Return the persisted merge-conflict state for ``tid``, or None.

    The dict shape is ``{"branch": str, "files": [str], "raised_at": iso}``.
    Set by ``merge_thread`` when ``merge_to_main`` raises
    :class:`MergeConflictError`; cleared when a subsequent merge
    succeeds.  Stored separately from ``status.json`` so the agent
    can keep transitioning through ``processing`` ↔ ``ready`` while
    the user works through the conflict.
    """
    path = _conflict_path(tid)
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _atomic_write(path: str, data: str, *, create_parent: bool = True) -> None:
    """Write ``data`` to ``path`` atomically: a uniquely-named temp file in the
    same directory + os.replace. ``create_parent=False`` makes a vanished parent
    fail rather than resurrecting it. The unique temp name means concurrent writers
    for the same path can't clobber a shared temp or leave a half-written file."""
    if create_parent:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path),
                               prefix=os.path.basename(path) + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _set_conflict(tid: str, branch: str, files: list[str]) -> None:
    data = {
        "branch": branch,
        "files": files,
        "raised_at": datetime.now().isoformat(timespec="seconds"),
    }
    _atomic_write(_conflict_path(tid), json.dumps(data))


def _clear_conflict(tid: str) -> None:
    path = _conflict_path(tid)
    if os.path.isfile(path):
        os.remove(path)


def _evict_caches(tid: str) -> None:
    """on_delete callback: drop the thread from in-process caches.

    Passed to ``MANAGER.hard_delete`` so the web process forgets a
    thread the moment its dir + DB rows are gone.  Keeps
    ``assist/thread.py`` web-agnostic — it never imports these
    module-level dicts.
    """
    DOMAIN_MANAGERS.pop(tid, None)
    DESCRIPTION_CACHE.pop(tid, None)
    _UNSEEN.discard(tid)  # the marker file went with the dir; drop the set entry too
    _URGENT.discard(tid)  # same for the urgent flag


# --- Thread status tracking ----------------------------------------------
# Stages used for the async thread creation flow:
#   initializing      - thread row created, background task queued
#   cloning           - git clone in progress
#   starting_sandbox  - docker container starting
#   queued            - waiting for another thread's LLM affinity hold
#   processing        - agent is running on a message
#   paused            - yielded the slot at its quantum so a waiting turn could
#                       run; the resume scheduler will continue it (fair scheduling)
#   ready             - idle, accepting input
#   error             - something failed; see status["error"]
INIT_STAGES = {"initializing", "cloning", "starting_sandbox"}
# `paused` is BUSY (not INIT): the turn is mid-flight and will resume — yielded at
# its fair-scheduling quantum, OR awaiting restart recovery (the lifespan rewrites
# every interrupted busy thread to `paused`) — so the page renders history + keeps
# input enabled, and _mark_pending won't clobber it.
BUSY_STAGES = INIT_STAGES | {"processing", "queued", "paused"}

STAGE_LABELS = {
    "initializing": "Setting up thread...",
    "cloning": "Cloning repository...",
    "starting_sandbox": "Starting sandbox container...",
    "queued": "Waiting for another thread to finish...",
    "processing": "Processing your message...",
    "paused": "Paused — will resume...",
}

# Single-word variants for the compact thread-LIST page only. The in-thread banner
# keeps the fuller STAGE_LABELS sentences — it has room and the sentence explains *why*
# a thread is busy (esp. queued/paused). List rows are dense, so one word each reads better.
LIST_STAGE_LABELS = {
    "initializing": "Initializing",
    "cloning": "Cloning",
    "starting_sandbox": "Starting",
    "queued": "Queued",
    "processing": "Processing",
    "paused": "Paused",
}


def _status_path(tid: str) -> str:
    return os.path.join(MANAGER.thread_dir(tid), "status.json")


def _get_status(tid: str) -> dict:
    path = _status_path(tid)
    if not os.path.isfile(path):
        return {"stage": "ready"}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {"stage": "ready"}


def _set_status(tid: str, stage: str, **kwargs) -> None:
    _atomic_write(_status_path(tid), json.dumps({"stage": stage, **kwargs}))
    # Single choke point for the "unseen AI response" badge: `ready` and
    # `awaiting_approval` are set ONLY at _process_message's three response-success
    # exits (incl. the supersede early-return), and nowhere else — so marking here
    # catches every response/draft the user should see, and can't miss an exit.
    # Errors don't mark (the "error" badge, above "new" in precedence, covers them).
    if stage in ("ready", "awaiting_approval"):
        _mark_unseen_response(tid)


# --- Per-turn timing (completed human→AI elapsed) -------------------------
# A per-thread sidecar keyed by human-message ordinal → elapsed seconds (int), the
# durations shown as badges on completed replies. Separate from status.json (which holds
# only the single CURRENT status and drops started_at at ready) because it must carry N
# completed-turn durations; separate from the messages/checkpoint (langchain messages have
# no timestamp, and stamping them would touch assist/thread.py + break the render tests).
# Mirrors the merge_conflict.json trio: written off-loop by _process_message's success
# exits, read on-loop by render_thread (a bare missing-ok read, no lock — same as
# _get_status / _get_conflict). One writer per tid (THREAD_QUEUE + serial resume), and
# _atomic_write makes a concurrent render-read see the whole old or whole new file.
def _timings_path(tid: str) -> str:
    return os.path.join(MANAGER.thread_dir(tid), "turn_timings.json")


def _get_timings(tid: str) -> dict:
    """Map of human-message ordinal (str) → elapsed seconds (int). {} if missing/corrupt.

    Validates the parsed value is a dict: a syntactically-valid but wrong-shaped file
    (a bare number/bool/list/string) parses fine but would break the on-loop render's
    ``str(ord) in timings`` / index — so anything non-dict degrades to {} (no badges)."""
    path = _timings_path(tid)
    if not os.path.isfile(path):
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        # Enforce the {ordinal: number} contract so a malformed value (e.g. {"1": "oops"})
        # can't reach _format_elapsed's int(round(...)) and crash the on-loop render.
        return {k: v for k, v in data.items() if isinstance(v, (int, float))}
    except Exception:
        return {}


def _append_timing(tid: str, ordinal: int, seconds: float) -> None:
    """Record one completed turn's elapsed. Read-modify-write of the small dict; off-loop."""
    timings = _get_timings(tid)
    timings[str(ordinal)] = round(seconds)
    _atomic_write(_timings_path(tid), json.dumps(timings))


# --- "unseen AI response" badge state -------------------------------------
# A thread carries an "unseen" AI response from when a turn produces a
# response/draft until the user OPENS that thread's page.  Two representations of
# the one bit, kept in sync (see docs/2026-07-03-unread-badge.org):
#   - a marker file per thread (durable — survives restart),
#   - the _UNSEEN set (the badge READ PATH — render_index tests membership per row,
#     O(1), no FS stat; Pierre's cache to keep list page-loads fast).
# set add/discard/`in` are each atomic under the GIL, so no lock is needed across
# the event loop (render_index read, get_thread clear) and the bg _process_message
# thread (mark) — preserving event-loop liveness.
_UNSEEN: set[str] = set()


def _unseen_path(tid: str) -> str:
    return os.path.join(MANAGER.thread_dir(tid), "unseen_response")


def _mark_unseen_response(tid: str) -> None:
    """Record an AI response the user hasn't seen: the live set + an idempotent
    marker file (existence is the whole signal — touch, tolerate exists)."""
    _UNSEEN.add(tid)
    # The durable marker is BEST-EFFORT: this runs in the terminal path (via
    # _set_status on ready/awaiting_approval), so a file-write OSError must NEVER
    # propagate and flip an otherwise-successful turn to error. The in-memory set
    # carries the badge this session; on write failure we just lose restart-survival.
    try:
        open(_unseen_path(tid), "w").close()
    except OSError:
        pass


def _clear_unseen_response(tid: str) -> None:
    """Mark ``tid`` seen (its page was opened): drop the set entry + the marker.
    Bare, missing-ok unlink — no _atomic_write, no lock (runs on the event loop)."""
    _UNSEEN.discard(tid)
    # Truly missing-ok AND never-raise: this runs on the event loop (get_thread), so
    # neither a missing file nor any OSError may 500 the thread page. A try/remove
    # (not isfile→remove) also avoids the TOCTOU race with the marker's writer.
    try:
        os.remove(_unseen_path(tid))
    except OSError:
        pass


def _has_unseen_response(tid: str) -> bool:
    return tid in _UNSEEN


def load_unseen_cache() -> None:
    """Rebuild _UNSEEN from the on-disk markers at startup, so a scheduled/SMS
    response's badge survives a restart.  One stat per thread, once.  A true
    REBUILD — clears first, so it's idempotent and drops stale entries if ever
    re-invoked with a non-empty set."""
    _UNSEEN.clear()
    for tid in MANAGER.list():
        try:
            if os.path.isfile(_unseen_path(tid)):
                _UNSEEN.add(tid)
        except OSError:
            pass  # a per-thread dir hiccup must not abort startup


# --- "urgent" (agent notify-flagged) badge state --------------------------
# The notify() agent tool flags a thread URGENT — a STRONGER signal than v1's
# "unseen".  Same durable-marker + in-memory-set + startup-rebuild pattern as
# _UNSEEN, lock-free (GIL-atomic set ops across the on-loop reads/clear and the
# off-loop tool call).  Read path: _has_urgent (the thread-list "urgent" pill).
_URGENT: set[str] = set()


def _urgent_path(tid: str) -> str:
    return os.path.join(MANAGER.thread_dir(tid), "urgent_response")


def _mark_urgent(tid: str) -> None:
    """Flag ``tid`` urgent (from the notify tool): the live set + a BEST-EFFORT
    durable marker (OSError swallowed — a marker write must never fail the turn)."""
    _URGENT.add(tid)
    try:
        open(_urgent_path(tid), "w").close()
    except OSError:
        pass


def _clear_urgent(tid: str) -> None:
    """Clear ``tid``'s urgent flag (its page was opened): drop the set entry + the
    marker.  Bare, missing-ok, never-raise (runs on the event loop in get_thread)."""
    _URGENT.discard(tid)
    try:
        os.remove(_urgent_path(tid))
    except OSError:
        pass


def _has_urgent(tid: str) -> bool:
    return tid in _URGENT


def load_urgent_cache() -> None:
    """Rebuild _URGENT from the on-disk markers at startup (mirror of
    load_unseen_cache) — a notify-flagged urgent survives a restart."""
    _URGENT.clear()
    for tid in MANAGER.list():
        try:
            if os.path.isfile(_urgent_path(tid)):
                _URGENT.add(tid)
        except OSError:
            pass


def _thread_title(tid: str) -> str:
    """Display title for a thread; placeholder if still initializing."""
    status = _get_status(tid)
    if status.get("stage") in BUSY_STAGES:
        if status.get("origin") == "continuation":
            # A background follow-up turn: the pending text is the AGENT's task
            # note, not the user's words — show the thread's real description.
            # CACHE/FILE READ ONLY, never get_cached_description's generating
            # path: this runs on the event loop while the single LLM slot is
            # GUARANTEED busy (the continuation turn holds it) — a cache-miss
            # generation here would park the whole server behind that turn.
            if tid in DESCRIPTION_CACHE:
                return DESCRIPTION_CACHE[tid]
            try:
                with open(os.path.join(MANAGER.thread_dir(tid),
                                       "description.txt")) as f:
                    return f.read()
            except OSError:
                return "Following up..."
        pending = (status.get("pending_message") or "").strip()
        if pending:
            short = pending.splitlines()[0]
            return short[:60] + ("..." if len(short) > 60 or len(pending) > len(short) else "")
        return "New thread"
    return get_cached_description(tid)


def get_cached_description(tid: str) -> str:
    """Get thread description from cache, or read from FS and cache if miss."""
    if tid in DESCRIPTION_CACHE:
        return DESCRIPTION_CACHE[tid]

    # Cache miss - read the saved description, else generate one (which also
    # writes + caches it via set_description, so the two write paths stay one).
    try:
        description_file = os.path.join(MANAGER.thread_dir(tid), "description.txt")
        if os.path.isfile(description_file):
            with open(description_file, 'r') as f:
                DESCRIPTION_CACHE[tid] = f.read()
        elif read_thread_engine(MANAGER.thread_dir(tid)).name == "pi":
            return "Pi thread"
        else:
            set_description(tid, MANAGER.get(tid).description())
        return DESCRIPTION_CACHE[tid]
    except Exception:
        return tid


def set_description(tid: str, description: str) -> None:
    """Persist a user-set thread description (the displayed title) and refresh
    the cache. Because ``get_cached_description`` only generates when
    description.txt is ABSENT, a value written here is never auto-regenerated —
    the rename sticks across later turns."""
    with DESCRIPTION_LOCK:
        _write_description(tid, description)


def set_description_if_absent(tid: str, description: str) -> bool:
    """Persist an initial title only when no user or system title exists yet."""
    with DESCRIPTION_LOCK:
        path = os.path.join(MANAGER.thread_dir(tid), "description.txt")
        if os.path.exists(path):
            return False
        try:
            _write_description(tid, description)
        except FileNotFoundError:
            return False  # deletion won; never recreate a thread for its title
        return True


def _write_description(tid: str, description: str) -> None:
    """Write one caller-authorized title while holding ``DESCRIPTION_LOCK``."""
    _atomic_write(os.path.join(MANAGER.thread_dir(tid), "description.txt"), description,
                  create_parent=False)
    DESCRIPTION_CACHE[tid] = description


def _recover_interrupted_threads() -> None:
    """Worker-thread startup migration + recovery queueing.

    Run-store locks, legacy-journal reads, and checkpoint inspection are blocking, so
    lifespan invokes this function through ``run_in_threadpool`` before serving. Busy
    status is first projected to paused so a live submit cannot outrun the recovered
    head; ``queue_recovery_runs`` then imports legacy tickets and queues run IDs in
    dependency order.
    """
    from manage.web.threads import queue_recovery_runs
    for tid in MANAGER.list():
        status = _get_status(tid)
        if status.get("stage") in BUSY_STAGES:
            _set_status(tid, "paused",
                        **{k: v for k, v in status.items() if k != "stage"})
    queue_recovery_runs()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure thread root exists at startup
    os.makedirs(ROOT, exist_ok=True)

    # Recover threads a previous server run left busy, instead of erroring them
    # (extracted so the test pins the REAL scan, not a copy).
    from manage.web.threads import start_scheduler, stop_scheduler
    from manage.voice.service import VoiceConfig, VoiceService
    from manage.voice.wire import configure_call_runner
    try:
        config = VoiceConfig.from_environ()
    except ValueError as exc:
        logging.getLogger(__name__).error("voice runner disabled: %s", exc)
        config = None
    configure_call_runner(VoiceService(config) if config else None)
    await run_in_threadpool(_recover_interrupted_threads)

    # Populate description cache at startup
    for tid in MANAGER.list():
        get_cached_description(tid)
    # Rebuild the "unseen AI response" badge cache from on-disk markers, so a
    # scheduled/SMS response's badge survives a restart.
    load_unseen_cache()
    load_urgent_cache()
    # Worker ownership and recovery scan touch the private capture filesystem.
    # Keep them out of the single-worker event loop just like shutdown below.
    await anyio.to_thread.run_sync(
        CAPTURE_WORKER.start, limiter=CAPTURE_THREAD_LIMITER)
    # Start the schedule poll loop + the serial worker that drains recovery jobs.
    start_scheduler()
    try:
        yield
    finally:
        configure_call_runner(None)
        try:
            stop_scheduler()
        except Exception:
            logging.getLogger(__name__).warning("scheduler shutdown failed", exc_info=True)
        try:
            await anyio.to_thread.run_sync(
                CAPTURE_WORKER.stop, limiter=CAPTURE_THREAD_LIMITER)
        except Exception:
            logging.getLogger(__name__).warning("capture worker shutdown failed", exc_info=True)
        # Clean up Docker sandbox containers
        try:
            SandboxManager.cleanup_all()
        except Exception:
            pass
        # Close shared resources (e.g., sqlite connection) to avoid leaks
        try:
            MANAGER.close()
        except Exception:
            pass
