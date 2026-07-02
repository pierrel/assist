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
import sys
import tempfile
import threading
from contextlib import asynccontextmanager
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Dict

from fastapi import FastAPI

from assist.domain_manager import DomainManager
from assist.env import load_dev_env
from assist.sandbox_manager import SandboxManager
from assist.schedule.store import ScheduleStore
from assist.schedule.tools import schedule_tools
from assist.events.store import SubscriptionStore
from assist.events.tools import subscription_tools
from assist.events.reply import reply_tools, REPLY_INTERRUPT_ON
from assist.thread_manager import ThreadManager, set_web_tools, set_web_interrupt_on


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

# The one shared schedule store (disk-as-truth on the thread root) — used by the
# Scheduler (started in the lifespan, see threads.py) AND the schedule tools AND the
# /schedules view, so they share its single read-modify-write lock. The web app is a
# single uvicorn worker, so exactly one Scheduler/store pair exists.
SCHEDULE_STORE = ScheduleStore(ROOT)
# The subscription store shares the thread root the same way (disk-as-truth, per-thread).
# Subscription tools let the agent set up message-event triage; the inbound-SMS route reads
# the same store to route a message to its subscription's thread.
SUBSCRIPTION_STORE = SubscriptionStore(ROOT)
# send_reply rides the web tools too but is HITL-gated (see REPLY_INTERRUPT_ON, applied in
# the web AgentSpec) — an inbound-message triage turn proposes a reply; the user approves.
set_web_tools(schedule_tools(SCHEDULE_STORE)
              + subscription_tools(SUBSCRIPTION_STORE)
              + reply_tools())
set_web_interrupt_on(REPLY_INTERRUPT_ON)
_raw = os.getenv("ASSIST_DOMAINS", "")
DOMAINS: list[str] = [d.strip() for d in _raw.split(",") if d.strip()]
DESCRIPTION_CACHE: Dict[str, str] = {}
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


def _get_sandbox_backend(tid: str, tz: str | None = None):
    """Get sandbox backend for a thread, or None if Docker is unavailable.

    ``tz`` is the per-turn context-rider timezone, so this turn's sandbox ``date``
    runs in the user's local time (else the host/server zone).

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
    return SandboxManager.get_sandbox_backend(work_dir, tz=tz)


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


def _atomic_write(path: str, data: str) -> None:
    """Write ``data`` to ``path`` atomically: a uniquely-named temp file in the
    same directory + os.replace. The unique temp name means concurrent writers
    for the same path can't clobber a shared temp or leave a half-written file."""
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


# --- Thread status tracking ----------------------------------------------
# Stages used for the async thread creation flow:
#   initializing      - thread row created, background task queued
#   cloning           - git clone in progress
#   starting_sandbox  - docker container starting
#   queued            - waiting for another thread's LLM affinity hold
#   processing        - agent is running on a message
#   ready             - idle, accepting input
#   error             - something failed; see status["error"]
INIT_STAGES = {"initializing", "cloning", "starting_sandbox"}
BUSY_STAGES = INIT_STAGES | {"processing", "queued"}

STAGE_LABELS = {
    "initializing": "Setting up thread...",
    "cloning": "Cloning repository...",
    "starting_sandbox": "Starting sandbox container...",
    "queued": "Waiting for another thread to finish...",
    "processing": "Processing your message...",
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


def _thread_title(tid: str) -> str:
    """Display title for a thread; placeholder if still initializing."""
    status = _get_status(tid)
    if status.get("stage") in BUSY_STAGES:
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
    _atomic_write(os.path.join(MANAGER.thread_dir(tid), "description.txt"), description)
    DESCRIPTION_CACHE[tid] = description


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure thread root exists at startup
    os.makedirs(ROOT, exist_ok=True)

    # Recover any threads left mid-init by a previous server crash.
    # Their background task is no longer running, so mark them errored
    # so the user gets feedback instead of a forever-spinning page.
    for tid in MANAGER.list():
        status = _get_status(tid)
        if status.get("stage") in BUSY_STAGES:
            _set_status(
                tid,
                "error",
                error="Server restarted while this thread was being set up.",
                pending_message=status.get("pending_message", ""),
            )

    # Populate description cache at startup
    for tid in MANAGER.list():
        get_cached_description(tid)
    # Start the schedule poll loop (local import breaks the state<->threads cycle).
    from manage.web.threads import start_scheduler, stop_scheduler
    start_scheduler()
    try:
        yield
    finally:
        try:
            stop_scheduler()
        except Exception:
            logging.getLogger(__name__).warning("scheduler shutdown failed", exc_info=True)
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
