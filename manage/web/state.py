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
from contextlib import asynccontextmanager
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Dict

from fastapi import FastAPI

from assist.domain_manager import DomainManager
from assist.env import load_dev_env
from assist.sandbox_manager import SandboxManager
from assist.thread import ThreadManager


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
_raw = os.getenv("ASSIST_DOMAINS", "")
DOMAINS: list[str] = [d.strip() for d in _raw.split(",") if d.strip()]
DESCRIPTION_CACHE: Dict[str, str] = {}
DOMAIN_MANAGERS: Dict[str, DomainManager] = {}  # tid -> DomainManager


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
    """
    if tid in DOMAIN_MANAGERS:
        return DOMAIN_MANAGERS[tid]
    twdir = MANAGER.thread_default_working_dir(tid)
    try:
        dm = DomainManager(twdir, domain)
        DOMAIN_MANAGERS[tid] = dm
        return dm
    except Exception:
        return None


def _get_sandbox_backend(tid: str):
    """Get sandbox backend for a thread, or None if Docker is unavailable."""
    work_dir = MANAGER.thread_default_working_dir(tid)
    return SandboxManager.get_sandbox_backend(work_dir)


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
    path = _status_path(tid)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {"stage": stage, **kwargs}
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, path)


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

    # Cache miss - read from FS or thread and cache
    try:
        chat = MANAGER.get(tid)
        thread_dir = MANAGER.thread_dir(tid)
        description_file = os.path.join(thread_dir, "description.txt")
        if os.path.isfile(description_file):
            with open(description_file, 'r') as f:
                description = f.read()
        else:
            description = chat.description()
            os.makedirs(os.path.dirname(description_file), exist_ok=True)
            with open(description_file, 'w') as f:
                f.write(description)
        DESCRIPTION_CACHE[tid] = description
        return description
    except Exception:
        return tid


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
    try:
        yield
    finally:
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
