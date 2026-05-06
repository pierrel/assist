"""manage.web package — FastAPI web UI for the assist agent.

Composed from sibling modules:

- ``app``    — the ``FastAPI`` instance (separated so route modules can
               import it without a circular dep on this ``__init__``).
- ``state``  — module-level config (``MANAGER``, ``DOMAINS``, caches),
               status helpers, lifespan.
- ``diff``   — per-file diff rendering with the inline cap, used by
               both the thread page and the review page.
- ``review`` — ``/thread/{tid}/review`` page + GET/POST routes.
- ``threads``— index, thread page, and the message/capture/merge/delete
               routes.  Owns ``_process_message``.
- ``evals``  — ``/evals`` and ``/evals/run/{id}`` routes.

Importing ``manage.web`` triggers route registration on ``app`` via the
submodule imports below.  ``uvicorn manage.web:app`` continues to work.
"""
from manage.web.app import app

# Importing the route modules registers their endpoints on ``app``.
# Order matters only insofar as ``review`` imports from ``threads``.
from manage.web import threads, review, evals  # noqa: E402,F401

# Re-export the names that external scripts (and any direct
# ``from manage.web import X`` consumers) historically relied on, so
# the package boundary is invisible from outside.  Tests that need to
# *patch* these names should target the module that defines them
# (e.g., ``manage.web.threads._process_message``); patching the
# re-export here only affects this namespace.
from manage.web.state import (  # noqa: E402,F401
    BUSY_STAGES,
    DESCRIPTION_CACHE,
    DOMAIN_MANAGERS,
    DOMAINS,
    INIT_STAGES,
    MANAGER,
    ROOT,
    STAGE_LABELS,
    _domain_label,
    _domain_selector_html,
    _evict_caches,
    _get_domain_manager,
    _get_sandbox_backend,
    _get_status,
    _set_status,
    _thread_domain_html,
    _thread_title,
    get_cached_description,
    lifespan,
)
from manage.web.diff import (  # noqa: E402,F401
    INLINE_FILE_BYTE_CAP,
    INLINE_FILE_LINE_CAP,
    INLINE_TOTAL_BYTE_CAP,
    _classify_diff_line,
    _diff_stats,
    _is_binary_diff,
    _rename_pair,
    render_file_diff,
)
from manage.web.review import (  # noqa: E402,F401
    _format_review_message,
    _REVIEW_HEADER,
    _REVIEW_OPENER,
    render_review_page,
)
from manage.web.threads import (  # noqa: E402,F401
    _capture_conversation,
    _initialize_thread,
    _process_message,
    render_index,
    render_thread,
)


if __name__ == "__main__":
    import os
    import uvicorn

    os.makedirs(ROOT, exist_ok=True)
    port = int(os.getenv("ASSIST_PORT", "8000"))
    uvicorn.run("manage.web:app", host="0.0.0.0", port=port, log_level="info", reload=False)
