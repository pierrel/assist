"""Retention sweeper: keep the most-recent N threads, hard-delete the rest.

Layer 0 of the threads.db growth plan
(docs/2026-05-04-threads-db-layer-0-thread-retention.org).

Two entry points:

- ``prune_to_n_threads(threads_root, min_threads, manager)`` — pure
  Python; the caller (typically ``scripts/vacuum-prod-db.sh``) is
  expected to have already stopped any concurrent writers (e.g.
  ``systemctl stop assist-web``).
- ``__main__`` — CLI form for the cron path.  Reads
  ``ASSIST_THREADS_DIR`` and ``MIN_THREADS`` from env, runs an
  ``lsof`` guard against foreign writers on ``threads.db``, then
  prunes.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
from typing import List

from assist.thread import ThreadManager

logger = logging.getLogger(__name__)

DEFAULT_MIN_THREADS = 100


def prune_to_n_threads(
    threads_root: str,
    min_threads: int,
    manager: ThreadManager,
) -> List[str]:
    """Hard-delete threads beyond the most-recent ``min_threads`` by mtime.

    Walks ``threads_root`` once, ranks subdirectories by mtime
    descending, keeps the first ``min_threads``, and calls
    ``manager.hard_delete`` on each of the rest.  Returns the list of
    tids actually deleted (the order matches the deletion order).

    Errors during a single thread's hard_delete are logged and
    swallowed so one bad thread can't block the rest of the sweep —
    the caller still wants the VACUUM to run.
    """
    if min_threads < 0:
        raise ValueError(f"min_threads must be >= 0, got {min_threads}")

    if not os.path.isdir(threads_root):
        logger.warning(
            "threads_root %s does not exist; nothing to prune", threads_root
        )
        return []

    candidates: list[tuple[str, float]] = []
    for name in os.listdir(threads_root):
        dpath = os.path.join(threads_root, name)
        if not os.path.isdir(dpath) or name == "__pycache__":
            continue
        try:
            mtime = os.path.getmtime(dpath)
        except OSError as e:
            logger.warning("Skipping %s: stat failed: %s", dpath, e)
            continue
        candidates.append((name, mtime))

    candidates.sort(key=lambda x: x[1], reverse=True)

    if len(candidates) <= min_threads:
        logger.info(
            "Nothing to prune: %d threads <= MIN_THREADS=%d",
            len(candidates), min_threads,
        )
        return []

    to_delete = candidates[min_threads:]
    deleted: list[str] = []
    for tid, _mtime in to_delete:
        try:
            manager.hard_delete(tid)
            deleted.append(tid)
        except Exception as e:
            logger.warning("hard_delete failed for %s: %s", tid, e)
    logger.info(
        "Pruned %d/%d threads (kept most-recent %d)",
        len(deleted), len(candidates), min_threads,
    )
    return deleted


def _check_no_foreign_writers(db_path: str) -> None:
    """Abort with non-zero exit if any other process has ``db_path`` open.

    The cron path runs after ``systemctl stop assist-web``, but a
    stray dev server / REPL / ``make web`` could still hold the DB.
    Better to fail fast than corrupt mid-sweep.
    """
    if not os.path.exists(db_path):
        return  # Nothing to guard.

    try:
        result = subprocess.run(
            ["lsof", "--", db_path],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        # ``lsof`` is not installed.  Per AGENTS-style guidance, this
        # is a deploy-host hygiene problem; refuse to proceed.
        print(
            "[retention] lsof not found on PATH; refusing to run "
            "without the foreign-writer guard.",
            file=sys.stderr,
        )
        sys.exit(2)

    # lsof returns 1 when there are no holders — that's the safe case.
    if result.returncode == 1:
        return
    if result.returncode != 0:
        print(
            f"[retention] lsof returned unexpected code "
            f"{result.returncode}: {result.stderr.strip()}",
            file=sys.stderr,
        )
        sys.exit(2)

    # Filter the lsof header line and our own PID.
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if len(lines) <= 1:
        # Header-only.
        return

    own_pid = str(os.getpid())
    foreign = []
    for line in lines[1:]:
        # lsof default format: COMMAND PID USER FD TYPE DEVICE SIZE NODE NAME
        parts = line.split()
        if len(parts) < 2:
            continue
        if parts[1] == own_pid:
            continue
        foreign.append(line)

    if foreign:
        print(
            "[retention] refusing to run: other processes hold "
            f"{db_path}:",
            file=sys.stderr,
        )
        for line in foreign:
            print(f"  {line}", file=sys.stderr)
        sys.exit(3)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    threads_root = os.environ.get("ASSIST_THREADS_DIR")
    if not threads_root:
        print(
            "[retention] ASSIST_THREADS_DIR must be set",
            file=sys.stderr,
        )
        return 2

    try:
        min_threads = int(os.environ.get("MIN_THREADS", DEFAULT_MIN_THREADS))
    except ValueError:
        print(
            "[retention] MIN_THREADS must be an integer",
            file=sys.stderr,
        )
        return 2

    db_path = os.path.join(threads_root, "threads.db")
    _check_no_foreign_writers(db_path)

    manager = ThreadManager(threads_root)
    try:
        deleted = prune_to_n_threads(threads_root, min_threads, manager)
    finally:
        manager.close()

    print(
        f"[retention] pruned {len(deleted)} threads; "
        f"kept most-recent {min_threads}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
