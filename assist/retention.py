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
import sqlite3
import subprocess
import sys
from typing import List

from assist.thread_manager import ThreadManager

logger = logging.getLogger(__name__)

DEFAULT_MIN_THREADS = 100

# Deleting a thread's rows in one transaction is unbounded for a huge orphan
# (2026-06-24: a 101GB / 24k-checkpoint thread thrashed a single delete for
# 75 min / 488GB read). Bound each transaction to this many rows.
_DELETE_BATCH = 2000


def _thread_dirs(threads_root: str) -> set[str]:
    """The on-disk thread directory names — the set of LIVE thread ids.

    One definition shared by ``prune_to_n_threads`` (which ranks these by
    mtime) and ``purge_orphaned_checkpoints`` (which treats any checkpoint
    thread_id NOT in this set as deletable); they must agree on what 'live'
    means."""
    if not os.path.isdir(threads_root):
        return set()
    return {
        name for name in os.listdir(threads_root)
        if name != "__pycache__"
        and os.path.isdir(os.path.join(threads_root, name))
    }


def _delete_thread_in_batches(conn, tid: str, batch: int = _DELETE_BATCH) -> None:
    """Delete a thread's ``checkpoints`` + ``writes`` rows in separately
    committed batches, so one giant orphan can't hold a single multi-GB
    transaction. ``thread_id`` leads both tables' primary key, so the LIMIT
    subquery is an index range scan."""
    for table in ("checkpoints", "writes"):
        while True:
            cur = conn.execute(
                f"DELETE FROM {table} WHERE rowid IN "
                f"(SELECT rowid FROM {table} WHERE thread_id = ? LIMIT ?)",
                (tid, batch),
            )
            conn.commit()
            if cur.rowcount < batch:
                break


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
    for name in _thread_dirs(threads_root):
        try:
            mtime = os.path.getmtime(os.path.join(threads_root, name))
        except OSError as e:
            logger.warning("Skipping %s: stat failed: %s", name, e)
            continue
        candidates.append((name, mtime))

    # Tiebreak by name: candidates come from a set (unstable iteration order),
    # so equal-mtime dirs must sort deterministically or the prune boundary
    # could keep different threads across runs.
    candidates.sort(key=lambda x: (x[1], x[0]), reverse=True)

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


def purge_orphaned_checkpoints(
    threads_root: str,
    manager: ThreadManager,
) -> List[str]:
    """Delete checkpoint+writes rows whose ``thread_id`` has no live thread dir.

    The web app derives a thread's directory name AND its checkpoint
    ``thread_id`` from one id (``ThreadManager.get`` / ``new``), so every LIVE
    web thread has a matching dir and is never touched here. Orphans are
    checkpoint thread_ids with no dir: leftovers from threads deleted before
    checkpoint-purging existed, plus non-web invocations that share this db
    under their own ids (e.g. an ``AgentHarness`` run that defaults
    ``thread_id`` to a ``uuid``). These accumulate unseen by thread-count
    retention — the 2026-06-24 incident found 166 of 262 checkpoint thread_ids
    had no dir, one holding 101 GB across 24k checkpoints.

    (Sub-agent runs are NOT orphans: deepagents reuses the parent's
    ``thread_id`` with a derived ``checkpoint_ns``, so their rows carry the
    live parent's dir name and are kept. Enumeration is keyed on the
    ``checkpoints`` table; a thread_id present only in ``writes`` — not an
    observed state — is not swept.)

    Caller must have stopped concurrent writers (same contract as
    ``prune_to_n_threads``: ``main`` runs after the cron stops assist-web and
    behind the lsof foreign-writer guard). Deletes run in bounded batches so
    one giant orphan can't hold a multi-GB transaction. Returns the purged
    thread_ids.
    """
    live = _thread_dirs(threads_root)
    try:
        rows = manager.conn.execute(
            "SELECT DISTINCT thread_id FROM checkpoints"
        ).fetchall()
    except sqlite3.OperationalError as e:
        if "no such table" in str(e).lower():
            # Fresh db whose checkpoints table doesn't exist yet (a deploy's
            # first sweep before any thread) — nothing to purge.
            return []
        raise  # a real fault (locked/corrupt) must surface, not look like success
    orphans = sorted({row[0] for row in rows} - live)
    purged: list[str] = []
    for tid in orphans:
        try:
            # count(*) is an index-range count (thread_id leads the PK) — the
            # size signal without scanning the rows we're about to delete.
            n = manager.conn.execute(
                "SELECT count(*) FROM checkpoints WHERE thread_id = ?", (tid,)
            ).fetchone()[0]
            if n > 1000:
                logger.warning(
                    "Purging large orphan %s: %d checkpoints (batched delete)",
                    tid, n,
                )
            _delete_thread_in_batches(manager.conn, tid)
            purged.append(tid)
        except sqlite3.Error:
            raise  # a DB fault (locked/corrupt) must surface, not be masked
        except Exception as e:
            logger.warning("purge failed for orphan %s: %s", tid, e)
    logger.info(
        "Purged %d/%d orphaned checkpoint-threads (no live dir)",
        len(purged), len(orphans),
    )
    return purged


def _check_no_foreign_writers(db_path: str) -> None:
    """Abort with non-zero exit if any other process has ``db_path`` open.

    The cron path runs after ``systemctl stop assist-web``, but a
    stray dev server / REPL / ``make web`` could still hold the DB.
    Better to fail fast than corrupt mid-sweep.
    """
    if not os.path.exists(db_path):
        return  # Nothing to guard.

    # Try lsof first (richer output: PID, command, FD, mode); fall back
    # to fuser if lsof isn't installed.  Both are part of the standard
    # Linux toolset; psmisc (fuser) is on this host even though lsof
    # isn't.  Refuse to run if neither is present — the foreign-writer
    # guard is non-optional.
    try:
        result = subprocess.run(
            ["lsof", "--", db_path],
            capture_output=True,
            text=True,
            check=False,
        )
        # lsof returncode 1 = no holders (header-only or empty).
        no_holders_codes = {1}
    except FileNotFoundError:
        try:
            result = subprocess.run(
                ["fuser", db_path],
                capture_output=True,
                text=True,
                check=False,
            )
            # fuser returncode 1 = no holders.  stderr carries the
            # filename echo; stdout is the pid list.
            no_holders_codes = {1}
        except FileNotFoundError:
            print(
                "[retention] neither lsof nor fuser found on PATH; "
                "refusing to run without the foreign-writer guard.",
                file=sys.stderr,
            )
            sys.exit(2)
        # Normalize fuser output to lsof-shaped lines so the rest of
        # the function doesn't branch.  fuser stdout is "pid pid pid"
        # on a single line.
        if result.returncode == 0 and result.stdout.strip():
            pids = result.stdout.strip().split()
            self_pid = str(os.getpid())
            foreign = [pid for pid in pids if pid != self_pid]
            if foreign:
                print(
                    f"[retention] foreign processes hold {db_path}: "
                    f"{' '.join(foreign)} — refusing to sweep.",
                    file=sys.stderr,
                )
                sys.exit(2)
        elif result.returncode in no_holders_codes:
            return
        else:
            print(
                f"[retention] fuser returned unexpected code "
                f"{result.returncode}: {result.stderr.strip()}",
                file=sys.stderr,
            )
            sys.exit(2)
        return  # fuser branch handled the verdict; skip lsof parsing.

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
        purged = purge_orphaned_checkpoints(threads_root, manager)
    finally:
        manager.close()

    print(
        f"[retention] pruned {len(deleted)} threads; "
        f"purged {len(purged)} orphaned checkpoint-threads; "
        f"kept most-recent {min_threads}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
