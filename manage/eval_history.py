"""Eval history parsing and caching for the web UI.

Scans edd/history/ for JUnit XML result files, caches parsed data to
edd/history/.cache.json, and returns structured run/test data.

Two file naming conventions are supported, both grouped by the
``YYYYMMDD-HHMM`` timestamp suffix:

- ``results-YYYYMMDD-HHMM.xml`` — legacy single-file format from the
  pre-2026-05-03 ``make eval`` (one combined JUnit XML per run).
- ``<base>-YYYYMMDD-HHMM.xml`` — per-file format from the current
  ``scripts/run-evals.sh`` (one JUnit XML per test file in the suite).
  Multiple files share a run id; this module aggregates them on read.
"""

import json
import os
import re
import xml.etree.ElementTree as ET

HISTORY_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "edd", "history")
CACHE_FILE = os.path.join(HISTORY_DIR, ".cache.json")

# Match either `results-YYYYMMDD-HHMM.xml` or `<base>-YYYYMMDD-HHMM.xml`.
# Group 1 is the prefix (test-file label or "results"); group 2 is the
# run id used to aggregate per-file XMLs into a single run.
_RUN_ID_RE = re.compile(r"^(.+?)-(\d{8}-\d{4})\.xml$")


def _run_id_from_filename(filename: str) -> str | None:
    m = _RUN_ID_RE.match(filename)
    return m.group(2) if m else None


def _parse_xml(filepath: str) -> dict:
    """Parse a JUnit XML file into a run dict."""
    tree = ET.parse(filepath)
    root = tree.getroot()

    suites = root.findall("testsuite") if root.tag == "testsuites" else [root]

    timestamp = None
    tests: dict[str, dict] = {}

    for suite in suites:
        if timestamp is None:
            timestamp = suite.get("timestamp")
        for tc in suite.findall("testcase"):
            classname = tc.get("classname", "")
            name = tc.get("name", "")
            time_val = float(tc.get("time", "0") or "0")
            key = f"{classname}::{name}" if classname else name

            failure = tc.find("failure")
            error = tc.find("error")
            skipped = tc.find("skipped")

            if failure is not None:
                status = "failed"
                message = failure.get("message", "") or ""
                details = failure.text or ""
            elif error is not None:
                status = "error"
                message = error.get("message", "") or ""
                details = error.text or ""
            elif skipped is not None:
                status = "skipped"
                message = skipped.get("message", "") or ""
                details = skipped.text or ""
            else:
                status = "passed"
                message = ""
                details = ""

            tests[key] = {
                "status": status,
                "time": time_val,
                "message": message,
                "details": details,
            }

    filename = os.path.basename(filepath)
    run_id = _run_id_from_filename(filename) or filename.removesuffix(".xml")
    return {
        "id": run_id,
        "timestamp": timestamp or "",
        "file": filename,
        "file_mtime": os.path.getmtime(filepath),
        "tests": tests,
    }


def _load_cache() -> dict:
    if os.path.isfile(CACHE_FILE):
        try:
            with open(CACHE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {"runs": []}


def _save_cache(cache: dict) -> None:
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


def get_runs(limit: int = 10) -> list[dict]:
    """Return the last `limit` runs, refreshing the cache for any new/changed files.

    Per-file XMLs that share a ``YYYYMMDD-HHMM`` run id are aggregated
    into a single run dict — tests merged across files, file_mtime is
    the latest of the group.  Runs are sorted by id (which encodes
    date-time), most-recent first.
    """
    cache = _load_cache()
    cached_by_file: dict[str, dict] = {r["file"]: r for r in cache.get("runs", [])}

    try:
        xml_files = sorted(
            f for f in os.listdir(HISTORY_DIR)
            if _run_id_from_filename(f) is not None
        )
    except FileNotFoundError:
        return []

    # Drop cache entries for files that no longer exist on disk.
    cached_by_file = {f: r for f, r in cached_by_file.items() if f in set(xml_files)}

    changed = False
    for filename in xml_files:
        filepath = os.path.join(HISTORY_DIR, filename)
        try:
            mtime = os.path.getmtime(filepath)
        except OSError:
            continue
        existing = cached_by_file.get(filename)
        if existing is None or existing.get("file_mtime", 0) != mtime:
            try:
                run = _parse_xml(filepath)
                cached_by_file[filename] = run
                changed = True
            except Exception:
                pass

    if changed:
        cache["runs"] = list(cached_by_file.values())
        _save_cache(cache)

    # Aggregate per-file entries by run id.  Multiple per-file XMLs from
    # one nightly run share the same id and merge into one run dict.
    by_id: dict[str, dict] = {}
    for entry in cached_by_file.values():
        run_id = entry["id"]
        if run_id not in by_id:
            by_id[run_id] = {
                "id": run_id,
                "timestamp": entry.get("timestamp", ""),
                "file": entry["file"],  # representative file (for legacy display)
                "file_mtime": entry.get("file_mtime", 0),
                "tests": {},
            }
        by_id[run_id]["tests"].update(entry["tests"])
        by_id[run_id]["file_mtime"] = max(
            by_id[run_id]["file_mtime"], entry.get("file_mtime", 0)
        )
        # Prefer earliest non-empty timestamp seen for the run.
        if not by_id[run_id]["timestamp"] and entry.get("timestamp"):
            by_id[run_id]["timestamp"] = entry["timestamp"]

    all_runs = sorted(by_id.values(), key=lambda r: r["id"])
    return all_runs[-limit:][::-1]
