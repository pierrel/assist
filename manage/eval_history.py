"""Eval history parsing and caching for the web UI.

Scans edd/history/ for JUnit XML result files, caches parsed data to
edd/history/.cache.json, and returns structured run/test data.
"""

import json
import os
import xml.etree.ElementTree as ET

HISTORY_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "edd", "history")
CACHE_FILE = os.path.join(HISTORY_DIR, ".cache.json")


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
    run_id = filename.removeprefix("results-").removesuffix(".xml")
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

    Runs are sorted by ID (which encodes date-time), most-recent first — so
    the web grid shows newest runs on the left and the user doesn't have to
    scroll right to see the latest results.
    """
    cache = _load_cache()
    cached_by_file: dict[str, dict] = {r["file"]: r for r in cache.get("runs", [])}

    try:
        xml_files = sorted(
            f for f in os.listdir(HISTORY_DIR)
            if f.startswith("results-") and f.endswith(".xml")
        )
    except FileNotFoundError:
        return []

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

    all_runs = sorted(cached_by_file.values(), key=lambda r: r["id"])

    if changed:
        cache["runs"] = all_runs
        _save_cache(cache)

    return all_runs[-limit:][::-1]
