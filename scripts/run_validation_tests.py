#!/usr/bin/env python3
"""Run validation tests repeatedly and summarize results."""

from __future__ import annotations

import argparse
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Tuple

TEST_PATH = Path("tests/integration/validation")


def parse_results(output: str) -> Iterable[Tuple[str, str]]:
    """Yield (test_name, status) pairs from pytest output."""
    pattern = re.compile(r"^(tests[^\s]+) (PASSED|FAILED|ERROR|SKIPPED)", re.MULTILINE)
    for match in pattern.finditer(output):
        yield match.group(1), match.group(2)


def run_once(cwd: Path) -> Iterable[Tuple[str, str]]:
    """Run the validation tests once and return parsed results."""
    proc = subprocess.run(
        ["pytest", str(TEST_PATH), "-vv"],
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return parse_results(proc.stdout)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run validation tests multiple times and report pass rates."
    )
    parser.add_argument(
        "runs",
        nargs="?",
        type=int,
        default=10,
        help="Number of times to run the validation tests",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"pass": 0, "fail": 0})

    for _ in range(args.runs):
        for test_name, status in run_once(root):
            key = "pass" if status == "PASSED" else "fail"
            stats[test_name][key] += 1

    for test_name, counts in sorted(stats.items()):
        total = counts["pass"] + counts["fail"]
        pct = (counts["pass"] / total * 100) if total else 0.0
        print(
            f"{test_name}: {counts['pass']} pass, {counts['fail']} fail, {pct:.1f}% pass"
        )


if __name__ == "__main__":
    main()
