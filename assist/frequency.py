"""Stable, occasional maintenance decisions for one visible web Run.

The model asks whether a named maintenance policy should run on this Run.  The
host chooses once, persists the answer, and returns it unchanged on retries so
an agent cannot sample repeatedly until it gets the answer it wants.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
from dataclasses import dataclass

from langgraph.config import get_config


FREQUENCY_RUN_ID_KEY = "frequency_run_id"
_POLICY_RE = re.compile(r"[a-z][a-z0-9-]{0,63}\Z")
_VERSION = "v1"


@dataclass(frozen=True)
class FrequencyDecision:
    policy: str
    probability: float
    should_run: bool


class FrequencyDecisionStore:
    """Durable, one-policy decisions for visible Runs in each thread."""

    FILENAME = "frequency-decisions.json"

    def __init__(self, root_dir: str):
        self._root = root_dir
        self._lock = threading.Lock()

    @staticmethod
    def _part(value: str, name: str) -> str:
        if not value or value in {".", ".."} or os.sep in value \
                or (os.altsep and os.altsep in value):
            raise ValueError(f"invalid {name}")
        return value

    def _path(self, thread_id: str) -> str:
        return os.path.join(self._root, self._part(thread_id, "thread ID"),
                            self.FILENAME)

    @staticmethod
    def _validate(policy: str, probability: float) -> None:
        if not isinstance(policy, str) or not _POLICY_RE.fullmatch(policy):
            raise ValueError("policy must be a short lowercase hyphenated name")
        if isinstance(probability, bool) or not isinstance(probability, (int, float)) \
                or not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("probability must be a finite number from 0 through 1")

    def decide(self, thread_id: str, run_id: str, policy: str,
               probability: float) -> FrequencyDecision:
        """Return one stable decision, rejecting changed policy inputs on retry."""
        self._validate(policy, probability)
        run_id = self._part(run_id, "run ID")
        key = f"{run_id}:{policy}"
        path = self._path(thread_id)
        probability = float(probability)
        with self._lock:
            try:
                with open(path, encoding="utf-8") as handle:
                    decisions = json.load(handle)
            except (FileNotFoundError, json.JSONDecodeError):
                decisions = {}
            existing = decisions.get(key)
            if existing is not None:
                if existing["probability"] != probability:
                    raise ValueError(
                        "this policy already has a decision for this run; "
                        "reuse its original probability")
                return FrequencyDecision(
                    policy=policy, probability=probability,
                    should_run=bool(existing["should_run"]))
            if any(saved_key.startswith(f"{run_id}:") for saved_key in decisions):
                raise ValueError(
                    "this run already has a maintenance decision; do not try "
                    "another policy")
            digest = hashlib.sha256(
                f"{_VERSION}\0{thread_id}\0{run_id}\0{policy}".encode()).digest()
            threshold = int.from_bytes(digest[:8], "big") / 2**64
            decision = FrequencyDecision(
                policy=policy, probability=probability,
                should_run=threshold < probability)
            decisions[key] = {
                "probability": probability,
                "should_run": decision.should_run,
            }
            os.makedirs(os.path.dirname(path), exist_ok=True)
            temporary = f"{path}.{os.getpid()}.tmp"
            with open(temporary, "w", encoding="utf-8") as handle:
                json.dump(decisions, handle, sort_keys=True)
            os.replace(temporary, path)
            return decision


def _config_value(name: str) -> str | None:
    value = ((get_config() or {}).get("configurable") or {}).get(name)
    return value if isinstance(value, str) else None


def frequency_tools(store: FrequencyDecisionStore) -> list:
    """Return the visible web tool for policy-guided occasional maintenance."""

    def should_run_maintenance(policy: str, probability: float) -> str:
        """Decide once whether this named occasional maintenance runs on this turn.

        Call only from the loaded maintenance skill. Use the returned answer;
        do not retry with another probability or policy on this turn.
        """
        thread_id = _config_value("thread_id")
        run_id = _config_value(FREQUENCY_RUN_ID_KEY)
        if not thread_id or not run_id:
            return "Maintenance decisions are available only during an ordinary web run."
        if policy != "thread-checkpoint":
            return "Use the thread-checkpoint policy for this maintenance decision."
        try:
            decision = store.decide(thread_id, run_id, policy, probability)
        except ValueError as error:
            return f"Couldn't decide maintenance: {error}."
        return ("Run this maintenance now." if decision.should_run
                else "Skip this maintenance for this run.")

    return [should_run_maintenance]
