from __future__ import annotations

import hashlib
import os
import tempfile
import subprocess
import unittest
from pathlib import Path

from pi_runtime_p0 import (
    MAX_SESSION_BYTES,
    ActiveWorkLedger,
    P0Error,
    ProviderAdmission,
    ProviderBinding,
    open_session_candidate,
    recover_verified_suffix,
)


class SessionPreflightTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="assist-pi-session-test-")
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write(self, name: str, data: bytes, mode: int = 0o600) -> Path:
        path = self.root / name
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
        try:
            os.write(descriptor, data)
        finally:
            os.close(descriptor)
        return path

    def test_accepts_empty_and_bounded_typed_jsonl(self) -> None:
        empty = self.write("empty.jsonl", b"")
        descriptor = open_session_candidate(empty)
        os.close(descriptor)
        valid = self.write(
            "valid.jsonl",
            b'{"type":"session","version":3}\n'
            b'{"type":"message","id":"one"}\n',
        )
        descriptor = open_session_candidate(valid)
        os.close(descriptor)

    def test_rejects_symlink_hardlink_nonregular_and_wrong_mode(self) -> None:
        target = self.write("target.jsonl", b'{"type":"session"}\n')
        symlink = self.root / "symlink.jsonl"
        symlink.symlink_to(target)
        with self.assertRaises(OSError):
            open_session_candidate(symlink)

        hardlink = self.root / "hardlink.jsonl"
        os.link(target, hardlink)
        with self.assertRaisesRegex(P0Error, "multiple links"):
            open_session_candidate(target)

        fifo = self.root / "fifo"
        os.mkfifo(fifo, 0o600)
        with self.assertRaisesRegex(P0Error, "not a regular"):
            open_session_candidate(fifo)

        wrong = self.write("wrong.jsonl", b'{"type":"session"}\n', 0o640)
        with self.assertRaisesRegex(P0Error, "0600"):
            open_session_candidate(wrong)

    def test_rejects_malformed_oversized_duplicate_and_truncated_data(self) -> None:
        fixtures = (
            ("malformed.jsonl", b'{"type":\n', "malformed"),
            ("oversized.jsonl", b"x" * (MAX_SESSION_BYTES + 1), "size bound"),
            (
                "duplicate.jsonl",
                b'{"type":"message","id":"same"}\n'
                b'{"type":"message","id":"same"}\n',
                "duplicate",
            ),
            ("truncated.jsonl", b'{"type":"session"}', "truncated"),
        )
        for name, data, message in fixtures:
            with self.subTest(name=name):
                path = self.write(name, data)
                with self.assertRaisesRegex(P0Error, message):
                    open_session_candidate(path)

    def test_suffix_recovery_requires_the_exact_committed_prefix(self) -> None:
        prefix = b'{"type":"session"}\n'
        suffix = b'{"type":"partial"'
        path = self.write("recover.jsonl", prefix + suffix)
        descriptor = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
        try:
            recover_verified_suffix(
                descriptor,
                len(prefix),
                hashlib.sha256(prefix).hexdigest(),
            )
            self.assertEqual(os.pread(descriptor, len(prefix), 0), prefix)
            with self.assertRaisesRegex(P0Error, "outside"):
                recover_verified_suffix(descriptor, len(prefix) + 1, "0" * 64)
            with self.assertRaisesRegex(P0Error, "no longer matches"):
                recover_verified_suffix(descriptor, len(prefix), "0" * 64)
        finally:
            os.close(descriptor)


class ProviderAdmissionTest(unittest.TestCase):
    def binding(self, **changes: str) -> ProviderBinding:
        values = {
            "run_id": "run",
            "release": "release",
            "model": "model",
            "profile_digest": "profile",
            "request_digest": "request",
            **changes,
        }
        return ProviderBinding(**values)

    def test_lease_is_exact_one_use_and_capacity_one(self) -> None:
        admission = ProviderAdmission()
        binding = self.binding()
        lease = admission.issue(binding)
        admission.admit(lease, binding)
        with self.assertRaisesRegex(P0Error, "unavailable"):
            admission.issue(binding)
        admission.observe_terminal(lease)
        with self.assertRaisesRegex(P0Error, "consumed"):
            admission.admit(lease, binding)

    def test_binding_and_generations_fail_closed(self) -> None:
        admission = ProviderAdmission()
        lease = admission.issue(self.binding())
        with self.assertRaisesRegex(P0Error, "binding"):
            admission.admit(lease, self.binding(model="other"))
        lease = admission.issue(self.binding())
        admission.gateway_restart()
        with self.assertRaisesRegex(P0Error, "consumed"):
            admission.admit(lease, self.binding())

    def test_cancel_requires_terminal_idle_or_bounded_server_restart(self) -> None:
        admission = ProviderAdmission()
        binding = self.binding()
        lease = admission.issue(binding)
        admission.admit(lease, binding)
        admission.cancel(lease)
        with self.assertRaisesRegex(P0Error, "unavailable"):
            admission.issue(binding)
        with self.assertRaisesRegex(P0Error, "exact inactive"):
            admission.observe_single_slot_idle(True)
        admission.observe_single_slot_idle(False)
        next_lease = admission.issue(binding)
        admission.admit(next_lease, binding)
        admission.gateway_restart()
        with self.assertRaisesRegex(P0Error, "unavailable"):
            admission.issue(binding)
        admission.server_restart_complete()
        admission.admit(admission.issue(binding), binding)


class ActiveWorkLedgerTest(unittest.TestCase):
    def test_holder_releases_only_after_exit_and_cap_survives_reopen(self) -> None:
        with tempfile.TemporaryDirectory(prefix="assist-pi-ledger-test-") as directory:
            path = Path(directory) / "ledger.json"
            ledger = ActiveWorkLedger(path, 10)
            child = subprocess.Popen(["/usr/bin/sleep", "0.02"])
            ledger.acquire("first")
            with self.assertRaisesRegex(P0Error, "occupied"):
                ledger.acquire("second")
            with self.assertRaisesRegex(P0Error, "before child exit"):
                ledger.release_after_exit("first", 6, child.pid)
            child.wait(timeout=1)
            ledger.release_after_exit("first", 6, child.pid)
            reopened = ActiveWorkLedger(path, 10)
            reopened.acquire("second")
            exited = subprocess.Popen(["/usr/bin/true"])
            exited.wait(timeout=1)
            reopened.release_after_exit("second", 4, exited.pid)
            with self.assertRaisesRegex(P0Error, "cap is exhausted"):
                ActiveWorkLedger(path, 10).acquire("third")


if __name__ == "__main__":
    unittest.main()
