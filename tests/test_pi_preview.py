from __future__ import annotations

import os
import threading

import pytest

from assist.pi_preview import PiPreviewPolicy, PiPreviewUnavailable


def test_preview_is_off_by_default_and_deep_always_admits(tmp_path):
    policy = PiPreviewPolicy(tmp_path, health_admits=lambda: True)
    assert not policy.enabled()
    assert policy.admits("deepagents")
    assert not policy.admits("pi")


def test_enabled_preview_requires_health_and_never_falls_back(tmp_path):
    health = [False]
    policy = PiPreviewPolicy(tmp_path, health_admits=lambda: health[0])
    policy.set_enabled(True)
    assert policy.enabled()
    assert not policy.admits("pi")
    with pytest.raises(PiPreviewUnavailable):
        policy.require_admission("pi")
    policy.set_enabled(False)
    health[0] = True
    assert not policy.admits("pi")


def test_health_is_cached_only_while_the_setting_is_unchanged(tmp_path):
    calls = []
    policy = PiPreviewPolicy(tmp_path, health_admits=lambda: calls.append(1) or True)
    policy.set_enabled(True)
    assert policy.admits("pi")
    assert policy.admits("pi")
    assert calls == [1]
    policy.set_enabled(False)
    assert not policy.admits("pi")
    assert calls == [1]


def test_fresh_claim_check_bypasses_cached_health_and_requires_a_bool(tmp_path):
    health = [True]
    policy = PiPreviewPolicy(tmp_path, health_admits=lambda: health[0])
    policy.set_enabled(True)
    assert policy.admits("pi")
    health[0] = "unhealthy"
    assert policy.admits("pi")  # page/form cache
    assert not policy.claim_admits("pi")


def test_disable_wins_over_an_inflight_fresh_claim_even_after_reenable(tmp_path):
    started = threading.Event()
    release = threading.Event()

    def health():
        started.set()
        assert release.wait(timeout=1)
        return True

    policy = PiPreviewPolicy(tmp_path, health_admits=health)
    policy.set_enabled(True)
    result = []
    worker = threading.Thread(target=lambda: result.append(policy.claim_admits("pi")))
    worker.start()
    assert started.wait(timeout=1)
    policy.set_enabled(False)
    policy.set_enabled(True)
    release.set()
    worker.join(timeout=1)
    assert not worker.is_alive()
    assert result == [False]


def test_unsafe_or_malformed_setting_disables_preview(tmp_path):
    policy = PiPreviewPolicy(tmp_path, health_admits=lambda: True)
    policy.set_enabled(True)
    os.chmod(policy.path, 0o666)
    assert not policy.enabled()
    policy.path.unlink()
    policy.path.write_text("not json")
    os.chmod(policy.path, 0o600)
    assert not policy.enabled()
    target = tmp_path / "target.json"
    target.write_text('{"enabled":true,"version":1}')
    policy.path.unlink()
    policy.path.symlink_to(target)
    assert not policy.enabled()


def test_unsafe_or_symlinked_state_root_disables_preview(tmp_path):
    root = tmp_path / "state"
    policy = PiPreviewPolicy(root, health_admits=lambda: True)
    policy.set_enabled(True)
    os.chmod(root, 0o777)
    assert not policy.enabled()
    os.chmod(root, 0o700)
    linked = PiPreviewPolicy(tmp_path / "linked", health_admits=lambda: True)
    (tmp_path / "linked").symlink_to(root, target_is_directory=True)
    assert not linked.enabled()
    with pytest.raises(PiPreviewUnavailable):
        linked.set_enabled(True)
