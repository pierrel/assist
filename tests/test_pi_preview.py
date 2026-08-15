from __future__ import annotations

import os

import pytest

from assist.pi_preview import PiPreviewPolicy, PiPreviewUnavailable


def test_preview_is_off_by_default_and_deep_always_admits(tmp_path):
    policy = PiPreviewPolicy(tmp_path)

    assert not policy.enabled()
    assert policy.admits("deepagents")
    assert not policy.admits("pi")


def test_enabled_preview_admits_pi_until_the_operator_disables_it(tmp_path):
    policy = PiPreviewPolicy(tmp_path)

    policy.set_enabled(True)
    assert policy.enabled()
    assert policy.admits("pi")

    policy.set_enabled(False)
    assert not policy.admits("pi")


def test_unsafe_or_malformed_setting_disables_preview(tmp_path):
    policy = PiPreviewPolicy(tmp_path)
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
    policy = PiPreviewPolicy(root)
    policy.set_enabled(True)
    os.chmod(root, 0o777)
    assert not policy.enabled()
    os.chmod(root, 0o700)
    linked = PiPreviewPolicy(tmp_path / "linked")
    (tmp_path / "linked").symlink_to(root, target_is_directory=True)
    assert not linked.enabled()
    with pytest.raises(PiPreviewUnavailable):
        linked.set_enabled(True)
