"""Cross-turn auth state stays private, bounded and generation-owned."""
import os

import pytest

from assist.browser import authority, storage


@pytest.fixture
def owner(tmp_path):
    thread_id = "thread"
    (tmp_path / thread_id).mkdir()
    authority.mark_new_thread(str(tmp_path), thread_id)
    with authority.fence(str(tmp_path), thread_id) as current:
        current.begin("run", 1)
        current.add_generation("run", "sandbox-generation")
    return str(tmp_path), thread_id


def _state(value="secret"):
    return {"cookies": [{"name": "session", "value": value}], "origins": []}


def test_storage_is_private_and_old_owner_cannot_overwrite(owner):
    root, thread_id = owner
    storage.save(root, thread_id, "run", "sandbox-generation", _state())
    path = os.path.join(root, thread_id, storage.FILE)
    assert os.stat(path).st_mode & 0o777 == 0o600
    assert storage.load(root, thread_id) == _state()
    with pytest.raises(RuntimeError, match="owner changed"):
        storage.save(root, thread_id, "older-run", "sandbox-generation",
                     _state("replacement"))
    assert storage.load(root, thread_id) == _state()


def test_storage_rejects_unsafe_file_and_oversized_state(owner, tmp_path):
    root, thread_id = owner
    path = tmp_path / thread_id / storage.FILE
    path.symlink_to(tmp_path / "outside")
    with pytest.raises(OSError):
        storage.load(root, thread_id)
    path.unlink()
    storage.save(root, thread_id, "run", "sandbox-generation", _state())
    path.chmod(0o644)
    with pytest.raises(ValueError, match="unsafe"):
        storage.load(root, thread_id)
    with pytest.raises(ValueError, match="exceeds limit"):
        storage.save(root, thread_id, "run", "sandbox-generation",
                     _state("x" * storage.MAX_BYTES))


def test_storage_does_not_recreate_deleted_thread(owner, tmp_path):
    root, thread_id = owner
    (tmp_path / thread_id / authority.STATE_FILE).unlink()
    (tmp_path / thread_id / authority.LOCK_FILE).unlink()
    (tmp_path / thread_id).rmdir()
    with pytest.raises(RuntimeError, match="directory unavailable"):
        storage.save(root, thread_id, "run", "sandbox-generation", _state())
    assert not (tmp_path / thread_id).exists()
