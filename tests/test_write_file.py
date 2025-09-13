import subprocess
import tempfile
from pathlib import Path
import pytest
from assist.tools.filesystem import write_file_user, write_file_tmp


def init_repo(path):
    subprocess.run(["git", "init"], cwd=path, check=True, stdout=subprocess.PIPE)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Tester"], cwd=path, check=True)


def test_write_file_user_new(tmp_path):
    init_repo(tmp_path)
    target = tmp_path / "new.txt"
    write_file_user.invoke({"path": str(target), "content": "hello"})
    assert target.read_text() == "hello"


def test_write_file_user_existing_untracked(tmp_path):
    init_repo(tmp_path)
    target = tmp_path / "new.txt"
    target.write_text("a")
    with pytest.raises(ValueError):
        write_file_user.invoke({"path": str(target), "content": "b"})


def test_write_file_user_existing_tracked_overwrite(tmp_path):
    init_repo(tmp_path)
    target = tmp_path / "t.txt"
    target.write_text("old")
    subprocess.run(["git", "add", "t.txt"], cwd=tmp_path, check=True)
    write_file_user.invoke({"path": str(target), "content": "new", "overwrite": True})
    assert target.read_text() == "new"


def test_write_file_tmp_new():
    res = write_file_tmp.invoke({"path": "tmp.txt", "content": "hi"})
    target = Path(res.removeprefix("Wrote ").strip())
    assert target.read_text() == "hi"
    assert target.is_absolute()
    temp_root = Path(tempfile.gettempdir()).resolve()
    assert temp_root in target.parents


def test_write_file_tmp_overwrite():
    res = write_file_tmp.invoke({"path": "tmp.txt", "content": "old"})
    target = Path(res.removeprefix("Wrote ").strip())
    with pytest.raises(ValueError):
        write_file_tmp.invoke({"path": str(target), "content": "new"})
    write_file_tmp.invoke({"path": str(target), "content": "new", "overwrite": True})
    assert target.read_text() == "new"
