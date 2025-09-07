import subprocess
import pytest
from assist.tools.filesystem import write_file


def init_repo(path):
    subprocess.run(["git", "init"], cwd=path, check=True, stdout=subprocess.PIPE)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Tester"], cwd=path, check=True)


def test_write_file_new(tmp_path):
    init_repo(tmp_path)
    target = tmp_path / "new.txt"
    write_file.invoke({"path": str(target), "content": "hello"})
    assert target.read_text() == "hello"


def test_write_file_existing_untracked(tmp_path):
    init_repo(tmp_path)
    target = tmp_path / "new.txt"
    target.write_text("a")
    with pytest.raises(ValueError):
        write_file.invoke({"path": str(target), "content": "b"})


def test_write_file_existing_tracked_overwrite(tmp_path):
    init_repo(tmp_path)
    target = tmp_path / "t.txt"
    target.write_text("old")
    subprocess.run(["git", "add", "t.txt"], cwd=tmp_path, check=True)
    write_file.invoke({"path": str(target), "content": "new", "overwrite": True})
    assert target.read_text() == "new"
