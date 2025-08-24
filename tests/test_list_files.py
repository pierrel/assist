from assist.tools.filesystem import list_files


def test_list_files_respects_gitignore(tmp_path):
    visible = tmp_path / "visible.txt"
    visible.write_text("visible")
    ignored = tmp_path / "ignored.txt"
    ignored.write_text("ignored")
    sub = tmp_path / "sub"
    sub.mkdir()
    secret = sub / "secret.txt"
    secret.write_text("secret")

    gitignore = tmp_path / ".gitignore"
    gitignore.write_text("ignored.txt\nsub/\n")

    files, msg = list_files(str(tmp_path))
    result = "\n".join(files)
    assert str(visible) in result
    assert str(ignored) not in result
    assert str(secret) not in result
    assert msg is None


def test_list_files_uses_parent_gitignore(tmp_path):
    (tmp_path / ".gitignore").write_text("sub/ignored.txt\n")
    sub = tmp_path / "sub"
    sub.mkdir()
    visible = sub / "visible.txt"
    visible.write_text("ok")
    ignored = sub / "ignored.txt"
    ignored.write_text("secret")

    files, _ = list_files(str(sub))
    result = "\n".join(files)
    assert str(visible) in result
    assert str(ignored) not in result


def test_list_files_limits_depth(tmp_path):
    deep_dir = tmp_path / "a" / "b" / "c" / "d" / "e"
    deep_dir.mkdir(parents=True)
    file_depth4 = deep_dir.parent / "f4.txt"
    file_depth4.write_text("ok")
    file_depth5 = deep_dir / "f5.txt"
    file_depth5.write_text("no")

    files, _ = list_files(str(tmp_path))
    joined = "\n".join(files)
    assert str(file_depth4) in joined
    assert str(file_depth5) not in joined


def test_list_files_limit_and_message(tmp_path):
    for i in range(205):
        (tmp_path / f"file_{i}.txt").write_text("x")

    files, msg = list_files(str(tmp_path))
    assert len(files) == 200
    assert msg == "Over 200 files found, only returned the first 200 files"
