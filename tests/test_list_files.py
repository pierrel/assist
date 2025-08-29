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

    result = "\n".join(list_files(str(tmp_path)))
    assert str(visible) in result
    assert str(ignored) not in result
    assert str(secret) not in result


def test_list_files_limit(tmp_path):
    for i in range(205):
        f = tmp_path / f"file_{i}.txt"
        f.write_text("x")

    result = list_files(str(tmp_path))
    assert len(result) == 201
    assert result[-1] == "Limit of 200 files reached"
