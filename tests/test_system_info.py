from assist.tools.system_info import SystemInfoIndex


def _create_info_root(tmp_path):
    info_root = tmp_path / "info"
    info_root.mkdir()

    # Sample info documents
    (info_root / "alpha.info").write_text(
        "Alpha file about bananas and other fruit."
    )
    (info_root / "beta.info").write_text(
        "Beta file documenting apples."
    )
    (info_root / ".hidden.info").write_text("secret")

    # Directory listing used by ``list_tool``
    dir_content = (
        "* alpha: (alpha).\tAlpha info file.\n"
        "* beta: (beta).\tBeta info file.\n"
    )
    (info_root / "dir").write_text(dir_content)

    return info_root


def test_list_system_info_files(tmp_path):
    info_root = _create_info_root(tmp_path)
    idx = SystemInfoIndex(info_root)
    list_tool = idx.list_tool()
    files = list_tool.invoke({})
    assert "alpha - Alpha info file." in files
    assert "beta - Beta info file." in files


def test_system_info_search(tmp_path):
    info_root = _create_info_root(tmp_path)
    idx = SystemInfoIndex(info_root)
    search_tool = idx.search_tool()
    result = search_tool.invoke("bananas")
    assert "bananas" in result


def test_system_info_ignores_hidden(tmp_path):
    info_root = _create_info_root(tmp_path)
    idx = SystemInfoIndex(info_root)
    search_tool = idx.search_tool()
    result = search_tool.invoke("secret")
    assert "secret" not in result
