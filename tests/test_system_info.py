from assist.tools.system_info import SystemInfoIndex


def test_list_system_info_files():
    idx = SystemInfoIndex()
    list_tool = idx.list_tool()
    files = list_tool.invoke({})
    assert isinstance(files, list)
    assert any(files)


def test_system_info_search():
    idx = SystemInfoIndex()
    search_tool = idx.search_tool()
    result = search_tool.invoke("directory")
    assert isinstance(result, str)
    assert result.strip()
