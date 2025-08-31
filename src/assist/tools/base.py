from typing import List
from langchain_core.tools import BaseTool
from langchain_tavily import TavilySearch
from assist.tools import filesystem, project_index
from assist.tools.system_info import SystemInfoIndex
from pathlib import Path


def base_tools(index_path: Path) -> List[BaseTool]:
    sys_index = SystemInfoIndex(base_dir=index_path)
    return [TavilySearch(max_results=10),
            project_index.ProjectIndex(base_dir=index_path).search_tool(),
            sys_index.search_tool(),
            sys_index.list_tool()]
