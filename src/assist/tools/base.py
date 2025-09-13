from typing import List
from langchain_core.tools import BaseTool
from langchain_tavily import TavilySearch
from assist.tools import filesystem, project_index
from assist.tools.system_info import SystemInfoIndex
from assist.tools.unit_conversion import UnitConversionTool
from assist.tools.timer import TimerTool
from assist.tools.web_search import search_site, search_page
from assist.tools.date_utils import get_current_date, offset_date, diff_dates
from assist.tools.safe_python import SafePythonTool
from pathlib import Path


def base_tools(index_path: Path) -> List[BaseTool]:
    sys_index = SystemInfoIndex(base_dir=index_path)
    return [
        TavilySearch(max_results=10),
        project_index.ProjectIndex(base_dir=index_path).search_tool(),
        #sys_index.search_tool(),
        #sys_index.list_tool(),
        filesystem.write_file_user,
        filesystem.write_file_tmp,
        filesystem.list_files,
        filesystem.file_contents,
        filesystem.project_context,
        UnitConversionTool(),
        TimerTool(),
        SafePythonTool(),
        search_site,
        search_page,
        get_current_date,
        offset_date,
        diff_dates,
    ]
