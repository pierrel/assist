from __future__ import annotations

import gzip
import shutil
from pathlib import Path
from typing import List

from langchain_core.tools import BaseTool, tool

from .project_index import ProjectIndex


class SystemInfoIndex:
    """Index and search system info (``info``) documentation."""

    def __init__(
        self,
        info_root: Path | str = Path("/usr/share/info"),
        base_dir: Path | str | None = None,
        keywords: List[str] | None = None,
    ) -> None:
        self._info_root = Path(info_root)
        base = Path(base_dir) if base_dir is not None else None
        if base is not None:
            base = base / "system"
        self._index = ProjectIndex(base)
        self._prepared_dir: Path | None = None
        self._keywords = keywords or []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _matches_keywords(self, path: Path) -> bool:
        if not self._keywords:
            return True
        if path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as src:
                content = src.read()
        else:
            content = path.read_text(encoding="utf-8", errors="ignore")
        return any(k in content for k in self._keywords)

    def _prepare_dir(self) -> Path:
        if self._prepared_dir is not None:
            return self._prepared_dir

        dest = self._index.index_dir(self._info_root) / "files"
        dest.mkdir(parents=True, exist_ok=True)

        for f in self._info_root.iterdir():
            if not f.is_file():
                continue
            name = f.name
            if "info" not in name:
                continue
            if not self._matches_keywords(f):
                continue
            target = dest / name.replace(".gz", "")
            if f.suffix == ".gz":
                with gzip.open(f, "rb") as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
            else:
                shutil.copy(f, target)

        self._prepared_dir = dest
        return dest

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def search(self, query: str) -> str:
        root = self._prepare_dir()
        return self._index.search(root, query)

    def search_tool(self) -> BaseTool:
        @tool
        def system_info_search(query: str) -> str:
            """Search system ``info`` files for technical information about ``query``."""
            return self.search(query)

        return system_info_search

    def list_tool(self) -> BaseTool:
        @tool
        def list_system_info_files() -> List[str]:
            """List available ``info`` files with short descriptions."""
            dir_file = self._info_root / "dir"
            if not dir_file.exists():
                return []

            entries: List[str] = []
            with open(dir_file, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line.startswith("* "):
                        continue
                    line = line[2:]
                    if ":" not in line:
                        continue
                    name, rest = line.split(":", 1)
                    info_file = self._info_root / f"{name.strip()}.info"
                    if not info_file.exists():
                        info_file = info_file.with_suffix(".info.gz")
                    if not info_file.exists():
                        continue
                    if not self._matches_keywords(info_file):
                        continue
                    # Description usually follows after '.\t'
                    desc = rest.split(".\t", 1)[-1].strip()
                    entries.append(f"{name.strip()} - {desc}")
            return entries

        return list_system_info_files


__all__ = ["SystemInfoIndex"]
