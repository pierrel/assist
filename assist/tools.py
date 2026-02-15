import re
import time
import threading

import requests
from ddgs import DDGS

_last_call_time = 0.0
_rate_lock = threading.Lock()
_MIN_DELAY = 0.5


def _rate_limit():
    """Enforce minimum delay between DuckDuckGo API calls."""
    global _last_call_time
    with _rate_lock:
        now = time.time()
        elapsed = now - _last_call_time
        if elapsed < _MIN_DELAY:
            time.sleep(_MIN_DELAY - elapsed)
        _last_call_time = time.time()


def read_url(url: str) -> str:
    """Extract the content from the given url."""
    _rate_limit()
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36"},
            timeout=15,
        )
        resp.raise_for_status()
        text = resp.text
        text = re.sub(r"<script[\s\S]*?</script>|<style[\s\S]*?</style>", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:4000]
    except Exception as e:
        return f"Error fetching URL: {e}"


def search_internet(
        query: str,
        max_results: int = 5,
) -> str:
    """Used to search the internet for information on a given topic using a query string."""
    _rate_limit()
    try:
        results = DDGS().text(query,
                              max_results=max_results,
                              backend="duckduckgo")
    except Exception:
        return "[]"
    normalized = [{"title": r["title"], "url": r["href"], "content": r["body"]} for r in results]
    return str(normalized)
