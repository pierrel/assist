import os
from tavily import TavilyClient

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


def read_url(url: str) -> str:
    """Extract the content from the given url."""
    return str(tavily.extract([url]))


def search_internet(
        query: str,
        max_results: int = 5,
) -> str:
    """Used to search the internet for information on a given topic using a query string."""
    search_docs = tavily.search(query,
                                max_results=max_results)
    return str(search_docs)


