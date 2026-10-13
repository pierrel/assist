"""Natural research completion with source-backed book recommendations.

The model and leaf research construction are real; canned search/page results
isolate generation behavior from search throttling and publisher availability.
"""
import re
import tempfile
from unittest import TestCase

from assist.agent import AgentHarness, create_research_agent
from assist.model_manager import select_assistant_model
from .utils import ResearchToolSpy


_BOOKS = [
    {"title": "The Odyssey", "url": "https://example.org/books/cross-odyssey",
     "content": "The Odyssey, retold by Gillian Cross and illustrated by Neil Packer. "
                "Candlewick Press. Ages 8-12. A standalone illustrated narrative retelling "
                "of Odysseus's journey home, with full-color illustrations."},
    {"title": "The Iliad", "url": "https://example.org/books/cross-iliad",
     "content": "The Iliad, retold by Gillian Cross and illustrated by Neil Packer. "
                "Candlewick Press. Ages 8-12. An illustrated narrative retelling of the "
                "Trojan War. A companion to their Odyssey, also readable independently."},
    {"title": "Tales from the Odyssey", "url": "https://example.org/books/osborne-odyssey",
     "content": "Tales from the Odyssey by Mary Pope Osborne, Disney-Hyperion. Ages 7-10. "
                "A series of short illustrated chapter books retelling Odysseus's adventures. "
                "Part One collects the opening adventures in accessible prose."},
    {"title": "The Iliad and the Odyssey", "url": "https://example.org/books/williams-homer",
     "content": "The Iliad and the Odyssey retold and illustrated by Marcia Williams. "
                "Walker Books. Ages 7-10. Both epics retold with comic-strip pictures, "
                "speech bubbles and short text. A visual introduction rather than chapter-book prose."},
]


def _count_sourced_recommendations(result, recommendations):
    body, *sources = re.split(
        r"(?im)^\s*(?:#{1,6}\s*)?(?:\*\*)?Sources:?(?:\*\*)?\s*$", result, maxsplit=1)
    url_pattern = r"https?://[^\s\])>]+"
    source_urls = set(re.findall(url_pattern, sources[0])) if sources else set()
    paragraphs = [(paragraph, set(re.findall(url_pattern, paragraph)))
                  for paragraph in re.split(r"\n\s*\n|\n(?=\s*#{1,6}\s)", body)]
    # The worker supports a final Sources list. An explicit inline link must
    # match the recommendation rather than being rescued by that common list.
    return sum(any(title in paragraph and author in paragraph
                   and url in (inline_urls or source_urls)
                   for paragraph, inline_urls in paragraphs)
               for author, title, url in recommendations)


class TestResearchGeneration(TestCase):
    def test_recommends_illustrated_retellings(self):
        prompt = ("Recommend a few illustrated retellings of the Iliad or Odyssey "
                  "for a child who enjoys chapter books.")
        model = select_assistant_model(0.1)
        with tempfile.TemporaryDirectory() as root, ResearchToolSpy(search_fixture=_BOOKS,
                read_fixture={book["url"]: book["content"] for book in _BOOKS}) as spy:
            agent = AgentHarness(create_research_agent(model, root, leaf=True))
            result = agent.message(prompt)
            self.assertNotEqual(agent.all_messages()[-1].response_metadata.get("finish_reason"), "length",
                                "A capped partial response is not a completed recommendation")
        self.assertTrue(spy.searched, "Recommendations must use retrieved evidence")
        recommendations = (
            ("Gillian Cross", "The Odyssey", _BOOKS[0]["url"]),
            ("Gillian Cross", "The Iliad", _BOOKS[1]["url"]),
            ("Mary Pope Osborne", "Tales from the Odyssey", _BOOKS[2]["url"]),
            ("Marcia Williams", "The Iliad and the Odyssey", _BOOKS[3]["url"]),
        )
        self.assertGreaterEqual(_count_sourced_recommendations(result, recommendations), 2,
                                "Each recommendation needs its author and matching source: " + result)
        self.assertRegex(result.lower(), r"chapter.books?|comic.strip|narrative")
        self.assertFalse(spy.guessed_fetches(), "Fetched URLs must come from source results")
        self.assertNotIn("timed out", result.lower())
