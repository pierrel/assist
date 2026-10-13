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
    """Count sourced book identities with their own supported format information."""
    body, *sources = re.split(
        r"(?im)^\s*(?:#{1,6}\s*)?(?:\*\*)?Sources:?(?:\*\*)?\s*$", result, maxsplit=1)
    url_pattern = r"https?://[^\s\])>|]+"
    source_urls = {url.rstrip(".,;") for url in re.findall(url_pattern, sources[0])} if sources else set()
    item_start = r"[ \t]*(?:#{1,6}\s|[-*+]\s|\d{1,9}[.)]\s|[^\n]*\|)"
    paragraphs = re.split(r"\n\s*\n|\n(?=" + item_start + ")", body)
    identities = {(author, title) for author, title, _, _ in recommendations}
    names = sorted({name for pair in identities for name in pair}, key=len, reverse=True)
    identity_tokens = "|".join(re.escape(name) for name in names)
    sections = []
    for paragraph in paragraphs:
        tokens = list(re.finditer(identity_tokens, paragraph))
        starts = {0: None}
        index = 0
        while index < len(tokens):
            pair = tuple(token.group() for token in tokens[index:index + 2])
            if len(pair) == 2 and (pair in identities or pair[::-1] in identities):
                starts[tokens[index].start()] = pair if pair in identities else pair[::-1]
                index += 2
            else:
                starts[tokens[index].start()] = None
                index += 1
        boundaries = sorted(starts) + [len(paragraph)]
        sections.extend((paragraph[start:end], starts[start])
                        for start, end in zip(boundaries, boundaries[1:]))
    paragraphs = sections
    count = 0
    for author, title, url, format_pattern in recommendations:
        for index, (paragraph, identity) in enumerate(paragraphs):
            if identity != (author, title):
                continue
            description = paragraph
            if index + 1 < len(paragraphs):
                following = paragraphs[index + 1][0]
                if not re.match(item_start, following) and not re.match(identity_tokens, following):
                    description += "\n\n" + following
            # Support the worker's final Sources list, but an explicit inline
            # link in the identity or its description must match this book.
            inline_urls = {url.rstrip(".,;") for url in re.findall(url_pattern, description)}
            if url in (inline_urls or source_urls) and re.search(format_pattern, description.lower()):
                count += 1
                break
    return count


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
            ("Gillian Cross", "The Odyssey", _BOOKS[0]["url"], r"narrative|prose|chapter.book|continuous.story|full.story"),
            ("Gillian Cross", "The Iliad", _BOOKS[1]["url"], r"narrative|prose|chapter.book|continuous.story|full.story"),
            ("Mary Pope Osborne", "Tales from the Odyssey", _BOOKS[2]["url"], r"chapter|series|episodic|short (?:books|stories|prose)"),
            ("Marcia Williams", "The Iliad and the Odyssey", _BOOKS[3]["url"], r"comic|graphic|visual|speech.bubbles"),
        )
        self.assertGreaterEqual(_count_sourced_recommendations(result, recommendations), 2,
                                "Each recommendation needs its author, matching source and own format: " + result)
        self.assertFalse(spy.guessed_fetches(), "Fetched URLs must come from source results")
        self.assertNotIn("timed out", result.lower())
