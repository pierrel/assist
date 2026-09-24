"""Stateful target references cannot drift to a new link between observations."""
import pytest

from assist.browser.runner import BrowserInputError, BrowserWorker


class _Page:
    url = "https://example.com/"

    def is_closed(self):
        return False


class _Link:
    def __init__(self, href):
        self.href = href
        self.clicked = False

    def is_visible(self):
        return True

    def get_attribute(self, name):
        return self.href if name == "href" else None

    def click(self, **_kwargs):
        self.clicked = True


@pytest.mark.parametrize("selector", [
    {"ref": "observed"},
    {"href": "https://example.com/original"},
    {"role": "link", "name": "Manual"},
])
def test_changed_observed_link_fails_for_every_target_selector(selector):
    worker = BrowserWorker()
    worker.pages["page"] = _Page()
    worker.snapshots["page"] = "snapshot"
    link = _Link("/changed")
    worker.targets["page"] = {"observed": (
        link, "link", "Manual", "https://example.com/original")}

    with pytest.raises(BrowserInputError, match="observed link changed"):
        worker.act("page", "snapshot", "click", selector)
    assert not link.clicked
