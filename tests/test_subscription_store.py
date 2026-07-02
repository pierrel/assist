"""Unit tests for the message-event subscription model + store."""
import os
import tempfile
import unittest

from assist.events.model import Subscription, validate_regexp, InvalidRegexp
from assist.events.store import SubscriptionStore, SubscriptionCapExceeded, SubscriptionNotFound


def _sub(sid, tid, regexp, template="from {sender}: {text}", created="2026-01-01T00:00:00"):
    return Subscription(id=sid, thread_id=tid, sender_regexp=regexp,
                        template=template, created_at=created)


class TestSubscriptionModel(unittest.TestCase):
    def test_matches_regexp(self):
        s = _sub("a", "t1", r"^\+1555")
        self.assertTrue(s.matches("+15551234567"))
        self.assertFalse(s.matches("+16505550000"))

    def test_disabled_never_matches(self):
        s = _sub("a", "t1", r".*").with_enabled(False)
        self.assertFalse(s.matches("anything"))

    def test_bad_regexp_does_not_raise_at_match(self):
        self.assertFalse(_sub("a", "t1", r"(unclosed").matches("x"))

    def test_render_uses_literal_tokens(self):
        s = _sub("a", "t1", r".*", template="msg from {sender}: {text}\nrules: reply nicely")
        out = s.render("+1555", "hi there")
        self.assertIn("from +1555", out)
        self.assertIn("hi there", out)

    def test_render_leaves_other_braces_untouched(self):
        s = _sub("a", "t1", r".*", template='{sender} said {text} — ignore {"json": 1}')
        self.assertIn('{"json": 1}', s.render("X", "Y"))

    def test_validate_regexp(self):
        validate_regexp(r"^\+1\d{10}$")  # ok
        with self.assertRaises(InvalidRegexp):
            validate_regexp(r"(unclosed")

    def test_roundtrip(self):
        s = _sub("a", "t1", r"^\+1555", template="t")
        self.assertEqual(Subscription.from_dict(s.to_dict()), s)


class TestSubscriptionStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        for tid in ("t1", "t2"):
            os.makedirs(os.path.join(self.tmp, tid))
        self.store = SubscriptionStore(self.tmp)

    def test_add_and_for_thread(self):
        self.store.add(_sub("a", "t1", r".*"))
        self.assertEqual(len(self.store.for_thread("t1")), 1)
        self.assertEqual(self.store.for_thread("t2"), [])

    def test_route_first_match_by_creation_order(self):
        self.store.add(_sub("early", "t1", r"^\+1555", created="2026-01-01T00:00:00"))
        self.store.add(_sub("late", "t2", r"^\+1555", created="2026-02-01T00:00:00"))
        self.assertEqual(self.store.route("+15551234567").id, "early")

    def test_route_no_match_returns_none(self):
        self.store.add(_sub("a", "t1", r"^\+1555"))
        self.assertIsNone(self.store.route("+16505550000"))

    def test_route_catch_all(self):
        self.store.add(_sub("all", "t1", r".*"))
        self.assertEqual(self.store.route("shortcode-ABC").id, "all")

    def test_cap(self):
        for i in range(10):
            self.store.add(_sub(f"s{i}", "t1", r".*"))
        with self.assertRaises(SubscriptionCapExceeded):
            self.store.add(_sub("overflow", "t1", r".*"))

    def test_remove_missing_raises(self):
        with self.assertRaises(SubscriptionNotFound):
            self.store.remove("t1", "nope")


if __name__ == "__main__":
    unittest.main()


def test_subscriptions_vanish_when_thread_dir_removed(tmp_path):
    # Deleting a thread rmtrees <root>/<tid>/, which holds subscriptions.json — so a deleted
    # thread's subscriptions disappear from the store by construction (no orphan routing).
    import shutil
    for tid in ("t1", "t2"):
        os.makedirs(os.path.join(tmp_path, tid))
    store = SubscriptionStore(str(tmp_path))
    store.add(_sub("keep", "t1", r"^\+1555", created="2026-01-01T00:00:00"))
    store.add(_sub("gone", "t2", r"^\+1555", created="2026-02-01T00:00:00"))
    assert store.route("+15551234567").id == "keep"      # t1's is earliest
    shutil.rmtree(os.path.join(tmp_path, "t2"))           # hard_delete removes the thread dir
    assert store.for_thread("t2") == []                  # gone
    assert store.route("+15551234567").id == "keep"      # t2's no longer routed
    assert all(s.thread_id != "t2" for s in store.all())
