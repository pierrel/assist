import re
from unittest import TestCase


def greet_node(state: dict) -> str:
    return f"Hello, {state['name']}!"


class TestGreetNode(TestCase):
    def setUp(self):
        self.node = greet_node

    def test_alice(self):
        result = self.node({"name": "Alice"})
        self.assertRegex(result, r"Hello, Alice!")

    def test_bob(self):
        result = self.node({"name": "Bob"})
        self.assertTrue(result.endswith("Bob!"))
