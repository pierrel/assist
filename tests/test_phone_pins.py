"""Durable phone-pin store contracts."""
from __future__ import annotations

import json

import pytest

from assist import phone_pins


def test_pin_is_idempotent_and_survives_a_reopen(tmp_path):
    first = phone_pins.create_pin(str(tmp_path), "m-1", "useful answer")
    duplicate = phone_pins.create_pin(str(tmp_path), "m-1", "changed ignored text")

    assert duplicate == first
    assert phone_pins.list_pins(str(tmp_path)) == [first]
    assert json.loads((tmp_path / "phone-pins.json").read_text())[0]["text"] == "useful answer"


def test_pins_are_newest_first_and_delete_is_idempotent(tmp_path):
    first = phone_pins.create_pin(str(tmp_path), "m-1", "first")
    second = phone_pins.create_pin(str(tmp_path), "m-2", "second")

    assert phone_pins.list_pins(str(tmp_path)) == [second, first]
    assert phone_pins.delete_pin(str(tmp_path), "m-1")
    assert not phone_pins.delete_pin(str(tmp_path), "m-1")
    assert phone_pins.list_pins(str(tmp_path)) == [second]


def test_pin_count_is_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(phone_pins, "MAX_PINS", 2)
    phone_pins.create_pin(str(tmp_path), "m-1", "one")
    phone_pins.create_pin(str(tmp_path), "m-2", "two")

    with pytest.raises(ValueError, match="pin limit"):
        phone_pins.create_pin(str(tmp_path), "m-3", "three")


def test_invalid_store_is_not_silently_replaced(tmp_path):
    (tmp_path / "phone-pins.json").write_text("{}")

    with pytest.raises(ValueError, match="invalid phone pin store"):
        phone_pins.list_pins(str(tmp_path))
