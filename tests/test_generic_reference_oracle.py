"""Status-oracle controls use no model and reject negation/history false passes."""

import pytest

from edd.eval.test_generic_reference_binding import SHIPMENTS, _assert_current_status


@pytest.mark.parametrize("layout", ["fields", "table"])
@pytest.mark.parametrize("first,second", [("Passed inspection", "Awaiting inspection"),
                                          ("Passed inspection", "Awaiting inspection (pending)")])
def test_explicit_current_statuses_pass_across_saved_note_formats(layout, first, second):
    names = [row[0] for row in SHIPMENTS]
    if layout == "fields":
        note = f"## {names[0]}\n- **Status:** {first}\n## {names[1]}\n- Current status: {second}\n"
    else:
        note = f"| Recipient | Current status |\n| --- | --- |\n| {names[0]} | {first} |\n| {names[1]} | {second} |\n"
    _assert_current_status(note, names[0], ("Passed inspection",))
    _assert_current_status(note, names[1], ("Awaiting inspection", "Pending"))


@pytest.mark.parametrize("value,allowed", [
    ("Awaiting inspection — has not passed inspection", ("Passed inspection",)),
    ("Not passed inspection", ("Passed inspection",)),
    ("Passed inspection (previously pending)", ("Awaiting inspection", "Pending")),
    ("Passed inspection — was pending", ("Awaiting inspection", "Pending")),
    ("Pending (passed inspection)", ("Awaiting inspection", "Pending")),
    ("Passed inspection (pending)", ("Passed inspection",)),
])
@pytest.mark.parametrize("layout", ["fields", "table"])
def test_negated_or_historical_status_words_do_not_count_as_current(value, allowed, layout):
    name = SHIPMENTS[0][0]
    note = (f"{name}\nStatus: {value}\n" if layout == "fields" else
            f"| Recipient | Status |\n| --- | --- |\n| {name} | {value} |\n")
    with pytest.raises(AssertionError):
        _assert_current_status(note, name, allowed)


def test_wrong_duplicate_current_field_cannot_hide_behind_correct_one():
    name = SHIPMENTS[0][0]
    with pytest.raises(AssertionError):
        _assert_current_status(f"{name}\nStatus: Passed inspection\nStatus: Pending\n",
                               name, ("Passed inspection",))


def test_only_historical_status_is_not_current_status_evidence():
    name = SHIPMENTS[0][0]
    with pytest.raises(AssertionError):
        _assert_current_status(f"{name}\nPrevious status: Passed inspection\n",
                               name, ("Passed inspection",))
