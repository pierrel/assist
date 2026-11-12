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


INLINE_BEFORE = ("# Note\n\nThis note refers to Mîra Vale.\n\n"
                 "| Name | ID | Status |\n| --- | --- | --- |\n"
                 "| Mîra Vale | ZX-0Il1-7F0b | Pending |\n")


def test_capability_status_only_update_preserves_original_sentence_and_table():
    from edd.eval.test_generic_reference_binding import _assert_inline_file_update
    _assert_inline_file_update(INLINE_BEFORE, INLINE_BEFORE.replace("| Pending |", "| Ready |"),
                               "Mîra Vale", "ZX-0Il1-7F0b")


@pytest.mark.parametrize("after", [
    "Mîra Vale ZX-0Il1-7F0b Ready\n",
    INLINE_BEFORE.replace("This note refers to Mîra Vale.\n\n", "").replace("| Pending |", "| Ready |"),
    "This note refers to Mîra Vale. ZX-0Il1-7F0b Ready.\n",
    INLINE_BEFORE + "Ready\n",
    INLINE_BEFORE.replace("This note refers", "A changed sentence refers").replace("| Pending |", "| Ready |"),
    INLINE_BEFORE.replace("| Name | ID | Status |\n| --- | --- | --- |\n", "").replace("| Pending |", "| Ready |"),
])
def test_capability_cannot_pass_after_deleting_or_changing_original_content(after):
    from edd.eval.test_generic_reference_binding import _assert_inline_file_update
    with pytest.raises(AssertionError):
        _assert_inline_file_update(INLINE_BEFORE, after, "Mîra Vale", "ZX-0Il1-7F0b")


@pytest.mark.parametrize("before", [
    INLINE_BEFORE[INLINE_BEFORE.index("| Name |"):],
    "Name | ID | Status\n--- | --- | ---\nMîra Vale | ZX-0Il1-7F0b | Pending\n",
    "This note refers to Mîra Vale and ZX-0Il1-7F0b. Pending.\n",
])
def test_capability_requires_both_structures_in_first_saved_file(before):
    from edd.eval.test_generic_reference_binding import _assert_inline_file_update
    with pytest.raises(AssertionError):
        _assert_inline_file_update(before, before.replace("Pending", "Ready"),
                                   "Mîra Vale", "ZX-0Il1-7F0b")


@pytest.mark.parametrize("original,changed", [("Mîra Vale", "mîra vale"),
                                               ("ZX-0Il1-7F0b", "zx-0il1-7f0b")])
def test_capability_requires_exact_table_data_case(original, changed):
    from edd.eval.test_generic_reference_binding import _assert_inline_file_update
    # Correct paragraph values cannot hide a mutated table cell.
    before = INLINE_BEFORE.replace("This note refers to Mîra Vale.",
                                 "This note refers to Mîra Vale and ZX-0Il1-7F0b.")
    before = before.replace(f"| {original} |", f"| {changed} |")
    with pytest.raises(AssertionError):
        _assert_inline_file_update(before, before.replace("| Pending |", "| Ready |"),
                                   "Mîra Vale", "ZX-0Il1-7F0b")
