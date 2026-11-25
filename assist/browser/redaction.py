"""Mask known private values in raw browser text, including ordinary URL encoding."""
from __future__ import annotations

import re
from urllib.parse import unquote_to_bytes

_PERCENT_TEXT = re.compile(r"%[0-9a-fA-F]{2}")
_PERCENT_BYTES = re.compile(rb"%[0-9a-fA-F]{2}")
MARKER = "[private value hidden]"
# 64 catalog entries, each with at most 1,024 password and 320 username bytes,
# can expand sixfold under JSON escaping. Allow the existing 32 KiB public
# command alongside that complete private envelope.
MAX_PRIVATE_COMMAND = 600_000


def _canonical_percent(value: str | bytes) -> str | bytes:
    pattern = _PERCENT_TEXT if isinstance(value, str) else _PERCENT_BYTES
    return pattern.sub(lambda match: match.group().upper(), value)


def _decoded_spans(value: str, plus: bool) -> tuple[bytes, list[tuple[int, int]]]:
    data = bytearray()
    spans = []
    index = 0
    while index < len(value):
        char = value[index]
        if (char == "%" and index + 2 < len(value)
                and all(digit in "0123456789abcdefABCDEF"
                        for digit in value[index + 1:index + 3])):
            encoded = bytes([int(value[index + 1:index + 3], 16)])
            end = index + 3
        else:
            encoded = (" " if plus and char == "+" else char).encode("utf-8")
            end = index + 1
        data.extend(encoded)
        spans.extend([(index, end)] * len(encoded))
        index = end
    return bytes(data), spans


def contains_hidden(value: str, hidden: tuple[str, ...]) -> bool:
    if not hidden:
        return False
    raw = _canonical_percent(value.encode("utf-8"))
    decoded = (_canonical_percent(unquote_to_bytes(value))
               if "%" in value else raw)
    form = (_canonical_percent(unquote_to_bytes(value.replace("+", " ")))
            if "+" in value else decoded)
    return any((needle := _canonical_percent(word.encode("utf-8"))) in raw
               or needle in decoded or needle in form for word in hidden)


def hidden_spans(value: str, hidden: tuple[str, ...]) -> list[tuple[int, int]]:
    """Return merged original-text spans for the same matches used by text masking."""
    if not contains_hidden(value, hidden):
        return []
    changes = [0] * (len(value) + 1)
    decoded = [_decoded_spans(value, False)] if "%" in value else []
    if "+" in value:
        decoded.append(_decoded_spans(value, True))
    decoded = [(_canonical_percent(data), positions) for data, positions in decoded]
    canonical = _canonical_percent(value)
    for word in hidden:
        canonical_word = _canonical_percent(word)
        start = canonical.find(canonical_word)
        while start >= 0:
            changes[start] += 1
            changes[start + len(word)] -= 1
            start = canonical.find(canonical_word, start + 1)
        needle = _canonical_percent(word.encode("utf-8"))
        for data, positions in decoded:
            start = data.find(needle)
            while start >= 0:
                beginning = positions[start][0]
                end = positions[start + len(needle) - 1][1]
                changes[beginning] += 1
                changes[end] -= 1
                start = data.find(needle, start + 1)
    spans = []
    beginning = 0
    depth = 0
    for index, change in enumerate(changes):
        previous = depth
        depth += change
        if previous == 0 and depth:
            beginning = index
        elif previous and depth == 0:
            spans.append((beginning, index))
    return spans


def replace_hidden(value: str, hidden: tuple[str, ...], marker: str = MARKER) -> str:
    result = []
    cursor = 0
    for beginning, end in hidden_spans(value, hidden):
        result.extend((value[cursor:beginning], marker))
        cursor = end
    result.append(value[cursor:])
    return "".join(result)
