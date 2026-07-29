"""Unit tests for the ``type: map`` render block (manage/web/threads.py).

The map data is agent-authored (untrusted small-model output), so alongside the
happy path these pin the containment contract: a null-origin sandboxed iframe, a
CSP, non-executable escaped data, text-node labels, and count/size caps.
"""
import json
import re
from unittest import TestCase

from manage.web.threads import (
    _parse_render_block, _render_map_block, _render_assistant_content,
    _MAP_MAX_PINS, _MAP_MAX_PATHS,
)


def _block(body: str) -> dict:
    return _parse_render_block(body)


class TestParseRenderBlock(TestCase):
    def test_repeated_keys_become_lists(self):
        b = _block("type: map\npin: 1,2 a\npin: 3,4 b\npin: 5,6 c")
        self.assertEqual(b["type"], "map")
        self.assertEqual(b["pin"], ["1,2 a", "3,4 b", "5,6 c"])

    def test_single_key_stays_string(self):
        # file blocks rely on path/lines being plain strings — must not regress.
        b = _block("type: file\npath: /workspace/notes.org\nlines: 10-20")
        self.assertEqual(b["path"], "/workspace/notes.org")
        self.assertEqual(b["lines"], "10-20")

    def test_mixed_single_path_still_string(self):
        b = _block("type: map\npin: 1,2 a\npath: xyz walk")
        self.assertIsInstance(b["pin"], str)   # one pin -> string
        self.assertEqual(b["path"], "xyz walk")


class TestRenderMapBlock(TestCase):
    def test_valid_map_is_sandboxed_null_origin_iframe(self):
        html = _render_map_block("t", _block("type: map\npin: 37.76,-122.42 Four Barrel"))
        self.assertIn('class="show-map"', html)
        self.assertIn('sandbox="allow-scripts allow-popups"', html)
        self.assertNotIn("allow-same-origin", html)   # opaque origin = containment
        self.assertIn("srcdoc=", html)
        self.assertIn("Content-Security-Policy", html)
        self.assertIn("tile.openstreetmap.org", html)

    def test_pins_and_paths_reach_the_data_island(self):
        html = _render_map_block(
            "t", _block("type: map\npin: 37.76,-122.42 Shop\npath: abc123 Walk"))
        # the data island JSON (</ escaped) carries the parsed pins + paths
        self.assertIn("37.76", html)
        self.assertIn("abc123", html)

    def test_empty_or_invalid_returns_none(self):
        self.assertIsNone(_render_map_block("t", _block("type: map")))
        self.assertIsNone(_render_map_block("t", _block("type: map\npin: nope label")))
        self.assertIsNone(_render_map_block("t", _block("type: map\npin: 200,999 oob")))

    def test_bad_pin_dropped_good_pin_kept(self):
        html = _render_map_block(
            "t", _block("type: map\npin: nope bad\npin: 37.7,-122.4 good"))
        self.assertIsNotNone(html)
        self.assertIn("37.7", html)

    def test_pin_origin_marker(self):
        # The agent only marks the ORIGIN; the renderer owns the color (origin green,
        # every other pin the default blue). No color is ever read from agent text.
        from manage.web.threads import _parse_pin
        self.assertEqual(_parse_pin("origin 37.77,-122.42 You are here")["color"], "#15803d")  # green
        self.assertEqual(_parse_pin("37.78,-122.41 Blue Bottle")["color"], "#1d4ed8")          # default blue
        # a legacy/stray leading word ("green") is stripped and rendered as a DEFAULT pin
        # (not dropped) — a stray word never drops a valid coordinate; and it's not origin.
        self.assertEqual(_parse_pin("green 37.7,-122.4 x")["color"], "#1d4ed8")   # default, not green
        self.assertIsNone(_parse_pin("origin nope label"))                        # no valid coord -> dropped
        # the model may copy the message-context location ("sent from ~37.77, -122.42")
        # verbatim — a leading ~ and a space after the comma must still parse, else the
        # origin pin (the user's own location) is silently dropped.
        tilde = _parse_pin("origin ~37.77, -122.42 Home")
        self.assertEqual((tilde["lat"], tilde["lon"], tilde["color"]), (37.77, -122.42, "#15803d"))
        # the origin pin's green hex reaches the data island the map JS reads
        html = _render_map_block("t", _block("type: map\npin: origin 37.77,-122.42 Home"))
        self.assertIn("#15803d", html)

    def test_count_caps(self):
        many = "type: map\n" + "".join(f"pin: 1,{i} p{i}\n" for i in range(_MAP_MAX_PINS + 1))
        self.assertIsNone(_render_map_block("t", _block(many)))

    def test_polyline_decoded_at_motis_precision_7(self):
        # MOTIS emits precision-7 polylines; decoding at Google's default 5 puts
        # every point 100x off the globe, dragging fitBounds off-map and blanking
        # the view (thread 20260703130532 bug). Pin the precision in the init.
        html = _render_map_block("t", _block("type: map\npath: abc123 route"))
        decoder = html.partition("function decode(str){")[2].partition("function popup")[0]
        self.assertIn("1e7", decoder)
        self.assertNotIn("1e5", decoder)

    def test_polyline_decoder_is_arithmetic_not_32bit_bitwise(self):
        # At precision 7 a coordinate's zigzag value exceeds 2^31, so JS 32-bit
        # bitwise (<<, |=, >>) overflows to a garbage point ("path in China"). The
        # decoder must accumulate arithmetically (Math.pow), which is exact to 2^53.
        html = _render_map_block("t", _block("type: map\npath: abc123 route"))
        self.assertIn("Math.pow(2,shift)", html)

    def test_map_iframe_is_lazy_loaded(self):
        # the map must not block the rest of the conversation from rendering.
        html = _render_map_block("t", _block("type: map\npin: 37.7,-122.4 x"))
        self.assertIn('loading="lazy"', html)

    def test_map_iframe_has_title(self):
        # a11y: screen readers identify the embedded iframe by its title.
        html = _render_map_block("t", _block("type: map\npin: 37.7,-122.4 x"))
        self.assertIn('title="Map"', html)

    def test_oversized_polyline_dropped(self):
        html = _render_map_block(
            "t", _block("type: map\npin: 1,2 ok\npath: " + "a" * 20001 + " toolong"))
        # the oversized path is dropped; the valid pin still renders
        self.assertIsNotNone(html)
        self.assertNotIn("a" * 20001, html)


class TestMapXSS(TestCase):
    def test_label_markup_is_not_executable(self):
        html = _render_map_block(
            "t", _block('type: map\npin: 37.7,-122.4 </script><img src=x onerror=alert(1)>'))
        # the whole srcdoc is HTML-escaped into the attribute; no raw executable
        # breakout of the data island or the srcdoc.
        self.assertNotIn("</script><img", html)
        self.assertNotIn("<img src=x onerror", html)

    def test_data_island_cannot_break_out(self):
        # a label containing </script> must be </-escaped inside the JSON island
        html = _render_map_block("t", _block("type: map\npin: 1,2 </script>evil"))
        # the unescaped closing-script sequence must not appear anywhere
        self.assertNotIn("</script>evil", html)

    def test_labels_bind_as_textnode_not_html(self):
        # OUR init binds popups via a text node (element), never bindPopup(string)
        # which Leaflet would treat as HTML.  (Leaflet's own vendored code uses
        # innerHTML internally on its trusted DOM — not on agent labels — so we
        # assert our binding path, not the absence of innerHTML in the whole lib.)
        html = _render_map_block("t", _block("type: map\npin: 1,2 x"))
        self.assertIn("el.textContent = label", html)
        self.assertIn("bindPopup(el)", html)


class TestRenderRobustness(TestCase):
    """Regression: the repeated-key -> list parse tweak must NOT crash the scalar
    consumers (`type` dispatch, file-block `path`).  `_render_assistant_content`
    runs inline on the async thread handler, so an uncaught raise here 500s the
    page and permanently bricks that thread (the message is persisted)."""

    def test_repeated_type_line_does_not_crash(self):
        # two type: lines -> block["type"] would be a list -> _RENDER_DISPATCH.get
        # (list) is `unhashable type` without the _scalar guard.
        raw = "```render\ntype: map\ntype: file\npin: 1,2 x\n```"
        self.assertIsInstance(_render_assistant_content("t", raw), str)

    def test_repeated_file_path_does_not_crash(self):
        # two path: lines in a file block -> os.path.splitext(list) without _scalar.
        raw = "```render\ntype: file\npath: a.md\npath: b.md\n```"
        self.assertIsInstance(_render_assistant_content("t", raw), str)


class TestMapFullscreen(TestCase):
    def test_map_has_fullscreen_control(self):
        html = _render_map_block("t", _block("type: map\npin: 37.7,-122.4 x"))
        self.assertIn("classList.toggle('fs')", html)  # CSS pseudo-fullscreen toggle
        self.assertIn("invalidateSize", html)          # Leaflet re-fits on resize
        self.assertIn("show-cap", html)   # same caption row as the file embed's ↗


class TestOneMapPerTurn(TestCase):
    def test_only_first_map_renders(self):
        raw = ("```render\ntype: map\npin: 1,2 a\n```\n\n"
               "```render\ntype: map\npin: 3,4 b\n```")
        out = _render_assistant_content("t", raw)
        self.assertEqual(out.count('class="show-map"'), 1)

    def test_second_map_falls_through_to_code(self):
        raw = ("```render\ntype: map\npin: 1,2 a\n```\n\n"
               "```render\ntype: map\npin: 3,4 second\n```")
        out = _render_assistant_content("t", raw)
        # the second map block stays in the markdown stream (rendered as code)
        self.assertIn("second", out)
