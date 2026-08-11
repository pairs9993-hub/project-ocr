"""Tests for the image-based accent verifier.

These pin the safety contract, not the accuracy: the verifier currently fails
its synthetic holdout (see Gate F1) and is not fit for runtime use. What must
hold regardless is that it never invents an accent, never touches anything but
the glyph it was asked about, and refuses to answer when unsure.
"""

import inspect
import unittest

import numpy as np

from ocr_roi_validator.accent_verifier import (
    ACCENT_ABSENT,
    ACCENT_PRESENT,
    UNKNOWN,
    verify_accent_glyph,
)


def make_line(width=120, height=30, background=0):
    return np.full((height, width, 3), background, dtype=np.uint8)


def draw_block(image, x0, x1, y0, y1, value=240):
    image[y0:y1, x0:x1] = value


class ApiSafetyTests(unittest.TestCase):
    def test_api_has_no_expected_text_parameter(self) -> None:
        parameters = inspect.signature(verify_accent_glyph).parameters
        self.assertEqual(list(parameters), ["line_image", "x0", "x1"])
        for name in parameters:
            self.assertNotIn("expected", name.lower())
            self.assertNotIn("reference", name.lower())
            self.assertNotIn("truth", name.lower())

    def test_verdicts_are_limited_to_three_values(self) -> None:
        line = make_line()
        draw_block(line, 10, 24, 10, 24)
        verdict = verify_accent_glyph(line, 10, 24).verdict
        self.assertIn(verdict, {ACCENT_PRESENT, ACCENT_ABSENT, UNKNOWN})


class FailClosedTests(unittest.TestCase):
    """Anything the verifier cannot measure must come back as unknown."""

    def test_empty_image_is_unknown(self) -> None:
        self.assertEqual(
            verify_accent_glyph(np.zeros((0, 0, 3), np.uint8), 0, 5).verdict, UNKNOWN
        )

    def test_none_image_is_unknown(self) -> None:
        self.assertEqual(verify_accent_glyph(None, 0, 5).verdict, UNKNOWN)

    def test_blank_line_is_unknown(self) -> None:
        self.assertEqual(verify_accent_glyph(make_line(), 10, 24).verdict, UNKNOWN)

    def test_narrow_span_is_unknown(self) -> None:
        line = make_line()
        draw_block(line, 10, 24, 10, 24)
        self.assertEqual(verify_accent_glyph(line, 10, 12).verdict, UNKNOWN)

    def test_span_with_no_ink_is_unknown(self) -> None:
        line = make_line()
        draw_block(line, 10, 24, 10, 24)
        self.assertEqual(verify_accent_glyph(line, 60, 80).verdict, UNKNOWN)

    def test_short_line_is_unknown(self) -> None:
        line = make_line(height=12)
        draw_block(line, 10, 24, 2, 6)
        self.assertEqual(verify_accent_glyph(line, 10, 24).verdict, UNKNOWN)

    def test_ambiguous_top_is_unknown(self) -> None:
        """A glyph starting between the two thresholds must not be decided."""
        line = make_line(height=40)
        draw_block(line, 0, 30, 4, 34)           # reference ink elsewhere on the line
        draw_block(line, 40, 56, 11, 34)         # top ratio ~= 0.23
        self.assertEqual(verify_accent_glyph(line, 40, 56).verdict, UNKNOWN)


class DirectionTests(unittest.TestCase):
    """The verifier may only ever support removing an accent, never adding one."""

    def test_glyph_reaching_line_top_reports_accent_present(self) -> None:
        line = make_line(height=40)
        draw_block(line, 0, 30, 4, 34)           # reference ink elsewhere
        draw_block(line, 40, 56, 4, 34)          # starts at the very top
        result = verify_accent_glyph(line, 40, 56)
        self.assertEqual(result.verdict, ACCENT_PRESENT)
        self.assertFalse(result.is_accent_absent)

    def test_glyph_starting_at_x_height_reports_accent_absent(self) -> None:
        line = make_line(height=40)
        draw_block(line, 0, 30, 4, 34)
        draw_block(line, 40, 56, 16, 34)         # clearly below the line top
        result = verify_accent_glyph(line, 40, 56)
        self.assertEqual(result.verdict, ACCENT_ABSENT)
        self.assertTrue(result.is_accent_absent)

    def test_only_absent_verdict_licenses_a_change(self) -> None:
        """is_accent_absent is the single flag a caller may act on."""
        line = make_line(height=40)
        draw_block(line, 0, 30, 4, 34)
        draw_block(line, 40, 56, 4, 34)
        self.assertFalse(verify_accent_glyph(line, 40, 56).is_accent_absent)
        self.assertFalse(verify_accent_glyph(make_line(), 10, 24).is_accent_absent)


class NeighbourInkTests(unittest.TestCase):
    """A sliver of the neighbouring glyph must not fake an accent."""

    def test_thin_neighbour_column_is_ignored(self) -> None:
        line = make_line(height=40)
        draw_block(line, 0, 30, 4, 34)           # reference ink elsewhere
        draw_block(line, 40, 56, 16, 34)         # the real `e` body
        # Two columns of a tall neighbour clipped into a 16px span: below the
        # 20%-of-width floor, so they must not raise the measured top edge.
        draw_block(line, 40, 42, 4, 16)
        result = verify_accent_glyph(line, 40, 56)
        self.assertEqual(result.verdict, ACCENT_ABSENT)


class PolarityTests(unittest.TestCase):
    def test_dark_text_on_light_background(self) -> None:
        line = np.full((40, 100, 3), 240, np.uint8)
        line[4:34, :] = 240
        draw_block(line, 40, 56, 16, 34, value=10)
        # Give the line a dark ink band so the reference band exists.
        draw_block(line, 0, 40, 4, 34, value=10)
        result = verify_accent_glyph(line, 40, 56)
        self.assertIn(result.verdict, {ACCENT_ABSENT, UNKNOWN})


if __name__ == "__main__":
    unittest.main()
