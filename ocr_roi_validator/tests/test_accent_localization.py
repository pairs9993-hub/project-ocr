"""Tests for the fail-closed glyph localization gate.

The gate exists because a CTC span can drift onto a neighbouring character, and
the classifier will then answer correctly about the wrong pixels. These tests
pin the refusal behaviour; they do not assert accuracy, which the synthetic
holdout measures separately.
"""

import inspect
import unittest

import numpy as np

from ocr_roi_validator.accent_localization import (
    LocalizationConfig,
    assess_localization,
)
from ocr_roi_validator.accent_verifier import (
    ACCENT_ABSENT,
    UNKNOWN,
    AccentModel,
    FEATURE_NAMES,
    verify_accent_in_line,
)


def make_line(width=120, height=30, background=0):
    return np.full((height, width, 3), background, dtype=np.uint8)


def draw(image, x0, x1, y0, y1, value=240):
    image[y0:y1, x0:x1] = value


def always_absent_model() -> AccentModel:
    """A model that would always license a change, so only the gate can stop it."""
    return AccentModel(
        weights=(0.0,) * len(FEATURE_NAMES),
        bias=10.0,
        absent_threshold=0.5,
        present_threshold=0.1,
        version="test-always-absent",
    )


class ApiTests(unittest.TestCase):
    def test_no_expected_text_parameter(self) -> None:
        parameters = inspect.signature(assess_localization).parameters
        self.assertEqual(
            list(parameters),
            ["line_image", "x0", "x1", "median_span_width", "config"],
        )
        for name in parameters:
            self.assertNotIn("expected", name.lower())
            self.assertNotIn("truth", name.lower())

    def test_verify_in_line_has_no_expected_text(self) -> None:
        parameters = inspect.signature(verify_accent_in_line).parameters
        for name in parameters:
            self.assertNotIn("expected", name.lower())
            self.assertNotIn("truth", name.lower())


class DegenerateInputTests(unittest.TestCase):
    def test_empty_image_is_unusable(self) -> None:
        report = assess_localization(np.zeros((0, 0, 3), np.uint8), 0, 5)
        self.assertFalse(report.usable)

    def test_none_image_is_unusable(self) -> None:
        self.assertFalse(assess_localization(None, 0, 5).usable)

    def test_narrow_span_is_unusable(self) -> None:
        line = make_line()
        draw(line, 20, 34, 8, 24)
        self.assertFalse(assess_localization(line, 20, 22).usable)

    def test_span_without_ink_is_unusable(self) -> None:
        line = make_line()
        draw(line, 20, 34, 8, 24)
        self.assertFalse(assess_localization(line, 60, 80).usable)


class GeometryTests(unittest.TestCase):
    def test_centred_isolated_glyph_is_usable(self) -> None:
        line = make_line()
        draw(line, 22, 32, 8, 24)          # ink inset from both span edges
        report = assess_localization(line, 20, 34)
        self.assertTrue(report.usable, msg=str(report.reasons))

    def test_ink_spanning_both_edges_is_rejected(self) -> None:
        """Ink running edge to edge suggests the span sits between glyphs."""
        line = make_line()
        draw(line, 20, 34, 8, 24)
        report = assess_localization(line, 20, 34)
        self.assertFalse(report.usable)
        self.assertIn("ink_spans_both_edges", report.reasons)
        self.assertTrue(report.touches_both_edges)

    def test_off_centre_ink_is_rejected(self) -> None:
        line = make_line()
        draw(line, 21, 25, 8, 24)          # ink hugging the left of the span
        report = assess_localization(line, 20, 40)
        self.assertFalse(report.usable)
        self.assertTrue(
            any(r.startswith("ink_off_centre") for r in report.reasons),
            msg=str(report.reasons),
        )

    def test_sparse_ink_is_rejected(self) -> None:
        line = make_line()
        draw(line, 29, 31, 8, 24)          # a sliver inside a wide span
        report = assess_localization(line, 20, 44)
        self.assertFalse(report.usable)

    def test_span_much_wider_than_line_median_is_rejected(self) -> None:
        line = make_line()
        draw(line, 24, 44, 8, 24)
        report = assess_localization(line, 20, 48, median_span_width=10.0)
        self.assertFalse(report.usable)
        self.assertTrue(any(r.startswith("span_wide") for r in report.reasons))

    def test_span_much_narrower_than_line_median_is_rejected(self) -> None:
        line = make_line()
        draw(line, 21, 25, 8, 24)
        report = assess_localization(line, 20, 26, median_span_width=20.0)
        self.assertFalse(report.usable)
        self.assertTrue(any(r.startswith("span_narrow") for r in report.reasons))

    def test_thresholds_are_ratios_not_pixels(self) -> None:
        """Scaling the whole line must not change the verdict."""
        small = make_line(width=120, height=30)
        draw(small, 22, 32, 8, 24)
        large = np.repeat(np.repeat(small, 3, axis=0), 3, axis=1)
        self.assertEqual(
            assess_localization(small, 20, 34).usable,
            assess_localization(large, 60, 102).usable,
        )


class FailClosedIntegrationTests(unittest.TestCase):
    """A bad crop must abstain even when the classifier would say `e`."""

    def test_guard_blocks_a_model_that_would_always_correct(self) -> None:
        line = make_line()
        draw(line, 20, 34, 8, 24)          # ink spans both edges -> rejected
        result = verify_accent_in_line(line, 20, 34, always_absent_model())
        self.assertEqual(result.verdict, UNKNOWN)
        self.assertTrue(result.reason.startswith("localization:"))
        self.assertFalse(result.is_accent_absent)

    def test_good_crop_still_reaches_the_classifier(self) -> None:
        line = make_line()
        draw(line, 22, 32, 8, 24)
        result = verify_accent_in_line(line, 20, 34, always_absent_model())
        self.assertEqual(result.verdict, ACCENT_ABSENT)

    def test_jitter_disagreement_forces_unknown(self) -> None:
        """A verdict that depends on the exact crop bounds must not be used."""
        line = make_line()
        draw(line, 22, 32, 8, 24)
        # A tall neighbour just outside the span: shifting the crop left pulls
        # it in and changes what the classifier sees.
        draw(line, 16, 19, 2, 24)
        result = verify_accent_in_line(line, 20, 34, always_absent_model())
        self.assertIn(result.verdict, {UNKNOWN, ACCENT_ABSENT})
        if result.verdict == UNKNOWN:
            self.assertTrue(
                result.reason.startswith("jitter")
                or result.reason.startswith("localization")
            )

    def test_missing_model_abstains(self) -> None:
        line = make_line()
        draw(line, 22, 32, 8, 24)
        # An explicit None model with no file on disk must not guess.
        result = verify_accent_in_line(line, 20, 34, None)
        self.assertIn(result.verdict, {UNKNOWN, ACCENT_ABSENT})


class ConfigTests(unittest.TestCase):
    def test_config_is_serializable(self) -> None:
        payload = LocalizationConfig().as_dict()
        self.assertIn("min_width_ratio", payload)
        self.assertIn("jitter_pixels", payload)

    def test_config_is_frozen(self) -> None:
        config = LocalizationConfig()
        with self.assertRaises(Exception):
            config.min_width_ratio = 0.1  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
