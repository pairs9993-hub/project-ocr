"""Tests for the image-based accent verifier.

These pin the safety contract rather than the accuracy: the verifier may only
ever license removing an accent, must abstain when it cannot measure a glyph,
and must never see the expected text. Accuracy is measured separately by the
synthetic holdout gate.
"""

import inspect
import json
import unittest
from pathlib import Path

import numpy as np

from ocr_roi_validator.accent_verifier import (
    ACCENT_ABSENT,
    ACCENT_PRESENT,
    FEATURE_NAMES,
    UNKNOWN,
    AccentModel,
    extract_features,
    load_model,
    verify_accent_glyph,
)

MODEL_PATH = Path(__file__).resolve().parents[1] / "ocr_roi_validator" / "accent_model.json"


def glyph_image(height=24, width=14, ink_rows=(8, 24), ink_columns=(2, 12)):
    """Dark background with a bright ink block, like the real line crops."""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[ink_rows[0]:ink_rows[1], ink_columns[0]:ink_columns[1]] = 240
    return image


class ApiSafetyTests(unittest.TestCase):
    def test_api_has_no_expected_text_parameter(self) -> None:
        parameters = inspect.signature(verify_accent_glyph).parameters
        self.assertEqual(list(parameters), ["glyph", "model"])
        for name in parameters:
            self.assertNotIn("expected", name.lower())
            self.assertNotIn("reference", name.lower())
            self.assertNotIn("truth", name.lower())

    def test_feature_extraction_takes_only_pixels(self) -> None:
        parameters = inspect.signature(extract_features).parameters
        self.assertEqual(list(parameters), ["glyph"])

    def test_verdicts_are_limited_to_three_values(self) -> None:
        verdict = verify_accent_glyph(glyph_image()).verdict
        self.assertIn(verdict, {ACCENT_PRESENT, ACCENT_ABSENT, UNKNOWN})


class FailClosedTests(unittest.TestCase):
    """Anything unmeasurable must come back unknown, never a guess."""

    def _model(self) -> AccentModel:
        return AccentModel(
            weights=(0.0,) * len(FEATURE_NAMES),
            bias=10.0,                      # would otherwise say "absent"
            absent_threshold=0.5,
            present_threshold=0.1,
            version="test",
        )

    def test_none_glyph_is_unknown(self) -> None:
        self.assertEqual(verify_accent_glyph(None, self._model()).verdict, UNKNOWN)

    def test_empty_glyph_is_unknown(self) -> None:
        empty = np.zeros((0, 0, 3), dtype=np.uint8)
        self.assertEqual(verify_accent_glyph(empty, self._model()).verdict, UNKNOWN)

    def test_blank_glyph_is_unknown(self) -> None:
        blank = np.zeros((24, 14, 3), dtype=np.uint8)
        self.assertEqual(verify_accent_glyph(blank, self._model()).verdict, UNKNOWN)

    def test_narrow_glyph_is_unknown(self) -> None:
        narrow = glyph_image(width=3, ink_columns=(0, 3))
        self.assertEqual(verify_accent_glyph(narrow, self._model()).verdict, UNKNOWN)

    def test_missing_model_yields_unknown(self) -> None:
        """A missing model must degrade to 'change nothing'."""
        result = verify_accent_glyph(glyph_image(), None)
        if load_model() is None:
            self.assertEqual(result.verdict, UNKNOWN)
            self.assertEqual(result.reason, "no_model_available")

    def test_extract_features_returns_none_for_unusable_input(self) -> None:
        self.assertIsNone(extract_features(None))
        self.assertIsNone(extract_features(np.zeros((0, 0, 3), np.uint8)))
        self.assertIsNone(extract_features(np.zeros((24, 14, 3), np.uint8)))


class DirectionTests(unittest.TestCase):
    """Only a confident `e` may license a change."""

    def test_is_accent_absent_only_true_for_absent_verdict(self) -> None:
        confident_absent = AccentModel(
            weights=(0.0,) * len(FEATURE_NAMES), bias=10.0,
            absent_threshold=0.5, present_threshold=0.1, version="test",
        )
        confident_present = AccentModel(
            weights=(0.0,) * len(FEATURE_NAMES), bias=-10.0,
            absent_threshold=0.9, present_threshold=0.5, version="test",
        )
        abstaining = AccentModel(
            weights=(0.0,) * len(FEATURE_NAMES), bias=0.0,
            absent_threshold=0.99, present_threshold=0.01, version="test",
        )
        glyph = glyph_image()
        self.assertTrue(verify_accent_glyph(glyph, confident_absent).is_accent_absent)
        self.assertFalse(verify_accent_glyph(glyph, confident_present).is_accent_absent)
        self.assertFalse(verify_accent_glyph(glyph, abstaining).is_accent_absent)

    def test_abstain_band_between_thresholds(self) -> None:
        model = AccentModel(
            weights=(0.0,) * len(FEATURE_NAMES), bias=0.0,   # probability 0.5
            absent_threshold=0.9, present_threshold=0.1, version="test",
        )
        result = verify_accent_glyph(glyph_image(), model)
        self.assertEqual(result.verdict, UNKNOWN)
        self.assertEqual(result.reason, "below_confidence_margin")


class FeatureTests(unittest.TestCase):
    def test_features_are_scale_invariant(self) -> None:
        """Doubling the glyph must not move the features much."""
        small = glyph_image(height=24, width=14, ink_rows=(8, 24), ink_columns=(2, 12))
        large = np.repeat(np.repeat(small, 2, axis=0), 2, axis=1)
        small_features = extract_features(small)
        large_features = extract_features(large)
        self.assertIsNotNone(small_features)
        self.assertIsNotNone(large_features)
        np.testing.assert_allclose(small_features, large_features, atol=0.12)

    def test_accent_mark_changes_the_separating_features(self) -> None:
        """A detached mark above the body must show up in the shape features.

        Note it *lowers* upper_ink_fraction rather than raising it: a small
        mark contributes little ink while stretching the box upward, whereas a
        solid body fills its own upper third. What identifies an accent is the
        gap beneath the mark, the taller box and the narrower upper band.
        """
        bare = glyph_image(height=30, ink_rows=(14, 28), ink_columns=(2, 12))
        accented = bare.copy()
        accented[3:7, 5:10] = 240          # accent mark above the body
        bare_features = extract_features(bare)
        accented_features = extract_features(accented)

        gap = FEATURE_NAMES.index("upper_gap_fraction")
        aspect = FEATURE_NAMES.index("aspect_ratio")
        upper_width = FEATURE_NAMES.index("upper_width_fraction")
        self.assertGreater(accented_features[gap], bare_features[gap])
        self.assertGreater(accented_features[aspect], bare_features[aspect])
        self.assertLess(accented_features[upper_width], bare_features[upper_width])

    def test_polarity_is_handled(self) -> None:
        """Dark-on-light must give the same features as light-on-dark."""
        dark_background = glyph_image(height=30, ink_rows=(14, 28))
        light_background = 255 - dark_background
        np.testing.assert_allclose(
            extract_features(dark_background),
            extract_features(light_background),
            atol=1e-9,
        )


class ShippedModelTests(unittest.TestCase):
    """The frozen model file must stay loadable and conservative."""

    def test_model_file_loads(self) -> None:
        if not MODEL_PATH.is_file():
            self.skipTest("no frozen model in this checkout")
        model = load_model(MODEL_PATH)
        self.assertIsNotNone(model)
        self.assertEqual(len(model.weights), len(FEATURE_NAMES))

    def test_model_feature_order_is_pinned(self) -> None:
        if not MODEL_PATH.is_file():
            self.skipTest("no frozen model in this checkout")
        payload = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
        self.assertEqual(tuple(payload["feature_names"]), FEATURE_NAMES)

    def test_model_rejects_mismatched_feature_order(self) -> None:
        with self.assertRaises(ValueError):
            AccentModel.from_dict(
                {
                    "feature_names": ["something_else"],
                    "weights": [0.0],
                    "bias": 0.0,
                    "absent_threshold": 0.5,
                    "present_threshold": 0.1,
                }
            )

    def test_absent_threshold_is_high(self) -> None:
        """Changing text requires real confidence, not a coin flip."""
        if not MODEL_PATH.is_file():
            self.skipTest("no frozen model in this checkout")
        model = load_model(MODEL_PATH)
        self.assertGreaterEqual(model.absent_threshold, 0.9)
        self.assertLess(model.present_threshold, model.absent_threshold)


if __name__ == "__main__":
    unittest.main()
