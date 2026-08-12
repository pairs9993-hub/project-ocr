"""Tests for the accent-v3 decision path.

These pin the safety contract, which is independent of how well the network
classifies: a correction requires every check to agree, any single abstention
blocks it, and an accent can never be added. Accuracy is measured by the
synthetic holdout gate, not here.
"""

import inspect
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ocr_roi_validator.accent_cnn_input import AccentInputConfig
from ocr_roi_validator.accent_cnn_verifier import (
    ACCENT_ABSENT,
    ACCENT_PRESENT,
    UNKNOWN,
    AccentCnnVerdict,
    AccentCnnVerifier,
)
from ocr_roi_validator.accent_localization import LocalizationConfig

MODEL = Path(__file__).resolve().parents[1] / "ocr_roi_validator"
CONFIG_PATH = MODEL / "accent_cnn_config.json"


def line_with_glyph(width=120, height=30, x0=22, x1=32, y0=8, y1=24):
    """Dark line with a single centred ink block inside the span."""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[y0:y1, x0:x1] = 240
    return image


class StubVerifier(AccentCnnVerifier):
    """Verifier with the network replaced by a fixed probability."""

    def __init__(self, probabilities, absent=0.9, present=0.05):
        self._probabilities = probabilities
        self.version = "stub"
        self.absent_threshold = absent
        self.present_threshold = present
        self.input_config = AccentInputConfig()
        self.localization_config = LocalizationConfig()
        self._onnx_path = Path("unused.onnx")
        self._session = object()
        self.calls = 0

    def _run(self, batch):
        self.calls += 1
        values = self._probabilities
        if callable(values):
            return np.asarray(values(batch), dtype=np.float64)
        if np.isscalar(values):
            return np.full(batch.shape[0], float(values))
        return np.asarray(values, dtype=np.float64)[: batch.shape[0]]


class ApiSafetyTests(unittest.TestCase):
    def test_verify_has_no_expected_text_parameter(self) -> None:
        parameters = inspect.signature(AccentCnnVerifier.verify).parameters
        self.assertEqual(
            list(parameters),
            ["self", "line_image", "x0", "x1", "median_span_width"],
        )
        for name in parameters:
            self.assertNotIn("expected", name.lower())
            self.assertNotIn("truth", name.lower())
            self.assertNotIn("label", name.lower())

    def test_verdicts_are_limited_to_three_values(self) -> None:
        verifier = StubVerifier(0.99)
        verdict = verifier.verify(line_with_glyph(), 20, 34).verdict
        self.assertIn(verdict, {ACCENT_ABSENT, ACCENT_PRESENT, UNKNOWN})

    def test_only_absent_licenses_a_change(self) -> None:
        self.assertTrue(AccentCnnVerdict(ACCENT_ABSENT, 1.0, "x").is_accent_absent)
        self.assertFalse(AccentCnnVerdict(ACCENT_PRESENT, 0.0, "x").is_accent_absent)
        self.assertFalse(AccentCnnVerdict(UNKNOWN, 0.5, "x").is_accent_absent)


class DecisionPathTests(unittest.TestCase):
    def test_all_views_confident_yields_a_correction(self) -> None:
        verifier = StubVerifier(0.99)
        result = verifier.verify(line_with_glyph(), 20, 34)
        self.assertEqual(result.verdict, ACCENT_ABSENT)
        self.assertEqual(result.reason, "all_views_confident_absent")
        self.assertGreater(len(result.view_probabilities), 1)

    def test_one_unconfident_view_blocks_the_correction(self) -> None:
        """A verdict that depends on the crop bounds must not be acted on."""
        verifier = StubVerifier([0.99, 0.99, 0.50, 0.99, 0.99])
        result = verifier.verify(line_with_glyph(), 20, 34)
        self.assertEqual(result.verdict, UNKNOWN)
        self.assertEqual(result.reason, "view_disagreement")

    def test_accent_present_on_any_view_vetoes(self) -> None:
        verifier = StubVerifier([0.99, 0.99, 0.01, 0.99, 0.99])
        result = verifier.verify(line_with_glyph(), 20, 34)
        self.assertEqual(result.verdict, UNKNOWN)
        self.assertEqual(result.reason, "accent_present_veto")
        self.assertFalse(result.is_accent_absent)

    def test_confident_accent_on_primary_view_reports_present(self) -> None:
        verifier = StubVerifier(0.01)
        result = verifier.verify(line_with_glyph(), 20, 34)
        self.assertEqual(result.verdict, ACCENT_PRESENT)

    def test_middling_confidence_abstains(self) -> None:
        verifier = StubVerifier(0.5)
        result = verifier.verify(line_with_glyph(), 20, 34)
        self.assertEqual(result.verdict, UNKNOWN)
        self.assertEqual(result.reason, "below_confidence_margin")


class FailClosedTests(unittest.TestCase):
    def test_localization_rejection_blocks_before_inference(self) -> None:
        verifier = StubVerifier(0.999)
        # Ink spanning both edges is rejected by the guard.
        line = line_with_glyph(x0=20, x1=34)
        result = verifier.verify(line, 20, 34)
        self.assertEqual(result.verdict, UNKNOWN)
        self.assertTrue(result.reason.startswith("localization:"))
        self.assertEqual(verifier.calls, 0, "network must not run on a bad crop")

    def test_empty_line_abstains(self) -> None:
        verifier = StubVerifier(0.999)
        self.assertEqual(
            verifier.verify(np.zeros((0, 0, 3), np.uint8), 0, 5).verdict, UNKNOWN
        )

    def test_none_line_abstains(self) -> None:
        verifier = StubVerifier(0.999)
        self.assertEqual(verifier.verify(None, 0, 5).verdict, UNKNOWN)

    def test_unpreparable_view_abstains(self) -> None:
        verifier = StubVerifier(0.999)
        blank = np.zeros((30, 120, 3), np.uint8)
        result = verifier.verify(blank, 20, 34)
        self.assertEqual(result.verdict, UNKNOWN)

    def test_threshold_boundary_is_inclusive_for_absent(self) -> None:
        verifier = StubVerifier(0.9, absent=0.9)
        self.assertEqual(
            verifier.verify(line_with_glyph(), 20, 34).verdict, ACCENT_ABSENT
        )

    def test_just_below_threshold_abstains(self) -> None:
        verifier = StubVerifier(0.8999, absent=0.9)
        self.assertEqual(
            verifier.verify(line_with_glyph(), 20, 34).verdict, UNKNOWN
        )


class FrozenConfigTests(unittest.TestCase):
    def test_shipped_config_loads_and_is_conservative(self) -> None:
        if not CONFIG_PATH.is_file():
            self.skipTest("no frozen accent-v3 config in this checkout")
        settings = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        self.assertGreaterEqual(settings["absent_threshold"], 0.9)
        self.assertLess(settings["present_threshold"], settings["absent_threshold"])
        self.assertEqual(settings["opset"], 11)

    def test_config_records_training_provenance(self) -> None:
        if not CONFIG_PATH.is_file():
            self.skipTest("no frozen accent-v3 config in this checkout")
        settings = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        training = settings["training"]
        for field in ("python", "torch", "seed", "deterministic"):
            self.assertIn(field, training)

    def test_missing_model_file_is_reported_on_use(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            config = Path(temp) / "config.json"
            config.write_text(
                json.dumps(
                    {
                        "version": "test",
                        "absent_threshold": 0.9,
                        "present_threshold": 0.05,
                        "input_config": AccentInputConfig().as_dict(),
                    }
                ),
                encoding="utf-8",
            )
            verifier = AccentCnnVerifier(Path(temp) / "missing.onnx", config)
            with self.assertRaises(Exception):
                verifier.verify(line_with_glyph(), 20, 34)


if __name__ == "__main__":
    unittest.main()
