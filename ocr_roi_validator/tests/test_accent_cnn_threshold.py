"""Tests for the accent CNN threshold and decision rules.

The threshold is the safety mechanism: it must sit strictly above every
probability the model assigns to a glyph that genuinely has an accent, so no
real diacritic can be converted. These tests pin that property and the
three-way decision derived from it, independently of any trained weights.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from check_accent_onnx_parity import decisions, softmax_bare  # noqa: E402
from train_accent_cnn import choose_threshold  # noqa: E402


class ThresholdSelectionTests(unittest.TestCase):
    def test_threshold_sits_above_every_real_accent(self) -> None:
        # label 0 = has an accent, label 1 = bare
        probabilities = np.array([0.10, 0.42, 0.88, 0.95, 0.97, 0.99])
        labels = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        threshold, stats = choose_threshold(probabilities, labels)
        self.assertGreater(threshold, probabilities[labels == 0.0].max())
        self.assertEqual(stats["validation_false_corrections"], 0)

    def test_no_false_corrections_at_chosen_threshold(self) -> None:
        rng = np.random.default_rng(0)
        accent = rng.uniform(0.0, 0.9, size=400)
        bare = rng.uniform(0.5, 1.0, size=400)
        probabilities = np.concatenate([accent, bare])
        labels = np.concatenate([np.zeros(400), np.ones(400)])
        threshold, stats = choose_threshold(probabilities, labels)
        converted = probabilities[labels == 0.0] >= threshold
        self.assertEqual(int(converted.sum()), 0)
        self.assertEqual(stats["validation_false_corrections"], 0)

    def test_overlapping_distributions_collapse_coverage(self) -> None:
        """If a real accent scores highest, safety costs all coverage.

        That is the intended behaviour: abstaining everywhere is safe, and the
        gate should surface it rather than trade it away.
        """
        probabilities = np.array([0.99, 0.20, 0.30, 0.40])
        labels = np.array([0.0, 1.0, 1.0, 1.0])
        threshold, stats = choose_threshold(probabilities, labels)
        self.assertGreater(threshold, 0.99)
        self.assertEqual(stats["validation_bare_coverage_at_threshold"], 0.0)

    def test_threshold_is_not_capped_below_the_worst_accent(self) -> None:
        """Regression: a tidy cap silently re-admitted forbidden corrections.

        An earlier version clamped the threshold to 0.9995. When the model
        assigned 0.9999929 to a real accent, that cap sat *below* it and 20
        genuine accents became convertible. The threshold must clear the worst
        real accent no matter how close to 1.0 that is.
        """
        probabilities = np.array([0.9999929, 0.99999])
        labels = np.array([0.0, 1.0])
        threshold, stats = choose_threshold(probabilities, labels)
        self.assertGreater(threshold, 0.9999929)
        self.assertEqual(stats["validation_false_corrections"], 0)

    def test_saturated_probabilities_still_yield_zero_false_corrections(self) -> None:
        probabilities = np.array([1.0, 1.0])
        labels = np.array([0.0, 1.0])
        threshold, stats = choose_threshold(probabilities, labels)
        self.assertEqual(stats["validation_false_corrections"], 0)
        self.assertEqual(stats["validation_bare_coverage_at_threshold"], 0.0)


class DecisionRuleTests(unittest.TestCase):
    def test_three_way_split(self) -> None:
        probabilities = np.array([0.01, 0.50, 0.99])
        verdicts = decisions(probabilities, absent=0.95, present=0.05)
        # 0 = accent present, 1 = absent (correct), 2 = unknown
        np.testing.assert_array_equal(verdicts, np.array([0, 2, 1]))

    def test_abstain_band_is_inclusive_of_the_middle(self) -> None:
        probabilities = np.array([0.94, 0.06])
        verdicts = decisions(probabilities, absent=0.95, present=0.05)
        np.testing.assert_array_equal(verdicts, np.array([2, 2]))

    def test_only_absent_verdict_licenses_a_change(self) -> None:
        probabilities = np.linspace(0.0, 1.0, 21)
        verdicts = decisions(probabilities, absent=0.95, present=0.05)
        changed = probabilities[verdicts == 1]
        self.assertTrue((changed >= 0.95).all())


class SoftmaxTests(unittest.TestCase):
    def test_matches_manual_softmax(self) -> None:
        logits = np.array([[2.0, 1.0], [0.0, 3.0]])
        expected = np.array([
            np.exp(1.0) / (np.exp(2.0) + np.exp(1.0)),
            np.exp(3.0) / (np.exp(0.0) + np.exp(3.0)),
        ])
        np.testing.assert_allclose(softmax_bare(logits), expected, rtol=1e-9)

    def test_is_stable_for_large_logits(self) -> None:
        logits = np.array([[1000.0, 999.0]])
        value = softmax_bare(logits)
        self.assertTrue(np.isfinite(value).all())
        self.assertGreaterEqual(float(value[0]), 0.0)
        self.assertLessEqual(float(value[0]), 1.0)


if __name__ == "__main__":
    unittest.main()
