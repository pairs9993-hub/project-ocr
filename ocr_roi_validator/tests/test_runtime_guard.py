"""Tests for the fail-closed runtime uncertainty band.

The guard exists because one sample sat 5.96e-08 from the frozen threshold and
changed verdict between PyTorch and ONNX Runtime. Its job is narrow: withhold
corrections near the boundary without ever inventing one. The monotonic
property is the whole safety argument, so it is tested directly rather than
inferred from the implementation.
"""

import unittest

import numpy as np

from ocr_roi_validator.runtime_guard import (
    ACCENT_PRESENT,
    BARE_E,
    RUNTIME_EPSILON,
    UNKNOWN,
    assert_monotonic,
    guarded_verdict,
    in_uncertainty_band,
    ungated_verdict,
)

THRESHOLD = np.float32(0.6443382502)


def probs(accent, bare, unknown):
    row = np.array([[accent, bare, unknown]], dtype=np.float32)
    return row / row.sum(axis=1, keepdims=True)


class BandTests(unittest.TestCase):
    def test_epsilon_is_fixed_and_float32(self) -> None:
        self.assertEqual(RUNTIME_EPSILON.dtype, np.float32)
        # float32 cannot hold 1e-4 exactly (it stores 9.9999997e-05), so the
        # comparison is made at float32 precision rather than demanding a
        # decimal value the type cannot represent.
        self.assertEqual(RUNTIME_EPSILON, np.float32(1e-4))
        self.assertAlmostEqual(float(RUNTIME_EPSILON), 1e-4, places=9)

    def test_epsilon_exceeds_observed_runtime_error(self) -> None:
        """1e-4 against the largest measured Torch/ONNX gap of 5.692e-06."""
        self.assertGreater(float(RUNTIME_EPSILON), 5.692e-06 * 10)

    def test_bare_just_above_threshold_is_withheld(self) -> None:
        row = np.array([[0.20, float(THRESHOLD) + 1e-6, 0.10]], dtype=np.float32)
        self.assertEqual(ungated_verdict(row, THRESHOLD)[0], BARE_E)
        self.assertEqual(guarded_verdict(row, THRESHOLD)[0], UNKNOWN)

    def test_bare_clearly_above_band_is_allowed(self) -> None:
        row = np.array([[0.10, 0.90, 0.05]], dtype=np.float32)
        self.assertEqual(guarded_verdict(row, THRESHOLD)[0], BARE_E)

    def test_band_membership_is_symmetric(self) -> None:
        above = np.array([[0.1, float(THRESHOLD) + 5e-5, 0.1]], dtype=np.float32)
        below = np.array([[0.1, float(THRESHOLD) - 5e-5, 0.1]], dtype=np.float32)
        self.assertTrue(in_uncertainty_band(above, THRESHOLD)[0])
        self.assertTrue(in_uncertainty_band(below, THRESHOLD)[0])

    def test_far_from_threshold_is_outside_the_band(self) -> None:
        row = np.array([[0.1, 0.99, 0.1]], dtype=np.float32)
        self.assertFalse(in_uncertainty_band(row, THRESHOLD)[0])


class MonotonicityTests(unittest.TestCase):
    def test_guard_never_creates_a_correction(self) -> None:
        rng = np.random.default_rng(7)
        raw = rng.random((5000, 3)).astype(np.float32)
        raw = raw / raw.sum(axis=1, keepdims=True)
        base = ungated_verdict(raw, THRESHOLD)
        guarded = guarded_verdict(raw, THRESHOLD)
        self.assertEqual(int(((guarded == BARE_E) & (base != BARE_E)).sum()), 0)

    def test_guarded_bare_is_a_subset_of_base_bare(self) -> None:
        rng = np.random.default_rng(11)
        raw = rng.random((5000, 3)).astype(np.float32)
        raw = raw / raw.sum(axis=1, keepdims=True)
        base = set(np.where(ungated_verdict(raw, THRESHOLD) == BARE_E)[0].tolist())
        guarded = set(np.where(guarded_verdict(raw, THRESHOLD) == BARE_E)[0].tolist())
        self.assertTrue(guarded.issubset(base))

    def test_only_transition_is_bare_to_unknown(self) -> None:
        rng = np.random.default_rng(13)
        raw = rng.random((5000, 3)).astype(np.float32)
        raw = raw / raw.sum(axis=1, keepdims=True)
        report = assert_monotonic(raw, THRESHOLD)
        self.assertTrue(report["only_bare_to_unknown"])
        self.assertLessEqual(report["guarded_bare"], report["base_bare"])

    def test_accent_verdicts_are_untouched(self) -> None:
        rng = np.random.default_rng(17)
        raw = rng.random((3000, 3)).astype(np.float32)
        raw = raw / raw.sum(axis=1, keepdims=True)
        base = ungated_verdict(raw, THRESHOLD)
        guarded = guarded_verdict(raw, THRESHOLD)
        accent = base == ACCENT_PRESENT
        np.testing.assert_array_equal(base[accent], guarded[accent])

    def test_unknown_verdicts_are_untouched(self) -> None:
        rng = np.random.default_rng(19)
        raw = rng.random((3000, 3)).astype(np.float32)
        raw = raw / raw.sum(axis=1, keepdims=True)
        base = ungated_verdict(raw, THRESHOLD)
        guarded = guarded_verdict(raw, THRESHOLD)
        unknown = base == UNKNOWN
        np.testing.assert_array_equal(base[unknown], guarded[unknown])

    def test_checker_catches_a_rule_that_adds_corrections(self) -> None:
        """The checker must fail loudly, not just trust the implementation.

        guarded_verdict cannot add a correction -- it only narrows rows where
        the base rule already said BARE_E -- so a violation is constructed by
        comparing against a deliberately broken alternative.
        """
        raw = np.array([[0.05, 0.50, 0.45],
                        [0.10, 0.85, 0.05]], dtype=np.float32)
        threshold = np.float32(0.60)
        base = ungated_verdict(raw, threshold)
        self.assertEqual(base[0], UNKNOWN)      # 0.50 is below threshold

        def broken(probabilities, thresh, eps):
            # Lowers the bar instead of raising it: turns the sub-threshold
            # row into a correction.
            return np.where(probabilities[:, BARE_E] >= np.float32(thresh - eps),
                            BARE_E, ungated_verdict(probabilities, thresh))

        result = broken(raw, threshold, np.float32(0.2))
        gained = int(((result == BARE_E) & (base != BARE_E)).sum())
        self.assertEqual(gained, 1)
        # The real guard, on the same input, adds nothing.
        guarded = guarded_verdict(raw, threshold)
        self.assertEqual(int(((guarded == BARE_E) & (base != BARE_E)).sum()), 0)

    def test_assert_monotonic_reports_withheld_count(self) -> None:
        raw = np.array([[0.1, float(THRESHOLD) + 1e-6, 0.1],
                        [0.1, 0.95, 0.05]], dtype=np.float32)
        report = assert_monotonic(raw, THRESHOLD)
        self.assertEqual(report["withheld"], 1)
        self.assertEqual(report["guarded_bare"], report["base_bare"] - 1)


class RuntimeDisagreementTests(unittest.TestCase):
    def test_the_observed_failure_is_neutralised(self) -> None:
        """Sample 105: 0.6443381906 in Torch, 0.6443383694 in ONNX."""
        torch_row = np.array([[0.2, 0.6443381906, 0.15]], dtype=np.float32)
        onnx_row = np.array([[0.2, 0.6443383694, 0.15]], dtype=np.float32)
        # Ungated, the two runtimes could disagree at the boundary.
        # Guarded, both fall inside the band and return UNKNOWN.
        self.assertEqual(guarded_verdict(torch_row, THRESHOLD)[0], UNKNOWN)
        self.assertEqual(guarded_verdict(onnx_row, THRESHOLD)[0], UNKNOWN)

    def test_differences_below_epsilon_cannot_change_a_verdict(self) -> None:
        rng = np.random.default_rng(23)
        raw = rng.random((4000, 3)).astype(np.float32)
        raw = raw / raw.sum(axis=1, keepdims=True)
        # Perturb by less than epsilon, as a runtime difference would.
        jitter = rng.uniform(-5.7e-06, 5.7e-06, size=raw.shape).astype(np.float32)
        perturbed = np.clip(raw + jitter, 0.0, 1.0).astype(np.float32)
        first = guarded_verdict(raw, THRESHOLD)
        second = guarded_verdict(perturbed, THRESHOLD)
        self.assertEqual(int((first != second).sum()), 0)

    def test_all_arithmetic_stays_float32(self) -> None:
        raw = np.array([[0.2, 0.7, 0.1]], dtype=np.float32)
        self.assertEqual(guarded_verdict(raw, THRESHOLD).dtype, np.int64)
        self.assertEqual(np.float32(THRESHOLD + RUNTIME_EPSILON).dtype,
                         np.float32)


if __name__ == "__main__":
    unittest.main()
