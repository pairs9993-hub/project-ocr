"""Tests for the small-sample interaction statistics.

These are hand-rolled because the product venv has neither statsmodels nor
scipy, so they are checked against values computed independently: textbook
Fisher tables, chi-square tail probabilities from published tables, and
logistic fits whose coefficients follow from the cell odds directly.

The behaviour that matters most is the ability to return "not enough
evidence". A test that always produces a verdict would let fifty events spread
over eighteen cells masquerade as a finding.
"""

import math
import unittest

import numpy as np

from ocr_roi_validator.interaction_stats import (
    chi_square_sf,
    detect_separation,
    fisher_exact,
    fit_logistic,
    holm_adjust,
    likelihood_ratio_test,
)


class ChiSquareTests(unittest.TestCase):
    """Checked against standard chi-square tables."""

    def test_known_upper_tail_values(self) -> None:
        self.assertAlmostEqual(chi_square_sf(3.841, 1), 0.05, places=3)
        self.assertAlmostEqual(chi_square_sf(5.991, 2), 0.05, places=3)
        self.assertAlmostEqual(chi_square_sf(9.488, 4), 0.05, places=3)
        self.assertAlmostEqual(chi_square_sf(6.635, 1), 0.01, places=3)

    def test_zero_statistic_is_certain(self) -> None:
        self.assertEqual(chi_square_sf(0.0, 3), 1.0)
        self.assertEqual(chi_square_sf(-1.0, 3), 1.0)

    def test_no_degrees_of_freedom(self) -> None:
        self.assertEqual(chi_square_sf(5.0, 0), 1.0)

    def test_tail_decreases_as_statistic_grows(self) -> None:
        values = [chi_square_sf(s, 2) for s in (1, 3, 6, 12, 25)]
        self.assertEqual(values, sorted(values, reverse=True))


class FisherTests(unittest.TestCase):
    def test_tea_tasting_table(self) -> None:
        """Fisher's original 3/1 vs 1/3 table: two-sided p = 0.4857."""
        self.assertAlmostEqual(fisher_exact(((3, 1), (1, 3))), 0.4857, places=4)

    def test_strong_association(self) -> None:
        self.assertAlmostEqual(fisher_exact(((10, 0), (0, 10))),
                               1.0825e-05, places=8)

    def test_no_association_is_not_significant(self) -> None:
        self.assertGreater(fisher_exact(((5, 5), (5, 5))), 0.9)

    def test_degenerate_tables_return_one(self) -> None:
        self.assertEqual(fisher_exact(((0, 0), (0, 0))), 1.0)
        self.assertEqual(fisher_exact(((0, 0), (5, 5))), 1.0)

    def test_sparse_table_cannot_reach_significance(self) -> None:
        """One event against zero proves nothing, and must not claim to."""
        self.assertGreater(fisher_exact(((1, 100), (0, 100))), 0.30)


class HolmTests(unittest.TestCase):
    def test_smallest_gets_the_full_penalty(self) -> None:
        adjusted = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03})
        self.assertAlmostEqual(adjusted["a"], 0.03)

    def test_adjustment_is_monotone(self) -> None:
        raw = {"a": 0.001, "b": 0.02, "c": 0.03, "d": 0.04}
        adjusted = holm_adjust(raw)
        ordered = [adjusted[k] for k, _ in sorted(raw.items(), key=lambda i: i[1])]
        self.assertEqual(ordered, sorted(ordered))

    def test_values_are_capped_at_one(self) -> None:
        adjusted = holm_adjust({"a": 0.6, "b": 0.7, "c": 0.9})
        self.assertTrue(all(v <= 1.0 for v in adjusted.values()))

    def test_empty_family(self) -> None:
        self.assertEqual(holm_adjust({}), {})


class LogisticTests(unittest.TestCase):
    def test_recovers_a_known_intercept(self) -> None:
        """200 trials at p=0.25 gives an intercept of log(0.25/0.75)."""
        outcome = np.array([1.0] * 50 + [0.0] * 150)
        design = np.ones((200, 1))
        fit = fit_logistic(design, outcome)
        self.assertTrue(fit.converged)
        self.assertAlmostEqual(fit.coefficients[0], math.log(50 / 150), places=3)

    def test_recovers_a_known_slope(self) -> None:
        # group 0: 10/100, group 1: 30/100 -> log odds ratio
        outcome = np.array([1.0] * 10 + [0.0] * 90 + [1.0] * 30 + [0.0] * 70)
        design = np.zeros((200, 2))
        design[:, 0] = 1.0
        design[100:, 1] = 1.0
        fit = fit_logistic(design, outcome)
        expected = math.log((30 / 70) / (10 / 90))
        self.assertAlmostEqual(fit.coefficients[1], expected, places=2)

    def test_log_likelihood_is_negative_and_finite(self) -> None:
        outcome = np.array([1.0] * 20 + [0.0] * 80)
        fit = fit_logistic(np.ones((100, 1)), outcome)
        self.assertLess(fit.log_likelihood, 0.0)
        self.assertTrue(math.isfinite(fit.log_likelihood))

    def test_separation_is_flagged_not_hidden(self) -> None:
        outcome = np.array([0.0] * 50 + [1.0] * 50)
        design = np.zeros((100, 2))
        design[:, 0] = 1.0
        design[50:, 1] = 1.0
        self.assertTrue(fit_logistic(design, outcome).separated)


class LikelihoodRatioTests(unittest.TestCase):
    def test_added_noise_parameter_is_not_significant(self) -> None:
        rng = np.random.default_rng(11)
        outcome = (rng.random(400) < 0.2).astype(float)
        base = np.ones((400, 1))
        noise = np.column_stack([base, rng.normal(size=400)])
        result = likelihood_ratio_test(fit_logistic(base, outcome),
                                       fit_logistic(noise, outcome))
        self.assertEqual(result["degrees_of_freedom"], 1)
        self.assertGreater(result["p_value"], 0.05)

    def test_real_effect_is_detected(self) -> None:
        outcome = np.array([1.0] * 10 + [0.0] * 190 + [1.0] * 60 + [0.0] * 140)
        base = np.ones((400, 1))
        group = np.zeros((400, 2))
        group[:, 0] = 1.0
        group[200:, 1] = 1.0
        result = likelihood_ratio_test(fit_logistic(base, outcome),
                                       fit_logistic(group, outcome))
        self.assertLess(result["p_value"], 0.001)

    def test_statistic_is_never_negative(self) -> None:
        outcome = np.array([1.0] * 5 + [0.0] * 95)
        fit = fit_logistic(np.ones((100, 1)), outcome)
        self.assertGreaterEqual(likelihood_ratio_test(fit, fit)["statistic"], 0.0)


class SeparationTests(unittest.TestCase):
    def test_empty_cells_are_reported(self) -> None:
        report = detect_separation({"a": (100, 5), "b": (100, 0), "c": (0, 0)})
        self.assertIn("b", report["cells_with_zero_events"])
        self.assertIn("c", report["cells_with_no_exposure"])
        self.assertTrue(report["quasi_separation_risk"])

    def test_healthy_table_has_no_risk(self) -> None:
        report = detect_separation({"a": (100, 20), "b": (100, 30)})
        self.assertFalse(report["quasi_separation_risk"])

    def test_sparse_cells_are_listed(self) -> None:
        report = detect_separation({"a": (100, 3), "b": (100, 40)})
        self.assertEqual(report["cells_with_fewer_than_five_events"], ["a"])


if __name__ == "__main__":
    unittest.main()
