"""Tests for quota-success budget sizing.

The bug being guarded against is subtle and expensive: sizing a budget as
``required / rate`` gives the N where the expected event count equals the quota,
and at that N the quota is met only about half the time. These tests pin that
the returned N actually reaches 95% success, and that a cell with no observed
events yields no budget rather than a fabricated one.
"""

import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from size_v2_budget import (  # noqa: E402
    binomial_at_least, renderings_for_quota, wilson_lower,
)


class BinomialTests(unittest.TestCase):
    def test_known_probabilities(self) -> None:
        # P[Bin(10, 0.5) >= 5] = 0.623046875
        self.assertAlmostEqual(binomial_at_least(10, 0.5, 5), 0.623046875,
                               places=9)
        # P[Bin(1, 0.3) >= 1] = 0.3
        self.assertAlmostEqual(binomial_at_least(1, 0.3, 1), 0.3, places=10)

    def test_expectation_matching_n_is_about_half(self) -> None:
        """The precise reason required/rate is the wrong budget."""
        probability = binomial_at_least(1000, 0.06, 60)
        self.assertGreater(probability, 0.4)
        self.assertLess(probability, 0.6)

    def test_degenerate_inputs(self) -> None:
        self.assertEqual(binomial_at_least(100, 0.0, 1), 0.0)
        self.assertEqual(binomial_at_least(10, 0.5, 0), 1.0)
        self.assertEqual(binomial_at_least(3, 0.5, 5), 0.0)

    def test_probability_increases_with_n(self) -> None:
        values = [binomial_at_least(n, 0.01, 10) for n in (500, 1000, 2000, 4000)]
        self.assertEqual(values, sorted(values))


class WilsonLowerTests(unittest.TestCase):
    def test_zero_events_gives_zero_lower_bound(self) -> None:
        self.assertEqual(wilson_lower(0, 5000), 0.0)

    def test_lower_bound_is_below_the_point_estimate(self) -> None:
        self.assertLess(wilson_lower(30, 1000), 30 / 1000)

    def test_bound_tightens_with_more_data(self) -> None:
        self.assertLess(wilson_lower(3, 100), wilson_lower(30, 1000))

    def test_no_trials(self) -> None:
        self.assertEqual(wilson_lower(0, 0), 0.0)


class BudgetTests(unittest.TestCase):
    def test_returned_n_actually_reaches_95_percent(self) -> None:
        for rate, required in ((0.01, 60), (0.002, 20), (0.05, 100)):
            n = renderings_for_quota(rate, required)
            self.assertIsNotNone(n)
            self.assertGreaterEqual(binomial_at_least(n, rate, required), 0.95)

    def test_n_is_minimal(self) -> None:
        """One fewer rendering must fall short of the requirement."""
        n = renderings_for_quota(0.01, 60)
        self.assertLess(binomial_at_least(n - 1, 0.01, 60), 0.95)

    def test_exceeds_the_expectation_matching_budget(self) -> None:
        rate, required = 0.002, 20
        expectation_n = required / rate
        self.assertGreater(renderings_for_quota(rate, required), expectation_n)

    def test_zero_rate_yields_no_budget(self) -> None:
        """A rate that might be zero cannot produce an N."""
        self.assertIsNone(renderings_for_quota(0.0, 20))

    def test_absurdly_small_rate_is_refused_not_extrapolated(self) -> None:
        self.assertIsNone(renderings_for_quota(1e-9, 60))

    def test_higher_confidence_costs_more(self) -> None:
        modest = renderings_for_quota(0.01, 60, confidence=0.80)
        strict = renderings_for_quota(0.01, 60, confidence=0.95)
        self.assertGreater(strict, modest)


if __name__ == "__main__":
    unittest.main()
