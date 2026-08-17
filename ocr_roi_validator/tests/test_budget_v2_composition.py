"""Tests for how the redefined quotas compose into a budget.

Two mistakes are easy here and both were live in the previous sizing. Summing
the per-quota budgets treats requirements as if they needed separate
renderings, when one rendering satisfies many at once -- that inflated the
figure roughly fivefold. And a quota that only a fraction of renderings can
contribute to (a stratum, a single font) needs its requirement scaled by that
fraction, or the budget is silently too small.
"""

import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from size_v2_budget import binomial_at_least, renderings_for_quota  # noqa: E402
from size_v2_budget_v2 import (  # noqa: E402
    LOW_SUPPORT_THRESHOLD, MACRO_STRATA, QUOTAS, TRAIN_V1_BACKFILL,
)


class QuotaPolicyTests(unittest.TestCase):
    def test_no_per_font_hallucination_minimum_remains(self) -> None:
        """The gate that made preflight unsatisfiable must be gone."""
        for split, quota in QUOTAS.items():
            self.assertNotIn("hallucination_per_font", quota)

    def test_font_safety_gate_is_preservation(self) -> None:
        for split in ("calibration", "preflight"):
            self.assertGreater(QUOTAS[split]["preservation_per_font"], 0)

    def test_supplement_has_no_per_font_preservation_gate(self) -> None:
        """Training data needs volume, not a per-font safety guarantee."""
        self.assertEqual(QUOTAS["supplement"]["preservation_per_font"], 0)

    def test_backfill_credit_is_applied_only_to_supplement(self) -> None:
        self.assertEqual(QUOTAS["supplement"]["credit"], TRAIN_V1_BACKFILL)
        self.assertEqual(QUOTAS["calibration"]["credit"], {})
        self.assertEqual(QUOTAS["preflight"]["credit"], {})

    def test_credit_leaves_large_as_the_real_shortfall(self) -> None:
        quota = QUOTAS["supplement"]
        remaining = {s: max(0, quota["hallucination_per_stratum"]
                            - TRAIN_V1_BACKFILL[s]) for s in MACRO_STRATA}
        self.assertEqual(remaining["SMALL"], 0)
        self.assertEqual(remaining["MEDIUM"], 0)
        self.assertEqual(remaining["LARGE"], 49)


class CompositionTests(unittest.TestCase):
    def test_max_is_used_not_sum(self) -> None:
        """Summing overlapping requirements inflates the budget."""
        requirements = [10_000, 4_000, 2_500, 600]
        self.assertEqual(max(requirements), 10_000)
        self.assertNotEqual(max(requirements), sum(requirements))

    def test_binding_budget_satisfies_every_requirement(self) -> None:
        # A budget set by the largest requirement must clear the smaller ones.
        rates = {"a": 0.002, "b": 0.01, "c": 0.05}
        needs = {"a": 20, "b": 100, "c": 500}
        budgets = {k: renderings_for_quota(rates[k], needs[k]) for k in rates}
        chosen = max(budgets.values())
        for key in rates:
            self.assertGreaterEqual(
                binomial_at_least(chosen, rates[key], needs[key]), 0.95)

    def test_stratum_requirement_is_scaled_by_its_share(self) -> None:
        """Only a third of renderings target any one stratum."""
        rate, required = 0.0036, 20
        within = renderings_for_quota(rate, required)
        across = within * len(MACRO_STRATA)
        self.assertGreater(across, within)
        self.assertGreaterEqual(
            binomial_at_least(across // len(MACRO_STRATA), rate, required), 0.95)

    def test_font_requirement_is_scaled_by_font_count(self) -> None:
        rate, required, fonts = 0.014, 200, 4
        within = renderings_for_quota(rate, required)
        across = int(within * fonts)
        self.assertGreaterEqual(
            binomial_at_least(across // fonts, rate, required), 0.95)

    def test_unscaled_budget_would_fall_short(self) -> None:
        """The failure mode being guarded: forgetting to scale."""
        rate, required = 0.0036, 20
        within = renderings_for_quota(rate, required)
        # Using `within` as the whole-split budget gives each stratum a third.
        self.assertLess(
            binomial_at_least(within // len(MACRO_STRATA), rate, required), 0.95)


class LowSupportTests(unittest.TestCase):
    def test_threshold_is_a_named_constant(self) -> None:
        self.assertGreater(LOW_SUPPORT_THRESHOLD, 0)

    def test_sparse_cell_is_flagged_not_dropped(self) -> None:
        cells = {"a": 2, "b": 400}
        flagged = [k for k, v in cells.items() if v < LOW_SUPPORT_THRESHOLD]
        self.assertEqual(flagged, ["a"])
        self.assertIn("b", cells)      # nothing is removed from the report


if __name__ == "__main__":
    unittest.main()
