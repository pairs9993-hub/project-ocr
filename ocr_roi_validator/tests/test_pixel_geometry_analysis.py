"""Tests for the analysis gates that decide whether a verdict may be issued.

Both gates here exist because an earlier revision got the answer wrong in a way
that looked convincing. One divided by a zero-count bin, produced ``inf``, and
declared a dependence from thirteen events whose intervals all overlapped. The
other counted the recognizer's own misreads as pipeline bias, which would have
failed the diagnostic for correctly observing what the recognizer does.
"""

import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from analyze_pixel_geometry_v1 import (  # noqa: E402
    PIPELINE_ATTRITION_LIMIT, funnel_attrition, wilson, zero_upper_bound,
)


def funnel(eligible=1000, clean=400):
    return {"N_RENDERED_ELIGIBLE": eligible, "N_CLEAN_ELIGIBLE": clean}


def losses(**overrides):
    base = {"detector_miss": 0, "wrong_line_selected": 0,
            "recognizer_failure": 0, "alignment_ambiguity": 0,
            "render_or_detector_error": 0, "insertion": 0, "deletion": 0,
            "change_elsewhere": 0, "multiple_changes": 0, "accent_lost": 0,
            "other_substitution": 0}
    base.update(overrides)
    return base


class AttritionSplitTests(unittest.TestCase):
    def test_recognizer_misreads_do_not_gate_the_verdict(self) -> None:
        """600 dropped characters are the recognizer working, not selection."""
        split = funnel_attrition(funnel(1000, 400), losses(deletion=600))
        self.assertAlmostEqual(split["pipeline_attrition"], 0.0)
        self.assertAlmostEqual(split["recognizer_attrition"], 0.6)
        self.assertAlmostEqual(split["total_attrition"], 0.6)
        self.assertTrue(split["funnel_usable"])

    def test_pipeline_losses_do_gate_the_verdict(self) -> None:
        split = funnel_attrition(
            funnel(1000, 400), losses(detector_miss=350, wrong_line_selected=250))
        self.assertAlmostEqual(split["pipeline_attrition"], 0.6)
        self.assertFalse(split["funnel_usable"])

    def test_the_two_kinds_are_not_summed(self) -> None:
        """The bug being pinned: totalling both would fail a sound funnel."""
        split = funnel_attrition(
            funnel(1000, 380), losses(detector_miss=20, deletion=600))
        self.assertGreater(split["total_attrition"], PIPELINE_ATTRITION_LIMIT)
        self.assertLessEqual(split["pipeline_attrition"], PIPELINE_ATTRITION_LIMIT)
        self.assertTrue(split["funnel_usable"])

    def test_boundary_is_inclusive(self) -> None:
        at_limit = funnel_attrition(funnel(1000, 400), losses(detector_miss=200))
        over = funnel_attrition(funnel(1000, 400), losses(detector_miss=201))
        self.assertTrue(at_limit["funnel_usable"])
        self.assertFalse(over["funnel_usable"])

    def test_empty_funnel_does_not_divide_by_zero(self) -> None:
        split = funnel_attrition(funnel(0, 0), losses())
        self.assertEqual(split["pipeline_attrition"], 0.0)


class IntervalTests(unittest.TestCase):
    def test_zero_events_gives_a_bounded_interval_not_certainty(self) -> None:
        low, high = wilson(0, 200)
        self.assertEqual(low, 0.0)
        self.assertGreater(high, 0.0)
        self.assertLess(high, 0.05)

    def test_upper_bound_tightens_with_more_trials(self) -> None:
        self.assertGreater(zero_upper_bound(100), zero_upper_bound(1000))
        self.assertAlmostEqual(zero_upper_bound(1000), 0.002994, places=5)

    def test_small_samples_produce_wide_intervals(self) -> None:
        """Three events in thirty trials must not look like a firm rate."""
        low, high = wilson(3, 30)
        self.assertLess(low, 0.10)
        self.assertGreater(high, 0.20)

    def test_no_trials_spans_the_whole_range(self) -> None:
        self.assertEqual(wilson(0, 0), (0.0, 1.0))


if __name__ == "__main__":
    unittest.main()
