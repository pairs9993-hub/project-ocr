"""Tests fixing the terminal-reason / diagnostic-flag separation.

The invariant is the point: terminal reasons partition the rows and must
reconcile exactly, while flags may co-occur and must never be used as a
denominator. The v2 report's loss dictionary summed to 16,564 against 19,200
rows, which was correct arithmetic on an incomplete list but indistinguishable
from double counting -- these tests make the difference checkable.
"""

import unittest

from ocr_roi_validator.terminal_reason import (
    CLEAN_TERMINALS,
    DIAGNOSTIC_FLAGS,
    PIPELINE_TERMINALS,
    RECOGNIZER_TERMINALS,
    TERMINAL_REASONS,
    assert_terminal_reason_invariant,
    derive_flags,
    summarise_terminal_reasons,
)


def row(outcome, **extra):
    base = {"outcome": outcome, "decoded_length": None,
            "expected_substring": None, "detector_box_count": 1,
            "clipped": False, "horizontal_padding_ratio": 0.2}
    base.update(extra)
    return base


class TaxonomyTests(unittest.TestCase):
    def test_groups_are_disjoint(self) -> None:
        groups = (set(PIPELINE_TERMINALS), set(RECOGNIZER_TERMINALS),
                  set(CLEAN_TERMINALS))
        for first in range(len(groups)):
            for second in range(first + 1, len(groups)):
                self.assertEqual(groups[first] & groups[second], set())

    def test_terminal_reasons_are_unique(self) -> None:
        self.assertEqual(len(TERMINAL_REASONS), len(set(TERMINAL_REASONS)))

    def test_flags_and_terminals_do_not_share_names(self) -> None:
        """A name in both systems would make the two impossible to tell apart."""
        self.assertEqual(set(DIAGNOSTIC_FLAGS) & set(TERMINAL_REASONS), set())


class InvariantTests(unittest.TestCase):
    def test_counts_reconcile_with_row_total(self) -> None:
        rows = [row("DELETION")] * 5 + [row("CLEAN_HALLUCINATION")] * 2
        counts = assert_terminal_reason_invariant(rows)
        self.assertEqual(sum(counts.values()), 7)

    def test_unknown_reason_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            assert_terminal_reason_invariant([row("SOMETHING_ELSE")])

    def test_missing_reason_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            assert_terminal_reason_invariant([{"decoded_length": 3}])

    def test_explicit_field_takes_precedence(self) -> None:
        rows = [{"terminal_reason": "DETECTOR_MISS", "outcome": "IGNORED"}]
        self.assertEqual(assert_terminal_reason_invariant(rows),
                         {"DETECTOR_MISS": 1})


class SummaryTests(unittest.TestCase):
    def test_summary_reconciles(self) -> None:
        rows = ([row("DELETION")] * 10 + [row("DETECTOR_MISS")] * 3
                + [row("CLEAN_PRESERVATION")] * 4)
        summary = summarise_terminal_reasons(rows)
        self.assertTrue(summary["reconciles"])
        self.assertEqual(summary["terminal_reason_total"], 17)
        self.assertEqual(summary["row_total"], 17)
        self.assertEqual(sum(summary["group_totals"].values()), 17)

    def test_group_totals_partition_the_rows(self) -> None:
        rows = [row("DELETION"), row("DETECTOR_MISS"), row("CLEAN_HALLUCINATION")]
        totals = summarise_terminal_reasons(rows)["group_totals"]
        self.assertEqual(totals, {"pipeline": 1, "recognizer": 1, "clean": 1})

    def test_flag_total_may_exceed_row_total(self) -> None:
        """Flags count observations; exceeding the row count is expected."""
        rows = [row("MULTIPLE_CHANGES", detector_box_count=3, clipped=True,
                    expected_substring="reglage", decoded_length=5,
                    horizontal_padding_ratio=0.8)]
        summary = summarise_terminal_reasons(rows)
        self.assertEqual(summary["row_total"], 1)
        self.assertGreater(summary["diagnostic_flag_total"], 1)
        self.assertTrue(summary["flags_are_multi_label"])


class FlagDerivationTests(unittest.TestCase):
    def test_split_line_is_flagged(self) -> None:
        self.assertIn("DETECTOR_SPLIT_LINE",
                      derive_flags(row("DELETION", detector_box_count=4)))

    def test_shorter_decode_flags_deletion(self) -> None:
        flags = derive_flags(row("DELETION", expected_substring="reglage",
                                 decoded_length=4))
        self.assertIn("DELETION_PRESENT", flags)
        self.assertNotIn("INSERTION_PRESENT", flags)

    def test_longer_decode_flags_insertion(self) -> None:
        flags = derive_flags(row("INSERTION", expected_substring="re",
                                 decoded_length=6))
        self.assertIn("INSERTION_PRESENT", flags)

    def test_multiple_e_forms_are_flagged(self) -> None:
        self.assertIn("MULTIPLE_E_FORMS_IN_LINE",
                      derive_flags(row("DELETION", expected_substring="réglagé")))
        self.assertNotIn("MULTIPLE_E_FORMS_IN_LINE",
                         derive_flags(row("DELETION", expected_substring="bd 1,5")))

    def test_clean_outcomes_flag_unchanged_target(self) -> None:
        self.assertIn("TARGET_UNCHANGED", derive_flags(row("CLEAN_PRESERVATION")))

    def test_high_padding_ratio_threshold(self) -> None:
        self.assertNotIn("HIGH_PADDING_RATIO",
                         derive_flags(row("DELETION", horizontal_padding_ratio=0.5)))
        self.assertIn("HIGH_PADDING_RATIO",
                      derive_flags(row("DELETION", horizontal_padding_ratio=0.51)))


if __name__ == "__main__":
    unittest.main()
