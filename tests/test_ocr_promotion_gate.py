from __future__ import annotations

import unittest

from ocr_validator.promotion_gate import (
    align_characters,
    cer,
    character_metrics,
    compare_runs,
    defect_false_passes,
)


class OCRPromotionGateTests(unittest.TestCase):
    def test_cer_preserves_diacritic_differences(self) -> None:
        self.assertEqual(cer("l’eau", "l'eau"), 0.0)
        self.assertGreater(cer("Veuillez", "Véuillez"), 0.0)

    def test_alignment_preserves_accent_and_i_l_substitutions(self) -> None:
        alignment = align_characters("Veuillez alimentation", "Véuillez alimentatlon")
        self.assertIn(("e", "é"), alignment)
        self.assertIn(("i", "l"), alignment)

    def test_character_metrics_separate_hallucination_deletion_and_confusable(self) -> None:
        rows = [
            {"reference": "e é i", "prediction": "é e l"},
        ]
        metrics = character_metrics(rows)
        self.assertEqual(metrics["diacritic_hallucinations"], 1)
        self.assertEqual(metrics["diacritic_deletions"], 1)
        self.assertEqual(metrics["confusable_substitutions"], 1)
        self.assertEqual(metrics["diacritic_reference_count"], 1)
        self.assertEqual(metrics["diacritic_prediction_count"], 1)
        self.assertEqual(metrics["diacritic_precision"], 0.0)
        self.assertEqual(metrics["diacritic_recall"], 0.0)

    def test_compare_runs_detects_pass_and_exact_regressions(self) -> None:
        baseline = [
            {"filename": "a.png", "cer": 0.0, "verdict": "PASS", "reference": "é", "prediction": "é"},
            {"filename": "b.png", "cer": 0.2, "verdict": "WARN", "reference": "abc", "prediction": "axc"},
        ]
        candidate = [
            {"filename": "a.png", "cer": 1.0, "verdict": "FAIL", "reference": "é", "prediction": "e"},
            {"filename": "b.png", "cer": 0.0, "verdict": "PASS", "reference": "abc", "prediction": "abc"},
        ]
        report = compare_runs(baseline, candidate)
        self.assertEqual(report["pairwise"]["cer_improved"], 1)
        self.assertEqual(report["pairwise"]["cer_worse"], 1)
        self.assertEqual(report["pairwise"]["baseline_pass_regressions"], ["a.png"])
        self.assertEqual(report["pairwise"]["exact_regressions"], ["a.png"])
        self.assertFalse(report["gates"]["no_pass_regressions"])

    def test_compare_runs_blocks_added_defect_false_passes(self) -> None:
        baseline = [
            {
                "filename": "defect.png",
                "visible_text": "Verifiez",
                "expected": "Vérifiez",
                "reference": "Verifiez",
                "prediction": "Veriflez",
                "cer": 0.125,
                "verdict": "WARN",
            }
        ]
        candidate = [
            {
                **baseline[0],
                "prediction": "Vérifiez",
                "cer": 0.125,
            }
        ]

        report = compare_runs(baseline, candidate)

        self.assertEqual(report["baseline_defects"]["false_pass_count"], 0)
        self.assertEqual(report["candidate_defects"]["false_pass_count"], 1)
        self.assertFalse(report["gates"]["no_added_defect_false_passes"])

    def test_compare_runs_reports_category_metrics(self) -> None:
        row = {
            "filename": "accent.png",
            "category": "missing_accent",
            "reference": "Vérifiez",
            "prediction": "Verifiez",
            "cer": 0.125,
            "verdict": "WARN",
        }

        report = compare_runs([row], [row])
        metrics = report["candidate_categories"]["missing_accent"]

        self.assertEqual(metrics["rows"], 1)
        self.assertEqual(metrics["exact"], 0)
        self.assertEqual(metrics["diacritic_deletions"], 1)

    def test_defect_false_pass_requires_visible_defect_to_be_corrected_to_spec(self) -> None:
        rows = [
            {
                "filename": "hidden.png",
                "visible_text": "Véuillez",
                "expected": "Veuillez",
                "prediction": "Veuillez",
            },
            {
                "filename": "detected.png",
                "visible_text": "MAN.",
                "expected": "MAÑ.",
                "prediction": "MAN.",
            },
        ]
        metrics = defect_false_passes(rows)
        self.assertEqual(metrics["eligible_rows"], 2)
        self.assertEqual(metrics["false_passes"], ["hidden.png"])


if __name__ == "__main__":
    unittest.main()