"""Tests for the fail-closed behaviour of the offline router evaluator."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EVALUATOR = REPO_ROOT / "scripts" / "evaluate_fr_specialist_router.py"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_fr_specialist_router import (  # noqa: E402
    AlignmentError,
    align_runs,
    build_rows,
)


def make_row(key: str, prediction: str, reference: str = "Veuillez", **overrides) -> dict:
    row = {
        "filename": key,
        "reference": reference,
        "prediction": prediction,
        "language": "fr",
        "category": "over_accent",
        "state": "normal",
        "visible_text": reference,
        "expected": reference,
    }
    row.update(overrides)
    return row


class AlignmentTests(unittest.TestCase):
    def test_empty_baseline_fails(self) -> None:
        with self.assertRaises(AlignmentError) as ctx:
            align_runs([], [make_row("a.png", "Veuillez")])
        self.assertIn("baseline run has no rows", str(ctx.exception))

    def test_empty_specialist_fails(self) -> None:
        with self.assertRaises(AlignmentError) as ctx:
            align_runs([make_row("a.png", "Veuillez")], [])
        self.assertIn("specialist run has no rows", str(ctx.exception))

    def test_duplicate_baseline_key_fails(self) -> None:
        baseline = [make_row("a.png", "Veuillez"), make_row("a.png", "Véuillez")]
        specialist = [make_row("a.png", "Veuillez")]
        with self.assertRaises(AlignmentError) as ctx:
            align_runs(baseline, specialist)
        self.assertIn("duplicate key", str(ctx.exception))
        self.assertEqual(ctx.exception.details["duplicate_keys_baseline"], ["a.png"])

    def test_duplicate_specialist_key_fails(self) -> None:
        baseline = [make_row("a.png", "Veuillez")]
        specialist = [make_row("a.png", "Veuillez"), make_row("a.png", "Véuillez")]
        with self.assertRaises(AlignmentError) as ctx:
            align_runs(baseline, specialist)
        self.assertEqual(ctx.exception.details["duplicate_keys_specialist"], ["a.png"])

    def test_missing_key_fails(self) -> None:
        baseline = [make_row("a.png", "Veuillez"), make_row("b.png", "Veuillez")]
        specialist = [make_row("a.png", "Veuillez")]
        with self.assertRaises(AlignmentError) as ctx:
            align_runs(baseline, specialist)
        self.assertEqual(ctx.exception.details["missing_keys"], ["b.png"])

    def test_extra_key_fails(self) -> None:
        baseline = [make_row("a.png", "Veuillez")]
        specialist = [make_row("a.png", "Veuillez"), make_row("z.png", "Veuillez")]
        with self.assertRaises(AlignmentError) as ctx:
            align_runs(baseline, specialist)
        self.assertEqual(ctx.exception.details["extra_keys"], ["z.png"])

    def test_reference_mismatch_fails(self) -> None:
        baseline = [make_row("a.png", "Veuillez", reference="Veuillez")]
        specialist = [make_row("a.png", "Veuillez", reference="Autre chose")]
        with self.assertRaises(AlignmentError) as ctx:
            align_runs(baseline, specialist)
        self.assertIn("reference", ctx.exception.details["metadata_mismatches"]["a.png"])

    def test_metadata_mismatch_fails(self) -> None:
        baseline = [make_row("a.png", "Veuillez", category="over_accent")]
        specialist = [make_row("a.png", "Veuillez", category="i_l_confusion")]
        with self.assertRaises(AlignmentError) as ctx:
            align_runs(baseline, specialist)
        self.assertIn("category", ctx.exception.details["metadata_mismatches"]["a.png"])

    def test_missing_optional_field_on_one_side_fails(self) -> None:
        baseline_row = make_row("a.png", "Veuillez")
        specialist_row = make_row("a.png", "Veuillez")
        del specialist_row["state"]
        with self.assertRaises(AlignmentError) as ctx:
            align_runs([baseline_row], [specialist_row])
        self.assertIn(
            "state(presence)", ctx.exception.details["metadata_mismatches"]["a.png"]
        )

    def test_valid_pairing_succeeds(self) -> None:
        baseline = [make_row("a.png", "Véuillez"), make_row("b.png", "1.5")]
        specialist = [make_row("a.png", "Veuillez"), make_row("b.png", "1.s5")]
        shared, report = align_runs(baseline, specialist)
        self.assertEqual(shared, ["a.png", "b.png"])
        self.assertEqual(report["scored_rows"], 2)
        self.assertEqual(report["missing_keys"], [])
        self.assertEqual(report["extra_keys"], [])
        self.assertEqual(report["metadata_mismatches"], {})


class RoutedRowTests(unittest.TestCase):
    def test_unrouted_final_equals_baseline_exactly(self) -> None:
        baseline = [make_row("b.png", "1.5", reference="1.5")]
        specialist = [make_row("b.png", "1.s5", reference="1.5")]
        rows, _ = build_rows(baseline, specialist)
        row = rows[0]
        self.assertFalse(row["specialist_applied"])
        self.assertEqual(row["final_prediction"], row["baseline_prediction"])
        self.assertEqual(row["final_raw_cer"], row["baseline_raw_cer"])
        self.assertEqual(row["final_canonical_cer"], row["baseline_canonical_cer"])
        self.assertEqual(row["final_raw_exact"], row["baseline_raw_exact"])
        self.assertEqual(row["final_canonical_exact"], row["baseline_canonical_exact"])


class ExitCodeTests(unittest.TestCase):
    """The evaluator must exit non-zero on failure without extra flags."""

    def _write(self, path: Path, rows: list[dict]) -> None:
        path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
            encoding="utf-8",
        )

    def _run(self, baseline: list[dict], specialist: list[dict], *extra: str):
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            baseline_path = temp_path / "baseline.jsonl"
            specialist_path = temp_path / "specialist.jsonl"
            self._write(baseline_path, baseline)
            self._write(specialist_path, specialist)
            return subprocess.run(
                [
                    sys.executable,
                    str(EVALUATOR),
                    "--baseline", str(baseline_path),
                    "--specialist", str(specialist_path),
                    "--out-jsonl", str(temp_path / "out.jsonl"),
                    "--report-dir", str(temp_path / "report"),
                    *extra,
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )

    def test_alignment_failure_exits_non_zero(self) -> None:
        result = self._run(
            [make_row("a.png", "Veuillez"), make_row("b.png", "Veuillez")],
            [make_row("a.png", "Veuillez")],
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ALIGNMENT FAILED", result.stderr)

    def test_clean_run_exits_zero(self) -> None:
        result = self._run(
            [make_row("a.png", "Veuillez", reference="Veuillez")],
            [make_row("a.png", "Veuillez", reference="Veuillez")],
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("gates=PASS", result.stdout)

    def test_gate_failure_exits_non_zero_without_flags(self) -> None:
        # Specialist adds an accent the baseline did not have: blocked, but the
        # baseline itself is wrong, so we craft a genuine gate failure instead
        # by making the specialist win and regress the reference.
        baseline = [make_row("a.png", "Véuillez", reference="Véuillez")]
        specialist = [make_row("a.png", "Veuillez", reference="Véuillez")]
        result = self._run(baseline, specialist)
        self.assertNotEqual(result.returncode, 0, msg=result.stdout)
        self.assertIn("gates=FAIL", result.stdout)

    def test_report_only_opts_out_of_non_zero_exit(self) -> None:
        baseline = [make_row("a.png", "Véuillez", reference="Véuillez")]
        specialist = [make_row("a.png", "Veuillez", reference="Véuillez")]
        result = self._run(baseline, specialist, "--report-only")
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("gates=FAIL", result.stdout)


if __name__ == "__main__":
    unittest.main()
