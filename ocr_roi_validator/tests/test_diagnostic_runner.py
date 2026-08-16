"""Tests for the crash-safe diagnostic runner.

These pin the properties that were missing when a fifteen-hour run was lost to
a shutdown: that finished work survives an interruption, that resuming
reproduces the same random stream, and that a report is never half-written.
"""

import json
import random
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ocr_roi_validator.diagnostic_runner import (
    CheckpointWriter,
    atomic_write_json,
    load_checkpoint,
    resumable_units,
    verify_resume_parity,
)


def workload(path: Path, stop_after: int | None, total: int = 500) -> None:
    """A stand-in diagnostic: RNG-driven units appended to a checkpoint."""
    state = load_checkpoint(path)
    rng = random.Random(4242)
    written = 0
    with CheckpointWriter(path, state.digests, flush_every=7) as writer:
        for index, value in resumable_units(
                total, lambda _i: rng.random(), state.completed_units):
            # Deliberately no resume_count here: a measurement row must not
            # record how many times the run was restarted, or resumed data
            # could never be byte-identical to an uninterrupted run. Run
            # provenance belongs in the manifest instead.
            writer.append({
                "unit": index,
                "row_digest": f"unit-{index}",
                "value": round(value, 12),
            })
            written += 1
            if stop_after is not None and written >= stop_after:
                return


class CheckpointTests(unittest.TestCase):
    def test_rows_survive_an_interruption(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.jsonl"
            workload(path, stop_after=30)
            state = load_checkpoint(path)
            self.assertEqual(len(state.rows), 30)
            self.assertEqual(state.next_unit, 30)

    def test_truncated_final_line_is_dropped_not_repaired(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.jsonl"
            workload(path, stop_after=10)
            with path.open("a", encoding="utf-8") as handle:
                handle.write('{"unit": 10, "row_dig')       # killed mid-write
            state = load_checkpoint(path)
            self.assertEqual(len(state.rows), 10)

    def test_duplicate_digests_are_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.jsonl"
            with CheckpointWriter(path) as writer:
                self.assertTrue(writer.append({"unit": 0, "row_digest": "a"}))
                self.assertFalse(writer.append({"unit": 0, "row_digest": "a"}))
                self.assertEqual(writer.duplicates_rejected, 1)
            self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 1)

    def test_run_local_fields_are_refused(self) -> None:
        """resume_count in a row would make parity unachievable by design."""
        with TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.jsonl"
            with CheckpointWriter(path) as writer:
                for field in ("resume_count", "pid", "started_at"):
                    with self.assertRaises(ValueError):
                        writer.append({"unit": 0, "row_digest": "a", field: 1})

    def test_resume_count_counts_resumes_not_runs(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.jsonl"
            workload(path, stop_after=20)
            # The first run created the file; nothing has been resumed yet.
            self.assertEqual(load_checkpoint(path).resume_count, 0)
            workload(path, stop_after=20)
            self.assertEqual(load_checkpoint(path).resume_count, 1)
            workload(path, stop_after=20)
            self.assertEqual(load_checkpoint(path).resume_count, 2)

    def test_reading_a_checkpoint_has_no_side_effect(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.jsonl"
            workload(path, stop_after=20)
            workload(path, stop_after=20)
            counts = {load_checkpoint(path).resume_count for _ in range(5)}
            self.assertEqual(counts, {1})


class RngReplayTests(unittest.TestCase):
    def test_completed_units_are_still_built(self) -> None:
        """Skipping outright would shift the shared RNG stream."""
        rng = random.Random(1)
        built: list[int] = []

        def build(index: int) -> float:
            built.append(index)
            return rng.random()

        yielded = [i for i, _ in resumable_units(10, build, {0, 1, 2, 3})]
        self.assertEqual(built, list(range(10)))     # all ten were constructed
        self.assertEqual(yielded, [4, 5, 6, 7, 8, 9])

    def test_resumed_values_match_uninterrupted(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            workload(root / "straight.jsonl", None, total=60)
            workload(root / "split.jsonl", 25, total=60)
            workload(root / "split.jsonl", None, total=60)
            straight = [json.loads(l)["value"] for l
                        in (root / "straight.jsonl").read_text().splitlines()]
            split = [json.loads(l)["value"] for l
                     in (root / "split.jsonl").read_text().splitlines()]
            self.assertEqual(straight, split)


class ParityTests(unittest.TestCase):
    def test_five_hundred_unit_resume_parity(self) -> None:
        """The gate the real diagnostic must clear before it may start."""
        with TemporaryDirectory() as directory:
            result = verify_resume_parity(
                lambda path, stop: workload(path, stop, total=500),
                Path(directory), total=500, interrupt_at=213)
            self.assertTrue(result["parity"], result)
            self.assertTrue(result["bytes_identical"])
            self.assertEqual(result["uninterrupted_rows"], 500)
            self.assertEqual(result["resumed_rows"], 500)
            self.assertIsNone(result["first_mismatch_row"])
            self.assertEqual(result["uninterrupted_sha256"],
                             result["resumed_sha256"])

    def test_parity_detects_a_divergent_resume(self) -> None:
        """A runner that reseeds on resume must be caught, not passed."""
        def broken(path: Path, stop_after: int | None) -> None:
            state = load_checkpoint(path)
            # Reseeding per invocation instead of replaying the stream.
            rng = random.Random(len(state.rows))
            written = 0
            with CheckpointWriter(path, state.digests) as writer:
                for index in range(60):
                    if index in state.completed_units:
                        continue
                    writer.append({"unit": index, "row_digest": f"u{index}",
                                   "value": rng.random()})
                    written += 1
                    if stop_after is not None and written >= stop_after:
                        return

        with TemporaryDirectory() as directory:
            result = verify_resume_parity(broken, Path(directory), 60, 20)
            self.assertFalse(result["parity"])


class AtomicWriteTests(unittest.TestCase):
    def test_report_is_written_and_hashed(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            digest = atomic_write_json(path, {"b": 2, "a": 1})
            self.assertTrue(path.is_file())
            self.assertEqual(json.loads(path.read_text())["a"], 1)
            self.assertEqual(len(digest), 64)

    def test_failed_write_leaves_no_partial_file(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"

            class Unserializable:
                pass

            with self.assertRaises(TypeError):
                atomic_write_json(path, {"bad": Unserializable()})
            self.assertFalse(path.exists())
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
