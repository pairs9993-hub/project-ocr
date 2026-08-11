"""Phase 1: the failure diagnostic must store the real recognizer input.

Reconstructing the crop after a failure can diverge from what the recognizer
actually saw. These tests pin the guarantee that the image is captured at the
``OCREngine.run()`` call site and saved verbatim, and that every weaker source
is labelled as such.
"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from ocr_roi_validator.gui import (
    FIDELITY_EXACT,
    FIDELITY_RECONSTRUCTED,
    OCRInputRecord,
    ROIItem,
    _save_failed_roi_diagnostic,
)
from ocr_roi_validator.roi_preprocess import RoiPreprocessConfig, crop_roi

VALIDATOR_ROOT = Path(__file__).resolve().parents[1]
REPLAY = VALIDATOR_ROOT / "scripts" / "replay_fr_specialist_roi.py"


def make_image(width: int, height: int, seed: int = 0) -> Image.Image:
    image = Image.new("RGB", (width, height), (10, 20, 30))
    for x in range(0, width, 3):
        for y in range(0, height, 2):
            image.putpixel((x, y), ((x * 7 + seed) % 256, (y * 11) % 256, (x + y) % 256))
    return image


class RecordedInputTests(unittest.TestCase):
    def test_recorded_image_is_saved_verbatim(self) -> None:
        source = make_image(400, 300)
        # Deliberately unrelated to `source`, so a reconstruction from the
        # source image could not possibly produce these pixels.
        recorded = make_image(123, 45, seed=99)
        roi = ROIItem(roi_id=1, rect=(50, 40, 300, 160), expected="Veuillez", actual="Véuillez")
        record = OCRInputRecord(
            image=recorded,
            path_kind="direct",
            language="fr",
            exact=True,
            raw_ocr_text="Véuillez allumer l'eau.",
        )

        with tempfile.TemporaryDirectory() as temp:
            saved_dir = _save_failed_roi_diagnostic(
                source,
                roi,
                Path(temp),
                preprocess=RoiPreprocessConfig(),
                ocr_input=record,
            )
            with Image.open(saved_dir / "roi_ocr_input.png") as handle:
                saved = handle.convert("RGB")
                saved_bytes = saved.tobytes()
                saved_size = saved.size
            metadata = json.loads((saved_dir / "metadata.json").read_text(encoding="utf-8"))

        self.assertEqual(saved_size, recorded.size)
        self.assertEqual(saved_bytes, recorded.tobytes())
        self.assertEqual(metadata["ocr_input_fidelity"], FIDELITY_EXACT)
        self.assertEqual(metadata["language"], "fr")
        self.assertEqual(metadata["ocr_path"], "direct")
        self.assertEqual(metadata["ocr_raw_output"], "Véuillez allumer l'eau.")
        self.assertEqual(metadata["actual"], "Véuillez")

    def test_exact_save_does_not_recompute_from_source(self) -> None:
        """The saved input must not equal a fresh crop of the source image."""
        source = make_image(400, 300)
        config = RoiPreprocessConfig()
        roi = ROIItem(roi_id=1, rect=(50, 40, 300, 160), expected="x", actual="y")
        recorded = make_image(77, 31, seed=7)
        record = OCRInputRecord(
            image=recorded, path_kind="direct", language="fr", exact=True
        )

        with tempfile.TemporaryDirectory() as temp:
            saved_dir = _save_failed_roi_diagnostic(
                source, roi, Path(temp), preprocess=config, ocr_input=record
            )
            with Image.open(saved_dir / "roi_ocr_input.png") as handle:
                saved_bytes = handle.convert("RGB").tobytes()

        reconstructed = crop_roi(source, roi.rect, config)
        self.assertNotEqual(saved_bytes, reconstructed.tobytes())
        self.assertEqual(saved_bytes, recorded.tobytes())

    def test_context_record_is_not_exact(self) -> None:
        source = make_image(200, 150)
        roi = ROIItem(roi_id=2, rect=(10, 10, 90, 60), expected="x", actual="y")
        record = OCRInputRecord(
            image=make_image(80, 50), path_kind="context", language="fr", exact=False
        )

        with tempfile.TemporaryDirectory() as temp:
            saved_dir = _save_failed_roi_diagnostic(
                source, roi, Path(temp), ocr_input=record
            )
            metadata = json.loads((saved_dir / "metadata.json").read_text(encoding="utf-8"))

        self.assertEqual(metadata["ocr_input_fidelity"], FIDELITY_RECONSTRUCTED)
        self.assertEqual(metadata["ocr_path"], "context")

    def test_reconstruction_without_record_is_not_exact(self) -> None:
        source = make_image(200, 150)
        roi = ROIItem(roi_id=3, rect=(10, 10, 90, 60), expected="x", actual="y")

        with tempfile.TemporaryDirectory() as temp:
            saved_dir = _save_failed_roi_diagnostic(
                source, roi, Path(temp), preprocess=RoiPreprocessConfig()
            )
            metadata = json.loads((saved_dir / "metadata.json").read_text(encoding="utf-8"))

        self.assertEqual(metadata["ocr_input_fidelity"], FIDELITY_RECONSTRUCTED)


class RunOnceCaptureTests(unittest.TestCase):
    """The direct Run Once path must record exactly what the engine received."""

    def test_run_engine_records_engine_input_pixels(self) -> None:
        seen: dict = {}

        class FakeResult:
            def __init__(self) -> None:
                self.text = "Véuillez allumer l'eau."
                self.boxes = [object()]
                self.mean_score = 0.99
                self.n_boxes = 1

        class FakeEngine:
            def run(self, image, language, preserve_small_left_noise=False):
                # Capture precisely what the engine was handed.
                seen["image_bytes"] = image.convert("RGB").tobytes()
                seen["size"] = image.size
                seen["language"] = language
                return FakeResult()

        gui = _make_headless_gui(FakeEngine())
        source = make_image(400, 300)
        rect = (50, 40, 300, 160)
        gui.rois[1] = ROIItem(roi_id=1, rect=rect, expected="Veuillez allumer l'eau.")

        result = gui._run_roi_ocr(source, rect, gui.rois[1].expected)

        self.assertEqual(result.text, "Véuillez allumer l'eau.")
        record = gui._last_ocr_input
        self.assertIsNotNone(record)
        self.assertTrue(record.exact)
        self.assertEqual(record.path_kind, "direct")
        self.assertEqual(record.language, "fr")
        self.assertEqual(record.raw_ocr_text, "Véuillez allumer l'eau.")
        # Pixel-for-pixel identity with what the engine received.
        self.assertEqual(record.image.size, seen["size"])
        self.assertEqual(record.image.convert("RGB").tobytes(), seen["image_bytes"])

    def test_recorded_input_survives_later_mutation_of_the_crop(self) -> None:
        """The record is a copy, so mutating the crop afterwards cannot corrupt it."""

        class FakeResult:
            def __init__(self) -> None:
                self.text = "abc"
                self.boxes = [object()]
                self.mean_score = 1.0
                self.n_boxes = 1

        holder: dict = {}

        class FakeEngine:
            def run(self, image, language, preserve_small_left_noise=False):
                holder["image"] = image
                return FakeResult()

        gui = _make_headless_gui(FakeEngine())
        source = make_image(200, 160)
        rect = (10, 10, 150, 100)
        before = gui._run_roi_ocr(source, rect, "abc") and gui._last_ocr_input.image.tobytes()

        # Mutate the very object handed to the engine.
        holder["image"].putpixel((0, 0), (255, 0, 255))
        self.assertEqual(gui._last_ocr_input.image.tobytes(), before)


def _make_headless_gui(engine):
    """Build a GUI instance without Tk, with only what these tests touch."""
    from ocr_roi_validator.gui import OCRValidatorGUI

    gui = OCRValidatorGUI.__new__(OCRValidatorGUI)
    gui.engine = engine
    gui.rois = {}
    gui._last_ocr_input = None
    gui._ocr_inputs = {}

    class Var:
        def __init__(self, value):
            self._value = value

        def get(self):
            return self._value

    gui.language_var = Var("fr")
    gui.context_detect_var = Var(False)
    gui.roi_margin_var = Var("8")
    gui.min_roi_side_var = Var("160")
    gui.fast_long_roi_var = Var(True)
    gui.auto_upscale_var = Var(True)
    return gui


class ReplayFidelityLabelTests(unittest.TestCase):
    """Each replay input mode must report its own fidelity level."""

    def _fidelity_for(self, *args: str) -> str:
        result = subprocess.run(
            [sys.executable, str(REPLAY), "--help"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        return result.stdout

    def test_help_documents_all_three_modes(self) -> None:
        help_text = self._fidelity_for()
        self.assertIn("--ocr-input", help_text)
        self.assertIn("--roi", help_text)
        self.assertIn("--source-image", help_text)

    def test_no_input_errors_out(self) -> None:
        result = subprocess.run(
            [
                sys.executable, str(REPLAY),
                "--package", "x",
                "--baseline-model", "a.onnx",
                "--specialist-model", "b.onnx",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--ocr-input", result.stderr)

    def test_ocr_input_mode_does_not_reapply_preprocessing(self) -> None:
        """Source check: the exact branch must bypass crop/pad/resize."""
        source_text = REPLAY.read_text(encoding="utf-8")
        marker = 'if fidelity == "exact_recorded_ocr_input":'
        self.assertIn(marker, source_text)
        branch = source_text.split(marker, 1)[1].split("elif", 1)[0]
        self.assertNotIn("crop_roi(", branch)
        self.assertNotIn("pad_long_roi(", branch)
        self.assertNotIn("resize_short_side(", branch)


if __name__ == "__main__":
    unittest.main()


class CaptureCheckerTests(unittest.TestCase):
    """The capture checker must accept only genuinely usable captures."""

    CHECKER = VALIDATOR_ROOT / "scripts" / "check_target_capture.py"

    def _write_capture(self, root: Path, name: str, metadata: dict, with_image=True) -> None:
        directory = root / name
        directory.mkdir(parents=True)
        (directory / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False), encoding="utf-8"
        )
        if with_image:
            make_image(40, 20).save(directory / "roi_ocr_input.png")

    def _good_metadata(self) -> dict:
        return {
            "ocr_input_fidelity": "exact_recorded_ocr_input",
            "ocr_path": "direct",
            "language": "fr",
            "expected": "Veuillez allumer l'eau.",
            "actual": "Véuillez allumer l'eau.",
            "ocr_raw_output": "Véuillez allumer l'eau.",
            "roi_ocr_input_size": [320, 160],
            "models": {
                "detector": {
                    "sha256": "21af37f36ce3940ba2fd201c6035571ae5807cf0333f1734d6d5b95c62135b7c"
                },
                "dictionary": {
                    "sha256": "7ff72cdde593c6f80ebd573dddb67b1a103a1607a444c11c4b2b7db57ae1d627"
                },
                "recognizers": {
                    "fr": {
                        "sha256": "d6a439c2b59b46051ea3e07a9d7df69cb76589489b4e487b3d365a773b903b0d"
                    }
                },
            },
        }

    def _run(self, root: Path):
        return subprocess.run(
            [sys.executable, str(self.CHECKER), "--failures-dir", str(root)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )

    def test_good_capture_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._write_capture(root, "failure_1_roi1", self._good_metadata())
            result = self._run(root)
        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
        self.assertIn("USABLE CAPTURE FOUND", result.stdout)

    def test_reconstructed_capture_rejected(self) -> None:
        metadata = self._good_metadata()
        metadata["ocr_input_fidelity"] = "representative_or_reconstructed_not_exact"
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._write_capture(root, "failure_1_roi1", metadata)
            result = self._run(root)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("NO USABLE CAPTURE", result.stdout)

    def test_context_path_rejected(self) -> None:
        metadata = self._good_metadata()
        metadata["ocr_path"] = "context_fallback"
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._write_capture(root, "failure_1_roi1", metadata)
            result = self._run(root)
        self.assertNotEqual(result.returncode, 0)

    def test_empty_actual_rejected(self) -> None:
        metadata = self._good_metadata()
        metadata["actual"] = ""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._write_capture(root, "failure_1_roi1", metadata)
            result = self._run(root)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("actual is empty", result.stdout)

    def test_wrong_model_hash_reported(self) -> None:
        metadata = self._good_metadata()
        metadata["models"]["recognizers"]["fr"]["sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._write_capture(root, "failure_1_roi1", metadata)
            result = self._run(root)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("MODEL HASH MISMATCH", result.stdout)
