import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from ocr_roi_validator.gui import ROIItem, _save_failed_roi_diagnostic


class FailureDiagnosticTests(unittest.TestCase):
    def test_saves_only_roi_pixels_and_metadata(self) -> None:
        source = Image.new("RGB", (20, 12), "white")
        roi = ROIItem(
            roi_id=2,
            rect=(3, 2, 13, 8),
            expected="Veuillez",
            actual="Véuillez",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            saved_dir = _save_failed_roi_diagnostic(source, roi, Path(temp_dir))
            with Image.open(saved_dir / "roi.png") as saved_image:
                saved_size = saved_image.size
            metadata = json.loads((saved_dir / "metadata.json").read_text(encoding="utf-8"))

        self.assertEqual(saved_size, (10, 6))
        self.assertEqual(metadata["expected"], "Veuillez")
        self.assertEqual(metadata["actual"], "Véuillez")
        self.assertEqual(metadata["rect"], [3, 2, 13, 8])


if __name__ == "__main__":
    unittest.main()

class OcrInputDiagnosticTests(unittest.TestCase):
    """The diagnostic can also record the exact image the recognizer saw."""

    def test_saves_ocr_input_and_preprocess_metadata(self) -> None:
        from ocr_roi_validator.roi_preprocess import RoiPreprocessConfig, crop_roi

        source = Image.new("RGB", (400, 300), "white")
        roi = ROIItem(roi_id=1, rect=(50, 40, 300, 160), expected="Veuillez", actual="Véuillez")
        config = RoiPreprocessConfig()

        with tempfile.TemporaryDirectory() as temp_dir:
            saved_dir = _save_failed_roi_diagnostic(
                source, roi, Path(temp_dir), preprocess=config
            )
            metadata = json.loads((saved_dir / "metadata.json").read_text(encoding="utf-8"))
            with Image.open(saved_dir / "roi.png") as raw_image:
                raw_size = raw_image.size
            with Image.open(saved_dir / "roi_ocr_input.png") as ocr_image:
                ocr_bytes = ocr_image.convert("RGB").tobytes()
                ocr_size = ocr_image.size

        expected_input = crop_roi(source, roi.rect, config)
        self.assertEqual(ocr_size, expected_input.size)
        self.assertEqual(ocr_bytes, expected_input.tobytes())
        # The OCR input carries the 8px margin the raw ROI crop does not.
        self.assertEqual(raw_size, (250, 120))
        self.assertNotEqual(ocr_size, raw_size)
        self.assertTrue(metadata["roi_ocr_input_is_exact"])
        self.assertEqual(metadata["preprocess"]["margin"], 8)
        self.assertEqual(metadata["roi_raw_size"], [250, 120])

    def test_without_preprocess_keeps_legacy_shape(self) -> None:
        source = Image.new("RGB", (200, 120), "white")
        roi = ROIItem(roi_id=3, rect=(10, 10, 60, 40), expected="a", actual="b")

        with tempfile.TemporaryDirectory() as temp_dir:
            saved_dir = _save_failed_roi_diagnostic(source, roi, Path(temp_dir))
            metadata = json.loads((saved_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertFalse((saved_dir / "roi_ocr_input.png").exists())

        self.assertFalse(metadata["roi_ocr_input_is_exact"])
        self.assertNotIn("preprocess", metadata)
