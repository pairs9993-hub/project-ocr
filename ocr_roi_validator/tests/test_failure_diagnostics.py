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