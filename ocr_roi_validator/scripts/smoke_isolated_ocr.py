"""Run a real blank-image OCR call through the cancellable Windows spawn service."""
from pathlib import Path
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image
from ocr_roi_validator.automation_ocr import FrozenValue
from ocr_roi_validator.gui import OCRValidatorGUI
from ocr_roi_validator.isolated_ocr import IsolatedOCR
from ocr_roi_validator.ocr_engine import OCREngine


def main():
    processor = OCRValidatorGUI.__new__(OCRValidatorGUI)
    processor.engine = OCREngine(use_rapid_default=True)
    settings = {"language_var": "en_es", "roi_margin_var": "0", "min_roi_side_var": "32",
                "auto_upscale_var": False, "fast_long_roi_var": False,
                "context_detect_var": False, "context_margin_var": "0"}
    for key, value in settings.items():
        setattr(processor, key, FrozenValue(value))
    service = IsolatedOCR(processor)
    try:
        started = time.monotonic()
        result = service.run(Image.new("RGB", (320, 240), "white"), (0, 0, 320, 240), "",
                             threading.Event(), started+30)
        print(f"Isolated OCR OK: text={result.text!r}, boxes={result.n_boxes}, elapsed={time.monotonic()-started:.2f}s")
        assert processor._last_ocr_input is not None
    finally:
        service.close()
    assert service.process is None
    print("Worker cleanup OK")


if __name__ == "__main__":
    main()