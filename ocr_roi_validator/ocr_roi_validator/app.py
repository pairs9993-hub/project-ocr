from __future__ import annotations

import argparse
from pathlib import Path

from .gui import run_gui
from .model_package import ModelPackageError, load_model_package
from .paddle_package import PaddlePackageError, load_paddle_model_package
from .ocr_engine import OCREngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ROI OCR Validator")
    parser.add_argument(
        "--backend",
        choices=["paddle", "rapid"],
        default="paddle",
        help="OCR runtime backend. Default is paddle.",
    )
    parser.add_argument(
        "--model-package",
        type=Path,
        default=Path("models_package_example"),
        help="Path to OCR model package folder containing manifest.json",
    )
    parser.add_argument(
        "--rapid-default",
        action="store_true",
        help="Use RapidOCR built-in default models (no custom model package required).",
    )
    parser.add_argument(
        "--paddle-package",
        type=Path,
        default=None,
        help="Path to Paddle model package directory (detector/recognizers inference dirs + dict).",
    )
    return parser.parse_args()


def _run_rapid(args: argparse.Namespace) -> int:
    if args.rapid_default:
        engine = OCREngine(use_rapid_default=True, backend="rapid")
        run_gui(engine)
        return 0

    try:
        package = load_model_package(args.model_package)
    except ModelPackageError as exc:
        print(f"[WARN] {exc}")
        print("Falling back to RapidOCR built-in default models.")
        print("Tip: use --rapid-default explicitly or pass a valid --model-package path.")
        engine = OCREngine(use_rapid_default=True, backend="rapid")
        run_gui(engine)
        return 0

    engine = OCREngine(package, backend="rapid")
    run_gui(engine)
    return 0


def main() -> int:
    args = parse_args()

    if args.backend == "paddle":
        try:
            from paddleocr import PaddleOCR  # noqa: F401

            paddle_package = None
            if args.paddle_package is not None:
                paddle_package = load_paddle_model_package(args.paddle_package)

            engine = OCREngine(backend="paddle", paddle_package=paddle_package)
            run_gui(engine)
            return 0
        except (ImportError, RuntimeError, PaddlePackageError) as exc:
            print(f"[WARN] Paddle backend unavailable: {exc}")
            print("Falling back to RapidOCR backend.")
            print("For Paddle on Windows CPU, use Python 3.10 and install paddleocr/paddlepaddle.")
            return _run_rapid(args)

    return _run_rapid(args)
