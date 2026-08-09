from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ocr_roi_validator.compare import normalize_ocr_ui_text
from ocr_roi_validator.gui import _pad_long_roi
from ocr_roi_validator.model_package import load_model_package
from ocr_roi_validator.ocr_engine import OCREngine


def parse_candidate(value: str) -> tuple[str, Path]:
    try:
        name, path = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("candidate must be NAME=MODEL_PATH") from exc
    if not name or not path:
        raise argparse.ArgumentTypeError("candidate must be NAME=MODEL_PATH")
    return name, Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay a saved failed ROI across recognizer models")
    parser.add_argument("--failure-dir", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--language", default="fr")
    parser.add_argument("--candidate", action="append", type=parse_candidate, default=[])
    parser.add_argument("--no-pad-long-roi", action="store_true")
    args = parser.parse_args()

    metadata = json.loads((args.failure_dir / "metadata.json").read_text(encoding="utf-8"))
    expected = str(metadata["expected"])
    with Image.open(args.failure_dir / "roi.png") as saved_image:
        image = saved_image.convert("RGB")
    if not args.no_pad_long_roi:
        image = _pad_long_roi(image)

    package = load_model_package(args.package)
    candidates = [("package_default", package.recognizers[args.language]), *args.candidate]
    for name, model_path in candidates:
        model_path = model_path.resolve()
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        recognizers = dict(package.recognizers)
        recognizers[args.language] = model_path
        engine = OCREngine(package=replace(package, recognizers=recognizers), backend="rapid")
        result = engine.run(image, args.language)
        normalized = normalize_ocr_ui_text(result.text, expected)
        print(
            json.dumps(
                {
                    "name": name,
                    "model": model_path.as_posix(),
                    "input_size": list(image.size),
                    "text": normalized,
                    "mean_score": result.mean_score,
                    "boxes": result.n_boxes,
                },
                ensure_ascii=False,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())