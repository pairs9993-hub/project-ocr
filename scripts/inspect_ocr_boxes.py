from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from PIL import Image

from ocr_runner import _get_engine, pil_to_bgr


def box_bounds(box: list[list[float]]) -> list[float]:
    xs = [point[0] for point in box]
    ys = [point[1] for point in box]
    return [round(min(xs), 1), round(min(ys), 1), round(max(xs), 1), round(max(ys), 1)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--rec-model", type=Path, default=ROOT / "artifacts/models/real_ui_1m/rec.onnx")
    parser.add_argument("--rec-keys", type=Path, default=ROOT / "artifacts/models/real_ui_1m/ppocr_keys.txt")
    parser.add_argument("--det-model", type=Path, default=ROOT / "artifacts/models/real_ui_1m/det.onnx")
    parser.add_argument("--det-unclip-ratio", type=float, default=3.5)
    args = parser.parse_args()

    engine = _get_engine(str(args.rec_model), str(args.rec_keys), str(args.det_model), args.det_unclip_ratio)
    for image_path in args.images:
        resolved = image_path if image_path.is_absolute() else ROOT / image_path
        image = Image.open(resolved)
        image.load()
        result, _ = engine(pil_to_bgr(image))
        print(f"IMAGE {resolved.as_posix()} size={image.size} boxes={len(result or [])}")
        for item in result or []:
            row = {
                "bounds": box_bounds(item[0]),
                "score": round(float(item[2]), 6),
                "text": item[1],
            }
            print(json.dumps(row, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())