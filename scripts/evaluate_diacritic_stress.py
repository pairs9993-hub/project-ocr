from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image

from ocr_runner import _get_engine, ocr_image
from ocr_validator.preprocess import PreprocessConfig, preprocess
from ocr_validator.promotion_gate import cer, verdict


PREPROCESS_MODES = ("raw", "bicubic_2x", "lanczos_2x", "clahe", "tophat_clahe")


def prepare_image(image: Image.Image, mode: str) -> Image.Image:
    image = image.convert("RGB")
    if mode == "raw":
        return image
    if mode in {"bicubic_2x", "lanczos_2x"}:
        resample = Image.Resampling.BICUBIC if mode == "bicubic_2x" else Image.Resampling.LANCZOS
        return image.resize((image.width * 2, image.height * 2), resample)

    bgr = np.asarray(image)[:, :, ::-1].copy()
    processed = preprocess(
        bgr,
        PreprocessConfig(
            apply_tophat=mode == "tophat_clahe",
            apply_clahe=True,
            apply_invert=False,
            pad_white=0,
        ),
    )
    return Image.fromarray(cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB))


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate an OCR model on the diacritic stress set")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--rec-model", type=Path, required=True)
    parser.add_argument("--rec-keys", type=Path, required=True)
    parser.add_argument("--det-model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--language", choices=("fr", "es"))
    parser.add_argument("--preprocess", choices=PREPROCESS_MODES, default="raw")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--progress-interval", type=int, default=50)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line]
    if args.language:
        rows = [row for row in rows if row.get("language") == args.language]
    if args.start_index:
        rows = rows[args.start_index :]
    if args.limit:
        rows = rows[: args.limit]
    _get_engine.cache_clear()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    evaluated = []
    with args.out.open("a" if args.append else "w", encoding="utf-8") as output:
        for index, row in enumerate(rows, 1):
            image_path = args.manifest.parent / row["image_path"]
            with Image.open(image_path) as image:
                prepared = prepare_image(image, args.preprocess)
                result = ocr_image(prepared, args.rec_model, args.rec_keys, args.det_model)
            cer_value = cer(row["visible_text"], result.text)
            evaluated_row = {
                **row,
                "image_path": image_path.as_posix(),
                "reference": row["visible_text"],
                "prediction": result.text,
                "cer": cer_value,
                "verdict": verdict(cer_value),
                "n_boxes": result.n_boxes,
                "mean_score": result.mean_score,
                "elapsed_ms": result.elapsed_ms,
                "preprocess": args.preprocess,
            }
            evaluated.append(evaluated_row)
            output.write(json.dumps(evaluated_row, ensure_ascii=False) + "\n")
            output.flush()
            if args.progress_interval and index % args.progress_interval == 0:
                print(f"processed {index}/{len(rows)}", flush=True)

    normal = [row for row in evaluated if row["state"] == "normal"]
    defects = [row for row in evaluated if row["state"] == "defect"]
    false_passes = [row for row in defects if row["prediction"] == row["expected"]]
    print(
        f"rows={len(evaluated)} normal={len(normal)} defects={len(defects)} "
        f"normal_exact={sum(row['cer'] == 0 for row in normal)} "
        f"defect_false_pass={len(false_passes)}"
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())