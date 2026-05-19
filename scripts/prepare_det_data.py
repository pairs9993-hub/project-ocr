"""Convert manifests to PaddleOCR detection label format.

PaddleOCR det reads label lines of the form:
    <image_path>\t<json list of {"transcription": str, "points": [[x,y]*4]}>

Image paths in the output are RELATIVE TO THE PROJECT ROOT, so the matching
PaddleOCR config must set `data_dir: e:/OCR_Project`.

Inputs:
  train  — dataset/train_manifest.jsonl (998k, image_path already project-root-relative)
  val    — dataset/test/labels.jsonl     (1500, image_path "images/..." -> prepend "dataset/test/")

Outputs:
  data/det_dataset_full/det_train.txt
  data/det_dataset_full/det_val.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def bbox_to_points(bbox, canvas_size=None):
    x1, y1, x2, y2 = bbox
    if canvas_size is not None:
        W, H = canvas_size
        x1 = max(0, min(W, x1))
        x2 = max(0, min(W, x2))
        y1 = max(0, min(H, y1))
        y2 = max(0, min(H, y2))
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


# Minimum polygon area (px^2). PaddleOCR's MakeBorderMap can crash on
# tiny polygons because pyclipper offset returns empty list.
MIN_POLY_AREA = 16


def convert_entry(entry: dict, image_path_prefix: str = "") -> tuple[str, list] | None:
    """Return (image_path, polygons) or None if no text boxes."""
    elements = entry.get("elements", [])
    canvas_size = entry.get("canvas_size")
    polygons = []
    for el in elements:
        if el.get("type") != "text":
            continue
        bbox = el.get("bbox")
        text = el.get("text", "")
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = bbox
        # clip to canvas
        if canvas_size is not None:
            W, H = canvas_size
            x1 = max(0, min(W, x1))
            x2 = max(0, min(W, x2))
            y1 = max(0, min(H, y1))
            y2 = max(0, min(H, y2))
        # filter degenerate / tiny boxes
        w = x2 - x1
        h = y2 - y1
        if w <= 0 or h <= 0 or w * h < MIN_POLY_AREA or min(w, h) < 3:
            continue
        polygons.append({
            "transcription": text,
            "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        })
    if not polygons:
        return None
    img_path = entry["image_path"]
    if image_path_prefix:
        img_path = f"{image_path_prefix}/{img_path}"
    return img_path, polygons


def write_split(in_path: Path, out_path: Path, image_path_prefix: str, limit: int | None):
    n_in = 0
    n_out = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with in_path.open(encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            if limit is not None and n_out >= limit:
                break
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            result = convert_entry(entry, image_path_prefix)
            if result is None:
                continue
            img_path, polygons = result
            fout.write(f"{img_path}\t{json.dumps(polygons, ensure_ascii=False)}\n")
            n_out += 1
    print(f"  {in_path.name}: read {n_in}, wrote {n_out} -> {out_path}")
    return n_out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", default=".", help="OCR_Project root")
    p.add_argument("--out-dir", default="data/det_dataset_full")
    p.add_argument("--train-limit", type=int, default=None, help="cap train rows (None=all)")
    p.add_argument("--val-limit", type=int, default=None, help="cap val rows (None=all)")
    args = p.parse_args()

    root = Path(args.project_root).resolve()
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("converting TRAIN (dataset/train_manifest.jsonl) ...")
    n_train = write_split(
        in_path=root / "dataset/train_manifest.jsonl",
        out_path=out_dir / "det_train.txt",
        image_path_prefix="",  # paths already project-root-relative
        limit=args.train_limit,
    )

    print("converting VAL (dataset/test/labels.jsonl) ...")
    n_val = write_split(
        in_path=root / "dataset/test/labels.jsonl",
        out_path=out_dir / "det_val.txt",
        image_path_prefix="dataset/test",
        limit=args.val_limit,
    )

    print(f"\ndone. train={n_train}, val={n_val}, out_dir={out_dir}")


if __name__ == "__main__":
    main()
