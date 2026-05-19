"""Crop text regions from screen images and emit PaddleOCR recognition format.

PaddleOCR rec training expects:
  rec_train.txt  : "<image_path>\\t<text>" per line
  rec_val.txt    : same
  <crops_dir>/   : the image files referenced by the text files

Usage:
  python scripts/prepare_rec_data.py \
      --train-source dataset/train \
      --val-source   dataset/test \
      --output-dir   data/rec_dataset \
      --margin 4 \
      --val-cap 1500

Notes:
  - Skips degenerate boxes (h < 8 or w < 8 px)
  - Skips empty text after stripping
  - Skips elements whose `text` would not be in the char dict (logged)
  - Written paths in rec_train.txt are relative to --output-dir, so the
    Colab notebook can simply unzip and use them as-is.
"""

import argparse
import json
import sys
import unicodedata
from pathlib import Path

import cv2


def load_dict(path: Path) -> set:
    return set(path.read_text(encoding="utf-8").splitlines())


def iter_labels(split_dir: Path):
    """Yield (full_image_path, label_dict)."""
    labels_path = split_dir / "labels.jsonl"
    with labels_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            img_path = split_dir / d["image_path"]
            yield img_path, d


def iter_manifest(project: Path, manifest_path: Path):
    with manifest_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            img_path = project / d["image_path"]
            yield img_path, d


def crop_text_elements(d: dict, bgr, margin: int):
    """Yield (text, crop_array) pairs for each text element with a bbox."""
    H, W = bgr.shape[:2]
    for idx, el in enumerate(d.get("elements", [])):
        if el.get("type") != "text":
            continue
        text = el.get("text") or ""
        text = unicodedata.normalize("NFC", text).strip()
        if not text:
            continue
        bbox = el.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = bbox
        x1 = max(0, int(x1) - margin)
        y1 = max(0, int(y1) - margin)
        x2 = min(W, int(x2) + margin)
        y2 = min(H, int(y2) + margin)
        if x2 - x1 < 8 or y2 - y1 < 8:
            continue
        crop = bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        yield idx, text, crop


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-source", help="dir with labels.jsonl OR a manifest.jsonl path")
    p.add_argument("--val-source", help="dir with labels.jsonl")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--char-dict", default="data/dict/ppocr_keys.txt")
    p.add_argument("--margin", type=int, default=4)
    p.add_argument("--train-cap", type=int, default=None,
                   help="cap number of input images for train (debug/prototype)")
    p.add_argument("--val-cap", type=int, default=None)
    p.add_argument("--project-root", default=".")
    args = p.parse_args()

    project = Path(args.project_root).resolve()
    out_dir = Path(args.output_dir).resolve()
    if not out_dir.is_absolute():
        out_dir = project / out_dir
    train_crops = out_dir / "train_crops"
    val_crops = out_dir / "val_crops"
    train_crops.mkdir(parents=True, exist_ok=True)
    val_crops.mkdir(parents=True, exist_ok=True)
    char_set = load_dict(project / args.char_dict)

    def process_source(source_arg: str, crops_dir: Path, list_path: Path,
                        cap: int | None, label: str):
        source = Path(source_arg)
        if not source.is_absolute():
            source = project / source
        if source.is_dir():
            iterator = iter_labels(source)
        else:
            iterator = iter_manifest(project, source)

        n_in = 0
        n_crops = 0
        n_skipped_oov = 0
        n_skipped_size = 0
        with list_path.open("w", encoding="utf-8") as out_f:
            for img_path, d in iterator:
                if cap is not None and n_in >= cap:
                    break
                bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                if bgr is None:
                    continue
                stem = img_path.stem
                produced_any = False
                for idx, text, crop in crop_text_elements(d, bgr, args.margin):
                    # OOV check
                    if any(ch not in char_set and ch != " " for ch in text):
                        n_skipped_oov += 1
                        continue
                    crop_name = f"{stem}__{idx:02d}.png"
                    crop_path = crops_dir / crop_name
                    ok = cv2.imwrite(str(crop_path), crop)
                    if not ok:
                        n_skipped_size += 1
                        continue
                    rel = f"{crops_dir.name}/{crop_name}"
                    out_f.write(f"{rel}\t{text}\n")
                    n_crops += 1
                    produced_any = True
                if produced_any:
                    n_in += 1
                if n_in % 2000 == 0 and n_in > 0:
                    print(f"  [{label}] processed {n_in} images, {n_crops} crops",
                          flush=True)

        print(f"\n[{label}] images consumed: {n_in}")
        print(f"[{label}] crops produced : {n_crops}")
        print(f"[{label}] skipped (OOV)  : {n_skipped_oov}")
        print(f"[{label}] skipped (write): {n_skipped_size}")

    if args.train_source:
        process_source(args.train_source, train_crops, out_dir / "rec_train.txt",
                       args.train_cap, "train")
    if args.val_source:
        process_source(args.val_source, val_crops, out_dir / "rec_val.txt",
                       args.val_cap, "val")

    # also drop a copy of the char dict alongside for portability
    (out_dir / "ppocr_keys.txt").write_text(
        (project / args.char_dict).read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    print(f"\noutput root: {out_dir}")


if __name__ == "__main__":
    main()
