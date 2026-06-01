from __future__ import annotations

import argparse
import hashlib
import json
import sys
import unicodedata
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from ocr_runner import _get_engine, pil_to_bgr


def load_charset(path: Path) -> set[str]:
    return set(path.read_text(encoding="utf-8").splitlines())


def iter_label_rows(labels_path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with labels_path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            entry = json.loads(line)
            image_path = entry.get("image_path") or entry.get("image")
            text = entry.get("text", "")
            if image_path and text:
                rows.append({"image_path": image_path, "text": text})
    return rows


def resolve_image(labels_path: Path, images_dir: Path, image_path: str) -> Path:
    path = Path(image_path)
    candidates = [
        images_dir / path.name,
        images_dir / path,
        labels_path.parent / path,
        ROOT / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def group_result_lines(result, tolerance: float) -> list[list[tuple[list[float], str, float]]]:
    items = []
    for box, text, score in result or []:
        xs = [point[0] for point in box]
        ys = [point[1] for point in box]
        bounds = [min(xs), min(ys), max(xs), max(ys)]
        items.append((sum(ys) / 4.0, sum(xs) / 4.0, bounds, text, float(score)))
    items.sort(key=lambda item: (item[0], item[1]))

    lines: list[list[tuple[list[float], str, float]]] = []
    current: list[tuple[list[float], str, float]] = []
    current_y: float | None = None
    for y, _x, bounds, text, score in items:
        if current_y is None or abs(y - current_y) <= tolerance:
            current.append((bounds, text, score))
            current_y = y if current_y is None else (current_y + y) / 2.0
        else:
            lines.append(current)
            current = [(bounds, text, score)]
            current_y = y
    if current:
        lines.append(current)
    return lines


def union_bounds(line: list[tuple[list[float], str, float]], width: int, height: int, margin: int) -> tuple[int, int, int, int]:
    x1 = min(item[0][0] for item in line)
    y1 = min(item[0][1] for item in line)
    x2 = max(item[0][2] for item in line)
    y2 = max(item[0][3] for item in line)
    return (
        max(0, int(x1) - margin),
        max(0, int(y1) - margin),
        min(width, int(x2) + margin),
        min(height, int(y2) + margin),
    )


def split_name(image_name: str, line_index: int, train_ratio: float) -> str:
    digest = hashlib.sha1(f"{image_name}:{line_index}".encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return "train" if bucket < train_ratio else "val"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, default=ROOT / "artifacts/company_real_screens/llm_labels/gpt_test_1045.jsonl")
    parser.add_argument("--images", type=Path, default=ROOT / "artifacts/company_real_screens/images_for_llm")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/rec_dataset_company_pseudo")
    parser.add_argument("--rec-model", type=Path, default=ROOT / "artifacts/models/real_ui_1m/rec.onnx")
    parser.add_argument("--rec-keys", type=Path, default=ROOT / "artifacts/models/real_ui_1m/ppocr_keys.txt")
    parser.add_argument("--det-model", type=Path, default=ROOT / "artifacts/models/real_ui_1m/det.onnx")
    parser.add_argument("--det-unclip-ratio", type=float, default=3.5)
    parser.add_argument("--line-tolerance", type=float, default=12.0)
    parser.add_argument("--margin", type=int, default=2)
    parser.add_argument("--train-ratio", type=float, default=0.85)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    train_dir = output_dir / "train_crops"
    val_dir = output_dir / "val_crops"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    charset = load_charset(args.rec_keys)
    rows = iter_label_rows(args.labels)
    if args.limit > 0:
        rows = rows[: args.limit]

    engine = _get_engine(str(args.rec_model), str(args.rec_keys), str(args.det_model), args.det_unclip_ratio)
    written = {"train": 0, "val": 0}
    rejected = {"line_count": 0, "oov": 0, "size": 0, "missing": 0}

    with (output_dir / "rec_train.txt").open("w", encoding="utf-8") as train_file, (output_dir / "rec_val.txt").open("w", encoding="utf-8") as val_file, (output_dir / "rejects.jsonl").open("w", encoding="utf-8") as reject_file:
        for index, row in enumerate(rows, 1):
            image_path = resolve_image(args.labels, args.images, row["image_path"])
            if not image_path.exists():
                rejected["missing"] += 1
                continue
            truth_lines = [unicodedata.normalize("NFC", line).strip() for line in row["text"].splitlines() if line.strip()]
            if not truth_lines:
                continue
            if any(any(ch not in charset and ch != " " for ch in line) for line in truth_lines):
                rejected["oov"] += 1
                reject_file.write(json.dumps({"image": image_path.name, "reason": "oov", "truth": truth_lines}, ensure_ascii=False) + "\n")
                continue

            image = Image.open(image_path).convert("RGB")
            result, _ = engine(pil_to_bgr(image))
            line_groups = group_result_lines(result, args.line_tolerance)
            if len(line_groups) != len(truth_lines):
                rejected["line_count"] += 1
                reject_file.write(json.dumps({"image": image_path.name, "reason": "line_count", "truth_lines": len(truth_lines), "det_lines": len(line_groups)}, ensure_ascii=False) + "\n")
                continue

            for line_index, (line_group, truth) in enumerate(zip(line_groups, truth_lines)):
                x1, y1, x2, y2 = union_bounds(line_group, image.width, image.height, args.margin)
                if x2 - x1 < 8 or y2 - y1 < 8:
                    rejected["size"] += 1
                    continue
                split = split_name(image_path.name, line_index, args.train_ratio)
                crop_dir = train_dir if split == "train" else val_dir
                list_file = train_file if split == "train" else val_file
                crop_name = f"{image_path.stem}__line{line_index:02d}.png"
                image.crop((x1, y1, x2, y2)).save(crop_dir / crop_name)
                list_file.write(f"{crop_dir.name}/{crop_name}\t{truth}\n")
                written[split] += 1
            if index % 100 == 0:
                print(f"processed {index}/{len(rows)} train={written['train']} val={written['val']} rejected={rejected}", flush=True)

    args.rec_keys.read_text(encoding="utf-8")
    (output_dir / "ppocr_keys.txt").write_text(args.rec_keys.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"output root: {output_dir}")
    print(f"written: {written}")
    print(f"rejected: {rejected}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())