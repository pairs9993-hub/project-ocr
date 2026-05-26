from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from PIL import Image

from ocr_runner import _get_engine, ocr_image


START_ICON_RE = re.compile(r"\s*▶\s*(?:Ⅱ|II)\s*")
START_KEY_ARTIFACT_RE = re.compile(
    r"\b(Press|Presione|Appuyez)\s+(?:▶\s*(?:[fIl1|Ⅱ]{1,2}|A\s*(?:Ⅱ|II)?\s*[lI1|]?)|A\s*[lI1|]|™\s*[lI1|]|[fIl1|Ⅱ]{1,2})\s+(?=(?:to|para|pour|or)\b)",
    re.IGNORECASE,
)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = START_KEY_ARTIFACT_RE.sub(r"\1 ▶II ", text)
    text = START_ICON_RE.sub(" ▶II ", text)
    return re.sub(r"\s+", " ", text).strip()


def levenshtein(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, 1):
        current = [i]
        for j, right_char in enumerate(right, 1):
            current.append(
                min(
                    previous[j] + 1,
                    current[-1] + 1,
                    previous[j - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def cer(reference: str, prediction: str) -> float:
    reference = normalize_text(reference)
    prediction = normalize_text(prediction)
    return levenshtein(reference, prediction) / max(1, len(reference))


def verdict(cer_value: float) -> str:
    if cer_value <= 0.05:
        return "PASS"
    if cer_value <= 0.20:
        return "WARN"
    return "FAIL"


def filename_parts(filename: str) -> tuple[str, str]:
    parts = filename.rsplit(".", 1)[0].split("_")
    screen_type = parts[2] if len(parts) > 2 else ""
    language = "_".join(parts[3:]) if len(parts) > 3 else ""
    if screen_type == "title" and len(parts) > 4 and parts[3] == "subtitle":
        return "title_subtitle", "_".join(parts[4:])
    if screen_type == "list" and len(parts) > 4 and parts[3] == "check":
        return "list_check", "_".join(parts[4:])
    return screen_type, language


def labels_to_truth(labels_path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with labels_path.open(encoding="utf-8") as file:
        for line in file:
            entry = json.loads(line)
            image_path = entry.get("image_path") or entry.get("image")
            if not image_path:
                continue
            if "text" in entry and not entry.get("elements"):
                truth_text = entry.get("text", "")
            else:
                elements = []
                for element in entry.get("elements", []):
                    if element.get("type") != "text":
                        continue
                    x1, y1, x2, y2 = element["bbox"]
                    elements.append(((y1 + y2) / 2, (x1 + x2) / 2, element.get("text", "")))
                elements.sort(key=lambda item: (item[0], item[1]))

                lines: list[str] = []
                current: list[tuple[float, str]] = []
                current_y: float | None = None
                for y, x, text in elements:
                    if current_y is None or abs(y - current_y) <= 12.0:
                        current.append((x, text))
                        current_y = y if current_y is None else (current_y + y) / 2
                    else:
                        current.sort()
                        lines.append(" ".join(text for _, text in current))
                        current = [(x, text)]
                        current_y = y
                if current:
                    current.sort()
                    lines.append(" ".join(text for _, text in current))
                truth_text = "\n".join(lines)
            rows.append({
                "image_path": image_path,
                "filename": Path(image_path).name,
                "truth": truth_text,
            })
    return rows


def resolve_image_path(labels_path: Path, images_path: Path, entry_image_path: str) -> Path:
    entry_path = Path(entry_image_path)
    candidates = [
        images_path / entry_path.name,
        images_path / entry_path,
        labels_path.parent / entry_path,
        ROOT / entry_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, default=ROOT / "dataset/test/labels.jsonl")
    parser.add_argument("--images", type=Path, default=ROOT / "dataset/test/images")
    parser.add_argument("--rec-model", type=Path, default=ROOT / "app/models/rec_v0.onnx")
    parser.add_argument("--rec-keys", type=Path, default=ROOT / "app/models/ppocr_keys.txt")
    parser.add_argument("--det-model", type=Path, default=ROOT / "app/models/det_v0.onnx")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/app_eval/ocr_vs_labels.jsonl")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    _get_engine.cache_clear()
    label_rows = labels_to_truth(args.labels)
    label_rows.sort(key=lambda item: item["image_path"])
    if args.limit > 0:
        label_rows = label_rows[: args.limit]
    print(f"running {len(label_rows)} images", flush=True)

    rows = []
    with args.out.open("w", encoding="utf-8") as output:
        for index, label_row in enumerate(label_rows, 1):
            image_path = resolve_image_path(args.labels, args.images, label_row["image_path"])
            image = Image.open(image_path)
            image.load()
            result = ocr_image(image, args.rec_model, args.rec_keys, args.det_model)
            cer_value = cer(label_row["truth"], result.text)
            row = {
                "filename": label_row["filename"],
                "image_path": image_path.as_posix(),
                "cer": cer_value,
                "verdict": verdict(cer_value),
                "n_boxes": result.n_boxes,
                "mean_score": result.mean_score,
                "reference": label_row["truth"],
                "prediction": result.text,
            }
            rows.append(row)
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
            if args.progress_interval > 0 and index % args.progress_interval == 0:
                print(f"processed {index}/{len(label_rows)}", flush=True)

    counts = Counter(row["verdict"] for row in rows)
    values = sorted(row["cer"] for row in rows)
    print(f"rows={len(rows)} PASS={counts['PASS']} WARN={counts['WARN']} FAIL={counts['FAIL']}")
    print(
        "CER "
        f"avg={sum(values) / len(values):.4f} "
        f"p50={values[len(values) // 2]:.4f} "
        f"p90={values[int(len(values) * 0.90)]:.4f} "
        f"p95={values[int(len(values) * 0.95)]:.4f} "
        f"max={max(values):.4f}"
    )

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        screen_type, _ = filename_parts(row["filename"])
        grouped[screen_type].append(row)
    print("by_type")
    for screen_type, group in sorted(
        grouped.items(), key=lambda item: sum(row["cer"] for row in item[1]) / len(item[1]), reverse=True
    ):
        group_counts = Counter(row["verdict"] for row in group)
        avg_cer = sum(row["cer"] for row in group) / len(group)
        print(
            f"{screen_type}: avg={avg_cer:.4f} "
            f"PASS={group_counts['PASS']} WARN={group_counts['WARN']} FAIL={group_counts['FAIL']} n={len(group)}"
        )

    print("top_bad")
    for row in sorted(rows, key=lambda item: item["cer"], reverse=True)[:20]:
        print(
            json.dumps(
                {
                    "filename": row["filename"],
                    "cer": round(row["cer"], 4),
                    "reference": row["reference"],
                    "prediction": row["prediction"],
                },
                ensure_ascii=False,
            )
        )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())