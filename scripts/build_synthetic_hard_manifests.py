from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REAL_SCREEN_MARKERS = (
    "french_c_column",
    "company_real",
    "ocr_validation",
    "images_for_llm",
    "llm_labels",
)
EXTRACTED_SCREEN_NAME_RE = re.compile(r"^\d+-\d+\.png$", re.IGNORECASE)


def project_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError as exc:
        raise ValueError(f"Path is outside project root: {resolved}") from exc


def is_real_screen_path(path: str) -> bool:
    normalized = path.replace("\\", "/").lower()
    if any(marker in normalized for marker in REAL_SCREEN_MARKERS):
        return True
    return EXTRACTED_SCREEN_NAME_RE.match(Path(normalized).name) is not None


def split_for_path(path: str, train_ratio: float, seed: int) -> str:
    digest = hashlib.sha1(f"{seed}:{path}".encode("utf-8")).hexdigest()
    bucket = int(digest[:12], 16) / float(0xFFFFFFFFFFFF)
    return "train" if bucket < train_ratio else "val"


def iter_source(source_dir: Path) -> list[dict]:
    labels_path = source_dir / "labels.jsonl"
    if not labels_path.exists():
        raise FileNotFoundError(labels_path)

    rows: list[dict] = []
    with labels_path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            entry = json.loads(line)
            image_path_text = entry.get("image_path") or entry.get("image")
            if not image_path_text:
                raise ValueError(f"Missing image_path in {labels_path}:{line_no}")
            image_path = Path(image_path_text)
            if not image_path.is_absolute():
                image_path = source_dir / image_path
            if not image_path.exists():
                raise FileNotFoundError(image_path)
            rel_image_path = project_relative(image_path)
            if is_real_screen_path(rel_image_path):
                raise ValueError(f"Refusing real-screen path in training manifest: {rel_image_path}")
            entry["image_path"] = rel_image_path
            entry.pop("image", None)
            rows.append(entry)
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize(rows: list[dict]) -> dict:
    return {
        "rows": len(rows),
        "patterns": dict(Counter(row.get("pattern", "") for row in rows)),
        "languages": dict(Counter(row.get("language", "") for row in rows)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", required=True, help="Synthetic labels directory containing labels.jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=20260528)
    args = parser.parse_args()

    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError("--train-ratio must be between 0 and 1")

    all_rows: list[dict] = []
    for source in args.source:
        source_dir = Path(source)
        if not source_dir.is_absolute():
            source_dir = ROOT / source_dir
        all_rows.extend(iter_source(source_dir))

    splits = {"train": [], "val": []}
    for row in all_rows:
        split = split_for_path(row["image_path"], args.train_ratio, args.seed)
        splits[split].append(row)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    write_jsonl(output_dir / "train.jsonl", splits["train"])
    write_jsonl(output_dir / "val.jsonl", splits["val"])

    summary = {
        "sources": args.source,
        "train_ratio": args.train_ratio,
        "seed": args.seed,
        "total": summarize(all_rows),
        "train": summarize(splits["train"]),
        "val": summarize(splits["val"]),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())