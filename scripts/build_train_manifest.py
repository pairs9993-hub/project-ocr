"""
Build a unified training manifest spanning multiple synthetic data sources.

Reads:
  dataset/train/labels.jsonl          (18,491 from initial 20k batch)
  generated_980000_balanced_vi/chunks/*/labels.jsonl  (8 × 122,500 = 980,000)

Writes:
  dataset/train_manifest.jsonl

Each entry's `image_path` is rewritten to be relative to the project root
(e.g. "dataset/train/images/screen_5000_carousel_en.png" or
       "generated_980000_balanced_vi/chunks/chunk_000000_122499/images/...").

No files are copied or moved. The manifest is a single source of truth that
training scripts can iterate over.
"""

import argparse
import json
from pathlib import Path


def iter_labels(labels_path: Path, image_root_rel: str):
    """Yield label dicts with `image_path` rewritten to a project-relative path."""
    with labels_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            d["image_path"] = f"{image_root_rel}/{d['image_path']}"
            yield d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", default=".", help="OCR_Project root")
    p.add_argument("--output", default="dataset/train_manifest.jsonl")
    args = p.parse_args()

    root = Path(args.project_root).resolve()
    out_path = root / args.output

    sources = []

    seed_train = root / "dataset/train/labels.jsonl"
    if seed_train.exists():
        sources.append((seed_train, "dataset/train"))

    chunks_dir = root / "generated_980000_balanced_vi/chunks"
    if chunks_dir.exists():
        for chunk in sorted(chunks_dir.iterdir()):
            if (chunk / "labels.jsonl").exists():
                rel = f"generated_980000_balanced_vi/chunks/{chunk.name}"
                sources.append((chunk / "labels.jsonl", rel))

    if not sources:
        raise SystemExit("no sources found")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    per_source = []
    seen_paths = set()
    duplicates = 0
    with out_path.open("w", encoding="utf-8") as out_f:
        for labels_path, rel in sources:
            n = 0
            for d in iter_labels(labels_path, rel):
                if d["image_path"] in seen_paths:
                    duplicates += 1
                    continue
                seen_paths.add(d["image_path"])
                out_f.write(json.dumps(d, ensure_ascii=False) + "\n")
                n += 1
                total += 1
            per_source.append((rel, n))

    print(f"manifest: {out_path}")
    print(f"total entries: {total}")
    if duplicates:
        print(f"duplicates skipped: {duplicates}")
    print("per-source counts:")
    for rel, n in per_source:
        print(f"  {n:>9d}  {rel}")


if __name__ == "__main__":
    main()
