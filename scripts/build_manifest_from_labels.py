"""Build a project-relative training manifest from labels.jsonl files.

Each source labels.jsonl is expected to contain image_path values relative to
the labels file's parent directory, which matches synth_generator.py and
scripts/generate_real_ui_synth.py output.

Example:
  python scripts/build_manifest_from_labels.py ^
    --source generated_1000000_real_ui_en_fr_es/chunks/*/labels.jsonl ^
    --output dataset/train_manifest_real_ui_1m.jsonl
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def expand_sources(root: Path, sources: list[str]) -> list[Path]:
    labels_paths: list[Path] = []
    for source in sources:
        pattern_path = resolve_path(root, source)
        matches = [Path(match) for match in glob.glob(str(pattern_path))]
        if not matches and pattern_path.exists():
            matches = [pattern_path]
        for match in sorted(matches):
            if match.is_dir():
                candidate = match / "labels.jsonl"
                if candidate.exists():
                    labels_paths.append(candidate)
            elif match.name == "labels.jsonl" and match.exists():
                labels_paths.append(match)
    unique: list[Path] = []
    seen = set()
    for path in labels_paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def project_relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root).as_posix()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--source", action="append", required=True,
                        help="labels.jsonl, directory containing labels.jsonl, or glob pattern")
    parser.add_argument("--output", default="dataset/train_manifest.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    output = resolve_path(root, args.output)
    labels_paths = expand_sources(root, args.source)
    if not labels_paths:
        raise SystemExit("no labels.jsonl sources found")

    output.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    duplicates = 0
    per_source: list[tuple[str, int]] = []
    seen_images = set()
    with output.open("w", encoding="utf-8", newline="\n") as out_file:
        for labels_path in labels_paths:
            source_count = 0
            with labels_path.open(encoding="utf-8") as in_file:
                for line in in_file:
                    if args.limit is not None and total >= args.limit:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    image_path = labels_path.parent / row["image_path"]
                    row["image_path"] = project_relative(root, image_path)
                    if row["image_path"] in seen_images:
                        duplicates += 1
                        continue
                    seen_images.add(row["image_path"])
                    out_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                    source_count += 1
                    total += 1
            per_source.append((project_relative(root, labels_path), source_count))
            if args.limit is not None and total >= args.limit:
                break

    print(f"manifest: {output}")
    print(f"total entries: {total}")
    if duplicates:
        print(f"duplicates skipped: {duplicates}")
    print("per-source counts:")
    for source, count in per_source:
        print(f"  {count:>9d}  {source}")


if __name__ == "__main__":
    main()