from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def rooted_label_rows(label_path: Path, image_base: Path) -> list[str]:
    rows = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            image_path, label = line.split("\t", 1)
        except ValueError as exc:
            raise ValueError(f"Invalid label row at {label_path}:{line_number}") from exc
        absolute_image = (image_base / image_path).resolve()
        rows.append(f"{absolute_image.relative_to(ROOT).as_posix()}\t{label}")
    return rows


def stress_rows(manifest_path: Path, language: str) -> list[str]:
    rows = []
    for line_number, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), 1):
        row = json.loads(line)
        if row.get("language") != language:
            continue
        label = str(row.get("visible_text", ""))
        if not label or "\t" in label or "\n" in label:
            raise ValueError(f"Invalid visible label at {manifest_path}:{line_number}")
        absolute_image = (manifest_path.parent / row["image_path"]).resolve()
        rows.append(f"{absolute_image.relative_to(ROOT).as_posix()}\t{label}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a combined OCR hard-negative dataset")
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--stress-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--language", default="fr")
    args = parser.parse_args()

    base_dir = args.base_dir.resolve()
    manifest_path = args.stress_manifest.resolve()
    output_dir = args.output_dir.resolve()
    train_rows = rooted_label_rows(base_dir / "rec_train.txt", base_dir)
    hard_negative_rows = stress_rows(manifest_path, args.language)
    val_rows = rooted_label_rows(base_dir / "rec_val.txt", base_dir)
    if not hard_negative_rows:
        raise ValueError(f"No {args.language!r} rows found in {manifest_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rec_train.txt").write_text(
        "\n".join(train_rows + hard_negative_rows) + "\n",
        encoding="utf-8",
    )
    (output_dir / "rec_val.txt").write_text("\n".join(val_rows) + "\n", encoding="utf-8")
    shutil.copy2(base_dir / "ppocr_keys.txt", output_dir / "ppocr_keys.txt")
    metadata = {
        "base_dir": base_dir.as_posix(),
        "stress_manifest": manifest_path.as_posix(),
        "language": args.language,
        "base_train_rows": len(train_rows),
        "hard_negative_rows": len(hard_negative_rows),
        "combined_train_rows": len(train_rows) + len(hard_negative_rows),
        "validation_rows": len(val_rows),
        "label_policy": "visible_text",
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())