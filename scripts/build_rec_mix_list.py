"""Build PaddleOCR recognition lists that can reference multiple crop roots.

prepare_rec_data.py writes crop paths relative to each dataset directory. This
script rewrites those paths relative to the project root so one PaddleOCR config
can mix several prepared recognition datasets without copying crop images.
"""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path


START_ICON_RE = re.compile(r"\s*(?:▶\s*Ⅱ|▶|Ⅱ)\s*")


def project_relative(project_root: Path, path: Path) -> str:
    path = path.resolve()
    try:
        rel = path.relative_to(project_root)
    except ValueError:
        rel = path
    return rel.as_posix()


def normalize_label(text: str, remove_start_icon: bool) -> str:
    if remove_start_icon:
        text = START_ICON_RE.sub(" ", text)
        text = re.sub(r"\s+", " ", text).strip()
    return text


def read_source(project_root: Path, dataset_root: str, list_path: str, repeat: int,
                remove_start_icon: bool) -> list[str]:
    root = Path(dataset_root)
    if not root.is_absolute():
        root = project_root / root
    labels = Path(list_path)
    if not labels.is_absolute():
        labels = project_root / labels

    rows: list[str] = []
    with labels.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line or "\t" not in line:
                continue
            image_rel, text = line.split("\t", 1)
            text = normalize_label(text, remove_start_icon)
            if not text:
                continue
            image_path = root / image_rel.replace("/", "\\")
            rows.append(f"{project_relative(project_root, image_path)}\t{text}")
    return rows * repeat


def build_split(project_root: Path, sources: list[list[str]], out_path: Path, seed: int | None,
                remove_start_icon: bool) -> int:
    rows: list[str] = []
    for dataset_root, list_path, repeat_text in sources:
        repeat = int(repeat_text)
        if repeat <= 0:
            continue
        rows.extend(read_source(project_root, dataset_root, list_path, repeat, remove_start_icon))
    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(rows)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8", newline="\n")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--char-dict", required=True)
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--remove-start-icon", action="store_true", help="remove ▶/Ⅱ start-key symbols from labels")
    parser.add_argument("--train-source", action="append", nargs=3, metavar=("DATASET_ROOT", "LIST_PATH", "REPEAT"), required=True)
    parser.add_argument("--val-source", action="append", nargs=3, metavar=("DATASET_ROOT", "LIST_PATH", "REPEAT"), required=True)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir

    train_count = build_split(project_root, args.train_source, output_dir / "rec_train.txt", args.seed, args.remove_start_icon)
    val_count = build_split(project_root, args.val_source, output_dir / "rec_val.txt", None, args.remove_start_icon)

    char_dict = Path(args.char_dict)
    if not char_dict.is_absolute():
        char_dict = project_root / char_dict
    (output_dir / "ppocr_keys.txt").write_text(char_dict.read_text(encoding="utf-8"), encoding="utf-8")

    print(f"train rows: {train_count}")
    print(f"val rows  : {val_count}")
    print(f"output    : {output_dir}")


if __name__ == "__main__":
    main()