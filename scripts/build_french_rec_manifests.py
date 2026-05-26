from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path


def load_character_set(path: Path) -> set[str]:
    return set(path.read_text(encoding="utf-8").splitlines()) | {" ", "\n"}


def text_elements(row: dict) -> list[str]:
    return [
        str(element.get("text", ""))
        for element in row.get("elements", [])
        if element.get("type") == "text"
    ]


def has_start_key(texts: list[str]) -> bool:
    return any("▶" in text or "Ⅱ" in text or "II" in text for text in texts)


def iter_french_rows(chunks_dir: Path, character_set: set[str]):
    for labels_path in sorted(chunks_dir.glob("chunk_*/labels.jsonl")):
        chunk_dir = labels_path.parent
        with labels_path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("language") != "fr":
                    continue

                texts = text_elements(row)
                if not texts:
                    continue

                bad_chars = sorted({char for char in "\n".join(texts) if char not in character_set})
                if bad_chars:
                    yield None, bad_chars
                    continue

                item = dict(row)
                item["image_path"] = (chunk_dir / row["image_path"]).as_posix()
                item["raw_text"] = "\n".join(texts)
                yield item, []


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build French-only recognition manifests from generated real UI chunks.")
    parser.add_argument("--chunks-dir", type=Path, default=Path("generated_1000000_real_ui_en_fr_es/chunks"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/french_rec_v1/manifests"))
    parser.add_argument("--char-dict", type=Path, default=Path("artifacts/models/real_ui_company_pseudo_rec/ppocr_keys.txt"))
    parser.add_argument("--train-count", type=int, default=20000)
    parser.add_argument("--val-count", type=int, default=2000)
    parser.add_argument("--test-count", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260526)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    character_set = load_character_set(args.char_dict)
    rows: list[dict] = []
    patterns: Counter[str] = Counter()
    oov_counts: Counter[str] = Counter()
    start_key_rows = 0

    for row, bad_chars in iter_french_rows(args.chunks_dir, character_set):
        if row is None:
            oov_counts.update(bad_chars)
            continue
        rows.append(row)
        patterns[row.get("pattern", "?")] += 1
        if has_start_key(text_elements(row)):
            start_key_rows += 1

    rng = random.Random(args.seed)
    rng.shuffle(rows)

    train_end = args.train_count
    val_end = train_end + args.val_count
    test_end = val_end + args.test_count
    splits = {
        "train": rows[:train_end],
        "val": rows[train_end:val_end],
        "test": rows[val_end:test_end],
    }

    for name, split_rows in splits.items():
        write_jsonl(args.output_dir / f"fr_{name}.jsonl", split_rows)
        split_start_key_rows = sum(1 for row in split_rows if has_start_key(text_elements(row)))
        print(f"{name}\trows={len(split_rows)}\tstart_key_rows={split_start_key_rows}\tpath={args.output_dir / f'fr_{name}.jsonl'}")

    print(f"available_fr_rows={len(rows)}")
    print(f"available_start_key_rows={start_key_rows}")
    print(f"patterns={dict(patterns)}")
    print(f"oov_counts={dict(oov_counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())