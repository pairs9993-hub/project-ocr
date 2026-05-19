"""Scan every label across train manifest + test/demo and emit a PaddleOCR
character dictionary covering all 22 languages.

Output:
  data/dict/ppocr_keys.txt           one char per line (PaddleOCR convention)
  artifacts/charset/coverage.json    per-language and per-source counts
"""

import argparse
import json
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


def iter_label_files(project: Path):
    yield project / "dataset/train_manifest.jsonl"
    yield project / "dataset/test/labels.jsonl"
    yield project / "dataset/demo/labels.jsonl"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", default=".")
    args = p.parse_args()

    project = Path(args.project_root).resolve()
    out_dict = project / "data/dict/ppocr_keys.txt"
    out_cov = project / "artifacts/charset/coverage.json"
    out_dict.parent.mkdir(parents=True, exist_ok=True)
    out_cov.parent.mkdir(parents=True, exist_ok=True)

    # PaddleOCR's CTC reserves an implicit blank, but we should *not* include
    # ASCII control chars or unprintable ones. Whitespace handled below.
    char_counts: Counter = Counter()
    by_lang: dict = defaultdict(Counter)
    n_lines_per_source: dict = {}

    for lf in iter_label_files(project):
        if not lf.exists():
            print(f"skip (not found): {lf}", file=sys.stderr)
            continue
        n = 0
        with lf.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                lang = d.get("language", "?")
                # use raw_text + per-element text (raw_text already concatenates
                # but element list is the source of truth for labels)
                texts = []
                for el in d.get("elements", []):
                    t = el.get("text") or ""
                    if t:
                        texts.append(t)
                if not texts and "raw_text" in d:
                    texts = [d["raw_text"]]
                for t in texts:
                    t = unicodedata.normalize("NFC", t)
                    for ch in t:
                        if ch == "\n":
                            continue
                        # skip C0 controls (keep tab as no — we'll exclude tab too)
                        if ord(ch) < 0x20:
                            continue
                        char_counts[ch] += 1
                        by_lang[lang][ch] += 1
                n += 1
        n_lines_per_source[str(lf.relative_to(project))] = n
        print(f"  {lf.relative_to(project)}: {n} samples", flush=True)

    # write dict — sorted by code point
    chars = sorted(char_counts.keys())
    with out_dict.open("w", encoding="utf-8") as f:
        for ch in chars:
            f.write(ch + "\n")

    # coverage report
    coverage = {
        "total_unique_chars": len(chars),
        "samples_per_source": n_lines_per_source,
        "per_language_unique_chars": {lang: len(c) for lang, c in by_lang.items()},
        "top_50_global": char_counts.most_common(50),
    }
    out_cov.write_text(json.dumps(coverage, ensure_ascii=False, indent=2),
                       encoding="utf-8")

    print(f"\ndict: {out_dict}  ({len(chars)} chars)")
    print(f"coverage: {out_cov}")
    # quick CJK / Arabic / Cyrillic / Greek / Thai counts
    blocks = {
        "ASCII": (0x20, 0x7E),
        "Latin-1 Supplement": (0xA0, 0xFF),
        "Latin Extended-A": (0x100, 0x17F),
        "Latin Extended-B": (0x180, 0x24F),
        "Greek": (0x370, 0x3FF),
        "Cyrillic": (0x400, 0x4FF),
        "Arabic": (0x600, 0x6FF),
        "Thai": (0xE00, 0xE7F),
        "CJK": (0x4E00, 0x9FFF),
    }
    print("\nPer-block coverage:")
    for name, (lo, hi) in blocks.items():
        in_block = sum(1 for ch in chars if lo <= ord(ch) <= hi)
        print(f"  {name:25s} {in_block:>5d}")


if __name__ == "__main__":
    main()
