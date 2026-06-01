"""Mine OCR evaluation JSONL for retraining hard cases.

Inputs are rows written by scripts/evaluate_app_ocr_against_labels.py.
The script groups non-PASS rows and near-threshold PASS rows into pattern
categories so they can be used for detector/recognizer hard-example generation.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def classify(row: dict) -> str:
    reference = row.get("reference", "") or ""
    ref_l = reference.lower()

    if any(
        token in ref_l
        for token in [
            "washes clothes",
            "action that",
            "untangle",
            "detergent quickly",
            "minimizes damage",
        ]
    ):
        return "dense_small_explanatory_text"
    if any(
        token in ref_l
        for token in [
            "inicio retardado",
            "delay start",
            "iniciar a las",
            "tmrw",
            "mañ",
            "a.m.",
            "p.m.",
            "am 12",
            "pm 12",
        ]
    ):
        return "schedule_delay_time_text"
    if any(
        token in ref_l
        for token in ["40 min", "15 min", "59 min", "9 min", "1 h 9 min", "1 hr 9 min", "1 h 30 min", "1 hr 30 min"]
    ) and reference.count("\n") <= 4:
        return "timer_duration_confusion"
    if "+2" in reference or "+1" in reference or re.search(r"\n0$", reference):
        return "small_numeric_option_missing"
    if any(token in ref_l for token in ["more cycles", "más ciclos", "<cycle", "large load", "sweat stains", "hand/wool"]):
        return "list_rows_icon_residue"
    if any(token in ref_l for token in ["demo mode", "thinq", "turbowash", "coldwash"]):
        return "brand_badge_or_icon_residue"
    if any(ch in reference for ch in "áéíóúñÑÁÉÍÓÚ"):
        return "spanish_accent_spacing"
    return "misc_recognition"


def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", required=True, help="Evaluation JSONL from evaluate_app_ocr_against_labels.py")
    parser.add_argument("--out-dir", required=True, help="Output directory for mined hard-case manifests")
    parser.add_argument("--near-pass-cer", type=float, default=0.035, help="PASS rows at/above this CER are kept as near-threshold hard cases")
    args = parser.parse_args()

    eval_path = Path(args.eval)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern_dir = out_dir / "patterns"
    pattern_dir.mkdir(exist_ok=True)

    rows = load_rows(eval_path)
    hard_rows: list[dict] = []
    near_pass_rows: list[dict] = []
    by_pattern: dict[str, list[dict]] = defaultdict(list)

    for row in rows:
        verdict = row.get("verdict", "")
        cer = float(row.get("cer", 0.0))
        if verdict != "PASS" or cer >= args.near_pass_cer:
            enriched = dict(row)
            enriched["hard_pattern"] = classify(row)
            if verdict == "PASS":
                near_pass_rows.append(enriched)
            else:
                hard_rows.append(enriched)
            by_pattern[enriched["hard_pattern"]].append(enriched)

    write_jsonl(out_dir / "hard_cases.jsonl", hard_rows)
    write_jsonl(out_dir / "near_threshold_pass.jsonl", near_pass_rows)
    write_jsonl(out_dir / "all_mined_cases.jsonl", hard_rows + near_pass_rows)

    for pattern, pattern_rows in sorted(by_pattern.items()):
        write_jsonl(pattern_dir / f"{pattern}.jsonl", pattern_rows)

    summary_lines = [
        "# OCR Hard Case Mining Summary",
        "",
        f"source: `{eval_path}`",
        f"rows: {len(rows)}",
        f"non-pass hard cases: {len(hard_rows)}",
        f"near-threshold PASS cases: {len(near_pass_rows)} (CER >= {args.near_pass_cer})",
        "",
        "## Patterns",
        "",
        "| Pattern | Rows | Avg CER | Max CER | Worst file |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for pattern, pattern_rows in sorted(by_pattern.items(), key=lambda item: (-len(item[1]), item[0])):
        avg_cer = sum(float(row.get("cer", 0.0)) for row in pattern_rows) / len(pattern_rows)
        worst = max(pattern_rows, key=lambda row: float(row.get("cer", 0.0)))
        summary_lines.append(
            f"| {pattern} | {len(pattern_rows)} | {avg_cer:.4f} | {float(worst.get('cer', 0.0)):.4f} | {worst.get('filename', '')} |"
        )

    summary_lines.extend([
        "",
        "## Outputs",
        "",
        "- `hard_cases.jsonl`: all WARN/FAIL rows",
        "- `near_threshold_pass.jsonl`: PASS rows close to WARN threshold",
        "- `all_mined_cases.jsonl`: union of both sets",
        "- `patterns/*.jsonl`: per-pattern splits for targeted generation/training",
    ])
    (out_dir / "SUMMARY.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"rows={len(rows)} hard={len(hard_rows)} near_pass={len(near_pass_rows)} patterns={len(by_pattern)}")
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())