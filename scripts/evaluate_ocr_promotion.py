from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ocr_validator.promotion_gate import (
    compare_runs,
    defect_false_passes,
    load_jsonl,
    render_markdown,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare OCR runs against safety promotion gates")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--defects", type=Path)
    parser.add_argument("--language", choices=("fr", "es"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--fail-on-gate", action="store_true")
    args = parser.parse_args()

    baseline_rows = load_jsonl(args.baseline)
    candidate_rows = load_jsonl(args.candidate)
    if args.language:
        baseline_rows = [row for row in baseline_rows if row.get("language") == args.language]
        candidate_rows = [row for row in candidate_rows if row.get("language") == args.language]
    report = compare_runs(baseline_rows, candidate_rows)
    if args.defects:
        defect_rows = load_jsonl(args.defects)
        if args.language:
            defect_rows = [row for row in defect_rows if row.get("language") == args.language]
        report["defects"] = defect_false_passes(defect_rows)
        report["gates"]["no_defect_false_passes"] = report["defects"]["false_pass_count"] == 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "promotion_gate.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "PROMOTION_GATE.md").write_text(render_markdown(report), encoding="utf-8")

    passed = all(report["gates"].values())
    print(f"gates={'PASS' if passed else 'FAIL'}")
    print(f"wrote {args.out_dir}")
    return 1 if args.fail_on_gate and not passed else 0


if __name__ == "__main__":
    raise SystemExit(main())