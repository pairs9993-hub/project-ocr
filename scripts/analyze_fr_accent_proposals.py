"""Re-decompose Gate A results to isolate the accent question.

Gate A collapsed several distinct outcomes into one pass/fail. A candidate can
propose the correct accent and still be blocked because some *other* error in
the same ROI changes the string length. That distinction matters: it decides
whether any existing model produces a usable accent proposal at all.

This reads the stored Gate A JSONL and re-scores it. No inference runs here.
Ground truth is used only to score, never to select anything.
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path


def first_line(text: str) -> str:
    return (text or "").split("\n", 1)[0]


def first_word(text: str) -> str:
    line = first_line(text).strip()
    return line.split(" ", 1)[0] if line else ""


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text or "")


def other_lines(text: str) -> list[str]:
    return (text or "").split("\n")[1:]


def analyze_row(row: dict, target_expected: str) -> dict:
    """Score one perturbation row of one candidate."""
    baseline = row["baseline"]
    specialist = row["specialist"]
    final = row["final"]

    expected_first_word = first_word(target_expected)

    baseline_first = first_word(baseline)
    specialist_first = first_word(specialist)
    final_first = first_word(final)

    # Did the raw specialist propose the correct accent on the target word,
    # regardless of whether the router could accept it?
    baseline_has_accent_error = nfc(baseline_first) != nfc(expected_first_word)
    raw_accent_fixed = (
        baseline_has_accent_error and nfc(specialist_first) == nfc(expected_first_word)
    )
    routed_accent_fixed = (
        baseline_has_accent_error and nfc(final_first) == nfc(expected_first_word)
    )

    # Did the specialist disturb anything beyond the first word?
    other_text_changed = other_lines(specialist) != other_lines(baseline)
    rest_of_first_line_changed = (
        first_line(specialist)[len(specialist_first):]
        != first_line(baseline)[len(baseline_first):]
    )

    return {
        "perturbation": row["perturbation"],
        "baseline_full": baseline,
        "specialist_full": specialist,
        "baseline_first_line": first_line(baseline),
        "specialist_first_line": first_line(specialist),
        "baseline_first_word": baseline_first,
        "specialist_first_word": specialist_first,
        "final_first_word": final_first,
        "raw_accent_fixed": raw_accent_fixed,
        "other_text_changed": other_text_changed or rest_of_first_line_changed,
        "router_applied": row["specialist_applied"],
        "route": row["route"],
        "routed_accent_fixed": routed_accent_fixed,
        "full_roi_exact": nfc(final) == nfc(target_expected),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-a-jsonl", type=Path, required=True)
    parser.add_argument("--target-metadata", type=Path, required=True)
    parser.add_argument("--out-jsonl", type=Path)
    args = parser.parse_args()

    metadata = json.loads(args.target_metadata.read_text(encoding="utf-8"))
    if metadata.get("ocr_input_fidelity") != "exact_recorded_ocr_input":
        print("refusing: target metadata is not an exact recorded input", file=sys.stderr)
        return 1
    target_expected = metadata["expected"]

    candidates = []
    for line in args.gate_a_jsonl.read_text(encoding="utf-8").splitlines():
        if line.strip():
            candidates.append(json.loads(line))

    results = []
    for candidate in candidates:
        if candidate.get("status") != "evaluated":
            results.append(
                {
                    "candidate_path": candidate.get("candidate_path"),
                    "candidate_sha256": candidate.get("candidate_sha256"),
                    "status": candidate.get("status"),
                    "rows": [],
                }
            )
            continue
        rows = [analyze_row(r, target_expected) for r in candidate["target_rows"]]
        unmodified = next((r for r in rows if r["perturbation"] == "none"), None)
        results.append(
            {
                "candidate_path": candidate["candidate_path"],
                "candidate_sha256": candidate["candidate_sha256"],
                "status": "evaluated",
                "rows": rows,
                "raw_accent_fixed_count": sum(1 for r in rows if r["raw_accent_fixed"]),
                "routed_accent_fixed_count": sum(
                    1 for r in rows if r["routed_accent_fixed"]
                ),
                "full_roi_exact_count": sum(1 for r in rows if r["full_roi_exact"]),
                "raw_accent_fixed_unmodified": bool(
                    unmodified and unmodified["raw_accent_fixed"]
                ),
                "defect_false_pass": candidate.get("defect_false_pass"),
            }
        )

    evaluated = [r for r in results if r["status"] == "evaluated"]
    proposers_unmodified = [r for r in evaluated if r["raw_accent_fixed_unmodified"]]
    proposers_any = [r for r in evaluated if r["raw_accent_fixed_count"] > 0]

    print(f"candidates evaluated: {len(evaluated)}")
    print(f"raw accent proposal on the unmodified input : {len(proposers_unmodified)}")
    print(f"raw accent proposal on any perturbation     : {len(proposers_any)}")
    print()
    print(
        f"{'candidate':62s} {'sha':>13s} "
        f"{'rawFix':>7s} {'routed':>7s} {'fullEx':>7s}"
    )
    for result in sorted(
        evaluated, key=lambda r: (-r["raw_accent_fixed_count"], r["candidate_path"])
    ):
        print(
            f"{result['candidate_path'][:62]:62s} "
            f"{result['candidate_sha256'][:12]:>13s} "
            f"{result['raw_accent_fixed_count']:>5d}/7 "
            f"{result['routed_accent_fixed_count']:>5d}/7 "
            f"{result['full_roi_exact_count']:>5d}/7"
        )

    print()
    if proposers_unmodified:
        print("Candidates proposing the correct accent on the UNMODIFIED exact input:")
        for result in proposers_unmodified:
            row = next(r for r in result["rows"] if r["perturbation"] == "none")
            print(f"  {result['candidate_path']}  {result['candidate_sha256'][:12]}")
            print(f"      baseline first word   : {row['baseline_first_word']!r}")
            print(f"      specialist first word : {row['specialist_first_word']!r}")
            print(f"      other text changed    : {row['other_text_changed']}")
            print(f"      route                 : {row['route']}")
    else:
        print(
            "CONCLUSION: no existing model proposes the target accent correction on "
            "the unmodified exact input."
        )

    if args.out_jsonl:
        args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.out_jsonl.open("w", encoding="utf-8") as handle:
            for result in results:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(f"\nwrote {args.out_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
