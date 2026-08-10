"""Offline evaluation of the French specialist router over stored OCR runs.

Replays a stored baseline run and a stored specialist run through
``route_specialist_text``, writes a per-row JSONL of the routed outcome, and
reports baseline-vs-routed metrics.

This never invokes an OCR model: it operates purely on predictions already
recorded in the two input JSONL files, so it answers "what would the router
have produced?" without needing the candidate ONNX to be identified.

Metric consistency
------------------
Every metric is recomputed here for BOTH sides from the prediction strings.
The ``cer``/``verdict``/exact fields stored in the input JSONL are never reused
as comparison metrics: different runs were written by different tooling and do
not share a normalization convention, so mixing them silently manufactures
improvements. Two exact-match notions are reported separately:

* ``raw_exact``       -- codepoint-identical to the reference.
* ``canonical_exact`` -- identical after the explicitly allowed normalization
                         (NFC, typographic apostrophe folding, whitespace
                         collapsing) that :mod:`ocr_validator.promotion_gate`
                         defines.

When the router applies the specialist nowhere, the routed run must be
byte-identical to the baseline run. That invariant is asserted rather than
assumed, and a violation fails the gate.

Fail-closed
-----------
The evaluator refuses to score data it cannot pair exactly. Empty input,
duplicate row keys, keys present on only one side, and rows whose identity
fields (reference, filename, language, category, state, visible_text,
expected) disagree between the two runs are all fatal rather than warnings --
scoring an intersection would silently hide missing data. A failing gate exits
non-zero by default; ``--report-only`` is the explicit opt-out.

Expected text is carried into the output rows for reporting and for defect
false-pass accounting, but it is never passed to the router.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

VALIDATOR_ROOT = ROOT / "ocr_roi_validator"
if str(VALIDATOR_ROOT) not in sys.path:
    sys.path.insert(0, str(VALIDATOR_ROOT))

from ocr_roi_validator.fr_specialist_router import route_specialist_text  # noqa: E402
from ocr_validator.promotion_gate import (  # noqa: E402
    character_metrics,
    load_jsonl,
    normalize_text,
    row_key,
    verdict,
)
from ocr_validator.promotion_gate import cer as canonical_cer  # noqa: E402
from ocr_validator.promotion_gate import levenshtein  # noqa: E402

VERDICT_RANK = {"FAIL": 0, "WARN": 1, "PASS": 2}


def raw_cer(reference: str, prediction: str) -> float:
    """CER with no normalization at all -- pure codepoint edit distance."""
    reference = reference or ""
    prediction = prediction or ""
    return levenshtein(reference, prediction) / max(1, len(reference))


def score_row(reference: str, prediction: str) -> dict:
    """Compute every metric for one prediction, both raw and canonical."""
    raw_value = raw_cer(reference, prediction)
    canonical_value = canonical_cer(reference, prediction)
    return {
        "raw_cer": raw_value,
        "canonical_cer": canonical_value,
        "raw_exact": prediction == reference,
        "canonical_exact": normalize_text(prediction) == normalize_text(reference),
        "raw_verdict": verdict(raw_value),
        "canonical_verdict": verdict(canonical_value),
    }


def summarize(rows: list[dict], prediction_key: str) -> dict:
    """Summarize a run. Metrics are recomputed from ``prediction_key``."""
    raw_cers: list[float] = []
    canonical_cers: list[float] = []
    raw_exact = 0
    canonical_exact = 0
    raw_verdicts: Counter[str] = Counter()
    canonical_verdicts: Counter[str] = Counter()
    character_rows = []

    for row in rows:
        reference = str(row.get("reference", ""))
        prediction = str(row.get(prediction_key, ""))
        scored = score_row(reference, prediction)
        raw_cers.append(scored["raw_cer"])
        canonical_cers.append(scored["canonical_cer"])
        raw_exact += scored["raw_exact"]
        canonical_exact += scored["canonical_exact"]
        raw_verdicts[scored["raw_verdict"]] += 1
        canonical_verdicts[scored["canonical_verdict"]] += 1
        character_rows.append({"reference": reference, "prediction": prediction})

    count = len(rows)
    return {
        "rows": count,
        "raw_mean_cer": sum(raw_cers) / count if count else 0.0,
        "canonical_mean_cer": sum(canonical_cers) / count if count else 0.0,
        "raw_exact": raw_exact,
        "canonical_exact": canonical_exact,
        "raw_pass": raw_verdicts["PASS"],
        "raw_warn": raw_verdicts["WARN"],
        "raw_fail": raw_verdicts["FAIL"],
        "canonical_pass": canonical_verdicts["PASS"],
        "canonical_warn": canonical_verdicts["WARN"],
        "canonical_fail": canonical_verdicts["FAIL"],
        "characters": character_metrics(character_rows),
    }


def defect_false_passes(rows: list[dict], prediction_key: str) -> dict:
    """Rows whose visible text is a real defect but whose OCR matches expected."""
    eligible: list[str] = []
    false_passes: list[str] = []
    for row in rows:
        visible = normalize_text(str(row.get("visible_text", row.get("reference", ""))))
        expected = normalize_text(str(row.get("expected", "")))
        prediction = normalize_text(str(row.get(prediction_key, "")))
        if not expected or visible == expected:
            continue
        key = row.get("row_key") or row_key(row)
        eligible.append(key)
        if prediction == expected:
            false_passes.append(key)
    return {
        "eligible_rows": len(eligible),
        "false_pass_count": len(false_passes),
        "false_pass_rate": len(false_passes) / len(eligible) if eligible else 0.0,
        "false_passes": false_passes,
    }


def category_false_passes(rows: list[dict], prediction_key: str) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("category", "uncategorized")), []).append(row)
    return {
        category: defect_false_passes(category_rows, prediction_key)
        for category, category_rows in sorted(grouped.items())
    }


class AlignmentError(Exception):
    """Raised when the two runs cannot be paired row for row."""

    def __init__(self, message: str, details: dict) -> None:
        super().__init__(message)
        self.details = details


# Fields that identify a row. If two runs disagree on any of these for the same
# key, they did not evaluate the same thing and must not be compared.
IDENTITY_FIELDS = (
    "reference",
    "filename",
    "language",
    "category",
    "state",
    "visible_text",
    "expected",
)


def index_rows(rows: list[dict], side: str) -> tuple[dict[str, dict], list[str]]:
    """Index rows by key, collecting duplicates rather than silently overwriting."""
    indexed: dict[str, dict] = {}
    duplicates: list[str] = []
    for row in rows:
        key = row_key(row)
        if key in indexed:
            duplicates.append(key)
            continue
        indexed[key] = row
    if duplicates:
        print(
            f"error: {side} contains duplicate row keys: {sorted(set(duplicates))[:10]}",
            file=sys.stderr,
        )
    return indexed, sorted(set(duplicates))


def compare_identity(baseline_row: dict, specialist_row: dict) -> list[str]:
    """Return the identity fields on which two rows for the same key disagree.

    A field present on one side and absent on the other counts as a mismatch:
    that is a provenance difference, not a harmless omission.
    """
    mismatched: list[str] = []
    for field in IDENTITY_FIELDS:
        present_baseline = field in baseline_row
        present_specialist = field in specialist_row
        if present_baseline != present_specialist:
            mismatched.append(f"{field}(presence)")
        elif present_baseline and baseline_row[field] != specialist_row[field]:
            mismatched.append(field)
    return mismatched


def align_runs(baseline_rows: list[dict], specialist_rows: list[dict]) -> tuple[list[str], dict]:
    """Validate that the two runs pair exactly, or raise.

    Returns the sorted shared keys and an alignment report. Any missing key,
    extra key, duplicate key, empty input or identity mismatch is fatal: the
    evaluator must never quietly score a subset of the data.
    """
    baseline_by_key, baseline_duplicates = index_rows(baseline_rows, "baseline")
    specialist_by_key, specialist_duplicates = index_rows(specialist_rows, "specialist")

    missing = sorted(set(baseline_by_key) - set(specialist_by_key))
    extra = sorted(set(specialist_by_key) - set(baseline_by_key))
    shared = sorted(set(baseline_by_key) & set(specialist_by_key))

    mismatches: dict[str, list[str]] = {}
    for key in shared:
        fields = compare_identity(baseline_by_key[key], specialist_by_key[key])
        if fields:
            mismatches[key] = fields

    report = {
        "baseline_input_rows": len(baseline_rows),
        "specialist_input_rows": len(specialist_rows),
        "baseline_indexed_rows": len(baseline_by_key),
        "specialist_indexed_rows": len(specialist_by_key),
        "duplicate_keys_baseline": baseline_duplicates,
        "duplicate_keys_specialist": specialist_duplicates,
        "missing_keys": missing,
        "extra_keys": extra,
        "metadata_mismatches": mismatches,
        "scored_rows": len(shared),
    }

    problems = []
    if not baseline_rows:
        problems.append("baseline run has no rows after filtering")
    if not specialist_rows:
        problems.append("specialist run has no rows after filtering")
    if baseline_duplicates:
        problems.append(f"{len(baseline_duplicates)} duplicate key(s) in baseline")
    if specialist_duplicates:
        problems.append(f"{len(specialist_duplicates)} duplicate key(s) in specialist")
    if missing:
        problems.append(f"{len(missing)} key(s) missing from specialist")
    if extra:
        problems.append(f"{len(extra)} key(s) only in specialist")
    if mismatches:
        problems.append(f"{len(mismatches)} row(s) with mismatched identity fields")
    if not shared:
        problems.append("no shared rows to score")

    if problems:
        raise AlignmentError("; ".join(problems), report)
    return shared, report


def build_rows(
    baseline_rows: list[dict], specialist_rows: list[dict]
) -> tuple[list[dict], dict]:
    """Route each paired row and score baseline / specialist / final identically."""
    baseline_by_key = {row_key(row): row for row in baseline_rows}
    specialist_by_key = {row_key(row): row for row in specialist_rows}
    shared, alignment = align_runs(baseline_rows, specialist_rows)

    rows: list[dict] = []
    for key in shared:
        baseline_row = baseline_by_key[key]
        specialist_row = specialist_by_key[key]
        reference = str(baseline_row.get("reference", ""))
        baseline_prediction = str(baseline_row.get("prediction", ""))
        specialist_prediction = str(specialist_row.get("prediction", ""))

        # The router sees only the two OCR strings.
        decision = route_specialist_text(baseline_prediction, specialist_prediction)

        baseline_scores = score_row(reference, baseline_prediction)
        specialist_scores = score_row(reference, specialist_prediction)
        final_scores = score_row(reference, decision.final_text)

        rows.append(
            {
                "row_key": key,
                "filename": baseline_row.get("filename"),
                "language": baseline_row.get("language"),
                "category": baseline_row.get("category"),
                "state": baseline_row.get("state"),
                "reference": reference,
                "visible_text": baseline_row.get("visible_text"),
                "expected": baseline_row.get("expected"),
                "baseline_prediction": baseline_prediction,
                "specialist_prediction": specialist_prediction,
                "final_prediction": decision.final_text,
                "route": decision.route,
                "specialist_applied": decision.specialist_applied,
                **{f"baseline_{k}": v for k, v in baseline_scores.items()},
                **{f"specialist_{k}": v for k, v in specialist_scores.items()},
                **{f"final_{k}": v for k, v in final_scores.items()},
            }
        )
    return rows, alignment


def check_identity_invariant(rows: list[dict]) -> dict:
    """When nothing was routed, the final run must equal the baseline run.

    Checked per row rather than only in aggregate, so a pair of offsetting
    differences cannot hide.
    """
    applied = [row["row_key"] for row in rows if row["specialist_applied"]]
    text_mismatches = [
        row["row_key"] for row in rows if row["final_prediction"] != row["baseline_prediction"]
    ]
    metric_mismatches = [
        row["row_key"]
        for row in rows
        if (
            row["final_raw_cer"] != row["baseline_raw_cer"]
            or row["final_canonical_cer"] != row["baseline_canonical_cer"]
            or row["final_raw_exact"] != row["baseline_raw_exact"]
            or row["final_canonical_exact"] != row["baseline_canonical_exact"]
        )
        and not row["specialist_applied"]
    ]
    # Any row the router did not route must be byte-identical to the baseline.
    unrouted_text_mismatches = [
        row["row_key"]
        for row in rows
        if not row["specialist_applied"] and row["final_prediction"] != row["baseline_prediction"]
    ]

    result = {
        "specialist_applied_count": len(applied),
        "unrouted_text_mismatches": unrouted_text_mismatches,
        "unrouted_metric_mismatches": metric_mismatches,
        "holds": not unrouted_text_mismatches and not metric_mismatches,
    }
    if not applied:
        # Whole-dataset identity: nothing routed anywhere.
        result["dataset_identity_expected"] = True
        result["dataset_identity_holds"] = not text_mismatches
        result["dataset_text_mismatches"] = text_mismatches
        result["holds"] = result["holds"] and not text_mismatches
    else:
        result["dataset_identity_expected"] = False
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay stored OCR runs through the French specialist router"
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--specialist", type=Path, required=True)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--label", default="routed")
    parser.add_argument("--language", choices=("fr", "es"))
    parser.add_argument(
        "--report-only",
        action="store_true",
        help=(
            "write the report and exit 0 even if gates fail. Without this flag a "
            "failing gate or a broken alignment exits non-zero."
        ),
    )
    args = parser.parse_args()

    baseline_rows = load_jsonl(args.baseline)
    specialist_rows = load_jsonl(args.specialist)
    if args.language:
        baseline_rows = [r for r in baseline_rows if r.get("language") == args.language]
        specialist_rows = [r for r in specialist_rows if r.get("language") == args.language]

    try:
        rows, alignment = build_rows(baseline_rows, specialist_rows)
    except AlignmentError as exc:
        print(f"ALIGNMENT FAILED: {exc}", file=sys.stderr)
        args.report_dir.mkdir(parents=True, exist_ok=True)
        (args.report_dir / "promotion_gate.json").write_text(
            json.dumps(
                {
                    "inputs": {
                        "baseline": str(args.baseline),
                        "specialist": str(args.specialist),
                        "label": args.label,
                        "language": args.language,
                    },
                    "alignment": exc.details,
                    "alignment_error": str(exc),
                    "gates": {"alignment": False},
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        for field in ("duplicate_keys_baseline", "duplicate_keys_specialist",
                      "missing_keys", "extra_keys"):
            values = exc.details.get(field) or []
            if values:
                print(f"  {field}: {len(values)} e.g. {values[:5]}", file=sys.stderr)
        mismatches = exc.details.get("metadata_mismatches") or {}
        for key, fields in list(mismatches.items())[:5]:
            print(f"  metadata mismatch {key}: {fields}", file=sys.stderr)
        print("gates=FAIL")
        return 1

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Both sides scored by the same code path, from prediction strings only.
    baseline_summary = summarize(rows, "baseline_prediction")
    specialist_summary = summarize(rows, "specialist_prediction")
    final_summary = summarize(rows, "final_prediction")

    baseline_defects = defect_false_passes(rows, "baseline_prediction")
    final_defects = defect_false_passes(rows, "final_prediction")

    raw_exact_regressions = [
        r["row_key"] for r in rows if r["baseline_raw_exact"] and not r["final_raw_exact"]
    ]
    canonical_exact_regressions = [
        r["row_key"]
        for r in rows
        if r["baseline_canonical_exact"] and not r["final_canonical_exact"]
    ]
    pass_regressions = [
        r["row_key"]
        for r in rows
        if VERDICT_RANK.get(r["final_canonical_verdict"], -1)
        < VERDICT_RANK.get(r["baseline_canonical_verdict"], -1)
    ]

    invariant = check_identity_invariant(rows)
    routes = dict(sorted(Counter(r["route"] for r in rows).items()))

    gates = {
        "metric_identity_invariant": invariant["holds"],
        "raw_mean_cer_not_worse": (
            final_summary["raw_mean_cer"] <= baseline_summary["raw_mean_cer"]
        ),
        "canonical_mean_cer_not_worse": (
            final_summary["canonical_mean_cer"] <= baseline_summary["canonical_mean_cer"]
        ),
        "no_raw_exact_regressions": not raw_exact_regressions,
        "no_canonical_exact_regressions": not canonical_exact_regressions,
        "no_pass_regressions": not pass_regressions,
        "no_added_diacritic_hallucinations": (
            final_summary["characters"]["diacritic_hallucinations"]
            <= baseline_summary["characters"]["diacritic_hallucinations"]
        ),
        "no_defect_false_passes": final_defects["false_pass_count"] == 0,
        "no_added_defect_false_passes": (
            final_defects["false_pass_count"] <= baseline_defects["false_pass_count"]
        ),
    }

    report = {
        "inputs": {
            "baseline": str(args.baseline),
            "specialist": str(args.specialist),
            "label": args.label,
            "language": args.language,
        },
        "metric_note": (
            "All metrics recomputed from prediction strings for baseline, specialist "
            "and final by the same code path. Stored cer/verdict fields in the input "
            "JSONL are not reused."
        ),
        "alignment": alignment,
        "baseline": baseline_summary,
        "specialist": specialist_summary,
        "final": final_summary,
        "routes": routes,
        "identity_invariant": invariant,
        "baseline_defects": baseline_defects,
        "final_defects": final_defects,
        "baseline_category_defects": category_false_passes(rows, "baseline_prediction"),
        "final_category_defects": category_false_passes(rows, "final_prediction"),
        "regressions": {
            "raw_exact": raw_exact_regressions,
            "canonical_exact": canonical_exact_regressions,
            "pass": pass_regressions,
        },
        "gates": gates,
    }

    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "promotion_gate.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.report_dir / "PROMOTION_GATE.md").write_text(render_markdown(report), encoding="utf-8")

    print(
        f"alignment: baseline_in={alignment['baseline_input_rows']} "
        f"specialist_in={alignment['specialist_input_rows']} "
        f"scored={alignment['scored_rows']} "
        f"duplicates={len(alignment['duplicate_keys_baseline'])}"
        f"/{len(alignment['duplicate_keys_specialist'])} "
        f"missing={len(alignment['missing_keys'])} extra={len(alignment['extra_keys'])} "
        f"metadata_mismatches={len(alignment['metadata_mismatches'])}"
    )
    print(f"rows={final_summary['rows']} specialist_applied={invariant['specialist_applied_count']}")
    for route, count in routes.items():
        print(f"  {route}: {count}")
    print("           raw_exact  canon_exact  raw_meanCER  canon_meanCER")
    for name, summary in (
        ("baseline ", baseline_summary),
        ("specialist", specialist_summary),
        ("final    ", final_summary),
    ):
        print(
            f"  {name} {summary['raw_exact']:>9} {summary['canonical_exact']:>12} "
            f"{summary['raw_mean_cer']:>12.6f} {summary['canonical_mean_cer']:>14.6f}"
        )
    print(f"identity_invariant holds={invariant['holds']}")
    if invariant.get("dataset_identity_expected"):
        print("  (nothing routed: final must equal baseline everywhere)")
    for gate, ok in gates.items():
        print(f"  gate {gate}: {'PASS' if ok else 'FAIL'}")
    passed = all(gates.values())
    print(f"gates={'PASS' if passed else 'FAIL'}")
    print(f"wrote {args.out_jsonl}")
    print(f"wrote {args.report_dir}")
    if passed:
        return 0
    # Fail closed: a failing gate is a non-zero exit unless explicitly opted out.
    return 0 if args.report_only else 1


def render_markdown(report: dict) -> str:
    lines = [
        "# French Specialist Router Gate",
        "",
        f"- baseline: `{report['inputs']['baseline']}`",
        f"- specialist: `{report['inputs']['specialist']}`",
        "",
        report["metric_note"],
        "",
        "## Alignment",
        "",
        f"- baseline input rows: {report['alignment']['baseline_input_rows']}",
        f"- specialist input rows: {report['alignment']['specialist_input_rows']}",
        f"- scored rows: {report['alignment']['scored_rows']}",
        f"- duplicate keys (baseline): {len(report['alignment']['duplicate_keys_baseline'])}",
        f"- duplicate keys (specialist): {len(report['alignment']['duplicate_keys_specialist'])}",
        f"- missing keys: {len(report['alignment']['missing_keys'])}",
        f"- extra keys: {len(report['alignment']['extra_keys'])}",
        f"- metadata mismatches: {len(report['alignment']['metadata_mismatches'])}",
        "",
        "## Metrics (all recomputed by the same code path)",
        "",
        "| Run | Rows | Raw exact | Canonical exact | Raw mean CER | Canonical mean CER |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, key in (("Baseline", "baseline"), ("Specialist", "specialist"), ("Final", "final")):
        s = report[key]
        lines.append(
            f"| {name} | {s['rows']} | {s['raw_exact']} | {s['canonical_exact']} | "
            f"{s['raw_mean_cer']:.6f} | {s['canonical_mean_cer']:.6f} |"
        )
    lines.extend(["", "## Routes", ""])
    for route, count in report["routes"].items():
        lines.append(f"- {route}: {count}")
    invariant = report["identity_invariant"]
    lines.extend(
        [
            "",
            "## Prediction Identity Invariant",
            "",
            f"- specialist applied: {invariant['specialist_applied_count']}",
            f"- unrouted rows differing from baseline: {len(invariant['unrouted_text_mismatches'])}",
            f"- unrouted rows with differing metrics: {len(invariant['unrouted_metric_mismatches'])}",
            f"- holds: {invariant['holds']}",
        ]
    )
    lines.extend(
        [
            "",
            "## Defect Safety",
            "",
            f"- baseline false passes: {report['baseline_defects']['false_pass_count']}"
            f"/{report['baseline_defects']['eligible_rows']}",
            f"- final false passes: {report['final_defects']['false_pass_count']}"
            f"/{report['final_defects']['eligible_rows']}",
            "",
            "| Category | Baseline false passes | Final false passes |",
            "| --- | ---: | ---: |",
        ]
    )
    for category, metrics in report["final_category_defects"].items():
        baseline_metrics = report["baseline_category_defects"].get(
            category, {"false_pass_count": 0, "eligible_rows": 0}
        )
        lines.append(
            f"| {category} | {baseline_metrics['false_pass_count']}"
            f"/{baseline_metrics['eligible_rows']} | "
            f"{metrics['false_pass_count']}/{metrics['eligible_rows']} |"
        )
    lines.extend(["", "## Gates", ""])
    for gate, ok in report["gates"].items():
        lines.append(f"- {gate}: {'PASS' if ok else 'FAIL'}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
