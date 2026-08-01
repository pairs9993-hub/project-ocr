from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


DIACRITIC_CHARS = frozenset("àâäçéèêëîïôöùûüÿáíóúñÀÂÄÇÉÈÊËÎÏÔÖÙÛÜŸÁÍÓÚÑ")
CONFUSABLE_GROUPS = (
    frozenset("iIl1|Ⅱ"),
    frozenset("oO0"),
)
VERDICT_RANK = {"FAIL": 0, "WARN": 1, "PASS": 2}
_WS_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    text = text.translate(str.maketrans({"‘": "'", "’": "'", "ʼ": "'"}))
    return _WS_RE.sub(" ", text).strip()


def levenshtein(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, 1):
        current = [left_index]
        for right_index, right_character in enumerate(right, 1):
            current.append(
                min(
                    previous[right_index] + 1,
                    current[-1] + 1,
                    previous[right_index - 1] + (left_character != right_character),
                )
            )
        previous = current
    return previous[-1]


def cer(reference: str, prediction: str) -> float:
    reference = normalize_text(reference)
    prediction = normalize_text(prediction)
    return levenshtein(reference, prediction) / max(1, len(reference))


def verdict(cer_value: float) -> str:
    if cer_value <= 0.05:
        return "PASS"
    if cer_value <= 0.20:
        return "WARN"
    return "FAIL"


def base_character(character: str) -> str:
    return "".join(
        part
        for part in unicodedata.normalize("NFD", character)
        if unicodedata.category(part) != "Mn"
    )


def align_characters(reference: str, prediction: str) -> list[tuple[str, str]]:
    reference = normalize_text(reference)
    prediction = normalize_text(prediction)
    rows = len(reference) + 1
    columns = len(prediction) + 1
    costs = [[0] * columns for _ in range(rows)]
    moves = [[""] * columns for _ in range(rows)]

    for row in range(1, rows):
        costs[row][0] = row
        moves[row][0] = "delete"
    for column in range(1, columns):
        costs[0][column] = column
        moves[0][column] = "insert"

    for row in range(1, rows):
        for column in range(1, columns):
            substitution_cost = costs[row - 1][column - 1] + (
                reference[row - 1] != prediction[column - 1]
            )
            candidates = (
                (substitution_cost, "match"),
                (costs[row - 1][column] + 1, "delete"),
                (costs[row][column - 1] + 1, "insert"),
            )
            costs[row][column], moves[row][column] = min(candidates, key=lambda item: item[0])

    alignment: list[tuple[str, str]] = []
    row = len(reference)
    column = len(prediction)
    while row or column:
        move = moves[row][column]
        if move == "match":
            alignment.append((reference[row - 1], prediction[column - 1]))
            row -= 1
            column -= 1
        elif move == "delete":
            alignment.append((reference[row - 1], ""))
            row -= 1
        else:
            alignment.append(("", prediction[column - 1]))
            column -= 1
    alignment.reverse()
    return alignment


def character_metrics(rows: Iterable[dict]) -> dict:
    substitutions: Counter[str] = Counter()
    diacritic_hallucinations = 0
    diacritic_deletions = 0
    confusable_substitutions = 0
    target_reference_count = 0
    target_prediction_count = 0
    target_correct_count = 0

    for row in rows:
        reference = str(row.get("reference", row.get("ref", "")))
        prediction = str(row.get("prediction", row.get("hyp", "")))
        for expected, actual in align_characters(reference, prediction):
            if expected in DIACRITIC_CHARS:
                target_reference_count += 1
            if actual in DIACRITIC_CHARS:
                target_prediction_count += 1
            if expected in DIACRITIC_CHARS and expected == actual:
                target_correct_count += 1
            if not expected or not actual or expected == actual:
                continue

            substitutions[f"{expected}->{actual}"] += 1
            expected_base = base_character(expected).casefold()
            actual_base = base_character(actual).casefold()
            same_base = expected_base == actual_base
            if same_base and expected not in DIACRITIC_CHARS and actual in DIACRITIC_CHARS:
                diacritic_hallucinations += 1
            if same_base and expected in DIACRITIC_CHARS and actual not in DIACRITIC_CHARS:
                diacritic_deletions += 1
            if any(expected in group and actual in group for group in CONFUSABLE_GROUPS):
                confusable_substitutions += 1

    recall = target_correct_count / target_reference_count if target_reference_count else 0.0
    precision = target_correct_count / target_prediction_count if target_prediction_count else 0.0
    return {
        "diacritic_reference_count": target_reference_count,
        "diacritic_prediction_count": target_prediction_count,
        "diacritic_correct_count": target_correct_count,
        "diacritic_recall": recall,
        "diacritic_precision": precision,
        "diacritic_hallucinations": diacritic_hallucinations,
        "diacritic_deletions": diacritic_deletions,
        "confusable_substitutions": confusable_substitutions,
        "top_substitutions": dict(substitutions.most_common(30)),
    }


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL: {exc}") from exc
    return rows


def row_key(row: dict) -> str:
    return str(row.get("filename") or Path(str(row.get("image_path", ""))).name)


def summarize(rows: list[dict]) -> dict:
    verdicts = Counter(str(row.get("verdict", "")) for row in rows)
    cer_values = [float(row.get("cer", 0.0)) for row in rows]
    return {
        "rows": len(rows),
        "mean_cer": sum(cer_values) / len(cer_values) if cer_values else 0.0,
        "pass": verdicts["PASS"],
        "warn": verdicts["WARN"],
        "fail": verdicts["FAIL"],
        "exact": sum(value == 0.0 for value in cer_values),
        "characters": character_metrics(rows),
    }


def category_metrics(rows: list[dict]) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("category", "uncategorized"))].append(row)

    result = {}
    for category, category_rows in sorted(grouped.items()):
        summary = summarize(category_rows)
        result[category] = {
            "rows": summary["rows"],
            "mean_cer": summary["mean_cer"],
            "exact": summary["exact"],
            "diacritic_hallucinations": summary["characters"]["diacritic_hallucinations"],
            "diacritic_deletions": summary["characters"]["diacritic_deletions"],
            "confusable_substitutions": summary["characters"]["confusable_substitutions"],
            "defects": defect_false_passes(category_rows),
        }
    return result


def compare_runs(baseline_rows: list[dict], candidate_rows: list[dict]) -> dict:
    baseline = {row_key(row): row for row in baseline_rows}
    candidate = {row_key(row): row for row in candidate_rows}
    common = sorted(set(baseline) & set(candidate))
    missing_candidate = sorted(set(baseline) - set(candidate))
    extra_candidate = sorted(set(candidate) - set(baseline))

    cer_improved = []
    cer_worse = []
    verdict_improved = []
    verdict_worse = []
    baseline_pass_regressions = []
    exact_regressions = []
    for key in common:
        old = baseline[key]
        new = candidate[key]
        old_cer = float(old.get("cer", 0.0))
        new_cer = float(new.get("cer", 0.0))
        old_verdict = str(old.get("verdict", ""))
        new_verdict = str(new.get("verdict", ""))
        if new_cer < old_cer:
            cer_improved.append(key)
        elif new_cer > old_cer:
            cer_worse.append(key)
        if VERDICT_RANK.get(new_verdict, -1) > VERDICT_RANK.get(old_verdict, -1):
            verdict_improved.append(key)
        elif VERDICT_RANK.get(new_verdict, -1) < VERDICT_RANK.get(old_verdict, -1):
            verdict_worse.append(key)
        if old_verdict == "PASS" and new_verdict != "PASS":
            baseline_pass_regressions.append(key)
        if old_cer == 0.0 and new_cer > 0.0:
            exact_regressions.append(key)

    baseline_summary = summarize(baseline_rows)
    candidate_summary = summarize(candidate_rows)
    baseline_defects = defect_false_passes(baseline_rows)
    candidate_defects = defect_false_passes(candidate_rows)
    return {
        "baseline": baseline_summary,
        "candidate": candidate_summary,
        "baseline_categories": category_metrics(baseline_rows),
        "candidate_categories": category_metrics(candidate_rows),
        "baseline_defects": baseline_defects,
        "candidate_defects": candidate_defects,
        "pairwise": {
            "common_rows": len(common),
            "missing_candidate": missing_candidate,
            "extra_candidate": extra_candidate,
            "cer_improved": len(cer_improved),
            "cer_worse": len(cer_worse),
            "cer_same": len(common) - len(cer_improved) - len(cer_worse),
            "verdict_improved": len(verdict_improved),
            "verdict_worse": len(verdict_worse),
            "verdict_same": len(common) - len(verdict_improved) - len(verdict_worse),
            "baseline_pass_regressions": baseline_pass_regressions,
            "exact_regressions": exact_regressions,
        },
        "gates": {
            "same_rows": not missing_candidate and not extra_candidate,
            "mean_cer_not_worse": candidate_summary["mean_cer"] <= baseline_summary["mean_cer"],
            "no_pass_regressions": not baseline_pass_regressions,
            "no_exact_regressions": not exact_regressions,
            "no_added_diacritic_hallucinations": (
                candidate_summary["characters"]["diacritic_hallucinations"]
                <= baseline_summary["characters"]["diacritic_hallucinations"]
            ),
            "no_added_defect_false_passes": (
                candidate_defects["false_pass_count"] <= baseline_defects["false_pass_count"]
            ),
        },
    }


def defect_false_passes(rows: list[dict]) -> dict:
    eligible = []
    false_passes = []
    for row in rows:
        visible = normalize_text(str(row.get("visible_text", row.get("reference", ""))))
        expected = normalize_text(str(row.get("expected", "")))
        prediction = normalize_text(str(row.get("prediction", row.get("hyp", ""))))
        if not expected or visible == expected:
            continue
        key = row_key(row)
        eligible.append(key)
        if prediction == expected:
            false_passes.append(key)
    return {
        "eligible_rows": len(eligible),
        "false_pass_count": len(false_passes),
        "false_pass_rate": len(false_passes) / len(eligible) if eligible else 0.0,
        "false_passes": false_passes,
    }


def render_markdown(report: dict) -> str:
    baseline = report["baseline"]
    candidate = report["candidate"]
    pairwise = report["pairwise"]
    lines = [
        "# OCR Promotion Gate",
        "",
        "| Run | Rows | PASS | WARN | FAIL | Mean CER | Exact |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| Baseline | {baseline['rows']} | {baseline['pass']} | {baseline['warn']} | {baseline['fail']} | {baseline['mean_cer']:.4f} | {baseline['exact']} |",
        f"| Candidate | {candidate['rows']} | {candidate['pass']} | {candidate['warn']} | {candidate['fail']} | {candidate['mean_cer']:.4f} | {candidate['exact']} |",
        "",
        "## Pairwise",
        "",
        f"- CER improved/worse/same: {pairwise['cer_improved']}/{pairwise['cer_worse']}/{pairwise['cer_same']}",
        f"- Verdict improved/worse/same: {pairwise['verdict_improved']}/{pairwise['verdict_worse']}/{pairwise['verdict_same']}",
        f"- Baseline PASS regressions: {len(pairwise['baseline_pass_regressions'])}",
        f"- Exact regressions: {len(pairwise['exact_regressions'])}",
        "",
        "## Character Safety",
        "",
        "| Run | Diacritic precision | Diacritic recall | Hallucinations | Deletions | Confusable substitutions |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, summary in (("Baseline", baseline), ("Candidate", candidate)):
        metrics = summary["characters"]
        lines.append(
            f"| {name} | {metrics['diacritic_precision']:.2%} | {metrics['diacritic_recall']:.2%} | "
            f"{metrics['diacritic_hallucinations']} | "
            f"{metrics['diacritic_deletions']} | {metrics['confusable_substitutions']} |"
        )
    lines.extend(["", "## Gates", ""])
    for gate, passed in report["gates"].items():
        lines.append(f"- {gate}: {'PASS' if passed else 'FAIL'}")
    baseline_defects = report.get("baseline_defects")
    candidate_defects = report.get("candidate_defects")
    if baseline_defects and candidate_defects:
        lines.extend(
            [
                "",
                "## Paired Defect Safety",
                "",
                f"- Baseline false passes: {baseline_defects['false_pass_count']}/{baseline_defects['eligible_rows']}",
                f"- Candidate false passes: {candidate_defects['false_pass_count']}/{candidate_defects['eligible_rows']}",
            ]
        )
    baseline_categories = report.get("baseline_categories", {})
    candidate_categories = report.get("candidate_categories", {})
    categories = sorted(set(baseline_categories) | set(candidate_categories))
    if categories:
        lines.extend(
            [
                "",
                "## Category Breakdown",
                "",
                "| Category | Run | Exact | Mean CER | Hallucinations | Deletions | Confusables | Defect false passes |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for category in categories:
            for name, metrics in (
                ("Baseline", baseline_categories.get(category)),
                ("Candidate", candidate_categories.get(category)),
            ):
                if metrics is None:
                    continue
                lines.append(
                    f"| {category} | {name} | {metrics['exact']}/{metrics['rows']} | "
                    f"{metrics['mean_cer']:.4f} | {metrics['diacritic_hallucinations']} | "
                    f"{metrics['diacritic_deletions']} | {metrics['confusable_substitutions']} | "
                    f"{metrics['defects']['false_pass_count']}/{metrics['defects']['eligible_rows']} |"
                )
    if "defects" in report:
        defects = report["defects"]
        lines.extend(
            [
                "",
                "## Defect Injection",
                "",
                f"- Eligible rows: {defects['eligible_rows']}",
                f"- False passes: {defects['false_pass_count']} ({defects['false_pass_rate']:.2%})",
            ]
        )
    return "\n".join(lines) + "\n"