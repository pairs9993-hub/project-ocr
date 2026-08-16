"""Two separate accounting systems for what happened to an occurrence.

The v2 report listed a ``losses`` dictionary whose entries summed to 16,564
against 19,200 rows, which reads as double counting. It was not -- the losses
dictionary simply omitted the three clean outcomes -- but a reader cannot tell
those two situations apart, and an accounting scheme whose total does not
reconcile is not one you can check.

So the two questions are separated explicitly:

``terminal_reason``
    Where the occurrence stopped. Exactly one per row, mutually exclusive,
    summing to the row count. This is the funnel.

``diagnostic_flags``
    What was observed along the way. Zero or more per row. Useful detail, but
    never a denominator -- a row flagged both INSERTION_PRESENT and
    MULTIPLE_CHANGES_PRESENT is one row, not two.

:func:`assert_terminal_reason_invariant` fixes the reconciliation so the two can
never be silently conflated again.
"""

from __future__ import annotations

from collections import Counter

__all__ = [
    "TERMINAL_REASONS",
    "PIPELINE_TERMINALS",
    "RECOGNIZER_TERMINALS",
    "CLEAN_TERMINALS",
    "DIAGNOSTIC_FLAGS",
    "derive_flags",
    "assert_terminal_reason_invariant",
    "summarise_terminal_reasons",
]

# Where an occurrence stopped. Exactly one applies to any row.
PIPELINE_TERMINALS = (
    "NOT_ELIGIBLE_NO_TARGET", "RENDER_ERROR", "DETECTOR_ERROR", "DETECTOR_MISS",
    "WRONG_LINE_SELECTED", "NO_TARGET_TOKEN", "RECOGNIZER_FAILURE",
    "ALIGNMENT_AMBIGUITY",
)
RECOGNIZER_TERMINALS = (
    "INSERTION", "DELETION", "CHANGE_ELSEWHERE", "MULTIPLE_CHANGES",
    "ACCENT_LOST", "OTHER_SUBSTITUTION",
)
CLEAN_TERMINALS = (
    "CLEAN_CORRECT_BARE_E", "CLEAN_PRESERVATION", "CLEAN_HALLUCINATION",
)
TERMINAL_REASONS = PIPELINE_TERMINALS + RECOGNIZER_TERMINALS + CLEAN_TERMINALS

# Observations that may co-occur. Never a denominator.
DIAGNOSTIC_FLAGS = (
    "DETECTOR_SPLIT_LINE", "LENGTH_MISMATCH", "INSERTION_PRESENT",
    "DELETION_PRESENT", "MULTIPLE_CHANGES_PRESENT", "TARGET_UNCHANGED",
    "NON_TARGET_CHANGE_PRESENT", "CROP_CLIPPED", "HIGH_PADDING_RATIO",
    "MULTIPLE_E_FORMS_IN_LINE",
)


def derive_flags(row: dict) -> list[str]:
    """Non-exclusive observations for one row, derived from what was recorded."""
    flags: list[str] = []
    if (row.get("detector_box_count") or 0) > 1:
        flags.append("DETECTOR_SPLIT_LINE")
    expected = row.get("expected_substring")
    decoded_length = row.get("decoded_length")
    if expected is not None and decoded_length is not None:
        if len(expected) != decoded_length:
            flags.append("LENGTH_MISMATCH")
            flags.append("INSERTION_PRESENT" if decoded_length > len(expected)
                         else "DELETION_PRESENT")
    if row.get("outcome") == "MULTIPLE_CHANGES":
        flags.append("MULTIPLE_CHANGES_PRESENT")
    if row.get("outcome") in {"CLEAN_CORRECT_BARE_E", "CLEAN_PRESERVATION"}:
        flags.append("TARGET_UNCHANGED")
    if row.get("outcome") in {"CHANGE_ELSEWHERE", "MULTIPLE_CHANGES"}:
        flags.append("NON_TARGET_CHANGE_PRESENT")
    if row.get("clipped"):
        flags.append("CROP_CLIPPED")
    ratio = row.get("horizontal_padding_ratio")
    if ratio is not None and ratio > 0.5:
        flags.append("HIGH_PADDING_RATIO")
    if expected:
        if sum(1 for character in expected if character in {"e", "é"}) > 1:
            flags.append("MULTIPLE_E_FORMS_IN_LINE")
    return flags


def assert_terminal_reason_invariant(rows: list[dict]) -> dict:
    """Check that terminal reasons partition the rows exactly.

    Raises if a row carries an unknown or missing reason, or if the counts fail
    to reconcile with the row total.
    """
    counts = Counter()
    for index, row in enumerate(rows):
        reason = row.get("terminal_reason") or row.get("outcome")
        if reason is None:
            raise ValueError(f"row {index} has no terminal reason")
        if reason not in TERMINAL_REASONS:
            raise ValueError(f"row {index} has unknown terminal reason {reason!r}")
        counts[reason] += 1
    total = sum(counts.values())
    if total != len(rows):
        raise ValueError(
            f"terminal reasons sum to {total} but there are {len(rows)} rows")
    return dict(counts)


def summarise_terminal_reasons(rows: list[dict]) -> dict:
    """Reconciled funnel plus a clearly separated multi-label flag tally."""
    counts = assert_terminal_reason_invariant(rows)
    flags = Counter()
    for row in rows:
        for flag in row.get("diagnostic_flags") or derive_flags(row):
            flags[flag] += 1
    grouped = {
        "pipeline": {k: counts.get(k, 0) for k in PIPELINE_TERMINALS},
        "recognizer": {k: counts.get(k, 0) for k in RECOGNIZER_TERMINALS},
        "clean": {k: counts.get(k, 0) for k in CLEAN_TERMINALS},
    }
    return {
        "terminal_reason_counts": counts,
        "terminal_reason_total": sum(counts.values()),
        "row_total": len(rows),
        "reconciles": sum(counts.values()) == len(rows),
        "by_group": grouped,
        "group_totals": {name: sum(values.values())
                         for name, values in grouped.items()},
        "diagnostic_flag_counts": dict(flags),
        "diagnostic_flag_total": sum(flags.values()),
        "flags_are_multi_label": True,
        "note": ("diagnostic_flag_total counts observations, not rows, and is "
                 "expected to differ from row_total; only terminal reasons "
                 "partition the rows"),
    }
