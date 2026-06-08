from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher


@dataclass
class CompareResult:
    passed: bool
    score: float
    mode: str
    expected_processed: str
    actual_processed: str


def _normalize_ignore_case(text: str) -> str:
    return text.lower()


def _normalize_ignore_space(text: str) -> str:
    return re.sub(r"\s+", "", text)


def compare_text(
    expected: str,
    actual: str,
    mode: str,
    similarity_threshold: float = 0.9,
) -> CompareResult:
    expected = expected or ""
    actual = actual or ""

    if mode == "exact":
        exp = expected
        act = actual
        score = 1.0 if exp == act else 0.0
        return CompareResult(score == 1.0, score, mode, exp, act)

    if mode == "ignore_case":
        exp = _normalize_ignore_case(expected)
        act = _normalize_ignore_case(actual)
        score = 1.0 if exp == act else 0.0
        return CompareResult(score == 1.0, score, mode, exp, act)

    if mode == "ignore_space":
        exp = _normalize_ignore_space(expected)
        act = _normalize_ignore_space(actual)
        score = 1.0 if exp == act else 0.0
        return CompareResult(score == 1.0, score, mode, exp, act)

    if mode == "similarity":
        exp = expected.strip()
        act = actual.strip()
        score = SequenceMatcher(None, exp, act).ratio()
        return CompareResult(score >= similarity_threshold, score, mode, exp, act)

    if mode == "regex":
        exp = expected
        act = actual
        try:
            matched = re.search(exp, act) is not None
        except re.error:
            matched = False
        return CompareResult(matched, 1.0 if matched else 0.0, mode, exp, act)

    raise ValueError(f"Unsupported compare mode: {mode}")
