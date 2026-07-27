from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher


@dataclass
class CompareResult:
    passed: bool
    score: float
    mode: str
    expected_processed: str
    actual_processed: str


_UI_IMAGE_TOKEN_RE = re.compile(r"\{\s*\d+\s*:\s*(img_[a-z0-9_]+)\s*\}", re.IGNORECASE)
_UI_IMAGE_SYMBOLS = {
    "img_start": "▶Ⅱ",
    "img_check": "✓",
    "img_checked": "✓",
    "img_checkmark": "✓",
}


def normalize_ui_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    text = text.translate(
        str.maketrans(
            {
                "‘": "'",
                "’": "'",
                "ʼ": "'",
                "\u00a0": " ",
                "\u202f": " ",
            }
        )
    )

    def replace_image_token(match: re.Match[str]) -> str:
        token = match.group(1).lower()
        return _UI_IMAGE_SYMBOLS.get(token, match.group(0))

    return _UI_IMAGE_TOKEN_RE.sub(replace_image_token, text)


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
    expected = normalize_ui_text(expected)
    actual = normalize_ui_text(actual)

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
