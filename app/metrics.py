"""Tiny text-metric utilities (self-contained, no project imports).

Keeps the app folder runnable in isolation when copied to another machine.
"""

from __future__ import annotations

import re
import unicodedata

from rapidfuzz.distance import Levenshtein

_WS = re.compile(r"\s+")


def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFC", s)
    s = s.replace("\u200f", "").replace("\u200e", "")  # strip RTL/LTR marks
    return _WS.sub(" ", s).strip()


def cer(ref: str, hyp: str) -> float:
    ref = normalize_text(ref)
    hyp = normalize_text(hyp)
    if not ref:
        return 0.0 if not hyp else 1.0
    return Levenshtein.distance(ref, hyp) / len(ref)


def wer(ref: str, hyp: str) -> float:
    ref_tokens = normalize_text(ref).split(" ")
    hyp_tokens = normalize_text(hyp).split(" ")
    if not ref_tokens or (len(ref_tokens) == 1 and ref_tokens[0] == ""):
        return 0.0 if not hyp_tokens or hyp_tokens == [""] else 1.0
    return Levenshtein.distance(ref_tokens, hyp_tokens) / len(ref_tokens)


def verdict(cer_v: float) -> str:
    """Return PASS / WARN / FAIL bucket for visualization."""
    if cer_v <= 0.05:
        return "PASS"
    if cer_v <= 0.20:
        return "WARN"
    return "FAIL"
