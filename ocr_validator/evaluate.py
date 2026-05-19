"""Evaluation utilities: CER/WER and aggregation by metadata.

CER (Character Error Rate) = Levenshtein(ref, hyp) / max(len(ref), 1)
WER (Word Error Rate)      = Levenshtein over word-tokenised sequences

Both are computed on whitespace-normalized but case-preserving text. Use
`normalize_text` to strip arbitrary whitespace and BOM but keep accents
and non-Latin scripts untouched.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

from rapidfuzz.distance import Levenshtein


_WS = re.compile(r"\s+")


def normalize_text(s: str) -> str:
    """Collapse whitespace and normalize Unicode (NFC). Preserve case + scripts."""
    if s is None:
        return ""
    s = unicodedata.normalize("NFC", s)
    s = s.replace("‏", "").replace("‎", "")  # strip RTL/LTR marks
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


@dataclass
class Sample:
    image_path: str
    language: str
    pattern: str
    background: str
    ref: str
    hyp: str
    cer_v: float
    wer_v: float
    elapsed_ms: float = 0.0


def aggregate(samples: Iterable[Sample], group_key) -> List[Tuple[str, int, float, float]]:
    """Return [(group, count, mean_cer, mean_wer)] sorted by count desc."""
    buckets: Dict[str, List[Sample]] = {}
    for s in samples:
        k = group_key(s)
        buckets.setdefault(k, []).append(s)
    rows = []
    for k, items in buckets.items():
        n = len(items)
        mc = sum(s.cer_v for s in items) / n
        mw = sum(s.wer_v for s in items) / n
        rows.append((k, n, mc, mw))
    rows.sort(key=lambda r: -r[1])
    return rows


def overall(samples: List[Sample]) -> Tuple[float, float]:
    if not samples:
        return 0.0, 0.0
    n = len(samples)
    return (
        sum(s.cer_v for s in samples) / n,
        sum(s.wer_v for s in samples) / n,
    )


# Map language tags to script families (for grouping in reports)
LANG_SCRIPT = {
    "en": "latin", "fr": "latin", "de": "latin", "it": "latin", "es": "latin",
    "pt": "latin", "nl": "latin", "no": "latin", "pl": "latin", "cs": "latin",
    "lt": "latin", "lv": "latin", "vi": "latin",
    "ru": "cyrillic", "uk": "cyrillic", "bg": "cyrillic",
    "ar": "arabic", "ar_eg": "arabic",
    "el": "greek",
    "th": "thai",
    "zh_cn": "chinese_simplified",
    "zh_tw": "chinese_traditional",
}


def script_of(lang: str) -> str:
    return LANG_SCRIPT.get(lang, "other")
