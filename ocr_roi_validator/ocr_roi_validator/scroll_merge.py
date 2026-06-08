from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
import re


def normalize_scroll_text(text: str) -> str:
    text = text or ""
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _best_overlap_suffix_prefix(left: str, right: str) -> int:
    max_len = min(len(left), len(right))
    max_len = min(max_len, 128)
    for overlap in range(max_len, 1, -1):
        if left[-overlap:] == right[:overlap]:
            return overlap
    return 0


def merge_scroll_text(existing: str, incoming: str) -> str:
    existing = normalize_scroll_text(existing)
    incoming = normalize_scroll_text(incoming)

    if not existing:
        return incoming
    if not incoming:
        return existing
    if incoming in existing:
        return existing
    if existing in incoming:
        return incoming

    suffix_prefix = _best_overlap_suffix_prefix(existing, incoming)
    prefix_suffix = _best_overlap_suffix_prefix(incoming, existing)

    if suffix_prefix >= 2 and suffix_prefix >= prefix_suffix:
        return normalize_scroll_text(existing + incoming[suffix_prefix:])
    if prefix_suffix >= 2:
        return normalize_scroll_text(incoming + existing[prefix_suffix:])

    similarity = SequenceMatcher(None, existing, incoming).ratio()
    if len(incoming) > len(existing) and similarity >= 0.45:
        return incoming
    return existing


@dataclass
class ScrollTextAccumulator:
    min_length: int = 3
    min_score: float = 0.25
    best_text: str = ""
    best_score: float = 0.0
    accepted_count: int = 0
    history: list[str] = field(default_factory=list)

    def add(self, text: str, score: float) -> bool:
        text = normalize_scroll_text(text)
        if len(text) < self.min_length:
            return False
        if score < self.min_score:
            return False

        if not self.best_text:
            self.best_text = text
            self.best_score = score
            self.accepted_count += 1
            self.history.append(text)
            return True

        merged = merge_scroll_text(self.best_text, text)
        similarity = SequenceMatcher(None, self.best_text, text).ratio()

        merged_is_better = len(merged) > len(self.best_text)
        candidate_is_better = len(text) > len(self.best_text) and similarity >= 0.45
        strong_candidate = score >= self.best_score + 0.15 and similarity >= 0.45

        if merged_is_better or candidate_is_better or strong_candidate:
            if merged_is_better:
                self.best_text = merged
            elif candidate_is_better or strong_candidate:
                self.best_text = text
            self.best_score = max(self.best_score, score)
            self.accepted_count += 1
            self.history.append(text)
            return True

        if similarity >= 0.65 and score >= self.best_score * 0.9:
            self.best_text = merged
            self.best_score = max(self.best_score, score)
            self.accepted_count += 1
            self.history.append(text)
            return True

        return False

    @property
    def final_text(self) -> str:
        return self.best_text
