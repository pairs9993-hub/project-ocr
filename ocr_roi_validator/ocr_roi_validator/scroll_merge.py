from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
import re

from PIL import Image, ImageChops, ImageStat


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


def _collapse_missing_runs(text: str) -> str:
    return re.sub(r"□+", "…", text).strip()


@dataclass
class AdaptiveFrameSampler:
    min_interval_sec: float = 0.4
    max_interval_sec: float = 1.0
    change_threshold: float = 0.5
    signature_size: tuple[int, int] = (64, 16)
    _last_sample_time: float | None = None
    _last_signature: Image.Image | None = None

    def should_sample(self, image: Image.Image, now: float) -> bool:
        signature = image.convert("L").resize(self.signature_size, Image.Resampling.BILINEAR)
        if self._last_sample_time is None or self._last_signature is None:
            self._accept(signature, now)
            return True

        elapsed = now - self._last_sample_time
        epsilon = 1e-9
        if elapsed + epsilon < self.min_interval_sec:
            return False

        difference = ImageStat.Stat(ImageChops.difference(signature, self._last_signature)).mean[0]
        if elapsed + epsilon >= self.max_interval_sec or difference >= self.change_threshold:
            self._accept(signature, now)
            return True
        return False

    def _accept(self, signature: Image.Image, now: float) -> None:
        self._last_signature = signature
        self._last_sample_time = now


@dataclass
class ScrollTextAccumulator:
    min_length: int = 3
    min_score: float = 0.25
    expected_text: str = ""
    best_text: str = ""
    best_score: float = 0.0
    accepted_count: int = 0
    history: list[str] = field(default_factory=list)
    _covered_positions: set[int] = field(default_factory=set)
    _last_alignment_start: int | None = None
    _order_violations: int = 0
    _completed_wraps: int = 0
    _last_coverage_improvement_time: float | None = None

    def __post_init__(self) -> None:
        self.expected_text = normalize_scroll_text(self.expected_text)

    def add(self, text: str, score: float, observed_at: float | None = None) -> bool:
        text = normalize_scroll_text(text)
        if len(text) < self.min_length:
            return False
        if score < self.min_score:
            return False

        self.accepted_count += 1
        self.history.append(text)
        coverage_before = len(self._covered_positions)
        self._update_expected_coverage(text)
        if len(self._covered_positions) > coverage_before and observed_at is not None:
            self._last_coverage_improvement_time = observed_at

        if not self.best_text:
            self.best_text = text
            self.best_score = score
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
            return True

        if similarity >= 0.65 and score >= self.best_score * 0.9:
            self.best_text = merged
            self.best_score = max(self.best_score, score)
            return True

        return False

    def _update_expected_coverage(self, observed: str) -> None:
        if not self.expected_text:
            return

        expected = self.expected_text
        cycle = expected + " "
        doubled = cycle + cycle
        best_ratio = 0.0
        best_start = 0
        best_target = ""
        for start in range(len(cycle)):
            target = doubled[start:start + len(observed)]
            ratio = SequenceMatcher(None, target.casefold(), observed.casefold()).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_start = start
                best_target = target

        if best_ratio < 0.45:
            return

        if self._last_alignment_start is not None and best_ratio >= 0.65:
            cycle_length = len(cycle)
            signed_delta = (
                (best_start - self._last_alignment_start + cycle_length // 2)
                % cycle_length
            ) - cycle_length // 2
            backward_tolerance = max(2, min(5, cycle_length // 20))
            raw_delta = best_start - self._last_alignment_start
            if raw_delta < 0 and signed_delta >= -backward_tolerance:
                self._completed_wraps += 1
            if signed_delta < -backward_tolerance:
                self._order_violations += 1
        self._last_alignment_start = best_start

        matcher = SequenceMatcher(None, best_target.casefold(), observed.casefold())
        for block in matcher.get_matching_blocks():
            for offset in range(block.size):
                cycle_position = (best_start + block.a + offset) % len(cycle)
                if cycle_position < len(expected):
                    self._covered_positions.add(cycle_position)

    @property
    def coverage(self) -> float:
        if not self.expected_text:
            return 0.0
        return len(self._covered_positions) / len(self.expected_text)

    @property
    def start_seen(self) -> bool:
        if not self.expected_text:
            return False
        edge_length = max(3, min(10, len(self.expected_text) // 5))
        return all(position in self._covered_positions for position in range(edge_length))

    @property
    def end_seen(self) -> bool:
        if not self.expected_text:
            return False
        edge_length = max(3, min(10, len(self.expected_text) // 5))
        start = len(self.expected_text) - edge_length
        return all(position in self._covered_positions for position in range(start, len(self.expected_text)))

    @property
    def cycle_complete(self) -> bool:
        return self.start_seen and self.end_seen and self.order_valid and self.coverage >= 0.9

    @property
    def order_valid(self) -> bool:
        return self._order_violations == 0

    @property
    def completed_wraps(self) -> int:
        return self._completed_wraps

    def ready_to_stop(self, now: float, stagnation_sec: float = 4.0) -> bool:
        if not self.cycle_complete:
            return False
        if self.coverage >= 0.99:
            return True
        if self.completed_wraps < 2 or self._last_coverage_improvement_time is None:
            return False
        return now - self._last_coverage_improvement_time >= stagnation_sec

    @property
    def reconstructed_text(self) -> str:
        if not self.expected_text:
            return self.best_text
        reconstructed = "".join(
            char if index in self._covered_positions else "□"
            for index, char in enumerate(self.expected_text)
        )
        return _collapse_missing_runs(reconstructed)

    @property
    def final_text(self) -> str:
        return self.reconstructed_text if self.expected_text else self.best_text
