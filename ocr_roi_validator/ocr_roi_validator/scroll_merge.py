from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
import re

from PIL import Image, ImageChops, ImageStat

from .compare import normalize_ui_text


def normalize_scroll_text(text: str) -> str:
    text = normalize_ui_text(text)
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
        best_new_positions = -1
        for start in range(len(cycle)):
            target = doubled[start:start + len(observed)]
            ratio = SequenceMatcher(None, target.casefold(), observed.casefold()).ratio()
            matcher = SequenceMatcher(None, target.casefold(), observed.casefold())
            new_positions = sum(
                1
                for block in matcher.get_matching_blocks()
                for offset in range(block.size)
                if (start + block.a + offset) % len(cycle) < len(expected)
                and (start + block.a + offset) % len(cycle) not in self._covered_positions
            )
            if ratio > best_ratio or (ratio == best_ratio and new_positions > best_new_positions):
                best_ratio = ratio
                best_start = start
                best_target = target
                best_new_positions = new_positions

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


@dataclass
class VerticalListAccumulator:
    expected_rows: list[str]
    require_loop: bool = True
    min_score: float = 0.25
    min_match_ratio: float = 0.65
    observed_rows: list[str] = field(default_factory=list)
    observed_indices: list[int] = field(default_factory=list)
    _candidate_index: int | None = None
    _candidate_text: str = ""
    _candidate_ratio: float = 0.0
    _candidate_hits: int = 0
    _flat_accumulator: ScrollTextAccumulator = field(init=False)
    _window_mode: bool = False
    _covered_row_indices: set[int] = field(default_factory=set)
    _last_window_start: int | None = None
    _window_order_violations: int = 0
    _window_wraps: int = 0
    _best_window_rows: dict[int, tuple[str, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.expected_rows = [normalize_scroll_text(row) for row in self.expected_rows if normalize_scroll_text(row)]
        self._flat_accumulator = ScrollTextAccumulator(
            min_length=1,
            min_score=self.min_score,
            expected_text=re.sub(r"\s+", "", "".join(self.expected_rows)),
        )

    def add(self, text: str, score: float) -> bool:
        rows = [normalize_scroll_text(row) for row in text.splitlines() if normalize_scroll_text(row)]
        if not rows or score < self.min_score or not self.expected_rows:
            return False

        self._flat_accumulator.add(re.sub(r"\s+", "", "".join(rows)), score)
        if len(rows) > 1:
            return self._add_window(rows)
        return self._add_row(rows[0])

    def _add_window(self, rows: list[str]) -> bool:
        self._window_mode = True
        count = len(self.expected_rows)
        best_start = 0
        best_ratios: list[float] = []
        best_score = -1.0
        best_new_rows = -1

        for start in range(count):
            ratios = [
                SequenceMatcher(None, self.expected_rows[(start + offset) % count].casefold(), row.casefold()).ratio()
                for offset, row in enumerate(rows)
            ]
            matching = [ratio for ratio in ratios if ratio >= self.min_match_ratio]
            alignment_score = sum(matching) / max(len(rows), 1)
            new_rows = sum(
                1
                for offset, ratio in enumerate(ratios)
                if ratio >= self.min_match_ratio and (start + offset) % count not in self._covered_row_indices
            )
            if alignment_score > best_score or (alignment_score == best_score and new_rows > best_new_rows):
                best_start = start
                best_ratios = ratios
                best_score = alignment_score
                best_new_rows = new_rows

        matched_indices = [
            (best_start + offset) % count
            for offset, ratio in enumerate(best_ratios)
            if ratio >= self.min_match_ratio
        ]
        minimum_matches = min(2, len(rows))
        if len(matched_indices) < minimum_matches or best_score < self.min_match_ratio * 0.5:
            return False

        misordered_rows = sum(
            1
            for row, aligned_ratio in zip(rows, best_ratios)
            if (best_independent_ratio := max(
                SequenceMatcher(None, expected.casefold(), row.casefold()).ratio()
                for expected in self.expected_rows
            )) >= self.min_match_ratio
            and best_independent_ratio >= aligned_ratio + 0.15
        )
        if misordered_rows >= 2:
            self._window_order_violations += 1

        if self._last_window_start is not None:
            forward_delta = (best_start - self._last_window_start) % count
            if best_start < self._last_window_start and 0 < forward_delta <= max(2, len(rows)):
                self._window_wraps += 1
            elif forward_delta > max(2, len(rows)):
                self._window_order_violations += 1
        self._last_window_start = best_start

        changed = False
        for offset, ratio in enumerate(best_ratios):
            if ratio < self.min_match_ratio:
                continue
            index = (best_start + offset) % count
            if index not in self._covered_row_indices:
                changed = True
            self._covered_row_indices.add(index)
            current = self._best_window_rows.get(index)
            if current is None or ratio > current[1]:
                self._best_window_rows[index] = (rows[offset], ratio)
        return changed

    def _add_row(self, text: str) -> bool:
        ratios = [SequenceMatcher(None, row.casefold(), text.casefold()).ratio() for row in self.expected_rows]
        best_ratio = max(ratios)
        candidate_indices = [
            index
            for index, ratio in enumerate(ratios)
            if ratio >= self.min_match_ratio and ratio >= best_ratio - 0.05
        ]
        if not candidate_indices:
            return False

        reference_index = self._candidate_index
        if reference_index is None and self.observed_indices:
            reference_index = self.observed_indices[-1]
        if self._candidate_index in candidate_indices:
            index = self._candidate_index
        elif reference_index is not None:
            count = len(self.expected_rows)
            index = min(
                candidate_indices,
                key=lambda candidate: ((candidate - reference_index) % count) or count,
            )
        else:
            index = candidate_indices[0]
        ratio = ratios[index]

        if index == self._candidate_index:
            self._candidate_hits += 1
            if ratio > self._candidate_ratio:
                self._candidate_text = text
                self._candidate_ratio = ratio
        else:
            self._commit_candidate()
            self._candidate_index = index
            self._candidate_text = text
            self._candidate_ratio = ratio
            self._candidate_hits = 1

        if self._candidate_hits >= 2 and (not self.observed_indices or self.observed_indices[-1] != index):
            self._commit_candidate()
            return True
        return False

    def _commit_candidate(self) -> None:
        if self._candidate_index is None or self._candidate_hits < 1:
            return
        if not self.observed_indices or self.observed_indices[-1] != self._candidate_index:
            self.observed_indices.append(self._candidate_index)
            self.observed_rows.append(self._candidate_text)
        current = self._best_window_rows.get(self._candidate_index)
        if current is None or self._candidate_ratio > current[1]:
            self._best_window_rows[self._candidate_index] = (self._candidate_text, self._candidate_ratio)
        self._candidate_index = None
        self._candidate_text = ""
        self._candidate_ratio = 0.0
        self._candidate_hits = 0

    def finalize(self) -> None:
        self._commit_candidate()

    @property
    def unique_indices(self) -> set[int]:
        return set(self._covered_row_indices) | set(self.observed_indices)

    @property
    def coverage(self) -> float:
        if not self.expected_rows:
            return 0.0
        return len(self.unique_indices) / len(self.expected_rows)

    @property
    def missing_indices(self) -> list[int]:
        return [index for index in range(len(self.expected_rows)) if index not in self.unique_indices]

    @property
    def order_valid(self) -> bool:
        single_row_order_valid = True
        if len(self.observed_indices) >= 2:
            count = len(self.expected_rows)
            max_forward_skip = max(2, count // 4)
            single_row_order_valid = all(
            0 < (current - previous) % count <= max_forward_skip
            for previous, current in zip(self.observed_indices, self.observed_indices[1:])
            )
        return single_row_order_valid and (not self._window_mode or self._window_order_violations == 0)

    @property
    def loop_seen(self) -> bool:
        count = len(self.expected_rows)
        max_forward_skip = max(2, count // 4)
        single_row_loop_seen = any(
            current < previous and 0 < (current - previous) % count <= max_forward_skip
            for previous, current in zip(self.observed_indices, self.observed_indices[1:])
        )
        return single_row_loop_seen or (self._window_mode and self._window_wraps > 0)

    @property
    def layout_passed(self) -> bool:
        return self.coverage == 1.0 and self.order_valid

    @property
    def flat_coverage(self) -> float:
        expected_length = 0
        matched_length = 0
        for index, expected_row in enumerate(self.expected_rows):
            expected_flat = re.sub(r"\s+", "", expected_row).casefold()
            observed_flat = re.sub(
                r"\s+",
                "",
                self._best_window_rows.get(index, ("", 0.0))[0],
            ).casefold()
            expected_length += len(expected_flat)
            matched_length += sum(
                block.size
                for block in SequenceMatcher(None, expected_flat, observed_flat).get_matching_blocks()
            )
        anchored_coverage = matched_length / expected_length if expected_length else 0.0
        if not self._window_mode:
            return anchored_coverage
        return max(anchored_coverage, self._flat_accumulator.coverage)

    @property
    def flat_passed(self) -> bool:
        if self.flat_coverage >= 0.99 and len(self._best_window_rows) == len(self.expected_rows):
            return self.flat_coverage >= 0.99
        if not self._window_mode:
            return False
        return (
            self._flat_accumulator.start_seen
            and self._flat_accumulator.end_seen
            and self._flat_accumulator.order_valid
            and self.flat_coverage >= 0.99
        )

    @property
    def passed(self) -> bool:
        return self.layout_passed and self.flat_passed

    @property
    def loop_status(self) -> str:
        if not self.require_loop:
            return "N/A"
        return "PASS" if self.loop_seen else "PENDING"

    @property
    def final_text(self) -> str:
        if self._best_window_rows:
            return "\n".join(
                self._best_window_rows.get(index, ("…", 0.0))[0]
                for index in range(len(self.expected_rows))
            )
        return "\n".join(self.observed_rows)
