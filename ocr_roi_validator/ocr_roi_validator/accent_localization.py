"""Fail-closed quality gate for a CTC-derived glyph crop.

The accent classifier answers a question about pixels, and it answers it well
when those pixels are the glyph it was asked about. The accent-v1 failure was
not a classification error: the CTC span had drifted a character left, so the
crop held a neighbouring ``r`` and no accent at all. The classifier said "no
accent", which was true of the crop and wrong about the text.

The fix belongs here rather than in a threshold. Before the classifier is
consulted, the crop must be shown to isolate a single character.

Design constraints
------------------
Two observations shaped this, both measured on the validation split:

* Blank borders are useless as a signal. At UI resolution the median CTC span
  has *zero* margin on both sides for genuine and drifted crops alike, so
  requiring a clear border rejects nearly everything.
* Ink aspect ratio does separate the known drift case from correct catches,
  but only by a margin narrower than the spread of correct catches themselves.
  Using it would mean fitting a threshold to one image, which is exactly the
  overfitting this gate is meant to avoid.

What is left is structural: a span that holds one character should be about as
wide as its neighbours on the same line, and its ink should sit inside it
rather than run out of one end. Those are compared against the line's own
statistics, so they carry no absolute pixel or font assumptions.

No expected text is involved at any point.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

__all__ = [
    "LocalizationConfig",
    "LocalizationReport",
    "assess_localization",
]


@dataclass(frozen=True)
class LocalizationConfig:
    """Thresholds for the localization quality gate.

    Every value is a ratio against the line's own geometry, never an absolute
    pixel count, so the gate behaves the same at any font size or resolution.
    """

    # Span width relative to the median CTC span width on the same line. A
    # character span far outside this band is merging or clipping characters.
    min_width_ratio: float = 0.55
    max_width_ratio: float = 1.90
    # How far the ink centroid may sit from the middle of the span. A crop
    # holding its neighbour rather than its own glyph is off-centre.
    max_centre_offset_ratio: float = 0.22
    # Ink must fill a plausible share of the span: too little means the glyph
    # is mostly outside it.
    min_ink_width_ratio: float = 0.30
    # Ink extending to both edges at once suggests the span sits between two
    # characters rather than around one. Either edge alone is normal.
    reject_ink_on_both_edges: bool = True
    # How far the crop bounds are moved when checking verdict stability.
    jitter_pixels: int = 2

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class LocalizationReport:
    """Whether a crop is trustworthy enough to classify."""

    usable: bool
    reasons: tuple[str, ...]
    width_ratio: float
    centre_offset_ratio: float
    ink_width_ratio: float
    touches_both_edges: bool


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        return image[..., :3].mean(axis=2)
    return image.astype(np.float64)


def _ink_mask(gray: np.ndarray) -> np.ndarray:
    low, high = float(gray.min()), float(gray.max())
    if high - low < 12.0:
        return np.zeros(gray.shape, dtype=bool)
    midpoint = (low + high) / 2.0
    bright = gray >= midpoint
    return bright if bright.mean() < 0.5 else ~bright


def assess_localization(
    line_image: np.ndarray,
    x0: int,
    x1: int,
    median_span_width: float | None = None,
    config: LocalizationConfig | None = None,
) -> LocalizationReport:
    """Judge whether ``[x0:x1]`` of ``line_image`` isolates a single glyph.

    ``median_span_width`` is the median CTC span width on the same line and is
    the reference for "one character wide". Without it the width check is
    skipped; the remaining geometry checks still apply.

    Returns a report; ``usable`` false means the caller must abstain.
    """
    config = config or LocalizationConfig()
    reasons: list[str] = []

    if line_image is None or line_image.size == 0:
        return LocalizationReport(False, ("empty_line_image",), 0.0, 0.0, 0.0, False)

    height, width = line_image.shape[:2]
    x0 = max(0, int(x0))
    x1 = min(width, int(x1))
    span_width = x1 - x0
    if span_width < 4:
        return LocalizationReport(False, ("span_too_narrow",), 0.0, 0.0, 0.0, False)

    width_ratio = 0.0
    if median_span_width and median_span_width > 0:
        width_ratio = span_width / float(median_span_width)
        if width_ratio < config.min_width_ratio:
            reasons.append(f"span_narrow_vs_line_{width_ratio:.2f}")
        elif width_ratio > config.max_width_ratio:
            reasons.append(f"span_wide_vs_line_{width_ratio:.2f}")

    crop = line_image[:, x0:x1]
    ink = _ink_mask(_to_grayscale(crop))
    if not ink.any():
        return LocalizationReport(
            False, ("no_ink_in_span",), width_ratio, 0.0, 0.0, False
        )

    columns = np.where(ink.any(axis=0))[0]
    left, right = int(columns[0]), int(columns[-1])
    ink_width = right - left + 1
    ink_width_ratio = ink_width / float(span_width)

    touches_left = left == 0
    touches_right = right == span_width - 1
    touches_both = touches_left and touches_right
    if config.reject_ink_on_both_edges and touches_both:
        reasons.append("ink_spans_both_edges")

    if ink_width_ratio < config.min_ink_width_ratio:
        reasons.append(f"ink_too_sparse_{ink_width_ratio:.2f}")

    column_mass = ink.sum(axis=0).astype(np.float64)
    total = column_mass.sum()
    centroid = float((column_mass * np.arange(span_width)).sum() / total)
    centre_offset_ratio = abs(centroid - (span_width - 1) / 2.0) / float(span_width)
    if centre_offset_ratio > config.max_centre_offset_ratio:
        reasons.append(f"ink_off_centre_{centre_offset_ratio:.2f}")

    return LocalizationReport(
        usable=not reasons,
        reasons=tuple(reasons),
        width_ratio=width_ratio,
        centre_offset_ratio=centre_offset_ratio,
        ink_width_ratio=ink_width_ratio,
        touches_both_edges=touches_both,
    )
