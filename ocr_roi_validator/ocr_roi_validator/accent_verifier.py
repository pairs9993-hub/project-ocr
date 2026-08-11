"""Image-based e / é / unknown verification for a single glyph.

The text router cannot tell a hallucinated accent from a real one: both look
like the same string difference. This module answers the question from pixels
instead, by asking whether the glyph actually rises into the accent band.

Method
------
Measuring ink inside the glyph box alone is not enough: at UI resolution the
acute accent touches the letter body, so there is no blank separator row to
find, and an ``e`` and an ``é`` have similar ink density.

What does separate them is *where the glyph sits within its own text line*. A
lowercase ``e`` occupies the x-height band and its top edge sits well below the
line's ascender height. An ``é`` carries a mark that reaches up into the
ascender band. So the discriminant is the glyph's top edge measured against the
line's own ink extent -- a ratio, and therefore independent of font size,
resolution and the crop's exact bounds.

Scope and safety
----------------
* Only ``é`` -> ``e`` is ever proposed. The reverse is impossible by
  construction: this module is only ever asked about a predicted ``é``, and
  never reports ``é`` for a glyph read as ``e``.
* Anything ambiguous returns :data:`UNKNOWN` and the caller keeps the baseline.
  Refusing to answer is safe; guessing is not.
* i/l, digits, timers and punctuation are out of scope.
* No expected text, reference string or dictionary is an input. The verdict is
  a function of pixels alone.

Thresholds come from Latin script geometry -- an accent occupies the band above
the x-height -- and are not fitted to any captured UI image.

.. warning::

   **This prototype fails its synthetic holdout and must not be used at
   runtime.**

   On the exact target ROI it is correct on every perturbation, but that is
   partly luck: the reference band is the *line's* ink extent, so any capital
   or ascender elsewhere on the line (``É``, ``l``, ``d``, ``t``) raises the
   top of that band and pushes a genuine ``é`` down into the "bare e" range.
   Across a 4-font holdout it erased 53 of 104 real accents. A line-relative
   measurement cannot separate the two classes; a usable verifier needs a
   per-glyph baseline and x-height estimate, or a small classifier trained on
   synthetic glyphs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "ACCENT_PRESENT",
    "ACCENT_ABSENT",
    "UNKNOWN",
    "AccentVerdict",
    "verify_accent_glyph",
]

ACCENT_PRESENT = "é"
ACCENT_ABSENT = "e"
UNKNOWN = "unknown"

# Where the glyph's top edge sits inside the line's ink band, as a fraction of
# that band's height. An accented glyph reaches close to the top; a bare `e`
# starts around the x-height, roughly a third of the way down.
ACCENT_TOP_RATIO = 0.18
BARE_TOP_RATIO = 0.28

# Guard rails: below these sizes the measurement is not meaningful.
MIN_GLYPH_WIDTH = 4
MIN_LINE_INK_HEIGHT = 10


@dataclass(frozen=True)
class AccentVerdict:
    """Outcome of inspecting one glyph."""

    verdict: str
    top_ratio: float
    reason: str

    @property
    def is_accent_absent(self) -> bool:
        """True only for a confident `e` -- the sole case that may change text."""
        return self.verdict == ACCENT_ABSENT


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        # Luma; channel order is irrelevant for a presence test.
        return image[..., :3].mean(axis=2)
    return image.astype(np.float64)


def _ink_mask(gray: np.ndarray) -> np.ndarray:
    """Boolean ink mask, handling light-on-dark and dark-on-light alike."""
    low, high = float(gray.min()), float(gray.max())
    if high - low < 12.0:
        return np.zeros(gray.shape, dtype=bool)
    midpoint = (low + high) / 2.0
    bright = gray >= midpoint
    # Ink is the sparser polarity: text covers less area than background.
    return bright if bright.mean() < 0.5 else ~bright


def verify_accent_glyph(
    line_image: np.ndarray,
    x0: int,
    x1: int,
) -> AccentVerdict:
    """Decide whether the glyph at ``[x0:x1]`` of ``line_image`` has an accent.

    The whole line is required, not just the glyph crop: the verdict depends on
    where the glyph sits relative to the line's own ink band.

    There is deliberately no parameter for the expected character.
    """
    if line_image is None or line_image.size == 0:
        return AccentVerdict(UNKNOWN, 0.0, "empty_line_image")

    height, width = line_image.shape[:2]
    x0 = max(0, int(x0))
    x1 = min(width, int(x1))
    if x1 - x0 < MIN_GLYPH_WIDTH:
        return AccentVerdict(UNKNOWN, 0.0, f"glyph_too_narrow_{x1 - x0}")

    ink = _ink_mask(_to_grayscale(line_image))
    if not ink.any():
        return AccentVerdict(UNKNOWN, 0.0, "no_ink_in_line")

    line_rows = np.where(ink.any(axis=1))[0]
    line_top, line_bottom = int(line_rows[0]), int(line_rows[-1])
    line_ink_height = line_bottom - line_top + 1
    if line_ink_height < MIN_LINE_INK_HEIGHT:
        return AccentVerdict(UNKNOWN, 0.0, f"line_ink_height_{line_ink_height}")

    glyph = ink[:, x0:x1]
    if not glyph.any():
        return AccentVerdict(UNKNOWN, 0.0, "no_ink_in_glyph")

    # CTC spans are approximate and can clip a sliver of the neighbouring
    # character. A stray column or two of that neighbour would raise the top
    # edge and fake an accent, so ignore rows holding only a trace of ink.
    row_counts = glyph.sum(axis=1)
    glyph_width = x1 - x0
    min_row_ink = max(2, int(np.ceil(glyph_width * 0.20)))
    substantial = np.where(row_counts >= min_row_ink)[0]
    if substantial.size == 0:
        return AccentVerdict(UNKNOWN, 0.0, "glyph_ink_too_sparse")

    glyph_top = int(substantial[0])
    glyph_bottom = int(substantial[-1])

    # A glyph whose body sits above the line's baseline band is not the
    # lowercase letter we were asked about.
    if glyph_bottom < line_top + line_ink_height * 0.4:
        return AccentVerdict(UNKNOWN, 0.0, "glyph_does_not_reach_baseline_band")

    top_ratio = (glyph_top - line_top) / float(line_ink_height)

    if top_ratio <= ACCENT_TOP_RATIO:
        return AccentVerdict(ACCENT_PRESENT, top_ratio, "glyph_reaches_accent_band")
    if top_ratio >= BARE_TOP_RATIO:
        return AccentVerdict(ACCENT_ABSENT, top_ratio, "glyph_starts_at_x_height")
    return AccentVerdict(UNKNOWN, top_ratio, "ambiguous_glyph_top")
