"""Measured pixel geometry of a rendered line and of one target glyph.

The v1 mining recorded nominal point size and nothing else about how large the
text actually was in pixels. Nominal size is a poor stand-in: it is measured
before padding, before any upscale, before the detector's crop, and before the
recognizer resizes everything to a fixed height. Two lines at "size 12" can
reach the recognizer at very different scales.

Everything here is measured from pixels, or computed from the renderer's own
layout of a specific glyph -- never from the decoded string, and never from the
expected text.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont

__all__ = [
    "GlyphGeometry",
    "LineGeometry",
    "measure_line_geometry",
    "measure_target_glyph",
    "INK_HEIGHT_BINS",
    "GLYPH_HEIGHT_BINS",
    "OCCUPANCY_BINS",
    "RESIZE_BINS",
    "bin_value",
]

# Bins fixed before any diagnostic is run, so a boundary can never be moved to
# make a result look cleaner.
INK_HEIGHT_BINS = ((0, 8), (8, 11), (11, 14), (14, 18), (18, 24), (24, 32),
                   (32, 10_000))
GLYPH_HEIGHT_BINS = ((0, 5), (5, 7), (7, 9), (9, 12), (12, 16), (16, 22),
                     (22, 10_000))
OCCUPANCY_BINS = ((0.0, 0.02), (0.02, 0.04), (0.04, 0.07), (0.07, 0.12),
                  (0.12, 1.01))
RESIZE_BINS = ((0.0, 1.0), (1.0, 1.6), (1.6, 2.4), (2.4, 3.5), (3.5, 1000.0))


def bin_value(value: float, bins: tuple[tuple[float, float], ...]) -> str:
    for low, high in bins:
        if low <= value < high:
            return f"[{low},{high})"
    return "out_of_range"


@dataclass(frozen=True)
class LineGeometry:
    """Pixel measurements of a line crop as the recognizer receives it."""

    crop_width: int
    crop_height: int
    ink_height: int
    ink_width: int
    ink_top: int
    ink_bottom: int
    recognizer_resize_scale: float      # target height / crop height
    horizontal_padding_ratio: float     # non-ink columns / crop width

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class GlyphGeometry:
    """Foreground extent of one specific rendered character.

    Measured by rasterizing that character alone with the same font and size,
    so it describes the drawn glyph rather than anything the recognizer said.
    """

    glyph_width: int
    glyph_height: int
    glyph_occupancy: float              # glyph area / line crop area

    def as_dict(self) -> dict:
        return asdict(self)


def _ink_mask(crop: np.ndarray) -> np.ndarray | None:
    """Foreground mask, polarity-independent."""
    if crop is None or crop.size == 0:
        return None
    grey = crop.astype(float)
    if grey.ndim == 3:
        grey = grey.mean(axis=2)
    low, high = float(grey.min()), float(grey.max())
    if high - low < 12.0:
        return None
    # Distance from the background is symmetric, whereas thresholding with >=
    # and then inverting is not: pixels sitting exactly on the midpoint would
    # be counted as ink for light-on-dark and as background for dark-on-light,
    # which made the same text measure one pixel wider in one polarity.
    midpoint = (low + high) / 2.0
    background = low if abs(grey.mean() - low) < abs(grey.mean() - high) else high
    return np.abs(grey - background) > abs(midpoint - background)


def measure_line_geometry(crop: np.ndarray,
                          recognizer_height: int = 48) -> LineGeometry | None:
    """Measure one detector line crop. Returns None if it carries no ink."""
    mask = _ink_mask(crop)
    if mask is None:
        return None
    rows = np.where(mask.any(axis=1))[0]
    columns = np.where(mask.any(axis=0))[0]
    if rows.size == 0 or columns.size == 0:
        return None
    height, width = mask.shape
    ink_width = int(columns[-1] - columns[0] + 1)
    return LineGeometry(
        crop_width=int(width),
        crop_height=int(height),
        ink_height=int(rows[-1] - rows[0] + 1),
        ink_width=ink_width,
        ink_top=int(rows[0]),
        ink_bottom=int(rows[-1]),
        recognizer_resize_scale=recognizer_height / float(height),
        horizontal_padding_ratio=1.0 - ink_width / float(width),
    )


def measure_target_glyph(character: str, font_path: str, size: int,
                         line: LineGeometry,
                         upscale: float = 1.0) -> GlyphGeometry | None:
    """Measure one glyph's inked extent at the size it was drawn.

    ``upscale`` scales the result to the page's final pixels so the occupancy
    ratio is comparable with the line crop it sits in.
    """
    try:
        font = ImageFont.truetype(font_path, size)
    except OSError:
        return None
    canvas = Image.new("L", (size * 4 + 16, size * 4 + 16), 0)
    ImageDraw.Draw(canvas).text((8, 8), character, font=font, fill=255)
    mask = np.asarray(canvas) > 40
    rows = np.where(mask.any(axis=1))[0]
    columns = np.where(mask.any(axis=0))[0]
    if rows.size == 0 or columns.size == 0:
        return None
    glyph_height = float(rows[-1] - rows[0] + 1) * upscale
    glyph_width = float(columns[-1] - columns[0] + 1) * upscale
    area = float(line.crop_width * line.crop_height) or 1.0
    return GlyphGeometry(
        glyph_width=int(round(glyph_width)),
        glyph_height=int(round(glyph_height)),
        glyph_occupancy=(glyph_width * glyph_height) / area,
    )
