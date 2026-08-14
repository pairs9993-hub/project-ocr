"""Two strictly separated views of a rendered line.

An earlier revision cropped both members of a counterfactual pair using the
geometry measured from the *accented* member, so the two would line up. That is
label leakage: the accent raises the line's ink extent in roughly a third of
renderings, so a bare image cropped that way carries a trace of its
counterpart's accent -- information that does not exist at runtime, where only
one image is ever available.

The fix is to stop conflating two different jobs:

``runtime_view``
    What the model sees. Built from one image alone, using only measurements
    obtainable at inference time. It cannot see a counterpart, and its
    signature has no parameter for one.

``causal_audit_view``
    Used only to check that a pair differs solely at the target accent. Its
    geometry comes from font metrics and the template canvas -- quantities
    fixed before the e/é choice is made -- so it is identical for both members
    by construction rather than by copying one member's measurements.

The audit view must never reach training. :func:`assert_runtime_view` enforces
that at the loader.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont

__all__ = [
    "LineGeometryConfig",
    "LineView",
    "runtime_view",
    "causal_audit_view",
    "label_neutral_bounds",
    "assert_runtime_view",
]

RUNTIME = "runtime_view"
CAUSAL_AUDIT = "causal_audit_view"


@dataclass(frozen=True)
class LineGeometryConfig:
    """Geometry settings, frozen with the dataset recipe."""

    # Vertical margin around the measured band, as a fraction of its height.
    margin_ratio: float = 0.17
    minimum_margin: int = 2
    minimum_height: int = 6
    minimum_width: int = 12

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class LineView:
    """A cropped line plus the provenance of the crop."""

    image: np.ndarray            # BGR, as the recognizer receives it
    view: str                    # RUNTIME or CAUSAL_AUDIT
    top: int
    bottom: int
    ink_height: int


def _ink_rows(image: Image.Image) -> tuple[int, int] | None:
    gray = np.asarray(image.convert("L")).astype(float)
    low, high = gray.min(), gray.max()
    if high - low < 12:
        return None
    mask = gray >= (low + high) / 2.0
    if mask.mean() > 0.5:
        mask = ~mask
    rows = np.where(mask.any(axis=1))[0]
    return (int(rows[0]), int(rows[-1])) if rows.size else None


def label_neutral_bounds(
    text: str, font_path: str, size: int, origin_y: float, canvas_height: int,
) -> tuple[int, int]:
    """Vertical band from font metrics, independent of which e-form is drawn.

    The ascent/descent of a font are properties of the face and size, not of
    the glyphs chosen, so both members of a pair get the same band whichever
    carries the accent. Nothing here rasterizes the text.
    """
    font = ImageFont.truetype(font_path, size)
    ascent, descent = font.getmetrics()
    top = int(np.floor(origin_y))
    bottom = int(np.ceil(origin_y + ascent + descent))
    return max(0, top), min(canvas_height, max(top + 1, bottom))


def runtime_view(
    page: Image.Image, config: LineGeometryConfig | None = None,
) -> LineView | None:
    """Crop one rendered page the way the runtime would, from that page alone.

    There is deliberately no parameter for a counterpart image or for
    externally supplied bounds: at inference only this image exists, so
    anything derived elsewhere would not be reproducible.
    """
    config = config or LineGeometryConfig()
    extent = _ink_rows(page)
    if extent is None:
        return None
    ink_height = extent[1] - extent[0] + 1
    margin = max(config.minimum_margin, int(ink_height * config.margin_ratio))
    top = max(0, extent[0] - margin)
    bottom = min(page.height, extent[1] + margin + 1)
    image = np.asarray(page)[top:bottom, :, ::-1].copy()
    if (image.shape[0] < config.minimum_height
            or image.shape[1] < config.minimum_width):
        return None
    return LineView(image=image, view=RUNTIME, top=top, bottom=bottom,
                    ink_height=ink_height)


def causal_audit_view(
    page: Image.Image, text: str, font_path: str, size: int, origin_y: float,
    config: LineGeometryConfig | None = None,
) -> LineView | None:
    """Crop for pair auditing, using geometry fixed before the e/é choice.

    Both members of a pair receive identical bounds because the bounds come
    from font metrics and the canvas, never from either member's pixels. Not
    for training -- :func:`assert_runtime_view` rejects it.
    """
    config = config or LineGeometryConfig()
    top, bottom = label_neutral_bounds(text, font_path, size, origin_y,
                                       page.height)
    image = np.asarray(page)[top:bottom, :, ::-1].copy()
    if (image.shape[0] < config.minimum_height
            or image.shape[1] < config.minimum_width):
        return None
    return LineView(image=image, view=CAUSAL_AUDIT, top=top, bottom=bottom,
                    ink_height=bottom - top)


def assert_runtime_view(view: LineView) -> LineView:
    """Raise if an audit-only view is about to be used for training."""
    if view.view != RUNTIME:
        raise ValueError(
            f"{view.view} must not be used as model input; its geometry is "
            "label-neutral by construction and is not reproducible at runtime"
        )
    return view
