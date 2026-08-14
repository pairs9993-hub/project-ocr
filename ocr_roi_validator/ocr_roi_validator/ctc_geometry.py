"""Shared CTC timestep-to-pixel geometry.

The accent-v3 crop rule lived inline in two evaluator scripts, and a third
re-implementation in the Stage 3D-1 comparison got it wrong: it used a constant
8px window around the token centre, while v3 actually widens the token *span*
by 4px on each side, so v3's width grows with the span. Comparing against that
mistaken baseline understated how wide the real v3 crop is.

Putting the rule here means every caller shares one definition and a unit test
can pin it.

Terminology: consecutive timestep centres are ``stride`` crop pixels apart,
where ``stride = padded_width / timesteps * crop_width / resized_width``. That
is a spacing between sample points, not a receptive field -- the receptive
field would require a per-layer calculation of the recognizer backbone, which
has not been done here.
"""

from __future__ import annotations

import math

__all__ = [
    "timestep_stride",
    "token_span_pixels",
    "v3_crop_bounds",
    "V3_PAD_PIXELS",
]

# accent-v3 widens the token span by this many pixels on each side.
V3_PAD_PIXELS = 4


def timestep_stride(
    padded_width: int, timesteps: int, crop_width: int, resized_width: int
) -> float:
    """Crop pixels between consecutive CTC timestep centres."""
    if timesteps <= 0 or resized_width <= 0:
        return 0.0
    return (padded_width / timesteps) * (crop_width / resized_width)


def token_span_pixels(start: int, end: int, stride: float) -> tuple[float, float]:
    """Unclamped x-range of a collapsed token, in crop pixels.

    ``end`` is inclusive, matching the collapsed-CTC representation, so the
    right edge is taken from ``end + 1``.
    """
    return (start + 0.5) * stride, (end + 1 + 0.5) * stride


def v3_crop_bounds(
    start: int, end: int, stride: float, crop_width: int,
    pad: int = V3_PAD_PIXELS,
) -> tuple[int, int]:
    """The exact crop accent-v3 takes for a collapsed token.

    Reproduces the production rule including its floor/ceil rounding and its
    clamping to the crop, so a caller cannot drift from it by accident.
    """
    x0 = max(0, int(math.floor((start + 0.5) * stride)) - pad)
    x1 = min(crop_width, int(math.ceil((end + 1 + 0.5) * stride)) + pad)
    return x0, x1
