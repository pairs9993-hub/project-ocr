"""Turn a native-resolution glyph crop into the accent CNN's input tensor.

The pixel audit showed ``e`` and ``é`` stay distinguishable in the native
rectified line crop, and that the recognizer's own 48px input is that crop
upsampled -- bigger, but carrying nothing extra. So the verifier reads the
native crop, and this module defines the one transformation between it and the
network.

Two rules shape that transformation:

* **Never distort the aspect ratio.** A squeezed glyph changes exactly the
  vertical proportions the accent decision depends on.
* **Never upsample to manufacture detail.** A crop smaller than the target box
  is placed at its own scale and padded; only crops larger than the box are
  scaled down, and then uniformly.

The result is a fixed-size tensor with the glyph centred, plus a second view of
the accent band alone, so the network can look at where a diacritic would sit
without hunting for it.

This module is pure geometry. It takes pixels and returns pixels; no expected
text, no strings, no model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

__all__ = ["AccentInputConfig", "prepare_cnn_input", "prepare_views"]


@dataclass(frozen=True)
class AccentInputConfig:
    """Fixed geometry for the CNN input. Frozen alongside the model."""

    height: int = 32
    width: int = 24
    # Fraction of the glyph box treated as the accent band view.
    accent_band_fraction: float = 0.45
    accent_height: int = 16
    accent_width: int = 24
    # Downscaling interpolation. Upscaling is not performed at all.
    interpolation: str = "area"
    # Pad value in normalized units; 0.0 is mid-grey after normalization.
    pad_value: float = 0.0
    # Jitter used when checking that a verdict is stable.
    jitter_pixels: int = 2

    def as_dict(self) -> dict:
        return asdict(self)


_INTERPOLATIONS = {
    "area": cv2.INTER_AREA,
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
}


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        return image[..., :3].mean(axis=2)
    return image.astype(np.float64)


def _normalize(gray: np.ndarray) -> np.ndarray:
    """Scale to roughly [-1, 1] and make ink positive regardless of polarity.

    Product screens are usually light text on dark, but not always. Flipping
    dark-on-light here means the network never has to learn both conventions.
    """
    low, high = float(gray.min()), float(gray.max())
    if high - low < 1e-6:
        return np.zeros(gray.shape, dtype=np.float32)
    scaled = (gray - low) / (high - low)
    # Ink is the sparser polarity: text covers less area than background.
    if (scaled >= 0.5).mean() > 0.5:
        scaled = 1.0 - scaled
    return (scaled * 2.0 - 1.0).astype(np.float32)


def _fit_into(
    gray: np.ndarray, height: int, width: int, config: AccentInputConfig
) -> np.ndarray:
    """Centre ``gray`` in a height x width canvas without distorting it."""
    source_h, source_w = gray.shape[:2]
    if source_h <= 0 or source_w <= 0:
        return np.full((height, width), config.pad_value, dtype=np.float32)

    # Only ever shrink. A crop smaller than the canvas keeps its native scale,
    # because enlarging it would invent detail the sensor never captured.
    scale = min(height / source_h, width / source_w, 1.0)
    target_h = max(1, int(round(source_h * scale)))
    target_w = max(1, int(round(source_w * scale)))
    if (target_h, target_w) != (source_h, source_w):
        resized = cv2.resize(
            gray.astype(np.float32),
            (target_w, target_h),
            interpolation=_INTERPOLATIONS[config.interpolation],
        )
    else:
        resized = gray.astype(np.float32)

    canvas = np.full((height, width), config.pad_value, dtype=np.float32)
    top = (height - target_h) // 2
    left = (width - target_w) // 2
    canvas[top : top + target_h, left : left + target_w] = resized
    return canvas


def prepare_views(
    glyph: np.ndarray, config: AccentInputConfig | None = None
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (full glyph view, accent band view), or None if unusable.

    Both views are normalized before cropping, so the accent band keeps the
    contrast of the whole glyph rather than being re-stretched on its own --
    otherwise faint accent ink would be amplified into looking definite.
    """
    if glyph is None or glyph.size == 0:
        return None
    if glyph.shape[0] < 4 or glyph.shape[1] < 3:
        return None

    normalized = _normalize(_to_grayscale(glyph))

    # Restrict to the ink bounding box so padding does not dominate the view.
    ink = normalized > 0.0
    if not ink.any():
        return None
    rows = np.where(ink.any(axis=1))[0]
    columns = np.where(ink.any(axis=0))[0]
    top, bottom = int(rows[0]), int(rows[-1])
    left, right = int(columns[0]), int(columns[-1])
    box = normalized[top : bottom + 1, left : right + 1]
    if box.shape[0] < 4 or box.shape[1] < 3:
        return None

    full_view = _fit_into(box, config.height, config.width, config) if config else None
    if config is None:
        config = AccentInputConfig()
        full_view = _fit_into(box, config.height, config.width, config)

    band_height = max(1, int(round(box.shape[0] * config.accent_band_fraction)))
    accent_view = _fit_into(
        box[:band_height, :], config.accent_height, config.accent_width, config
    )
    return full_view, accent_view


def prepare_cnn_input(
    glyph: np.ndarray, config: AccentInputConfig | None = None
) -> np.ndarray | None:
    """Return a (1, 2, H, W) tensor: full glyph and accent band as channels.

    The two views have different natural sizes, so the accent band is placed
    into the top of a full-size plane and the rest padded. That keeps a single
    rectangular tensor without stretching either view.
    """
    config = config or AccentInputConfig()
    views = prepare_views(glyph, config)
    if views is None:
        return None
    full_view, accent_view = views

    accent_plane = np.full(
        (config.height, config.width), config.pad_value, dtype=np.float32
    )
    height = min(config.accent_height, config.height)
    width = min(config.accent_width, config.width)
    accent_plane[:height, :width] = accent_view[:height, :width]

    return np.stack([full_view, accent_plane])[np.newaxis, :].astype(np.float32)
