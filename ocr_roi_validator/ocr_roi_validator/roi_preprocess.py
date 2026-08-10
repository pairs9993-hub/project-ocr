"""Shared ROI preprocessing used by the GUI and by offline replay tools.

The GUI decides what pixels reach the recognizer. Replay tooling must apply the
same steps in the same order or its results describe a different pipeline than
the one that runs in production. This module is the single definition of those
steps so both callers stay in sync.

Order matters and mirrors the historical GUI behaviour exactly:

1. expand the ROI rect by ``margin`` pixels, clamped to the source image
2. pad a very long/tall crop toward 2:1 (``pad_long_roi``)
3. upscale with BICUBIC if the short side is below ``min_side``
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from PIL import Image

Rect = tuple[int, int, int, int]

__all__ = ["RoiPreprocessConfig", "pad_long_roi", "crop_roi", "resize_short_side"]


@dataclass(frozen=True)
class RoiPreprocessConfig:
    """Settings that determine the final OCR input image.

    Defaults match the GUI's default control values.
    """

    margin: int = 8
    min_side: int = 160
    pad_long_roi: bool = True
    auto_upscale: bool = True

    def as_dict(self) -> dict:
        return asdict(self)


def pad_long_roi(image: Image.Image) -> Image.Image:
    """Pad an extreme aspect ratio toward 2:1 using the top-left pixel colour."""
    width, height = image.size
    if width > height * 2:
        padded = Image.new(image.mode, (width, (width + 1) // 2), image.getpixel((0, 0)))
    elif height > width * 2:
        padded = Image.new(image.mode, ((height + 1) // 2, height), image.getpixel((0, 0)))
    else:
        return image
    padded.paste(image, (0, 0))
    return padded


def resize_short_side(image: Image.Image, min_side: int) -> Image.Image:
    """Upscale with BICUBIC until the short side reaches ``min_side``."""
    width, height = image.size
    short_side = min(width, height)
    if short_side >= min_side:
        return image
    scale = float(min_side) / float(max(1, short_side))
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return image.resize((new_width, new_height), Image.Resampling.BICUBIC)


def crop_roi(image: Image.Image, rect: Rect, config: RoiPreprocessConfig) -> Image.Image:
    """Produce the exact image the recognizer receives for ``rect``.

    ``rect`` is in ``image`` coordinates. The margin is clamped to the image
    bounds, so a ROI at the edge simply gets less context -- the same behaviour
    the GUI has always had.
    """
    x1, y1, x2, y2 = rect
    width, height = image.size
    expanded = (
        max(0, x1 - config.margin),
        max(0, y1 - config.margin),
        min(width, x2 + config.margin),
        min(height, y2 + config.margin),
    )
    crop = image.crop(expanded)
    if config.pad_long_roi:
        crop = pad_long_roi(crop)
    if not config.auto_upscale:
        return crop
    return resize_short_side(crop, config.min_side)
