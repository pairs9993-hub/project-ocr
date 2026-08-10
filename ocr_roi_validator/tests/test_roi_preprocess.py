"""Tests for the shared ROI preprocessing helpers.

The GUI was refactored to call this module instead of carrying its own copy of
the logic. These tests pin the behaviour to the original implementation so the
refactor cannot silently change what the recognizer sees.
"""

import unittest

from PIL import Image

from ocr_roi_validator.roi_preprocess import (
    RoiPreprocessConfig,
    crop_roi,
    pad_long_roi,
    resize_short_side,
)


def reference_crop_roi(image, rect, margin, min_side, fast_long_roi, auto_upscale):
    """The original GUI ``_crop_roi`` body, kept verbatim as a test oracle."""
    x1, y1, x2, y2 = rect
    img_w, img_h = image.size
    ex1 = max(0, x1 - margin)
    ey1 = max(0, y1 - margin)
    ex2 = min(img_w, x2 + margin)
    ey2 = min(img_h, y2 + margin)

    crop = image.crop((ex1, ey1, ex2, ey2))
    if fast_long_roi:
        crop = reference_pad_long_roi(crop)
    if not auto_upscale:
        return crop

    c_w, c_h = crop.size
    short_side = min(c_w, c_h)
    if short_side >= min_side:
        return crop

    scale = float(min_side) / float(max(1, short_side))
    new_w = max(1, int(round(c_w * scale)))
    new_h = max(1, int(round(c_h * scale)))
    return crop.resize((new_w, new_h), Image.Resampling.BICUBIC)


def reference_pad_long_roi(image):
    """The original GUI ``_pad_long_roi`` body, kept verbatim as a test oracle."""
    width, height = image.size
    if width > height * 2:
        padded = Image.new(image.mode, (width, (width + 1) // 2), image.getpixel((0, 0)))
    elif height > width * 2:
        padded = Image.new(image.mode, ((height + 1) // 2, height), image.getpixel((0, 0)))
    else:
        return image
    padded.paste(image, (0, 0))
    return padded


def make_image(width: int, height: int) -> Image.Image:
    image = Image.new("RGB", (width, height), (12, 34, 56))
    for x in range(0, width, 7):
        for y in range(0, height, 5):
            image.putpixel((x, y), ((x * 3) % 256, (y * 5) % 256, (x + y) % 256))
    return image


class PadLongRoiTests(unittest.TestCase):
    def test_wide_image_padded_to_half_height(self) -> None:
        padded = pad_long_roi(make_image(100, 10))
        self.assertEqual(padded.size, (100, 50))

    def test_tall_image_padded_to_half_width(self) -> None:
        padded = pad_long_roi(make_image(10, 100))
        self.assertEqual(padded.size, (50, 100))

    def test_normal_aspect_untouched(self) -> None:
        image = make_image(100, 80)
        self.assertIs(pad_long_roi(image), image)


class ResizeShortSideTests(unittest.TestCase):
    def test_upscales_below_threshold(self) -> None:
        resized = resize_short_side(make_image(200, 50), 160)
        self.assertEqual(min(resized.size), 160)

    def test_leaves_large_image_alone(self) -> None:
        image = make_image(400, 300)
        self.assertIs(resize_short_side(image, 160), image)


class GuiParityTests(unittest.TestCase):
    """crop_roi must match the original GUI implementation pixel for pixel."""

    def test_matches_reference_across_configurations(self) -> None:
        source = make_image(400, 300)
        rects = [
            (50, 40, 300, 160),
            (0, 0, 40, 20),          # clamped at the top-left edge
            (380, 290, 400, 300),    # clamped at the bottom-right edge
            (10, 10, 330, 30),       # very wide, triggers padding
            (10, 10, 30, 280),       # very tall, triggers padding
        ]
        configs = [
            RoiPreprocessConfig(),
            RoiPreprocessConfig(margin=0),
            RoiPreprocessConfig(margin=24, min_side=320),
            RoiPreprocessConfig(pad_long_roi=False),
            RoiPreprocessConfig(auto_upscale=False),
            RoiPreprocessConfig(margin=0, pad_long_roi=False, auto_upscale=False),
        ]
        for rect in rects:
            for config in configs:
                with self.subTest(rect=rect, config=config):
                    actual = crop_roi(source, rect, config)
                    expected = reference_crop_roi(
                        source,
                        rect,
                        config.margin,
                        config.min_side,
                        config.pad_long_roi,
                        config.auto_upscale,
                    )
                    self.assertEqual(actual.size, expected.size)
                    self.assertEqual(actual.tobytes(), expected.tobytes())


class GuiIntegrationTests(unittest.TestCase):
    def test_gui_pad_long_roi_delegates_to_shared_helper(self) -> None:
        from ocr_roi_validator.gui import _pad_long_roi

        image = make_image(100, 10)
        self.assertEqual(
            _pad_long_roi(image).tobytes(), reference_pad_long_roi(image).tobytes()
        )


if __name__ == "__main__":
    unittest.main()
