"""Tests for measured pixel geometry.

The point of these measurements is to replace nominal point size, which proved
useless as a stand-in for how large the text actually reaches the recognizer.
So the tests check that the measurements track real pixels: that they follow an
upscale, that they are blind to background polarity, and that the bins are
fixed rather than adjustable after the fact.
"""

import unittest

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ocr_roi_validator.glyph_geometry import (
    GLYPH_HEIGHT_BINS,
    INK_HEIGHT_BINS,
    OCCUPANCY_BINS,
    RESIZE_BINS,
    bin_value,
    measure_line_geometry,
    measure_target_glyph,
)

FONT = "C:/Windows/Fonts/arial.ttf"


def crop(text="Il reglage", size=16, dark=True, width=220, height=44):
    background, foreground = ((20, 20, 20), (240, 240, 240)) if dark else \
        ((240, 240, 240), (20, 20, 20))
    image = Image.new("RGB", (width, height), background)
    ImageDraw.Draw(image).text((10, 8), text,
                               font=ImageFont.truetype(FONT, size),
                               fill=foreground)
    return np.asarray(image)[:, :, ::-1].copy()


class LineGeometryTests(unittest.TestCase):
    def test_measures_ink_not_canvas(self) -> None:
        geometry = measure_line_geometry(crop(size=16))
        self.assertIsNotNone(geometry)
        self.assertLess(geometry.ink_height, geometry.crop_height)
        self.assertGreater(geometry.ink_height, 4)

    def test_polarity_does_not_change_measurements(self) -> None:
        dark = measure_line_geometry(crop(dark=True))
        light = measure_line_geometry(crop(dark=False))
        self.assertEqual(dark.ink_height, light.ink_height)
        self.assertEqual(dark.ink_width, light.ink_width)

    def test_ink_height_tracks_point_size(self) -> None:
        small = measure_line_geometry(crop(size=12))
        large = measure_line_geometry(crop(size=24))
        self.assertGreater(large.ink_height, small.ink_height)

    def test_resize_scale_is_relative_to_crop_height(self) -> None:
        geometry = measure_line_geometry(crop(height=24), recognizer_height=48)
        self.assertAlmostEqual(geometry.recognizer_resize_scale, 2.0, places=6)

    def test_padding_ratio_grows_with_empty_canvas(self) -> None:
        narrow = measure_line_geometry(crop(width=150))
        wide = measure_line_geometry(crop(width=400))
        self.assertGreater(wide.horizontal_padding_ratio,
                           narrow.horizontal_padding_ratio)

    def test_blank_and_degenerate_crops_return_none(self) -> None:
        self.assertIsNone(measure_line_geometry(None))
        self.assertIsNone(measure_line_geometry(np.zeros((0, 0, 3), np.uint8)))
        self.assertIsNone(measure_line_geometry(
            np.full((30, 90, 3), 18, np.uint8)))       # uniform, no ink


class GlyphGeometryTests(unittest.TestCase):
    def test_accented_glyph_is_taller_than_bare(self) -> None:
        line = measure_line_geometry(crop())
        bare = measure_target_glyph("e", FONT, 20, line)
        accented = measure_target_glyph("é", FONT, 20, line)
        self.assertGreater(accented.glyph_height, bare.glyph_height)

    def test_glyph_scales_with_upscale(self) -> None:
        line = measure_line_geometry(crop())
        plain = measure_target_glyph("e", FONT, 16, line, upscale=1.0)
        scaled = measure_target_glyph("e", FONT, 16, line, upscale=2.0)
        self.assertAlmostEqual(scaled.glyph_height / plain.glyph_height, 2.0,
                               delta=0.15)

    def test_occupancy_falls_as_the_crop_grows(self) -> None:
        small = measure_line_geometry(crop(width=150, height=40))
        large = measure_line_geometry(crop(width=600, height=80))
        self.assertGreater(measure_target_glyph("e", FONT, 16, small).glyph_occupancy,
                           measure_target_glyph("e", FONT, 16, large).glyph_occupancy)

    def test_unknown_font_returns_none(self) -> None:
        line = measure_line_geometry(crop())
        self.assertIsNone(measure_target_glyph("e", "C:/nope/missing.ttf", 16, line))


class BinTests(unittest.TestCase):
    def test_bins_are_contiguous_and_ordered(self) -> None:
        for bins in (INK_HEIGHT_BINS, GLYPH_HEIGHT_BINS, OCCUPANCY_BINS,
                     RESIZE_BINS):
            for (_, previous_high), (next_low, _) in zip(bins, bins[1:]):
                self.assertEqual(previous_high, next_low)

    def test_bin_value_is_half_open(self) -> None:
        self.assertEqual(bin_value(8, INK_HEIGHT_BINS), "[8,11)")
        self.assertEqual(bin_value(10.999, INK_HEIGHT_BINS), "[8,11)")
        self.assertEqual(bin_value(11, INK_HEIGHT_BINS), "[11,14)")

    def test_out_of_range_is_named_not_silently_clamped(self) -> None:
        self.assertEqual(bin_value(-1, INK_HEIGHT_BINS), "out_of_range")


if __name__ == "__main__":
    unittest.main()
