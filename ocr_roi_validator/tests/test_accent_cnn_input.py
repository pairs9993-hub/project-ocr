"""Tests for the accent CNN input transform.

The transform is the contract between the pixel audit and the network: if it
distorts or invents detail, the audit's conclusions no longer apply to what the
model actually sees. These tests pin the properties the audit relied on.
"""

import inspect
import unittest

import numpy as np

from ocr_roi_validator.accent_cnn_input import (
    AccentInputConfig,
    prepare_cnn_input,
    prepare_views,
)


def glyph(height=20, width=14, ink_rows=(6, 19), ink_columns=(2, 12), value=240):
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[ink_rows[0]:ink_rows[1], ink_columns[0]:ink_columns[1]] = value
    return image


class ApiTests(unittest.TestCase):
    def test_no_expected_text_parameter(self) -> None:
        for function in (prepare_cnn_input, prepare_views):
            for name in inspect.signature(function).parameters:
                self.assertNotIn("expected", name.lower())
                self.assertNotIn("truth", name.lower())
                self.assertNotIn("label", name.lower())

    def test_output_shape_matches_config(self) -> None:
        config = AccentInputConfig()
        tensor = prepare_cnn_input(glyph(), config)
        self.assertIsNotNone(tensor)
        self.assertEqual(tensor.shape, (1, 2, config.height, config.width))
        self.assertEqual(tensor.dtype, np.float32)


class DegenerateInputTests(unittest.TestCase):
    def test_none_returns_none(self) -> None:
        self.assertIsNone(prepare_cnn_input(None))

    def test_empty_returns_none(self) -> None:
        self.assertIsNone(prepare_cnn_input(np.zeros((0, 0, 3), np.uint8)))

    def test_blank_returns_none(self) -> None:
        self.assertIsNone(prepare_cnn_input(np.zeros((20, 14, 3), np.uint8)))

    def test_tiny_returns_none(self) -> None:
        self.assertIsNone(prepare_cnn_input(glyph(height=3, width=2,
                                                  ink_rows=(0, 2),
                                                  ink_columns=(0, 2))))


class GeometryTests(unittest.TestCase):
    def test_small_glyph_is_not_upsampled(self) -> None:
        """A crop smaller than the canvas keeps its own scale and is padded.

        Upsampling would invent detail the sensor never captured, which the
        pixel audit explicitly refused to count as information.
        """
        config = AccentInputConfig(height=32, width=24)
        small = glyph(height=12, width=8, ink_rows=(3, 11), ink_columns=(1, 7))
        tensor = prepare_cnn_input(small, config)
        ink = tensor[0, 0] > 0.0
        rows = np.where(ink.any(axis=1))[0]
        columns = np.where(ink.any(axis=0))[0]
        # The ink box is 8x6 in the source and must stay that size on canvas.
        self.assertEqual(int(rows[-1] - rows[0] + 1), 8)
        self.assertEqual(int(columns[-1] - columns[0] + 1), 6)

    def test_uniform_image_is_refused(self) -> None:
        """An image with no contrast holds no glyph, so it must abstain."""
        uniform = np.full((40, 20, 3), 240, dtype=np.uint8)
        self.assertIsNone(prepare_cnn_input(uniform))

    def test_large_glyph_is_scaled_down_uniformly(self) -> None:
        config = AccentInputConfig(height=32, width=24)
        # Ink inset from the border, so the crop has a background to measure.
        large = glyph(height=96, width=48, ink_rows=(4, 92), ink_columns=(3, 47))
        tensor = prepare_cnn_input(large, config)
        self.assertIsNotNone(tensor)
        ink = tensor[0, 0] > 0.0
        rows = np.where(ink.any(axis=1))[0]
        columns = np.where(ink.any(axis=0))[0]
        height = int(rows[-1] - rows[0] + 1)
        width = int(columns[-1] - columns[0] + 1)
        # Source aspect is 2:1; the placed ink must preserve it.
        self.assertAlmostEqual(height / width, 2.0, delta=0.25)
        self.assertLessEqual(height, config.height)
        self.assertLessEqual(width, config.width)

    def test_aspect_ratio_is_never_distorted(self) -> None:
        config = AccentInputConfig(height=32, width=24)
        wide = glyph(height=20, width=60, ink_rows=(4, 18), ink_columns=(0, 60))
        tensor = prepare_cnn_input(wide, config)
        ink = tensor[0, 0] > 0.0
        rows = np.where(ink.any(axis=1))[0]
        columns = np.where(ink.any(axis=0))[0]
        placed = (rows[-1] - rows[0] + 1) / (columns[-1] - columns[0] + 1)
        self.assertAlmostEqual(placed, 14 / 60, delta=0.12)

    def test_glyph_is_centred(self) -> None:
        config = AccentInputConfig(height=32, width=24)
        tensor = prepare_cnn_input(glyph(), config)
        ink = tensor[0, 0] > 0.0
        columns = np.where(ink.any(axis=0))[0]
        centre = (columns[0] + columns[-1]) / 2.0
        self.assertAlmostEqual(centre, (config.width - 1) / 2.0, delta=1.5)


class NormalizationTests(unittest.TestCase):
    def test_polarity_is_unified(self) -> None:
        """Dark-on-light and light-on-dark must yield the same tensor."""
        light_on_dark = glyph(value=240)
        dark_on_light = 255 - light_on_dark
        a = prepare_cnn_input(light_on_dark)
        b = prepare_cnn_input(dark_on_light)
        np.testing.assert_allclose(a, b, atol=1e-6)

    def test_values_are_bounded(self) -> None:
        tensor = prepare_cnn_input(glyph())
        self.assertGreaterEqual(float(tensor.min()), -1.001)
        self.assertLessEqual(float(tensor.max()), 1.001)

    def test_brightness_scaling_does_not_change_output(self) -> None:
        """Normalization is per-crop, so overall exposure must not matter."""
        bright = glyph(value=250)
        dim = glyph(value=120)
        np.testing.assert_allclose(
            prepare_cnn_input(bright), prepare_cnn_input(dim), atol=1e-6
        )


class AccentBandTests(unittest.TestCase):
    def test_accent_band_view_differs_when_a_mark_is_present(self) -> None:
        config = AccentInputConfig()
        bare = glyph(height=26, ink_rows=(12, 25), ink_columns=(2, 12))
        accented = bare.copy()
        accented[2:6, 5:10] = 240
        bare_views = prepare_views(bare, config)
        accented_views = prepare_views(accented, config)
        self.assertIsNotNone(bare_views)
        self.assertIsNotNone(accented_views)
        difference = np.abs(accented_views[1] - bare_views[1]).max()
        self.assertGreater(difference, 0.5)

    def test_band_is_cropped_after_normalization(self) -> None:
        """Faint accent ink must not be re-stretched into looking definite."""
        config = AccentInputConfig()
        image = glyph(height=26, ink_rows=(12, 25), ink_columns=(2, 12))
        image[2:6, 5:10] = 90          # deliberately faint mark
        views = prepare_views(image, config)
        self.assertIsNotNone(views)
        # The faint mark must stay below the body's own intensity.
        self.assertLess(float(views[1].max()), float(views[0].max()) + 1e-6)


if __name__ == "__main__":
    unittest.main()
