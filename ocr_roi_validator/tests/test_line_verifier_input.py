"""Tests for the target-query line verifier input contract.

The contract is the safety mechanism here. If the decoded text, a word, or the
expected string can reach the network, it can answer from spelling instead of
pixels and every downstream measurement becomes meaningless. These tests pin
what may and may not be passed, and the geometric properties the audits assume.
"""

import inspect
import unittest

import numpy as np

from ocr_roi_validator.line_verifier_input import (
    CHANNEL_NAMES,
    LineVerifierInputConfig,
    assert_no_text_leakage,
    build_line_input,
)


def line_image(width=160, height=24, ink_columns=((20, 30), (50, 60))):
    image = np.zeros((height, width, 3), dtype=np.uint8)
    for x0, x1 in ink_columns:
        image[6:18, x0:x1] = 240
    return image


def ctc_probabilities(timesteps=20, labels=8, target=1, start=6, end=7):
    probabilities = np.full((timesteps, labels), 0.01, dtype=np.float32)
    probabilities[:, 0] = 0.8
    probabilities[start:end + 1, 0] = 0.1
    probabilities[start:end + 1, target] = 0.85
    return probabilities


class ContractTests(unittest.TestCase):
    def test_signature_takes_no_text(self) -> None:
        parameters = inspect.signature(build_line_input).parameters
        for name in parameters:
            for banned in ("expected", "decoded_text", "word", "truth", "label_text"):
                self.assertNotIn(banned, name.lower())

    def test_query_is_position_only(self) -> None:
        """The query carries an index and a length -- never a character."""
        prepared = build_line_input(
            line_image(), ctc_probabilities(), token_label=1, token_start=6,
            token_end=7, target_ordinal=2, decoded_length=8)
        self.assertIsNotNone(prepared)
        self.assertEqual(prepared.query.shape, (2,))
        self.assertTrue(np.issubdtype(prepared.query.dtype, np.floating))

    def test_leakage_guard_rejects_strings(self) -> None:
        with self.assertRaises(ValueError):
            assert_no_text_leakage({"planes": np.zeros(3), "word": "Veuillez"})
        with self.assertRaises(ValueError):
            assert_no_text_leakage({"planes": np.zeros(3), "chars": ["V", "e"]})

    def test_leakage_guard_accepts_a_valid_input(self) -> None:
        prepared = build_line_input(
            line_image(), ctc_probabilities(), 1, 6, 7, 2, 8)
        assert_no_text_leakage(prepared)      # must not raise

    def test_channel_names_are_fixed(self) -> None:
        self.assertEqual(
            CHANNEL_NAMES, ("line_image", "ctc_position_map", "valid_width_mask"))


class ShapeTests(unittest.TestCase):
    def test_planes_match_config(self) -> None:
        config = LineVerifierInputConfig()
        prepared = build_line_input(
            line_image(), ctc_probabilities(), 1, 6, 7, 2, 8, config)
        self.assertEqual(prepared.planes.shape, (3, config.height, config.width))
        self.assertEqual(prepared.planes.dtype, np.float32)

    def test_aspect_ratio_is_preserved(self) -> None:
        config = LineVerifierInputConfig()
        prepared = build_line_input(
            line_image(width=160, height=24), ctc_probabilities(), 1, 6, 7, 2, 8,
            config)
        expected = int(round(160 * (config.height / 24)))
        self.assertAlmostEqual(prepared.scale_x * 160, min(expected, config.width),
                               delta=1.0)

    def test_valid_mask_marks_the_padding(self) -> None:
        config = LineVerifierInputConfig()
        prepared = build_line_input(
            line_image(width=40, height=24), ctc_probabilities(), 1, 6, 7, 2, 8,
            config)
        valid = prepared.planes[2]
        self.assertTrue((valid[:, 0] == 1.0).all())
        self.assertTrue((valid[:, -1] == 0.0).all())

    def test_degenerate_inputs_return_none(self) -> None:
        self.assertIsNone(build_line_input(None, ctc_probabilities(), 1, 6, 7, 2, 8))
        self.assertIsNone(build_line_input(
            np.zeros((0, 0, 3), np.uint8), ctc_probabilities(), 1, 6, 7, 2, 8))
        self.assertIsNone(build_line_input(
            line_image(), ctc_probabilities(), 1, 6, 7, target_ordinal=9,
            decoded_length=8))
        self.assertIsNone(build_line_input(
            line_image(), ctc_probabilities(), 1, 6, 7, target_ordinal=0,
            decoded_length=0))


class PositionMapTests(unittest.TestCase):
    def test_map_peaks_near_the_token(self) -> None:
        prepared = build_line_input(
            line_image(width=160), ctc_probabilities(timesteps=20, start=10,
                                                     end=11), 1, 10, 11, 2, 8)
        position = prepared.planes[1]
        peak_column = int(position[0].argmax())
        scaled_width = int(round(160 * prepared.scale_x))
        self.assertAlmostEqual(peak_column / scaled_width, 11 / 20, delta=0.12)

    def test_map_is_amplitude_normalized(self) -> None:
        """Confidence must not leak through the map's height."""
        confident = ctc_probabilities()
        unsure = ctc_probabilities()
        unsure[6:8, 1] = 0.30           # much lower confidence, same position
        first = build_line_input(line_image(), confident, 1, 6, 7, 2, 8)
        second = build_line_input(line_image(), unsure, 1, 6, 7, 2, 8)
        self.assertAlmostEqual(float(first.planes[1].max()),
                               float(second.planes[1].max()), places=6)
        np.testing.assert_allclose(first.planes[1], second.planes[1], atol=1e-6)

    def test_ordinal_changes_only_the_query(self) -> None:
        """Two ordinals over the same line share their image planes."""
        first = build_line_input(line_image(), ctc_probabilities(), 1, 6, 7, 1, 8)
        second = build_line_input(line_image(), ctc_probabilities(), 1, 6, 7, 5, 8)
        np.testing.assert_array_equal(first.planes[0], second.planes[0])
        np.testing.assert_array_equal(first.planes[2], second.planes[2])
        self.assertFalse(np.array_equal(first.query, second.query))


class PolarityTests(unittest.TestCase):
    def test_dark_and_light_backgrounds_agree(self) -> None:
        dark = line_image()
        light = 255 - dark
        first = build_line_input(dark, ctc_probabilities(), 1, 6, 7, 2, 8)
        second = build_line_input(light, ctc_probabilities(), 1, 6, 7, 2, 8)
        np.testing.assert_allclose(first.planes[0], second.planes[0], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
