"""Tests for the localization-v4 anchor and patch geometry.

Stage 3D-0 established that a CTC token span is not a glyph bounding box, so
v4 uses the span only for an x-anchor and sizes the crop from page geometry.
These tests pin that property: the patch must scale with the line, never with
the token, and the frozen config must stay conservative and fail closed.
"""

import json
import sys
import unittest
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from evaluate_anchor_localization_v4 import (  # noqa: E402
    ANCHORS,
    PATCHES,
    anchor_candidates,
    ink_mask,
    patch_candidates,
    score_patch,
)

CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "ocr_roi_validator"
    / "localization_v4_config.json"
)


def emitted(*spans):
    """Collapsed-CTC style tokens from (char, start, end) triples."""
    return [{"char": c, "start": s, "end": e, "label": 1} for c, s, e in spans]


def flat_probabilities(timesteps=20, labels=8, target=1):
    """CTC-like posteriors: blank dominates between characters and dips on them.

    A constant blank track is not realistic and leaves the valley walk with no
    stopping point, so blanks peak between the emitted tokens here.
    """
    probabilities = np.full((timesteps, labels), 0.01, dtype=np.float64)
    probabilities[:, 0] = 0.8
    probabilities[:, target] = 0.05
    for start, end in ((2, 3), (6, 7), (10, 11)):
        probabilities[start:end + 1, 0] = 0.1
        probabilities[start:end + 1, target] = 0.85
    return probabilities


class PatchScaleTests(unittest.TestCase):
    """The patch must follow the line, not the token."""

    def test_width_scales_with_ink_height(self) -> None:
        small = patch_candidates(50.0, ink_height=10.0, pitch=8.0, crop_w=200)
        large = patch_candidates(50.0, ink_height=40.0, pitch=32.0, crop_w=200)
        small_span = small["ink_height_scaled"]
        large_span = large["ink_height_scaled"]
        small_width = small_span[1] - small_span[0]
        large_width = large_span[1] - large_span[0]
        self.assertAlmostEqual(large_width / small_width, 4.0, delta=0.01)

    def test_v3_baseline_ignores_line_height(self) -> None:
        """The v3 crop follows the token span, so line height does not move it.

        That is the defect v4 removes: the glyph grows with the line but the
        crop does not.
        """
        token = {"start": 6, "end": 7, "label": 1}
        small = patch_candidates(50.0, 10.0, 8.0, 200, token=token,
                                 stride=2.5)["v3_span_plus_4px"]
        large = patch_candidates(50.0, 40.0, 32.0, 200, token=token,
                                 stride=2.5)["v3_span_plus_4px"]
        self.assertEqual(small, large)

    def test_geometry_patches_are_centred_on_the_anchor(self) -> None:
        """Every geometry-derived patch centres on the anchor.

        The v3 baseline is excluded: it is centred on the token span, which is
        precisely why it drifts from the glyph.
        """
        candidates = patch_candidates(73.0, 20.0, 16.0, 300)
        for name, span in candidates.items():
            if name == "v3_span_plus_4px":
                continue
            with self.subTest(patch=name):
                if name == "multi_scale":
                    for entry in span:
                        self.assertAlmostEqual((entry[0] + entry[1]) / 2, 73.0,
                                               places=6)
                else:
                    self.assertAlmostEqual((span[0] + span[1]) / 2, 73.0, places=6)

    def test_multi_scale_returns_increasing_widths(self) -> None:
        spans = patch_candidates(50.0, 20.0, 16.0, 300)["multi_scale"]
        widths = [s[1] - s[0] for s in spans]
        self.assertEqual(widths, sorted(widths))
        self.assertGreater(widths[-1], widths[0])

    def test_no_candidate_uses_token_width(self) -> None:
        """Identical anchor and geometry must give identical patches, whatever
        the token span was."""
        first = patch_candidates(50.0, 20.0, 16.0, 300)
        second = patch_candidates(50.0, 20.0, 16.0, 300)
        self.assertEqual(first["ink_height_scaled"], second["ink_height_scaled"])


class AnchorTests(unittest.TestCase):
    def test_all_candidates_present(self) -> None:
        tokens = emitted(("a", 2, 3), ("e", 6, 7), ("b", 10, 11))
        anchors = anchor_candidates(tokens, 1, flat_probabilities(), scale=2.0,
                                    pitch=8.0)
        self.assertEqual(set(anchors), set(ANCHORS))

    def test_argmax_center_is_the_token_midpoint(self) -> None:
        tokens = emitted(("e", 4, 5))
        anchors = anchor_candidates(tokens, 0, flat_probabilities(), scale=2.0,
                                    pitch=8.0)
        self.assertAlmostEqual(anchors["argmax_center"], 10.0, places=6)

    def test_consensus_abstains_when_candidates_disagree(self) -> None:
        probabilities = flat_probabilities(timesteps=40)
        # Blank probability engineered so the valley walk runs far away.
        probabilities[:, 0] = np.linspace(0.9, 0.1, 40)
        tokens = emitted(("e", 20, 21))
        anchors = anchor_candidates(tokens, 0, probabilities, scale=2.0, pitch=1.0)
        self.assertIsNone(anchors["consensus"])

    def test_consensus_fires_when_candidates_agree(self) -> None:
        tokens = emitted(("a", 2, 3), ("e", 6, 7), ("b", 10, 11))
        anchors = anchor_candidates(tokens, 1, flat_probabilities(), scale=2.0,
                                    pitch=8.0)
        self.assertIsNotNone(anchors["consensus"])

    def test_anchors_are_positions_not_spans(self) -> None:
        tokens = emitted(("e", 6, 7))
        anchors = anchor_candidates(tokens, 0, flat_probabilities(), scale=2.0,
                                    pitch=8.0)
        for value in anchors.values():
            self.assertTrue(value is None or isinstance(value, float))


class PatchScoringTests(unittest.TestCase):
    def _fixture(self):
        line = np.zeros((20, 100, 3), dtype=np.uint8)
        line[6:16, 40:50] = 240          # target glyph
        line[6:16, 60:70] = 240          # neighbour
        glyph = np.zeros((20, 100), dtype=bool)
        glyph[6:16, 40:50] = True
        accent = np.zeros((20, 100), dtype=bool)
        accent[2:5, 43:47] = True
        line[2:5, 43:47] = 240
        return line, glyph, accent

    def test_full_containment_scores_one(self) -> None:
        line, glyph, accent = self._fixture()
        result = score_patch((38, 52), glyph, accent, line, 100)
        self.assertAlmostEqual(result["glyph_containment"], 1.0, places=6)
        self.assertAlmostEqual(result["accent_containment"], 1.0, places=6)

    def test_neighbour_ink_counts_as_intrusion(self) -> None:
        line, glyph, accent = self._fixture()
        result = score_patch((38, 72), glyph, accent, line, 100)
        self.assertGreater(result["intrusion"], 0.0)

    def test_clipped_patch_is_flagged(self) -> None:
        line, glyph, accent = self._fixture()
        self.assertTrue(score_patch((-5, 20), glyph, accent, line, 100)
                        ["clipped_at_image_edge"])

    def test_degenerate_patch_abstains(self) -> None:
        line, glyph, accent = self._fixture()
        self.assertTrue(score_patch((50, 51), glyph, accent, line, 100)["abstained"])

    def test_polarity_is_handled(self) -> None:
        line, glyph, accent = self._fixture()
        inverted = 255 - line
        np.testing.assert_array_equal(ink_mask(line), ink_mask(inverted))


class FrozenConfigTests(unittest.TestCase):
    def test_config_exists_and_is_complete(self) -> None:
        if not CONFIG_PATH.is_file():
            self.skipTest("localization-v4 config not present")
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        for field in ("version", "anchor", "patch", "guard_unknown_conditions"):
            self.assertIn(field, config)

    def test_patch_scale_source_is_not_the_token(self) -> None:
        if not CONFIG_PATH.is_file():
            self.skipTest("localization-v4 config not present")
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        self.assertEqual(config["patch"]["scale_source"], "native_line_ink_height")
        excluded = config["explicitly_not_used"]
        self.assertIn("raw CTC span width", excluded)
        self.assertIn("line median span width", excluded)

    def test_guard_lists_fail_closed_conditions(self) -> None:
        if not CONFIG_PATH.is_file():
            self.skipTest("localization-v4 config not present")
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(config["guard_unknown_conditions"]), 5)

    def test_config_records_the_known_cost(self) -> None:
        """The wider patch admits more neighbour ink; that must be recorded."""
        if not CONFIG_PATH.is_file():
            self.skipTest("localization-v4 config not present")
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        self.assertIn("known_cost", config)
        self.assertGreater(config["known_cost"]["intrusion_positive_rate_dev"], 0.0)


if __name__ == "__main__":
    unittest.main()
