"""Pin the accent-v3 crop rule so comparisons cannot drift from it.

Stage 3D-1 compared its candidates against a baseline it called
``fixed_pad_4px``, implemented as a constant 8px window around the token
centre. accent-v3 does something different: it widens the token *span* by 4px
on each side, so its width grows with the span. Measuring against the wrong
baseline understated how wide the real v3 crop is.

These tests hold every caller to one shared definition.
"""

import math
import sys
import unittest
from pathlib import Path

VALIDATOR_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = VALIDATOR_ROOT / "scripts"
for path in (str(VALIDATOR_ROOT), str(SCRIPTS)):
    if path not in sys.path:
        sys.path.insert(0, path)

from evaluate_anchor_localization_v4 import patch_candidates  # noqa: E402
from ocr_roi_validator.ctc_geometry import (  # noqa: E402
    V3_PAD_PIXELS,
    timestep_stride,
    token_span_pixels,
    v3_crop_bounds,
)


def production_rule(start, end, stride, crop_width):
    """The accent-v3 crop, written out literally as the evaluator had it.

    Kept as an independent oracle: if the shared helper is ever changed, this
    catches it rather than both moving together.
    """
    x0 = max(0, int(math.floor((start + 0.5) * stride)) - 4)
    x1 = min(crop_width, int(math.ceil((end + 1 + 0.5) * stride)) + 4)
    return x0, x1


class V3CropParityTests(unittest.TestCase):
    FIXTURES = [
        (6, 7, 2.5, 200),
        (0, 0, 8.04, 229),
        (3, 3, 1.2, 40),
        (12, 15, 4.2, 300),
        (40, 41, 2.5, 60),      # clamps at the right edge
        (0, 1, 9.0, 50),        # clamps at the left edge
    ]

    def test_shared_helper_matches_the_production_rule(self) -> None:
        for start, end, stride, crop_width in self.FIXTURES:
            with self.subTest(start=start, end=end, stride=stride):
                self.assertEqual(
                    v3_crop_bounds(start, end, stride, crop_width),
                    production_rule(start, end, stride, crop_width),
                )

    def test_evaluator_candidate_matches_the_production_rule(self) -> None:
        """The v4 comparison's baseline must be the genuine v3 crop."""
        for start, end, stride, crop_width in self.FIXTURES:
            with self.subTest(start=start, end=end, stride=stride):
                anchor = ((start + end + 1) / 2.0) * stride
                candidate = patch_candidates(
                    anchor, ink_height=20.0, pitch=16.0, crop_w=crop_width,
                    token={"start": start, "end": end, "label": 1},
                    stride=stride,
                )["v3_span_plus_4px"]
                self.assertEqual(
                    tuple(int(v) for v in candidate),
                    production_rule(start, end, stride, crop_width),
                )

    def test_v3_width_grows_with_the_token_span(self) -> None:
        """The property the old constant-window baseline got wrong."""
        narrow = v3_crop_bounds(6, 6, 2.5, 500)
        wide = v3_crop_bounds(6, 12, 2.5, 500)
        self.assertGreater(wide[1] - wide[0], narrow[1] - narrow[0])

    def test_constant_window_is_not_the_v3_crop(self) -> None:
        """Guard against the earlier mistake being reintroduced."""
        start, end, stride, crop_width = 6, 7, 2.5, 200
        anchor = ((start + end + 1) / 2.0) * stride
        constant_window = (anchor - 4.0, anchor + 4.0)
        self.assertNotEqual(
            tuple(int(v) for v in constant_window),
            v3_crop_bounds(start, end, stride, crop_width),
        )

    def test_pad_is_applied_to_both_sides(self) -> None:
        stride = 4.0
        unclamped_start, unclamped_end = token_span_pixels(5, 6, stride)
        x0, x1 = v3_crop_bounds(5, 6, stride, 10_000)
        self.assertEqual(x0, int(math.floor(unclamped_start)) - V3_PAD_PIXELS)
        self.assertEqual(x1, int(math.ceil(unclamped_end)) + V3_PAD_PIXELS)

    def test_bounds_are_clamped_to_the_crop(self) -> None:
        x0, x1 = v3_crop_bounds(0, 0, 1.0, 5)
        self.assertGreaterEqual(x0, 0)
        self.assertLessEqual(x1, 5)


class StrideTests(unittest.TestCase):
    def test_stride_is_spacing_between_timestep_centres(self) -> None:
        # 48px-high line, padded to 458, 57 timesteps, crop 229 wide.
        stride = timestep_stride(458, 57, 229, 458)
        self.assertAlmostEqual(stride, (458 / 57) * (229 / 458), places=9)

    def test_stride_scales_with_line_height_not_glyph_width(self) -> None:
        """Two lines of the same aspect but different height: stride scales."""
        short = timestep_stride(400, 50, 100, 400)
        tall = timestep_stride(400, 50, 200, 400)
        self.assertAlmostEqual(tall / short, 2.0, places=9)

    def test_degenerate_inputs_return_zero(self) -> None:
        self.assertEqual(timestep_stride(400, 0, 100, 400), 0.0)
        self.assertEqual(timestep_stride(400, 50, 100, 0), 0.0)


if __name__ == "__main__":
    unittest.main()
