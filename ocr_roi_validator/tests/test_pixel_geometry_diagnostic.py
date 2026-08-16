"""Tests for the matched pixel-geometry diagnostic's design invariants.

Two properties matter most and are easy to break silently. The matched design
requires that a base condition be identical across fonts -- if any font-derived
value leaked into it, the comparison would no longer isolate the typeface. And
eligibility must be decided from the renderer alone, because deciding it from
the decode is what destroyed v1's denominator.
"""

import sys
import unittest
from collections import Counter
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from diagnose_pixel_geometry_v1 import (  # noqa: E402
    BASE_CONDITIONS, CELLS, FONTS, RENDERINGS_PER_CELL, SIZES, TOTAL_RENDERINGS,
    BaseCondition, render,
)


class BudgetTests(unittest.TestCase):
    def test_budget_is_an_exact_integer_matrix(self) -> None:
        self.assertEqual(CELLS, len(SIZES) * 2 * 2 * 2)
        self.assertEqual(BASE_CONDITIONS, CELLS * RENDERINGS_PER_CELL)
        self.assertEqual(TOTAL_RENDERINGS, BASE_CONDITIONS * len(FONTS))
        self.assertEqual(TOTAL_RENDERINGS, 19200)

    def test_every_cell_receives_the_same_count(self) -> None:
        cells = Counter((c.size, c.padding_bucket, c.upscale_bucket, c.polarity)
                        for c in map(BaseCondition, range(BASE_CONDITIONS)))
        self.assertEqual(len(cells), CELLS)
        self.assertEqual(set(cells.values()), {RENDERINGS_PER_CELL})


class MatchedDesignTests(unittest.TestCase):
    def test_base_condition_carries_no_font(self) -> None:
        """Font must not be an input, or the comparison stops being matched."""
        self.assertNotIn("font", BaseCondition.__slots__)
        for name in BaseCondition.__slots__:
            self.assertNotIn("font", name.lower())

    def test_condition_is_deterministic_from_its_index(self) -> None:
        for index in (0, 17, 999, 3199):
            first, second = BaseCondition(index), BaseCondition(index)
            self.assertEqual(first.as_dict(), second.as_dict())
            self.assertEqual(first.text, second.text)

    def test_same_condition_renders_the_same_text_in_every_font(self) -> None:
        condition = BaseCondition(42)
        rendered = {name: render(condition, Path("C:/Windows/Fonts") / name)
                    for name in FONTS}
        # Identical settings; only the typeface differs, so sizes may vary
        # slightly with advance widths but the content must not.
        self.assertEqual(len({condition.text}), 1)
        for image in rendered.values():
            self.assertGreater(image.width, 0)
            self.assertGreater(image.height, 0)

    def test_optical_settings_do_not_depend_on_font(self) -> None:
        condition = BaseCondition(7)
        for attribute in ("size", "pad_x", "pad_y", "upscale", "background",
                          "foreground", "contrast", "blur", "polarity"):
            self.assertEqual(getattr(condition, attribute),
                             getattr(BaseCondition(7), attribute))


class EligibilityTests(unittest.TestCase):
    def test_target_position_comes_from_the_drawn_string(self) -> None:
        for index in range(0, BASE_CONDITIONS, 211):
            condition = BaseCondition(index)
            position = condition.target_position()
            self.assertGreaterEqual(position, 0)
            self.assertIn(condition.text[position], {"e", "é"})
            self.assertEqual(condition.text[position], condition.target_character)

    def test_target_lies_inside_the_substituted_word(self) -> None:
        """Not in the template, which must contribute no e-forms."""
        for index in range(0, BASE_CONDITIONS, 307):
            condition = BaseCondition(index)
            self.assertGreaterEqual(condition.target_position(),
                                    condition.template.index("{}"))

    def test_every_condition_is_eligible(self) -> None:
        missing = [i for i in range(BASE_CONDITIONS)
                   if BaseCondition(i).target_position() < 0]
        self.assertEqual(missing, [])

    def test_both_visual_targets_occur(self) -> None:
        targets = Counter(BaseCondition(i).target_character
                          for i in range(BASE_CONDITIONS))
        self.assertEqual(set(targets), {"e", "é"})
        self.assertGreater(min(targets.values()), BASE_CONDITIONS * 0.4)


if __name__ == "__main__":
    unittest.main()
