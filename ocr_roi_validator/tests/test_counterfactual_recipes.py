"""Tests for counterfactual recipe arithmetic and pair isolation.

The arithmetic tests exist because "24 pairs plus 25% UNKNOWN" did not state
what the 60 renderings per context actually were. The isolation tests exist
because a previous revision cropped both pair members using geometry measured
from the accented one, which put the answer into the bare image.
"""

import sys
import unittest
from collections import Counter
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from ocr_roi_validator.counterfactual_recipes import (  # noqa: E402
    COUNTERFACTUAL_RECIPES,
    FORBIDDEN,
    MEMBERS_PER_PAIR,
    PAIRS_PER_CONTEXT,
    RENDERINGS_PER_CONTEXT,
    UNKNOWN_KINDS,
    UNKNOWN_PER_CONTEXT,
    assert_context_independence,
    build_word_contexts,
)


class ArithmeticTests(unittest.TestCase):
    def test_composition_sums_to_sixty(self) -> None:
        self.assertEqual(PAIRS_PER_CONTEXT * MEMBERS_PER_PAIR
                         + UNKNOWN_PER_CONTEXT, 60)
        self.assertEqual(RENDERINGS_PER_CONTEXT, 60)

    def test_unknown_kinds_sum_to_the_unknown_count(self) -> None:
        self.assertEqual(sum(UNKNOWN_KINDS.values()), UNKNOWN_PER_CONTEXT)

    def test_declared_totals_match_the_parts(self) -> None:
        for recipe in COUNTERFACTUAL_RECIPES.values():
            self.assertEqual(recipe.renderings,
                             recipe.pair_members + recipe.unknown_cases)
            self.assertEqual(
                recipe.renderings,
                recipe.word_context_count * recipe.renderings_per_context)

    def test_shipped_totals(self) -> None:
        train = COUNTERFACTUAL_RECIPES["line_counterfactual_train_v1"]
        calibration = COUNTERFACTUAL_RECIPES["line_counterfactual_calibration_v1"]
        self.assertEqual(train.renderings, 12_000)
        self.assertEqual(calibration.renderings, 3_000)
        self.assertEqual(train.word_context_count, 200)
        self.assertEqual(calibration.word_context_count, 50)

    def test_context_counts_meet_the_minimums(self) -> None:
        self.assertGreaterEqual(
            COUNTERFACTUAL_RECIPES["line_counterfactual_train_v1"]
            .word_context_count, 200)
        self.assertGreaterEqual(
            COUNTERFACTUAL_RECIPES["line_counterfactual_calibration_v1"]
            .word_context_count, 50)


class IndependenceTests(unittest.TestCase):
    def test_shipped_recipes_are_independent(self) -> None:
        report = assert_context_independence()
        for pair, shared in report.items():
            self.assertEqual(shared, [], f"{pair} overlaps")

    def test_no_forbidden_string_appears(self) -> None:
        for recipe in COUNTERFACTUAL_RECIPES.values():
            words = {w.lower() for pair in recipe.words for w in pair}
            self.assertEqual(words & FORBIDDEN, set())

    def test_target_string_is_absent(self) -> None:
        """The real UI text must never be rendered."""
        for recipe in COUNTERFACTUAL_RECIPES.values():
            for bare, accented in recipe.words:
                self.assertNotIn("veuillez", bare.lower())
                self.assertNotIn("veuillez", accented.lower())

    def test_every_context_has_a_bare_e(self) -> None:
        for recipe in COUNTERFACTUAL_RECIPES.values():
            for bare, _ in recipe.words:
                self.assertIn("e", bare)

    def test_accented_form_differs_only_in_accents(self) -> None:
        for recipe in COUNTERFACTUAL_RECIPES.values():
            for bare, accented in recipe.words:
                self.assertEqual(len(bare), len(accented))
                self.assertEqual(accented, bare.replace("e", "é"))

    def test_word_lists_are_deterministic(self) -> None:
        first = build_word_contexts(30, "same-salt", frozenset())
        second = build_word_contexts(30, "same-salt", frozenset())
        self.assertEqual(first, second)

    def test_different_salts_give_different_lists(self) -> None:
        first = build_word_contexts(30, "salt-a", frozenset())
        second = build_word_contexts(30, "salt-b", frozenset())
        self.assertNotEqual(first, second)

    def test_exclusions_are_honoured(self) -> None:
        base = build_word_contexts(20, "exclusion-test", frozenset())
        banned = frozenset({base[0][0].lower()})
        after = build_word_contexts(20, "exclusion-test", banned)
        self.assertNotIn(base[0][0].lower(),
                         {w.lower() for pair in after for w in pair})


class PairIsolationTests(unittest.TestCase):
    def setUp(self) -> None:
        from generate_counterfactual_v1 import Rendering
        self.Rendering = Rendering
        self.recipe = COUNTERFACTUAL_RECIPES["line_counterfactual_train_v1"]

    def test_pair_members_share_every_optical_setting(self) -> None:
        first = self.Rendering(0, self.recipe)
        second = self.Rendering(1, self.recipe)
        for attribute in ("font", "size", "upscale", "template", "pad_x",
                          "pad_y", "background", "foreground", "contrast",
                          "blur", "jitter_x", "jitter_y", "stratum"):
            self.assertEqual(getattr(first, attribute),
                             getattr(second, attribute), attribute)

    def test_pair_members_differ_only_in_the_glyph(self) -> None:
        first = self.Rendering(0, self.recipe)
        second = self.Rendering(1, self.recipe)
        self.assertEqual(first.pair_id, second.pair_id)
        self.assertNotEqual(first.word, second.word)
        self.assertEqual(first.word, second.word.replace("é", "e"))

    def test_labels_come_from_the_drawn_glyph(self) -> None:
        self.assertEqual(self.Rendering(0, self.recipe).label, "BARE_E")
        self.assertEqual(self.Rendering(1, self.recipe).label, "ACCENT_PRESENT")

    def test_context_composition_is_exactly_as_declared(self) -> None:
        labels = Counter(self.Rendering(i, self.recipe).label
                         for i in range(RENDERINGS_PER_CONTEXT))
        self.assertEqual(labels["BARE_E"], PAIRS_PER_CONTEXT)
        self.assertEqual(labels["ACCENT_PRESENT"], PAIRS_PER_CONTEXT)
        self.assertEqual(labels["UNKNOWN"], UNKNOWN_PER_CONTEXT)

    def test_unknown_kinds_are_evenly_produced(self) -> None:
        kinds = Counter(self.Rendering(i, self.recipe).unknown_kind
                        for i in range(RENDERINGS_PER_CONTEXT)
                        if self.Rendering(i, self.recipe).unknown_kind)
        self.assertEqual(dict(kinds), dict(UNKNOWN_KINDS))

    def test_unknown_cases_carry_no_pair_id(self) -> None:
        for offset in range(UNKNOWN_PER_CONTEXT):
            rendering = self.Rendering(
                PAIRS_PER_CONTEXT * MEMBERS_PER_PAIR + offset, self.recipe)
            self.assertIsNone(rendering.pair_id)
            self.assertEqual(rendering.label, "UNKNOWN")

    def test_context_boundary_is_exact(self) -> None:
        self.assertEqual(self.Rendering(59, self.recipe).context_index, 0)
        self.assertEqual(self.Rendering(60, self.recipe).context_index, 1)

    def test_rendering_is_a_function_of_index(self) -> None:
        for index in (0, 137, 5000):
            first = self.Rendering(index, self.recipe)
            second = self.Rendering(index, self.recipe)
            self.assertEqual(first.text, second.text)
            self.assertEqual(first.font, second.font)
            self.assertEqual(first.upscale, second.upscale)

    def test_every_pair_id_has_exactly_two_members(self) -> None:
        seen = Counter()
        for index in range(RENDERINGS_PER_CONTEXT * 3):
            rendering = self.Rendering(index, self.recipe)
            if rendering.pair_id:
                seen[rendering.pair_id] += 1
        self.assertEqual(set(seen.values()), {MEMBERS_PER_PAIR})


if __name__ == "__main__":
    unittest.main()
