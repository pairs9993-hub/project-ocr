"""Pre-launch gate for the v2 generator.

Each test here corresponds to a way a long generation has already gone wrong in
this work, or could: cohorts overlapping between splits, a resume quietly
continuing a different experiment, two writers on one checkpoint, a run that
stops when results look sufficient, or one that reports success while a quota
is short.
"""

import json
import unittest
from collections import Counter
from pathlib import Path

from ocr_roi_validator.v2_recipes import (
    MACRO_STRATA,
    STRATUM_TARGETS,
    V2_RECIPES,
    SplitRecipe,
    assert_cohort_independence,
)


def counts(hallucination=0, preservation=0, unknown=0, by_stratum=None,
           preservation_by_stratum=None, preservation_by_font=None):
    return {
        "hallucination_total": hallucination,
        "preservation_total": preservation,
        "unknown_total": unknown,
        "hallucination_by_stratum": Counter(by_stratum or {}),
        "preservation_by_stratum": Counter(preservation_by_stratum or {}),
        "preservation_by_font": Counter(preservation_by_font or {}),
    }


class CohortIndependenceTests(unittest.TestCase):
    def test_shipped_recipes_are_independent(self) -> None:
        report = assert_cohort_independence(V2_RECIPES)
        for pair, overlap in report.items():
            for axis, values in overlap.items():
                self.assertEqual(values, [], f"{pair} shares {axis}")

    def test_overlapping_fonts_are_refused(self) -> None:
        broken = dict(V2_RECIPES)
        first = V2_RECIPES["line_calibration_v2"]
        broken["line_calibration_v2"] = SplitRecipe(
            name=first.name, role=first.role,
            fonts=first.fonts + ("arial.ttf",),      # already in supplement
            words=first.words, templates=first.templates, seed=first.seed,
            max_renderings=first.max_renderings, quotas=first.quotas)
        with self.assertRaises(ValueError):
            assert_cohort_independence(broken)

    def test_shared_seed_is_refused(self) -> None:
        broken = dict(V2_RECIPES)
        first = V2_RECIPES["line_preflight_v2"]
        broken["line_preflight_v2"] = SplitRecipe(
            name=first.name, role=first.role, fonts=first.fonts,
            words=first.words, templates=first.templates,
            seed=V2_RECIPES["line_calibration_v2"].seed,
            max_renderings=first.max_renderings, quotas=first.quotas)
        with self.assertRaises(ValueError):
            assert_cohort_independence(broken)

    def test_external_cohort_overlap_is_refused(self) -> None:
        """A split must not reuse the rate pilot's words."""
        external = {"rate_pilot_v2": {
            "words": ["reglage"], "templates": [], "seed": 20260817}}
        with self.assertRaises(ValueError):
            assert_cohort_independence(V2_RECIPES, external)

    def test_bare_placeholder_template_is_exempt(self) -> None:
        """'{}' carries no lexical content, so sharing it is not leakage."""
        recipes = dict(V2_RECIPES)
        first = V2_RECIPES["line_calibration_v2"]
        recipes["line_calibration_v2"] = SplitRecipe(
            name=first.name, role=first.role, fonts=first.fonts,
            words=first.words,
            templates=first.templates + ("{}",),   # also used by the supplement
            seed=first.seed, max_renderings=first.max_renderings,
            quotas=first.quotas)
        assert_cohort_independence(recipes)         # must not raise

    def test_a_shared_contentful_template_is_refused(self) -> None:
        recipes = dict(V2_RECIPES)
        first = V2_RECIPES["line_calibration_v2"]
        recipes["line_calibration_v2"] = SplitRecipe(
            name=first.name, role=first.role, fonts=first.fonts,
            words=first.words,
            templates=first.templates + ("Il {} lla",),   # supplement's
            seed=first.seed, max_renderings=first.max_renderings,
            quotas=first.quotas)
        with self.assertRaises(ValueError):
            assert_cohort_independence(recipes)


class QuotaEvaluationTests(unittest.TestCase):
    def test_all_quotas_short_is_not_met(self) -> None:
        recipe = V2_RECIPES["line_calibration_v2"]
        self.assertFalse(recipe.quotas_met(counts()))

    def test_one_short_quota_blocks_the_rest(self) -> None:
        """The failure the joint gate exists for."""
        recipe = V2_RECIPES["line_calibration_v2"]
        satisfied = counts(
            hallucination=100, preservation=500, unknown=500,
            by_stratum={s: 40 for s in MACRO_STRATA},
            preservation_by_stratum={s: 200 for s in MACRO_STRATA},
            preservation_by_font={f: 200 for f in recipe.fonts})
        self.assertTrue(recipe.quotas_met(satisfied))
        satisfied["preservation_by_font"]["georgia.ttf"] = 99
        self.assertFalse(recipe.quotas_met(satisfied))

    def test_supplement_credit_is_applied(self) -> None:
        """train_v1's backfilled events count toward the stratum quotas."""
        recipe = V2_RECIPES["line_train_supplement_v2"]
        state = recipe.quota_state(counts(by_stratum={"SMALL": 0}))
        self.assertEqual(state["hallucination_SMALL"]["credited"], 107)
        self.assertTrue(state["hallucination_SMALL"]["met"])
        self.assertFalse(state["hallucination_LARGE"]["met"])

    def test_large_is_the_supplement_shortfall(self) -> None:
        recipe = V2_RECIPES["line_train_supplement_v2"]
        state = recipe.quota_state(counts(by_stratum={"LARGE": 49}))
        self.assertTrue(state["hallucination_LARGE"]["met"])
        state = recipe.quota_state(counts(by_stratum={"LARGE": 48}))
        self.assertFalse(state["hallucination_LARGE"]["met"])

    def test_credit_does_not_leak_into_other_splits(self) -> None:
        for name in ("line_calibration_v2", "line_preflight_v2"):
            self.assertEqual(V2_RECIPES[name].credit, {})

    def test_supplement_has_no_per_font_preservation_quota(self) -> None:
        recipe = V2_RECIPES["line_train_supplement_v2"]
        state = recipe.quota_state(counts(preservation=2000))
        self.assertFalse(any(k.startswith("preservation_font_") for k in state))


class BudgetTests(unittest.TestCase):
    def test_max_renderings_match_the_joint_gate(self) -> None:
        self.assertEqual(V2_RECIPES["line_train_supplement_v2"].max_renderings,
                         23095)
        self.assertEqual(V2_RECIPES["line_calibration_v2"].max_renderings, 39189)
        self.assertEqual(V2_RECIPES["line_preflight_v2"].max_renderings, 62772)

    def test_every_stratum_has_a_target(self) -> None:
        self.assertEqual(tuple(STRATUM_TARGETS), MACRO_STRATA)

    def test_recipes_serialise_deterministically(self) -> None:
        for recipe in V2_RECIPES.values():
            first = json.dumps(recipe.as_dict(), sort_keys=True)
            second = json.dumps(recipe.as_dict(), sort_keys=True)
            self.assertEqual(first, second)

    def test_recipe_is_immutable(self) -> None:
        with self.assertRaises(Exception):
            V2_RECIPES["line_calibration_v2"].seed = 1   # type: ignore[misc]


class RenderingDeterminismTests(unittest.TestCase):
    def setUp(self) -> None:
        import sys
        scripts = Path(__file__).resolve().parents[1] / "scripts"
        if str(scripts) not in sys.path:
            sys.path.insert(0, str(scripts))

    def test_rendering_is_a_function_of_index(self) -> None:
        from generate_v2_split import Rendering
        recipe = V2_RECIPES["line_calibration_v2"]
        for index in (0, 7, 1234):
            first, second = Rendering(index, recipe), Rendering(index, recipe)
            self.assertEqual(first.text, second.text)
            self.assertEqual(first.font, second.font)
            self.assertEqual(first.upscale, second.upscale)
            self.assertEqual(first.stratum, second.stratum)

    def test_strata_rotate_so_prefixes_stay_balanced(self) -> None:
        """Stopping early must not skew the stratum mix."""
        from generate_v2_split import Rendering
        recipe = V2_RECIPES["line_preflight_v2"]
        for prefix in (300, 900, 3000):
            seen = Counter(Rendering(i, recipe).stratum for i in range(prefix))
            self.assertEqual(len(seen), len(MACRO_STRATA))
            self.assertLessEqual(max(seen.values()) - min(seen.values()), 1)

    def test_fonts_are_evenly_exposed(self) -> None:
        from generate_v2_split import Rendering
        recipe = V2_RECIPES["line_preflight_v2"]
        seen = Counter(Rendering(i, recipe).font for i in range(6000))
        self.assertEqual(set(seen), set(recipe.fonts))
        self.assertLessEqual(max(seen.values()) - min(seen.values()), 3)

    def test_every_font_reaches_every_stratum(self) -> None:
        from generate_v2_split import Rendering
        for recipe in V2_RECIPES.values():
            pairs = {(Rendering(i, recipe).font, Rendering(i, recipe).stratum)
                     for i in range(2000)}
            self.assertEqual(len(pairs), len(recipe.fonts) * len(MACRO_STRATA))

    def test_target_is_inside_the_substituted_word(self) -> None:
        from generate_v2_split import Rendering
        for recipe in V2_RECIPES.values():
            for index in range(0, 600, 37):
                rendering = Rendering(index, recipe)
                position = rendering.target_position()
                self.assertGreaterEqual(position, 0)
                self.assertEqual(rendering.text[position],
                                 rendering.target_character)
                self.assertGreaterEqual(position,
                                        rendering.template.index("{}"))

    def test_templates_contribute_no_e_forms(self) -> None:
        for recipe in V2_RECIPES.values():
            for template in recipe.templates:
                body = template.replace("{}", "")
                self.assertEqual({c for c in body if c in {"e", "é"}}, set())


if __name__ == "__main__":
    unittest.main()
