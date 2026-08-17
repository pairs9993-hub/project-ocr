"""Immutable recipes for the three v2 quota-driven splits.

One generator serves all three; only these declarations differ. Keeping the
split-specific parts as data rather than as three near-identical scripts means
the funnel, the parity guarantees and the stopping rule cannot drift apart
between splits.

Cohorts are disjoint by construction: fonts, words, templates, sizes and seeds
are partitioned here, and :func:`assert_cohort_independence` refuses to build a
recipe set that violates it rather than leaving the check to a later audit.

Budgets come from the joint-quota gate -- the smallest N at which *all* of a
split's quotas are met together with probability 0.95, estimated by two-stage
bootstrap of the rate pilot. They are maxima, not targets: generation stops
early the moment the quotas are satisfied.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

__all__ = [
    "SplitRecipe",
    "V2_RECIPES",
    "MACRO_STRATA",
    "assert_cohort_independence",
]

MACRO_STRATA = ("SMALL", "MEDIUM", "LARGE")

# Targeting settings measured on the pilot: the (size, upscale) combinations
# that land in each stratum 81-96% of the time. Fixed before generation.
STRATUM_TARGETS: Mapping[str, dict] = {
    "SMALL": {"sizes": (11, 12, 13), "upscale": (1.0, 1.0)},
    "MEDIUM": {"sizes": (12, 13, 14), "upscale": (1.45, 1.75)},
    "LARGE": {"sizes": (17, 18, 20), "upscale": (1.55, 2.05)},
}

# Shared perturbation grid. Vertical padding is held at 20px because the
# detector resizes the short side to 640 and small canvases were being blown up
# 20-30x, destroying the text before detection.
PERTURBATION = {
    "pad_x": (12, 30),
    "pad_y": 20,
    "dark_background_probability": 0.75,
    "dark_background": (8, 34), "dark_foreground": (220, 252),
    "light_background": (226, 250), "light_foreground": (8, 40),
    "contrast": (0.85, 1.20),
    "blur": (0.0, 0.5),
    "jitter": (-0.5, 0.5),
    "resample": ("bicubic", "lanczos"),
}


@dataclass(frozen=True)
class SplitRecipe:
    """Everything that distinguishes one split. Frozen once constructed."""

    name: str
    role: str
    fonts: tuple[str, ...]
    words: tuple[tuple[str, str], ...]
    templates: tuple[str, ...]
    seed: int
    max_renderings: int
    quotas: Mapping[str, object]
    credit: Mapping[str, int] = field(default_factory=dict)

    def quota_state(self, counts: Mapping[str, object]) -> dict:
        """Evaluate every quota against observed counts.

        Returns one entry per quota with its requirement, its current value and
        whether it is met, so a stop decision and a failure report are the same
        computation rather than two that might disagree.
        """
        state: dict[str, dict] = {}
        credited = sum(self.credit.values())
        state["hallucination_total"] = {
            "required": self.quotas["hallucination_total"],
            "observed": counts["hallucination_total"] + credited,
            "credited": credited,
        }
        for stratum in MACRO_STRATA:
            state[f"hallucination_{stratum}"] = {
                "required": self.quotas["hallucination_per_stratum"],
                "observed": (counts["hallucination_by_stratum"].get(stratum, 0)
                             + self.credit.get(stratum, 0)),
                "credited": self.credit.get(stratum, 0),
            }
        state["preservation_total"] = {
            "required": self.quotas["preservation_total"],
            "observed": counts["preservation_total"], "credited": 0,
        }
        if self.quotas["preservation_per_stratum"]:
            for stratum in MACRO_STRATA:
                state[f"preservation_{stratum}"] = {
                    "required": self.quotas["preservation_per_stratum"],
                    "observed": counts["preservation_by_stratum"].get(stratum, 0),
                    "credited": 0,
                }
        if self.quotas["preservation_per_font"]:
            for font in self.fonts:
                state[f"preservation_font_{font}"] = {
                    "required": self.quotas["preservation_per_font"],
                    "observed": counts["preservation_by_font"].get(font, 0),
                    "credited": 0,
                }
        state["unknown_total"] = {
            "required": self.quotas["unknown_total"],
            "observed": counts["unknown_total"], "credited": 0,
        }
        for entry in state.values():
            entry["met"] = entry["observed"] >= entry["required"]
        return state

    def quotas_met(self, counts: Mapping[str, object]) -> bool:
        return all(entry["met"] for entry in self.quota_state(counts).values())

    def as_dict(self) -> dict:
        return {
            "name": self.name, "role": self.role, "fonts": list(self.fonts),
            "words": [list(pair) for pair in self.words],
            "templates": list(self.templates), "seed": self.seed,
            "max_renderings": self.max_renderings,
            "quotas": dict(self.quotas), "credit": dict(self.credit),
            "stratum_targets": {k: {"sizes": list(v["sizes"]),
                                    "upscale": list(v["upscale"])}
                                for k, v in STRATUM_TARGETS.items()},
            "perturbation": {k: (list(v) if isinstance(v, tuple) else v)
                             for k, v in PERTURBATION.items()},
        }


# train_v1's hallucinations, each verified by re-render parity, count toward the
# supplement's stratum quotas. Only LARGE falls short, by 49.
TRAIN_V1_CREDIT = {"SMALL": 107, "MEDIUM": 82, "LARGE": 11}

V2_RECIPES: dict[str, SplitRecipe] = {
    "line_train_supplement_v2": SplitRecipe(
        name="line_train_supplement_v2",
        role="training_only",
        fonts=("arial.ttf", "calibri.ttf", "segoeui.ttf", "verdana.ttf",
               "corbel.ttf", "Candara.ttf"),
        words=(("reglage", "réglagé"), ("element", "élémént"),
               ("decale", "décalé"), ("reserve", "résérvé"),
               ("general", "général"), ("montee", "montéé")),
        templates=("{}", "Vous {} localis", "Il {} lla", "{} kd/hb 2,5",
                   "L'{} du bac 1,5"),
        seed=30100000,
        max_renderings=23095,
        quotas={
            "hallucination_total": 300, "hallucination_per_stratum": 60,
            "preservation_total": 2000, "preservation_per_stratum": 0,
            "preservation_per_font": 0, "unknown_total": 2000,
        },
        credit=TRAIN_V1_CREDIT,
    ),
    "line_calibration_v2": SplitRecipe(
        name="line_calibration_v2",
        role="threshold_and_calibration_only",
        fonts=("trebuc.ttf", "georgia.ttf", "palab.ttf"),
        words=(("repare", "réparé"), ("melange", "mélangé"),
               ("semelle", "séméllé")),
        templates=("Application {} tot", "H'{} ktb 1,5", "{} zr 8"),
        seed=30200000,
        max_renderings=39189,
        quotas={
            "hallucination_total": 100, "hallucination_per_stratum": 20,
            "preservation_total": 500, "preservation_per_stratum": 100,
            "preservation_per_font": 100, "unknown_total": 500,
        },
    ),
    "line_preflight_v2": SplitRecipe(
        name="line_preflight_v2",
        role="frozen_model_pre_gate_not_final_safety_evidence",
        fonts=("times.ttf", "framd.ttf", "tahoma.ttf", "consola.ttf"),
        words=(("degage", "dégagé"), ("severe", "sévéré"),
               ("deneige", "dénéigé"), ("relever", "rélévér")),
        templates=("{} du top", "Tk {} biffl", "{}: tud 30", "Wm {} 6,1"),
        seed=30300000,
        max_renderings=62772,
        quotas={
            "hallucination_total": 100, "hallucination_per_stratum": 20,
            "preservation_total": 1000, "preservation_per_stratum": 200,
            "preservation_per_font": 200, "unknown_total": 1000,
        },
    ),
}


def assert_cohort_independence(recipes: Mapping[str, SplitRecipe],
                               external: Mapping[str, dict] | None = None) -> dict:
    """Refuse a recipe set whose cohorts overlap.

    Checked here rather than only in a later audit, so an overlapping cohort
    cannot reach generation at all. Templates legitimately share the bare
    placeholder ``{}``, which carries no lexical content, so it is exempt.
    """
    names = list(recipes)
    report: dict[str, dict] = {}
    for index, first in enumerate(names):
        for second in names[index + 1:]:
            a, b = recipes[first], recipes[second]
            words_a = {w for pair in a.words for w in pair}
            words_b = {w for pair in b.words for w in pair}
            overlap = {
                "font": sorted(set(a.fonts) & set(b.fonts)),
                "word": sorted(words_a & words_b),
                "template": sorted((set(a.templates) & set(b.templates)) - {"{}"}),
                "seed": [a.seed] if a.seed == b.seed else [],
            }
            report[f"{first} vs {second}"] = overlap
            for axis, values in overlap.items():
                if values:
                    raise ValueError(
                        f"{first} and {second} share {axis}: {values}")

    for name, recipe in recipes.items():
        for other_name, other in (external or {}).items():
            words = {w for pair in recipe.words for w in pair}
            shared_words = sorted(words & set(other.get("words", ())))
            shared_templates = sorted(
                (set(recipe.templates) & set(other.get("templates", ()))) - {"{}"})
            if shared_words or shared_templates:
                raise ValueError(
                    f"{name} overlaps {other_name}: words={shared_words} "
                    f"templates={shared_templates}")
            if recipe.seed == other.get("seed"):
                raise ValueError(f"{name} shares a seed with {other_name}")
            report[f"{name} vs {other_name}"] = {
                "word": shared_words, "template": shared_templates, "seed": [],
            }
    return report
