"""Counterfactual pair recipes for the line verifier's primary training data.

Every attempt so far has tried to collect training data from lines where the
baseline happened to hallucinate. That has now failed three times, and
discovery showed why: of 150 words, four carry robust support and 130 produced
no event in 400 renderings. Mining natural hallucinations is a poor way to
teach the model what an accent looks like, because the supply is governed by
something other than the visual question being asked.

So the primary training signal moves to counterfactual pairs. The same word,
same font, same size, same optical settings, rendered once with a bare e and
once with an accent -- a supervision signal available for any word, not only the
rare ones the baseline stumbles on. Natural hallucinations keep a role, but as
coverage evidence rather than as the training set.

The leakage hazard is specific and was already found once in this work: cropping
both pair members with geometry measured from the accented one leaks the answer
into the bare image. Each member's runtime view is therefore built from its own
image alone, and the audit view that compares them is barred from the loader.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Mapping

__all__ = [
    "CounterfactualRecipe",
    "COUNTERFACTUAL_RECIPES",
    "build_word_contexts",
    "assert_context_independence",
]

# Fonts span all cohorts: the visual question does not depend on a font's
# trigger rate, so none is excluded.
COUNTERFACTUAL_FONTS = (
    "arial.ttf", "calibri.ttf", "segoeui.ttf", "verdana.ttf", "corbel.ttf",
    "Candara.ttf", "trebuc.ttf", "georgia.ttf", "palab.ttf", "times.ttf",
    "framd.ttf", "tahoma.ttf", "consola.ttf",
)

# Strings that must never be rendered or used to select anything.
FORBIDDEN = frozenset({"veuillez", "véuillez", "allumer", "eau", "l'eau"})


def build_word_contexts(count: int, salt: str,
                        exclude: frozenset[str]) -> tuple[tuple[str, str], ...]:
    """Construct synthetic word contexts deterministically from a salt.

    Syllables are combined by a hash of the salt and position, so a recipe's
    word list is fixed by its name rather than chosen by anyone, and two
    recipes with different salts cannot collide by accident.
    """
    onsets = ("br", "cl", "dr", "fl", "gr", "pl", "sp", "st", "tr", "vl",
              "kr", "bl", "gl", "pr", "sc", "sk", "sm", "sn", "sv", "tw")
    codas = ("nd", "rm", "lk", "st", "pt", "nk", "rd", "mp", "ls", "rt",
             "nt", "sk", "ft", "lp", "rn", "mb")
    tails = ("o", "a", "i", "u", "ol", "ar", "in", "us", "al", "or")
    prefixes = ("", "", "", "b", "d", "k", "l", "t", "m", "n", "r", "s",
                "g", "p", "v", "z", "B", "D", "K", "L", "M", "N", "P", "R")

    contexts: list[tuple[str, str]] = []
    seen: set[str] = set()
    position = 0
    while len(contexts) < count and position < count * 200:
        digest = hashlib.sha256(f"{salt}|{position}".encode()).digest()
        prefix = prefixes[digest[0] % len(prefixes)]
        onset = onsets[digest[1] % len(onsets)]
        coda = codas[digest[2] % len(codas)]
        tail = tails[digest[3] % len(tails)]
        # The e sits between onset and coda; the prefix shifts its position.
        word = f"{prefix}{onset}e{coda}{tail}"
        position += 1
        lowered = word.lower()
        if lowered in seen or lowered in FORBIDDEN or lowered in exclude:
            continue
        if "e" not in word:
            continue
        seen.add(lowered)
        contexts.append((word, word.replace("e", "é")))
    return tuple(contexts)


@dataclass(frozen=True)
class CounterfactualRecipe:
    """One counterfactual split."""

    name: str
    role: str
    salt: str
    word_context_count: int
    pairs_per_context: int
    seed: int
    fonts: tuple[str, ...] = COUNTERFACTUAL_FONTS
    unknown_fraction: float = 0.25
    words: tuple[tuple[str, str], ...] = field(default=(), compare=False)

    @property
    def renderings(self) -> int:
        # Two members per pair, plus the UNKNOWN corruptions drawn alongside.
        base = self.word_context_count * self.pairs_per_context * 2
        return int(round(base * (1.0 + self.unknown_fraction)))

    def as_dict(self) -> dict:
        return {
            "name": self.name, "role": self.role, "salt": self.salt,
            "word_context_count": self.word_context_count,
            "pairs_per_context": self.pairs_per_context,
            "seed": self.seed, "fonts": list(self.fonts),
            "unknown_fraction": self.unknown_fraction,
            "renderings": self.renderings,
            "words": [list(pair) for pair in self.words],
        }


def _make(name: str, role: str, salt: str, count: int, pairs: int, seed: int,
          exclude: frozenset[str]) -> CounterfactualRecipe:
    return CounterfactualRecipe(
        name=name, role=role, salt=salt, word_context_count=count,
        pairs_per_context=pairs, seed=seed,
        words=build_word_contexts(count, salt, exclude))


def _prior_words() -> frozenset[str]:
    """Every word already used anywhere, so counterfactual contexts are new."""
    from ocr_roi_validator.v2_recipes import V2_RECIPES
    from ocr_roi_validator.word_candidates import CANDIDATES

    words: set[str] = set()
    for recipe in V2_RECIPES.values():
        words |= {w.lower() for pair in recipe.words for w in pair}
    words |= {c.bare.lower() for c in CANDIDATES}
    words |= {c.accented.lower() for c in CANDIDATES}
    return frozenset(words)


_EXCLUDE = _prior_words()

COUNTERFACTUAL_RECIPES: Mapping[str, CounterfactualRecipe] = {
    "line_counterfactual_train_v1": _make(
        "line_counterfactual_train_v1", "primary_training_data",
        "counterfactual-train-v1", 200, 24, 50100000, _EXCLUDE),
    "line_counterfactual_calibration_v1": _make(
        "line_counterfactual_calibration_v1", "threshold_calibration_only",
        "counterfactual-calibration-v1", 50, 24, 50200000, _EXCLUDE),
}


def assert_context_independence(
        recipes: Mapping[str, CounterfactualRecipe] = COUNTERFACTUAL_RECIPES,
        prior: frozenset[str] = _EXCLUDE) -> dict:
    """Refuse recipes whose word contexts overlap each other or earlier data."""
    report: dict[str, object] = {}
    names = list(recipes)
    for index, first in enumerate(names):
        for second in names[index + 1:]:
            shared = sorted({w.lower() for pair in recipes[first].words
                             for w in pair}
                            & {w.lower() for pair in recipes[second].words
                               for w in pair})
            report[f"{first} vs {second}"] = shared
            if shared:
                raise ValueError(f"{first} and {second} share contexts: {shared}")
    for name, recipe in recipes.items():
        words = {w.lower() for pair in recipe.words for w in pair}
        clash = sorted(words & prior)
        if clash:
            raise ValueError(f"{name} reuses earlier words: {clash[:8]}")
        forbidden = sorted(words & FORBIDDEN)
        if forbidden:
            raise ValueError(f"{name} contains forbidden strings: {forbidden}")
        report[f"{name} vs prior"] = []
        if len(recipe.words) < recipe.word_context_count:
            raise ValueError(
                f"{name} produced {len(recipe.words)} contexts, "
                f"needs {recipe.word_context_count}")
    return report
