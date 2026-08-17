"""Candidate words for discovering where the baseline's e -> é trigger lives.

Three cohorts have now been chosen without knowing whether their words provoke
the defect at all, and calibration produced zero events in 39,189 renderings
across three words. The pilot used five words, so word-level variance was
invisible; supplement then showed it spans 0.000 to 0.299 between words drawn
from the same list.

This set exists to measure that variance directly, over enough words that a
later split can be assembled from ones with demonstrated support instead of
hope. Candidates are built by crossing the context factors that plausibly
matter -- where the e sits, what surrounds it, how long the word is, whether an
apostrophe or digit is adjacent -- so the report can separate a word-identity
effect from a context effect rather than asserting one.

Every candidate is synthetic and carries at least one bare ``e``. None appears
in any earlier cohort, and the real UI strings are excluded outright.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

__all__ = [
    "WordCandidate",
    "CANDIDATES",
    "EXCLUDED_STRINGS",
    "assert_no_prior_overlap",
    "context_of",
]

# The real on-screen text and its misread form. Never a candidate, never
# rendered, never used to select anything.
EXCLUDED_STRINGS = frozenset({
    "veuillez", "véuillez", "allumer", "eau", "l'eau",
})

VOWELS = frozenset("aeiouyàâéèêëîïôùûü")
ASCENDERS = frozenset("bdfhklt")
DESCENDERS = frozenset("gjpqy")


@dataclass(frozen=True)
class WordCandidate:
    """One synthetic word plus the accented form used for the pair member."""

    bare: str
    accented: str
    group: str          # which construction produced it, for balance reporting

    def target_index(self) -> int:
        """First bare e -- the occurrence the diagnostic queries."""
        return self.bare.index("e")


def _accent(word: str) -> str:
    """Accent every e, matching how the other cohorts build their pair member."""
    return word.replace("e", "é")


def _build() -> tuple[WordCandidate, ...]:
    """Cross the context factors into a fixed candidate list.

    Construction is deterministic and declared before any inference; nothing
    here consults a measured rate.
    """
    candidates: list[WordCandidate] = []
    seen: set[str] = set()

    def add(word: str, group: str) -> None:
        lowered = word.lower()
        if lowered in seen or lowered in EXCLUDED_STRINGS or "e" not in lowered:
            return
        seen.add(lowered)
        candidates.append(WordCandidate(word, _accent(word), group))

    # Position of the e within the word: leading, middle, trailing.
    for prefix, suffix in (("", "tra"), ("", "lmo"), ("", "spri"),
                           ("", "nklo"), ("", "chva")):
        add(f"e{suffix}", "e_leading")
    for stem in ("bral", "trom", "clin", "sprad", "vulm", "graft", "plond"):
        add(f"{stem}e", "e_trailing")
    for head, tail in (("br", "tal"), ("cl", "mor"), ("gr", "nils"),
                       ("pl", "dorm"), ("tr", "vask"), ("fl", "grom"),
                       ("sp", "linq"), ("dr", "moult")):
        add(f"{head}e{tail}", "e_middle")

    # Preceding character class.
    for before in ("b", "d", "f", "h", "k", "l", "t"):        # ascenders
        add(f"{before}emrat", "prev_ascender")
    for before in ("g", "j", "p", "q", "y"):                  # descenders
        add(f"{before}elnort", "prev_descender")
    for before in ("a", "i", "o", "u"):                       # vowels
        add(f"{before}erbint", "prev_vowel")
    for before in ("m", "n", "r", "s", "v", "z"):             # neutral
        add(f"{before}eldunt", "prev_neutral")

    # Following character class.
    for after in ("b", "d", "f", "h", "k", "l", "t"):
        add(f"mre{after}ison", "next_ascender")
    for after in ("g", "j", "p", "q", "y"):
        add(f"clu{after}e{after}na", "next_descender")
    for after in ("a", "i", "o", "u"):
        add(f"vand{after}e{after}rs", "next_vowel")
    for after in ("m", "n", "r", "s", "v", "z"):
        add(f"pol{after}e{after}dit", "next_neutral")

    # An e directly after a capital.
    for head in ("B", "D", "K", "L", "M", "N", "P", "R", "S", "T"):
        add(f"{head}elvrat", "after_capital")

    # Word length, holding the surrounding letters fixed.
    for length, filler in ((4, "b"), (5, "bl"), (6, "blo"), (7, "blon"),
                           (8, "blonk"), (9, "blonkr"), (10, "blonkri"),
                           (11, "blonkris"), (12, "blonkrist")):
        add(f"tre{filler}", f"length_{length}")

    # Repeated e, adjacent and separated.
    for stem in ("kree", "vleem", "sprees", "gleent", "treemol", "bleendra"):
        add(stem, "repeated_e_adjacent")
    for stem in ("keral", "verim", "temoli", "gerondi", "sereplan"):
        add(f"{stem}e", "repeated_e_separated")

    # Apostrophes, digits and punctuation adjacent to the target.
    for stem in ("l'ebrant", "d'eklor", "n'evrist", "s'emondr", "j'eplant"):
        add(stem, "apostrophe_before")
    for stem in ("mre2t", "kli7e", "vro4en", "spa9el", "dru3emp"):
        add(f"{stem}alo", "digit_adjacent")
    for stem in ("bre,mal", "clo.ent", "gru;esk", "prin:elm", "vald-emo"):
        add(stem, "punctuation_adjacent")

    # Ascender/descender density around the target.
    for stem in ("bdlethk", "hklebdt", "tblhedk", "kdhtelb"):
        add(stem, "dense_ascenders")
    for stem in ("gpjeqyg", "qygepjq", "jpgeyqp", "ygqejpg"):
        add(stem, "dense_descenders")

    # Consonant clusters immediately around the e.
    for stem in ("strempl", "schrend", "sprektl", "chlemps", "phrendt",
                 "thremkl", "scrempt", "splendr"):
        add(stem, "cluster_around_e")

    # Narrow versus wide letters, which change the line's aspect ratio.
    for stem in ("iliejli", "iltefil", "jilekil", "tilrejl"):
        add(stem, "narrow_glyphs")
    for stem in ("mwoemwo", "wmuemwa", "mowemwu", "wamemow"):
        add(stem, "wide_glyphs")

    # Filler to reach the required count, varying the two letters that
    # bracket the target while keeping the frame constant.
    frame = ("br", "cl", "dr", "fl", "gr", "pl", "sp", "st", "tr", "vl")
    tails = ("nd", "rm", "lk", "st", "pt", "nk", "rd", "mp")
    for head in frame:
        for tail in tails:
            add(f"{head}e{tail}o", "bracket_grid")

    return tuple(candidates)


RENDERINGS_PER_WORD = 400
DISCOVERY_MAX_RENDERINGS = 60_000
MAX_CANDIDATES = DISCOVERY_MAX_RENDERINGS // RENDERINGS_PER_WORD    # 150


def _trim(candidates: tuple[WordCandidate, ...]) -> tuple[WordCandidate, ...]:
    """Cut to the exposure budget without disturbing the factor balance.

    The construction yields more words than 60,000 renderings at 400 each can
    cover. Taking a prefix would drop whole groups, so words are ranked by a
    stable hash *within* each group and the groups are drawn round-robin --
    every construction survives, thinned evenly. The hash depends only on the
    spelling, so the selection is fixed before any inference and cannot be
    nudged by a result.
    """
    if len(candidates) <= MAX_CANDIDATES:
        return candidates
    grouped: dict[str, list[WordCandidate]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.group, []).append(candidate)
    for group in grouped.values():
        group.sort(key=lambda c: hashlib.sha256(c.bare.encode()).hexdigest())

    kept: list[WordCandidate] = []
    order = sorted(grouped)
    position = 0
    while len(kept) < MAX_CANDIDATES:
        progressed = False
        for name in order:
            if position < len(grouped[name]):
                kept.append(grouped[name][position])
                progressed = True
                if len(kept) == MAX_CANDIDATES:
                    break
        if not progressed:
            break
        position += 1
    return tuple(sorted(kept, key=lambda c: c.bare))


CANDIDATES: tuple[WordCandidate, ...] = _trim(_build())


def context_of(candidate: WordCandidate) -> dict:
    """Context features of the target e, from the spelling alone."""
    word = candidate.bare
    index = candidate.target_index()
    previous = word[index - 1] if index > 0 else ""
    following = word[index + 1] if index + 1 < len(word) else ""

    def classify(character: str) -> str:
        if not character:
            return "none"
        if character.isdigit():
            return "digit"
        if character in "'":
            return "apostrophe"
        if not character.isalpha():
            return "punctuation"
        if character.isupper():
            return "uppercase"
        if character in VOWELS:
            return "vowel"
        if character in ASCENDERS:
            return "ascender"
        if character in DESCENDERS:
            return "descender"
        return "neutral"

    return {
        "word_length": len(word),
        "target_index": index,
        "normalized_position": round(index / max(1, len(word) - 1), 4),
        "preceding_char": previous,
        "following_char": following,
        "preceding_class": classify(previous),
        "following_class": classify(following),
        "local_context": word[max(0, index - 2):index + 3],
        "e_count": word.count("e"),
        "has_apostrophe": "'" in word,
        "has_digit": any(c.isdigit() for c in word),
        "group": candidate.group,
    }


def assert_no_prior_overlap(prior_words: set[str]) -> None:
    """Refuse the candidate set if it reuses any earlier cohort's word."""
    lowered = {w.lower() for w in prior_words}
    clash = sorted({c.bare.lower() for c in CANDIDATES} & lowered)
    if clash:
        raise ValueError(f"candidates overlap prior cohorts: {clash}")
    excluded = sorted({c.bare.lower() for c in CANDIDATES} & EXCLUDED_STRINGS)
    if excluded:
        raise ValueError(f"candidates include excluded strings: {excluded}")
