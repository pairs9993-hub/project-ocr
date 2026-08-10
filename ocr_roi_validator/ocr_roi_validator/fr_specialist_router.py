"""Routing between the French baseline recognizer and a confusable specialist.

The baseline French recognizer emits a spurious acute accent on some real
screens (``Veuillez`` -> ``Véuillez``), producing a false FAIL. A
recovery-trained specialist repairs it on some renderings but regresses
elsewhere, notably ``1.5`` -> ``1.s5``, so it cannot replace the baseline.

This module decides, per ROI, whether the specialist output may replace the
baseline output. The policy is deliberately narrow: the specialist wins only
when it differs from the baseline by exactly one character, and only when that
difference removes an acute accent that the baseline hallucinated. Everything
else -- length changes, insertions, deletions, digit edits, whitespace or
punctuation edits, or several edits at once -- keeps the baseline.

The routing decision is a pure function of the two OCR strings. It never sees
the expected text: expected text is used only for the exact comparison that
happens after OCR is complete, never to pick a recognizer or repair a result.

.. warning::

   **This router is an experimental offline policy for evaluating specialist
   proposals. It must not be used for runtime enforcement without an
   image-based glyph verifier.**

   The router compares two strings. It cannot tell the difference between
   "the baseline hallucinated an accent that is not on screen" and "the screen
   genuinely shows ``Véuillez`` and the specialist wrongly erased the accent".
   Both look identical at the string level. Accepting ``é`` -> ``e`` therefore
   risks silently correcting a real on-screen typo, which is exactly the
   failure this project must never produce. Deciding that safely requires
   inspecting the glyph pixels, not the decoded text.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

__all__ = [
    "ALLOWED_PAIRS",
    "ROUTE_IDENTICAL",
    "ROUTE_ALLOWED_SUBSTITUTION",
    "ROUTE_BLOCKED_LENGTH_CHANGE",
    "ROUTE_BLOCKED_MULTIPLE_CHANGES",
    "ROUTE_BLOCKED_NON_CONFUSABLE_CHANGE",
    "SpecialistRouteResult",
    "route_specialist_text",
]


# Substitutions the specialist is allowed to win, as
# (baseline_character, specialist_character).
#
# Scoped to the single failure under investigation: the baseline adds an acute
# accent that is not on screen, and the specialist removes it. The reverse
# direction (``e`` -> ``é``) and the i/l pair are deliberately NOT allowed --
# they were not needed for this target and every additional pair widens the
# window in which a genuine on-screen typo could be masked.
ALLOWED_PAIRS = frozenset({("é", "e")})

ROUTE_IDENTICAL = "IDENTICAL"
ROUTE_ALLOWED_SUBSTITUTION = "ALLOWED_SINGLE_CONFUSABLE_SUBSTITUTION"
ROUTE_BLOCKED_LENGTH_CHANGE = "BLOCKED_LENGTH_CHANGE"
ROUTE_BLOCKED_MULTIPLE_CHANGES = "BLOCKED_MULTIPLE_CHANGES"
ROUTE_BLOCKED_NON_CONFUSABLE_CHANGE = "BLOCKED_NON_CONFUSABLE_CHANGE"


@dataclass(frozen=True)
class SpecialistRouteResult:
    """Outcome of a single baseline-vs-specialist routing decision.

    ``baseline_text`` and ``specialist_text`` are the caller's original
    strings, preserved codepoint for codepoint. ``final_text`` is whichever of
    those two the router selected -- never a normalized rewrite of either.
    """

    final_text: str
    baseline_text: str
    specialist_text: str
    route: str
    specialist_applied: bool


def _normalize_for_comparison(text: str) -> str:
    """NFC-normalize a string *for comparison only*.

    Composed and decomposed forms of the same character must not read as a
    substitution, so the comparison runs on NFC. The normalized form is never
    returned to the caller: routing must not silently rewrite text it decided
    to keep.

    Note this deliberately does not collapse whitespace or fold typographic
    apostrophes, unlike the scoring helpers in the promotion gate -- doing so
    would hide precisely the differences this router exists to block.
    """
    return unicodedata.normalize("NFC", text or "")


def route_specialist_text(baseline_text: str, specialist_text: str) -> SpecialistRouteResult:
    """Decide whether ``specialist_text`` may replace ``baseline_text``.

    Both inputs are OCR outputs. There is deliberately no parameter for the
    expected text: the routing decision must not depend on the answer key.

    Returns a :class:`SpecialistRouteResult` whose ``final_text`` is the string
    the caller should use. Unless the route is
    :data:`ROUTE_ALLOWED_SUBSTITUTION`, ``final_text`` is the original
    ``baseline_text`` unchanged -- byte for byte, including any decomposed
    characters it contained.
    """
    baseline_original = baseline_text or ""
    specialist_original = specialist_text or ""
    baseline = _normalize_for_comparison(baseline_original)
    specialist = _normalize_for_comparison(specialist_original)

    def keep_baseline(route: str) -> SpecialistRouteResult:
        return SpecialistRouteResult(
            final_text=baseline_original,
            baseline_text=baseline_original,
            specialist_text=specialist_original,
            route=route,
            specialist_applied=False,
        )

    if baseline == specialist:
        return keep_baseline(ROUTE_IDENTICAL)

    # A length change means an insertion or a deletion. Those are never
    # allowed, so this also covers `1.5` -> `1.s5`.
    if len(baseline) != len(specialist):
        return keep_baseline(ROUTE_BLOCKED_LENGTH_CHANGE)

    # Equal lengths: compare position by position. Because the lengths match,
    # every difference is a substitution.
    differing = [
        index
        for index, (left, right) in enumerate(zip(baseline, specialist))
        if left != right
    ]

    if len(differing) != 1:
        return keep_baseline(ROUTE_BLOCKED_MULTIPLE_CHANGES)

    index = differing[0]
    if (baseline[index], specialist[index]) not in ALLOWED_PAIRS:
        return keep_baseline(ROUTE_BLOCKED_NON_CONFUSABLE_CHANGE)

    return SpecialistRouteResult(
        final_text=specialist_original,
        baseline_text=baseline_original,
        specialist_text=specialist_original,
        route=ROUTE_ALLOWED_SUBSTITUTION,
        specialist_applied=True,
    )
