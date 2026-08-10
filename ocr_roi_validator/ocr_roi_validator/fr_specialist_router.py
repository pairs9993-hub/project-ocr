"""Routing between the French baseline recognizer and a confusable specialist.

The baseline French recognizer occasionally emits a spurious acute accent
(``Veuillez`` -> ``Véuillez``). A recovery-trained specialist fixes some of
those rows but introduces its own regressions elsewhere, notably numeric ones
such as ``1.5`` -> ``1.s5``. Replacing the baseline outright is therefore not
safe.

This module decides, per ROI, whether the specialist output may replace the
baseline output. The policy is deliberately narrow: the specialist wins only
when it differs from the baseline by exactly one character, and only when that
single difference is one of the confusable pairs the specialist was trained to
repair. Everything else -- length changes, insertions, deletions, digit edits,
whitespace or punctuation edits, or several edits at once -- keeps the
baseline.

The routing decision is a pure function of the two OCR strings. It never sees
the expected text: expected text is used only for the exact comparison that
happens after OCR is complete, never to pick a recognizer or repair a result.
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


# Substitutions the specialist is allowed to win. Ordered as
# (baseline_character, specialist_character).
ALLOWED_PAIRS = frozenset(
    {
        ("e", "é"),
        ("é", "e"),
        ("i", "l"),
        ("l", "i"),
    }
)

ROUTE_IDENTICAL = "IDENTICAL"
ROUTE_ALLOWED_SUBSTITUTION = "ALLOWED_SINGLE_CONFUSABLE_SUBSTITUTION"
ROUTE_BLOCKED_LENGTH_CHANGE = "BLOCKED_LENGTH_CHANGE"
ROUTE_BLOCKED_MULTIPLE_CHANGES = "BLOCKED_MULTIPLE_CHANGES"
ROUTE_BLOCKED_NON_CONFUSABLE_CHANGE = "BLOCKED_NON_CONFUSABLE_CHANGE"


@dataclass(frozen=True)
class SpecialistRouteResult:
    """Outcome of a single baseline-vs-specialist routing decision."""

    final_text: str
    baseline_text: str
    specialist_text: str
    route: str
    specialist_applied: bool


def _normalize(text: str) -> str:
    """Apply NFC normalization only.

    This is the sole normalization the router performs. Unlike the scoring
    helpers used by the promotion gate, it does not collapse whitespace and
    does not fold typographic apostrophes -- doing so would hide precisely the
    differences this router exists to block.
    """
    return unicodedata.normalize("NFC", text or "")


def route_specialist_text(baseline_text: str, specialist_text: str) -> SpecialistRouteResult:
    """Decide whether ``specialist_text`` may replace ``baseline_text``.

    Both inputs are OCR outputs. There is deliberately no parameter for the
    expected text: the routing decision must not depend on the answer key.

    Returns a :class:`SpecialistRouteResult` whose ``final_text`` is the string
    the caller should use. ``final_text`` equals the baseline unless the route
    is :data:`ROUTE_ALLOWED_SUBSTITUTION`.
    """
    baseline = _normalize(baseline_text)
    specialist = _normalize(specialist_text)

    def keep_baseline(route: str) -> SpecialistRouteResult:
        return SpecialistRouteResult(
            final_text=baseline,
            baseline_text=baseline,
            specialist_text=specialist,
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
    pair = (baseline[index], specialist[index])
    if pair not in ALLOWED_PAIRS:
        return keep_baseline(ROUTE_BLOCKED_NON_CONFUSABLE_CHANGE)

    return SpecialistRouteResult(
        final_text=specialist,
        baseline_text=baseline,
        specialist_text=specialist,
        route=ROUTE_ALLOWED_SUBSTITUTION,
        specialist_applied=True,
    )
