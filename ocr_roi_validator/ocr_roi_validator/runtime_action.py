"""Map an internal verdict to the action the product actually performs.

Three of the four internal outcomes do the same thing: they leave the baseline
text alone. Only BARE_E changes anything. So two runtimes can disagree on a
verdict and still produce identical output, and the distinction matters for
deployment even though it does not for diagnosis.

That is not an excuse to look away from the verdict mismatch -- it stays on the
record as a diagnostic. But the question "would the user see a different
string" has its own answer, and this module is where it is computed.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "APPLY_E_CORRECTION", "KEEP_BASELINE",
    "ACTION_BY_VERDICT", "action_of", "apply_action",
]

ACCENT_PRESENT, BARE_E, UNKNOWN, PREMODEL_UNKNOWN = 0, 1, 2, 3

APPLY_E_CORRECTION = "APPLY_E_CORRECTION"
KEEP_BASELINE = "KEEP_BASELINE"

ACTION_BY_VERDICT = {
    ACCENT_PRESENT: KEEP_BASELINE,
    BARE_E: APPLY_E_CORRECTION,
    UNKNOWN: KEEP_BASELINE,
    PREMODEL_UNKNOWN: KEEP_BASELINE,
}


def action_of(verdicts) -> np.ndarray:
    """Action for each verdict. Only BARE_E does anything."""
    verdicts = np.asarray(verdicts)
    return np.where(verdicts == BARE_E, APPLY_E_CORRECTION, KEEP_BASELINE)


def apply_action(baseline_text: str, action: str, position: int) -> str:
    """Rewrite the accented character at ``position`` back to a bare e.

    Only that one character may change. The baseline's own codepoints are
    returned unchanged everywhere else, and an out-of-range position or a
    non-accent character at the target leaves the string untouched.
    """
    if action != APPLY_E_CORRECTION:
        return baseline_text
    if position < 0 or position >= len(baseline_text):
        return baseline_text
    if baseline_text[position] not in ("é", "É"):
        return baseline_text
    replacement = "e" if baseline_text[position] == "é" else "E"
    return baseline_text[:position] + replacement + baseline_text[position + 1:]
