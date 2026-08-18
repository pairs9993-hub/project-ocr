"""Fail-closed uncertainty band around the frozen decision threshold.

The base threshold was derived from one sample's own confidence stepped by a
single float32 ulp, which left that sample sitting 5.96e-08 from the boundary.
PyTorch and ONNX Runtime differ by up to 5.7e-06 on these inputs, so the sample
lands on opposite sides depending on which runtime evaluates it -- the class
argmax agreed everywhere, but the thresholded verdict did not.

Rather than move the boundary, a band is placed around it. Anything inside the
band returns UNKNOWN regardless of runtime, so a numerical difference smaller
than the band can no longer change a decision. The base threshold is untouched
and stays authoritative for what it was calibrated on.

The band is one-directional in effect: it can only turn a BARE_E into an
UNKNOWN, never create a correction that the ungated rule would not have made.
:func:`assert_monotonic` checks that rather than trusting it.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "ACCENT_PRESENT", "BARE_E", "UNKNOWN",
    "RUNTIME_EPSILON", "guarded_verdict", "assert_monotonic",
]

ACCENT_PRESENT, BARE_E, UNKNOWN = 0, 1, 2

# Fixed once, at roughly 17.6x the largest observed Torch/ONNX disagreement of
# 5.692e-06. Not to be retuned after seeing results.
RUNTIME_EPSILON = np.float32(1e-4)


def ungated_verdict(probabilities: np.ndarray, threshold) -> np.ndarray:
    """The original rule: argmax, applied only above the base threshold."""
    probabilities = np.asarray(probabilities, dtype=np.float32)
    threshold = np.float32(threshold)
    predicted = probabilities.argmax(axis=1).astype(np.int64)
    confident = probabilities.max(axis=1) >= threshold
    return np.where(confident, predicted, UNKNOWN)


def guarded_verdict(probabilities: np.ndarray, threshold,
                    epsilon=RUNTIME_EPSILON) -> np.ndarray:
    """Apply the base rule, then withhold BARE_E inside the band.

    A correction is allowed only when the bare-e probability clears the
    threshold by more than epsilon. Everything within epsilon of the boundary
    becomes UNKNOWN, which is the fail-closed direction: the baseline text
    stands unchanged.
    """
    probabilities = np.asarray(probabilities, dtype=np.float32)
    threshold = np.float32(threshold)
    epsilon = np.float32(epsilon)

    verdict = ungated_verdict(probabilities, threshold)
    bare_probability = probabilities[:, BARE_E]
    # Only a bare-e correction is gated. Accent and UNKNOWN outcomes are left
    # exactly as the base rule decided them, so the band cannot manufacture a
    # new correction anywhere.
    clears_band = bare_probability >= np.float32(threshold + epsilon)
    return np.where((verdict == BARE_E) & ~clears_band, UNKNOWN, verdict)


def in_uncertainty_band(probabilities: np.ndarray, threshold,
                        epsilon=RUNTIME_EPSILON) -> np.ndarray:
    """Rows whose bare-e probability sits within epsilon of the threshold."""
    probabilities = np.asarray(probabilities, dtype=np.float32)
    distance = np.abs(probabilities[:, BARE_E] - np.float32(threshold))
    return distance <= np.float32(epsilon)


def assert_monotonic(probabilities: np.ndarray, threshold,
                     epsilon=RUNTIME_EPSILON) -> dict:
    """Verify the guard only ever removes corrections.

    Raises if any row gains a BARE_E it did not have, or if a verdict changes
    in any way other than BARE_E to UNKNOWN.
    """
    base = ungated_verdict(probabilities, threshold)
    guarded = guarded_verdict(probabilities, threshold, epsilon)

    gained = np.where((guarded == BARE_E) & (base != BARE_E))[0]
    if gained.size:
        raise ValueError(
            "guard created %d BARE_E verdicts that the base rule did not make; "
            "rows %s" % (gained.size, gained[:8].tolist()))

    changed = np.where(base != guarded)[0]
    illegal = [int(i) for i in changed
               if not (base[i] == BARE_E and guarded[i] == UNKNOWN)]
    if illegal:
        raise ValueError(
            "guard produced transitions other than BARE_E->UNKNOWN at rows %s"
            % illegal[:8])

    return {
        "rows": int(len(base)),
        "base_bare": int((base == BARE_E).sum()),
        "guarded_bare": int((guarded == BARE_E).sum()),
        "withheld": int(len(changed)),
        "guarded_bare_is_subset": True,
        "only_bare_to_unknown": True,
    }
