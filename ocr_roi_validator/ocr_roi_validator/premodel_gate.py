"""Pre-model validation: reject unanswerable queries without inference.

Some queries cannot be answered by looking at pixels because they are
malformed: an ordinal below zero, an ordinal past the end of the decode, an
empty decode, or an input the contract cannot represent. None of these needs a
model to adjudicate, and running one on a malformed input would only invent an
answer.

This gate runs first and returns UNKNOWN without touching the network. The
baseline string passes through untouched -- no correction is applied on this
path, which is what makes it fail-closed rather than merely fail-fast.

It exists because ORDINAL_OUT_OF_RANGE was originally built as a trainable
UNKNOWN class, and the Stage 3E-0 input contract cannot represent such a query
at all. Rather than widen the contract to admit inputs the runtime can reject
for free, the case moved here.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "PremodelVerdict",
    "PREMODEL_REASONS",
    "check_premodel",
]

PREMODEL_REASONS = (
    "ORDINAL_NEGATIVE",
    "ORDINAL_OUT_OF_RANGE",
    "DECODED_LENGTH_NOT_POSITIVE",
    "INPUT_BUILD_FAILED",
)


@dataclass(frozen=True)
class PremodelVerdict:
    """Outcome of pre-model validation.

    ``network_invoked`` is part of the record rather than an implementation
    detail: a caller must be able to prove the model never saw a malformed
    query.
    """

    verdict: str                 # "UNKNOWN" or "PROCEED"
    reason: str | None
    network_invoked: bool
    correction_allowed: bool

    @property
    def rejected(self) -> bool:
        return self.verdict == "UNKNOWN"


PROCEED = PremodelVerdict("PROCEED", None, False, True)


def check_premodel(target_ordinal: int, decoded_length: int,
                   input_built: bool = True) -> PremodelVerdict:
    """Decide whether the query is answerable at all.

    Returns a rejecting verdict for anything the input contract cannot
    represent. On rejection no correction may be applied and the baseline text
    stands as decoded.
    """
    if decoded_length is None or decoded_length <= 0:
        return PremodelVerdict("UNKNOWN", "DECODED_LENGTH_NOT_POSITIVE",
                               False, False)
    if target_ordinal is None or target_ordinal < 0:
        return PremodelVerdict("UNKNOWN", "ORDINAL_NEGATIVE", False, False)
    if target_ordinal >= decoded_length:
        return PremodelVerdict("UNKNOWN", "ORDINAL_OUT_OF_RANGE", False, False)
    if not input_built:
        return PremodelVerdict("UNKNOWN", "INPUT_BUILD_FAILED", False, False)
    return PROCEED
