"""localization-v4.1: strict, fail-closed glyph anchoring.

v4 kept a posterior-centre fallback whenever the anchor candidates failed to
agree. That contradicts the project's own safety rule -- disagreement means
UNKNOWN -- and it is where most of the accepted misassignments came from: the
fallback answered confidently in exactly the cases the consensus had flagged as
untrustworthy.

v4.1 removes the fallback. Every path that cannot establish, from the image and
the CTC output alone, which character is the target ends in UNKNOWN with a named
reason. There is no branch that guesses.

The crop is a *context patch* centred on the anchor, not an attempt at a glyph
bounding box. It is sized from the line's ink height, so it scales with the
text rather than with the token span. Neighbouring ink inside the patch is
expected and acceptable -- a classifier can see context -- but only while the
target remains unambiguous, which the central-zone checks below enforce.

No expected text, no ground truth and no model output is an input here.
"""

from __future__ import annotations

import statistics
import unicodedata
from dataclasses import asdict, dataclass

import numpy as np

__all__ = [
    "AnchorConfig",
    "AnchorResult",
    "Reason",
    "locate_target",
]


class Reason:
    """Named outcomes. Every UNKNOWN carries one of these."""

    ACCEPTED = "ACCEPTED"
    NO_TARGET_TOKEN = "NO_TARGET_TOKEN"
    SEQUENCE_MISMATCH = "SEQUENCE_MISMATCH"
    AMBIGUOUS_REPEATED_CHARACTER = "AMBIGUOUS_REPEATED_CHARACTER"
    NON_MONOTONIC_TOKENS = "NON_MONOTONIC_TOKENS"
    PITCH_UNAVAILABLE = "PITCH_UNAVAILABLE"
    PITCH_OUT_OF_RANGE = "PITCH_OUT_OF_RANGE"
    TOO_FEW_ANCHORS = "TOO_FEW_ANCHORS"
    ANCHOR_SPREAD_EXCEEDED = "ANCHOR_SPREAD_EXCEEDED"
    ANCHOR_NEAR_CELL_BOUNDARY = "ANCHOR_NEAR_CELL_BOUNDARY"
    NEIGHBOUR_CLOSER_THAN_TARGET = "NEIGHBOUR_CLOSER_THAN_TARGET"
    NEIGHBOUR_IN_CENTRAL_ZONE = "NEIGHBOUR_IN_CENTRAL_ZONE"
    PATCH_CLIPPED = "PATCH_CLIPPED"
    PATCH_DEGENERATE = "PATCH_DEGENERATE"
    INK_HEIGHT_UNAVAILABLE = "INK_HEIGHT_UNAVAILABLE"
    MULTI_SCALE_DISAGREEMENT = "MULTI_SCALE_DISAGREEMENT"


@dataclass(frozen=True)
class AnchorConfig:
    """Frozen geometry. Every value is a ratio, so nothing is font-specific."""

    # Anchor estimators that must agree, and how many must be available.
    required_anchors: tuple[str, ...] = (
        "argmax_center", "posterior_center", "viterbi_center",
        "blank_valley_center",
    )
    minimum_agreeing: int = 4
    # Maximum spread between estimates, as a fraction of local character pitch.
    anchor_spread_tolerance: float = 0.35
    # Plausible character pitch, relative to the line's ink height.
    minimum_pitch_ratio: float = 0.15
    maximum_pitch_ratio: float = 2.50
    # The anchor must sit inside the middle of its own character cell.
    cell_boundary_margin: float = 0.30
    # Half-width of the context patch, as a fraction of ink height.
    patch_half_width_ratio: float = 0.75
    # The zone around the anchor that must belong to the target alone.
    central_zone_ratio: float = 0.30
    # Scales used to check that target assignment is scale-stable.
    multi_scale_ratios: tuple[float, ...] = (0.60, 0.75, 0.90)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class AnchorResult:
    accepted: bool
    reason: str
    anchor_x: float | None = None
    patch: tuple[int, int] | None = None
    anchor_spread: float | None = None
    pitch: float | None = None
    ink_height: float | None = None


def _ink_mask(image: np.ndarray) -> np.ndarray:
    gray = image[..., :3].mean(axis=2) if image.ndim == 3 else image.astype(float)
    low, high = float(gray.min()), float(gray.max())
    if high - low < 12.0:
        return np.zeros(gray.shape, dtype=bool)
    bright = gray >= (low + high) / 2.0
    return bright if bright.mean() < 0.5 else ~bright


def _anchor_estimates(
    emitted: list[dict], position: int, probabilities: np.ndarray, stride: float
) -> dict[str, float]:
    item = emitted[position]
    out: dict[str, float] = {}
    out["argmax_center"] = ((item["start"] + item["end"] + 1) / 2.0) * stride

    window = slice(max(0, item["start"] - 2),
                   min(probabilities.shape[0], item["end"] + 3))
    weights = probabilities[window, item["label"]]
    if float(weights.sum()) > 1e-9:
        indices = np.arange(window.start, window.stop) + 0.5
        out["posterior_center"] = float(
            (indices * weights).sum() / weights.sum()) * stride
    else:
        out["posterior_center"] = out["argmax_center"]

    peak = int(np.argmax(probabilities[window, item["label"]])) + window.start
    out["viterbi_center"] = (peak + 0.5) * stride

    blank = probabilities[:, 0]
    left, right = item["start"], item["end"]
    while left > 0 and blank[left - 1] >= blank[left]:
        left -= 1
    while right + 1 < probabilities.shape[0] and blank[right + 1] >= blank[right]:
        right += 1
    out["blank_valley_center"] = ((left + right + 1) / 2.0) * stride
    return out


def locate_target(
    line_image: np.ndarray,
    emitted: list[dict],
    decoded: str,
    position: int,
    probabilities: np.ndarray,
    stride: float,
    config: AnchorConfig | None = None,
) -> AnchorResult:
    """Locate the character at ``position``, or refuse and say why.

    ``position`` indexes ``emitted``, the collapsed CTC tokens. There is
    deliberately no parameter for expected text or ground truth.
    """
    config = config or AnchorConfig()

    if line_image is None or line_image.size == 0:
        return AnchorResult(False, Reason.INK_HEIGHT_UNAVAILABLE)
    crop_h, crop_w = line_image.shape[:2]

    if not emitted or not (0 <= position < len(emitted)):
        return AnchorResult(False, Reason.NO_TARGET_TOKEN)

    # The collapsed path must reproduce the decoded text, or the token-to-
    # character correspondence is not established.
    if "".join(item["char"] for item in emitted) != decoded:
        return AnchorResult(False, Reason.SEQUENCE_MISMATCH)

    # Tokens must be strictly ordered; anything else breaks the assumption
    # that neighbours in the list are neighbours on the line.
    starts = [item["start"] for item in emitted]
    if any(b <= a for a, b in zip(starts, starts[1:])):
        return AnchorResult(False, Reason.NON_MONOTONIC_TOKENS)

    # If the same character repeats adjacently, which token corresponds to the
    # glyph in question is not uniquely determined.
    target_char = unicodedata.normalize("NFC", emitted[position]["char"])
    for neighbour in (position - 1, position + 1):
        if 0 <= neighbour < len(emitted):
            if unicodedata.normalize("NFC", emitted[neighbour]["char"]) == target_char:
                return AnchorResult(False, Reason.AMBIGUOUS_REPEATED_CHARACTER)

    ink = _ink_mask(line_image)
    rows = np.where(ink.any(axis=1))[0]
    if rows.size == 0:
        return AnchorResult(False, Reason.INK_HEIGHT_UNAVAILABLE)
    ink_height = float(rows[-1] - rows[0] + 1)
    if ink_height < 4:
        return AnchorResult(False, Reason.INK_HEIGHT_UNAVAILABLE)

    centres = [((item["start"] + item["end"] + 1) / 2.0) * stride
               for item in emitted]
    gaps = [b - a for a, b in zip(centres, centres[1:]) if b > a]
    if not gaps:
        return AnchorResult(False, Reason.PITCH_UNAVAILABLE)
    pitch = float(statistics.median(gaps))
    if not (config.minimum_pitch_ratio * ink_height
            <= pitch <= config.maximum_pitch_ratio * ink_height):
        return AnchorResult(False, Reason.PITCH_OUT_OF_RANGE, pitch=pitch,
                            ink_height=ink_height)

    estimates = _anchor_estimates(emitted, position, probabilities, stride)
    available = [estimates[name] for name in config.required_anchors
                 if name in estimates]
    if len(available) < config.minimum_agreeing:
        return AnchorResult(False, Reason.TOO_FEW_ANCHORS, pitch=pitch,
                            ink_height=ink_height)

    spread = max(available) - min(available)
    if spread > config.anchor_spread_tolerance * pitch:
        # No fallback: disagreement is exactly the case v4 answered wrongly.
        return AnchorResult(False, Reason.ANCHOR_SPREAD_EXCEEDED,
                            anchor_spread=spread, pitch=pitch,
                            ink_height=ink_height)

    anchor = float(statistics.median(available))

    # The anchor must sit near the middle of its own character cell, not near
    # the boundary it shares with a neighbour.
    for neighbour_index in (position - 1, position + 1):
        if 0 <= neighbour_index < len(emitted):
            boundary = (anchor + centres[neighbour_index]) / 2.0
            if abs(anchor - boundary) < config.cell_boundary_margin * pitch:
                return AnchorResult(False, Reason.ANCHOR_NEAR_CELL_BOUNDARY,
                                    anchor_x=anchor, anchor_spread=spread,
                                    pitch=pitch, ink_height=ink_height)
            if abs(anchor - centres[neighbour_index]) < abs(anchor - centres[position]):
                return AnchorResult(False, Reason.NEIGHBOUR_CLOSER_THAN_TARGET,
                                    anchor_x=anchor, anchor_spread=spread,
                                    pitch=pitch, ink_height=ink_height)

    # A neighbour's centre inside the central zone means the patch cannot say
    # which glyph the decision is about.
    central = config.central_zone_ratio * pitch
    for neighbour_index in (position - 1, position + 1):
        if 0 <= neighbour_index < len(emitted):
            if abs(centres[neighbour_index] - anchor) < central:
                return AnchorResult(False, Reason.NEIGHBOUR_IN_CENTRAL_ZONE,
                                    anchor_x=anchor, anchor_spread=spread,
                                    pitch=pitch, ink_height=ink_height)

    half = config.patch_half_width_ratio * ink_height
    x0, x1 = anchor - half, anchor + half
    if x0 < 0 or x1 > crop_w:
        return AnchorResult(False, Reason.PATCH_CLIPPED, anchor_x=anchor,
                            anchor_spread=spread, pitch=pitch,
                            ink_height=ink_height)
    if x1 - x0 < 4:
        return AnchorResult(False, Reason.PATCH_DEGENERATE, anchor_x=anchor,
                            pitch=pitch, ink_height=ink_height)

    # Target assignment must not depend on the scale chosen.
    for ratio in config.multi_scale_ratios:
        scaled_half = ratio * ink_height
        left, right = anchor - scaled_half, anchor + scaled_half
        if left < 0 or right > crop_w:
            return AnchorResult(False, Reason.MULTI_SCALE_DISAGREEMENT,
                                anchor_x=anchor, anchor_spread=spread,
                                pitch=pitch, ink_height=ink_height)
        inside = [index for index, centre in enumerate(centres)
                  if left <= centre <= right]
        if position not in inside:
            return AnchorResult(False, Reason.MULTI_SCALE_DISAGREEMENT,
                                anchor_x=anchor, anchor_spread=spread,
                                pitch=pitch, ink_height=ink_height)
        nearest = min(inside, key=lambda index: abs(centres[index] - anchor))
        if nearest != position:
            return AnchorResult(False, Reason.MULTI_SCALE_DISAGREEMENT,
                                anchor_x=anchor, anchor_spread=spread,
                                pitch=pitch, ink_height=ink_height)

    return AnchorResult(
        True, Reason.ACCEPTED, anchor_x=anchor,
        patch=(int(round(x0)), int(round(x1))),
        anchor_spread=spread, pitch=pitch, ink_height=ink_height,
    )
